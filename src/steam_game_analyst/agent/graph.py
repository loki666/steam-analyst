"""The LangGraph agent.

    START -> router --(out of scope)--> refuse -> END
                    --(needs documents)--> retrieve -> agent
                    --------------------------------> agent
    agent --(tool calls, under the cap)--> tools --(retry edge: results or errors)--> agent
    agent --(malformed tool call, under the cap)--> agent
    agent --(DONE / cap / too many errors)--> answer -> END

`router` plans and decides whether to retrieve; `retrieve` reuses the W2 retriever filtered
to the games named in the question; `agent` is the chat model with the four tools bound;
`tools` validates every call against its pydantic schema and feeds errors back so the model
can correct them; `answer` writes the final text from tool results [Tn] and documents [n]
and keeps only citations that point at real evidence."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from steam_game_analyst.agent import prompts
from steam_game_analyst.agent.state import AgentState, RouterDecision
from steam_game_analyst.citations import cited_ids
from steam_game_analyst.retrieval.filters import SearchFilters
from steam_game_analyst.retrieval.query import GameIndex
from steam_game_analyst.retrieval.rag import _defang, _fit, describe
from steam_game_analyst.tools.events import patch_candidates
from steam_game_analyst.tools.registry import TOOL_NAMES, ToolBox

MAX_ITERATIONS = 6
MAX_ERRORS = 4
MAX_CALLS_PER_TURN = 5  # a runaway turn of dozens of calls costs minutes on a 14B model
RETRIEVE_K = 6
DOC_BUDGET_TOKENS = 2400  # documents share of the answer prompt (num_ctx 8192)
TOOL_BUDGET_CHARS = 14_000  # ~4k tokens of tool output, shared by all results
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


class Retriever(Protocol):
    def retrieve(
        self, question: str, k: int = 8, overrides: SearchFilters | None = None
    ) -> Any: ...


def parse_router(text: str) -> RouterDecision | None:
    m = _JSON_OBJ.search(text or "")
    if not m:
        return None
    try:
        return RouterDecision.model_validate(json.loads(m.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


def text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Tool calls the model wrote as text ({"name": ..., "arguments": {...}}, possibly inside
    <tool_call> tags) instead of using the native tool-call channel."""
    decoder = json.JSONDecoder()
    calls, i = [], content.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            i = content.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, dict):
                calls.append(
                    {"name": obj["name"], "args": args, "id": f"txt-{uuid.uuid4().hex[:8]}"}
                )
        i = content.find("{", end)
    return calls


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f" …[{len(text) - n} more chars cut]"


def appids_in(content: str) -> set[str]:
    """appids a tool result names: "appid" fields, and the appid column of query rows."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return set()
    found: set[str] = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if isinstance(o.get("columns"), list) and isinstance(o.get("rows"), list):
                cols = [str(c).lower() for c in o["columns"]]
                if "appid" in cols:
                    j = cols.index("appid")
                    found.update(str(r[j]) for r in o["rows"] if isinstance(r, list) and len(r) > j)
            for k, v in o.items():
                if k == "appid" and isinstance(v, int | str):
                    found.add(str(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


class Agent:
    def __init__(
        self,
        llm: BaseChatModel,
        toolbox: ToolBox,
        games: GameIndex,
        corpus_appids: Iterable[int],
        retriever: Retriever | None = None,
        router_llm: BaseChatModel | None = None,
        max_iterations: int = MAX_ITERATIONS,
        today: date | None = None,
    ) -> None:
        self.llm = llm
        self.router_llm = router_llm or llm
        self.tool_llm = llm.bind_tools(toolbox.langchain_tools())
        self.toolbox = toolbox
        self.games = games
        self.corpus = set(corpus_appids)
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.today = today or date.today()
        self._schema_text: str | None = None
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("router", self.router)
        g.add_node("refuse", self.refuse)
        g.add_node("retrieve", self.retrieve)
        g.add_node("agent", self.agent)
        g.add_node("tools", self.tools)
        g.add_node("answer", self.answer)
        g.add_edge(START, "router")
        g.add_conditional_edges("router", self.after_router, ["refuse", "retrieve", "agent"])
        g.add_edge("refuse", END)
        g.add_edge("retrieve", "agent")
        g.add_conditional_edges("agent", self.after_agent, ["tools", "agent", "answer"])
        g.add_conditional_edges("tools", self.after_tools, ["agent", "answer"])
        g.add_edge("answer", END)
        return g.compile()

    def run(
        self, question: str, on_step: Callable[[str, dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Final state for one question. `on_step(node, update)` is called after each node
        finishes (the UI shows progress with it); without it the graph runs as one invoke."""
        state = {
            "question": question,
            "iterations": 0,
            "errors": 0,
            "notes": [],
            "tool_results": [],
            "chunks": [],
        }
        config: Any = {"recursion_limit": 4 * self.max_iterations + 12}
        if on_step is None:
            return self.graph.invoke(state, config=config)
        final: dict[str, Any] = state
        for mode, chunk in self.graph.stream(
            state, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                final = chunk
                continue
            for node, update in chunk.items():
                on_step(node, update or {})
        return final

    # ------------------------------------------------------------------ nodes
    def _games_text(self, games: list[dict[str, Any]]) -> str:
        if not games:
            return "none recognised (use query_catalog to look games up by name)"
        return "\n".join(
            f"- {g['name']} (appid {g['appid']}; "
            f"{'in' if g['in_review_corpus'] else 'NOT in'} the review corpus)"
            for g in games
        )

    @staticmethod
    def _events_text(events: list[str]) -> str:
        head = "Patch notes whose titles match the question"
        if not events:
            return f"{head}: none"
        return (
            f"{head} (for an event the question names, use the date of its release note as "
            "event_date / split_date, not a date guessed from the question's wording):\n"
            + "\n".join(f"- {e}" for e in events)
        )

    def router(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]
        games = [
            {"appid": a, "name": n, "in_review_corpus": a in self.corpus}
            for a, n in self.games.find(q)
        ]
        msgs = [
            SystemMessage(prompts.ROUTER.format(tools=", ".join(TOOL_NAMES))),
            HumanMessage(f"Question: {q}\nGames recognised: {self._games_text(games)}"),
        ]
        decision = parse_router(str(self.router_llm.invoke(msgs).content))
        notes = list(state.get("notes", []))
        if decision is None:
            notes.append("router reply was not valid JSON: default plan (retrieve + tools)")
            decision = RouterDecision(retrieval_query=q)
        events = patch_candidates(self.toolbox.db, games, q) if decision.in_scope else []
        return {
            "games": games,
            "route": decision,
            "plan": decision.plan,
            "events": events,
            "notes": notes,
        }

    def after_router(self, state: AgentState) -> Literal["refuse", "retrieve", "agent"]:
        route = state["route"]
        if not route.in_scope:
            return "refuse"
        return "retrieve" if route.needs_retrieval and self.retriever else "agent"

    def refuse(self, state: AgentState) -> dict[str, Any]:
        return {"answer": prompts.REFUSAL, "citations": []}

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        assert self.retriever is not None
        query = state["route"].retrieval_query or state["question"]
        appids = tuple(g["appid"] for g in state.get("games", []) if g["in_review_corpus"])
        try:
            hits = self.retriever.retrieve(
                query, k=RETRIEVE_K, overrides=SearchFilters(appids=appids)
            ).hits
        except Exception as e:  # noqa: BLE001 — the vector store is optional for tool questions
            return {"notes": [*state.get("notes", []), f"retrieval failed: {e}"]}
        per_doc = DOC_BUDGET_TOKENS // max(len(hits), 1)
        chunks = [
            {
                "n": i,
                "id": h.id,
                "label": describe(h),
                "text": _fit(h.payload.get("text", ""), per_doc),
                "url": h.payload.get("url") or None,
            }
            for i, h in enumerate(hits, start=1)
        ]
        return {"chunks": chunks}

    def _schema(self) -> str:
        if self._schema_text is None:
            self._schema_text = "\n".join(
                f"- {t}({', '.join(cols)})" for t, cols in self.toolbox.db.schema().items()
            )
        return self._schema_text

    def agent(self, state: AgentState) -> dict[str, Any]:
        data_end = self.toolbox.trend.data_end
        system = prompts.AGENT.format(
            today=self.today.isoformat(),
            data_through=data_end.isoformat() if data_end else "unknown",
            games=self._games_text(state.get("games", [])),
            events=self._events_text(state.get("events", [])),
            plan=state.get("plan") or "(none)",
            chunks="; ".join(f"[{c['n']}] {c['label']}" for c in state.get("chunks", [])) or "none",
            schema=self._schema(),
        )
        msgs: list[BaseMessage] = [SystemMessage(system), HumanMessage(state["question"])]
        reply = self.tool_llm.invoke(msgs + list(state.get("messages", [])))
        assert isinstance(reply, AIMessage)
        update: dict[str, Any] = {"iterations": state.get("iterations", 0) + 1}
        if not reply.tool_calls and (calls := text_tool_calls(str(reply.content))):
            reply = AIMessage(content="", tool_calls=calls)
        if reply.invalid_tool_calls and not reply.tool_calls:
            bad = "; ".join(f"{c.get('name')}: {c.get('error')}" for c in reply.invalid_tool_calls)
            update["messages"] = [
                AIMessage(content=str(reply.content) or "(malformed tool call)"),
                HumanMessage(
                    f"Your tool call could not be parsed ({bad}). Call the tool again with "
                    "valid JSON arguments, or reply DONE."
                ),
            ]
            update["errors"] = state.get("errors", 0) + 1
            return update
        update["messages"] = [reply]
        return update

    def after_agent(self, state: AgentState) -> Literal["tools", "agent", "answer"]:
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, HumanMessage):  # malformed call fed back: retry edge
            return "answer" if self._capped(state) else "agent"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "answer"

    def _capped(self, state: AgentState) -> bool:
        return (
            state.get("iterations", 0) >= self.max_iterations
            or state.get("errors", 0) >= MAX_ERRORS
        )

    def tools(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        results = list(state.get("tool_results", []))
        failed = list(state.get("tool_errors", []))
        errors = state.get("errors", 0)
        seen = {str(g["appid"]) for g in state.get("games", [])}
        for r in results:
            seen |= appids_in(r["content"])
        out: list[BaseMessage] = []
        for n, tc in enumerate(last.tool_calls):
            text, result, error = self._execute(n, tc, results, failed, seen)
            if error is not None:
                errors += 1
                failed.append(error)
            if result is not None:
                results.append(result)
                seen |= appids_in(result["content"])  # later calls in this turn may use them
            status = "error" if error is not None or text.startswith("skipped") else "success"
            out.append(ToolMessage(text, tool_call_id=tc["id"], status=status))
        return {"messages": out, "tool_results": results, "errors": errors, "tool_errors": failed}

    def _execute(
        self,
        n: int,
        tc: dict[str, Any],
        results: list[dict[str, Any]],
        failed: list[dict[str, Any]],
        seen: set[str],
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        """One tool call -> (message for the model, new result or None, error record or None)."""
        args = tc.get("args") or {}
        if n >= MAX_CALLS_PER_TURN:
            return (
                (
                    f"skipped: at most {MAX_CALLS_PER_TURN} tool calls per turn; repeat this "
                    "call in the next turn if you still need it"
                ),
                None,
                None,
            )
        appid = args.get("appid")
        if str(appid).isdigit() and str(appid) not in seen:  # non-numeric: schema validation
            msg = (
                f"appid {appid} was not seen in the question or in any tool result; look "
                "the game up with query_catalog first, never guess an appid"
            )
            return f"ERROR {msg}", None, {"tool": tc["name"], "args": args, "error": msg}
        res = self.toolbox.invoke(tc["name"], args)
        if not res.valid:
            msg = res.content
            if any(f["tool"] == res.name and f["args"] == args for f in failed):
                msg = (
                    f"you already made this exact call and it failed: {msg} -- change the arguments"
                )
            return f"ERROR {msg}", None, {"tool": tc["name"], "args": args, "error": msg[:300]}
        dup = next((t for t in results if t["tool"] == res.name and t["args"] == res.args), None)
        if dup is not None:  # same validated call again: point at the earlier result
            return f"same call as [{dup['id']}]; see that result", None, None
        tid = f"T{len(results) + 1}"
        result = {
            "id": tid,
            "tool": res.name,
            "args": res.args,
            "ok": res.ok,
            "content": res.content,
        }
        return f"[{tid}] {res.content}", result, None

    def after_tools(self, state: AgentState) -> Literal["agent", "answer"]:
        if self._capped(state):
            return "answer"
        return "agent"

    def answer(self, state: AgentState) -> dict[str, Any]:
        notes = list(state.get("notes", []))
        if state.get("iterations", 0) >= self.max_iterations:
            notes.append(f"stopped at the iteration cap ({self.max_iterations})")
        if state.get("errors", 0) >= MAX_ERRORS:
            notes.append(f"stopped after {MAX_ERRORS} invalid tool calls")
        tools = state.get("tool_results", [])
        chunks = state.get("chunks", [])
        evidence = (
            "\n\n".join(
                [
                    f'<tool id="{t["id"]}" name="{t["tool"]}" args=\'{json.dumps(t["args"])}\'>\n'
                    f"{_clip(t['content'], TOOL_BUDGET_CHARS // max(len(tools), 1))}\n</tool>"
                    for t in tools
                ]
                + [
                    f'<source id="{c["n"]}">\n[{c["n"]}] {_defang(c["label"])}\n'
                    f"{_defang(c['text'])}\n</source>"
                    for c in chunks
                ]
            )
            or "(no evidence was gathered)"
        )
        prompt = prompts.ANSWER.format(
            today=self.today.isoformat(), evidence=evidence, question=state["question"]
        )
        msgs: list[BaseMessage] = [HumanMessage(prompt)]
        text = str(self.llm.invoke(msgs).content).strip()
        citations = self._citations(text, tools, chunks)
        if not citations and (tools or chunks):  # one rewrite: every answer must cite
            msgs += [AIMessage(text), HumanMessage(prompts.CITE_REMINDER)]
            text = str(self.llm.invoke(msgs).content).strip()
            citations = self._citations(text, tools, chunks)
            if not citations:
                notes.append("the answer cites no evidence")
        return {"answer": text, "citations": citations, "notes": notes}

    @staticmethod
    def _citations(
        text: str, tools: list[dict[str, Any]], chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_tool = {t["id"]: t for t in tools}
        by_doc = {str(c["n"]): c for c in chunks}
        out = []
        for ref in cited_ids(text):
            if ref in by_tool:
                t = by_tool[ref]
                out.append({"id": ref, "kind": "tool", "tool": t["tool"], "args": t["args"]})
            elif ref in by_doc:
                c = by_doc[ref]
                out.append(
                    {
                        "id": ref,
                        "kind": "document",
                        "doc_id": c["id"],
                        "label": c["label"],
                        "url": c["url"],
                    }
                )
        return out
