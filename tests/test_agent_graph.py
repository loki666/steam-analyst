"""LangGraph agent with a scripted chat model (no Ollama) over the fixture data."""

import json
from typing import Any

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from steam_game_analyst.agent.graph import Agent, parse_router
from steam_game_analyst.retrieval.query import GameIndex
from steam_game_analyst.retrieval.store import Hit
from steam_game_analyst.tools.registry import ToolBox


class ScriptedLLM(BaseChatModel):
    """Replays canned replies in order and records every prompt it received."""

    replies: list[Any]
    seen: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        if not self.replies:
            raise AssertionError("LLM called more often than scripted")
        r = self.replies.pop(0)
        msg = r if isinstance(r, AIMessage) else AIMessage(content=r)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, args: dict, i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def router(**kw) -> str:
    d = {
        "in_scope": True,
        "needs_retrieval": False,
        "retrieval_query": "",
        "tools": [],
        "plan": "p",
    }
    return json.dumps({**d, **kw})


class Dummy:
    def predict_proba(self, texts):
        return np.full(len(texts), 0.8)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, question, k=8, overrides=None):
        self.queries.append((question, overrides))
        hit = Hit(
            "h1",
            0.9,
            {
                "game": "Cyberpunk 2077",
                "doc_type": "patch_note",
                "created_at": 1695300000,
                "title": "Update 2.0",
                "text": "Update 2.0 overhauls performance.",
                "url": "https://x",
            },
        )

        class R:
            hits = (hit,)

        return R()


@pytest.fixture
def toolbox(built):
    box = ToolBox(built, sentiment_model=Dummy())
    yield box
    box.close()


def make_agent(toolbox, replies, retriever=None, **kw):
    llm = ScriptedLLM(replies=replies, seen=[])
    games = GameIndex.from_names({1091500: "Cyberpunk 2077", 892970: "Valheim"})
    agent = Agent(llm, toolbox, games, corpus_appids={1091500, 892970}, retriever=retriever, **kw)
    return agent, llm


def test_tool_loop_then_cited_answer(toolbox):
    agent, llm = make_agent(
        toolbox,
        [
            router(tools=["review_trend"]),
            call("review_trend", {"appid": 1091500, "last_n_months": 12}),
            "DONE",
            "Reviews trend up [T1].",
        ],
    )
    out = agent.run("Are Cyberpunk 2077 reviews trending up?")
    assert out["games"] == [{"appid": 1091500, "name": "Cyberpunk 2077", "in_review_corpus": True}]
    assert [t["tool"] for t in out["tool_results"]] == ["review_trend"]
    assert out["tool_results"][0]["ok"] and out["tool_results"][0]["id"] == "T1"
    assert out["answer"] == "Reviews trend up [T1]."
    assert out["citations"] == [
        {
            "id": "T1",
            "kind": "tool",
            "tool": "review_trend",
            "args": {
                "appid": 1091500,
                "granularity": "month",
                "last_n_months": 12,
                "window_months": 6,
            },
        }
    ]
    # the agent prompt carries the resolved appid; the answer prompt carries the tool output
    assert "appid 1091500" in llm.seen[1][0].content
    assert '<tool id="T1" name="review_trend"' in llm.seen[3][-1].content


def test_invalid_arguments_go_back_to_the_model_and_it_retries(toolbox):
    agent, llm = make_agent(
        toolbox,
        [
            router(),
            call("review_trend", {"appid": "Cyberpunk"}, 0),
            call("review_trend", {"appid": 1091500}, 1),
            "DONE",
            "Answer [T1].",
        ],
    )
    out = agent.run("Cyberpunk 2077 trend?")
    assert out["errors"] == 1
    retry_prompt = llm.seen[2]
    err = [m for m in retry_prompt if isinstance(m, ToolMessage)][0]
    assert "invalid arguments" in err.content and "appid" in err.content
    assert [t["ok"] for t in out["tool_results"]] == [True]  # only successful calls are evidence
    assert (
        out["tool_errors"][0]["tool"] == "review_trend"
        and "appid" in out["tool_errors"][0]["error"]
    )


def test_iteration_cap_stops_the_loop(toolbox):
    replies = [router()] + [call("query_catalog", {"sql": f"SELECT {i}"}, i) for i in range(3)]
    agent, _ = make_agent(toolbox, replies + ["Capped answer [T1]."], max_iterations=3)
    out = agent.run("count games")
    assert out["iterations"] == 3
    assert len(out["tool_results"]) == 3
    assert any("iteration cap" in n for n in out["notes"])


def test_out_of_scope_is_refused_without_tools(toolbox):
    agent, llm = make_agent(toolbox, [router(in_scope=False)])
    out = agent.run("What should I play tonight?")
    assert out["answer"].startswith("I can only answer questions about PC games on Steam")
    assert not out.get("tool_results") and len(llm.seen) == 1


def test_router_garbage_falls_back_to_in_scope_with_retrieval(toolbox):
    retriever = FakeRetriever()
    agent, _ = make_agent(
        toolbox, ["not json at all", "DONE", "From the patch notes [1]."], retriever=retriever
    )
    out = agent.run("What changed in Cyberpunk 2077 update 2.0?")
    assert any("router" in n for n in out["notes"])
    assert retriever.queries and retriever.queries[0][1].appids == (1091500,)
    assert out["citations"][0]["kind"] == "document" and out["citations"][0]["url"] == "https://x"


def test_tool_call_written_as_text_is_still_executed(toolbox):
    text_call = '{"name": "query_catalog", "arguments": {"sql": "SELECT count(*) FROM games"}}'
    agent, _ = make_agent(toolbox, [router(), text_call, "DONE", "4 games [T1]."])
    out = agent.run("How many games are in the subset?")
    assert out["tool_results"][0]["tool"] == "query_catalog"


def test_citations_out_of_range_are_dropped(toolbox):
    agent, _ = make_agent(toolbox, [router(), "DONE", "Made up [T7] and [3]."])
    out = agent.run("anything about games")
    assert out["citations"] == []


def test_uncited_answer_is_rewritten_once_with_citations(toolbox):
    agent, llm = make_agent(
        toolbox,
        [
            router(),
            call("query_catalog", {"sql": "SELECT count(*) FROM games"}),
            "DONE",
            "There are 4 games.",
            "There are 4 games [T1].",
        ],
    )
    out = agent.run("how many games")
    assert out["answer"] == "There are 4 games [T1]." and out["citations"][0]["id"] == "T1"
    assert "citing the evidence" in llm.seen[-1][-1].content


def test_parse_router_accepts_fenced_json():
    d = parse_router('```json\n{"in_scope": false, "plan": "x"}\n```')
    assert d is not None and d.in_scope is False
    assert parse_router("nope") is None


def test_malformed_tool_call_is_fed_back_and_retried(toolbox):
    bad = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "review_trend",
                "args": "{appid: oops",
                "id": "x",
                "error": "bad json",
                "type": "invalid_tool_call",
            }
        ],
    )
    agent, llm = make_agent(
        toolbox, [router(), bad, call("review_trend", {"appid": 1091500}), "DONE", "ok [T1]"]
    )
    out = agent.run("Cyberpunk 2077 trend")
    assert out["errors"] == 1 and out["tool_results"][0]["ok"]
    assert "could not be parsed" in llm.seen[2][-1].content


def test_trace_record_is_json_serialisable(toolbox):
    from steam_game_analyst.agent.factory import trace_record

    agent, _ = make_agent(
        toolbox, [router(), call("review_trend", {"appid": 1091500}), "DONE", "ok [T1]"]
    )
    rec = trace_record(agent.run("Cyberpunk 2077 trend"), latency_s=1.5, model="m")
    line = json.dumps(rec)
    assert '"tool": "review_trend"' in line and rec["latency_s"] == 1.5
    assert "content" not in rec["tool_calls"][0]  # outputs stay out of the trace


def test_parse_router_accepts_plan_as_a_list_and_tools_as_a_string():
    d = parse_router('{"plan": ["a", "b"], "tools": "review_trend", "needs_retrieval": "false"}')
    assert d is not None and d.plan == "1. a 2. b" and d.tools == ["review_trend"]
    assert d.needs_retrieval is False


def test_identical_tool_calls_reuse_the_first_result(toolbox):
    args = {"sql": "SELECT count(*) FROM games"}
    agent, llm = make_agent(
        toolbox,
        [
            router(),
            call("query_catalog", args, 0),
            call("query_catalog", args, 1),
            "DONE",
            "4 [T1]",
        ],
    )
    out = agent.run("how many games")
    assert len(out["tool_results"]) == 1
    assert "same call as [T1]" in llm.seen[3][-1].content


def test_tool_calls_per_turn_are_capped(toolbox):
    many = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_catalog", "args": {"sql": f"SELECT {i}"}, "id": f"c{i}"}
            for i in range(9)
        ],
    )
    agent, llm = make_agent(toolbox, [router(), many, "DONE", "ok [T1]"])
    out = agent.run("count games")
    assert len(out["tool_results"]) == 5
    skipped = [m for m in llm.seen[2] if isinstance(m, ToolMessage) and "skipped" in m.content]
    assert len(skipped) == 4


def test_unseen_appids_are_refused_until_a_tool_returns_them(toolbox):
    agent, llm = make_agent(
        toolbox,
        [
            router(),
            call("review_trend", {"appid": 12345}, 0),  # made up: not in question or results
            call("query_catalog", {"sql": "SELECT appid FROM games WHERE name = 'Hades'"}, 1),
            call("review_trend", {"appid": 1145360}, 2),  # seen in [T1] now
            "DONE",
            "ok [T2]",
        ],
    )
    out = agent.run("how is that roguelike doing")
    assert out["tool_errors"][0]["tool"] == "review_trend"
    assert "not seen" in out["tool_errors"][0]["error"]
    assert [t["tool"] for t in out["tool_results"]] == ["query_catalog", "review_trend"]


def test_appid_from_a_lookup_can_be_used_in_the_same_turn(toolbox):
    both = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_catalog",
                "args": {"sql": "SELECT appid, name FROM games WHERE name = 'Hades'"},
                "id": "a",
            },
            {"name": "review_trend", "args": {"appid": 1145360}, "id": "b"},
        ],
    )
    agent, _ = make_agent(toolbox, [router(), both, "DONE", "ok [T1][T2]"])
    out = agent.run("how is that roguelike doing")
    assert [t["tool"] for t in out["tool_results"]] == ["query_catalog", "review_trend"]


def test_numbers_that_are_not_appids_do_not_unlock_an_appid(toolbox):
    agent, _ = make_agent(
        toolbox,
        [
            router(),
            call("query_catalog", {"sql": "SELECT 999001 AS votes"}, 0),
            call("review_trend", {"appid": 999001}, 1),
            "DONE",
            "ok [T1]",
        ],
    )
    out = agent.run("count")
    assert out["tool_errors"] and "not seen" in out["tool_errors"][0]["error"]


def test_text_tool_calls_parse_several_objects():
    from steam_game_analyst.agent.graph import text_tool_calls

    txt = (
        '{"name": "review_trend", "arguments": {"appid": 1}} then '
        '{"name": "query_catalog", "arguments": {"sql": "SELECT 1"}}'
    )
    assert [c["name"] for c in text_tool_calls(txt)] == ["review_trend", "query_catalog"]


def test_answer_prompt_has_a_total_evidence_budget(toolbox):
    from steam_game_analyst.agent import graph

    calls = [
        call("query_catalog", {"sql": f"SELECT review_text, {i} FROM reviews"}, i) for i in range(5)
    ]
    agent, llm = make_agent(toolbox, [router(), *calls, "DONE", "ok [T1]"], max_iterations=6)
    agent.run("dump reviews")
    prompt = llm.seen[-1][-1].content
    assert len(prompt) < graph.TOOL_BUDGET_CHARS + 3000


def test_on_step_reports_each_node_and_returns_the_same_final_state(toolbox):
    replies = [
        router(tools=["review_trend"]),
        call("review_trend", {"appid": 1091500, "last_n_months": 12}),
        "DONE",
        "Reviews trend up [T1].",
    ]
    agent, _ = make_agent(toolbox, list(replies))
    steps: list[str] = []
    out = agent.run("Are Cyberpunk 2077 reviews trending up?", on_step=lambda n, u: steps.append(n))
    assert steps == ["router", "agent", "tools", "agent", "answer"]
    plain, _ = make_agent(toolbox, list(replies))
    ref = plain.run("Are Cyberpunk 2077 reviews trending up?")
    assert out["answer"] == ref["answer"] and out["citations"] == ref["citations"]
    assert out["tool_results"] == ref["tool_results"]
