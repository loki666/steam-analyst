"""Pure view models over the agent's final state (`Agent.run`): no Streamlit, no LLM.

The UI reads the state dict the graph returns: answer, citations, chunks (retrieved documents
[n]), tool_results (T-ids), tool_errors, notes, route and games. Tool outputs arrive as JSON
text that the registry may have cut at its size cap, so `parse_content` salvages the complete
top-level fields of a truncated output instead of dropping it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

# one bracket can hold several ids: "[3]", "[T2]", "[3, 2]", "[T1][2]"
CITE_GROUP = re.compile(r"\[((?:T?\d+)(?:\s*,\s*T?\d+)*)\]")
_ONLY_CITES = re.compile(r"^\s*(?:\[(?:T?\d+)(?:\s*,\s*T?\d+)*\][\s.]*)+$")
SNIPPET_CHARS = 280
ARGS_CHARS = 140
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Source:
    n: int
    label: str
    snippet: str
    text: str
    url: str | None


@dataclass(frozen=True)
class ToolRow:
    id: str
    tool: str
    args: str
    status: Literal["ok", "no data", "error"]
    summary: str


@dataclass(frozen=True)
class Banner:
    kind: Literal["warn", "info"]
    text: str


# ------------------------------------------------------------------ parsing
def _skip(text: str, i: int, chars: str) -> int:
    while i < len(text) and text[i] in chars:
        i += 1
    return i


def _salvage(text: str) -> dict[str, Any]:
    """Every complete top-level `"key": value` pair of a JSON object cut short."""
    dec = json.JSONDecoder()
    out: dict[str, Any] = {}
    i = _skip(text, 0, " \n\t")
    if not text.startswith("{", i):
        return out
    i += 1
    while True:
        i = _skip(text, i, " \n\t,")
        try:
            key, i = dec.raw_decode(text, i)
            i = _skip(text, i, " \n\t")
            if not text.startswith(":", i):
                break
            value, i = dec.raw_decode(text, _skip(text, i + 1, " \n\t"))
        except json.JSONDecodeError:
            break
        out[str(key)] = value
    return out


def parse_content(content: str) -> dict[str, Any]:
    """A tool output as a dict; a truncated one keeps its complete fields and `_truncated`."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {**_salvage(content or ""), "_truncated": True}
    return obj if isinstance(obj, dict) else {"value": obj}


def route_of(result: dict[str, Any]) -> dict[str, Any]:
    route = result.get("route")
    if route is None:
        return {}
    if hasattr(route, "model_dump"):
        return dict(route.model_dump())
    return dict(route)


def is_refusal(result: dict[str, Any]) -> bool:
    return route_of(result).get("in_scope") is False


# ------------------------------------------------------------------ citations
def cited_ids(answer: str) -> list[str]:
    """Ids the answer cites, in reading order, each once ("T2", "4", ...)."""
    seen: dict[str, None] = {}
    for group in CITE_GROUP.findall(answer or ""):
        for ref in group.split(","):
            seen.setdefault(ref.strip(), None)
    return list(seen)


def snippet(text: str, n: int = SNIPPET_CHARS) -> str:
    flat = _WS.sub(" ", text or "").strip()
    if len(flat) <= n:
        return flat
    cut = flat[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "…"


def _source(chunk: dict[str, Any]) -> Source:
    return Source(
        n=int(chunk["n"]),
        label=str(chunk.get("label", "")).replace("patch_note", "patch note"),
        snippet=snippet(str(chunk.get("text", ""))),
        text=str(chunk.get("text", "")),
        url=chunk.get("url") or None,
    )


def sources(result: dict[str, Any]) -> tuple[list[Source], list[Source]]:
    """(documents the answer cites in reading order, retrieved documents it does not cite)."""
    chunks = {str(c["n"]): c for c in result.get("chunks", [])}
    order = [i for i in cited_ids(result.get("answer", "")) if i in chunks]
    cited = [_source(chunks[i]) for i in order]
    rest = [_source(c) for k, c in chunks.items() if k not in order]
    return cited, rest


def _badge(ref: str, known: set[str]) -> str:
    if ref not in known:
        return f"[{ref}]"
    color = "gray" if ref.startswith("T") else "blue"
    return f":{color}-badge[{ref}]"


def attach_lone_citations(text: str) -> str:
    """A line that is only citations ("[T1]" alone after a blank line) joins the line above, so
    the chips sit at the end of the claim instead of forming a paragraph of their own."""
    out: list[str] = []
    for line in text.split("\n"):
        if _ONLY_CITES.match(line) and any(prev.strip() for prev in out):
            while not out[-1].strip():
                out.pop()
            out[-1] = f"{out[-1].rstrip()} {line.strip()}"
            continue
        out.append(line)
    return "\n".join(out)


def answer_markdown(result: dict[str, Any]) -> str:
    """The answer as Streamlit markdown: citations become badges, `$` stays literal (st.markdown
    would read "$25 ... $15" as LaTeX)."""
    known = {str(c["n"]) for c in result.get("chunks", [])}
    known |= {t["id"] for t in result.get("tool_results", [])}

    def sub(m: re.Match[str]) -> str:
        return " ".join(_badge(r.strip(), known) for r in m.group(1).split(","))

    text = attach_lone_citations(result.get("answer") or "").replace("$", r"\$")
    return CITE_GROUP.sub(sub, text)


# ------------------------------------------------------------------ tool trace
def compact_args(args: dict[str, Any], n: int = ARGS_CHARS) -> str:
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return text if len(text) <= n else text[: n - 1] + "…"


def whole(x: float | None) -> str:
    """Effective sample sizes everywhere: a whole number ("523", never "523.5")."""
    return "?" if x is None else f"{round(x):,}"


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _game(out: dict[str, Any]) -> str:
    game = out.get("game") or {}
    return str(game.get("name") or f"appid {game.get('appid', '?')}")


def _sum_trend(out: dict[str, Any]) -> str:
    if "limitation" in out:
        return f"{_game(out)} · no dated reviews: catalog facts only"
    w, s, t = out.get("window", {}), out.get("summary", {}), out.get("trend", {})
    parts = [_game(out), f"{w.get('from', '?')} → {w.get('to', '?')}"]
    if s:
        parts.append(f"{s.get('n_reviews', 0):,} reviews · {pct(s.get('positive_share'))} positive")
    if t.get("direction"):
        p = t.get("p_value")
        parts.append(f"trend {t['direction']}" + (f" (p {p:.2g})" if p is not None else ""))
    return " · ".join(parts)


def _sum_sentiment(out: dict[str, Any]) -> str:
    w = out.get("window", {})
    return (
        f"{_game(out)} · {w.get('from', '?')} → {w.get('to', '?')} · "
        f"{out.get('n_reviews_sampled', '?')} sampled · effective n {whole(out.get('effective_n'))}"
        f" · {pct(out.get('positive_share'))} positive"
    )


def _sum_catalog(out: dict[str, Any]) -> str:
    if "no_match" in out:
        return "no match: " + snippet(str(out["no_match"]), 120)
    cols = ", ".join(str(c) for c in out.get("columns", [])[:6])
    more = " (more rows cut)" if out.get("truncated") else ""
    return f"{out.get('row_count', 0)} rows{more} · {cols}"


def _sum_reception(out: dict[str, Any]) -> str:
    probs = out.get("probabilities", {})
    bucket = out.get("bucket")
    if bucket is None:
        return "no prediction"
    return f"most likely {bucket.replace('_', ' ')} ({pct(probs.get(bucket))})"


_SUMMARIES = {
    "review_trend": _sum_trend,
    "review_sentiment": _sum_sentiment,
    "query_catalog": _sum_catalog,
    "predict_reception": _sum_reception,
}


def summarize(tool: str, content: str) -> str:
    out = parse_content(content)
    if "error" in out:
        return str(out["error"])
    fn = _SUMMARIES.get(tool)
    text = fn(out) if fn else snippet(content, 160)
    return text + (" · output cut at the size cap" if out.get("_truncated") else "")


def tool_rows(result: dict[str, Any]) -> list[ToolRow]:
    """Every tool call in run order: the T-id evidence, then the calls that were refused."""
    rows = [
        ToolRow(
            id=t["id"],
            tool=t["tool"],
            args=compact_args(t.get("args", {})),
            status="ok" if t.get("ok") else "no data",
            summary=summarize(t["tool"], t.get("content", "")),
        )
        for t in result.get("tool_results", [])
    ]
    rows += [
        ToolRow(
            id="–",
            tool=str(e.get("tool", "?")),
            args=compact_args(e.get("args") or {}),
            status="error",
            summary=str(e.get("error", "")),
        )
        for e in result.get("tool_errors", [])
    ]
    return rows
