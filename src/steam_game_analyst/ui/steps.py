"""What the agent did, in words: the live progress lines, the step strip under each answer and
the honesty banners (degraded paths, small samples, missing data). Pure functions."""

from __future__ import annotations

from typing import Any

from steam_game_analyst.ui.results import Banner, parse_content, route_of

TOOL_VERB = {
    "query_catalog": "Queried the catalog",
    "review_sentiment": "Ran review_sentiment",
    "review_trend": "Ran review_trend",
    "predict_reception": "Ran predict_reception",
}


def elapsed(seconds: float) -> str:
    s = max(int(round(seconds)), 0)
    return f"{s // 60}:{s % 60:02d}"


def _games(games: list[dict[str, Any]]) -> str:
    return ", ".join(g["name"] for g in games)


def _router_line(update: dict[str, Any]) -> str:
    if not route_of(update).get("in_scope", True):
        return "Understood the question · outside what this tool covers"
    games = update.get("games") or []
    return "Understood the question" + (f" · {_games(games)}" if games else "")


def _agent_line(update: dict[str, Any]) -> str:
    msgs = update.get("messages") or []
    calls = getattr(msgs[-1], "tool_calls", None) if msgs else None
    if calls:
        names = ", ".join(dict.fromkeys(c["name"] for c in calls))
        return f"Planned the next step · calling {names}"
    return "Gathered enough evidence"


def _tools_line(update: dict[str, Any], seen: int) -> str:
    new = (update.get("tool_results") or [])[seen:]
    if not new:
        return "Tool call refused: the model will correct it"
    return " · ".join(f"[{t['id']}] {TOOL_VERB.get(t['tool'], t['tool'])}" for t in new)


def progress_line(node: str, update: dict[str, Any], seen_results: int = 0) -> str:
    """One finished graph node as a progress line. `seen_results` = tool results already shown."""
    if node == "router":
        return _router_line(update)
    if node == "retrieve":
        if "chunks" not in update:
            return "Document search unavailable · continuing with tools only"
        return f"Searched sources · {len(update['chunks'])} found"
    if node == "agent":
        return _agent_line(update)
    if node == "tools":
        return _tools_line(update, seen_results)
    if node == "answer":
        return "Wrote the answer"
    if node == "refuse":
        return "Declined: out of scope"
    return node


def step_strip(result: dict[str, Any], latency_s: float | None = None) -> list[str]:
    """The one-line summary above an answer: game, sources, tools, time."""
    out = [f"Found game: {g['name']}" for g in result.get("games", [])]
    if result.get("chunks"):
        out.append(f"Searched {len(result['chunks'])} sources")
    tools = [t["tool"] for t in result.get("tool_results", [])]
    out += [f"Ran {name}" for name in dict.fromkeys(tools)]
    if latency_s is not None:
        out.append(f"{latency_s:.0f} s")
    return out


def _tool_banners(t: dict[str, Any]) -> list[Banner]:
    out = parse_content(t.get("content", ""))
    found: list[Banner] = []
    if t["tool"] == "review_sentiment" and out.get("small_sample"):
        found.append(
            Banner(
                "warn",
                f"Small sample in [{t['id']}]: only {out.get('n_reviews_sampled')} sampled "
                f"reviews (effective n {out.get('effective_n')}). Treat the shares as indicative.",
            )
        )
    if t["tool"] == "review_trend" and out.get("limitation"):
        found.append(Banner("info", f"[{t['id']}] {out['limitation']}."))
    elif t["tool"] == "review_trend" and out.get("note"):
        found.append(Banner("info", f"[{t['id']}] {out['note']}."))
    if t["tool"] == "predict_reception":
        found += [Banner("warn", f"[{t['id']}] {w}.") for w in out.get("warnings", [])]
    if out.get("_truncated"):
        found.append(Banner("info", f"[{t['id']}] output was cut at the tool's size cap."))
    return found


def banners(result: dict[str, Any]) -> list[Banner]:
    """Degraded-path notes first (the answer is weaker than it looks), then data caveats."""
    found = [Banner("warn", note[:1].upper() + note[1:] + ".") for note in result.get("notes", [])]
    for t in result.get("tool_results", []):
        found += _tool_banners(t)
    return found
