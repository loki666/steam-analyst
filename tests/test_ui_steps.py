"""Progress lines, step strip and honesty banners from real agent states."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from steam_game_analyst.ui import steps

FIX = Path(__file__).parent / "fixtures" / "ui"


def load(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text())


def test_progress_lines_name_what_each_node_did():
    state = load("cyberpunk_patch")
    router = {"games": state["games"], "route": state["route"]}
    assert steps.progress_line("router", router) == "Understood the question · Cyberpunk 2077"
    out_of_scope = {"route": load("refusal")["route"]}
    assert "outside" in steps.progress_line("router", out_of_scope)
    assert steps.progress_line("retrieve", {"chunks": state["chunks"]}) == (
        "Searched sources · 6 found"
    )
    assert "unavailable" in steps.progress_line("retrieve", {"notes": ["retrieval failed: x"]})
    calls = AIMessage(content="", tool_calls=[{"name": "review_trend", "args": {}, "id": "a"}])
    assert steps.progress_line("agent", {"messages": [calls]}).endswith("calling review_trend")
    assert steps.progress_line("agent", {"messages": [AIMessage("DONE")]}) == (
        "Gathered enough evidence"
    )


def test_tools_progress_reports_only_the_new_results():
    results = load("cyberpunk_patch")["tool_results"]
    update = {"tool_results": results}
    assert steps.progress_line("tools", update, seen_results=1) == "[T2] Ran review_sentiment"
    assert "refused" in steps.progress_line("tools", update, seen_results=2)
    assert steps.progress_line("answer", {}) == "Wrote the answer"


def test_step_strip_lists_games_sources_tools_and_time():
    assert steps.step_strip(load("valheim_trend"), 17.8) == [
        "Found game: Valheim",
        "Ran review_trend",
        "18 s",
    ]
    strip = steps.step_strip(load("cyberpunk_patch"))
    assert strip == [
        "Found game: Cyberpunk 2077",
        "Searched 6 sources",
        "Ran review_trend",
        "Ran review_sentiment",
    ]


def test_elapsed_is_minutes_and_seconds():
    assert steps.elapsed(83.4) == "1:23" and steps.elapsed(5) == "0:05"


def test_banners_put_degraded_paths_first_then_data_caveats():
    b = steps.banners(load("tool_errors"))
    assert b[0].kind == "warn" and b[0].text == "Stopped after 4 invalid tool calls."
    reception = steps.banners(load("reception"))
    assert reception[0].kind == "warn" and "only 4 known tags" in reception[0].text
    starfield = steps.banners(load("starfield_no_data"))
    assert [x.kind for x in starfield] == ["info"]
    assert "no reviews for this game" in starfield[0].text
    assert steps.banners(load("valheim_trend")) == []


def test_a_split_with_few_effective_reviews_is_warned_about():
    b = steps.banners(load("cyberpunk_patch"))  # T2: effective n 7.2 before, 24.8 after
    assert b == [
        steps.Banner(
            "warn",
            "Only 7 effective reviews before and 25 after 21 Sep 2023: "
            "differences under ~10 pts are noise.",
        )
    ]
    one_side = {"split_date": "2023-09-21", "effective_n": {"before": 40.0, "after": 12.4}}
    banner = steps.split_banner(one_side)
    assert banner is not None and banner.text.startswith("Only 12 effective reviews after 21 Sep")
    assert steps.split_banner({"effective_n": {"before": 30.0, "after": 99.0}}) is None


def test_next_activity_names_what_runs_after_each_node():
    route = load("cyberpunk_patch")["route"]
    assert steps.next_activity("router", {"route": route}) == "Searching sources…"
    no_docs = {**route, "needs_retrieval": False}
    assert steps.next_activity("router", {"route": no_docs}) == "Planning the next step…"
    assert steps.next_activity("router", {"route": load("refusal")["route"]}) is None
    calls = AIMessage(content="", tool_calls=[{"name": "review_trend", "args": {}, "id": "a"}])
    assert steps.next_activity("agent", {"messages": [calls]}) == "Running review_trend…"
    assert steps.next_activity("agent", {"messages": [AIMessage("DONE")]}) == (
        "Writing the answer…"
    )
    assert steps.next_activity("answer", {}) is None


def test_progress_log_checks_finished_steps_and_bolds_only_the_current_one():
    log = steps.ProgressLog()
    assert log.markdown() == "**Understanding the question…**"
    state = load("valheim_trend")
    log.feed("router", {"games": state["games"], "route": state["route"]})
    call = AIMessage("", tool_calls=[{"name": "review_trend", "args": {}, "id": "a"}])
    log.feed("agent", {"messages": [call]})
    log.feed("tools", {"tool_results": state["tool_results"]})
    assert log.markdown().split("  \n") == [
        "✓ Understood the question · Valheim",
        "✓ Planned the next step · calling review_trend",
        "✓ [T1] Ran review_trend",
        "**Planning the next step…**",
    ]
    log.feed("answer", {})
    assert log.current is None and not log.markdown().endswith("**")


def test_small_sample_and_truncated_outputs_are_flagged():
    t = {"id": "T1", "tool": "review_sentiment"}
    small = {**t, "content": '{"small_sample":true,"n_reviews_sampled":22,"effective_n":14.0}'}
    cut = {**t, "content": '{"game":{"appid":1,"name":"x"},"aspects":{"bugs":{"n_m'}
    texts = [b.text for b in steps.banners({"tool_results": [small, cut]})]
    assert "only 22 sampled reviews (effective n 14.0)" in texts[0]
    assert "size cap" in texts[1]
