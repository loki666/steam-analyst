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


def test_small_sample_and_truncated_outputs_are_flagged():
    t = {"id": "T1", "tool": "review_sentiment"}
    small = {**t, "content": '{"small_sample":true,"n_reviews_sampled":22,"effective_n":14.0}'}
    cut = {**t, "content": '{"game":{"appid":1,"name":"x"},"aspects":{"bugs":{"n_m'}
    texts = [b.text for b in steps.banners({"tool_results": [small, cut]})]
    assert "only 22 sampled reviews (effective n 14.0)" in texts[0]
    assert "size cap" in texts[1]
