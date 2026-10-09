"""The Streamlit app end to end with AppTest: a fake agent replays real captured states, so no
LLM, Qdrant or GPU is needed."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from steam_game_analyst.agent.state import RouterDecision  # noqa: E402
from steam_game_analyst.ui import render, services  # noqa: E402

APP = Path(__file__).parents[1] / "src" / "steam_game_analyst" / "ui" / "app.py"
FIX = Path(__file__).parent / "fixtures" / "ui"


class ReplayAgent:
    def __init__(self, name: str) -> None:
        self.state = json.loads((FIX / f"{name}.json").read_text())
        # the graph returns the router's pydantic model, the fixture stores its dump
        self.state["route"] = RouterDecision.model_validate(self.state["route"])

    def run(self, question, on_step=None):
        if on_step:
            on_step("router", {"games": self.state["games"], "route": self.state["route"]})
            on_step("answer", {})
        return dict(self.state)


def walk(node):
    """Every element under an AppTest block, in render order."""
    yield node
    for child in getattr(node, "children", {}).values():
        yield from walk(child)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from steam_game_analyst.config import get_settings

    monkeypatch.setenv("SGA_DATA_DIR", str(tmp_path / "data"))  # traces land here
    get_settings.cache_clear()
    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(services, "health", lambda s: {"llm": None, "qdrant": None})
    monkeypatch.setattr(services, "build_retriever", lambda s, mode: object())
    yield lambda: AppTest.from_file(str(APP), default_timeout=60)
    get_settings.cache_clear()


def test_empty_state_offers_example_questions(app):
    at = app().run()
    assert not at.exception
    assert any("Ask about a Steam game" in m.value for m in at.markdown)
    keys = [b.key for b in at.button if (b.key or "").startswith("example-")]
    assert keys == [f"example-{i}" for i in range(4)]


def test_an_answered_question_renders_cards_sources_and_trace(app, monkeypatch):
    monkeypatch.setattr(services, "build", lambda s, r: ReplayAgent("hades_complaints"))
    at = app().run()
    at.button(key="example-0").click().run()
    assert not at.exception
    assert any(":gray-badge[T1]" in m.value for m in at.markdown)
    assert [e.label for e in at.expander][:2] == ["Sources (1 cited)", "Tool trace (1 call)"]
    assert [t.label for t in at.tabs] == ["Cited (1)", "Also retrieved (5)"]
    assert any(m.value.startswith(":blue-badge[1] Hades II · review") for m in at.markdown)
    title = '<span class="sga-title">Review sentiment · Hades II</span>'
    assert any(title in m.value for m in at.markdown)
    tiles = {m.label: m for m in at.metric}
    assert tiles["positive (Steam votes)"].value == "95.8%"
    assert tiles["positive (Steam votes)"].help.startswith("Share of reviews voted up")
    assert [c.avatar for c in at.chat_message] == [render.USER_AVATAR, render.BOT_AVATAR]
    assert at.session_state["history"][0]["progress"] == [
        "Understood the question · Hades II",
        "Wrote the answer",
    ]


def test_out_of_scope_question_is_declined_with_example_buttons(app, monkeypatch):
    monkeypatch.setattr(services, "build", lambda s, r: ReplayAgent("refusal"))
    at = app()
    at.run()
    at.chat_input[0].set_value("What should I play tonight?").run()
    assert not at.exception
    assert any(m.value.startswith("I can only answer questions") for m in at.markdown)
    assert not at.expander and not at.metric
    tries = [b for b in at.button if (b.key or "").startswith("t0-try-")]
    assert [b.label for b in tries] == list(render.EXAMPLES[:2])
    tries[0].click().run()  # asks the example as the next question
    assert not at.exception
    assert [t["question"] for t in at.session_state["history"]] == [
        "What should I play tonight?",
        render.EXAMPLES[0],
    ]


def test_model_down_keeps_the_question_and_explains_the_fix(app, monkeypatch):
    monkeypatch.setattr(
        services, "health", lambda s: {"llm": "ConnectError: refused", "qdrant": None}
    )
    at = app().run()
    at.button(key="example-1").click().run()
    assert not at.exception
    assert any(m.value == "**The language model isn't running**" for m in at.markdown)
    assert at.button(key="t0-retry") and at.button(key="t0-search")
    assert at.session_state["history"][0]["question"].startswith("Did performance complaints")


def test_split_warning_sits_inside_the_sentiment_card(app, monkeypatch):
    monkeypatch.setattr(services, "build", lambda s, r: ReplayAgent("cyberpunk_patch"))
    at = app().run()
    at.button(key="example-1").click().run()
    assert not at.exception
    text = "Only 7 effective reviews before and 25 after 21 Sep 2023"
    hits = [w for w in at.warning if w.value.startswith(text)]
    assert len(hits) == 1  # once, not also with the answer-level banners
    # in reading order: after the sentiment card's header, not above the trend card
    order = [getattr(n, "value", "") for n in walk(at.main)]
    head = next(i for i, v in enumerate(order) if "Review sentiment · Cyberpunk" in str(v))
    trend = next(i for i, v in enumerate(order) if "Positive share by month" in str(v))
    warn = next(i for i, v in enumerate(order) if str(v).startswith(text))
    assert trend < head < warn


def test_tool_errors_show_in_the_trace_and_as_a_warning(app, monkeypatch):
    monkeypatch.setattr(services, "build", lambda s, r: ReplayAgent("tool_errors"))
    at = app().run()
    at.chat_input[0].set_value("Which roguelikes released in 2024 are best under $20?").run()
    assert not at.exception
    assert any(w.value == "Stopped after 4 invalid tool calls." for w in at.warning)
    assert any(e.label == "Tool trace (7 calls)" for e in at.expander)
