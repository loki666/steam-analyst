"""Streamlit UI (W5): chat with the agent, cited answers, tool cards with charts, tool trace.

    streamlit run src/steam_game_analyst/ui/app.py      (docker compose serves it on :8501)

Thin on purpose: what to show comes from the pure modules (results, steps, views, charts);
this file holds the session state, the cached backends and the run loop."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from steam_game_analyst.agent.factory import append_trace, trace_record
from steam_game_analyst.config import get_settings
from steam_game_analyst.ui import render, services
from steam_game_analyst.ui.steps import elapsed, progress_line


@st.cache_resource(show_spinner="Loading the search index…")
def get_retriever(mode: str) -> Any:
    return services.build_retriever(get_settings(), mode)


@st.cache_resource(show_spinner="Loading the agent…")
def get_agent(mode: str, with_search: bool) -> Any:
    retriever = get_retriever(mode) if with_search else None
    return services.build(get_settings(), retriever)


@st.cache_data(ttl=600, show_spinner=False)
def corpus_facts() -> str | None:
    return services.corpus_facts(get_settings())


def ask(question: str) -> None:
    st.session_state["pending"] = question


def retry(index: int) -> None:
    turn = st.session_state["history"].pop(index)
    ask(turn["question"])


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## Steam Game Analyst")
        if facts := corpus_facts():
            st.caption(facts)
        quality = st.radio(
            "Search quality",
            list(services.SEARCH_MODES),
            help="Thorough re-ranks the top 50 sources with a cross-encoder.",
        )
        if st.button("New conversation", width="stretch"):
            st.session_state["history"] = []
        st.caption("Each question is answered on its own: the agent keeps no chat memory.")
    return services.SEARCH_MODES[quality]


def _progress(box: Any, start: float) -> tuple[list[str], Any]:
    lines: list[str] = []
    shown = [0]  # tool results already reported

    def on_step(node: str, update: dict[str, Any]) -> None:
        lines.append(progress_line(node, update, shown[0]))
        shown[0] = len(update.get("tool_results") or []) or shown[0]
        box.write(f"**{lines[-1]}**")
        box.update(label=f"Working on it · {elapsed(time.perf_counter() - start)}", expanded=True)

    return lines, on_step


def execute(question: str, mode: str) -> dict[str, Any]:
    """Run one question with live progress; a failure becomes a turn, never a traceback."""
    settings = get_settings()
    down = services.health(settings)
    if down["ollama"]:
        return {"question": question, "failure": services.llm_down(settings, down["ollama"])}
    start = time.perf_counter()
    try:
        agent = get_agent(mode, with_search=down["qdrant"] is None)
        with st.status("Working on it", expanded=True) as box:
            box.caption("The local model usually answers in 10–90 s. Steps appear as they finish.")
            lines, on_step = _progress(box, start)
            result = agent.run(question, on_step=on_step)
            box.update(label=f"Done in {elapsed(time.perf_counter() - start)}", state="complete")
    except Exception as e:  # noqa: BLE001 — shown as a card with the details folded away
        return {"question": question, "failure": services.classify(e, settings)}
    latency = time.perf_counter() - start
    result.pop("messages", None)
    try:
        append_trace(
            settings.data_dir / "traces" / "agent.jsonl",
            trace_record(result, latency, settings.llm_model),
        )
    except OSError as e:
        st.toast(f"Could not write the trace: {e}")
    note = None if down["qdrant"] is None else "Document search is unavailable (Qdrant down)."
    return {
        "question": question,
        "result": result,
        "latency_s": latency,
        "progress": lines,
        "service_note": note,
    }


def show_turn(i: int, turn: dict[str, Any], mode: str) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"].replace("$", r"\$"))
    with st.chat_message("assistant"):
        if turn.get("chunks") is not None:
            render.sources_only(turn["chunks"])
        elif turn.get("failure") is not None:
            searchable = services.health(get_settings())["qdrant"] is None
            if render.failure(turn["failure"], f"t{i}", lambda: retry(i), searchable):
                turn["chunks"] = services.search_only(get_retriever(mode), turn["question"])
                st.rerun()
        else:
            if turn.get("service_note"):
                st.info(turn["service_note"] + " This answer uses the tools only.")
            render.answer(turn, f"t{i}")


def main() -> None:
    st.set_page_config(page_title="Steam Game Analyst", page_icon=":material/insights:")
    mode = sidebar()
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    typed = st.chat_input("Ask about a game, a patch, or the catalog…")  # pinned to the bottom
    question = typed or st.session_state.pop("pending", None)
    intro = st.empty()  # same slot every run, so the examples clear as soon as a run starts
    if not history and not question:
        with intro.container():
            render.empty_state(ask)
    else:
        intro.empty()
    for i, turn in enumerate(history):
        show_turn(i, turn, mode)
    if question:
        with st.chat_message("user"):
            st.markdown(question.replace("$", r"\$"))
        with st.chat_message("assistant"):
            turn = execute(question, mode)
        history.append(turn)
        st.rerun()


main()
