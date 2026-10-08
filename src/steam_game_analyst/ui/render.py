"""Streamlit rendering of one agent turn. Thin: the content comes from the pure view models in
results, steps, views and charts; this module only lays it out."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import streamlit as st

from steam_game_analyst.ui import charts, views
from steam_game_analyst.ui.results import (
    Source,
    answer_markdown,
    is_refusal,
    parse_content,
    route_of,
    snippet,
    sources,
    tool_rows,
)
from steam_game_analyst.ui.services import Failure
from steam_game_analyst.ui.steps import banners, split_banner, step_strip
from steam_game_analyst.ui.style import CSS

EXAMPLES = (
    "What do players complain about most in Hades II?",
    "Did performance complaints drop after patch 2.0 of Cyberpunk 2077?",
    "Are reviews for Valheim trending up or down over the last 12 months?",
    "A co-op survival game at $25 with crafting and an open world launches in November 2026: "
    "how is it likely to be received?",
)
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/insights:"
STATUS_MARK = {"ok": ":gray[✓ ok]", "no data": ":orange[no data]", "error": ":red[✗ error]"}

Pick = Callable[[str], None]


def inject_style() -> None:
    st.html(f"<style>{CSS}</style>")


def _tiles(tiles: list[views.Tile]) -> None:
    if not tiles:
        return
    for col, tile in zip(st.columns(len(tiles)), tiles, strict=True):
        col.metric(tile.caption, tile.value, help=tile.help)


def _chart(fig: Any, key: str) -> None:
    if fig is not None:
        st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


def card_head_html(tid: str, title: str, meta: str) -> str:
    """T-id chip (same badge as the answer's citations), title in the heading font, meta in
    mono on the same row."""
    meta_html = f' <span class="sga-meta">{html.escape(meta)}</span>' if meta else ""
    return (
        f':gray-badge[{tid}] <span class="sga-title">{html.escape(title)}</span>{meta_html}'
    ).replace("$", r"\$")


def _card_head(tid: str, title: str, meta: str) -> None:
    st.markdown(card_head_html(tid, title, meta), unsafe_allow_html=True)


def _trend_card(t: dict[str, Any], out: dict[str, Any], key: str) -> None:
    if "limitation" in out:
        title = f"Catalog facts · {views.game_name(out)}"
        _card_head(t["id"], title, "no dated reviews in the corpus")
        _tiles(views.catalog_tiles(out))
        return
    _card_head(t["id"], views.trend_title(out), views.trend_meta(out))
    _chart(charts.trend_figure(out), f"{key}-trend")
    _tiles(views.trend_tiles(out))


def _comparison(cmp: dict[str, Any], key: str) -> None:
    st.markdown(f"**Complaints before vs after {views.day_label(cmp.get('split_date'))}**")
    if warning := split_banner(cmp):  # in the card it is about, not above the answer
        st.warning(warning.text)
    _chart(charts.comparison_figure(cmp), f"{key}-cmp")
    if quiet := views.quiet_aspects(cmp):
        st.caption("No complaints either side: " + ", ".join(a.capitalize() for a in quiet))
    _tiles(views.comparison_tiles(cmp))


def _sentiment_card(t: dict[str, Any], out: dict[str, Any], key: str) -> None:
    title = f"Review sentiment · {views.game_name(out)}"
    _card_head(t["id"], title, views.sentiment_meta(out))
    _tiles(views.sentiment_tiles(out))
    if cmp := out.get("comparison"):
        _comparison(cmp, key)
    _chart(charts.aspect_figure(out.get("aspects") or {}), f"{key}-aspects")
    st.caption(out.get("method", ""))


def _reception_card(t: dict[str, Any], out: dict[str, Any], key: str) -> None:
    meta = views.reception_meta(t.get("args", {}), out)
    _card_head(t["id"], "Reception forecast", meta)
    _chart(charts.reception_figure(out), f"{key}-buckets")
    st.markdown("**What drove it (SHAP)**")
    _chart(charts.shap_figure(out), f"{key}-shap")


def _catalog_card(t: dict[str, Any], out: dict[str, Any], key: str) -> None:
    _card_head(t["id"], "Catalog query", views.plural(out.get("row_count", 0), "row"))
    if out.get("no_match"):
        st.info(out["no_match"])
    if out.get("columns"):
        rows = [dict(zip(out["columns"], r, strict=False)) for r in out.get("rows", [])]
        st.dataframe(rows, hide_index=True, key=f"{key}-rows")
    st.code(t.get("args", {}).get("sql", ""), language="sql")


CARDS: dict[str, Callable[[dict[str, Any], dict[str, Any], str], None]] = {
    "review_trend": _trend_card,
    "review_sentiment": _sentiment_card,
    "predict_reception": _reception_card,
    "query_catalog": _catalog_card,
}


def tool_cards(result: dict[str, Any], key: str) -> None:
    """One bordered card per successful tool call, headed by its T-id."""
    for t in result.get("tool_results", []):
        out = parse_content(t.get("content", ""))
        card = CARDS.get(t["tool"])
        if card is None or "error" in out:
            continue
        with st.container(border=True):
            card(t, out, f"{key}-{t['id']}")


def _source_row(s: Source) -> None:
    link = f" · [open]({s.url})" if s.url else ""
    st.markdown(f":blue-badge[{s.n}] {html.escape(s.label)}{link}")
    st.markdown(f"> {html.escape(s.snippet)}".replace("$", r"\$"))
    if len(s.text) > len(s.snippet) + 20:
        body = html.escape(s.text).replace("\n", "<br>")
        st.html(f"<details><summary>Full excerpt</summary><p>{body}</p></details>")


def sources_panel(result: dict[str, Any]) -> None:
    """Cited sources in reading order; the retrieved but uncited ones in a second tab."""
    cited, rest = sources(result)
    if not cited and not rest:
        return
    label = f"Sources ({len(cited)} cited)" if cited else "Sources (none cited)"
    with st.expander(label, expanded=bool(cited)):
        tab_cited, tab_rest = st.tabs([f"Cited ({len(cited)})", f"Also retrieved ({len(rest)})"])
        with tab_cited:
            for s in cited:
                _source_row(s)
            if not cited:
                st.caption("The answer cites tool results only.")
        with tab_rest:
            for s in rest:
                _source_row(s)


def trace_panel(result: dict[str, Any], progress: list[str]) -> None:
    """Route and plan, every tool call with its outcome, and the degraded-path notes."""
    rows = tool_rows(result)
    with st.expander(f"Tool trace ({views.plural(len(rows), 'call')})"):
        route = route_of(result)
        games = ", ".join(f"{g['name']} ({g['appid']})" for g in result.get("games", []))
        st.markdown(f"**Games:** {games or 'none recognised'}")
        st.markdown(f"**Plan:** {route.get('plan') or result.get('plan') or '–'}")
        if route.get("retrieval_query"):
            st.markdown(f"**Search query:** {route['retrieval_query']}")
        for r in rows:
            summary = html.escape(r.summary).replace("$", r"\$")
            st.markdown(f"**{r.id}** `{r.tool}` {STATUS_MARK[r.status]} · {summary}")
            st.code(r.args, language=None, wrap_lines=True)
        for note in result.get("notes", []):
            st.warning(note)
        if progress:
            st.caption("Steps: " + " → ".join(progress))
        st.caption(f"{result.get('iterations', 0)} agent steps · {result.get('errors', 0)} errors")


def _banners(result: dict[str, Any]) -> None:
    for b in banners(result):
        (st.warning if b.kind == "warn" else st.info)(b.text)


def examples(on_pick: Pick, key: str, items: tuple[str, ...] = EXAMPLES) -> None:
    for i, q in enumerate(items):
        st.button(q, key=f"{key}-{i}", on_click=on_pick, args=(q,))


def answer(turn: dict[str, Any], key: str, on_pick: Pick) -> None:
    """A finished turn: step strip, answer with citations, caveats, tool cards, panels."""
    result = turn["result"]
    if is_refusal(result):
        st.markdown(result.get("answer", ""))
        st.caption("Try one of these:")
        examples(on_pick, f"{key}-try", EXAMPLES[:2])
        return
    st.caption(" · ".join(step_strip(result, turn.get("latency_s"))))
    st.markdown(answer_markdown(result))
    _banners(result)
    tool_cards(result, key)
    sources_panel(result)
    trace_panel(result, turn.get("progress", []))


def failure(f: Failure, key: str, on_retry: Callable[[], None], can_search: bool) -> bool:
    """A service or run failure as a card. Returns True when "Search sources only" is asked."""
    with st.container(border=True):
        st.markdown(f"**{f.title}**")
        st.markdown(f.body)
        with st.expander("Details"):
            st.code(f.detail, language=None, wrap_lines=True)
        with st.container(horizontal=True):
            st.button("Retry", key=f"{key}-retry", type="primary", on_click=on_retry)
            wanted = can_search and st.button("Search sources only", key=f"{key}-search")
        if can_search:
            st.caption(
                "“Search sources only” lists matching reviews, patch notes, news and wiki pages, "
                "without a written answer."
            )
    return bool(wanted)


def sources_only(chunks: list[dict[str, Any]]) -> None:
    st.caption(f"Search only · {len(chunks)} matching sources · no written answer")
    for c in chunks:
        _source_row(Source(int(c["n"]), c["label"], snippet(c["text"]), c["text"], c.get("url")))


def empty_state(on_pick: Pick) -> None:
    st.markdown("### Ask about a Steam game")
    st.markdown(
        "Every answer cites its sources [n] and the tool results [T1] it used. "
        "The local model takes 10–90 s."
    )
    examples(on_pick, "example")
