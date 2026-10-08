"""Plotly figures for tool outputs (parsed dicts from `results.parse_content`). Pure: each
function returns a Figure, or None when the output has nothing to plot.

Colours follow the design system: signal blue for what was measured, burnt orange for
complaints and warnings, a grey-green for neutral fills. Blue vs orange also differ in
lightness, so the charts read in greyscale."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from steam_game_analyst.ui.views import BUCKET_LABEL, comparison_rows, feature_label, month_label

ACCENT = "#2340C8"
WARN = "#A04E0C"
MENTION = "#B9C2BB"
MUTED = "#55605B"
INK = "#18201D"
LINE = "#D2D8D0"
CLEAR = "rgba(0,0,0,0)"  # the chart sits on its card: one surface, no white box
# CJK fallbacks: review text includes Chinese and Japanese
FONT = '"Public Sans", "Noto Sans CJK SC", "Noto Sans", system-ui, sans-serif'
SIGNIFICANCE = 0.05


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        paper_bgcolor=CLEAR,
        plot_bgcolor=CLEAR,
        font={"family": FONT, "color": INK, "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"font": {"family": FONT}},
    )
    # automargin is off by default in plotly.js: tick labels and titles get clipped
    fig.update_xaxes(gridcolor=LINE, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, automargin=True)
    return fig


def _marker_line(fig: go.Figure, when: str, label: str) -> None:
    """Dashed vertical line with a label, on the share panel (row 1)."""
    fig.add_vline(x=when, line={"color": MUTED, "dash": "dot", "width": 1}, row=1, col=1)
    fig.add_annotation(
        x=when,
        y=1.0,
        yref="y domain",
        text=label,
        showarrow=False,
        xanchor="left",
        font={"color": MUTED, "size": 11},
        row=1,
        col=1,
    )


def _level(fig: go.Figure, x0: str, x1: str, share: float | None, name: str, first: bool) -> None:
    if share is None or x0 is None or x1 is None:
        return
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[share, share],
            mode="lines",
            name=name,
            showlegend=first,
            line={"color": WARN, "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )


def _shift_overlay(fig: go.Figure, out: dict[str, Any], periods: list[str]) -> None:
    """The event date (before/after test) or the most likely change point, with the average
    positive share on each side as dashed levels."""
    name = "average before / after"
    if ev := out.get("event_test"):
        _marker_line(fig, ev["event_date"], f"event {ev['event_date']}")
        b, a = ev.get("before", {}), ev.get("after", {})
        _level(fig, b.get("from"), b.get("to"), b.get("positive_share"), name, True)
        _level(fig, a.get("from"), a.get("to"), a.get("positive_share"), name, False)
    elif cp := out.get("change_point"):
        _marker_line(fig, cp["period"], f"shift, {month_label(cp['period'])}")
        _level(fig, periods[0], cp["period"], cp.get("positive_share_before"), name, True)
        _level(fig, cp["period"], periods[-1], cp.get("positive_share_after"), name, False)


def trend_figure(out: dict[str, Any]) -> go.Figure | None:
    """review_trend: positive share by period (line) over review volume (bars)."""
    series = [p for p in out.get("series", []) if p.get("period")]
    if not series:
        return None
    periods = [p["period"] for p in series]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=[p.get("positive_share") for p in series],
            name="positive share",
            mode="lines+markers",
            line={"color": ACCENT, "width": 2.5},
            marker={"size": 6},
            connectgaps=False,
            hovertemplate="%{x|%b %Y}: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=periods,
            y=[p.get("n_reviews", 0) for p in series],
            name="reviews per period",
            marker={"color": MENTION},
            hovertemplate="%{x|%b %Y}: %{y:,} reviews<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _shift_overlay(fig, out, periods)
    fig.update_yaxes(tickformat=".0%", title_text="positive", row=1, col=1)
    fig.update_yaxes(title_text="reviews", row=2, col=1)
    return _layout(fig, 420)


def aspect_figure(aspects: dict[str, Any]) -> go.Figure | None:
    """review_sentiment: share of reviews mentioning each aspect, and complaining about it."""
    rows = [(k, v) for k, v in aspects.items() if isinstance(v, dict)]
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1].get("mention_share") or 0)
    names = [k.capitalize() for k, _ in rows]
    mentions = [v.get("mention_share") or 0 for _, v in rows]
    complaints = [v.get("complaint_share") or 0 for _, v in rows]
    fig = go.Figure()
    fig.add_bar(
        y=names,
        x=mentions,
        customdata=complaints,
        orientation="h",
        name="mentions the aspect",
        marker={"color": MENTION},
        width=0.7,
        texttemplate="%{x:.1%} · %{customdata:.1%}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%} mention it<extra></extra>",
    )
    fig.add_bar(
        y=names,
        x=complaints,
        orientation="h",
        name="complains about it",
        marker={"color": WARN},
        width=0.35,
        hovertemplate="%{y}: %{x:.1%} complain<extra></extra>",
    )
    fig.update_layout(barmode="overlay", uniformtext={"minsize": 11})
    # head room on the right for the "mention · complaint" labels
    fig.update_xaxes(
        tickformat=".0%", title_text="share of all reviews", range=[0, max(mentions) * 1.35]
    )
    return _layout(fig, 70 + 44 * len(rows))


def p_mark(p: float | None) -> tuple[str, str]:
    """Right-hand label of a dumbbell row and its colour: accent when significant."""
    if p is None:
        return "p n/a", MUTED
    if p < SIGNIFICANCE:
        return f"p {p:.2g}", ACCENT
    return "n.s.", MUTED


def _dumbbell_labels(fig: go.Figure, rows: list[tuple[str, float, float, float | None]]) -> None:
    for aspect, _, _, p in rows:
        text, color = p_mark(p)
        fig.add_annotation(
            x=1.0,
            xref="paper",
            xanchor="left",
            y=aspect.capitalize(),
            text=text,
            showarrow=False,
            font={"color": color, "size": 12},
        )


def comparison_figure(comparison: dict[str, Any]) -> go.Figure | None:
    """review_sentiment split_date: complaint share before (muted) vs after (accent), one
    dumbbell per aspect; aspects with no complaints on either side are left out."""
    rows = sorted(comparison_rows(comparison), key=lambda r: max(r[1], r[2]))
    if not rows:
        return None
    names = [r[0].capitalize() for r in rows]
    fig = go.Figure()
    for name, (_, before, after, _) in zip(names, rows, strict=True):
        fig.add_scatter(
            x=[before, after],
            y=[name, name],
            mode="lines",
            line={"color": LINE, "width": 3},
            showlegend=False,
            hoverinfo="skip",
        )
    for i, (side, color) in enumerate((("before", MUTED), ("after", ACCENT)), start=1):
        fig.add_scatter(
            x=[r[i] for r in rows],
            y=names,
            mode="markers",
            name=side,
            cliponaxis=False,
            marker={"color": color, "size": 13},
            hovertemplate="%{y}: %{x:.1%} " + side + "<extra></extra>",
        )
    _dumbbell_labels(fig, rows)
    top = max(max(r[1], r[2]) for r in rows)
    fig.update_xaxes(
        tickformat=".0%", title_text="share of reviews complaining", range=[-0.02 * top, top * 1.08]
    )
    fig = _layout(fig, 70 + 44 * len(rows))
    fig.update_layout(margin={"r": 64})  # room for the p labels
    return fig


def reception_figure(out: dict[str, Any]) -> go.Figure | None:
    """predict_reception: probability of each review-ratio bucket."""
    probs = out.get("probabilities") or {}
    if not probs:
        return None
    colors = {"negative_or_mixed": WARN, "mostly_positive": MENTION, "very_positive": ACCENT}
    keys = [k for k in colors if k in probs][::-1]
    fig = go.Figure(
        go.Bar(
            y=[BUCKET_LABEL.get(k, k) for k in keys],
            x=[probs[k] for k in keys],
            orientation="h",
            marker={"color": [colors[k] for k in keys]},
            text=[f"{probs[k]:.1%}" for k in keys],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1.1])
    return _layout(fig, 200)


def shap_figure(out: dict[str, Any]) -> go.Figure | None:
    """predict_reception: SHAP contributions towards (orange) or away from (blue) the
    predicted bucket, largest first."""
    top = out.get("top_features") or []
    if not top:
        return None
    rows = top[::-1]
    bucket = BUCKET_LABEL.get(out.get("bucket", ""), out.get("bucket", ""))
    fig = go.Figure(
        go.Bar(
            y=[feature_label(f) for f in rows],
            x=[f["shap"] for f in rows],
            orientation="h",
            marker={"color": [WARN if f["shap"] > 0 else ACCENT for f in rows]},
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=f"← away from · towards {bucket.lower()} →", zeroline=True)
    return _layout(fig, 70 + 36 * len(rows))
