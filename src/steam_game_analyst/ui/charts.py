"""Plotly figures for tool outputs (parsed dicts from `results.parse_content`). Pure: each
function returns a Figure, or None when the output has nothing to plot.

Colours follow the design system: signal blue for what was measured, burnt orange for
complaints and warnings, a grey-green for neutral fills. Blue vs orange also differ in
lightness, so the charts read in greyscale."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from steam_game_analyst.ui.views import BUCKET_LABEL, feature_label, month_label

ACCENT = "#2340C8"
WARN = "#A04E0C"
MENTION = "#B9C2BB"
MUTED = "#55605B"
INK = "#18201D"
LINE = "#D2D8D0"
SURFACE = "#FFFFFF"
FONT = "Public Sans, system-ui, sans-serif"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
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
    fig = go.Figure()
    fig.add_bar(
        y=names,
        x=[v.get("mention_share") or 0 for _, v in rows],
        orientation="h",
        name="mentions the aspect",
        marker={"color": MENTION},
        width=0.7,
        hovertemplate="%{y}: %{x:.1%} mention it<extra></extra>",
    )
    fig.add_bar(
        y=names,
        x=[v.get("complaint_share") or 0 for _, v in rows],
        orientation="h",
        name="complains about it",
        marker={"color": WARN},
        width=0.35,
        hovertemplate="%{y}: %{x:.1%} complain<extra></extra>",
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(tickformat=".0%", title_text="share of all reviews")
    return _layout(fig, 70 + 44 * len(rows))


def _p_text(p: float | None) -> str:
    return "p n/a" if p is None else f"p {p:.2g}"


def comparison_figure(comparison: dict[str, Any]) -> go.Figure | None:
    """review_sentiment split_date: complaint share before vs after, one dumbbell per aspect."""
    by_aspect = comparison.get("complaint_share_by_aspect") or {}
    rows = [(k, v) for k, v in by_aspect.items() if v.get("before") is not None]
    if not rows:
        return None
    rows.sort(key=lambda kv: max(kv[1]["before"], kv[1].get("after") or 0))
    labels = [f"{k.capitalize()} · {_p_text(v.get('p_value'))}" for k, v in rows]
    fig = go.Figure()
    for label, (_, v) in zip(labels, rows, strict=True):
        fig.add_scatter(
            x=[v["before"], v.get("after")],
            y=[label, label],
            mode="lines",
            line={"color": LINE, "width": 3},
            showlegend=False,
            hoverinfo="skip",
        )
    for side, color in (("before", MENTION), ("after", ACCENT)):
        fig.add_scatter(
            x=[v.get(side) for _, v in rows],
            y=labels,
            mode="markers",
            name=side,
            marker={"color": color, "size": 13},
            hovertemplate="%{y}: %{x:.1%} " + side + "<extra></extra>",
        )
    fig.update_xaxes(
        tickformat=".0%", title_text="share of reviews complaining", rangemode="tozero"
    )
    return _layout(fig, 70 + 44 * len(rows))


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
