"""Plotly figures for the tool cards, built from real tool outputs."""

import json
from pathlib import Path

import pytest

pytest.importorskip("plotly")

from steam_game_analyst.ui import charts  # noqa: E402
from steam_game_analyst.ui.results import parse_content  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "ui"


def out(name: str, i: int = 0) -> dict:
    t = json.loads((FIX / f"{name}.json").read_text())["tool_results"][i]
    return parse_content(t["content"])


def test_trend_figure_plots_share_and_volume_and_marks_the_shift():
    fig = charts.trend_figure(out("valheim_trend"))
    assert fig is not None
    share, volume = fig.data[0], fig.data[1]
    assert share.name == "positive share" and len(share.x) == 12
    assert share.y[-1] == pytest.approx(0.9212)
    assert volume.type == "bar" and sum(volume.y) == 32183
    levels = [d for d in fig.data if d.name == "average before / after"]
    assert [lv.y[0] for lv in levels] == [0.8641, 0.8976]
    assert any("shift, Sep 2024" in a.text for a in fig.layout.annotations)
    assert fig.layout.shapes and fig.layout.shapes[0].x0 == "2024-09-01"


def test_trend_figure_marks_the_event_date_of_a_before_after_test():
    fig = charts.trend_figure(out("cyberpunk_patch", 0))
    assert fig is not None
    assert any(a.text == "event 2023-09-21" for a in fig.layout.annotations)
    assert len([d for d in fig.data if d.name == "average before / after"]) == 2


def test_no_series_means_no_trend_chart():
    assert charts.trend_figure(out("starfield_no_data")) is None


def test_aspect_figure_overlays_complaints_on_mentions():
    fig = charts.aspect_figure(out("hades_complaints")["aspects"])
    assert fig is not None
    mentions, complaints = fig.data
    assert len(mentions.y) == 6 and mentions.y[-1] == "Gameplay"  # most mentioned on top
    assert complaints.marker.color == charts.WARN
    # one "mention · complaint" label per row, outside the bar
    assert mentions.texttemplate == "%{x:.1%} · %{customdata:.1%}"
    assert mentions.textposition == "outside" and mentions.customdata[-1] == 0.0353
    assert charts.aspect_figure({}) is None


def test_charts_are_transparent_so_the_card_is_the_only_surface():
    fig = charts.trend_figure(out("valheim_trend"))
    assert fig is not None
    assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == charts.CLEAR
    assert "Noto Sans CJK SC" in fig.layout.font.family


def test_comparison_figure_has_one_dumbbell_per_aspect_with_p_labels():
    fig = charts.comparison_figure(out("cyberpunk_patch", 1)["comparison"])
    assert fig is not None
    before, after = fig.data[-2], fig.data[-1]
    assert (before.name, after.name) == ("before", "after")
    assert before.marker.color == charts.MUTED and after.marker.color == charts.ACCENT
    assert before.cliponaxis is False and fig.layout.xaxis.range[0] < 0
    # story and performance have no complaints either side: left out
    assert list(before.y) == ["Gameplay", "Controls", "Price", "Bugs"]
    labels = {a.y: (a.text, a.font.color) for a in fig.layout.annotations}
    assert labels["Controls"] == ("p 0.049", charts.ACCENT)
    assert labels["Bugs"] == ("n.s.", charts.MUTED)


def test_p_mark_flags_only_significant_rows():
    assert charts.p_mark(0.01) == ("p 0.01", charts.ACCENT)
    assert charts.p_mark(0.2) == ("n.s.", charts.MUTED)
    assert charts.p_mark(None) == ("p n/a", charts.MUTED)


def test_reception_figures_show_buckets_and_signed_shap():
    o = out("reception")
    buckets = charts.reception_figure(o)
    assert buckets is not None
    assert list(buckets.data[0].x)[::-1] == [0.7399, 0.1305, 0.1296]
    shap = charts.shap_figure(o)
    assert shap is not None
    colors = list(shap.data[0].marker.color)
    assert colors[-1] == charts.WARN  # price pushes towards the predicted bucket
    assert charts.ACCENT in colors  # release year pushes away
    assert charts.shap_figure({}) is None and charts.reception_figure({}) is None
