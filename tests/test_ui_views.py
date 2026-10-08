"""Card titles, meta lines and metric tiles: every value is read from the real tool output."""

import json
from pathlib import Path

from steam_game_analyst.ui import views
from steam_game_analyst.ui.results import parse_content

FIX = Path(__file__).parent / "fixtures" / "ui"


def tool(name: str, i: int = 0) -> tuple[dict, dict]:
    t = json.loads((FIX / f"{name}.json").read_text())["tool_results"][i]
    return t, parse_content(t["content"])


def test_trend_card_with_a_change_point():
    _, out = tool("valheim_trend")
    assert views.trend_title(out) == "Positive share by month · Valheim"
    assert views.trend_meta(out) == "Dec 2023 – Nov 2024 · every review, 32,183 in total"
    assert views.trend_tiles(out) == [
        ("flat", "trend · -0.1 pts/year, p 0.98"),
        ("86% → 90%", "most likely shift, Sep 2024"),
        ("2.9k → 2.5k", "reviews / period, 1st vs 2nd half"),
    ]


def test_trend_card_with_an_event_test():
    _, out = tool("cyberpunk_patch", 0)
    value, caption = views.trend_tiles(out)[1]
    assert caption.startswith("positive before / after event, p")
    assert "→" in value


def test_catalog_tiles_for_a_game_without_reviews():
    _, out = tool("starfield_no_data")
    assert views.catalog_tiles(out) == [
        ("56.8%", "positive, all-time votes"),
        ("165,527", "votes on Steam"),
        ("5 Sep 2023", "released"),
    ]


def test_sentiment_tiles_and_meta():
    _, out = tool("hades_complaints")
    assert views.sentiment_meta(out) == (
        "6 May 2024 – 24 Nov 2024 · 768 sampled reviews · effective n 523.5"
    )
    tiles = views.sentiment_tiles(out)
    assert tiles[:2] == [
        ("95.8%", "positive (Steam votes)"),
        ("94.3%", "positive (classifier)"),
    ]
    assert tiles[2] == ("3.5%", "complain about gameplay")


def test_comparison_tiles_read_the_split():
    _, out = tool("cyberpunk_patch", 1)
    assert views.comparison_tiles(out["comparison"]) == [
        ("91% → 96%", "positive reviews · p 0.55"),
        ("7.2 / 24.8", "effective reviews before / after"),
    ]


def test_reception_meta_and_feature_labels():
    t, out = tool("reception")
    assert views.reception_meta(t["args"], out) == (
        "tags: Co-op, Survival, Crafting, Open World · $25.00 · 2026-11-01"
    )
    labels = [views.feature_label(f) for f in out["top_features"]]
    assert labels == [
        "Price $25.00",
        "No Steam Cloud",
        "Tag: Survival",
        "Release year 2026",
        "Developer track record unknown",
        "No Mac version",
        "Tag: Crafting",
        "Tag: Open World",
    ]
    assert views.feature_label({"feature": "n_languages", "value": 3.0}) == "n languages = 3"


def test_small_formatters():
    assert views.p_label(4e-14) == "p < 0.001" and views.p_label(None) == "p n/a"
    assert views.fmt_k(2460.8) == "2.5k" and views.fmt_k(181.7) == "182"
    assert views.month_label("2024-09-01") == "Sep 2024" and views.day_label(None) == "?"
