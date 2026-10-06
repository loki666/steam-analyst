"""review_trend over the population monthly counts (review_monthly)."""

from datetime import date

import pytest

from steam_game_analyst.tools.catalog import CatalogDB
from steam_game_analyst.tools.trend import ReviewTrend, TrendArgs, aggregate, month_add


@pytest.fixture
def trend(built):
    db = CatalogDB(built.duckdb_path)
    yield ReviewTrend(db)
    db.close()


def test_month_add():
    assert month_add(date(2024, 11, 1), 3) == date(2025, 2, 1)
    assert month_add(date(2024, 1, 1), -1) == date(2023, 12, 1)


def test_aggregate_to_quarters():
    rows = [(date(2024, 1, 1), 10, 5), (date(2024, 2, 1), 10, 10), (date(2024, 4, 1), 4, 1)]
    q = aggregate(rows, "quarter")
    assert q == [(date(2024, 1, 1), 20, 15), (date(2024, 4, 1), 4, 1)]


def test_monthly_series_counts_the_population(trend):
    out = trend.run(TrendArgs(appid=1091500))
    assert out["game"]["name"] == "Cyberpunk 2077"
    total = sum(p["n_reviews"] for p in out["series"])
    assert total == out["summary"]["n_reviews"] > 30  # population, not the 30 sampled
    assert out["trend"]["direction"] in {"up", "down", "flat"}
    assert "change_point" in out and "event_test" not in out


def test_last_n_months_is_relative_to_the_data(trend):
    out = trend.run(TrendArgs(appid=1091500, last_n_months=12))
    assert len(out["series"]) <= 12
    assert out["window"]["to"] == out["data_through"]


def test_event_date_runs_a_before_after_test(trend):
    out = trend.run(TrendArgs(appid=1091500, event_date=date(2022, 6, 15), window_months=6))
    ev = out["event_test"]
    assert ev["before"]["to"] < "2022-06-15" <= ev["after"]["from"]
    assert 0 <= ev["p_value"] <= 1
    assert ev["diff"] == pytest.approx(
        ev["after"]["positive_share"] - ev["before"]["positive_share"]
    )


def test_unknown_game_reports_catalog_facts(trend):
    out = trend.run(TrendArgs(appid=999002))  # in the catalog, no reviews
    assert "error" in out and out["catalog"]["name"] == "Unreleased Thing"
    out = trend.run(TrendArgs(appid=123))
    assert "error" in out and out.get("catalog") is None


def test_args_validate():
    with pytest.raises(ValueError):
        TrendArgs(appid=1, granularity="day")
    with pytest.raises(ValueError):
        TrendArgs(appid=1, last_n_months=0)


def test_trend_slope_uses_calendar_months_not_row_positions():
    # two points 12 months apart, +12 pp: 0.12 per year even though they are adjacent rows
    rows = [
        (date(2023, 1, 1), 1000, 500),
        (date(2023, 2, 1), 1000, 510),
        (date(2024, 1, 1), 1000, 620),
        (date(2024, 2, 1), 1000, 630),
    ]
    t = ReviewTrend._trend(rows)
    assert t["positive_share_change_per_year"] == pytest.approx(0.12, abs=0.005)


def test_window_is_anchored_to_the_corpus_end(trend):
    end = trend.corpus_last_month
    out = trend.run(TrendArgs(appid=1091500, last_n_months=6))
    assert out["window"]["to"] <= end.isoformat()
    assert out["data_through"] == end.isoformat()
    assert len(out["series"]) == 6  # months without reviews are listed with 0
