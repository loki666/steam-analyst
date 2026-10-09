"""review_sentiment: weighted sentiment and aspect distribution over the review sample."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from steam_game_analyst.tools.catalog import CatalogDB
from steam_game_analyst.tools.sentiment import ReviewSentiment, SentimentArgs, stratum_weights


class KeywordModel:
    """Stand-in classifier: negative when the text has a complaint word."""

    BAD = ("crash", "terrible", "bugs", "awful", "not worth")

    def predict_proba(self, texts):
        return np.array([0.1 if any(b in t.lower() for b in self.BAD) else 0.9 for t in texts])


@pytest.fixture
def tool(built):
    db = CatalogDB(built.duckdb_path)
    yield ReviewSentiment(db, built.processed_dir / "reviews.parquet", KeywordModel())
    db.close()


def test_stratum_weights_undo_the_stratified_sample():
    sample = pl.DataFrame(
        {"appid": [1, 1, 1, 1], "year": [2023] * 4, "voted_up": [True, True, False, False]}
    )
    population = pl.DataFrame(
        {"appid": [1], "year": [2023], "n_reviews": [100], "n_positive": [90]}
    )
    w = stratum_weights(sample, population)
    # 2 sampled positives stand for 90 reviews, 2 negatives for 10
    assert w.sort("voted_up")["weight"].to_list() == [5.0, 5.0, 45.0, 45.0]


def test_window_summary_has_aspects_and_weighted_share(tool):
    out = tool.run(
        SentimentArgs(appid=1091500, date_from=date(2021, 1, 1), date_to=date(2024, 12, 31))
    )
    assert out["game"]["name"] == "Cyberpunk 2077"
    assert out["n_reviews_sampled"] > 0
    assert 0 <= out["positive_share"] <= 1
    assert set(out["aspects"]) == {"performance", "bugs", "gameplay", "story", "price", "controls"}
    perf = out["aspects"]["performance"]
    assert {"mention_share", "negative_share_of_mentions", "complaint_share", "n_mentions"} <= set(
        perf
    )
    # fixture phrases: "Crashes every hour, terrible performance." -> a performance complaint
    assert perf["complaint_share"] > 0
    assert all(len(e["text"]) <= 160 for a in out["aspects"].values() for e in a["examples"])


def test_split_date_compares_before_and_after(tool):
    out = tool.run(
        SentimentArgs(
            appid=1091500,
            date_from=date(2021, 1, 1),
            date_to=date(2024, 12, 31),
            split_date=date(2022, 6, 1),
        )
    )
    cmp = out["comparison"]
    assert cmp["split_date"] == "2022-06-01"
    perf = cmp["complaint_share_by_aspect"]["performance"]
    assert {"before", "after", "diff", "p_value"} <= set(perf)


def test_empty_window_is_an_error_not_a_crash(tool):
    out = tool.run(
        SentimentArgs(appid=1091500, date_from=date(2010, 1, 1), date_to=date(2010, 2, 1))
    )
    assert "error" in out


def test_args_reject_reversed_window():
    with pytest.raises(ValueError):
        SentimentArgs(appid=1, date_from=date(2024, 1, 1), date_to=date(2023, 1, 1))


def test_a_window_not_straddling_split_date_is_replaced_by_window_months():
    a = SentimentArgs(
        appid=1,
        date_from=date(2023, 9, 22),
        date_to=date(2023, 12, 22),
        split_date=date(2023, 9, 21),
        window_months=3,
    )
    assert (a.date_from, a.date_to) == (date(2023, 6, 21), date(2023, 12, 21))
    assert a.window_adjusted
    b = SentimentArgs(
        appid=1, date_from=date(2023, 1, 1), date_to=date(2023, 12, 31), split_date=date(2023, 6, 1)
    )
    assert not b.window_adjusted and b.date_from == date(2023, 1, 1)


def test_split_date_alone_defaults_to_window_months_each_side():
    a = SentimentArgs(appid=1, split_date=date(2023, 9, 21), window_months=6)
    assert a.date_from == date(2023, 3, 21) and a.date_to == date(2024, 3, 21)


def test_weights_use_the_whole_sample_not_the_window_or_one_language():
    # 4 sampled positives in 2023 (2 English in Jan, 2 French in Jul) stand for 400 reviews:
    # each review weighs 100 whatever window or language is analysed afterwards
    sample = pl.DataFrame(
        {
            "appid": [1] * 4,
            "year": [2023] * 4,
            "voted_up": [True] * 4,
            "language": ["english", "english", "french", "french"],
        }
    )
    population = pl.DataFrame(
        {"appid": [1], "year": [2023], "n_reviews": [400], "n_positive": [400]}
    )
    w = stratum_weights(sample, population)
    assert w["weight"].to_list() == [100.0] * 4


# ---------------------------------------------------------------- window widening
from steam_game_analyst.tools.sentiment import choose_window  # noqa: E402

SPLIT = date(2023, 9, 21)


def counter(per_month_before: float, per_month_after: float):
    """Fake effective-n per side: proportional to the months each side covers."""

    def n_eff(start: date, end: date) -> tuple[float, float]:
        before = (SPLIT - start).days / 30.4 * per_month_before
        after = (end - SPLIT).days / 30.4 * per_month_after
        return before, after

    return n_eff


def test_a_window_with_enough_reviews_is_kept():
    start, end, used = choose_window(counter(50, 50), SPLIT, date(2023, 8, 21), date(2023, 10, 21))
    assert (start, end, used) == (date(2023, 8, 21), date(2023, 10, 21), None)


def test_a_thin_window_widens_to_six_months_each_side():
    start, end, used = choose_window(counter(10, 10), SPLIT, date(2023, 8, 21), date(2023, 10, 21))
    assert used == 6 and (start, end) == (date(2023, 3, 21), date(2024, 3, 21))


def test_still_thin_at_six_months_widens_to_twelve_and_stops():
    start, end, used = choose_window(counter(1, 1), SPLIT, date(2023, 8, 21), date(2023, 10, 21))
    assert used == 12 and (start, end) == (date(2022, 9, 21), date(2024, 9, 21))


def test_one_thin_side_is_enough_to_widen():
    _, _, used = choose_window(counter(100, 2), SPLIT, date(2023, 6, 21), date(2023, 12, 21))
    assert used == 12  # 6 months after hold 12 reviews; 12 months hold 24 (still thin) -> cap


def test_a_wide_requested_window_is_never_shrunk():
    start, end, used = choose_window(counter(1, 1), SPLIT, date(2021, 9, 21), date(2025, 9, 21))
    assert (start, end, used) == (date(2021, 9, 21), date(2025, 9, 21), None)


def test_the_tool_reports_a_widened_window(tool):
    out = tool.run(SentimentArgs(appid=1091500, split_date=date(2022, 6, 1), window_months=1))
    widened = out["window_widened"]
    assert widened["requested"] == {"from": "2022-05-01", "to": "2022-07-01"}
    assert widened["months_each_side"] in (6, 12)
    assert "fewer than 30" in widened["reason"]
