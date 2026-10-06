"""End-to-end W1 pipeline on the synthetic fixture: raw -> parquet -> duckdb."""

import duckdb
import polars as pl

from steam_game_analyst.config import load_pipeline_config
from steam_game_analyst.data import build, subset, transform


def test_build_games_canonical(raw_dataset):
    raw = transform.read_catalog_raw(raw_dataset / "catalog" / "games.csv")
    games = transform.build_games(raw)
    assert games.height == 6
    row = games.filter(pl.col("appid") == 1245620).row(0, named=True)
    assert row["name"] == "ELDEN RING"
    assert str(row["release_date"]) == "2022-02-25"
    assert row["genres"] == ["Action", "RPG"]
    assert row["tags"][0] == "Souls-like"
    assert row["estimated_owners_low"] == 20_000_000
    assert row["metacritic_score"] == 90
    assert "<" not in row["about_the_game"] and "great" in row["about_the_game"]
    assert abs(row["positive_ratio"] - 500000 / 540000) < 1e-6
    tba = games.filter(pl.col("appid") == 999002).row(0, named=True)
    assert tba["release_date"] is None and tba["metacritic_score"] is None


def test_build_reviews_cleans_dedupes_and_drops_ids(raw_dataset):
    p = raw_dataset / "reviews" / "1245620_120.csv"
    raw = transform.read_review_file(p, appid=1245620)
    cfg = load_pipeline_config().cleaning
    reviews = transform.build_reviews(raw, cfg)
    assert "author.steamid" not in reviews.columns and "recommendationid" not in reviews.columns
    assert reviews["n_chars"].min() >= cfg.min_review_chars  # "ok" rows dropped
    assert reviews.filter(pl.col("review_text").str.contains("<")).height == 0  # html stripped
    assert reviews["review_id"].n_unique() == reviews.height
    # duplicate text row collapsed
    assert (
        reviews.filter(
            pl.col("review_text") == "Combat is superb and the world is huge. (3)"
        ).height
        <= 1
    )
    assert reviews["created_at"].dtype == pl.Datetime("us")
    assert reviews["voted_up"].null_count() == 0


def test_sample_reviews_caps_and_balances(raw_dataset):
    p = raw_dataset / "reviews" / "1091500_120.csv"
    reviews = transform.build_reviews(
        transform.read_review_file(p, appid=1091500), load_pipeline_config().cleaning
    )
    s = subset.sample_reviews(reviews, per_game=30, languages=["english"])
    assert s.height == 30
    assert set(s["language"].unique()) == {"english"}
    n_neg = s.filter(~pl.col("voted_up")).height
    assert 10 <= n_neg <= 15  # roughly half, topped up from the other polarity when short


def test_select_games_respects_window_and_must_include(raw_dataset):
    games = transform.build_games(transform.read_catalog_raw(raw_dataset / "catalog" / "games.csv"))
    counts = pl.DataFrame(
        {"appid": [1245620, 1091500, 892970, 1145360, 999001], "n_reviews": [120] * 5}
    )
    cfg = load_pipeline_config().subset
    cfg.n_games, cfg.min_english_reviews, cfg.must_include = 3, 100, [999001]
    chosen = subset.select_games(games, counts, cfg)
    assert chosen.height == 3
    assert 999001 in chosen["appid"].to_list()  # must_include wins even if less popular
    assert 999002 not in chosen["appid"].to_list()  # TBA release excluded


def test_build_news(raw_dataset):
    news = transform.build_news(raw_dataset / "news")
    assert news.height == 4
    patch = news.filter(pl.col("is_patch_note")).row(0, named=True)
    assert patch["title"] == "Patch 1.5 Notes"
    assert "- Fixed stutter" in patch["contents"]
    assert news["published_at"].dtype == pl.Datetime("us")


def test_build_all_creates_duckdb(settings):
    cfg = load_pipeline_config()
    build.build_all(settings, cfg)
    assert settings.duckdb_path.exists()
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    n_games = con.execute("select count(*) from games").fetchone()[0]
    n_reviews = con.execute("select count(*) from reviews").fetchone()[0]
    n_months = con.execute("select count(*) from review_monthly").fetchone()[0]
    summary = con.execute("select * from game_summary").pl()
    con.close()
    assert n_games == 4
    assert 0 < n_reviews <= 4 * 30
    assert n_months > 0
    assert summary.height == 4 and "n_patch_notes" in summary.columns
