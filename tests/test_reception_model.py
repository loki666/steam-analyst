"""Reception predictor on a synthetic catalog with a planted signal."""

import random
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from steam_game_analyst.data.schemas import GAMES_SCHEMA
from steam_game_analyst.ml import reception as rc


def catalog(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    """Tag "Polished" pushes the ratio up, "Janky" down; "Great Soundtrack" is a pure leak."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tags = rng.sample(["Action", "Puzzle", "Survival", "Co-op", "Indie", "2D"], 3)
        quality = rng.random()
        if quality > 0.6:
            tags.append("Polished")
        if quality < 0.3:
            tags.append("Janky")
        ratio = min(max(0.55 + 0.4 * quality + rng.gauss(0, 0.05), 0.0), 1.0)
        if ratio > 0.85:
            tags.append("Great Soundtrack")
        votes = rng.choice([20, 80, 500, 5000])
        pos = int(votes * ratio)
        rows.append(
            {
                "appid": i,
                "name": f"Game {i}",
                "release_date": date(2014, 1, 1) + timedelta(days=rng.randint(0, 11 * 365)),
                "price_usd": rng.choice([0.0, 4.99, 9.99, 19.99, 29.99]),
                "required_age": 0,
                "dlc_count": 0,
                "short_description": "short",
                "about_the_game": "about " * rng.randint(1, 50),
                "developers": [f"Dev {i % 300}"],
                "publishers": [f"Pub {i % 50}"],
                "genres": ["Action"] if i % 2 else ["Indie"],
                "categories": ["Single-player", "Steam Trading Cards"],
                "tags": tags,
                "supported_languages": ["English"],
                "windows": True,
                "mac": False,
                "linux": False,
                "metacritic_score": None,
                "positive": pos,
                "negative": votes - pos,
                "positive_ratio": pos / votes,
                "estimated_owners_low": 0,
                "estimated_owners_high": 0,
                "peak_ccu": 0,
                "average_playtime_forever": 0,
                "median_playtime_forever": 0,
                "recommendations": 0,
                "achievements": 0,
            }
        )
    return pl.DataFrame(rows, schema=GAMES_SCHEMA)


@pytest.fixture(scope="module")
def trained():
    return rc.train(catalog(), seed=0)


def test_bucket_boundaries_follow_steam_labels():
    assert [rc.bucket_of(r) for r in (0.1, 0.69, 0.70, 0.79, 0.80, 1.0)] == [0, 0, 1, 1, 2, 2]


def test_labelled_games_need_min_votes():
    lab = rc.labelled_games(catalog())
    assert ((lab["positive"] + lab["negative"]) >= rc.MIN_VOTES).all()


def test_split_is_by_release_date():
    tr, va, te = rc.split(rc.labelled_games(catalog()))
    assert tr["release_date"].max() < rc.VALID_FROM <= va["release_date"].min()
    assert va["release_date"].max() < rc.TEST_FROM <= te["release_date"].min()


def test_test_games_had_a_year_of_reviews_before_the_snapshot():
    data = rc.labelled_games(catalog())
    assert data["release_date"].max() >= rc.TEST_TO  # the fixture has younger games
    _, _, te = rc.split(data)
    assert te["release_date"].max() < rc.TEST_TO
    assert rc.TEST_TO == date(2024, 6, 1)  # CATALOG_SNAPSHOT 2025-06-01 minus 12 months


def test_spec_drops_evaluative_tags_and_post_launch_categories(trained):
    model, _ = trained
    assert "Great Soundtrack" not in model.spec.tags
    assert "Steam Trading Cards" not in model.spec.categories
    assert {"Polished", "Janky"} <= set(model.spec.tags)


def test_prior_track_record_only_counts_earlier_releases():
    games = pl.DataFrame(
        {
            "developers": [["A"], ["A"], ["A"], ["B"]],
            "publishers": [["P"], ["P"], ["P"], ["P"]],
            "release_date": [
                date(2020, 1, 1),
                date(2021, 1, 1),
                date(2021, 1, 1),
                date(2022, 1, 1),
            ],
            "positive": [90, 60, 70, 10],
            "negative": [10, 40, 30, 90],
            "positive_ratio": [0.9, 0.6, 0.7, 0.1],
        }
    )
    out = rc._with_priors(games, rc.track_record(games)).sort("release_date")
    assert out["dev_prior_games"].to_list() == [0, 1, 1, 0]  # same-day release not counted
    assert out["dev_prior_mean_ratio"][1] == pytest.approx(0.9)
    assert out["pub_prior_games"].to_list() == [0, 1, 1, 3]
    assert np.isnan(out["dev_prior_mean_ratio"][0])


def test_training_beats_majority_baseline(trained):
    _, m = trained
    assert m["test"]["accuracy"] > m["majority_baseline"]["accuracy"]
    assert m["test"]["log_loss"] < m["majority_baseline"]["log_loss"]
    lo, hi = m["test"]["accuracy_ci95"]
    assert lo <= m["test"]["accuracy"] <= hi
    assert 0 <= m["test"]["ece"] <= 1 and m["test"]["reliability"]
    assert any("Polished" in f or "Janky" in f for f in list(m["shap_mean_abs_top20"])[:3])


def test_predict_returns_bucket_probabilities_and_shap(trained):
    model, _ = trained
    good = model.predict(rc.GameSpec(price_usd=25, tags=["polished", "co-op", "Survival"]))
    bad = model.predict(rc.GameSpec(price_usd=25, tags=["Janky", "Co-op", "nonsense tag"]))
    assert set(good["probabilities"]) == set(rc.BUCKETS)
    assert sum(good["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert good["probabilities"]["very_positive"] > bad["probabilities"]["very_positive"]
    assert good["inputs_used"]["tags"] == ["Polished", "Co-op", "Survival"]
    assert bad["inputs_used"]["unknown"]["tags"] == ["nonsense tag"]
    feats = [f["feature"] for f in good["top_features"]]
    assert "tag: Polished" in feats


def test_save_and_load_round_trip(trained, tmp_path):
    model, metrics = trained
    rc.save(model, metrics, tmp_path)
    loaded = rc.load(tmp_path)
    spec = rc.GameSpec(price_usd=10, tags=["Polished"], release_date=date(2025, 5, 1))
    assert loaded.predict(spec)["probabilities"] == model.predict(spec)["probabilities"]


def test_game_spec_validates_price():
    with pytest.raises(ValueError):
        rc.GameSpec(price_usd=-1)


def test_unstated_listing_fields_get_typical_values_not_missing(trained):
    model, _ = trained
    out = model.predict(rc.GameSpec(price_usd=10, tags=["Polished"]))
    imputed = out["inputs_used"]["imputed_typical"]
    assert {"about_chars", "n_languages", "self_published"} <= set(imputed)
    assert imputed["about_chars"] == model.spec.defaults["about_chars"]
    stated = model.predict(rc.GameSpec(price_usd=10, n_languages=3, windows=True))
    assert "n_languages" not in stated["inputs_used"]["imputed_typical"]


def test_few_tags_come_with_a_warning(trained):
    model, _ = trained
    out = model.predict(rc.GameSpec(price_usd=10, tags=["Polished"]))
    assert any("tags" in w for w in out["warnings"])
