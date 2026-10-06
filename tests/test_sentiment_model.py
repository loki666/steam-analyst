"""TF-IDF + logistic regression sentiment baseline, trained on a synthetic corpus."""

import random
from datetime import datetime

import polars as pl
import pytest

from steam_game_analyst.ml import sentiment

POS = ["great", "fun", "love", "amazing", "superb", "recommend", "masterful", "smooth"]
NEG = ["boring", "awful", "hate", "refund", "broken", "terrible", "waste", "clunky"]
NEUTRAL = ["the", "game", "world", "map", "hours", "played", "level", "music", "it", "was"]


def corpus(n_games: int = 20, per_game: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for g in range(n_games):
        for i in range(per_game):
            up = rng.random() < 0.6
            words = rng.sample(NEUTRAL, 5) + rng.sample(POS if up else NEG, 2)
            if rng.random() < 0.1:  # label noise
                up = not up
            rng.shuffle(words)
            rows.append(
                {
                    "appid": 1000 + g,
                    "review_text": " ".join(words),
                    "voted_up": up,
                    "language": "english" if i % 10 else "french",
                    "created_at": datetime(2022, 1 + i % 12, 1),
                }
            )
    return pl.DataFrame(rows)


def test_split_by_game_keeps_games_disjoint():
    df = corpus()
    train, test = sentiment.split_by_game(df, test_frac=0.25, seed=1)
    assert set(train["appid"]).isdisjoint(set(test["appid"]))
    assert train.height + test.height == df.height
    assert train.height and test.height


def test_train_reports_metrics_next_to_majority_baseline():
    model, metrics = sentiment.train(corpus(), seed=0)
    m, base = metrics["test"], metrics["majority_baseline"]
    assert set(m) >= {"f1_positive", "f1_negative", "macro_f1", "roc_auc", "accuracy", "n"}
    assert base["roc_auc"] == 0.5
    assert m["roc_auc"] > 0.8 and m["macro_f1"] > base["macro_f1"]
    assert metrics["languages"] == ["english"]  # non-English reviews are left out
    assert metrics["C"] in sentiment.C_GRID
    p = model.predict_proba(["superb fun game, recommend", "awful broken waste, refund"])
    assert p[0] > 0.5 > p[1]


def test_save_and_load_round_trip(tmp_path):
    model, metrics = sentiment.train(corpus(), seed=0)
    path = sentiment.save(model, metrics, tmp_path)
    loaded = sentiment.load(tmp_path)
    assert path.exists() and (tmp_path / sentiment.METRICS_FILE).exists()
    assert loaded.predict_proba(["great fun"])[0] == pytest.approx(
        model.predict_proba(["great fun"])[0]
    )
