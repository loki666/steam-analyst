"""Review sentiment baseline: TF-IDF (word 1-2 grams) + logistic regression against voted_up.

English reviews only (the aspect lexicon is English too). Held-out data are whole games
(split by appid), so the test score measures transfer to unseen games rather than memorised
game names. C is picked on validation games carved out of the training games; the test
games are scored once."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

MODEL_FILE = "sentiment_tfidf_lr.joblib"
METRICS_FILE = "sentiment_tfidf_lr.metrics.json"
C_GRID: tuple[float, ...] = (0.25, 1.0, 4.0)
LANGUAGES = ["english"]


class SentimentModel:
    """P(review is positive) for a batch of texts."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return self.pipeline.predict_proba(list(texts))[:, 1]


def _bucket(appid: int, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{appid}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_game(
    df: pl.DataFrame, test_frac: float = 0.2, seed: int = 13
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic split on a hash of appid: every review of a game lands on one side."""
    games = df.select("appid").unique()
    test_ids = [a for a in games["appid"].to_list() if _bucket(int(a), seed) < test_frac]
    is_test = pl.col("appid").is_in(test_ids)
    return df.filter(~is_test), df.filter(is_test)


def _pipeline(c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    max_features=300_000,
                    sublinear_tf=True,
                ),
            ),
            ("lr", LogisticRegression(C=c, max_iter=2000, random_state=seed)),
        ]
    )


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p >= 0.5
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_positive": float(f1_score(y, pred, pos_label=True, zero_division=0)),
        "f1_negative": float(f1_score(y, pred, pos_label=False, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else 0.5,
        "positive_share": float(y.mean()),
    }


def train(reviews: pl.DataFrame, seed: int = 13) -> tuple[SentimentModel, dict[str, Any]]:
    df = reviews.filter(
        pl.col("language").is_in(LANGUAGES)
        & pl.col("voted_up").is_not_null()
        & pl.col("review_text").is_not_null()
    )
    train_df, test_df = split_by_game(df, 0.2, seed)
    fit_df, val_df = split_by_game(train_df, 0.2, seed + 1)

    def xy(d: pl.DataFrame) -> tuple[list[str], np.ndarray]:
        return d["review_text"].to_list(), d["voted_up"].to_numpy().astype(bool)

    x_fit, y_fit = xy(fit_df)
    x_val, y_val = xy(val_df)
    val_auc = {}
    for c in C_GRID:
        p = _pipeline(c, seed).fit(x_fit, y_fit).predict_proba(x_val)[:, 1]
        val_auc[c] = float(roc_auc_score(y_val, p)) if len(set(y_val.tolist())) > 1 else 0.5
    best_c = max(C_GRID, key=lambda c: (val_auc[c], -c))

    x_train, y_train = xy(train_df)
    x_test, y_test = xy(test_df)
    pipe = _pipeline(best_c, seed).fit(x_train, y_train)
    model = SentimentModel(pipe)
    majority = bool(y_train.mean() >= 0.5)
    metrics = {
        "model": "tfidf(1-2gram) + logistic regression",
        "target": "voted_up",
        "languages": LANGUAGES,
        "split": "by game (sha256 of appid), 20% of games held out",
        "C": best_c,
        "validation_roc_auc_by_C": {str(k): v for k, v in val_auc.items()},
        "n_train": len(y_train),
        "n_train_games": train_df["appid"].n_unique(),
        "n_test_games": test_df["appid"].n_unique(),
        "test": _scores(y_test, model.predict_proba(x_test)),
        "majority_baseline": {
            **_scores(y_test, np.full(len(y_test), 1.0 if majority else 0.0)),
            "roc_auc": 0.5,
            "predicts": "positive" if majority else "negative",
        },
    }
    return model, metrics


def save(model: SentimentModel, metrics: dict[str, Any], models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILE
    joblib.dump(model.pipeline, path)
    (models_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1))
    return path


def load(models_dir: Path) -> SentimentModel:
    path = models_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sga train sentiment`")
    return SentimentModel(joblib.load(path))
