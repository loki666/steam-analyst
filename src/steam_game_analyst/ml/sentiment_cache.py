"""Precomputed sentiment scores for the review corpus.

The review_sentiment tool scores each review and, per aspect, the review's sentences about it.
DistilBERT on the CPU takes 15-20 s for one game's window, while the reviews never change, so
every input the tool can ask for is scored once on the GPU (`sga train sentiment-scores`) and
looked up by a hash of its text. Anything not in the table (new reviews, a different sample)
is scored live by the fallback model."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from steam_game_analyst.ml.aspects import aspect_sentences

SCORES_FILE = "sentiment_distilbert.scores.parquet"


def aspect_inputs(text: str) -> dict[str, str]:
    """aspect -> the review's sentences about it, joined: what the tool scores per aspect."""
    return {a: " ".join(s) for a, s in aspect_sentences(text).items()}


def classifier_inputs(text: str) -> list[str]:
    """Everything the tool scores for one review: the review, then each aspect's input."""
    return [text, *aspect_inputs(text).values()]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def build(reviews: pl.Series, model: Any, batch_size: int = 4096) -> pl.DataFrame:
    """Score every distinct classifier input of `reviews` -> (key, prob)."""
    texts = sorted({x for t in reviews.to_list() if t for x in classifier_inputs(t)})
    probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="sentiment scores"):
        probs.append(np.asarray(model.predict_proba(texts[i : i + batch_size])))
    prob = np.concatenate(probs) if probs else np.zeros(0)
    return pl.DataFrame({"key": [_key(t) for t in texts], "prob": prob.astype(np.float32)})


class CachedClassifier:
    """predict_proba from the table; misses go to `fallback()` (built on first miss)."""

    def __init__(self, table: pl.DataFrame, fallback: Callable[[], Any]) -> None:
        self._probs = dict(zip(table["key"].to_list(), table["prob"].to_list(), strict=True))
        self._fallback_factory = fallback
        self._fallback: Any | None = None

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        out = np.array([self._probs.get(_key(t), np.nan) for t in texts], dtype=float)
        miss = np.flatnonzero(np.isnan(out))
        if len(miss):
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            out[miss] = self._fallback.predict_proba([texts[i] for i in miss])
        return out


def save(table: pl.DataFrame, models_dir: Path) -> Path:
    path = models_dir / SCORES_FILE
    table.write_parquet(path)
    return path


def load(models_dir: Path, fallback: Callable[[], Any]) -> CachedClassifier:
    path = models_dir / SCORES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sga train sentiment-scores`")
    return CachedClassifier(pl.read_parquet(path), fallback)
