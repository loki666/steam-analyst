"""Multi-label DistilBERT aspect model: which of the six aspects a review discusses.

Trained on silver labels (a local LLM applying the rubric in ml/aspect_labels.py to reviews of
training games) and scored once on the gold set (300 reviews of held-out games, labelled by two
Claude labellers plus adjudication). Training games and gold games never overlap. The keyword
lexicon (ml/aspects.py) is scored on the same gold set as the baseline. Hyperparameters are
fixed up front; validation games carved out of the silver set only pick the epoch. A review
has an aspect when its probability is >= 0.5."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

from steam_game_analyst.ml import bert
from steam_game_analyst.ml.aspects import ASPECTS, aspects_in
from steam_game_analyst.ml.sentiment import split_by_game

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "aspects_distilbert"
METRICS_FILE = "aspects_distilbert.metrics.json"
EPOCHS = 3
LR = 3e-5
BATCH_SIZE = 32
MAX_LENGTH = 256
THRESHOLD = 0.5
N_BOOT = 2000


class AspectModel(bert.BertClassifier):
    def __init__(self, tokenizer: Any, model: Any, device: str, max_length: int = MAX_LENGTH):
        super().__init__(tokenizer, model, device, max_length, head="sigmoid")

    def aspects_of(self, texts: Sequence[str]) -> list[set[str]]:
        probs = self.predict_proba(texts)
        return [{a for a, p in zip(ASPECTS, row, strict=True) if p >= THRESHOLD} for row in probs]


def _matrix(label_lists: Sequence[Sequence[str]]) -> np.ndarray:
    return np.array([[a in set(labels) for a in ASPECTS] for labels in label_lists], dtype=bool)


def scores(y: np.ndarray, pred: np.ndarray, names: Sequence[str] = ASPECTS) -> dict[str, Any]:
    """Per-aspect precision / recall / F1 and macro-F1 with a 95% bootstrap CI (reviews
    resampled)."""
    per = {
        a: {
            "n": int(y[:, j].sum()),
            "precision": float(precision_score(y[:, j], pred[:, j], zero_division=0)),
            "recall": float(recall_score(y[:, j], pred[:, j], zero_division=0)),
            "f1": float(f1_score(y[:, j], pred[:, j], zero_division=0)),
        }
        for j, a in enumerate(names)
    }
    rng = np.random.default_rng(0)
    boot = [
        f1_score(y[i], pred[i], average="macro", zero_division=0)
        for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))
    ]
    return {
        "n": int(len(y)),
        "macro_f1": float(np.mean([p["f1"] for p in per.values()])),
        "macro_f1_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "per_aspect": per,
    }


def train(
    silver: pl.DataFrame,
    gold: Sequence[Mapping[str, Any]],
    seed: int = 13,
    device: str = "cuda",
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    model_name: str = MODEL_NAME,
    cache_dir: str | None = None,
    model_factory: bert.ModelFactory | None = None,
) -> tuple[AspectModel, dict[str, Any]]:
    """silver: review_id, appid, text, aspects (list). gold: dicts with appid, text, aspects."""
    shared = set(silver["appid"].to_list()) & {g["appid"] for g in gold}
    if shared:
        raise ValueError(f"silver and gold share games {sorted(shared)[:5]}: gold must be unseen")
    bert.seed_everything(seed)
    fit_df, val_df = split_by_game(silver, 0.2, seed + 1)
    x_fit, y_fit = fit_df["text"].to_list(), _matrix(fit_df["aspects"].to_list())
    x_val, y_val = val_df["text"].to_list(), _matrix(val_df["aspects"].to_list())
    factory = model_factory or bert.pretrained(model_name, cache_dir, len(ASPECTS), "sigmoid")
    model = AspectModel(*factory(), device, max_length)

    def validate() -> float:
        pred = model.predict_proba(x_val) >= THRESHOLD
        return float(f1_score(y_val, pred, average="macro", zero_division=0))

    val_f1 = bert.fit(model, x_fit, y_fit, validate, epochs, lr, batch_size, seed)
    x_gold = [g["text"] for g in gold]
    y_gold = _matrix([g["aspects"] for g in gold])
    metrics = {
        "model": f"{model_name} fine-tuned, multi-label",
        "labels": list(ASPECTS),
        "train_labels": "silver: local LLM with the gold rubric, reviews of training games",
        "gold_labels": "300 reviews of held-out games, two Claude labellers + adjudication",
        "hyperparameters": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_length": max_length,
            "threshold": THRESHOLD,
            "seed": seed,
        },
        "validation_macro_f1_by_epoch": val_f1,
        "best_epoch": int(np.argmax(val_f1)) + 1,
        "n_fit": len(x_fit),
        "n_validation": len(x_val),
        "silver_share_by_aspect": {a: float(y_fit[:, j].mean()) for j, a in enumerate(ASPECTS)},
        "gold": {
            "model": scores(y_gold, model.predict_proba(x_gold) >= THRESHOLD),
            "lexicon": scores(y_gold, _matrix([sorted(aspects_in(t)) for t in x_gold])),
        },
    }
    return model, metrics


def save(model: AspectModel, metrics: dict[str, Any], models_dir: Path) -> Path:
    path = models_dir / MODEL_DIR
    path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(path)
    model.tokenizer.save_pretrained(path)
    (models_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1))
    return path


def load(models_dir: Path, device: str = "cpu") -> AspectModel:
    path = models_dir / MODEL_DIR
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sga train aspects`")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForSequenceClassification.from_pretrained(path)
    return AspectModel(tok, net, device)
