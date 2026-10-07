"""Review sentiment with a fine-tuned DistilBERT, against voted_up.

Same data and split as the TF-IDF baseline (ml/sentiment.py): English reviews, 20% of games
held out by a hash of appid, validation games carved out of the training games. The
hyperparameters are fixed up front (standard DistilBERT fine-tuning values), not searched; the
only choice made on validation games is the epoch to keep. The test games are scored once.

Batches group reviews of similar length (most reviews are short), which cuts padding and
makes an epoch over ~180k reviews take minutes on the 3080 in fp16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from steam_game_analyst.ml import bert
from steam_game_analyst.ml.sentiment import LANGUAGES, _scores, split_by_game

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "sentiment_distilbert"
METRICS_FILE = "sentiment_distilbert.metrics.json"
EPOCHS = 2
LR = 3e-5
BATCH_SIZE = 32
MAX_LENGTH = 256

ModelFactory = bert.ModelFactory


class BertSentiment(bert.BertClassifier):
    """P(review is positive) for a batch of texts."""

    def __init__(self, tokenizer: Any, model: Any, device: str, max_length: int = MAX_LENGTH):
        super().__init__(tokenizer, model, device, max_length, head="softmax")


def train(
    reviews: pl.DataFrame,
    seed: int = 13,
    device: str = "cuda",
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    model_name: str = MODEL_NAME,
    cache_dir: str | None = None,
    model_factory: ModelFactory | None = None,
) -> tuple[BertSentiment, dict[str, Any]]:
    bert.seed_everything(seed)
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
    x_test, y_test = xy(test_df)
    factory = model_factory or bert.pretrained(model_name, cache_dir, 2, "softmax")
    clf = BertSentiment(*factory(), device, max_length)

    def validate() -> float:
        p = clf.predict_proba(x_val)
        return float(roc_auc_score(y_val, p)) if len(set(y_val.tolist())) > 1 else 0.5

    val_auc = bert.fit(clf, x_fit, y_fit, validate, epochs, lr, batch_size, seed)
    majority = bool(y_fit.mean() >= 0.5)
    metrics = {
        "model": f"{model_name} fine-tuned",
        "target": "voted_up",
        "languages": LANGUAGES,
        "split": "by game (sha256 of appid), 20% of games held out (same as the TF-IDF baseline)",
        "hyperparameters": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_length": max_length,
            "warmup": bert.WARMUP,
            "weight_decay": bert.WEIGHT_DECAY,
            "seed": seed,
        },
        "validation_roc_auc_by_epoch": val_auc,
        "best_epoch": int(np.argmax(val_auc)) + 1,
        "n_fit": len(y_fit),
        "n_validation": len(y_val),
        "n_train_games": train_df["appid"].n_unique(),
        "n_test_games": test_df["appid"].n_unique(),
        "test": _scores(y_test, clf.predict_proba(x_test)),
        "majority_baseline": {
            **_scores(y_test, np.full(len(y_test), 1.0 if majority else 0.0)),
            "roc_auc": 0.5,
            "predicts": "positive" if majority else "negative",
        },
    }
    return clf, metrics


def save(model: BertSentiment, metrics: dict[str, Any], models_dir: Path) -> Path:
    path = models_dir / MODEL_DIR
    path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(path)
    model.tokenizer.save_pretrained(path)
    (models_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1))
    return path


def load(models_dir: Path, device: str = "cpu") -> BertSentiment:
    path = models_dir / MODEL_DIR
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sga train sentiment-bert`")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return BertSentiment(tok, model, device)
