"""Review sentiment with a fine-tuned DistilBERT, against voted_up.

Same data and split as the TF-IDF baseline (ml/sentiment.py): English reviews, 20% of games
held out by a hash of appid, validation games carved out of the training games. The
hyperparameters are fixed up front (standard DistilBERT fine-tuning values), not searched; the
only choice made on validation games is the epoch to keep. The test games are scored once.

Batches group reviews of similar length (most reviews are short), which cuts padding and
makes an epoch over ~180k reviews take minutes on the 3080 in fp16."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from steam_game_analyst.ml.sentiment import LANGUAGES, _scores, split_by_game

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "sentiment_distilbert"
METRICS_FILE = "sentiment_distilbert.metrics.json"
EPOCHS = 2
LR = 3e-5
BATCH_SIZE = 32
MAX_LENGTH = 256
WARMUP = 0.06
WEIGHT_DECAY = 0.01
_BUCKET = 50  # batches per length-sorted bucket

ModelFactory = Callable[[], tuple[Any, Any]]  # () -> (tokenizer, model)


class BertSentiment:
    """P(review is positive) for a batch of texts."""

    def __init__(self, tokenizer: Any, model: Any, device: str, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = device
        self.max_length = max_length

    def predict_proba(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros(0)
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts))
        with torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idx = order[i : i + batch_size]
                enc = self._encode([texts[j] for j in idx])
                with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                    logits = self.model(**enc).logits.float()
                out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}


def _pretrained(model_name: str, cache_dir: str | None) -> ModelFactory:
    def factory() -> tuple[Any, Any]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir=cache_dir
        )
        return tok, model

    return factory


def _batches(lengths: Sequence[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffled batches of similar-length examples: shuffle, sort within buckets, cut."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    span = batch_size * _BUCKET
    batches = []
    for s in range(0, len(idx), span):
        chunk = sorted(idx[s : s + span], key=lambda i: lengths[i])
        batches += [chunk[b : b + batch_size] for b in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def _fit_epochs(
    clf: BertSentiment,
    x_fit: list[str],
    y_fit: np.ndarray,
    x_val: list[str],
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    """Train; return validation ROC-AUC per epoch and the weights of the best epoch."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    model, device = clf.model, clf.device
    rng = random.Random(seed)
    lengths = [len(t) for t in x_fit]
    steps = epochs * -(-len(x_fit) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    labels = torch.tensor(y_fit.astype(np.int64))
    val_auc: list[float] = []
    best: dict[str, Any] = {}
    for _ in range(epochs):
        model.train()
        for batch in _batches(lengths, batch_size, rng):
            enc = clf._encode([x_fit[i] for i in batch])
            with torch.autocast("cuda", enabled=use_amp):
                loss = model(**enc, labels=labels[batch].to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        p = clf.predict_proba(x_val)
        auc = float(roc_auc_score(y_val, p)) if len(set(y_val.tolist())) > 1 else 0.5
        if not val_auc or auc > max(val_auc):
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        val_auc.append(auc)
    return val_auc, best


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
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    tok, net = (model_factory or _pretrained(model_name, cache_dir))()
    clf = BertSentiment(tok, net, device, max_length)
    val_auc, best = _fit_epochs(clf, x_fit, y_fit, x_val, y_val, epochs, lr, batch_size, seed)
    clf.model.load_state_dict(best)
    clf.model.eval()
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
            "warmup": WARMUP,
            "weight_decay": WEIGHT_DECAY,
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
