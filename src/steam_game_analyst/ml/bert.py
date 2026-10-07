"""Shared DistilBERT fine-tuning: a classifier wrapper and a length-bucketed training loop.

`head="softmax"` is single-label (sentiment: P(class 1)); `head="sigmoid"` is multi-label
(aspects: one probability per label)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

Head = Literal["softmax", "sigmoid"]
ModelFactory = Callable[[], tuple[Any, Any]]  # () -> (tokenizer, model)
WARMUP = 0.06
WEIGHT_DECAY = 0.01
_BUCKET = 50  # batches per length-sorted bucket


class BertClassifier:
    def __init__(self, tokenizer: Any, model: Any, device: str, max_length: int, head: Head):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.head = head

    def predict_proba(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """softmax head: shape (n,), P(label 1). sigmoid head: shape (n, n_labels)."""
        import torch

        n_out = self.model.config.num_labels
        if not texts:
            return np.zeros(0) if self.head == "softmax" else np.zeros((0, n_out))
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts)) if self.head == "softmax" else np.empty((len(texts), n_out))
        with torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idx = order[i : i + batch_size]
                enc = self.encode([texts[j] for j in idx])
                with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                    logits = self.model(**enc).logits.float()
                if self.head == "softmax":
                    out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    out[idx] = torch.sigmoid(logits).cpu().numpy()
        return out

    def encode(self, texts: list[str]) -> dict[str, Any]:
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}


def pretrained(model_name: str, cache_dir: str | None, num_labels: int, head: Head) -> ModelFactory:
    def factory() -> tuple[Any, Any]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        kw = {"problem_type": "multi_label_classification"} if head == "sigmoid" else {}
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir, **kw
        )
        return tok, model

    return factory


def batches(lengths: Sequence[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffled batches of similar-length examples: shuffle, sort within buckets, cut."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    span = batch_size * _BUCKET
    out = []
    for s in range(0, len(idx), span):
        chunk = sorted(idx[s : s + span], key=lambda i: lengths[i])
        out += [chunk[b : b + batch_size] for b in range(0, len(chunk), batch_size)]
    rng.shuffle(out)
    return out


def fit(
    clf: BertClassifier,
    texts: list[str],
    labels: np.ndarray,
    validate: Callable[[], float],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> list[float]:
    """Fine-tune `clf` in place; after each epoch call `validate` (higher is better) and keep
    the weights of the best epoch. Returns the validation score per epoch."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    model, device = clf.model, clf.device
    rng = random.Random(seed)
    lengths = [len(t) for t in texts]
    steps = epochs * -(-len(texts) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if clf.head == "softmax":
        y = torch.tensor(labels.astype(np.int64))
    else:
        y = torch.tensor(labels.astype(np.float32))
    scores: list[float] = []
    best: dict[str, Any] = {}
    for _ in range(epochs):
        model.train()
        for batch in batches(lengths, batch_size, rng):
            enc = clf.encode([texts[i] for i in batch])
            with torch.autocast("cuda", enabled=use_amp):
                loss = model(**enc, labels=y[batch].to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        score = validate()
        if not scores or score > max(scores):
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scores.append(score)
    model.load_state_dict(best)
    model.eval()
    return scores


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
