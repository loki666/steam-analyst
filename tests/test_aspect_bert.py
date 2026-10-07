"""Multi-label DistilBERT aspect model, with a tiny randomly initialised network."""

import random

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steam_game_analyst.ml import aspect_bert as ab  # noqa: E402
from steam_game_analyst.ml.aspects import ASPECTS  # noqa: E402

CUES = {
    "performance": "fps",
    "bugs": "crash",
    "gameplay": "combat",
    "story": "plot",
    "price": "price",
    "controls": "controller",
}
FILLER = ["the", "game", "is", "really", "quite", "honestly", "i", "think"]


def _rows(n, appid0, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        labels = [a for a in ASPECTS if rng.random() < 0.3]
        words = rng.sample(FILLER, 4) + [CUES[a] for a in labels]
        rng.shuffle(words)
        rows.append(
            {
                "review_id": f"r{appid0}-{i}",
                "appid": appid0 + i % 12,
                "text": " ".join(words),
                "aspects": labels,
            }
        )
    return rows


@pytest.fixture
def tiny_factory():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *CUES.values(), *FILLER]

    def factory():
        tok = transformers.DistilBertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
        torch.manual_seed(0)
        cfg = transformers.DistilBertConfig(
            vocab_size=len(vocab),
            # one layer of 32 dims cannot pick out six cues at once; this size can
            dim=64,
            n_layers=2,
            n_heads=4,
            hidden_dim=128,
            max_position_embeddings=64,
            num_labels=len(ASPECTS),
            problem_type="multi_label_classification",
        )
        return tok, transformers.DistilBertForSequenceClassification(cfg)

    return factory


def _train(factory, silver=None, gold=None, **kw):
    params = {"epochs": 8, "lr": 3e-3, "batch_size": 16, "max_length": 16, "device": "cpu", **kw}
    silver = pl.DataFrame(silver or _rows(600, 100, 0))
    gold = gold or _rows(120, 900, 1)
    return ab.train(silver, gold, seed=13, model_factory=factory, **params)


def test_scores_per_aspect_and_macro_f1():
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    p = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=bool)
    s = ab.scores(y, p, ["a", "b"])
    assert s["per_aspect"]["a"] == {"n": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert s["per_aspect"]["b"]["f1"] == 1.0
    assert s["macro_f1"] == 0.75
    assert s["macro_f1_ci95"][0] <= 0.75 <= s["macro_f1_ci95"][1]


def test_model_learns_aspects_and_reports_the_lexicon_too(tiny_factory):
    model, metrics = _train(tiny_factory)
    assert metrics["gold"]["model"]["macro_f1"] > 0.6
    assert set(metrics["gold"]) == {"model", "lexicon"}
    assert len(metrics["validation_macro_f1_by_epoch"]) == 8
    probs = model.predict_proba(["fps crash", "plot"])
    assert probs.shape == (2, len(ASPECTS))
    assert len(model.aspects_of(["fps crash plot", "the game"])) == 2


def test_training_refuses_reviews_from_gold_games(tiny_factory):
    gold = _rows(20, 100, 1)  # same appids as the silver rows
    with pytest.raises(ValueError, match="share games"):
        _train(tiny_factory, gold=gold, epochs=1)


def test_save_and_load_round_trip(tiny_factory, tmp_path):
    model, metrics = _train(tiny_factory, epochs=1)
    ab.save(model, metrics, tmp_path)
    loaded = ab.load(tmp_path, device="cpu")
    texts = ["controller price", "the game"]
    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts), atol=1e-5)
    with pytest.raises(FileNotFoundError, match="sga train aspects"):
        ab.load(tmp_path / "missing")
