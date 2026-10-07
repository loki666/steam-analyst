"""DistilBERT sentiment fine-tuning, exercised with a tiny randomly initialised model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steam_game_analyst.ml import sentiment_bert as sb  # noqa: E402
from test_sentiment_model import NEG, NEUTRAL, POS, corpus  # noqa: E402


@pytest.fixture
def tiny_factory():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *POS, *NEG, *NEUTRAL]

    def factory():
        # transformers 5 ignores vocab_file here: pass the vocab itself
        tok = transformers.DistilBertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
        torch.manual_seed(0)
        cfg = transformers.DistilBertConfig(
            vocab_size=len(vocab),
            dim=32,
            n_layers=1,
            n_heads=2,
            hidden_dim=64,
            max_position_embeddings=64,
            num_labels=2,
        )
        return tok, transformers.DistilBertForSequenceClassification(cfg)

    return factory


def _train(factory, **kw):
    params = {"epochs": 3, "lr": 3e-3, "batch_size": 16, "max_length": 16, "device": "cpu", **kw}
    return sb.train(corpus(n_games=24, per_game=60), seed=13, model_factory=factory, **params)


def test_fine_tuning_learns_and_reports_held_out_games(tiny_factory):
    model, metrics = _train(tiny_factory)
    assert metrics["test"]["roc_auc"] > 0.8  # word cues are easy; 10% label noise
    assert 1 <= metrics["best_epoch"] <= 3
    assert len(metrics["validation_roc_auc_by_epoch"]) == 3
    assert metrics["n_test_games"] > 0 and metrics["languages"] == ["english"]
    assert metrics["majority_baseline"]["roc_auc"] == 0.5
    p = model.predict_proba(["great fun game", "boring awful refund"])
    assert p.shape == (2,) and 0 <= p.min() and p.max() <= 1 and p[0] > p[1]
    assert model.predict_proba([]).shape == (0,)


def test_same_held_out_games_as_the_tfidf_baseline(tiny_factory):
    from steam_game_analyst.ml import sentiment

    _, bert = _train(tiny_factory, epochs=1)
    _, tfidf = sentiment.train(corpus(n_games=24, per_game=60), seed=13)
    assert bert["test"]["n"] == tfidf["test"]["n"]
    assert bert["n_test_games"] == tfidf["n_test_games"]


def test_save_and_load_give_the_same_probabilities(tiny_factory, tmp_path):
    model, metrics = _train(tiny_factory, epochs=1)
    sb.save(model, metrics, tmp_path / "models")
    loaded = sb.load(tmp_path / "models", device="cpu")
    texts = ["great smooth world", "clunky broken map"]
    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts), atol=1e-5)
    assert (tmp_path / "models" / sb.METRICS_FILE).exists()


def test_load_without_a_trained_model_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="sga train sentiment-bert"):
        sb.load(tmp_path)
