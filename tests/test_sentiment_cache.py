import numpy as np
import polars as pl
import pytest

from steam_game_analyst.ml import sentiment_cache as sc


class CountingModel:
    """P(positive) = 0.9 if 'good' in the text else 0.1; records what it scored."""

    def __init__(self):
        self.seen = []

    def predict_proba(self, texts):
        self.seen += list(texts)
        return np.array([0.9 if "good" in t else 0.1 for t in texts])


def test_inputs_are_the_review_and_each_aspect_s_sentences():
    text = "The story is good. Crashes every hour. The plot drags."
    inputs = sc.classifier_inputs(text)
    assert inputs[0] == text
    assert "The story is good. The plot drags." in inputs
    assert "Crashes every hour." in inputs


def test_cache_answers_known_texts_and_scores_only_the_misses():
    model = CountingModel()
    table = sc.build(pl.Series(["good game", "bad game"]), model, batch_size=1)
    assert table.height == 2 and set(table.columns) == {"key", "prob"}
    fallback = CountingModel()
    cached = sc.CachedClassifier(table, fallback=lambda: fallback)
    probs = cached.predict_proba(["bad game", "good stuff", "good game"])
    np.testing.assert_allclose(probs, [0.1, 0.9, 0.9])
    assert fallback.seen == ["good stuff"]
    assert cached.predict_proba([]).shape == (0,)


def test_build_scores_each_distinct_input_once(tmp_path):
    model = CountingModel()
    reviews = pl.Series(["Crashes a lot.", "Crashes a lot."])
    table = sc.build(reviews.map_elements(lambda t: t), model)  # duplicates collapse
    assert len(model.seen) == len(set(model.seen))
    sc.save(table, tmp_path)
    loaded = sc.load(tmp_path, fallback=lambda: model)
    assert loaded.predict_proba(["Crashes a lot."])[0] == pytest.approx(0.1)


def test_load_without_scores_says_how_to_make_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="sga train sentiment-scores"):
        sc.load(tmp_path, fallback=lambda: None)


def test_toolbox_rejects_an_unknown_sentiment_model(tmp_path):
    from steam_game_analyst.config import Settings
    from steam_game_analyst.tools.registry import _load_sentiment

    with pytest.raises(ValueError, match="distilbert or tfidf"):
        _load_sentiment(Settings(models_dir=tmp_path, sentiment_model="bert-large"))
