"""`sga train` writes models + metrics into SGA_MODELS_DIR."""

import json

from typer.testing import CliRunner

from steam_game_analyst.cli import app
from steam_game_analyst.config import load_pipeline_config
from steam_game_analyst.data import build


def test_train_sentiment_writes_model_and_metrics(settings, tmp_path, monkeypatch):
    build.build_all(settings, load_pipeline_config())
    models = tmp_path / "models"
    monkeypatch.setenv("SGA_MODELS_DIR", str(models))
    from steam_game_analyst.config import get_settings

    get_settings.cache_clear()
    res = CliRunner().invoke(app, ["train", "sentiment"])
    assert res.exit_code == 0, res.output
    metrics = json.loads((models / "sentiment_tfidf_lr.metrics.json").read_text())
    assert "test" in metrics and "majority_baseline" in metrics
