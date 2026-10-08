"""Service checks and failure messages: a down service is a message, not a traceback."""

import datetime
import re

import httpx
import pytest

from steam_game_analyst.config import Settings
from steam_game_analyst.ui import services


class ResponseError(Exception):
    """Stands in for ollama.ResponseError (matched by name, the package is optional)."""


@pytest.fixture
def cfg() -> Settings:
    return Settings(OLLAMA_HOST="http://ollama.invalid:11434", QDRANT_URL="http://q.invalid:6333")


def test_unreachable_ollama_is_a_kept_question_with_a_fix(cfg):
    f = services.classify(ConnectionError("Failed to connect to Ollama"), cfg)
    assert f.kind == "llm_down" and f.title == "The language model isn't running"
    assert "make up" in f.body and "http://ollama.invalid:11434" in f.detail
    assert services.classify(httpx.ConnectError("refused"), cfg).kind == "llm_down"


def test_missing_model_data_and_other_errors(cfg):
    missing = services.classify(ResponseError("model 'qwen' not found"), cfg)
    assert missing.kind == "model_missing" and "make pull-models" in missing.body
    data = services.classify(FileNotFoundError("sga.duckdb missing"), cfg)
    assert data.kind == "data_missing" and "make data" in data.body
    other = services.classify(ValueError("boom"), cfg)
    assert other.kind == "error" and other.detail == "ValueError: boom"


def test_health_reports_each_service(cfg, monkeypatch):
    def fake_get(url, timeout):
        if "ollama" in url:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(services.httpx, "get", fake_get)
    status = services.health(cfg)
    assert status["qdrant"] is None
    assert status["ollama"] == "ConnectError: connection refused"


def test_corpus_facts_read_the_tables(built):
    facts = services.corpus_facts(built)
    assert facts is not None
    assert re.fullmatch(
        r"4 games with review text · full Steam catalog · reviews through [A-Z][a-z]{2} \d{4}",
        facts,
    )


def test_corpus_facts_month_is_the_trend_tools_data_through(built):
    from steam_game_analyst.tools.catalog import CatalogDB
    from steam_game_analyst.tools.trend import ReviewTrend, TrendArgs

    db = CatalogDB(built.duckdb_path)
    try:
        through = ReviewTrend(db).run(TrendArgs(appid=1091500))["data_through"]
    finally:
        db.close()
    month = datetime.date.fromisoformat(through).strftime("%b %Y")
    assert (services.corpus_facts(built) or "").endswith(f"reviews through {month}")


def test_corpus_facts_without_data(tmp_path):
    assert services.corpus_facts(Settings(data_dir=tmp_path)) is None
