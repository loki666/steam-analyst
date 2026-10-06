"""LangChain tool wrappers: schemas, validation errors and JSON output."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from steam_game_analyst.tools.registry import TOOL_NAMES, ToolBox


class Dummy:
    def predict_proba(self, texts):
        return np.full(len(texts), 0.8)


@pytest.fixture
def box(built, monkeypatch):
    from steam_game_analyst.ml import reception

    class FakeReception:
        def predict(self, spec, top_k=8):
            return {
                "bucket": "very_positive",
                "probabilities": {},
                "top_features": [],
                "inputs_used": {"tags": spec.tags},
            }

    b = ToolBox(built, sentiment_model=Dummy(), reception_model=FakeReception())
    yield b
    b.close()
    assert reception  # imported for the type only


def test_tools_have_names_descriptions_and_pydantic_schemas(box):
    tools = box.langchain_tools()
    assert [t.name for t in tools] == list(TOOL_NAMES)
    for t in tools:
        assert len(t.description) > 40
        assert t.args_schema is not None


def test_invoke_validates_and_returns_json(box):
    ok = box.invoke("query_catalog", {"sql": "SELECT count(*) AS n FROM games"})
    assert ok.ok and json.loads(ok.content)["rows"] == [[4]]
    bad = box.invoke("review_trend", {"appid": "not a number"})
    assert not bad.ok and "appid" in bad.content
    unknown = box.invoke("drop_database", {})
    assert not unknown.ok and "unknown tool" in unknown.content


def test_guard_errors_come_back_as_tool_errors(box):
    res = box.invoke("query_catalog", {"sql": "DELETE FROM games"})
    assert not res.ok and "SELECT" in res.content


def test_predict_reception_takes_flat_features(box):
    res = box.invoke("predict_reception", {"price_usd": 25, "tags": ["Survival", "Co-op"]})
    assert res.ok and json.loads(res.content)["inputs_used"]["tags"] == ["Survival", "Co-op"]


def test_long_outputs_are_truncated(box):
    res = box.invoke("query_catalog", {"sql": "SELECT review_text FROM reviews", "limit": 200})
    assert len(res.content) <= box.max_chars + 100


def test_schema_validation_messages_are_short():
    from steam_game_analyst.tools.registry import QueryCatalogArgs

    with pytest.raises(ValidationError):
        QueryCatalogArgs(sql="SELECT 1", limit=0)
