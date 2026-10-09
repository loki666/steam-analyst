"""The four agent tools as LangChain StructuredTools with pydantic argument schemas.

The graph binds `langchain_tools()` to the chat model (so the model sees the JSON schemas)
but executes calls through `ToolBox.invoke`, which validates the arguments itself and turns
every failure (bad arguments, refused SQL, missing data) into an error message the model can
read and correct on the retry edge."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from steam_game_analyst.config import Settings
from steam_game_analyst.ml.reception import GameSpec
from steam_game_analyst.tools.catalog import CatalogDB, SqlGuardError
from steam_game_analyst.tools.sentiment import ReviewSentiment, SentimentArgs
from steam_game_analyst.tools.trend import ReviewTrend, TrendArgs

TOOL_NAMES = ("query_catalog", "review_sentiment", "predict_reception", "review_trend")


class QueryCatalogArgs(BaseModel):
    sql: str = Field(description="One DuckDB SELECT statement over the allowed tables")
    limit: int = Field(50, ge=1, le=CatalogDB.MAX_ROWS, description="Maximum rows returned")


DESCRIPTIONS = {
    "query_catalog": (
        "Run one read-only DuckDB SELECT over the game catalog and review tables to filter, "
        "rank, count or compare games. Tables: games_all (full Steam catalog), games (the "
        "300-game subset with review text), game_summary, review_monthly, reviews, news. "
        "Prices are in USD. tags/genres/categories are VARCHAR[] lists. positive_ratio is "
        "NULL for games without votes. To rank games by review ratio, ORDER BY "
        "positive_ratio_lower95 (Wilson lower bound, penalises few votes) but SELECT and "
        "report positive_ratio and n_votes; the lower bound is not the ratio."
    ),
    "review_sentiment": (
        "Sentiment and aspect distribution (performance, bugs, gameplay, story, price, "
        "controls) of one game's English reviews in a date window: share of reviews "
        "mentioning each aspect, share of those mentions that are negative (complaints), "
        "and example review excerpts. Give split_date (e.g. a patch release date) to compare "
        "aspect complaints before vs after it with a significance test. Needs the appid."
    ),
    "predict_reception": (
        "Predict how a (possibly unreleased) game will be received from launch metadata: "
        "returns the probability of each positive-review-ratio bucket (negative_or_mixed "
        "<70%, mostly_positive 70-80%, very_positive >=80%) and the SHAP features that "
        "drove the prediction. Pass Steam tags/genres/categories by name, the USD price and "
        "the planned release date; leave unknown fields out."
    ),
    "review_trend": (
        "Monthly or quarterly review volume and positive share for one game, from every "
        "review in the corpus (not a sample): direction of the trend with a p-value, the "
        "most likely change point, or a before/after test around event_date (e.g. a patch). "
        "last_n_months counts back from the last month with data. Needs the appid."
    ),
}


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict[str, Any]
    ok: bool  # False also for a valid call that found no data
    content: str  # JSON, or an error message when not valid
    valid: bool = True  # False: bad arguments, unknown tool, refused query, missing model


def _dumps(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, date):
            return o.isoformat()
        return str(o)

    return json.dumps(obj, default=default, ensure_ascii=False, separators=(",", ":"))


def _validation_message(e: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(x) for x in err['loc']) or 'args'}: {err['msg']}" for err in e.errors()
    ]
    return "invalid arguments: " + "; ".join(parts)


def _load_sentiment(settings: Settings) -> tuple[Any, str]:
    """The configured review classifier and how the tool's output names it."""
    models_dir = settings.models_dir
    if settings.sentiment_model == "tfidf":
        from steam_game_analyst.ml import sentiment

        return sentiment.load(models_dir), "TF-IDF+LR classifier"
    if settings.sentiment_model != "distilbert":
        raise ValueError(
            f"SGA_SENTIMENT_MODEL must be distilbert or tfidf, not {settings.sentiment_model!r}"
        )
    from steam_game_analyst.ml import sentiment_bert, sentiment_cache

    model = sentiment_cache.load(
        models_dir, fallback=lambda: sentiment_bert.load(models_dir, "cpu")
    )
    return model, "fine-tuned DistilBERT classifier"


class ToolBox:
    """Owns the data handles and models; lazily loads the trained models from models/."""

    def __init__(
        self,
        settings: Settings,
        sentiment_model: Any | None = None,
        reception_model: Any | None = None,
        max_chars: int = 6000,
    ) -> None:
        self.settings = settings
        self.db = CatalogDB(settings.duckdb_path)
        self._sentiment_model = sentiment_model
        self._reception_model = reception_model
        self._sentiment: ReviewSentiment | None = None
        self._trend: ReviewTrend | None = None
        self.max_chars = max_chars
        self._schemas: dict[str, type[BaseModel]] = {
            "query_catalog": QueryCatalogArgs,
            "review_sentiment": SentimentArgs,
            "predict_reception": GameSpec,
            "review_trend": TrendArgs,
        }
        self._runners: dict[str, Callable[[Any], dict[str, Any]]] = {
            "query_catalog": lambda a: self.db.query(a.sql, a.limit),
            "review_sentiment": lambda a: self.sentiment.run(a),
            "predict_reception": lambda a: self.reception.predict(a),
            "review_trend": lambda a: self.trend.run(a),
        }

    def close(self) -> None:
        self.db.close()

    @property
    def sentiment(self) -> ReviewSentiment:
        if self._sentiment is None:
            name = "sentiment classifier"
            if self._sentiment_model is None:
                self._sentiment_model, name = _load_sentiment(self.settings)
            self._sentiment = ReviewSentiment(
                self.db,
                self.settings.processed_dir / "reviews.parquet",
                self._sentiment_model,
                model_name=name,
            )
        return self._sentiment

    @property
    def reception(self) -> Any:
        if self._reception_model is None:
            from steam_game_analyst.ml import reception

            self._reception_model = reception.load(self.settings.models_dir)
        return self._reception_model

    @property
    def trend(self) -> ReviewTrend:
        if self._trend is None:
            self._trend = ReviewTrend(self.db)
        return self._trend

    def schema(self, name: str) -> type[BaseModel]:
        return self._schemas[name]

    def langchain_tools(self) -> list[StructuredTool]:
        def make(name: str) -> StructuredTool:
            def run(**kwargs: Any) -> str:
                return self.invoke(name, kwargs).content

            return StructuredTool.from_function(
                func=run,
                name=name,
                description=DESCRIPTIONS[name],
                args_schema=self._schemas[name],
            )

        return [make(n) for n in TOOL_NAMES]

    def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self._schemas:
            return ToolResult(
                name, args, False, f"unknown tool {name!r}; use one of {TOOL_NAMES}", valid=False
            )
        try:
            parsed = self._schemas[name].model_validate(args)
        except ValidationError as e:
            return ToolResult(name, args, False, _validation_message(e), valid=False)
        try:
            out = self._runners[name](parsed)
        except SqlGuardError as e:
            return ToolResult(name, args, False, f"query refused: {e}", valid=False)
        except FileNotFoundError as e:
            return ToolResult(name, args, False, f"missing data or model: {e}", valid=False)
        ok = "error" not in out
        text = _dumps(out)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f'..."[truncated {len(text) - self.max_chars} chars]"'
        return ToolResult(name, parsed.model_dump(mode="json", exclude_none=True), ok, text)
