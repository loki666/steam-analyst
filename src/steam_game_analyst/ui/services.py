"""Backends for the UI: build the agent the way `sga agent` does, check that Ollama and Qdrant
answer, and turn failures into a message a person can act on instead of a traceback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from steam_game_analyst.config import Settings

HEALTH_TIMEOUT_S = 2.0
MAX_ITERATIONS = 6
SEARCH_MODES = {"Standard": "hybrid", "Thorough (+3–5 s)": "rerank"}


@dataclass(frozen=True)
class Failure:
    kind: Literal["llm_down", "model_missing", "data_missing", "error"]
    title: str
    body: str
    detail: str


def ping(url: str, timeout: float = HEALTH_TIMEOUT_S) -> str | None:
    """None when the service answers, else a one-line reason."""
    try:
        httpx.get(url, timeout=timeout).raise_for_status()
    except httpx.HTTPError as e:
        return f"{type(e).__name__}: {e}"
    return None


def health(settings: Settings) -> dict[str, str | None]:
    return {
        "ollama": ping(f"{settings.ollama_host.rstrip('/')}/api/tags"),
        "qdrant": ping(f"{settings.qdrant_url.rstrip('/')}/collections"),
    }


def llm_down(settings: Settings, reason: str) -> Failure:
    return Failure(
        "llm_down",
        "The language model isn't running",
        "Your question was kept. Start the model with `make up`, then retry.",
        f"Ollama unreachable at {settings.ollama_host} ({reason})",
    )


def _is_connection_error(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, ConnectionError | httpx.ConnectError | httpx.ConnectTimeout) or (
        "failed to connect" in text or "connection refused" in text
    )


def classify(e: BaseException, settings: Settings) -> Failure:
    """An exception from building or running the agent, in words."""
    detail = f"{type(e).__name__}: {e}"
    if _is_connection_error(e):
        return llm_down(settings, detail)
    if "not found" in str(e).lower() and type(e).__name__ == "ResponseError":
        return Failure(
            "model_missing",
            "The language model isn't downloaded",
            f"Ollama is running but has no `{settings.llm_model}`. Run `make pull-models`, "
            "then retry.",
            detail,
        )
    if isinstance(e, FileNotFoundError):
        return Failure(
            "data_missing",
            "Data or a trained model is missing",
            "Build the tables with `make data` and the models with `make train`, then retry.",
            detail,
        )
    return Failure(
        "error", "Something went wrong while answering", "Retry, or rephrase the question.", detail
    )


def corpus_facts(settings: Settings) -> str | None:
    """The sidebar's one-line data description, read from the tables (None without data)."""
    from steam_game_analyst.tools.catalog import CatalogDB

    try:
        db = CatalogDB(settings.duckdb_path)
    except FileNotFoundError:
        return None
    try:
        n_games = db.fetch("SELECT count(DISTINCT appid) FROM review_monthly")[0][0]
        end = db.fetch("SELECT max(created_at)::DATE FROM reviews")[0][0]
    finally:
        db.close()
    until = f" · reviews to {end.strftime('%b %Y')}" if end else ""
    return f"{n_games} games with review text · full Steam catalog{until}"


def build_retriever(settings: Settings, mode: str) -> Any:
    """The W2 retriever over Qdrant (hybrid, or rerank for the thorough search)."""
    import polars as pl

    from steam_game_analyst.cli import _embedder, _reranker, _store
    from steam_game_analyst.retrieval import rag
    from steam_game_analyst.retrieval.query import GameIndex

    subset = GameIndex.from_frame(pl.read_parquet(settings.processed_dir / "games.parquet"))
    return rag.Retriever(
        _store(),
        _embedder(settings.query_embed_device),
        subset,
        mode=mode,  # type: ignore[arg-type]
        reranker=_reranker() if mode == "rerank" else None,
    )


def build(settings: Settings, retriever: Any | None) -> Any:
    """The agent exactly as `sga agent` builds it (the 14B model, 6 tool rounds)."""
    from steam_game_analyst.agent.factory import build_agent

    agent, _toolbox = build_agent(settings, settings.llm_model, retriever, MAX_ITERATIONS)
    return agent


def search_only(retriever: Any, question: str, k: int = 8) -> list[dict[str, Any]]:
    """Matching documents without a written answer, shaped like the agent's chunks."""
    from steam_game_analyst.retrieval.rag import describe

    hits = retriever.retrieve(question, k=k).hits
    return [
        {
            "n": i,
            "id": h.id,
            "label": describe(h),
            "text": h.payload.get("text", ""),
            "url": h.payload.get("url") or None,
        }
        for i, h in enumerate(hits, start=1)
    ]
