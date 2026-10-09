"""Wiring for the CLI: models, data handles, entity index, optional retriever, run traces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from steam_game_analyst.agent.graph import Agent
from steam_game_analyst.agent.llm import chat_models
from steam_game_analyst.config import Settings
from steam_game_analyst.retrieval.query import GameIndex
from steam_game_analyst.tools.registry import ToolBox

# catalog games outside the subset are linked by name when they are this popular, so a
# question about a well-known game resolves to its appid even without review text
LINK_MIN_VOTES = 1000


def game_index(processed_dir: Path) -> GameIndex:
    catalog = pl.read_parquet(
        processed_dir / "games_all.parquet", columns=["appid", "name", "positive", "negative"]
    ).filter((pl.col("positive") + pl.col("negative")) >= LINK_MIN_VOTES)
    subset = pl.read_parquet(processed_dir / "games.parquet", columns=["appid", "name"])
    names = {**dict(catalog.select("appid", "name").iter_rows()), **dict(subset.iter_rows())}
    return GameIndex.from_names(names)


def build_agent(
    settings: Settings,
    model: str,
    retriever: Any | None,
    max_iterations: int,
) -> tuple[Agent, ToolBox]:
    llm, router_llm = chat_models(settings, model)
    toolbox = ToolBox(settings)
    corpus = {r[0] for r in toolbox.db.fetch("SELECT DISTINCT appid FROM review_monthly")}
    agent = Agent(
        llm,
        toolbox,
        game_index(settings.processed_dir),
        corpus_appids=corpus,
        retriever=retriever,
        router_llm=router_llm,
        max_iterations=max_iterations,
    )
    return agent, toolbox


def trace_record(state: dict[str, Any], latency_s: float, model: str) -> dict[str, Any]:
    route = state.get("route")
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "question": state.get("question"),
        "games": state.get("games", []),
        "route": route.model_dump() if route is not None else None,
        "tool_calls": [
            {"id": t["id"], "tool": t["tool"], "args": t["args"], "ok": t["ok"]}
            for t in state.get("tool_results", [])
        ],
        "tool_errors": state.get("tool_errors", []),
        "documents": [
            {"n": c["n"], "id": c["id"], "label": c["label"]} for c in state.get("chunks", [])
        ],
        "iterations": state.get("iterations", 0),
        "errors": state.get("errors", 0),
        "notes": state.get("notes", []),
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "latency_s": round(latency_s, 2),
        "usage": state.get("usage"),
    }


def append_trace(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
