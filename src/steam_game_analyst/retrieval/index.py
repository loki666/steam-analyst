"""Processed tables -> chunks -> embeddings -> Qdrant (`sga index build`)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import polars as pl
from rich.console import Console
from tqdm import tqdm

from steam_game_analyst.retrieval import chunking
from steam_game_analyst.retrieval.chunking import Chunk
from steam_game_analyst.retrieval.embed import Embedder
from steam_game_analyst.retrieval.store import ChunkStore

console = Console()

# source name -> (processed table, chunker)
SOURCES: dict[str, tuple[str, Callable[[pl.DataFrame, dict[int, str]], list[Chunk]]]] = {
    "review": ("reviews.parquet", chunking.review_chunks),
    "news": ("news.parquet", chunking.news_chunks),
    "wiki": ("wiki.parquet", chunking.wiki_chunks),
}
# doc_type values each source writes (a news item is "patch_note" or "news")
DOC_TYPES: dict[str, tuple[str, ...]] = {
    "review": ("review",),
    "news": ("patch_note", "news"),
    "wiki": ("wiki",),
}


def build_index(
    processed_dir: Path,
    store: ChunkStore,
    embedder: Embedder,
    sources: tuple[str, ...] = tuple(SOURCES),
    recreate: bool = False,
    encode_batch: int = 4096,
) -> dict[str, int]:
    """Chunk, embed and upsert each source table, streaming `encode_batch` rows at a time.
    Points are tagged with this run's id; only after a source is fully written are its older
    points deleted, so a crash mid-way leaves the previous index intact (ids are stable, so
    unchanged chunks are simply overwritten). Returns chunks written per source."""
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; choose from {list(SOURCES)}")
    names = dict(
        pl.read_parquet(processed_dir / "games.parquet", columns=["appid", "name"]).iter_rows()
    )
    store.ensure(dim=embedder.dim, recreate=recreate)
    run_id = uuid.uuid4().hex
    counts: dict[str, int] = {}
    for source in sources:
        file, make_chunks = SOURCES[source]
        path = processed_dir / file
        if not path.exists():
            console.print(f"[yellow]index: {file} missing, skipping {source}[/]")
            counts[source] = 0
            continue
        table = pl.read_parquet(path)
        n = 0
        with tqdm(total=table.height, desc=f"embed {source}", unit="row") as bar:
            for rows in table.iter_slices(encode_batch):
                part = make_chunks(rows, names)
                if part:
                    vectors = embedder.encode([c.text for c in part])
                    store.upsert(part, vectors, extra_payload={"run_id": run_id})
                n += len(part)
                bar.update(rows.height)
        store.delete_stale(DOC_TYPES[source], run_id)
        counts[source] = n
        console.print(f"index: {source}: {n:,} chunks")
    return counts
