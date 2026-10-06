"""`sga` command line: data | index | eval | serve. W1 implements `data`; the others are stubs."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console

from steam_game_analyst import __version__
from steam_game_analyst.config import get_settings, load_pipeline_config

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Steam Game Analyst pipeline CLI"
)
data_app = typer.Typer(no_args_is_help=True, help="W1: download raw sources and build the tables")
index_app = typer.Typer(no_args_is_help=True, help="W2: chunk, embed and load into Qdrant")
eval_app = typer.Typer(no_args_is_help=True, help="W4/W5: evaluation suite")
app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def _root(version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


# ----------------------------------------------------------------------------- data
@data_app.command("download-catalog")
def download_catalog() -> None:
    """FronkonGames catalog from Hugging Face."""
    from steam_game_analyst.data.download import download_catalog as dl

    s, cfg = get_settings(), load_pipeline_config()
    src = cfg.sources.get("catalog", {})
    dl(s.raw_dir, src.get("hf_repo", "FronkonGames/steam-games-dataset"), src.get("hf_file"))


@data_app.command("download-reviews")
def download_reviews() -> None:
    """Mendeley 31M-review dataset (large: several GB)."""
    from steam_game_analyst.data.download import download_reviews as dl

    s, cfg = get_settings(), load_pipeline_config()
    src = cfg.sources.get("reviews", {})
    dl(s.raw_dir, src.get("mendeley_id", "jxy85cr3th"), int(src.get("mendeley_version", 2)))


@data_app.command("download-news")
def download_news() -> None:
    """Steam news / patch notes for the subset appids (requires games.parquet)."""
    from steam_game_analyst.data.download import download_news as dl

    s, cfg = get_settings(), load_pipeline_config()
    ids_file = s.processed_dir / "subset_appids.json"
    if not ids_file.exists():
        raise typer.BadParameter("subset_appids.json missing; run `sga data build` first")
    appids = json.loads(ids_file.read_text())
    dl(s.raw_dir, appids, s.steam_api_key, cfg.news)


@data_app.command("download-wiki")
def download_wiki() -> None:
    """Fandom wiki pages for the subset games (requires games.parquet)."""
    import httpx
    import polars as pl
    import yaml

    from steam_game_analyst.data.wiki import FandomClient, download_wikis

    s, cfg = get_settings(), load_pipeline_config()
    games_file = s.processed_dir / "games.parquet"
    if not games_file.exists():
        raise typer.BadParameter("games.parquet missing; run `sga data build` first")
    games = pl.read_parquet(games_file, columns=["appid", "name"]).rows()
    wcfg = cfg.wiki
    overrides_path = s.configs_dir / wcfg.overrides_file
    overrides = (
        (yaml.safe_load(overrides_path.read_text()) or {}) if overrides_path.exists() else {}
    )
    with httpx.Client(timeout=30, headers={"User-Agent": s.http_user_agent}) as client:
        fc = FandomClient(client, rate_per_sec=wcfg.rate_per_sec)
        download_wikis(
            s.raw_dir,
            games,
            fc,
            max_pages=wcfg.max_pages_per_wiki,
            min_length=wcfg.min_page_bytes,
            overrides={int(k): v for k, v in overrides.items()},
        )


@data_app.command("build")
def build(
    step: str = typer.Argument(
        "all", help="all | games | counts | subset | reviews | news | wiki | duckdb"
    ),
) -> None:
    """Raw -> processed Parquet -> DuckDB."""
    from steam_game_analyst.data import build as b

    s, cfg = get_settings(), load_pipeline_config()
    steps = {
        "games": lambda: b.step_games_all(s),
        "counts": lambda: b.step_review_counts(s, cfg.subset.languages),
        "subset": lambda: b.step_games_subset(s, cfg),
        "reviews": lambda: b.step_reviews(s, cfg),
        "news": lambda: b.step_news(s),
        "wiki": lambda: b.step_wiki(s),
        "duckdb": lambda: b.step_duckdb(s),
        "all": lambda: b.build_all(s, cfg),
    }
    if step not in steps:
        raise typer.BadParameter(f"unknown step {step}; choose from {list(steps)}")
    steps[step]()


@data_app.command("all")
def data_all(
    skip_news: bool = typer.Option(False, help="Skip the Steam news download"),
    skip_wiki: bool = typer.Option(False, help="Skip the Fandom wiki download"),
) -> None:
    """Download everything and build all tables (the `make data` target)."""
    from steam_game_analyst.data import build as b

    download_catalog()
    download_reviews()
    s, cfg = get_settings(), load_pipeline_config()
    b.step_games_all(s)
    b.step_review_counts(s, cfg.subset.languages)
    b.step_games_subset(s, cfg)
    if not skip_news:
        download_news()
    if not skip_wiki:
        download_wiki()
    b.step_reviews(s, cfg)
    b.step_news(s)
    b.step_wiki(s)
    b.step_duckdb(s)


@data_app.command("stats")
def stats() -> None:
    """Quick sanity numbers from the DuckDB file."""
    import duckdb

    s = get_settings()
    con = duckdb.connect(str(s.duckdb_path), read_only=True)
    for q in (
        "SELECT count(*) AS games FROM games",
        "SELECT count(*) AS reviews, round(avg(CASE WHEN voted_up THEN 1 ELSE 0 END),3) AS positive_share FROM reviews",
        "SELECT count(*) AS news, sum(CASE WHEN is_patch_note THEN 1 ELSE 0 END) AS patch_notes FROM news",
        "SELECT min(month) AS first_month, max(month) AS last_month FROM review_monthly",
    ):
        console.print(con.execute(q).pl())
    con.close()


# ----------------------------------------------------------------------------- index / ask
def _store():
    from qdrant_client import QdrantClient

    from steam_game_analyst.retrieval.store import ChunkStore

    s = get_settings()
    return ChunkStore(QdrantClient(url=s.qdrant_url, timeout=60), s.qdrant_collection)


def _embedder(device: str):
    from steam_game_analyst.retrieval.embed import SentenceEmbedder

    s = get_settings()
    return SentenceEmbedder(s.embed_model, device=device, cache_dir=str(s.models_dir / "hf"))


@index_app.command("build")
def index_build(
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="review | news | wiki (repeatable; default all)"),
    ] = None,
    recreate: bool = typer.Option(False, help="Drop and rebuild the Qdrant collection"),
    device: str = typer.Option(None, help="Embedding device (default: SGA_INDEX_EMBED_DEVICE)"),
) -> None:
    """Chunk the processed tables, embed them and load them into Qdrant."""
    from steam_game_analyst.retrieval.chunking import hf_token_counter
    from steam_game_analyst.retrieval.index import build_index

    s = get_settings()
    counts = build_index(
        s.processed_dir,
        _store(),
        _embedder(device or s.index_embed_device),
        sources=tuple(source or ("review", "news", "wiki")),
        recreate=recreate,
        count=hf_token_counter(s.embed_model, cache_dir=str(s.models_dir / "hf")),
    )
    console.print(counts)


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question about the games in the subset"),
    k: int = typer.Option(8, help="Chunks to retrieve"),
    model: str = typer.Option(None, help="Ollama model (default: SGA_LLM_MODEL, the 14B)"),
    game: Annotated[
        list[str] | None, typer.Option(help="Restrict to a game name (repeatable)")
    ] = None,
    doc_type: Annotated[
        list[str] | None, typer.Option(help="review | patch_note | news | wiki (repeatable)")
    ] = None,
    min_hours: int = typer.Option(None, help="Reviews from players with at least N hours"),
    language: str = typer.Option(None, help="Reviews in one language, e.g. english, chinese"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only show the retrieved sources"),
) -> None:
    """Baseline RAG: retrieve with metadata filters and answer with cited sources."""
    import polars as pl

    from steam_game_analyst.retrieval import rag
    from steam_game_analyst.retrieval.filters import SearchFilters
    from steam_game_analyst.retrieval.query import GameIndex

    s = get_settings()
    games = GameIndex.from_frame(pl.read_parquet(s.processed_dir / "games.parquet"))
    game, doc_type = game or [], doc_type or []
    appids = tuple(a for g in game for a, _ in games.find(g, strict=False))
    if game and not appids:
        raise typer.BadParameter(f"no game in the subset matches {game}")
    overrides = SearchFilters(
        appids=appids,
        doc_types=tuple(doc_type),
        min_playtime_min=min_hours * 60 if min_hours is not None else None,
        language=language.lower() if language else None,
    )
    retriever = rag.Retriever(_store(), _embedder(s.query_embed_device), games)
    if no_llm:
        r = retriever.retrieve(question, k=k, overrides=overrides)
        _print_retrieval(r)
        _print_sources(r.hits)
        return
    llm = rag.OllamaChat(s.ollama_host, model or s.llm_model)
    try:
        ans = rag.answer(question, retriever, llm, k=k, overrides=overrides)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    if ans.retrieval:
        _print_retrieval(ans.retrieval)
    console.print(f"\n{ans.text}\n")
    _print_sources(ans.sources, cited=set(ans.cited))


def _print_retrieval(r) -> None:
    games = ", ".join(n for _, n in r.parsed.games) or "none"
    f = {k: v for k, v in r.filters.__dict__.items() if v not in (None, ())}
    note = f"  [yellow](relaxed: {', '.join(r.relaxed)})[/]" if r.relaxed else ""
    console.print(f"[dim]games: {games} · filters: {f}[/]{note}")


def _print_sources(hits, cited: set[int] | None = None) -> None:
    from steam_game_analyst.retrieval.rag import describe

    for i, h in enumerate(hits, start=1):
        mark = "" if cited is None or i in cited else "[dim]"
        url = h.payload.get("url") or ""
        text = " ".join(h.payload.get("text", "").split())[:160]
        console.print(f"{mark}[{i}] {describe(h)} ({h.score:.2f}) {url}\n    {text}")


# ----------------------------------------------------------------------------- stubs


@eval_app.command("synth")
def eval_synth(
    per_type: int = typer.Option(
        25, help="Questions per doc type (review, patch_note, news, wiki)"
    ),
    model: str = typer.Option(None, help="Ollama model (default: SGA_LLM_DEV_MODEL)"),
    seed: int = typer.Option(13),
) -> None:
    """Start the gold set: one LLM-written question per sampled chunk (hand-check before use)."""
    import polars as pl

    from steam_game_analyst.eval.synth import generate_questions, sample_chunks
    from steam_game_analyst.retrieval.index import SOURCES
    from steam_game_analyst.retrieval.rag import OllamaChat

    s = get_settings()
    games_df = pl.read_parquet(s.processed_dir / "games.parquet", columns=["appid", "name"])
    names: dict[int, str] = dict(games_df.iter_rows())
    chunks = []
    for file, make_chunks in SOURCES.values():
        if (s.processed_dir / file).exists():
            chunks += make_chunks(pl.read_parquet(s.processed_dir / file), names)
    picked = sample_chunks(chunks, per_type=per_type, seed=seed)
    llm_model = model or s.llm_dev_model
    llm = OllamaChat(s.ollama_host, llm_model, options={"temperature": 0.7, "num_ctx": 4096})
    items = generate_questions(picked, llm, model=llm_model)
    out = s.data_dir / "eval" / "synthetic_questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    console.print(f"wrote {len(items)} questions -> {out}")


@eval_app.command("retrieval")
def eval_retrieval(
    split: str = typer.Option("dev", help="dev (tune on this) | test (frozen: needs --final)"),
    final: bool = typer.Option(False, "--final", help="Allow the one-off run on the test split"),
    label: str = typer.Option("run", help="Name for this configuration, e.g. baseline"),
    k: int = typer.Option(20, help="Documents considered per question"),
) -> None:
    """Document-level recall@5/@20 and MRR@20 on the synthetic gold set."""
    from datetime import UTC, datetime

    import polars as pl

    from steam_game_analyst.eval.retrieval import evaluate
    from steam_game_analyst.retrieval.query import GameIndex
    from steam_game_analyst.retrieval.rag import Retriever

    s = get_settings()
    gold = s.data_dir / "eval" / "synthetic_questions.jsonl"
    items = [json.loads(line) for line in gold.read_text().splitlines() if line.strip()]
    games = GameIndex.from_frame(pl.read_parquet(s.processed_dir / "games.parquet"))
    retriever = Retriever(_store(), _embedder(s.query_embed_device), games)
    try:
        res = evaluate(items, retriever, split=split, k=k, final=final)
    except PermissionError as e:
        raise typer.BadParameter(str(e)) from e
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    res = {"label": label, "created": stamp, "gold_file": gold.name, **res}
    out = s.data_dir / "eval" / "results" / f"retrieval_{split}_{label}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    console.print(res["overall"])
    for doc_type, m in res["by_doc_type"].items():
        console.print(f"  {doc_type:10} {m}")
    console.print(f"-> {out}")


@eval_app.command("all")
def eval_all() -> None:
    """W4/W5: not implemented yet."""
    console.print("[yellow]eval: coming in W4[/]")


if __name__ == "__main__":
    app()
