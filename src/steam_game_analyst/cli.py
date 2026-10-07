"""`sga` command line: data | index | train | ask | agent | eval."""

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
train_app = typer.Typer(no_args_is_help=True, help="W3: train the tool models into models/")
app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")
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

    from steam_game_analyst.data.wiki import FandomClient, download_wikis, load_category_specs

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
    categories_path = s.configs_dir / wcfg.categories_file
    try:
        categories = load_category_specs(
            (yaml.safe_load(categories_path.read_text()) or {}) if categories_path.exists() else {}
        )
    except ValueError as e:
        raise typer.BadParameter(f"{categories_path}: {e}") from e
    with httpx.Client(timeout=30, headers={"User-Agent": s.http_user_agent}) as client:
        fc = FandomClient(client, rate_per_sec=wcfg.rate_per_sec)
        download_wikis(
            s.raw_dir,
            games,
            fc,
            max_pages=wcfg.max_pages_per_wiki,
            min_length=wcfg.min_page_bytes,
            overrides={int(k): v for k, v in overrides.items()},
            categories=categories,
        )


@data_app.command("build")
def build(
    step: str = typer.Argument(
        "all", help="all | games | counts | subset | reviews | volume | news | wiki | duckdb"
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
        "volume": lambda: b.step_review_volume(s, cfg),
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
    b.step_review_volume(s, cfg)
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


# ----------------------------------------------------------------------------- train
@train_app.command("sentiment")
def train_sentiment(seed: int = typer.Option(13)) -> None:
    """TF-IDF + logistic regression on English reviews (held-out games) -> models/."""
    import polars as pl

    from steam_game_analyst.ml import sentiment

    s = get_settings()
    reviews = pl.read_parquet(
        s.processed_dir / "reviews.parquet",
        columns=["appid", "review_text", "voted_up", "language"],
    )
    model, metrics = sentiment.train(reviews, seed=seed)
    path = sentiment.save(model, metrics, s.models_dir)
    console.print({k: metrics[k] for k in ("C", "test", "majority_baseline")})
    console.print(f"-> {path}")


@train_app.command("sentiment-bert")
def train_sentiment_bert(
    seed: int = typer.Option(13),
    device: str = typer.Option("cuda", help="cuda | cpu (stop Ollama first: it needs the GPU)"),
) -> None:
    """Fine-tune DistilBERT on the same reviews and held-out games as the baseline -> models/."""
    import polars as pl

    from steam_game_analyst.ml import sentiment_bert

    s = get_settings()
    reviews = pl.read_parquet(
        s.processed_dir / "reviews.parquet",
        columns=["appid", "review_text", "voted_up", "language"],
    )
    model, metrics = sentiment_bert.train(
        reviews, seed=seed, device=device, cache_dir=str(s.models_dir / "hf")
    )
    path = sentiment_bert.save(model, metrics, s.models_dir)
    console.print(
        {k: metrics[k] for k in ("validation_roc_auc_by_epoch", "test", "majority_baseline")}
    )
    console.print(f"-> {path}")


@train_app.command("aspect-labels")
def train_aspect_labels(
    n: int = typer.Option(12000, help="Reviews from training games to label"),
    model: str = typer.Option(None, help="Ollama model (default: SGA_LLM_MODEL)"),
    seed: int = typer.Option(13),
) -> None:
    """Silver aspect labels from a local LLM for reviews of training games (resumable)."""
    import polars as pl

    from steam_game_analyst.ml import sentiment
    from steam_game_analyst.ml.aspect_labels import label_to_file
    from steam_game_analyst.retrieval.rag import OllamaChat

    s = get_settings()
    reviews = pl.read_parquet(
        s.processed_dir / "reviews.parquet",
        columns=["review_id", "appid", "review_text", "voted_up", "language"],
    ).filter(
        pl.col("language").is_in(sentiment.LANGUAGES)
        & pl.col("voted_up").is_not_null()
        & pl.col("review_text").is_not_null()
    )
    train_df, _ = sentiment.split_by_game(reviews, 0.2, seed)  # never the held-out games
    names = pl.read_parquet(s.processed_dir / "games.parquet", columns=["appid", "name"])
    picked = train_df.sample(min(n, train_df.height), seed=seed).join(names, on="appid", how="left")
    rows = [
        {"review_id": r["review_id"], "game": r["name"], "text": r["review_text"]}
        for r in picked.iter_rows(named=True)
    ]
    llm = OllamaChat(
        s.ollama_host, model or s.llm_model, options={"temperature": 0, "num_ctx": 2048}
    )
    out = s.processed_dir / "aspect_silver.jsonl"
    added = label_to_file(rows, llm, out, progress=True)
    console.print(f"labelled {added} new reviews -> {out}")


@train_app.command("aspects")
def train_aspects(
    seed: int = typer.Option(13),
    device: str = typer.Option("cuda", help="cuda | cpu (stop Ollama first: it needs the GPU)"),
) -> None:
    """Multi-label DistilBERT on silver aspect labels, scored on the gold set -> models/."""
    import polars as pl

    from steam_game_analyst.ml import aspect_bert

    s = get_settings()
    silver_file = s.processed_dir / "aspect_silver.jsonl"
    gold_file = s.data_dir / "eval" / "aspects_gold.jsonl"
    for f, how in ((silver_file, "sga train aspect-labels"), (gold_file, "the gold labelling")):
        if not f.exists():
            raise typer.BadParameter(f"{f} not found: run {how} first")
    labels = pl.read_ndjson(
        silver_file, schema={"review_id": pl.String, "aspects": pl.List(pl.String)}
    )
    reviews = pl.read_parquet(
        s.processed_dir / "reviews.parquet", columns=["review_id", "appid", "review_text"]
    )
    labels = labels.filter(pl.col("aspects").is_not_null())  # the LLM could not label these
    silver = labels.join(reviews, on="review_id").rename({"review_text": "text"})
    gold = [json.loads(line) for line in gold_file.read_text().splitlines() if line.strip()]
    model, metrics = aspect_bert.train(
        silver, gold, seed=seed, device=device, cache_dir=str(s.models_dir / "hf")
    )
    path = aspect_bert.save(model, metrics, s.models_dir)
    for name, m in metrics["gold"].items():
        f1 = {a: round(v["f1"], 3) for a, v in m["per_aspect"].items()}
        console.print(f"{name}: macro-F1 {m['macro_f1']:.3f} {m['macro_f1_ci95']} {f1}")
    console.print(f"-> {path}")


@train_app.command("reception")
def train_reception(seed: int = typer.Option(13)) -> None:
    """LightGBM reception-bucket model on catalog launch metadata (time split) -> models/."""
    import polars as pl

    from steam_game_analyst.ml import reception

    s = get_settings()
    model, metrics = reception.train(pl.read_parquet(s.processed_dir / "games_all.parquet"), seed)
    path = reception.save(model, metrics, s.models_dir)
    for key in ("test", "majority_baseline"):
        console.print(key, {k: v for k, v in metrics[key].items() if k != "reliability"})
    console.print("SHAP top features:", list(metrics["shap_mean_abs_top20"])[:10])
    console.print(f"-> {path}")


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


def _reranker():
    from steam_game_analyst.retrieval.rerank import CrossEncoderReranker

    s = get_settings()
    return CrossEncoderReranker(
        s.rerank_model, device=s.rerank_device, cache_dir=str(s.models_dir / "hf")
    )


def _token_counter():
    """The embedding model's tokenizer: chunk boundaries (and so chunk ids) must be the same
    wherever chunks are made, or gold-set chunk ids point at different text in the index."""
    from steam_game_analyst.retrieval.chunking import hf_token_counter

    s = get_settings()
    return hf_token_counter(s.embed_model, cache_dir=str(s.models_dir / "hf"))


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
    from steam_game_analyst.retrieval.index import build_index

    s = get_settings()
    counts = build_index(
        s.processed_dir,
        _store(),
        _embedder(device or s.index_embed_device),
        sources=tuple(source or ("review", "news", "wiki")),
        recreate=recreate,
        count=_token_counter(),
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


@app.command("agent")
def agent_cmd(
    question: str = typer.Argument(..., help="Analytical question about Steam games"),
    model: str = typer.Option(None, help="Ollama model (default: SGA_LLM_MODEL, the 14B)"),
    max_iterations: int = typer.Option(6, help="Cap on tool-calling rounds"),
    no_retrieval: bool = typer.Option(False, "--no-retrieval", help="Tools only, no Qdrant"),
    show_tools: bool = typer.Option(False, "--show-tools", help="Print every tool output"),
) -> None:
    """W3 agent: router -> retrieval -> tool loop -> cited answer (trace in data/traces/)."""
    import time

    import polars as pl

    from steam_game_analyst.agent.factory import append_trace, build_agent, trace_record

    s = get_settings()
    retriever = None
    if not no_retrieval:
        from steam_game_analyst.retrieval import rag
        from steam_game_analyst.retrieval.query import GameIndex

        subset = GameIndex.from_frame(pl.read_parquet(s.processed_dir / "games.parquet"))
        retriever = rag.Retriever(_store(), _embedder(s.query_embed_device), subset)
    llm_model = model or s.llm_model
    agent, toolbox = build_agent(s, llm_model, retriever, max_iterations)
    start = time.perf_counter()
    try:
        state = agent.run(question)
    except Exception as e:  # Ollama down, model missing: show it, do not dump a trace
        console.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(1) from e
    finally:
        toolbox.close()
    latency = time.perf_counter() - start
    append_trace(s.data_dir / "traces" / "agent.jsonl", trace_record(state, latency, llm_model))
    games = ", ".join(f"{g['name']} ({g['appid']})" for g in state.get("games", [])) or "none"
    console.print(f"[dim]games: {games} · plan: {state.get('plan') or '-'}[/]")
    for t in state.get("tool_results", []):
        mark = "[green]ok[/]" if t["ok"] else "[yellow]no data[/]"
        console.print(f"[dim][{t['id']}] {t['tool']}({t['args']}) {mark}[/]")
        if show_tools:
            console.print(t["content"], markup=False)
    for note in state.get("notes", []):
        console.print(f"[yellow]note: {note}[/]")
    console.print(f"\n{state.get('answer', '')}\n")
    for c in state.get("citations", []):
        if c["kind"] == "tool":
            console.print(f"[{c['id']}] tool {c['tool']} {c['args']}", markup=False)
        else:
            console.print(f"[{c['id']}] {c['label']} {c['url'] or ''}", markup=False)
    console.print(f"[dim]{latency:.1f}s · {state.get('iterations', 0)} agent steps[/]")


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
    count = _token_counter()
    chunks = []
    for file, make_chunks in SOURCES.values():
        if (s.processed_dir / file).exists():
            chunks += make_chunks(pl.read_parquet(s.processed_dir / file), names, count=count)
    picked = sample_chunks(chunks, per_type=per_type, seed=seed)
    llm_model = model or s.llm_dev_model
    llm = OllamaChat(s.ollama_host, llm_model, options={"temperature": 0.7, "num_ctx": 4096})
    items = generate_questions(picked, llm, model=llm_model)
    out = s.data_dir / "eval" / "synthetic_questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    console.print(f"wrote {len(items)} questions -> {out}")


def _gold_items() -> list[dict]:
    gold = get_settings().data_dir / "eval" / "synthetic_questions.jsonl"
    return [json.loads(line) for line in gold.read_text().splitlines() if line.strip()]


def _retriever(mode: str, store=None, embedder=None):
    import polars as pl

    from steam_game_analyst.retrieval.query import GameIndex
    from steam_game_analyst.retrieval.rag import Retriever

    s = get_settings()
    games = GameIndex.from_frame(pl.read_parquet(s.processed_dir / "games.parquet"))
    try:
        return Retriever(
            store or _store(),
            embedder or _embedder(s.query_embed_device),
            games,
            mode=mode,
            reranker=_reranker() if mode == "rerank" else None,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@eval_app.command("pool")
def eval_pool(
    split: str = typer.Option("dev", help="dev | test (frozen: needs --final)"),
    final: bool = typer.Option(False, "--final", help="Allow pooling the test split"),
    depth: int = typer.Option(10, help="Chunks taken from each retrieval mode"),
    new_only: bool = typer.Option(
        False, "--new-only", help="Leave out pairs already in data/eval/judgments.jsonl"
    ),
) -> None:
    """Pool every mode's top chunks per question (plus the generating chunk) for judging."""
    from steam_game_analyst.eval.judged import build_pool, load_judgments
    from steam_game_analyst.retrieval.rag import RETRIEVAL_MODES

    s = get_settings()
    store, embedder = _store(), _embedder(s.query_embed_device)
    retrievers = {m: _retriever(m, store, embedder) for m in RETRIEVAL_MODES}
    judgments_file = s.data_dir / "eval" / "judgments.jsonl"
    skip = set(load_judgments(judgments_file)) if new_only and judgments_file.exists() else set()
    try:
        pool = build_pool(
            _gold_items(), retrievers, store, split=split, depth=depth, final=final, skip=skip
        )
    except PermissionError as e:
        raise typer.BadParameter(str(e)) from e
    out = s.data_dir / "eval" / f"pool_{split}{'_new' if new_only else ''}.jsonl"
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool))
    n = sum(len(p["candidates"]) for p in pool)
    console.print(f"pool: {len(pool)} questions, {n} chunks to judge -> {out}")


@eval_app.command("retrieval")
def eval_retrieval(
    split: str = typer.Option("dev", help="dev (tune on this) | test (frozen: needs --final)"),
    final: bool = typer.Option(False, "--final", help="Allow the one-off run on the test split"),
    label: str = typer.Option("run", help="Name for this configuration, e.g. baseline"),
    k: int = typer.Option(None, help="Cut-off (default 20 documents; 10 chunks with --judged)"),
    mode: str = typer.Option(
        "hybrid", help="hybrid (dense + BM25, RRF) | dense | bm25 | rerank (hybrid + cross-encoder)"
    ),
    judged: bool = typer.Option(
        False, "--judged", help="Score against data/eval/judgments.jsonl (any relevant chunk)"
    ),
) -> None:
    """Retrieval metrics on the synthetic gold set: by default whether each question's own
    source document comes back; with --judged, hit/MRR/nDCG against judged relevance."""
    from datetime import UTC, datetime

    from steam_game_analyst.eval.judged import evaluate_judged, load_judgments
    from steam_game_analyst.eval.retrieval import evaluate

    s = get_settings()
    items = _gold_items()
    retriever = _retriever(mode)
    judgments_file = s.data_dir / "eval" / "judgments.jsonl"
    try:
        if judged:
            if not judgments_file.exists():
                raise typer.BadParameter(
                    f"{judgments_file} not found: run `sga eval pool` and judge it"
                )
            judgments = load_judgments(judgments_file)
            res = evaluate_judged(items, retriever, judgments, split=split, k=k or 10, final=final)
        else:
            res = evaluate(items, retriever, split=split, k=k or 20, final=final)
    except PermissionError as e:
        raise typer.BadParameter(str(e)) from e
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    kind = "judged" if judged else "source"
    res = {"label": label, "mode": mode, "scoring": kind, "created": stamp, **res}
    out = s.data_dir / "eval" / "results" / f"retrieval_{kind}_{split}_{label}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    console.print(res["overall"])
    for doc_type, m in res["by_doc_type"].items():
        console.print(f"  {doc_type:10} {m}")
    if judged and res["unjudged_hits"]:
        console.print(
            f"[yellow]{res['unjudged_hits']} retrieved chunks are unjudged and scored as "
            "irrelevant: pool and judge this configuration before comparing it[/]"
        )
    console.print(f"-> {out}")


@eval_app.command("all")
def eval_all(
    split: str = typer.Option("dev", help="dev | test (frozen: needs --final)"),
    final: bool = typer.Option(False, "--final", help="Allow the one-off run on the test split"),
    k: int = typer.Option(10, help="Chunks scored per question"),
    baseline: str = typer.Option("hybrid", help="Mode the others are compared with"),
) -> None:
    """Retrieval ablation: every mode scored against judged relevance, with bootstrap 95% CIs
    and paired differences from the baseline. Writes JSON and a Markdown table."""
    from datetime import UTC, datetime

    from steam_game_analyst.eval.ablation import ablation, to_markdown
    from steam_game_analyst.eval.judged import evaluate_judged, load_judgments
    from steam_game_analyst.retrieval.rag import RETRIEVAL_MODES

    s = get_settings()
    judgments_file = s.data_dir / "eval" / "judgments.jsonl"
    if not judgments_file.exists():
        raise typer.BadParameter(f"{judgments_file} not found: run `sga eval pool` and judge it")
    judgments = load_judgments(judgments_file)
    items = _gold_items()
    store, embedder = _store(), _embedder(s.query_embed_device)
    results = {}
    try:
        for mode in RETRIEVAL_MODES:
            retriever = _retriever(mode, store, embedder)
            results[mode] = evaluate_judged(
                items, retriever, judgments, split=split, k=k, final=final
            )
        table = ablation(results, baseline=baseline)
    except (PermissionError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    md = to_markdown(table, split=split)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = s.data_dir / "eval" / "results" / f"ablation_{split}_{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"split": split, "created": stamp, **table, "runs": results}, indent=1)
    )
    out.with_suffix(".md").write_text(md)
    console.print(md)
    console.print(f"-> {out}.json, {out}.md")


if __name__ == "__main__":
    app()
