"""Orchestrates W1: raw -> processed Parquet -> DuckDB.

Steps (each idempotent, each writes one artefact):
  1. games_all.parquet     full catalog, canonical columns
  2. review_counts.parquet appid, n_reviews (English) scanned from the raw review files
  3. games.parquet         the selected subset (+ n_reviews)
  4. reviews.parquet       cleaned, sampled reviews for the subset
  4b. review_volume.parquet monthly review / positive counts over every cleaned review (the
                           sample is stratified by year and rating, so it cannot carry trends)
  5. news.parquet          patch notes / announcements for the subset (needs raw/news)
  6. wiki.parquet          fandom wiki sections for the subset (needs raw/wiki)
  7. sga.duckdb            views over the parquet files + a few derived tables
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import polars as pl
from rich.console import Console

from steam_game_analyst.config import CleaningConfig, PipelineConfig, Settings
from steam_game_analyst.data import subset, transform

console = Console()


def _catalog_file(raw_dir: Path) -> Path:
    d = raw_dir / "catalog"
    for ext in ("*.parquet", "*.csv", "*.json"):
        files = sorted(d.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f"No catalog file in {d}; run `sga data download-catalog`")


def _review_files(raw_dir: Path) -> list[Path]:
    d = raw_dir / "reviews"
    files = [p for p in d.rglob("*.csv") if transform.appid_from_filename(p) is not None]
    files += [p for p in d.rglob("*.jsonl") if transform.appid_from_filename(p) is not None]
    return sorted(files)


def step_games_all(settings: Settings) -> pl.DataFrame:
    out = settings.processed_dir / "games_all.parquet"
    if out.exists():
        return pl.read_parquet(out)
    raw = transform.read_catalog_raw(_catalog_file(settings.raw_dir))
    df = transform.build_games(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    console.print(f"games_all: {df.height:,} rows -> {out}")
    return df


def step_review_counts(settings: Settings, languages: list[str]) -> pl.DataFrame:
    """Cheap pre-pass: count usable reviews per appid without materialising all text.

    The Mendeley file name carries the total review count; we use it as a fast proxy, then
    refine with the actual language column for files we end up reading."""
    out = settings.processed_dir / "review_counts.parquet"
    if out.exists():
        return pl.read_parquet(out)
    rows = []
    for p in _review_files(settings.raw_dir):
        appid = transform.appid_from_filename(p)
        parts = p.stem.split("_")
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if n is None:  # fall back to counting lines
            n = max(sum(1 for _ in open(p, "rb")) - 1, 0)
        rows.append({"appid": appid, "n_reviews": n, "path": str(p)})
    df = pl.DataFrame(rows, schema={"appid": pl.Int64, "n_reviews": pl.Int64, "path": pl.Utf8})
    df.write_parquet(out)
    console.print(f"review_counts: {df.height:,} games with review files")
    return df


def step_games_subset(settings: Settings, cfg: PipelineConfig) -> pl.DataFrame:
    out = settings.processed_dir / "games.parquet"
    games_all = step_games_all(settings)
    counts = step_review_counts(settings, cfg.subset.languages).select(["appid", "n_reviews"])
    df = subset.select_games(games_all, counts, cfg.subset)
    df.write_parquet(out)
    console.print(f"games (subset): {df.height:,} rows -> {out}")
    (settings.processed_dir / "subset_appids.json").write_text(json.dumps(df["appid"].to_list()))
    return df


def step_reviews(settings: Settings, cfg: PipelineConfig) -> pl.DataFrame:
    out = settings.processed_dir / "reviews.parquet"
    games = pl.read_parquet(settings.processed_dir / "games.parquet")
    counts = pl.read_parquet(settings.processed_dir / "review_counts.parquet")
    paths = counts.filter(pl.col("appid").is_in(games["appid"].implode()))
    frames: list[pl.DataFrame] = []
    for appid, path in zip(paths["appid"], paths["path"], strict=True):
        raw = transform.read_review_file(Path(path), appid=int(appid))
        df = transform.build_reviews(raw, cfg.cleaning)
        langs = cfg.subset.languages
        if langs:  # filtering by language: detect everything first
            df = transform.fill_languages(df)
        df = subset.sample_reviews(df, cfg.subset.reviews_per_game, langs)
        df = transform.fill_languages(df)  # no-op when already detected
        frames.append(df)
    reviews = (
        pl.concat(frames, how="vertical_relaxed")
        if frames
        else pl.DataFrame(schema=transform.REVIEWS_SCHEMA)
    )  # type: ignore[arg-type]
    reviews.write_parquet(out)
    console.print(
        f"reviews: {reviews.height:,} rows for {reviews['appid'].n_unique()} games -> {out}"
    )
    return reviews


def _monthly_volume(args: tuple[int, str, CleaningConfig]) -> pl.DataFrame:
    appid, path, cleaning = args
    df = transform.build_reviews(transform.read_review_file(Path(path), appid=appid), cleaning)
    return (
        df.filter(pl.col("created_at").is_not_null())
        .group_by("appid", pl.col("created_at").dt.truncate("1mo").cast(pl.Date).alias("month"))
        .agg(
            pl.len().cast(pl.Int64).alias("n_reviews"),
            pl.col("voted_up").cast(pl.Int64).sum().alias("n_positive"),
        )
    )


def step_review_volume(settings: Settings, cfg: PipelineConfig) -> pl.DataFrame:
    """Monthly review and positive counts over every cleaned review of each subset game (the
    population the sample was drawn from): exact trends, and inverse-probability weights for
    the stratified sample."""
    out = settings.processed_dir / "review_volume.parquet"
    games = pl.read_parquet(settings.processed_dir / "games.parquet", columns=["appid"])
    counts = pl.read_parquet(settings.processed_dir / "review_counts.parquet")
    paths = counts.filter(pl.col("appid").is_in(games["appid"].implode()))
    jobs = [
        (int(a), str(p), cfg.cleaning) for a, p in zip(paths["appid"], paths["path"], strict=True)
    ]
    schema = {"appid": pl.Int64, "month": pl.Date, "n_reviews": pl.Int64, "n_positive": pl.Int64}
    # spawn, not fork: polars' thread pool deadlocks in forked children
    workers = min(8, os.cpu_count() or 1, max(len(jobs), 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        frames = list(pool.map(_monthly_volume, jobs))
    vol = (
        pl.concat(frames, how="vertical_relaxed").cast(schema)  # type: ignore[arg-type]
        if frames
        else pl.DataFrame(schema=schema)
    ).sort("appid", "month")
    vol.write_parquet(out)
    console.print(
        f"review_volume: {int(vol['n_reviews'].sum()):,} reviews in {vol.height:,} game-months -> {out}"
    )
    return vol


def step_news(settings: Settings) -> pl.DataFrame:
    out = settings.processed_dir / "news.parquet"
    news_dir = settings.raw_dir / "news"
    appids = pl.read_parquet(settings.processed_dir / "games.parquet", columns=["appid"])
    df = transform.build_news(
        news_dir if news_dir.exists() else Path("/nonexistent"), appids["appid"].to_list()
    )
    df.write_parquet(out)
    console.print(
        f"news: {df.height:,} items ({int(df['is_patch_note'].sum()) if df.height else 0} patch notes) -> {out}"
    )
    return df


DUCKDB_DDL = """
CREATE OR REPLACE VIEW games      AS SELECT * FROM read_parquet('{p}/games.parquet');
CREATE OR REPLACE VIEW games_all  AS SELECT * FROM read_parquet('{p}/games_all.parquet');
CREATE OR REPLACE VIEW reviews    AS SELECT * FROM read_parquet('{p}/reviews.parquet');
CREATE OR REPLACE VIEW news       AS SELECT * FROM read_parquet('{p}/news.parquet');

CREATE OR REPLACE VIEW review_volume AS SELECT * FROM read_parquet('{p}/review_volume.parquet');

-- monthly review aggregates per game: the backbone of review_trend(). n_reviews, n_positive and
-- positive_share count every cleaned review (population); n_sampled counts the text sample.
CREATE OR REPLACE TABLE review_monthly AS
SELECT v.appid, v.month, v.n_reviews, v.n_positive,
       v.n_positive / v.n_reviews             AS positive_share,
       coalesce(s.n_sampled, 0)               AS n_sampled
FROM review_volume v
LEFT JOIN (SELECT appid, date_trunc('month', created_at)::DATE AS month, count(*) AS n_sampled
           FROM reviews WHERE created_at IS NOT NULL GROUP BY 1, 2) s USING (appid, month);

-- one row per game with a few headline numbers for the catalog tool
CREATE OR REPLACE TABLE game_summary AS
SELECT g.appid, g.name, g.release_date, g.price_usd, g.genres, g.tags,
       g.positive, g.negative, g.positive_ratio, g.metacritic_score,
       g.estimated_owners_low, g.estimated_owners_high, g.peak_ccu,
       g.average_playtime_forever, g.median_playtime_forever,
       r.n_sampled_reviews, r.sampled_positive_share, n.n_news, n.n_patch_notes
FROM games g
LEFT JOIN (SELECT appid, count(*) n_sampled_reviews,
                  avg(CASE WHEN voted_up THEN 1 ELSE 0 END) sampled_positive_share
           FROM reviews GROUP BY 1) r USING (appid)
LEFT JOIN (SELECT appid, count(*) n_news, sum(CASE WHEN is_patch_note THEN 1 ELSE 0 END) n_patch_notes
           FROM news GROUP BY 1) n USING (appid);
"""


def step_wiki(settings: Settings) -> pl.DataFrame:
    out = settings.processed_dir / "wiki.parquet"
    appids = pl.read_parquet(settings.processed_dir / "games.parquet", columns=["appid"])
    df = transform.build_wiki(settings.raw_dir / "wiki", appids["appid"].to_list())
    df.write_parquet(out)
    console.print(
        f"wiki: {df.height:,} sections from {df['pageid'].n_unique() if df.height else 0} pages "
        f"in {df['slug'].n_unique() if df.height else 0} wikis -> {out}"
    )
    return df


def step_duckdb(settings: Settings) -> Path:
    db = settings.duckdb_path
    db.parent.mkdir(parents=True, exist_ok=True)
    for name in ("games", "games_all", "reviews", "review_volume", "news"):
        if not (settings.processed_dir / f"{name}.parquet").exists():
            raise FileNotFoundError(f"missing {name}.parquet; run the earlier steps first")
    con = duckdb.connect(str(db))
    try:
        con.execute(DUCKDB_DDL.format(p=settings.processed_dir.as_posix()))
        if (settings.processed_dir / "wiki.parquet").exists():
            con.execute(
                "CREATE OR REPLACE VIEW wiki AS SELECT * FROM read_parquet("
                f"'{settings.processed_dir.as_posix()}/wiki.parquet')"
            )
        row = con.execute("SELECT count(*) FROM reviews").fetchone()
        n = row[0] if row else 0
    finally:
        con.close()
    console.print(f"duckdb: {db} ready ({n:,} reviews)")
    return db


def build_all(settings: Settings, cfg: PipelineConfig) -> None:
    step_games_all(settings)
    step_review_counts(settings, cfg.subset.languages)
    step_games_subset(settings, cfg)
    step_reviews(settings, cfg)
    step_review_volume(settings, cfg)
    step_news(settings)
    step_wiki(settings)
    step_duckdb(settings)
