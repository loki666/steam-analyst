"""Orchestrates W1: raw -> processed Parquet -> DuckDB.

Steps (each idempotent, each writes one artefact):
  1. games_all.parquet     full catalog, canonical columns
  2. review_counts.parquet appid, n_reviews (English) scanned from the raw review files
  3. games.parquet         the selected subset (+ n_reviews)
  4. reviews.parquet       cleaned, sampled reviews for the subset
  5. news.parquet          patch notes / announcements for the subset (needs raw/news)
  6. sga.duckdb            views over the parquet files + a few derived tables
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
from rich.console import Console

from steam_game_analyst.config import PipelineConfig, Settings
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
        df = subset.sample_reviews(df, cfg.subset.reviews_per_game, cfg.subset.languages)
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

-- monthly review aggregates per game: the backbone of review_trend()
CREATE OR REPLACE TABLE review_monthly AS
SELECT appid,
       date_trunc('month', created_at)::DATE AS month,
       count(*)                               AS n_reviews,
       avg(CASE WHEN voted_up THEN 1 ELSE 0 END) AS positive_share,
       avg(playtime_at_review_min)            AS avg_playtime_at_review_min
FROM reviews
WHERE created_at IS NOT NULL
GROUP BY 1, 2;

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


def step_duckdb(settings: Settings) -> Path:
    db = settings.duckdb_path
    db.parent.mkdir(parents=True, exist_ok=True)
    for name in ("games", "games_all", "reviews", "news"):
        if not (settings.processed_dir / f"{name}.parquet").exists():
            raise FileNotFoundError(f"missing {name}.parquet; run the earlier steps first")
    con = duckdb.connect(str(db))
    try:
        con.execute(DUCKDB_DDL.format(p=settings.processed_dir.as_posix()))
        n = con.execute("SELECT count(*) FROM reviews").fetchone()[0]
    finally:
        con.close()
    console.print(f"duckdb: {db} ready ({n:,} reviews)")
    return db


def build_all(settings: Settings, cfg: PipelineConfig) -> None:
    step_games_all(settings)
    step_review_counts(settings, cfg.subset.languages)
    step_games_subset(settings, cfg)
    step_reviews(settings, cfg)
    step_news(settings)
    step_duckdb(settings)
