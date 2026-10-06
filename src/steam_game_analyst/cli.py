"""`sga` command line: data | index | eval | serve. W1 implements `data`; the others are stubs."""

from __future__ import annotations

import json

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


@data_app.command("build")
def build(
    step: str = typer.Argument(
        "all", help="all | games | counts | subset | reviews | news | duckdb"
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
        "duckdb": lambda: b.step_duckdb(s),
        "all": lambda: b.build_all(s, cfg),
    }
    if step not in steps:
        raise typer.BadParameter(f"unknown step {step}; choose from {list(steps)}")
    steps[step]()


@data_app.command("all")
def data_all(skip_news: bool = typer.Option(False, help="Skip the Steam news download")) -> None:
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
    b.step_reviews(s, cfg)
    b.step_news(s)
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


# ----------------------------------------------------------------------------- stubs
@index_app.command("build")
def index_build() -> None:
    """W2: not implemented yet."""
    console.print("[yellow]index build: coming in W2[/]")


@eval_app.command("all")
def eval_all() -> None:
    """W4/W5: not implemented yet."""
    console.print("[yellow]eval: coming in W4[/]")


if __name__ == "__main__":
    app()
