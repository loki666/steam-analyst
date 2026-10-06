"""Subset selection: which games go into the project, and which reviews per game."""

from __future__ import annotations

import polars as pl

from steam_game_analyst.config import SubsetConfig


def select_games(
    games: pl.DataFrame, review_counts: pl.DataFrame, cfg: SubsetConfig
) -> pl.DataFrame:
    """Pick `cfg.n_games` appids.

    Rules, in order:
      1. `must_include` appids are always in (if present in the catalog).
      2. Candidates: released within [release_from, release_to], with at least
         `min_reviews` reviews available (any language) (from `review_counts`: appid, n_reviews),
         at least one genre and none of `exclude_genres` (non-game software).
      3. Spread across genres: at most `genre_quota` games per primary genre (first genre),
         filling by descending review count so popular games come first within each genre.
    Returns the selected rows of `games` with an added `n_reviews` column.
    """
    cand = (
        games.join(review_counts, on="appid", how="inner")
        .filter(
            pl.col("release_date").is_between(
                pl.lit(cfg.release_from).str.to_date(), pl.lit(cfg.release_to).str.to_date()
            )
        )
        .filter(pl.col("n_reviews") >= cfg.min_reviews)
        .filter(pl.col("genres").list.len().fill_null(0) > 0)
        .filter(~pl.col("genres").list.eval(pl.element().is_in(cfg.exclude_genres)).list.any())
        .with_columns(pl.col("genres").list.first().fill_null("Unknown").alias("primary_genre"))
        .sort("n_reviews", descending=True)
    )
    must = games.join(review_counts, on="appid", how="left").filter(
        pl.col("appid").is_in(cfg.must_include)
    )
    must = must.with_columns(
        pl.col("genres").list.first().fill_null("Unknown").alias("primary_genre")
    )

    chosen = must.select(cand.columns)
    remaining = cfg.n_games - chosen.height
    if remaining > 0:
        rest = cand.filter(~pl.col("appid").is_in(chosen["appid"].implode()))
        # rank within genre, then take round-robin by rank so genres interleave
        rest = rest.with_columns(pl.col("appid").cum_count().over("primary_genre").alias("_rank"))
        rest = rest.filter(pl.col("_rank") <= cfg.genre_quota).sort(
            ["_rank", "n_reviews"], descending=[False, True]
        )
        chosen = pl.concat([chosen, rest.head(remaining).drop("_rank")], how="vertical_relaxed")
    return chosen


def sample_reviews(
    reviews: pl.DataFrame, per_game: int, languages: list[str], seed: int = 42
) -> pl.DataFrame:
    """Stratified cap per game: split the quota between positive and negative reviews, and
    within each spread across years, so the sample keeps the time dimension for trend questions."""
    df = (
        reviews.filter(pl.col("language").is_in([lang.lower() for lang in languages]))
        if languages
        else reviews
    )
    df = df.with_columns(
        pl.col("created_at").dt.year().fill_null(0).alias("_year"),
        pl.col("voted_up").fill_null(True).alias("_up"),
    )
    half = max(per_game // 2, 1)
    # per (appid, voted_up): take up to `half`, round-robin across years by shuffled rank
    df = df.with_columns(
        pl.int_range(pl.len()).shuffle(seed=seed).over(["appid", "_up", "_year"]).alias("_r_year"),
    )
    df = df.sort(["appid", "_up", "_r_year"]).with_columns(
        pl.int_range(pl.len()).over(["appid", "_up"]).alias("_r_strat")
    )
    out = df.filter(pl.col("_r_strat") < half)
    # if one polarity is short, top up from the other
    counts = out.group_by("appid").len().rename({"len": "_n"})
    short = counts.filter(pl.col("_n") < per_game)
    if short.height:
        extra = (
            df.join(short, on="appid", how="inner")
            .filter(pl.col("_r_strat") >= half)
            .with_columns(pl.int_range(pl.len()).over("appid").alias("_r_extra"))
            .filter(pl.col("_r_extra") < (per_game - pl.col("_n")))
            .drop(["_n", "_r_extra"])
        )
        out = pl.concat([out, extra], how="vertical_relaxed")
    return out.drop([c for c in out.columns if c.startswith("_")])
