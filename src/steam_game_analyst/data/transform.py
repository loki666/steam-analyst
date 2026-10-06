"""Raw -> canonical tables (polars). Column names in the raw sources vary between
versions, so each loader maps through an alias table and fails loudly on missing essentials."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from steam_game_analyst.config import CleaningConfig
from steam_game_analyst.data import clean
from steam_game_analyst.data.schemas import GAMES_SCHEMA, NEWS_SCHEMA, REVIEWS_SCHEMA

# ----------------------------------------------------------------------------- helpers
_CATALOG_ALIASES: dict[str, list[str]] = {
    "appid": ["AppID", "appid", "app_id", "steam_appid"],
    "name": ["Name", "name"],
    "release_date": ["Release date", "release_date"],
    "price_usd": ["Price", "price"],
    "required_age": ["Required age", "required_age"],
    "dlc_count": ["DLC count", "dlc_count"],
    "short_description": ["Short description", "short_description"],
    "about_the_game": [
        "About the game",
        "about_the_game",
        "Detailed description",
        "detailed_description",
    ],
    "developers": ["Developers", "developers"],
    "publishers": ["Publishers", "publishers"],
    "genres": ["Genres", "genres"],
    "categories": ["Categories", "categories"],
    "tags": ["Tags", "tags"],
    "supported_languages": ["Supported languages", "supported_languages"],
    "windows": ["Windows", "windows"],
    "mac": ["Mac", "mac"],
    "linux": ["Linux", "linux"],
    "metacritic_score": ["Metacritic score", "metacritic_score"],
    "positive": ["Positive", "positive"],
    "negative": ["Negative", "negative"],
    "estimated_owners": ["Estimated owners", "estimated_owners"],
    "peak_ccu": ["Peak CCU", "peak_ccu"],
    "average_playtime_forever": ["Average playtime forever", "average_playtime_forever"],
    "median_playtime_forever": ["Median playtime forever", "median_playtime_forever"],
    "recommendations": ["Recommendations", "recommendations"],
    "achievements": ["Achievements", "achievements"],
}

_REVIEW_ALIASES: dict[str, list[str]] = {
    "appid": ["appid", "app_id", "AppID", "game_id"],
    "language": ["language", "lang"],
    "review_text": ["review", "review_text", "text", "content"],
    "voted_up": ["voted_up", "recommended", "voted up", "rating", "recommend"],
    "votes_up": ["votes_up", "helpful", "votes up", "helpfulness"],
    "votes_funny": ["votes_funny", "funny"],
    "weighted_vote_score": ["weighted_vote_score"],
    "playtime_at_review_min": [
        "author.playtime_at_review",
        "playtime_at_review",
        "author_playtime_at_review",
    ],
    "playtime_forever_min": [
        "author.playtime_forever",
        "playtime_forever",
        "author_playtime_forever",
    ],
    "steam_purchase": ["steam_purchase"],
    "received_for_free": ["received_for_free"],
    "written_during_early_access": ["written_during_early_access", "early_access_review"],
    "created_at": ["timestamp_created", "created", "date", "timestamp", "post_date"],
    # Mendeley: hours on record when scraped (not at review time)
    "playtime_hours": ["playtime"],
}

# Tried in order; format inference fails on a whole column, so formats are explicit.
_REVIEW_DATE_FORMATS = (
    "%B %d, %Y",  # Mendeley: "May 10, 2024"
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


def _resolve(df: pl.DataFrame, aliases: dict[str, list[str]], required: set[str]) -> dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    out: dict[str, str] = {}
    for canon, names in aliases.items():
        for n in names:
            if n.lower() in cols:
                out[canon] = cols[n.lower()]
                break
    missing = required - out.keys()
    if missing:
        raise ValueError(f"Missing required columns {missing}; available: {df.columns}")
    return out


def _to_bool(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(
            expr.cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["true", "1", "yes", "recommended", "positive", "t", "early access review"])
        )
        .then(True)
        .when(
            expr.cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["false", "0", "no", "not recommended", "negative", "f"])
        )
        .then(False)
        .otherwise(None)
    )


# ----------------------------------------------------------------------------- catalog
def read_catalog_raw(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".json":
        # FronkonGames games.json is {appid: {...}}
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [{"AppID": k, **v} for k, v in data.items()]
        return pl.DataFrame(rows, infer_schema_length=None)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first_row = next(reader, None)
    if "DiscountDLC count" in header:
        # FronkonGames ships "Discount" and "DLC count" merged in the header; without this
        # every column after Price is read one to the left (Genres holds Categories, ...).
        i = header.index("DiscountDLC count")
        fixed = [*header[:i], "Discount", "DLC count", *header[i + 1 :]]
        if first_row is not None and len(first_row) != len(fixed):
            raise ValueError(
                f"{path}: repaired header has {len(fixed)} columns but rows have "
                f"{len(first_row)}; the upstream CSV layout changed"
            )
        return pl.read_csv(
            path,
            has_header=False,
            skip_rows=1,
            new_columns=fixed,
            infer_schema_length=10_000,
            ignore_errors=True,
        )
    return pl.read_csv(
        path, infer_schema_length=10_000, ignore_errors=True, truncate_ragged_lines=True
    )


def build_games(raw: pl.DataFrame) -> pl.DataFrame:
    """Canonical `games` table from the FronkonGames catalog."""
    m = _resolve(raw, _CATALOG_ALIASES, required={"appid", "name"})

    def col(name: str, default: object = None) -> pl.Expr:
        return pl.col(m[name]) if name in m else pl.lit(default)

    owners = (
        raw[m["estimated_owners"]].cast(pl.Utf8).to_list()
        if "estimated_owners" in m
        else [None] * raw.height
    )
    lo_hi = [clean.parse_owners(o) for o in owners]

    df = raw.select(
        col("appid").cast(pl.Int64, strict=False).alias("appid"),
        col("name").cast(pl.Utf8).alias("name"),
        col("release_date").cast(pl.Utf8).alias("_release"),
        col("price_usd").cast(pl.Float64, strict=False).alias("price_usd"),
        col("required_age", 0).cast(pl.Int32, strict=False).alias("required_age"),
        col("dlc_count", 0).cast(pl.Int32, strict=False).alias("dlc_count"),
        col("short_description", "").cast(pl.Utf8).alias("_short"),
        col("about_the_game", "").cast(pl.Utf8).alias("_about"),
        col("developers", "").cast(pl.Utf8).alias("_dev"),
        col("publishers", "").cast(pl.Utf8).alias("_pub"),
        col("genres", "").cast(pl.Utf8).alias("_genres"),
        col("categories", "").cast(pl.Utf8).alias("_cats"),
        col("tags", "").cast(pl.Utf8).alias("_tags"),
        col("supported_languages", "").cast(pl.Utf8).alias("_langs"),
        _to_bool(col("windows", True)).alias("windows"),
        _to_bool(col("mac", False)).alias("mac"),
        _to_bool(col("linux", False)).alias("linux"),
        col("metacritic_score", 0).cast(pl.Int32, strict=False).alias("_meta"),
        col("positive", 0).cast(pl.Int64, strict=False).alias("positive"),
        col("negative", 0).cast(pl.Int64, strict=False).alias("negative"),
        col("peak_ccu", 0).cast(pl.Int64, strict=False).alias("peak_ccu"),
        col("average_playtime_forever", 0)
        .cast(pl.Int64, strict=False)
        .alias("average_playtime_forever"),
        col("median_playtime_forever", 0)
        .cast(pl.Int64, strict=False)
        .alias("median_playtime_forever"),
        col("recommendations", 0).cast(pl.Int64, strict=False).alias("recommendations"),
        col("achievements", 0).cast(pl.Int32, strict=False).alias("achievements"),
    ).with_columns(
        pl.Series("estimated_owners_low", [x[0] for x in lo_hi], dtype=pl.Int64),
        pl.Series("estimated_owners_high", [x[1] for x in lo_hi], dtype=pl.Int64),
    )

    df = df.with_columns(
        pl.col("_release")
        .map_elements(clean.parse_release_date, return_dtype=pl.Date)
        .alias("release_date"),
        pl.col("_short")
        .map_elements(clean.strip_html, return_dtype=pl.Utf8)
        .alias("short_description"),
        pl.col("_about")
        .map_elements(clean.strip_html, return_dtype=pl.Utf8)
        .alias("about_the_game"),
        pl.col("_dev")
        .map_elements(clean.split_list, return_dtype=pl.List(pl.Utf8))
        .alias("developers"),
        pl.col("_pub")
        .map_elements(clean.split_list, return_dtype=pl.List(pl.Utf8))
        .alias("publishers"),
        pl.col("_genres")
        .map_elements(clean.split_list, return_dtype=pl.List(pl.Utf8))
        .alias("genres"),
        pl.col("_cats")
        .map_elements(clean.split_list, return_dtype=pl.List(pl.Utf8))
        .alias("categories"),
        pl.col("_tags")
        .map_elements(clean.tags_from_field, return_dtype=pl.List(pl.Utf8))
        .alias("tags"),
        pl.col("_langs")
        .map_elements(clean.split_list, return_dtype=pl.List(pl.Utf8))
        .alias("supported_languages"),
        pl.when(pl.col("_meta") > 0)
        .then(pl.col("_meta"))
        .otherwise(None)
        .alias("metacritic_score"),
        (pl.col("positive") / (pl.col("positive") + pl.col("negative")).clip(lower_bound=1)).alias(
            "positive_ratio"
        ),
    )
    df = df.drop([c for c in df.columns if c.startswith("_")])
    df = df.filter(pl.col("appid").is_not_null() & pl.col("name").is_not_null()).unique(
        subset=["appid"], keep="first"
    )
    return df.select(list(GAMES_SCHEMA.keys())).cast(GAMES_SCHEMA, strict=False)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- reviews
def read_review_file(path: Path, appid: int | None = None) -> pl.DataFrame:
    """One Mendeley per-game CSV (or any appreviews-shaped CSV/JSONL)."""
    if path.suffix == ".jsonl":
        df = pl.read_ndjson(path)
    else:
        df = pl.read_csv(
            path,
            infer_schema_length=5_000,
            ignore_errors=True,
            truncate_ragged_lines=True,
            quote_char='"',
        )
    if appid is not None and not any(c.lower() in {"appid", "app_id"} for c in df.columns):
        df = df.with_columns(pl.lit(appid).alias("appid"))
    return df


def appid_from_filename(path: Path) -> int | None:
    """Mendeley names files `<AppID>_<reviewcount>.csv`."""
    stem = path.stem.split("_")[0]
    return int(stem) if stem.isdigit() else None


def build_reviews(raw: pl.DataFrame, cfg: CleaningConfig) -> pl.DataFrame:
    """Canonical `reviews` table. Drops author ids, dedupes, filters by length."""
    m = _resolve(raw, _REVIEW_ALIASES, required={"appid", "review_text"})

    def col(name: str, default: object = None) -> pl.Expr:
        return pl.col(m[name]) if name in m else pl.lit(default)

    created = col("created_at")
    created_expr = (
        (
            pl.when(created.cast(pl.Utf8).str.contains(r"^\d{9,11}$"))
            .then(
                pl.from_epoch(
                    created.cast(pl.Utf8).str.extract(r"^(\d{9,11})$").cast(pl.Int64, strict=False),
                    time_unit="s",
                )
            )
            .otherwise(
                pl.coalesce(
                    created.cast(pl.Utf8).str.strptime(pl.Datetime("us"), fmt, strict=False)
                    for fmt in _REVIEW_DATE_FORMATS
                )
            )
            .cast(pl.Datetime("us"))
        )
        if "created_at" in m
        else pl.lit(None, dtype=pl.Datetime("us"))
    )

    if "playtime_forever_min" in m:
        playtime_forever = col("playtime_forever_min").cast(pl.Int64, strict=False)
    else:
        hours = (
            col("playtime_hours", None)
            .cast(pl.Utf8)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
        )
        playtime_forever = (hours * 60).round().cast(pl.Int64)
    early_access = _to_bool(col("written_during_early_access", None))
    if m.get("written_during_early_access") == "early_access_review":
        # Mendeley marks early-access reviews with a label and leaves the rest empty
        early_access = early_access.fill_null(False)

    df = raw.select(
        col("appid").cast(pl.Int64, strict=False).alias("appid"),
        col("language", "english").cast(pl.Utf8).str.to_lowercase().alias("language"),
        col("review_text").cast(pl.Utf8).alias("review_text"),
        _to_bool(col("voted_up")).alias("voted_up"),
        col("votes_up", 0).cast(pl.Int64, strict=False).alias("votes_up"),
        col("votes_funny", 0).cast(pl.Int64, strict=False).alias("votes_funny"),
        col("weighted_vote_score", None)
        .cast(pl.Float64, strict=False)
        .alias("weighted_vote_score"),
        col("playtime_at_review_min", None)
        .cast(pl.Int64, strict=False)
        .alias("playtime_at_review_min"),
        playtime_forever.alias("playtime_forever_min"),
        _to_bool(col("steam_purchase", None)).alias("steam_purchase"),
        _to_bool(col("received_for_free", None)).alias("received_for_free"),
        early_access.alias("written_during_early_access"),
        created_expr.alias("created_at"),
    )
    if cfg.strip_html:
        df = df.with_columns(
            pl.col("review_text").map_elements(clean.strip_html, return_dtype=pl.Utf8)
        )
    else:
        df = df.with_columns(
            pl.col("review_text").map_elements(clean.normalise_ws, return_dtype=pl.Utf8)
        )
    df = (
        df.with_columns(pl.col("review_text").str.len_chars().cast(pl.Int32).alias("n_chars"))
        .filter(pl.col("appid").is_not_null())
        .filter(pl.col("n_chars").is_between(cfg.min_review_chars, cfg.max_review_chars))
        .unique(subset=cfg.drop_duplicates_on, keep="first")
        .with_columns(
            pl.struct(["appid", "review_text", "created_at"])
            .map_elements(
                lambda s: clean.stable_id(s["appid"], s["review_text"], s["created_at"]),
                return_dtype=pl.Utf8,
            )
            .alias("review_id")
        )
    )
    return df.select(list(REVIEWS_SCHEMA.keys())).cast(REVIEWS_SCHEMA, strict=False)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- news
def build_news(news_dir: Path, appids: Iterable[int] | None = None) -> pl.DataFrame:
    """data/raw/news/<appid>.json -> canonical `news` table, limited to `appids` if given
    (the raw folder keeps files from earlier subsets)."""
    keep = {str(a) for a in appids} if appids is not None else None
    rows: list[dict] = []
    for p in sorted(news_dir.glob("*.json")):
        if keep is not None and p.stem not in keep:
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        appnews = payload.get("appnews", {})
        appid = int(appnews.get("appid") or p.stem)
        for item in appnews.get("newsitems", []):
            tags = item.get("tags") or []
            title = item.get("title") or ""
            contents = clean.strip_bbcode(item.get("contents"))
            rows.append(
                {
                    "news_id": clean.stable_id(appid, item.get("gid") or title, item.get("date")),
                    "appid": appid,
                    "title": title,
                    "contents": contents,
                    "author": item.get("author") or "",
                    "feed_name": item.get("feedname") or "",
                    "feed_label": item.get("feedlabel") or "",
                    "url": item.get("url") or "",
                    "published_at": item.get("date"),
                    "tags": [str(t) for t in tags],
                    "is_patch_note": clean.is_patch_note(title, tags),
                }
            )
    if not rows:
        return pl.DataFrame(schema=NEWS_SCHEMA)  # type: ignore[arg-type]
    df = pl.DataFrame(rows).with_columns(
        (pl.col("published_at").cast(pl.Int64) * 1_000_000)
        .cast(pl.Datetime("us"))
        .alias("published_at")
    )
    return df.select(list(NEWS_SCHEMA.keys())).cast(NEWS_SCHEMA, strict=False)  # type: ignore[arg-type]
