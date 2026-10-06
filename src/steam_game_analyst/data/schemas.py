"""Canonical column schemas for the processed tables. Every table joins on `appid`."""

from __future__ import annotations

import polars as pl

GAMES_SCHEMA: dict[str, pl.DataType] = {
    "appid": pl.Int64,
    "name": pl.Utf8,
    "release_date": pl.Date,
    "price_usd": pl.Float64,
    "required_age": pl.Int32,
    "dlc_count": pl.Int32,
    "short_description": pl.Utf8,
    "about_the_game": pl.Utf8,  # HTML stripped
    "developers": pl.List(pl.Utf8),
    "publishers": pl.List(pl.Utf8),
    "genres": pl.List(pl.Utf8),
    "categories": pl.List(pl.Utf8),
    "tags": pl.List(pl.Utf8),  # tag names, ordered by vote count
    "supported_languages": pl.List(pl.Utf8),
    "windows": pl.Boolean,
    "mac": pl.Boolean,
    "linux": pl.Boolean,
    "metacritic_score": pl.Int32,  # 0 -> null
    "positive": pl.Int64,
    "negative": pl.Int64,
    "positive_ratio": pl.Float64,
    "estimated_owners_low": pl.Int64,
    "estimated_owners_high": pl.Int64,
    "peak_ccu": pl.Int64,
    "average_playtime_forever": pl.Int64,  # minutes
    "median_playtime_forever": pl.Int64,
    "recommendations": pl.Int64,
    "achievements": pl.Int32,
}

REVIEWS_SCHEMA: dict[str, pl.DataType] = {
    "review_id": pl.Utf8,  # stable hash of (appid, text, created) — no Steam ids kept
    "appid": pl.Int64,
    "language": pl.Utf8,
    "review_text": pl.Utf8,
    "voted_up": pl.Boolean,
    "votes_up": pl.Int64,
    "votes_funny": pl.Int64,
    "weighted_vote_score": pl.Float64,
    "playtime_at_review_min": pl.Int64,
    "playtime_forever_min": pl.Int64,
    "steam_purchase": pl.Boolean,
    "received_for_free": pl.Boolean,
    "written_during_early_access": pl.Boolean,
    "created_at": pl.Datetime,
    "n_chars": pl.Int32,
}

NEWS_SCHEMA: dict[str, pl.DataType] = {
    "news_id": pl.Utf8,
    "appid": pl.Int64,
    "title": pl.Utf8,
    "contents": pl.Utf8,  # BBCode/HTML stripped
    "author": pl.Utf8,
    "feed_name": pl.Utf8,
    "feed_label": pl.Utf8,
    "url": pl.Utf8,
    "published_at": pl.Datetime,
    "tags": pl.List(pl.Utf8),
    "is_patch_note": pl.Boolean,  # heuristic on title/tags
}

WIKI_SCHEMA: dict[str, pl.DataType] = {
    "page_id": pl.Utf8,
    "appid": pl.Int64,
    "wiki": pl.Utf8,  # e.g. "eldenring.fandom.com"
    "title": pl.Utf8,
    "text": pl.Utf8,  # wikitext cleaned to plain text with "## section" markers
    "categories": pl.List(pl.Utf8),
    "n_chars": pl.Int32,
}
