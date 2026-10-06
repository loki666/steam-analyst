"""Shared fixtures: a tiny synthetic raw dataset shaped like the real sources."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

CATALOG_ROWS = [
    # appid, name, release, price, genres, tags(json), positive, negative, owners, playtime
    (
        1245620,
        "ELDEN RING",
        "Feb 25, 2022",
        59.99,
        "Action,RPG",
        '{"Souls-like": 9000, "Open World": 8000, "Dark Fantasy": 7000}',
        500000,
        40000,
        "20000000 - 50000000",
        5000,
    ),
    (
        1091500,
        "Cyberpunk 2077",
        "Dec 10, 2020",
        59.99,
        "RPG",
        '{"Cyberpunk": 9000, "Open World": 8500, "Story Rich": 7000}',
        400000,
        90000,
        "10000000 - 20000000",
        4000,
    ),
    (
        892970,
        "Valheim",
        "Feb 2, 2021",
        19.99,
        "Adventure,Indie,RPG,Early Access",
        '{"Survival": 9000, "Co-op": 8000, "Crafting": 7000}',
        350000,
        15000,
        "10000000 - 20000000",
        3000,
    ),
    (
        1145360,
        "Hades",
        "Sep 17, 2020",
        24.99,
        "Action,Indie,RPG",
        '{"Rogue-like": 9000, "Action Roguelike": 8000}',
        200000,
        4000,
        "5000000 - 10000000",
        1500,
    ),
    (
        999001,
        "Tiny Shooter",
        "Mar 3, 2023",
        4.99,
        "Action",
        "Shooter,Indie",
        50,
        60,
        "0 - 20000",
        100,
    ),
    (999002, "Unreleased Thing", "TBA", 0.0, "", "", 0, 0, "0 - 0", 0),
]


def write_catalog(path: Path) -> Path:
    """FronkonGames-style CSV with the original human-readable headers and HTML in text fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "AppID",
                "Name",
                "Release date",
                "Price",
                "Genres",
                "Tags",
                "Positive",
                "Negative",
                "Estimated owners",
                "Average playtime forever",
                "About the game",
                "Short description",
                "Developers",
                "Publishers",
                "Metacritic score",
                "Windows",
                "Mac",
                "Linux",
                "Categories",
                "Supported languages",
                "Peak CCU",
                "Median playtime forever",
                "Recommendations",
                "Achievements",
                "Required age",
                "DLC count",
            ]
        )
        for appid, name, rel, price, genres, tags, pos, neg, owners, play in CATALOG_ROWS:
            w.writerow(
                [
                    appid,
                    name,
                    rel,
                    price,
                    genres,
                    tags,
                    pos,
                    neg,
                    owners,
                    play,
                    f"<p>{name} is a <b>great</b> game.<br>Second line.</p>",
                    f"{name} short.",
                    "Dev Studio",
                    "Pub Co",
                    90 if pos > 100000 else 0,
                    "True",
                    "False",
                    "False",
                    "Single-player,Multi-player",
                    "['English', 'French']",
                    1000,
                    play // 2,
                    pos // 10,
                    40,
                    0,
                    2,
                ]
            )
    return path


def write_reviews(dir_: Path, appids: list[int], n: int = 120, seed: int = 0) -> list[Path]:
    """Mendeley-style per-game CSVs named <appid>_<count>.csv with appreviews-like columns."""
    rng = random.Random(seed)
    dir_.mkdir(parents=True, exist_ok=True)
    paths = []
    phrases_pos = [
        "Runs great on Steam Deck.",
        "Combat is superb and the world is huge.",
        "Best crafting system I have played.",
        "Patch 1.5 fixed the stutter, finally.",
    ]
    phrases_neg = [
        "Crashes every hour, terrible performance.",
        "Too many bugs at launch.",
        "Not worth the price.",
        "Controls feel awful with a gamepad.",
    ]
    base = datetime(2021, 1, 1)
    for appid in appids:
        p = dir_ / f"{appid}_{n}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "recommendationid",
                    "author.steamid",
                    "language",
                    "review",
                    "timestamp_created",
                    "voted_up",
                    "votes_up",
                    "votes_funny",
                    "weighted_vote_score",
                    "author.playtime_at_review",
                    "author.playtime_forever",
                    "steam_purchase",
                    "received_for_free",
                    "written_during_early_access",
                ]
            )
            for i in range(n):
                up = rng.random() < 0.7
                lang = (
                    "english"
                    if rng.random() < 0.85
                    else rng.choice(["russian", "schinese", "french"])
                )
                text = rng.choice(phrases_pos if up else phrases_neg) + f" ({i})"
                if i % 40 == 0:
                    text = "ok"  # too short -> filtered
                if i % 50 == 1:
                    text = "<b>bold</b> review with html &amp; entities"
                ts = int((base + timedelta(days=rng.randint(0, 1200))).timestamp())
                w.writerow(
                    [
                        100000 + i,
                        7656119800000000 + i,
                        lang,
                        text,
                        ts,
                        "True" if up else "False",
                        rng.randint(0, 50),
                        rng.randint(0, 5),
                        round(rng.random(), 3),
                        rng.randint(10, 20000),
                        rng.randint(10, 30000),
                        "True",
                        "False",
                        "False",
                    ]
                )
            # exact duplicate row
            w.writerow(
                [
                    999999,
                    1,
                    "english",
                    "Combat is superb and the world is huge. (3)",
                    ts,
                    "True",
                    0,
                    0,
                    0.5,
                    100,
                    100,
                    "True",
                    "False",
                    "False",
                ]
            )
        paths.append(p)
    return paths


def write_news(dir_: Path, appids: list[int]) -> list[Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    paths = []
    for appid in appids:
        items = [
            {
                "gid": f"{appid}-1",
                "title": "Patch 1.5 Notes",
                "url": "https://store.steampowered.com/news/app/x",
                "author": "dev",
                "contents": "[h1]Patch 1.5[/h1][list][*]Fixed stutter[*]Balance: nerfed the sword[/list]{STEAM_CLAN_IMAGE}/abc.png",
                "feedlabel": "Community Announcements",
                "date": 1700000000,
                "feedname": "steam_community_announcements",
                "tags": ["patchnotes"],
            },
            {
                "gid": f"{appid}-2",
                "title": "Summer Sale!",
                "url": "",
                "author": "",
                "contents": "<p>50% off this week</p>",
                "feedlabel": "Community Announcements",
                "date": 1690000000,
                "feedname": "steam_community_announcements",
            },
        ]
        p = dir_ / f"{appid}.json"
        p.write_text(json.dumps({"appnews": {"appid": appid, "newsitems": items, "count": 2}}))
        paths.append(p)
    return paths


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """A complete fake data/raw tree."""
    raw = tmp_path / "data" / "raw"
    write_catalog(raw / "catalog" / "games.csv")
    write_reviews(raw / "reviews", [1245620, 1091500, 892970, 1145360, 999001])
    write_news(raw / "news", [1245620, 1091500])
    return raw


@pytest.fixture
def settings(tmp_path: Path, raw_dataset: Path, monkeypatch: pytest.MonkeyPatch):
    from steam_game_analyst.config import Settings, get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SGA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SGA_SUBSET_GAMES", "4")
    monkeypatch.setenv("SGA_REVIEWS_PER_GAME", "30")
    s = Settings()
    (s.processed_dir).mkdir(parents=True, exist_ok=True)
    yield s
    get_settings.cache_clear()


@pytest.fixture
def built(settings):
    """The fixture dataset run through the whole W1 build (parquet + sga.duckdb)."""
    from steam_game_analyst.config import load_pipeline_config
    from steam_game_analyst.data import build

    build.build_all(settings, load_pipeline_config())
    return settings
