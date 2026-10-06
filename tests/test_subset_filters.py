import json
from datetime import date

import polars as pl

from steam_game_analyst.config import SubsetConfig
from steam_game_analyst.data import subset, transform


def _games(rows):
    return pl.DataFrame(
        rows,
        schema={"appid": pl.Int64, "release_date": pl.Date, "genres": pl.List(pl.Utf8)},
        orient="row",
    )


def test_select_games_skips_software_and_genreless_apps():
    games = _games(
        [
            (1, date(2022, 1, 1), ["Action", "RPG"]),
            (2, date(2022, 1, 1), ["Utilities"]),
            (3, date(2022, 1, 1), ["Casual", "Animation & Modeling"]),
            (4, date(2022, 1, 1), []),
            (5, date(2022, 1, 1), None),
            (6, date(2022, 1, 1), ["Strategy"]),
        ]
    )
    counts = pl.DataFrame({"appid": [1, 2, 3, 4, 5, 6], "n_reviews": [900] * 6})
    cfg = SubsetConfig(
        n_games=10, min_english_reviews=500, exclude_genres=["Utilities", "Animation & Modeling"]
    )

    chosen = subset.select_games(games, counts, cfg)

    assert sorted(chosen["appid"].to_list()) == [1, 6]


def test_select_games_keeps_must_include_even_if_excluded_genre():
    games = _games([(2, date(2022, 1, 1), ["Utilities"]), (6, date(2022, 1, 1), ["Strategy"])])
    counts = pl.DataFrame({"appid": [2, 6], "n_reviews": [900, 900]})
    cfg = SubsetConfig(n_games=10, must_include=[2], exclude_genres=["Utilities"])

    chosen = subset.select_games(games, counts, cfg)

    assert sorted(chosen["appid"].to_list()) == [2, 6]


def _news_file(path, appid):
    item = {
        "gid": f"g{appid}",
        "title": "Patch 1.0",
        "url": "https://example.com",
        "author": "dev",
        "contents": "Fixed bugs",
        "feedlabel": "Community Announcements",
        "date": 1700000000,
        "feedname": "steam_community_announcements",
        "tags": ["patchnotes"],
    }
    path.write_text(json.dumps({"appnews": {"appid": appid, "newsitems": [item]}}))


def test_build_news_keeps_only_requested_appids(tmp_path):
    for appid in (10, 20, 30):
        _news_file(tmp_path / f"{appid}.json", appid)

    news = transform.build_news(tmp_path, appids=[10, 30])

    assert sorted(news["appid"].unique().to_list()) == [10, 30]


def test_build_news_without_appids_reads_everything(tmp_path):
    for appid in (10, 20):
        _news_file(tmp_path / f"{appid}.json", appid)

    assert transform.build_news(tmp_path)["appid"].n_unique() == 2
