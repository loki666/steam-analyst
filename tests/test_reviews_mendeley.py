from datetime import datetime

import polars as pl

from steam_game_analyst.config import CleaningConfig
from steam_game_analyst.data import transform

# Shape of the Mendeley jxy85cr3th per-game CSVs (`<AppID>_<count>.csv`).
MENDELEY_CSV = (
    "user,playtime,post_date,helpfulness,review,recommend,early_access_review\n"
    'Alice,"1,234.5","November 30, 2021",20,Great puzzles and a lovely art style.,Recommended,\n'
    'Bob,0.1,"May 10, 2024",3,Crashes on launch every single time.,Not Recommended,'
    "Early Access Review\n"
)


def _build(tmp_path):
    p = tmp_path / "1710290_2.csv"
    p.write_text(MENDELEY_CSV)
    raw = transform.read_review_file(p, appid=1710290)
    return transform.build_reviews(raw, CleaningConfig()).sort("created_at")


def test_mendeley_recommend_maps_to_voted_up(tmp_path):
    assert _build(tmp_path)["voted_up"].to_list() == [True, False]


def test_mendeley_post_date_parses_to_created_at(tmp_path):
    assert _build(tmp_path)["created_at"].to_list() == [
        datetime(2021, 11, 30),
        datetime(2024, 5, 10),
    ]


def test_mendeley_helpfulness_playtime_and_early_access(tmp_path):
    df = _build(tmp_path)
    assert df["votes_up"].to_list() == [20, 3]
    assert df["playtime_forever_min"].to_list() == [74070, 6]
    assert df["written_during_early_access"].to_list() == [False, True]


def test_mendeley_author_name_is_not_kept(tmp_path):
    df = _build(tmp_path)
    assert "user" not in df.columns
    flat = df.select(pl.concat_str(pl.all().cast(pl.Utf8), separator="|")).to_series()
    assert not flat.str.contains("Alice|Bob").any()


def test_mendeley_reviews_have_no_language_until_detected(tmp_path):
    df = _build(tmp_path)
    assert df["language"].null_count() == df.height

    filled = transform.fill_languages(df)
    assert filled["language"].to_list() == ["english", "english"]


def test_sample_reviews_without_language_filter_keeps_all_languages():
    from steam_game_analyst.data import subset

    df = pl.DataFrame(
        {
            "appid": [1, 1, 1],
            "language": ["english", "chinese", None],
            "voted_up": [True, False, True],
            "created_at": [datetime(2024, 1, 1)] * 3,
        }
    )
    assert subset.sample_reviews(df, per_game=10, languages=[]).height == 3
    assert subset.sample_reviews(df, per_game=10, languages=["english"]).height == 1
