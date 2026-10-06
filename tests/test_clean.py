from datetime import date

from steam_game_analyst.data import clean


def test_strip_html_keeps_line_breaks_and_decodes_entities():
    out = clean.strip_html("<p>Hello <b>world</b> &amp; friends</p><br><li>item</li>")
    assert "Hello world & friends" in out
    assert "item" in out
    assert "<" not in out


def test_strip_bbcode_bullets_and_images():
    out = clean.strip_bbcode(
        "[h1]Patch[/h1][list][*]Fixed stutter[*]Nerfed sword[/list]{STEAM_CLAN_IMAGE}/x.png"
    )
    assert out.startswith("Patch")
    assert "- Fixed stutter" in out
    assert "STEAM_CLAN_IMAGE" not in out
    assert "[" not in out


def test_parse_owners():
    assert clean.parse_owners("20000 - 50000") == (20000, 50000)
    assert clean.parse_owners("1,000,000 - 2,000,000") == (1_000_000, 2_000_000)
    assert clean.parse_owners("N/A") == (None, None)
    assert clean.parse_owners(None) == (None, None)


def test_parse_release_date_formats():
    assert clean.parse_release_date("Feb 25, 2022") == date(2022, 2, 25)
    assert clean.parse_release_date("2022-02-25") == date(2022, 2, 25)
    assert clean.parse_release_date("TBA") is None
    assert clean.parse_release_date("garbage") is None


def test_split_list_and_tags():
    assert clean.split_list("Action,RPG") == ["Action", "RPG"]
    assert clean.split_list("['English', 'French']") == ["English", "French"]
    assert clean.split_list(None) == []
    assert clean.tags_from_field('{"A": 10, "B": 30, "C": 20}') == ["B", "C", "A"]
    assert clean.tags_from_field("Shooter,Indie") == ["Shooter", "Indie"]


def test_stable_id_is_deterministic_and_short():
    a = clean.stable_id(1, "text", None)
    b = clean.stable_id(1, "text", None)
    assert a == b and len(a) == 16
    assert clean.stable_id(2, "text", None) != a


def test_is_patch_note():
    assert clean.is_patch_note("Patch 1.5 Notes")
    assert clean.is_patch_note("Hotfix for crashes")
    assert clean.is_patch_note("Summer Sale!", ["patchnotes"])
    assert not clean.is_patch_note("Summer Sale!")


def test_split_companies_keeps_corporate_suffixes_together():
    # FronkonGames separates companies with a bare comma; ", " belongs to a name.
    assert clean.split_companies("FromSoftware, Inc.,Bandai Namco Entertainment") == [
        "FromSoftware, Inc.",
        "Bandai Namco Entertainment",
    ]
    assert clean.split_companies("CAPCOM Co., Ltd.") == ["CAPCOM Co., Ltd."]
    assert clean.split_companies("Treyarch,Raven Software") == ["Treyarch", "Raven Software"]
    assert clean.split_companies("CyberStep, Inc.,Rideon Works Co. Ltd,") == [
        "CyberStep, Inc.",
        "Rideon Works Co. Ltd",
    ]
    # suffix written without the space is re-attached (and normalised)
    assert clean.split_companies("Kairosoft Co.,Ltd,Chiptune,Inc.") == [
        "Kairosoft Co., Ltd",
        "Chiptune, Inc.",
    ]
    assert clean.split_companies("2202GAMES,LLC") == ["2202GAMES, LLC"]
    assert clean.split_companies(None) == []
    assert clean.split_companies(["Larian Studios"]) == ["Larian Studios"]


def test_detect_languages_labels_and_short_texts():
    out = clean.detect_languages(
        [
            "Great game, love the combat and the music!",
            "bug是真的多 能不能修复一下骷髅头武器丢到地图外面",
            "Не понимаю откуда такое количество положительных отзывов",
            "gg",
            "10/10",
            None,
        ]
    )
    assert out == ["english", "chinese", "russian", "unknown", "unknown", "unknown"]
