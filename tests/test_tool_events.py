"""Patch notes whose titles match the question (synthetic games and titles only: the eval
questions are never used to build or check the matcher)."""

from datetime import date

from steam_game_analyst.tools.events import match_patch_titles

STARFORGE = [
    (
        date(2025, 3, 2),
        "Starforge | The Ember Tides update arrives March 20! Watch the trailer",
        True,
    ),
    (date(2025, 3, 20), "Ember Tides v1.4.0: new biome, ships and more", True),
    (date(2025, 1, 10), "Starforge 1.3 patch notes", True),
    (date(2025, 1, 12), "Starforge 1.3.1 hotfix", True),
    (date(2024, 12, 20), "Winter sale: 40% off", False),
]


def test_version_strings_match_exactly():
    hits = match_patch_titles(
        "Did Starforge reviews change after patch 1.3?", "Starforge", STARFORGE
    )
    assert [t for _, t in hits] == ["Starforge 1.3 patch notes"]


def test_a_named_update_matches_its_release_before_its_trailer():
    hits = match_patch_titles("How was the Ember Tides update received?", "Starforge", STARFORGE)
    assert hits[0][1] == "Ember Tides v1.4.0: new biome, ships and more"


def test_month_and_year_in_the_question_are_ignored():
    hits = match_patch_titles(
        "Did Starforge change after patch 1.3 in January 2025?", "Starforge", STARFORGE
    )
    assert [d for d, _ in hits] == [date(2025, 1, 10)]


def test_bare_numbers_only_count_after_a_version_word():
    assert match_patch_titles("Starforge over the last 12 months?", "Starforge", STARFORGE) == []
    titles = [(date(2025, 5, 1), "Update 12 is live", True)]
    assert match_patch_titles("Reviews after update 12?", "Starforge", titles)


def test_the_game_name_and_common_words_do_not_match():
    assert (
        match_patch_titles("Is Starforge getting better after the update?", "Starforge", STARFORGE)
        == []
    )


def test_numbered_patch_with_a_hash():
    titles = [(date(2025, 6, 1), "Patch #9 - balance", True), (date(2025, 7, 1), "Patch #19", True)]
    assert match_patch_titles("after patch #9?", "Starforge", titles) == [titles[0][:2]]


def test_at_most_k_matches_earliest_first_on_ties():
    titles = [(date(2025, 1, d), f"Ember Tides fix {d}", True) for d in range(8, 0, -1)]
    hits = match_patch_titles("the Ember Tides fixes", "Starforge", titles, k=3)
    assert [d.day for d, _ in hits] == [1, 2, 3]
