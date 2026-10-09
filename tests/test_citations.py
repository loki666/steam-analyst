"""Citation ids in an answer: "[3]", "[T2]", grouped "[3, 2]", adjacent "[T1][2]"."""

from steam_game_analyst.citations import cited_ids


def test_ids_in_reading_order_each_once():
    assert cited_ids("A [T2] b [3, 2] c [T1][2] d [3]") == ["T2", "3", "2", "T1"]


def test_text_without_citations():
    assert cited_ids("") == [] and cited_ids("price [cheap] and [a, b]") == []
