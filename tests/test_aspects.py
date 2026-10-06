"""Keyword aspect tagging (weak labels until the W4 DistilBERT model)."""

from steam_game_analyst.ml.aspects import ASPECTS, aspect_sentences, aspects_in


def test_aspect_names_match_the_prd():
    assert ASPECTS == ("performance", "bugs", "gameplay", "story", "price", "controls")


def test_aspects_in_finds_each_aspect_case_insensitively():
    assert aspects_in("Terrible FPS and constant stutter") == {"performance"}
    assert aspects_in("It crashes to desktop, so many glitches") == {"bugs"}
    assert aspects_in("The combat loop is great") == {"gameplay"}
    assert aspects_in("Loved the story and the characters") == {"story"}
    assert aspects_in("Not worth the full price") == {"price"}
    assert aspects_in("Keyboard controls are clunky") == {"controls"}


def test_aspects_in_uses_word_boundaries():
    # "lag" inside "flag", "bug" inside "debug" menu name, "price" inside "priceless"
    assert aspects_in("Capture the flag mode") == set()
    assert aspects_in("A priceless experience") == set()


def test_aspect_sentences_keeps_only_the_sentences_that_mention_the_aspect():
    text = "The story is wonderful. But the game runs at 20 fps on my PC! Price is fair."
    s = aspect_sentences(text)
    assert s["performance"] == ["But the game runs at 20 fps on my PC!"]
    assert s["story"] == ["The story is wonderful."]
    assert s["price"] == ["Price is fair."]
    assert "bugs" not in s
