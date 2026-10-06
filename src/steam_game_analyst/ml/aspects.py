"""Keyword aspect tagging for English reviews: weak labels for the six PRD aspects.

The lexicon is generic game-review vocabulary, written once and not tuned on any question or
eval item. It favours precision (word boundaries, few ambiguous words); the hand-labelled
macro-F1 check and the DistilBERT aspect model come in W4."""

from __future__ import annotations

import re

ASPECTS: tuple[str, ...] = ("performance", "bugs", "gameplay", "story", "price", "controls")

_LEXICON: dict[str, list[str]] = {
    "performance": [
        r"performance",
        r"fps",
        r"frame ?rates?",
        r"frames? per second",
        r"frame ?drops?",
        r"stutter\w*",
        r"lag\w*",
        r"optimi[sz]\w*",
        r"unoptimi[sz]\w*",
        r"loading times?",
        r"load times?",
        r"low[- ]end",
        r"high[- ]end",
        r"cpu",
        r"gpu",
        r"vram",
        r"runs (?:smooth\w*|well|great|fine|badly|poorly|terribl\w*|horribl\w*)",
    ],
    "bugs": [
        r"bug\w*",
        r"glitch\w*",
        r"crash\w*",
        r"freez\w*",
        r"froze",
        r"broken",
        r"soft ?lock\w*",
        r"corrupt\w*",
        r"error\w*",
        r"unplayable",
        r"desync\w*",
    ],
    "gameplay": [
        r"gameplay",
        r"mechanics?",
        r"combat",
        r"gunplay",
        r"level design",
        r"progression",
        r"grind\w*",
        r"difficulty",
        r"game ?loop",
        r"missions?",
        r"quests?",
        r"skill tree",
        r"boss(?:es)?",
        r"replayab\w*",
        r"multiplayer",
        r"co-?op",
    ],
    "story": [
        r"story",
        r"stories",
        r"storyline",
        r"narrative",
        r"plot",
        r"characters?",
        r"writing",
        r"ending",
        r"lore",
        r"dialogues?",
        r"voice acting",
        r"protagonist",
    ],
    "price": [
        r"price\w*",
        r"overpriced",
        r"worth (?:it|the money|every penny)",
        r"not worth",
        r"money",
        r"refund\w*",
        r"expensive",
        r"cheap",
        r"on sale",
        r"full price",
        r"microtransactions?",
        r"mtx",
        r"pay[- ]to[- ]win",
        r"\$\d+",
        r"€\d+",
        r"\d+ ?(?:usd|eur)",
    ],
    "controls": [
        r"controls?",
        r"controller",
        r"gamepad",
        r"keyboard",
        r"mouse",
        r"key ?binds?",
        r"key ?bindings?",
        r"rebind\w*",
        r"input lag",
        r"aim assist",
        r"camera",
    ],
}
# "priceless" is praise, not a price remark
_EXCLUDE = {"price": re.compile(r"\bpriceless\b", re.I)}

_PATTERNS: dict[str, re.Pattern[str]] = {
    a: re.compile(r"(?<![\w-])(?:" + "|".join(words) + r")(?![\w-])", re.I)
    for a, words in _LEXICON.items()
}
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")


def _mentions(aspect: str, text: str) -> bool:
    ex = _EXCLUDE.get(aspect)
    cleaned = ex.sub(" ", text) if ex else text
    return bool(_PATTERNS[aspect].search(cleaned))


def aspects_in(text: str) -> set[str]:
    return {a for a in ASPECTS if _mentions(a, text)}


def aspect_sentences(text: str) -> dict[str, list[str]]:
    """aspect -> the sentences of `text` that mention it (aspects with no mention omitted)."""
    out: dict[str, list[str]] = {}
    for m in _SENTENCE.finditer(text):
        sentence = m.group(0).strip()
        if not sentence:
            continue
        for a in aspects_in(sentence):
            out.setdefault(a, []).append(sentence)
    return out
