"""Patch notes whose titles match the question, looked up before the agent plans its calls.

Patch-impact answers stand or fall on the event date, and the model often takes one from the
question's wording ("in July 2023" -> 1 July) instead of the patch note. The agent prompt
therefore lists the news titles that best match the question, with their dates.

Matching is plain token overlap with fixed, generic rules (not tuned on eval questions):
version strings ("1.3", "2.0.1", "#9") match exactly; bare numbers count only after a version
word ("update 12"); other words count when they are not English stopwords, common patch and
review words, month names or part of the game's name. Titles that announce or tease an
update lose a point, patch notes get a small bonus, ties go to the earliest date."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any, Protocol

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"#?\d+(?:\.\d+)*|[a-z][a-z']*")
VERSION_WORDS = frozenset({"patch", "update", "hotfix", "version", "season", "chapter", "part"})
DOMAIN_WORDS = frozenset(
    {
        "patch", "patches", "update", "updates", "hotfix", "hotfixes", "notes", "release",
        "released", "version", "game", "games", "player", "players", "review", "reviews",
        "steam", "change", "changed", "changes",
    }
)  # fmt: skip
MONTHS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    }
)  # fmt: skip
ANNOUNCE_WORDS = frozenset(
    {
        "trailer", "teaser", "announcement", "announcing", "announced", "reveal", "revealed",
        "roadmap", "preview", "coming", "soon", "countdown", "livestream", "date",
    }
)  # fmt: skip
IGNORED = ENGLISH_STOP_WORDS | DOMAIN_WORDS | MONTHS
MIN_WORD_LEN = 3
PATCH_NOTE_BONUS = 0.5
MAX_GAMES = 3


class _Fetch(Protocol):
    def fetch(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]: ...


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _question_terms(question: str, game_name: str) -> tuple[set[str], set[str]]:
    """(version strings, distinctive words) of the question."""
    name = set(_tokens(game_name))
    toks = _tokens(question)
    versions: set[str] = set()
    words: set[str] = set()
    for prev, tok in zip(["", *toks], toks, strict=False):
        if tok[0].isdigit() or tok.startswith("#"):
            if "." in tok or tok.startswith("#") or prev in VERSION_WORDS:
                versions.add(tok)
        elif len(tok) >= MIN_WORD_LEN and tok not in IGNORED and tok not in name:
            words.add(tok)
    return versions, words


def match_patch_titles(
    question: str,
    game_name: str,
    titles: Sequence[tuple[Any, ...]],
    k: int = 3,
) -> list[tuple[date, str]]:
    """Up to k (date, title), best first, from items (date, title[, is_patch_note])."""
    versions, words = _question_terms(question, game_name)
    if not versions and not words:
        return []
    scored = []
    for item in titles:
        day, title = item[0], item[1]
        toks = set(_tokens(title))
        overlap = 2 * len(versions & toks) + len(words & toks)
        if overlap == 0:
            continue
        score: float = overlap - len(ANNOUNCE_WORDS & toks)
        score += PATCH_NOTE_BONUS if len(item) > 2 and item[2] else 0
        if score > PATCH_NOTE_BONUS:
            scored.append((-score, day, title))
    return [(day, title) for _, day, title in sorted(scored)[:k]]


def patch_candidates(db: _Fetch, games: Sequence[dict[str, Any]], question: str) -> list[str]:
    """Prompt lines 'Game (appid): YYYY-MM-DD "title"' for the games named in the question."""
    lines = []
    for g in games[:MAX_GAMES]:
        rows = db.fetch(
            "SELECT CAST(published_at AS DATE), title, is_patch_note FROM news WHERE appid = ?",
            [g["appid"]],
        )
        for day, title in match_patch_titles(question, g["name"], rows):
            lines.append(f'{g["name"]} ({g["appid"]}): {day.isoformat()} "{title}"')
    return lines
