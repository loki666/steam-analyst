"""Question -> games + metadata filters, with plain rules (the W3 agent replaces this with an LLM
router). Game names are matched against the catalog; filter phrases are pattern-matched."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from steam_game_analyst.retrieval.filters import SearchFilters

_HOURS = r"(?:hours?|hrs?|h)\b"
_MIN_HOURS = [
    re.compile(rf"(\d+)\s*\+\s*{_HOURS}", re.I),
    re.compile(rf"(?:at least|over|more than|>=?)\s*(\d+)\s*{_HOURS}", re.I),
]
_MAX_HOURS = re.compile(rf"(?:under|less than|fewer than|<=?)\s*(\d+)\s*{_HOURS}", re.I)
_QUOTED = re.compile(r"[\"“]([^\"”]{2,60})[\"”]")
_MENTION = re.compile(
    r"\bmention(?:s|ing|ed)?\s+(?:the\s+)?(.+?)"
    r"(?=\s+(?:for|from|in|on|by|with|about|and|that|who|where|when)\b|[?.!,;]|$)",
    re.I,
)
_YEAR_IN = re.compile(r"\b(?:in|during)\s+(20\d\d)\b", re.I)
_YEAR_SINCE = re.compile(r"\b(?:since|from)\s+(20\d\d)\b", re.I)
_YEAR_AFTER = re.compile(r"\bafter\s+(20\d\d)\b", re.I)
_YEAR_TO = re.compile(r"\bbefore\s+(20\d\d)\b", re.I)
_DOC_HINTS = [
    (re.compile(r"\b(?:reviews?|reviewers?|players?|complain\w*)\b", re.I), ("review",)),
    (
        re.compile(r"\b(?:patch(?:es)?|patch notes?|updates?|changelog)\b", re.I),
        ("patch_note", "news"),
    ),
    (re.compile(r"\bwiki\b", re.I), ("wiki",)),
]


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"[™®©]", "", s).casefold())


def _norm(s: str) -> str:
    return " " + " ".join(_tokens(s)) + " "


def _capitalised_mention(question: str, word: str) -> bool:
    """True when `word` appears in the question with a capital first letter ("Journey", not
    "a long journey")."""
    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
    return any(m.group(0)[0].isupper() for m in re.finditer(pattern, question, re.IGNORECASE))


@dataclass(frozen=True)
class _Entry:
    key: str  # normalised, space-padded
    appid: int
    name: str
    single_word: bool  # one Latin word: needs a capitalised mention
    cjk: bool  # matched as a plain substring (no word boundaries in CJK text)


@dataclass(frozen=True)
class GameIndex:
    """Game names (plus unique multi-word subtitle aliases) -> appid, longest names first.

    Guards against false positives: purely numeric names ("1980") are never auto-linked;
    single-word names ("Satisfactory", "Journey") need a capitalised mention; an alias shared
    by several games ("Prologue") is dropped."""

    entries: tuple[_Entry, ...]

    @classmethod
    def from_names(cls, names: Mapping[int, str]) -> GameIndex:
        full: dict[str, tuple[int, str]] = {}
        aliases: dict[str, set[tuple[int, str]]] = defaultdict(set)
        for appid, name in names.items():
            toks = _tokens(name)
            if not toks or (len(toks) == 1 and toks[0].isdigit()):
                continue
            full.setdefault(_norm(name), (appid, name))
            if ":" in name:  # "Warhammer 40,000: Darktide" -> no alias (one word)
                alias_toks = _tokens(name.split(":", 1)[1])
                if len(alias_toks) >= 2:
                    aliases[" " + " ".join(alias_toks) + " "].add((appid, name))
        merged = dict(full)
        for key, owners in aliases.items():
            if len(owners) == 1 and key not in full:
                merged[key] = next(iter(owners))
        entries = [
            _Entry(
                key,
                appid,
                name,
                single_word=len(key.split()) == 1 and not _CJK.search(key),
                cjk=bool(_CJK.search(key)),
            )
            for key, (appid, name) in merged.items()
        ]
        return cls(tuple(sorted(entries, key=lambda e: -len(e.key))))

    @classmethod
    def from_frame(cls, games: pl.DataFrame) -> GameIndex:
        return cls.from_names(dict(games.select("appid", "name").iter_rows()))

    def find(self, question: str, strict: bool = True) -> list[tuple[int, str]]:
        """Games named in the question, in order of appearance; overlapping matches go to the
        longest name ("Hades II" beats "Hades"). `strict=False` (an explicit --game value)
        skips the capitalisation rule."""
        text = _norm(question)
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, int, str]] = []
        for e in self.entries:
            needle = e.key.strip() if e.cjk else e.key
            if strict and e.single_word and not _capitalised_mention(question, e.key.strip()):
                continue
            start = text.find(needle)
            while start != -1:
                end = start + len(needle)
                if not any(s < end and start < t for s, t in taken):
                    taken.append((start, end))
                    found.append((start, e.appid, e.name))
                    break
                start = text.find(needle, start + 1)
        seen: set[int] = set()
        out = []
        for _, appid, name in sorted(found):
            if appid not in seen:
                seen.add(appid)
                out.append((appid, name))
        return out


@dataclass(frozen=True)
class ParsedQuery:
    question: str
    games: tuple[tuple[int, str], ...] = ()
    filters: SearchFilters = field(default_factory=SearchFilters)


def _first_int(patterns: list[re.Pattern[str]], text: str) -> int | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return int(m.group(1))
    return None


def _mention(text: str, game_names: list[str]) -> str | None:
    """Phrase to require in the chunk text: a quoted string or "mentioning X" (a quoted game
    name is the game itself, not a phrase)."""
    names = {_norm(n) for n in game_names}
    for m in _QUOTED.finditer(text):
        if _norm(m.group(1)) not in names:
            return m.group(1).strip().lower()
    mention = _MENTION.search(text)
    return mention.group(1).strip().lower() if mention else None


def _doc_types(text: str) -> tuple[str, ...]:
    types: list[str] = []
    for pattern, kinds in _DOC_HINTS:
        if pattern.search(text):
            types += [k for k in kinds if k not in types]
    return tuple(types)


def _dates(text: str) -> tuple[date | None, date | None]:
    if m := _YEAR_IN.search(text):
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)
    start = None
    if m := _YEAR_SINCE.search(text):
        start = date(int(m.group(1)), 1, 1)
    elif m := _YEAR_AFTER.search(text):
        start = date(int(m.group(1)) + 1, 1, 1)
    end = date(int(m.group(1)) - 1, 12, 31) if (m := _YEAR_TO.search(text)) else None
    return start, end


def parse_question(question: str, games: GameIndex) -> ParsedQuery:
    found = games.find(question)
    min_h = _first_int(_MIN_HOURS, question)
    max_h = _first_int([_MAX_HOURS], question)
    date_from, date_to = _dates(question)
    filters = SearchFilters(
        appids=tuple(a for a, _ in found),
        doc_types=_doc_types(question),
        min_playtime_min=min_h * 60 if min_h is not None else None,
        max_playtime_min=max_h * 60 if max_h is not None else None,
        date_from=date_from,
        date_to=date_to,
        text_contains=_mention(question, [n for _, n in found]),
    )
    return ParsedQuery(question, tuple(found), filters)
