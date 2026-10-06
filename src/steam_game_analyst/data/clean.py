"""Text cleaning helpers shared by all sources (pure functions, unit-tested)."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")
_BBCODE = re.compile(
    r"\[/?(b|i|u|h1|h2|h3|list|olist|url|img|strike|spoiler|code|table|tr|th|td|p|hr|noparse|quote)(=[^\]]*)?\]",
    re.I,
)
_BB_STAR = re.compile(r"\s*\[\*\]\s*")
_BB_BLOCK_END = re.compile(r"\[/?(h1|h2|h3|list|olist|p|table|tr|hr)\]", re.I)
_URL_PREVIEW = re.compile(r"\{STEAM_CLAN_IMAGE\}\S*", re.I)
_OWNERS = re.compile(r"(\d[\d,]*)\s*-\s*(\d[\d,]*)")

PATCH_WORDS = re.compile(
    r"\b(patch|hotfix|update|changelog|bug ?fix(es)?|balance|version \d|v\d+\.\d+|notes)\b", re.I
)


def strip_html(text: str | None) -> str:
    """Remove HTML tags, decode entities, normalise whitespace. Keeps line breaks for <br>/<p>/<li>."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    for br in soup.find_all(["br", "p", "li", "h1", "h2", "h3", "h4", "div", "tr"]):
        br.insert_after("\n")
    out = soup.get_text()
    return normalise_ws(out)


def strip_bbcode(text: str | None) -> str:
    """Steam news/announcements use BBCode-ish markup; strip it, keep bullets."""
    if not text:
        return ""
    t = _URL_PREVIEW.sub("", text)
    t = _BB_STAR.sub("\n- ", t)
    t = _BB_BLOCK_END.sub("\n", t)  # headings/lists end a line
    t = _BBCODE.sub("", t)
    t = strip_html(t)  # announcements sometimes mix HTML in
    return normalise_ws(t)


def normalise_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL.sub("\n\n", text)
    return text.strip()


def stable_id(*parts: object, n: int = 16) -> str:
    """Deterministic id from content, so re-runs produce the same keys and no Steam ids leak."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n]


def parse_owners(s: str | None) -> tuple[int | None, int | None]:
    """'20000 - 50000' -> (20000, 50000). Handles commas and 'N/A'."""
    if not s:
        return None, None
    m = _OWNERS.search(s)
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def parse_release_date(s: str | None) -> date | None:
    """FronkonGames uses 'Oct 21, 2022' or '2022-10-21'; Mendeley uses ISO or 'YYYY-MM-DD'."""
    if not s or s in {"TBA", "Coming soon", "To be announced"}:
        return None
    s = s.strip()
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%d %b, %Y", "%b %Y", "%Y-%m-%dT%H:%M:%S", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def split_list(s: str | list[str] | None, sep: str = ",") -> list[str]:
    """'Action,Indie' or '["Action","Indie"]' or list -> ['Action', 'Indie']."""
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    s = str(s).strip()
    if s.startswith("["):
        import json

        try:
            return [str(x).strip() for x in json.loads(s)]
        except json.JSONDecodeError:
            s = s.strip("[]").replace("'", "").replace('"', "")
    return [x.strip() for x in s.split(sep) if x.strip()]


_COMPANY_SEP = re.compile(r",(?=\S)")
_CORP_SUFFIX = re.compile(
    r"^(inc|llc|ltd|limited|co|corp|gmbh|s\.?a|s\.?r\.?l|b\.?v|pty|plc|ab|oy|kk)\.?$",
    re.IGNORECASE,
)


def split_companies(s: str | list[str] | None) -> list[str]:
    """Developer/publisher field -> names. Companies are separated by a bare comma, while
    ", " is part of a name ("FromSoftware, Inc.,Bandai Namco" -> 2 companies). A piece that
    is only a corporate suffix ("Kairosoft Co.,Ltd") is re-attached to the name before it."""
    if s is None or isinstance(s, list) or str(s).strip().startswith("["):
        return split_list(s)
    names: list[str] = []
    for part in _COMPANY_SEP.split(str(s).strip().rstrip(",")):
        part = part.strip()
        if not part:
            continue
        if names and _CORP_SUFFIX.match(part):
            names[-1] = f"{names[-1]}, {part}"
        else:
            names.append(part)
    return names


def tags_from_field(s: str | dict | None, top: int = 20) -> list[str]:
    """FronkonGames `tags` is either a JSON dict {tag: votes} or a comma list. Return names by votes."""
    if not s:
        return []
    if isinstance(s, dict):
        return [k for k, _ in sorted(s.items(), key=lambda kv: -int(kv[1] or 0))][:top]
    s = str(s).strip()
    if s.startswith("{"):
        import json

        try:
            d = json.loads(s)
            return tags_from_field(d, top)
        except json.JSONDecodeError:
            pass
    return split_list(s)[:top]


def is_patch_note(title: str | None, tags: list[str] | None = None) -> bool:
    if tags and any(t.lower() in {"patchnotes", "patch_notes", "patch notes"} for t in tags):
        return True
    return bool(title and PATCH_WORDS.search(title))
