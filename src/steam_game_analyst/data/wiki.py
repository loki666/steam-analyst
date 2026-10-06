"""Fandom wiki pages for the subset games, via each wiki's MediaWiki API (CC BY-SA 3.0).

The nyuuzyou/fandom HF dump has no wiki identifier per page, so pages are pulled per wiki:
resolve game -> <slug>.fandom.com (guessed from the name, validated against the wiki's
site name; `configs/wikis.yaml` overrides), list main-namespace pages, keep the largest,
fetch their wikitext in batches and convert it to plain-text sections.

Output (data/raw/wiki/):
  <slug>.jsonl     one page per line: slug, pageid, title, url, sections[{heading, text}]
  wiki_map.json    [{slug, sitename, appids}]  (several games can share a wiki: Hades, Hades II)
  resolved.json    appid -> wiki info | null  (cache, so reruns skip the slug probing)
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import mwparserfromhell as mw
from rich.console import Console
from tqdm import tqdm

console = Console()

API = "https://{slug}.fandom.com/api.php"
PAGE_URL = "https://{slug}.fandom.com/wiki/{title}"
FETCH_BATCH = 50  # MediaWiki limit for titles= per query

_ROMAN_OR_NUMBER = re.compile(r"^(?:[ivx]+|\d+)$")
_SITE_STOPWORDS = {"wiki", "wikia", "fandom", "the", "official"}
_SKIP_LINK_PREFIXES = ("file:", "image:", "category:", "media:")
_SKIP_TEMPLATES = {"delete", "stub", "cleanup", "citation needed", "cn", "clr", "clear", "-", "!"}
_SKIP_INFOBOX_KEYS = {"image", "icon", "caption", "imagecaption", "image1", "title1"}


# ----------------------------------------------------------------------------- naming
def _tokens(s: str) -> list[str]:
    s = re.sub(r"(?<=\d),(?=\d)", "", s.lower())  # 40,000 -> 40000
    return re.findall(r"[a-z0-9]+", s)


def slug_candidates(name: str) -> list[str]:
    """Likely <slug>.fandom.com names for a game, most specific first."""
    toks = _tokens(name)
    out = ["".join(toks), "-".join(toks)]
    if ":" in name:
        head, tail = name.split(":", 1)
        out += ["".join(_tokens(head)), "".join(_tokens(tail))]
    elif len(toks) > 1 and _ROMAN_OR_NUMBER.match(toks[-1]):
        out.append("".join(toks[:-1]))  # "Hades II" -> "hades", "Cyberpunk 2077" -> "cyberpunk"
    return list(dict.fromkeys(c for c in out if c))


def sitename_matches(sitename: str, game_name: str) -> bool:
    """A wiki belongs to a game when its name words (minus "Wiki") all appear in the game name."""
    site = set(_tokens(sitename)) - _SITE_STOPWORDS
    return bool(site) and site <= set(_tokens(game_name))


# ----------------------------------------------------------------------------- wikitext
@dataclass(frozen=True)
class Section:
    heading: str  # "Recipe > Upgrades"; "" for the lead
    text: str


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _render_template(t: Any) -> str:
    if str(t.name).strip().lower() in _SKIP_TEMPLATES:
        return ""
    named = [
        (str(p.name).strip(), p.value.strip_code().strip())
        for p in t.params
        if p.showkey and str(p.name).strip().lower() not in _SKIP_INFOBOX_KEYS
    ]
    positional = [p.value.strip_code().strip() for p in t.params if not p.showkey]
    lines = [f"{k}: {v}" for k, v in named if v]
    words = " ".join(v for v in positional if v)
    if lines:
        return "\n" + "\n".join(lines) + "\n" + (words + "\n" if words else "")
    return words


def _render(code: Any) -> None:
    """In place: drop file/category links, replace templates with their text (innermost first)."""
    for link in code.filter_wikilinks():
        if str(link.title).strip().lower().startswith(_SKIP_LINK_PREFIXES):
            try:
                code.remove(link)
            except ValueError:  # already gone with a removed parent
                pass
    for t in reversed(code.filter_templates()):
        try:
            code.replace(t, _render_template(t))
        except ValueError:
            pass


def wikitext_to_sections(wikitext: str) -> list[Section]:
    """Wikitext -> plain-text sections with their heading path; empty sections dropped."""
    code = mw.parse(wikitext)
    _render(code)
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    for sec in code.get_sections(flat=True, include_lead=True):
        headings = sec.filter_headings(recursive=False)
        if headings:
            h = headings[0]
            path = [(lvl, t) for lvl, t in path if lvl < h.level] + [
                (h.level, h.title.strip_code().strip())
            ]
            sec.remove(h)
        text = _clean(sec.strip_code(normalize=True, collapse=True))
        if text:
            sections.append(Section(" > ".join(t for _, t in path) if headings else "", text))
    return sections


# ----------------------------------------------------------------------------- API client
@dataclass(frozen=True)
class WikiInfo:
    slug: str
    sitename: str
    articles: int


class FandomClient:
    def __init__(
        self,
        client: httpx.Client,
        rate_per_sec: float = 2.0,
        retries: int = 3,
        min_articles: int = 20,
        backoff_s: float = 2.0,
    ) -> None:
        self._client = client
        self._delay = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._retries = retries
        self._min_articles = min_articles
        self._backoff = backoff_s

    def _get(self, slug: str, params: dict[str, Any]) -> httpx.Response | None:
        """GET the wiki API. None only when the wiki does not exist (404) or is closed (410);
        anything else that fails (403/408/429/5xx, timeouts) is retried with backoff and then
        raised, so a transient block is never mistaken for "no wiki"."""
        err: Exception | None = None
        for attempt in range(self._retries):
            if self._delay:
                time.sleep(self._delay)
            retry_after = 0.0
            try:
                r = self._client.get(
                    API.format(slug=slug),
                    params={**params, "format": "json"},
                    follow_redirects=True,
                )
            except httpx.TransportError as e:
                err = e
            else:
                if r.status_code < 400:
                    return r
                if r.status_code in (404, 410):
                    return None
                err = httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                header = r.headers.get("Retry-After", "")
                retry_after = float(header) if header.isdigit() else 0.0
            if attempt < self._retries - 1 and self._backoff:
                wait = max(retry_after, self._backoff * 2**attempt)
                time.sleep(wait + random.uniform(0, self._backoff))
        raise RuntimeError(f"fandom {slug}: giving up after {self._retries} tries") from err

    @staticmethod
    def _json(r: httpx.Response) -> dict[str, Any] | None:
        """API payload, or None when a redirect landed on an HTML page (dead wiki)."""
        if not r.url.path.endswith("/api.php"):
            return None
        try:
            body = r.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def site_info(self, slug: str) -> WikiInfo | None:
        r = self._get(slug, {"action": "query", "meta": "siteinfo", "siprop": "general|statistics"})
        body = self._json(r) if r is not None else None
        if r is None or body is None:
            return None
        q = body.get("query", {})
        general, stats = q.get("general", {}), q.get("statistics", {})
        if general.get("lang", "en") != "en":
            return None
        final_slug = r.url.host.split(".")[0] if r.url.host.endswith("fandom.com") else slug
        return WikiInfo(final_slug, general.get("sitename", ""), int(stats.get("articles", 0)))

    def resolve(self, game_name: str) -> WikiInfo | None:
        """First candidate slug whose English wiki is named after the game."""
        for slug in slug_candidates(game_name):
            info = self.site_info(slug)
            if (
                info
                and info.articles >= self._min_articles
                and sitename_matches(info.sitename, game_name)
            ):
                return info
        return None

    def largest_pages(self, slug: str, n: int, min_length: int = 1500) -> list[dict[str, Any]]:
        """The `n` longest main-namespace articles (byte length as a cheap proxy for substance)."""
        pages: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "action": "query",
            "generator": "allpages",
            "gapnamespace": 0,
            "gapfilterredir": "nonredirects",
            "gaplimit": 500,
            "prop": "info",
            "formatversion": 2,
        }
        while True:
            r = self._get(slug, params)
            body = self._json(r) if r is not None else None
            if body is None:
                break
            raw = body.get("query", {}).get("pages", [])
            pages += raw.values() if isinstance(raw, dict) else raw
            cont = body.get("continue")
            if not cont:
                break
            params = {**params, **cont}
        keep = [p for p in pages if int(p.get("length", 0)) >= min_length]
        return sorted(keep, key=lambda p: -int(p["length"]))[:n]

    def fetch_wikitext(self, slug: str, titles: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(titles), FETCH_BATCH):
            r = self._get(
                slug,
                {
                    "action": "query",
                    "titles": "|".join(titles[i : i + FETCH_BATCH]),
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "formatversion": 2,
                },
            )
            body = self._json(r) if r is not None else None
            if body is None:  # never write a wiki file with silently missing pages
                raise RuntimeError(f"fandom {slug}: page batch {i // FETCH_BATCH} unavailable")
            raw = body.get("query", {}).get("pages", [])
            for page in raw.values() if isinstance(raw, dict) else raw:
                revs = page.get("revisions") or []
                if revs:
                    main = revs[0].get("slots", {}).get("main", {})
                    out[page["title"]] = main.get("content", main.get("*", ""))
        return out


# ----------------------------------------------------------------------------- download
def _resolve_all(
    out_dir: Path,
    games: Iterable[tuple[int, str]],
    fc: FandomClient,
    overrides: dict[int, str | None],
) -> dict[str, tuple[str, list[int]]]:
    """slug -> (sitename, appids). Cached per appid in resolved.json."""
    cache_file = out_dir / "resolved.json"
    cache: dict[str, Any] = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    wikis: dict[str, tuple[str, list[int]]] = {}
    try:
        for appid, name in tqdm(list(games), desc="resolve wikis"):
            key = str(appid)
            if appid in overrides:
                slug = overrides[appid]
                info = fc.site_info(slug) if slug else None
            elif key in cache:
                info = WikiInfo(**cache[key]) if cache[key] else None
            else:
                try:
                    info = fc.resolve(name)
                except RuntimeError as e:  # transient: not cached, retried next run
                    console.print(f"[yellow]wiki: could not resolve {name!r}: {e}[/]")
                    continue
            cache[key] = info.__dict__ if info else None
            if info:
                wikis.setdefault(info.slug, (info.sitename, []))[1].append(appid)
    finally:  # keep what was resolved if the network gives up mid-way
        cache_file.write_text(json.dumps(cache, indent=1))
    return wikis


def _page_rows(slug: str, fc: FandomClient, max_pages: int, min_length: int) -> list[dict]:
    pages = fc.largest_pages(slug, n=max_pages, min_length=min_length)
    texts = fc.fetch_wikitext(slug, [p["title"] for p in pages])
    rows = []
    for p in pages:
        try:
            sections = wikitext_to_sections(texts.get(p["title"], ""))
        except Exception as e:  # odd markup (deep templates, parser bugs): skip the page
            console.print(f"[yellow]wiki {slug}: skipped {p['title']!r}: {type(e).__name__}[/]")
            continue
        if sections:
            rows.append(
                {
                    "slug": slug,
                    "pageid": p["pageid"],
                    "title": p["title"],
                    "url": PAGE_URL.format(slug=slug, title=quote(p["title"].replace(" ", "_"))),
                    "sections": [s.__dict__ for s in sections],
                }
            )
    return rows


def download_wikis(
    raw_dir: Path,
    games: Iterable[tuple[int, str]],
    fc: FandomClient,
    max_pages: int = 300,
    min_length: int = 1500,
    overrides: dict[int, str | None] | None = None,
) -> Path:
    """Resolve each game's wiki and save its largest pages under raw_dir/wiki. Idempotent per wiki."""
    out_dir = raw_dir / "wiki"
    out_dir.mkdir(parents=True, exist_ok=True)
    wikis = _resolve_all(out_dir, games, fc, overrides or {})
    mapping = [
        {"slug": slug, "sitename": sitename, "appids": sorted(appids)}
        for slug, (sitename, appids) in sorted(wikis.items())
    ]
    (out_dir / "wiki_map.json").write_text(json.dumps(mapping, indent=1))
    console.print(f"wiki: {len(mapping)} wikis for {sum(len(m['appids']) for m in mapping)} games")
    for slug in tqdm(sorted(wikis), desc="wiki pages"):
        dest = out_dir / f"{slug}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            rows = _page_rows(slug, fc, max_pages, min_length)
        except RuntimeError as e:
            console.print(f"[yellow]wiki {slug}: {e}[/]")
            continue
        tmp = dest.with_suffix(".jsonl.part")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(dest)
    return out_dir
