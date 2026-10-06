"""Documents -> retrieval chunks, one strategy per document type.

reviews     one chunk per review (long ones split), header "<game> review (recommended)"
news        patch notes / announcements split on paragraphs, header "<game> patch notes: <title>"
wiki        one chunk per page section (long ones split), header "<game> wiki: <page> > <section>"

The header is embedded with the body (it anchors short texts to their game); the payload
keeps the bare body plus every field retrieval can filter on. No author ids are ever stored.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

MAX_WORDS = 300  # ~400-450 bge-m3 tokens: well under its 512-token window
MAX_CHARS = 1200  # CJK: ~1 token per character, so cap characters too
OVERLAP_WORDS = 30
_PROFILE_LINK = re.compile(
    r"https?://(?:www\.)?steamcommunity\.com/(?:id|profiles)/[^\s/]+/?", re.IGNORECASE
)
_ID_NAMESPACE = uuid.UUID("5f0b8c1e-6d1a-4e53-9a0c-2b9a7c1d3e44")


@dataclass(frozen=True)
class Chunk:
    id: str  # uuid5 of (source id, chunk index): stable across rebuilds
    text: str  # what gets embedded: header + body
    payload: dict[str, Any]


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    step = max(max_chars - overlap_chars, 1)
    return [text[i : i + max_chars] for i in range(0, len(text), step) if text[i : i + max_chars]]


def _split_long(para: str, max_words: int, max_chars: int, overlap: int) -> list[str]:
    """Words windows with `overlap`; any window still over `max_chars` (CJK, no spaces) is
    cut by characters."""
    words = para.split()
    pieces: list[str] = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    out: list[str] = []
    for piece in pieces:
        out += [piece] if len(piece) <= max_chars else _hard_split(piece, max_chars, overlap * 4)
    return out


def split_text(
    text: str,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_WORDS,
    max_chars: int = MAX_CHARS,
) -> list[str]:
    """Pack paragraphs into chunks of at most `max_words` words and `max_chars` characters
    (the character cap matters for CJK text, which has no spaces); longer paragraphs are
    split with `overlap` words carried into the next piece."""
    chunks: list[str] = []
    current: list[str] = []
    n_words = n_chars = 0
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        words = len(para.split())
        if words > max_words or len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, n_words, n_chars = [], 0, 0
            chunks += _split_long(para, max_words, max_chars, overlap)
            continue
        if current and (n_words + words > max_words or n_chars + len(para) > max_chars):
            chunks.append("\n\n".join(current))
            current, n_words, n_chars = [], 0, 0
        current.append(para)
        n_words += words
        n_chars += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_id(source_id: str, i: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source_id}#{i}"))


def _epoch(ts: datetime | None) -> int | None:
    """Naive timestamps in the tables are UTC."""
    if ts is None:
        return None
    return int((ts if ts.tzinfo else ts.replace(tzinfo=UTC)).timestamp())


def _make(source_id: str, header: str, body: str, base: dict[str, Any]) -> Iterator[Chunk]:
    for i, piece in enumerate(split_text(body)):
        payload = {**base, "source_id": source_id, "chunk_index": i, "text": piece}
        yield Chunk(_chunk_id(source_id, i), f"{header}\n{piece}", payload)


def review_chunks(df: pl.DataFrame, names: Mapping[int, str]) -> list[Chunk]:
    out: list[Chunk] = []
    for r in df.iter_rows(named=True):
        game = names.get(r["appid"], str(r["appid"]))
        verdict = {True: "recommended", False: "not recommended"}.get(r["voted_up"], "unrated")
        base = {
            "doc_type": "review",
            "appids": [r["appid"]],
            "game": game,
            "title": f"{game} review",
            "url": None,
            "created_at": _epoch(r["created_at"]),
            "voted_up": r["voted_up"],
            "playtime_min": r["playtime_forever_min"],
            "votes_up": r["votes_up"],
            "early_access": r["written_during_early_access"],
            "language": r["language"],
        }
        body = _PROFILE_LINK.sub("[profile link]", r["review_text"])  # no player identities
        out += _make(r["review_id"], f"{game} review ({verdict})", body, base)
    return out


def news_chunks(df: pl.DataFrame, names: Mapping[int, str]) -> list[Chunk]:
    out: list[Chunk] = []
    for r in df.iter_rows(named=True):
        game = names.get(r["appid"], str(r["appid"]))
        doc_type = "patch_note" if r["is_patch_note"] else "news"
        label = "patch notes" if r["is_patch_note"] else "news"
        base = {
            "doc_type": doc_type,
            "appids": [r["appid"]],
            "game": game,
            "title": r["title"],
            "url": r["url"],
            "created_at": _epoch(r["published_at"]),
            "language": "english",
        }
        out += _make(r["news_id"], f"{game} {label}: {r['title']}", r["contents"] or "", base)
    return out


def wiki_chunks(df: pl.DataFrame, names: Mapping[int, str]) -> list[Chunk]:
    out: list[Chunk] = []
    for r in df.iter_rows(named=True):
        appids = list(r["appids"])
        game = names.get(appids[0], r["slug"]) if appids else r["slug"]
        where = f"{r['title']} > {r['heading']}" if r["heading"] else r["title"]
        base = {
            "doc_type": "wiki",
            "appids": appids,
            "game": game,
            "title": r["title"],
            "heading": r["heading"],
            "url": r["url"],
            "created_at": None,
            "language": "english",
        }
        out += _make(r["section_id"], f"{game} wiki: {where}", r["text"], base)
    return out
