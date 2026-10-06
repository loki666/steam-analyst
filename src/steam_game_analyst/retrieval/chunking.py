"""Documents -> retrieval chunks, one strategy per document type.

reviews     one chunk per review (long ones split), header "<game> review (recommended)"
news        patch notes / announcements, header "<game> patch notes: <title>"
wiki        one page section at a time, header "<game> wiki: <page> > <section>"

Splitting follows the LLM Engineer's Handbook: structure first (paragraphs; wiki pages are
already split into sections), then sentences, packed up to a token budget measured with the
embedding model's own tokenizer; only a sentence longer than the budget is cut by tokens.
Overlap is carried between pieces of one long paragraph, not across paragraph breaks.

The header is embedded with the body (it anchors short texts to their game); the payload
keeps the bare body plus every field retrieval can filter on. No author ids are ever stored.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

TokenCounter = Callable[[str], int]

# bge-m3 reads 512 tokens: body budget leaves room for the header and special tokens
MAX_TOKENS = 448
OVERLAP_TOKENS = 48
_CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
# handbook sentence regex (keeps "e.g.", "Mr.") + CJK full stops, which need no space after
_SENTENCE_END = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+|(?<=[。！？])")
_PROFILE_LINK = re.compile(
    r"https?://(?:www\.)?steamcommunity\.com/(?:id|profiles)/[^\s/]+/?", re.IGNORECASE
)
_ID_NAMESPACE = uuid.UUID("5f0b8c1e-6d1a-4e53-9a0c-2b9a7c1d3e44")


@dataclass(frozen=True)
class Chunk:
    id: str  # uuid5 of (source id, chunk index): stable across rebuilds
    text: str  # what gets embedded: header + body
    payload: dict[str, Any]


# ----------------------------------------------------------------------------- token counting
def approx_token_count(text: str) -> int:
    """Tokenizer-free estimate: one per word, one per CJK character."""
    cjk = len(_CJK_CHAR.findall(text))
    return cjk + len(_CJK_CHAR.sub(" ", text).split())


def hf_token_counter(model_name: str, cache_dir: str | None = None) -> TokenCounter:
    """Exact token count with the embedding model's own (fast) tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tok.model_max_length = 10**9  # counting only: silence the "longer than max length" warning

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


# ----------------------------------------------------------------------------- splitting
@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int
    para: int  # paragraph index: overlap never crosses a paragraph break
    whole_para: bool


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _cut(sentence: str, max_tokens: int, overlap_tokens: int, count: TokenCounter) -> list[str]:
    """Last resort for a sentence over budget: word windows, or character windows when there
    are no spaces (CJK, URLs). A token never spans less than one character, so a window of
    `max_tokens` characters always fits."""
    words = sentence.split()
    if len(words) <= 1:
        step = max(max_tokens - min(overlap_tokens, max_tokens // 2), 1)
        return [sentence[i : i + max_tokens] for i in range(0, len(sentence), step)]
    pieces: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        wn = count(w)
        if wn > max_tokens:  # one giant "word"
            pieces += [" ".join(cur)] if cur else []
            pieces += _cut(w, max_tokens, overlap_tokens, count)
            cur, n = [], 0
            continue
        if cur and n + wn > max_tokens:
            pieces.append(" ".join(cur))
            keep: list[str] = []
            kn = 0
            for prev in reversed(cur):  # word-level overlap
                pn = count(prev)
                if kn + pn > overlap_tokens:
                    break
                keep.insert(0, prev)
                kn += pn
            cur, n = keep, kn
        cur.append(w)
        n += wn
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _units(text: str, max_tokens: int, overlap_tokens: int, count: TokenCounter) -> list[_Unit]:
    units: list[_Unit] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for i, para in enumerate(paras):
        n = count(para)
        if n <= max_tokens:
            units.append(_Unit(para, n, i, True))
            continue
        for sent in split_sentences(para):
            sn = count(sent)
            pieces = [sent] if sn <= max_tokens else _cut(sent, max_tokens, overlap_tokens, count)
            units += [_Unit(p, sn if p is sent else count(p), i, False) for p in pieces]
    return units


def _join(units: list[_Unit]) -> str:
    out = units[0].text
    for prev, u in zip(units, units[1:], strict=False):
        out += (" " if u.para == prev.para else "\n\n") + u.text
    return out


def split_text(
    text: str,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    count: TokenCounter = approx_token_count,
) -> list[str]:
    """Structure first, then sentences, packed up to `max_tokens` (see module docstring)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_tokens or count(text) <= max_tokens:  # most reviews: no splitting
        return [text]
    chunks: list[str] = []
    cur: list[_Unit] = []
    n = 0
    for u in _units(text, max_tokens, overlap_tokens, count):
        if cur and n + u.tokens > max_tokens:
            chunks.append(_join(cur))
            carry: list[_Unit] = []
            if overlap_tokens and not u.whole_para and cur[-1].para == u.para:
                t = 0
                for prev in reversed(cur):
                    if prev.para != u.para or t + prev.tokens > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    t += prev.tokens
            cur = carry if sum(c.tokens for c in carry) + u.tokens <= max_tokens else []
            n = sum(c.tokens for c in cur)
        cur.append(u)
        n += u.tokens
    if cur:
        chunks.append(_join(cur))
    return chunks


# ----------------------------------------------------------------------------- chunkers
def _chunk_id(source_id: str, i: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source_id}#{i}"))


def _epoch(ts: datetime | None) -> int | None:
    """Naive timestamps in the tables are UTC."""
    if ts is None:
        return None
    return int((ts if ts.tzinfo else ts.replace(tzinfo=UTC)).timestamp())


def _make(
    source_id: str, header: str, body: str, base: dict[str, Any], count: TokenCounter
) -> Iterator[Chunk]:
    for i, piece in enumerate(split_text(body, count=count)):
        payload = {**base, "source_id": source_id, "chunk_index": i, "text": piece}
        yield Chunk(_chunk_id(source_id, i), f"{header}\n{piece}", payload)


def review_chunks(
    df: pl.DataFrame, names: Mapping[int, str], count: TokenCounter = approx_token_count
) -> list[Chunk]:
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
        out += _make(r["review_id"], f"{game} review ({verdict})", body, base, count)
    return out


def news_chunks(
    df: pl.DataFrame, names: Mapping[int, str], count: TokenCounter = approx_token_count
) -> list[Chunk]:
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
        out += _make(
            r["news_id"], f"{game} {label}: {r['title']}", r["contents"] or "", base, count
        )
    return out


def wiki_chunks(
    df: pl.DataFrame, names: Mapping[int, str], count: TokenCounter = approx_token_count
) -> list[Chunk]:
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
        out += _make(r["section_id"], f"{game} wiki: {where}", r["text"], base, count)
    return out
