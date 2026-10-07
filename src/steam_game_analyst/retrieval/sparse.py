"""BM25 as Qdrant sparse vectors, computed on CPU.

Documents store the BM25 term-frequency part, tf·(k1+1) / (tf + k1·(1 − b + b·dl/avg_len));
the collection's IDF modifier makes Qdrant supply the IDF at query time, so queries are just
their distinct terms with weight 1. Terms map to sparse indices by a stable 32-bit hash.

Tokens are language-agnostic: NFKC + casefold, Unicode word runs, no stemming or stopwords
(IDF already discounts common words, and the dense leg covers morphology and paraphrase).
Chinese and Japanese have no spaces, so their runs become character bigrams; Korean is
space-separated and tokenised like Latin text."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from qdrant_client.http import models as qm

K1 = 1.2
B = 0.75
# mean tokens per chunk (header + body) over all sources, measured 2026-10-07 (reviews 72,
# news 287, wiki 165); fixed so stored weights stay comparable across partial re-indexes
AVG_LEN = 117.0

_WORD = re.compile(r"\w+")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORD.findall(unicodedata.normalize("NFKC", text).casefold()):
        pos = 0
        for m in _CJK_RUN.finditer(word):
            if m.start() > pos:
                tokens.append(word[pos : m.start()])
            run = m.group()
            tokens.extend([run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)])
            pos = m.end()
        if pos < len(word):
            tokens.append(word[pos:])
    return tokens


@lru_cache(maxsize=1 << 20)
def _index(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little")


def _vector(weights: dict[int, float]) -> qm.SparseVector:
    indices = sorted(weights)
    return qm.SparseVector(indices=indices, values=[weights[i] for i in indices])


def encode_document(text: str, avg_len: float = AVG_LEN) -> qm.SparseVector:
    tokens = tokenize(text)
    norm = K1 * (1 - B + B * len(tokens) / avg_len)
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        i = _index(token)  # a (rare) hash collision adds up rather than overwrites
        weights[i] = weights.get(i, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _vector(weights)


def encode_query(text: str) -> qm.SparseVector:
    return _vector({_index(t): 1.0 for t in set(tokenize(text))})
