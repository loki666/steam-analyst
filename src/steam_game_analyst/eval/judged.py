"""Retrieval scored against judged relevance (pooled judging).

`build_pool` gathers, per question, the top `depth` chunks of every retrieval mode plus the
chunk the question was written from, in a blind order (by chunk id), for a judge to grade:
0 = does not help, 1 = partly answers / relevant evidence, 2 = answers it. `evaluate_judged`
then scores a ranking against every chunk judged relevant, not just the generating one, so
broad questions and questions that do not name the game count fairly. Hits outside the pool
are unjudged and score as irrelevant; `unjudged_hits` says when a new system needs judging.

Judgments are keyed by chunk id: re-judge after a chunking change."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

GRADES = (0, 1, 2)
HIT_KS = (1, 5, 10)
CANDIDATE_FIELDS = ("source_id", "doc_type", "game", "title", "text")

Judgments = Mapping[tuple[str, str], int]


class _Retriever(Protocol):
    def retrieve(self, question: str, k: int = ..., overrides: Any = ...) -> Any: ...


class _Store(Protocol):
    def fetch(self, ids: Sequence[str]) -> list[Any]: ...


def _chosen(items: Sequence[dict[str, Any]], split: str, final: bool) -> list[dict[str, Any]]:
    if split == "test" and not final:
        raise PermissionError("the test split is frozen: evaluate it once, with final=True")
    return [i for i in items if i.get("split") == split]


def _candidate(hit: Any) -> dict[str, Any]:
    return {"chunk_id": hit.id, **{f: hit.payload.get(f) or "" for f in CANDIDATE_FIELDS}}


def build_pool(
    items: Sequence[dict[str, Any]],
    retrievers: Mapping[str, _Retriever],
    store: _Store,
    split: str = "dev",
    depth: int = 10,
    final: bool = False,
) -> list[dict[str, Any]]:
    """Per question: the union of every retriever's top `depth` chunks and the generating
    chunk, sorted by chunk id so the judge cannot tell which system (or the generator)
    supplied a passage."""
    pool = []
    for item in tqdm(_chosen(items, split, final), desc=f"pool {split}"):
        found: dict[str, Any] = {}
        for retriever in retrievers.values():
            for h in retriever.retrieve(item["question"], k=depth).hits:
                found.setdefault(h.id, h)
        if item["chunk_id"] not in found:
            for h in store.fetch([item["chunk_id"]]):
                found[h.id] = h
        pool.append(
            {
                "qid": item["qid"],
                "question": item["question"],
                "candidates": [_candidate(found[c]) for c in sorted(found)],
            }
        )
    return pool


def load_judgments(path: Path) -> dict[tuple[str, str], int]:
    """(qid, chunk_id) -> grade; a later line overrides an earlier one."""
    out: dict[tuple[str, str], int] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("grade") not in GRADES:
            raise ValueError(f"{path}:{n}: grade must be one of {GRADES}, got {row.get('grade')!r}")
        out[(row["qid"], row["chunk_id"])] = row["grade"]
    return out


def _dcg(gains: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def _first(grades: Sequence[int], threshold: int) -> int | None:
    return next((i + 1 for i, g in enumerate(grades) if g >= threshold), None)


def _summary(rows: Sequence[dict[str, Any]], key: str, k: int) -> dict[str, float | int]:
    n = len(rows)
    if not n:
        return {"n": 0}
    ranks = [r[key] for r in rows]
    out: dict[str, float | int] = {"n": n}
    for h in HIT_KS:
        out[f"hit@{h}"] = sum(r is not None and r <= h for r in ranks) / n
    out[f"mrr@{k}"] = sum(1 / r for r in ranks if r) / n
    return out


def evaluate_judged(
    items: Sequence[dict[str, Any]],
    retriever: _Retriever,
    judgments: Judgments,
    split: str = "dev",
    k: int = 10,
    final: bool = False,
) -> dict[str, Any]:
    """hit@1/5/10, MRR, nDCG and precision at `k` against judged grades (relevant = grade
    >= 1; `strict` repeats hit/MRR with grade 2 only). Questions with no chunk judged
    relevant are counted in `no_relevant` and left out of the averages."""
    by_qid: dict[str, list[int]] = defaultdict(list)
    for (qid, _), g in judgments.items():
        by_qid[qid].append(g)
    per_question: list[dict[str, Any]] = []
    no_relevant = unjudged = 0
    for item in tqdm(_chosen(items, split, final), desc=f"judged eval {split}"):
        qid = item["qid"]
        ideal = sorted(by_qid.get(qid, []), reverse=True)[:k]
        if not ideal or ideal[0] == 0:
            no_relevant += 1
            continue
        hits = retriever.retrieve(item["question"], k=k).hits[:k]
        unjudged += sum((qid, h.id) not in judgments for h in hits)
        grades = [judgments.get((qid, h.id), 0) for h in hits]
        per_question.append(
            {
                "qid": qid,
                "doc_type": item["doc_type"],
                "rank": _first(grades, 1),
                "strict_rank": _first(grades, 2),
                "ndcg": _dcg(grades) / _dcg(ideal),
                "precision": sum(g >= 1 for g in grades) / k,
            }
        )

    def summary(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        out = _summary(rows, "rank", k)
        if rows:
            out[f"ndcg@{k}"] = sum(r["ndcg"] for r in rows) / len(rows)
            out[f"p@{k}"] = sum(r["precision"] for r in rows) / len(rows)
        return out

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        by_type[row["doc_type"]].append(row)
    return {
        "split": split,
        "k": k,
        "n": len(per_question),
        "no_relevant": no_relevant,
        "unjudged_hits": unjudged,
        "overall": summary(per_question),
        "strict": _summary(per_question, "strict_rank", k),
        "by_doc_type": {t: summary(r) for t, r in sorted(by_type.items())},
        "per_question": per_question,
    }
