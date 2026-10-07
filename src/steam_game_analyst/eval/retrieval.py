"""Retrieval metrics at document level: did the right review / patch note / wiki section come
back? Scoring by source document (not chunk id) keeps labels valid when chunking changes.

Protocol (PRD: Evaluation): tune on `dev` only; `test` runs once, with `final=True`."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from tqdm import tqdm

KS = (5, 20)
CHUNKS_PER_DOC = 3  # retrieve extra chunks so that k distinct documents remain after dedup


class _Retriever(Protocol):
    def retrieve(self, question: str, k: int = ..., overrides: Any = ...) -> Any: ...


def _doc_ranking(hits: Iterable[Any]) -> list[str]:
    """Source ids in rank order, each document counted once (its best chunk)."""
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(str(h.payload.get("source_id")), None)
    return list(seen)


def _summary(ranks: Sequence[int | None]) -> dict[str, float | int]:
    n = len(ranks)
    out: dict[str, float | int] = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(r is not None and r <= k for r in ranks) / n if n else 0.0
    out[f"mrr@{max(KS)}"] = sum(1 / r for r in ranks if r and r <= max(KS)) / n if n else 0.0
    return out


def evaluate(
    items: Sequence[dict[str, Any]],
    retriever: _Retriever,
    split: str = "dev",
    k: int = max(KS),
    final: bool = False,
) -> dict[str, Any]:
    """Recall@5/@20 and MRR@20 of the labelled source document, overall and per doc type.
    Questions labelled scope "broad" (many documents answer them equally well) have no single
    right document, so they are left out here and judged in the end-to-end answer eval."""
    if split == "test" and not final:
        raise PermissionError("the test split is frozen: evaluate it once, with final=True")
    in_split = [i for i in items if i.get("split") == split]
    chosen = [i for i in in_split if i.get("scope") != "broad"]
    ranks: list[int | None] = []
    by_type: dict[str, list[int | None]] = defaultdict(list)
    per_question: list[dict[str, Any]] = []
    for item in tqdm(chosen, desc=f"eval {split}"):
        docs = _doc_ranking(retriever.retrieve(item["question"], k=k * CHUNKS_PER_DOC).hits)[:k]
        rank = docs.index(item["source_id"]) + 1 if item["source_id"] in docs else None
        ranks.append(rank)
        by_type[item["doc_type"]].append(rank)
        per_question.append({"qid": item["qid"], "doc_type": item["doc_type"], "rank": rank})
    return {
        "split": split,
        "k": k,
        "n": len(chosen),
        "excluded_broad": len(in_split) - len(chosen),
        "overall": _summary(ranks),
        "by_doc_type": {t: _summary(r) for t, r in sorted(by_type.items())},
        "per_question": per_question,
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    """Paired comparison of two runs on the same questions: how many questions ranked their
    document better, worse or the same (a miss counts as rank infinity). Far more sensitive
    than comparing two averages on a small set."""
    inf = float("inf")
    old = {q["qid"]: q["rank"] or inf for q in before["per_question"]}
    new = {q["qid"]: q["rank"] or inf for q in after["per_question"]}
    common = old.keys() & new.keys()
    better = sum(new[q] < old[q] for q in common)
    worse = sum(new[q] > old[q] for q in common)
    return {
        "better": better,
        "worse": worse,
        "same": len(common) - better - worse,
        "n": len(common),
    }
