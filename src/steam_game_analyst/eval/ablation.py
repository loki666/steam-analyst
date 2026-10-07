"""Retrieval ablation: judged metrics per mode with bootstrap 95% CIs, and each mode's
question-by-question difference from a baseline (paired bootstrap over questions)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

METRICS = ("hit@5", "hit@10", "mrr@10", "ndcg@10", "p@10", "strict_hit@5")
# shown as differences from the baseline
DELTAS = ("ndcg@10", "mrr@10", "hit@5")
_LABELS = {
    "hit@5": "hit@5",
    "hit@10": "hit@10",
    "mrr@10": "MRR@10",
    "ndcg@10": "nDCG@10",
    "p@10": "P@10",
    "strict_hit@5": "strict hit@5",
}
N_BOOT = 2000


def value(row: Mapping[str, Any], metric: str, k: int) -> float:
    """One question's score on `metric`, from a judged-eval per-question row."""
    name, _, cut = metric.partition("@")
    if name in ("hit", "strict_hit"):
        rank = row["strict_rank" if name == "strict_hit" else "rank"]
        return float(rank is not None and rank <= int(cut))
    if name == "mrr":
        rank = row["rank"]
        return 1.0 / rank if rank and rank <= int(cut) else 0.0
    if name == "ndcg":
        return float(row["ndcg"])
    if name == "p":
        return float(row["precision"])
    raise ValueError(f"unknown metric {metric!r}")


def bootstrap_ci(
    values: Sequence[float], n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and its 95% percentile-bootstrap interval, resampling questions."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def _by_qid(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {r["qid"]: r for r in result["per_question"]}


def ablation(results: Mapping[str, Mapping[str, Any]], baseline: str = "hybrid") -> dict[str, Any]:
    """Per-mode metrics with CIs, plus paired differences from `baseline`. Every mode must
    be scored on the same questions (the judged eval keeps a question when any of its
    chunks is judged relevant, which does not depend on the mode)."""
    if baseline not in results:
        raise ValueError(f"baseline {baseline!r} is not among the modes {sorted(results)}")
    base = _by_qid(results[baseline])
    if not base:
        raise ValueError(
            "no question has a judged relevant chunk: pool and judge this split first "
            "(`sga eval pool`)"
        )
    k = int(results[baseline]["k"])
    qids = sorted(base)
    rows = []
    for mode, res in results.items():
        per_q = _by_qid(res)
        if sorted(per_q) != qids:
            raise ValueError(f"{mode} and {baseline} were not scored on the same questions")
        metrics = {}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([value(per_q[q], m, k) for q in qids])
            metrics[m] = {"mean": mean, "lo": lo, "hi": hi}
        deltas = {}
        if mode != baseline:
            for m in DELTAS:
                d = [value(per_q[q], m, k) - value(base[q], m, k) for q in qids]
                mean, lo, hi = bootstrap_ci(d)
                better = sum(x > 1e-12 for x in d)
                worse = sum(x < -1e-12 for x in d)
                deltas[m] = {"mean": mean, "lo": lo, "hi": hi, "better": better, "worse": worse}
        rows.append(
            {
                "mode": mode,
                "unjudged_hits": int(res.get("unjudged_hits", 0)),
                "metrics": metrics,
                "vs_baseline": deltas,
            }
        )
    return {"baseline": baseline, "k": k, "n": len(qids), "n_boot": N_BOOT, "rows": rows}


def _ci(c: Mapping[str, float], signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(c['mean'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def to_markdown(table: Mapping[str, Any], split: str) -> str:
    base = table["baseline"]
    head = [_LABELS[m] for m in METRICS]
    lines = [
        f"Judged retrieval on the {split} split: {table['n']} questions, top {table['k']} "
        f"chunks, mean [95% bootstrap CI over questions, {table['n_boot']} resamples].",
        "",
        "| Mode | " + " | ".join(head) + " |",
        "| --- |" + " --- |" * len(head),
    ]
    for r in table["rows"]:
        name = f"{r['mode']} (baseline)" if r["mode"] == base else r["mode"]
        lines.append(f"| {name} | " + " | ".join(_ci(r["metrics"][m]) for m in METRICS) + " |")
    base_row = next(r for r in table["rows"] if r["mode"] == base)
    if base_row["unjudged_hits"]:
        lines += ["", f"{base} returned {base_row['unjudged_hits']} unjudged chunks (scored 0)."]
    dhead = [f"Δ {_LABELS[m]} vs {base}" for m in DELTAS]
    lines += [
        "",
        f"Paired differences from {base} (better / worse = questions that gained / lost).",
        "",
        "| Mode | " + " | ".join(dhead) + " | Unjudged hits |",
        "| --- |" + " --- |" * (len(dhead) + 1),
    ]
    for r in table["rows"]:
        if r["mode"] == base:
            continue
        cells = [
            f"{_ci(r['vs_baseline'][m], signed=True)} "
            f"({r['vs_baseline'][m]['better']} / {r['vs_baseline'][m]['worse']})"
            for m in DELTAS
        ]
        unjudged = f"{r['unjudged_hits']} unjudged" if r["unjudged_hits"] else "0"
        lines.append(f"| {r['mode']} | " + " | ".join(cells) + f" | {unjudged} |")
    return "\n".join(lines) + "\n"
