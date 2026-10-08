"""Judging answers: blind batches for an LLM judge, judgment files, judge agreement and the
scores that go into the README.

A batch hides which configuration (label, model, retrieval mode) wrote each answer: items
get opaque ids, are shuffled, and the id -> (qid, label) key is kept apart from what the
judge reads. The judge grades each answer against the question's key points and against
the evidence the agent itself had (its tool outputs and retrieved documents), following
RUBRIC. Judgments are keyed by item id: re-judge after re-running the agent."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from steam_game_analyst.eval.ablation import bootstrap_ci

RUBRIC_VERSION = "v1"
RUBRIC = """Grade one answer of a Steam game-analytics assistant. You see the question, the key
points a correct answer contains (computed from the data by the eval author), and the evidence
the assistant had: its tool outputs [T1..] and retrieved documents [1..].

correct (0-2): does the answer get the key points right?
  2 = every key point that matters is present and right (small rounding is fine; other true
      details are fine). For a question the assistant should decline or cannot answer from
      the data, 2 = it declines or says the data is missing, without inventing an answer.
  1 = partly right: a main point missing or vague, or a minor error.
  0 = wrong, missing the main point, contradicts the key points, or answers a question that
      should have been declined.
  Key points about a model prediction (predict_reception) depend on the attributes passed:
  if the assistant mapped the game to different tags, judge whether its conclusion follows
  from its own tool output and is consistent in direction with the key points.
grounded (0-2): are the answer's numbers and claims supported by the assistant's evidence?
  2 = everything checkable is in the tool outputs or documents (or is plain general knowledge).
  1 = one unsupported or misread detail that does not change the conclusion.
  0 = invents numbers, facts or quotes, or misreads the evidence in a way that matters.
cites (0-1): 1 if claims taken from evidence carry [n]/[Tn] markers that point at the right
  source, or the answer has no evidence-based claims (e.g. a refusal); else 0.

Judge only what is written. Length and style do not matter. Return one JSON object per item:
{"jid": ..., "correct": 0|1|2, "grounded": 0|1|2, "cites": 0|1, "reason": "<one sentence>"}"""

GRADES = {"correct": (0, 1, 2), "grounded": (0, 1, 2), "cites": (0, 1)}
SCALE = {"correct": 2, "grounded": 2, "cites": 1}


def _jid(qid: str, label: str) -> str:
    return hashlib.sha1(f"{qid}|{label}".encode()).hexdigest()[:12]


def _item(jid: str, q: Mapping[str, Any], r: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "jid": jid,
        "question": q["question"],
        "key_points": q["key_points"],
        "note": q.get("note", ""),
        "answer": r["answer"],
        "tool_outputs": [
            {k: t[k] for k in ("id", "tool", "args", "content")} for t in r.get("tool_calls", [])
        ],
        "documents": [{k: d[k] for k in ("n", "label", "text")} for d in r.get("documents", [])],
    }


def make_batch(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    questions: Sequence[Mapping[str, Any]],
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Blind judge items from one or more runs (lists of answer records), and the key that
    maps each item id back to (qid, label). Failed runs are left out."""
    by_qid = {q["qid"]: q for q in questions}
    items, key = [], {}
    for records in runs:
        for r in records:
            if r.get("error") is not None:
                continue
            jid = _jid(r["qid"], r["label"])
            key[jid] = {"qid": r["qid"], "label": r["label"]}
            items.append(_item(jid, by_qid[r["qid"]], r))
    items.sort(key=lambda i: i["jid"])
    random.Random(seed).shuffle(items)
    return items, key


def load_judgments(path: Path) -> dict[str, dict[str, int]]:
    """jid -> grades; a later line overrides an earlier one."""
    out: dict[str, dict[str, int]] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field, allowed in GRADES.items():
            if row.get(field) not in allowed:
                raise ValueError(f"{path}:{n}: {field} must be one of {allowed}")
        out[row["jid"]] = {f: row[f] for f in GRADES}
    return out


def weighted_kappa(a: Sequence[int], b: Sequence[int], levels: Sequence[int] = (0, 1, 2)) -> float:
    """Cohen's kappa with quadratic weights: 1 = perfect, 0 = chance agreement."""
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    denom = float((w * exp).sum())
    return 1.0 if denom == 0 else 1.0 - float((w * obs).sum()) / denom


def agreement(
    a: Mapping[str, Mapping[str, int]], b: Mapping[str, Mapping[str, int]], field: str
) -> dict[str, Any]:
    """Exact agreement and quadratic-weighted kappa on the items both graded."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}
    x = [a[j][field] for j in shared]
    y = [b[j][field] for j in shared]
    return {
        "n": len(shared),
        "exact": sum(p == q for p, q in zip(x, y, strict=True)) / len(shared),
        "kappa_w": weighted_kappa(x, y, GRADES.get(field, (0, 1, 2))),
    }


def _values(g: Mapping[str, int]) -> dict[str, float]:
    v = {f: g[f] / SCALE[f] for f in GRADES}
    v["pass"] = float(g["correct"] == 2 and g["grounded"] == 2)
    return v


def _summary(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for m in ("correct", "grounded", "cites", "pass"):
        mean, lo, hi = bootstrap_ci([r[m] for r in rows])
        out[m] = {"mean": mean, "lo": lo, "hi": hi}
    return out


def score(
    judgments: Mapping[str, Mapping[str, int]],
    key: Mapping[str, Mapping[str, str]],
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per label: mean grades on a 0-1 scale and the pass rate (correct and fully
    grounded), with bootstrap 95% CIs, overall and per category."""
    cat = {q["qid"]: q["category"] for q in questions}
    rows: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for jid, g in judgments.items():
        if jid not in key:
            continue
        label, qid = key[jid]["label"], key[jid]["qid"]
        v = _values(g)
        rows[label]["__all__"].append(v)
        rows[label][cat[qid]].append(v)
    return {
        label: {
            "overall": _summary(by["__all__"]),
            "by_category": {c: _summary(r) for c, r in sorted(by.items()) if c != "__all__"},
        }
        for label, by in sorted(rows.items())
    }


def _paired_metric(diffs: Sequence[float]) -> dict[str, Any]:
    mean, lo, hi = bootstrap_ci(diffs) if diffs else (0.0, 0.0, 0.0)
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "better": sum(x > 0 for x in diffs),
        "worse": sum(x < 0 for x in diffs),
    }


def paired(
    judgments: Mapping[str, Mapping[str, int]],
    key: Mapping[str, Mapping[str, str]],
    label: str,
    baseline: str,
) -> dict[str, Any]:
    """Question-by-question difference `label` - `baseline` on correct (0-1 scale, as in
    `score`) and pass, with bootstrap CIs, over the questions judged for both."""
    grades: dict[str, dict[str, Mapping[str, int]]] = defaultdict(dict)
    for jid, g in judgments.items():
        if jid in key:
            grades[key[jid]["label"]][key[jid]["qid"]] = g
    shared = sorted(set(grades[label]) & set(grades[baseline]))
    a, b = grades[label], grades[baseline]
    return {
        "n": len(shared),
        **{
            m: _paired_metric([_values(a[q])[m] - _values(b[q])[m] for q in shared])
            for m in ("correct", "pass")
        },
    }
