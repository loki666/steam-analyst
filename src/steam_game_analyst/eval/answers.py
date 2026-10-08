"""Answer-level evaluation: run the agent on the W5 question set, record what it did and how
long it took, and score its routing against each question's expected tools.

A question lists `required_tools` (the answer needs their numbers) and `allowed_tools`
(also reasonable, e.g. query_catalog to look up a date). Routing is `exact` when the scope
call is right, every required tool was attempted, nothing outside the allowed set was
called, and retrieval ran when the question needs quoted text. Answer quality is judged
separately, from the records written here."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REQUIRED_FIELDS = (
    "qid",
    "split",
    "category",
    "question",
    "in_scope",
    "needs_retrieval",
    "required_tools",
    "allowed_tools",
    "key_points",
)
TOOL_OUTPUT_CHARS = 6000  # what the judge sees of each tool output
DOC_CHARS = 1500  # and of each retrieved document

RunFn = Callable[[str], Mapping[str, Any]]


def load_questions(path: Path) -> list[dict[str, Any]]:
    items = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            raise ValueError(f"{path}:{n}: missing {', '.join(missing)}")
        items.append(item)
    return items


def chosen(items: Sequence[dict[str, Any]], split: str, final: bool) -> list[dict[str, Any]]:
    if split == "test" and not final:
        raise PermissionError("the test split is frozen: evaluate it once, with final=True")
    return [i for i in items if i["split"] == split]


def _attempted(state: Mapping[str, Any]) -> set[str]:
    calls = [*state.get("tool_results", []), *state.get("tool_errors", [])]
    return {c["tool"] for c in calls}


def score_routing(item: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    route = state.get("route")
    answered = route is None or bool(route.in_scope)
    scope_ok = answered == item["in_scope"]
    called = _attempted(state)
    required = set(item["required_tools"])
    missing = sorted(required - called)
    unexpected = sorted(called - set(item["allowed_tools"]) - required)
    recall = 1.0 if not required else len(required & called) / len(required)
    retrieval_ok = not item["needs_retrieval"] or bool(state.get("chunks"))
    return {
        "scope_ok": scope_ok,
        "required_recall": recall,
        "unexpected_tools": unexpected,
        "missing_tools": missing,
        "retrieval_ok": retrieval_ok,
        "exact": scope_ok and not missing and not unexpected and retrieval_ok,
    }


def record(item: Mapping[str, Any], state: Mapping[str, Any], latency_s: float) -> dict[str, Any]:
    """What the judge and the report need from one run: the answer, the evidence it was
    given (tool outputs and documents, clipped) and the routing score."""
    route = state.get("route")
    return {
        "route": route.model_dump() if route is not None else None,
        "tool_calls": [
            {
                "id": t["id"],
                "tool": t["tool"],
                "args": t["args"],
                "ok": t["ok"],
                "content": str(t.get("content", ""))[:TOOL_OUTPUT_CHARS],
            }
            for t in state.get("tool_results", [])
        ],
        "tool_errors": list(state.get("tool_errors", [])),
        "documents": [
            {"n": c["n"], "id": c["id"], "label": c["label"], "text": c["text"][:DOC_CHARS]}
            for c in state.get("chunks", [])
        ],
        "notes": list(state.get("notes", [])),
        "iterations": state.get("iterations", 0),
        "answer": state.get("answer", ""),
        "citations": list(state.get("citations", [])),
        "latency_s": round(latency_s, 2),
        "routing": score_routing(item, state),
        "error": None,
    }


def _done(out: Path) -> dict[str, dict[str, Any]]:
    """Successful records already in `out`, by qid (a failed question is run again)."""
    if not out.exists():
        return {}
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return {r["qid"]: r for r in rows if r.get("error") is None}


def _write(out: Path, results: Mapping[str, dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results.values())
    out.write_text(text)


def run_answers(
    run: RunFn,
    items: Sequence[dict[str, Any]],
    out: Path,
    split: str = "dev",
    final: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run every question of `split` through `run` (the agent), saving `out` after each
    question. Questions already answered in `out` are kept, not re-run, so an interrupted
    run resumes; a question that raised is run again."""
    todo = chosen(items, split, final)
    results = _done(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for item in tqdm([i for i in todo if i["qid"] not in results], desc=f"answers {split}"):
        head = {k: item[k] for k in ("qid", "split", "category", "question")}
        start = time.perf_counter()
        try:
            state = run(item["question"])
            body = record(item, state, time.perf_counter() - start)
        except Exception as e:  # noqa: BLE001 — one failed question must not end the run
            body = {
                "latency_s": round(time.perf_counter() - start, 2),
                "routing": None,
                "error": f"{type(e).__name__}: {e}",
            }
        results[item["qid"]] = {**head, **(meta or {}), **body}
        _write(out, results)
    return [results[i["qid"]] for i in todo if i["qid"] in results]


def _routing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    routing = [r["routing"] for r in rows]
    latency = np.asarray([r["latency_s"] for r in rows], dtype=float)
    return {
        "n": n,
        "exact": sum(x["exact"] for x in routing) / n,
        "scope_ok": sum(x["scope_ok"] for x in routing) / n,
        "required_recall": sum(x["required_recall"] for x in routing) / n,
        "unexpected_rate": sum(bool(x["unexpected_tools"]) for x in routing) / n,
        "retrieval_ok": sum(x["retrieval_ok"] for x in routing) / n,
        "latency_p50_s": float(np.percentile(latency, 50)),
        "latency_p95_s": float(np.percentile(latency, 95)),
        "latency_mean_s": float(latency.mean()),
    }


def to_markdown(summary: Mapping[str, Any], title: str) -> str:
    """Routing and latency table, one row per category plus the overall row."""
    head = (
        "| Category | n | Routing exact | Required recall | Unexpected tool | Scope ok "
        "| p50 s | p95 s |\n|---|---|---|---|---|---|---|---|\n"
    )

    def row(name: str, m: Mapping[str, Any]) -> str:
        if not m.get("n"):
            return f"| {name} | 0 | | | | | | |\n"
        return (
            f"| {name} | {m['n']} | {m['exact']:.0%} | {m['required_recall']:.0%} "
            f"| {m['unexpected_rate']:.0%} | {m['scope_ok']:.0%} "
            f"| {m['latency_p50_s']:.1f} | {m['latency_p95_s']:.1f} |\n"
        )

    rows = "".join(row(c, m) for c, m in summary["by_category"].items())
    failed = f"\n{summary['failed']} of {summary['n']} runs failed.\n" if summary["failed"] else ""
    return f"## {title}\n\n{head}{rows}{row('**all**', summary['overall'])}{failed}"


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Routing accuracy and latency, overall and per category; failed runs are counted
    and left out of the averages."""
    ok = [r for r in records if r.get("error") is None]
    by_cat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    failed: dict[str, int] = defaultdict(int)
    for r in records:
        rows = by_cat[r["category"]]
        if r.get("error") is None:
            rows.append(r)
        else:
            failed[r["category"]] += 1
    return {
        "n": len(records),
        "failed": len(records) - len(ok),
        "overall": _routing_summary(ok),
        "by_category": {
            c: {**_routing_summary(rows), "failed": failed[c]} for c, rows in sorted(by_cat.items())
        },
    }
