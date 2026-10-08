import json

import pytest

from steam_game_analyst.eval import answer_judge as aj

QUESTIONS = [
    {"qid": "a", "category": "catalog", "question": "qa", "key_points": ["ka"]},
    {"qid": "b", "category": "rag", "question": "qb", "key_points": ["kb"], "note": "nb"},
]


def _rec(qid, label, answer="ans", error=None):
    return {
        "qid": qid,
        "label": label,
        "model": "m",
        "retrieval_mode": "hybrid",
        "answer": answer,
        "tool_calls": [
            {"id": "T1", "tool": "query_catalog", "args": {}, "ok": True, "content": "{}"}
        ],
        "documents": [{"n": 1, "id": "c1", "label": "doc", "text": "text"}],
        "error": error,
    }


def test_batch_is_blind_and_carries_the_evidence():
    runs = [[_rec("a", "hybrid"), _rec("b", "hybrid")], [_rec("a", "rerank"), _rec("b", "rerank")]]
    items, key = aj.make_batch(runs, QUESTIONS, seed=1)
    assert len(items) == 4 and len(key) == 4
    text = json.dumps(items)
    for leak in ("hybrid", "rerank", '"m"', "qid"):
        assert leak not in text
    b = next(i for i in items if key[i["jid"]] == {"qid": "b", "label": "hybrid"})
    assert b["key_points"] == ["kb"] and b["note"] == "nb"
    assert b["tool_outputs"][0]["tool"] == "query_catalog"
    assert b["documents"][0]["text"] == "text"


def test_batch_skips_failed_runs():
    items, key = aj.make_batch([[_rec("a", "x"), _rec("b", "x", error="boom")]], QUESTIONS)
    assert [key[i["jid"]]["qid"] for i in items] == ["a"]


def test_batch_order_is_deterministic_per_seed():
    runs = [[_rec("a", "x"), _rec("b", "x")]]
    assert aj.make_batch(runs, QUESTIONS, seed=3) == aj.make_batch(runs, QUESTIONS, seed=3)


def test_load_judgments_validates_grades(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"jid": "j1", "correct": 3, "grounded": 2, "cites": 1}) + "\n")
    with pytest.raises(ValueError, match="correct"):
        aj.load_judgments(p)
    first = {"jid": "j1", "correct": 2, "grounded": 1, "cites": 0}
    later = {"jid": "j1", "correct": 1, "grounded": 1, "cites": 0, "reason": "r"}
    p.write_text(json.dumps(first) + "\n" + json.dumps(later) + "\n")
    assert aj.load_judgments(p) == {"j1": {"correct": 1, "grounded": 1, "cites": 0}}


def test_weighted_kappa_perfect_and_chance():
    assert aj.weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert aj.weighted_kappa([0, 2, 0, 2], [0, 0, 2, 2]) == pytest.approx(0.0)


def test_weighted_kappa_penalises_far_disagreement_more():
    near = aj.weighted_kappa([0, 1, 2, 2, 0], [0, 2, 2, 1, 0])
    far = aj.weighted_kappa([0, 1, 2, 2, 0], [2, 1, 0, 2, 0])
    assert near > far


def test_agreement_on_shared_items():
    a = {"j1": {"correct": 2}, "j2": {"correct": 0}, "j3": {"correct": 1}}
    b = {"j1": {"correct": 2}, "j2": {"correct": 1}, "j9": {"correct": 0}}
    r = aj.agreement(a, b, "correct")
    assert r["n"] == 2 and r["exact"] == pytest.approx(0.5)


def test_score_by_label_and_category():
    key = {
        "j1": {"qid": "a", "label": "x"},
        "j2": {"qid": "b", "label": "x"},
        "j3": {"qid": "a", "label": "y"},
    }
    judgments = {
        "j1": {"correct": 2, "grounded": 2, "cites": 1},
        "j2": {"correct": 1, "grounded": 2, "cites": 1},
        "j3": {"correct": 0, "grounded": 1, "cites": 0},
    }
    s = aj.score(judgments, key, QUESTIONS)
    assert s["x"]["overall"]["n"] == 2
    # grades are reported on a 0-1 scale: (2 + 1) / 2 / 2
    assert s["x"]["overall"]["correct"]["mean"] == pytest.approx(0.75)
    assert s["x"]["overall"]["pass"]["mean"] == pytest.approx(0.5)
    assert s["x"]["by_category"]["rag"]["n"] == 1
    assert s["y"]["overall"]["pass"]["mean"] == 0.0


def test_paired_difference_only_on_shared_questions():
    key = {
        "j1": {"qid": "a", "label": "x"},
        "j2": {"qid": "b", "label": "x"},
        "j3": {"qid": "a", "label": "y"},
    }
    judgments = {k: {"correct": 2, "grounded": 2, "cites": 1} for k in key}
    judgments["j3"] = {"correct": 0, "grounded": 2, "cites": 1}
    d = aj.paired(judgments, key, "y", "x")
    assert d["n"] == 1
    assert d["correct"]["mean"] == pytest.approx(-1.0)
    assert d["correct"]["worse"] == 1 and d["correct"]["better"] == 0
