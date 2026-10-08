import json

import pytest

from steam_game_analyst.agent.state import RouterDecision
from steam_game_analyst.eval import answers as ev

ITEMS = [
    {
        "qid": "c1",
        "split": "dev",
        "category": "catalog",
        "question": "cheapest?",
        "in_scope": True,
        "needs_retrieval": False,
        "required_tools": ["query_catalog"],
        "allowed_tools": ["query_catalog"],
        "key_points": ["x"],
    },
    {
        "qid": "p1",
        "split": "dev",
        "category": "patch",
        "question": "did the patch help?",
        "in_scope": True,
        "needs_retrieval": True,
        "required_tools": ["review_sentiment", "review_trend"],
        "allowed_tools": ["query_catalog", "review_sentiment", "review_trend"],
        "key_points": ["y"],
    },
    {
        "qid": "o1",
        "split": "dev",
        "category": "out_of_scope",
        "question": "banana bread?",
        "in_scope": False,
        "needs_retrieval": False,
        "required_tools": [],
        "allowed_tools": [],
        "key_points": ["declines"],
    },
    {
        "qid": "t1",
        "split": "test",
        "category": "catalog",
        "question": "frozen",
        "in_scope": True,
        "needs_retrieval": False,
        "required_tools": ["query_catalog"],
        "allowed_tools": ["query_catalog"],
        "key_points": ["z"],
    },
]


def _state(question, tools=(), in_scope=True, chunks=0, errors=()):
    return {
        "question": question,
        "route": RouterDecision(in_scope=in_scope),
        "tool_results": [
            {"id": f"T{i}", "tool": t, "args": {}, "ok": True, "content": '{"rows": []}'}
            for i, t in enumerate(tools, 1)
        ],
        "tool_errors": [{"tool": t, "args": {}, "error": "bad"} for t in errors],
        "chunks": [
            {"n": i, "id": f"c{i}", "label": f"doc {i}", "text": "t" * 50, "url": None}
            for i in range(1, chunks + 1)
        ],
        "answer": "an answer [1]",
        "citations": [],
        "notes": [],
        "iterations": 1,
    }


def test_routing_exact_when_required_called_and_nothing_unexpected():
    r = ev.score_routing(ITEMS[0], _state("q", ["query_catalog"]))
    assert r == {
        "scope_ok": True,
        "required_recall": 1.0,
        "unexpected_tools": [],
        "missing_tools": [],
        "retrieval_ok": True,
        "exact": True,
    }


def test_routing_counts_missing_and_unexpected_tools():
    r = ev.score_routing(ITEMS[1], _state("q", ["review_trend", "predict_reception"], chunks=3))
    assert r["required_recall"] == 0.5
    assert r["missing_tools"] == ["review_sentiment"]
    assert r["unexpected_tools"] == ["predict_reception"]
    assert r["exact"] is False


def test_refused_tool_calls_count_as_attempts():
    r = ev.score_routing(ITEMS[0], _state("q", errors=["query_catalog"]))
    assert r["required_recall"] == 1.0


def test_retrieval_needed_but_skipped_is_not_exact():
    r = ev.score_routing(ITEMS[1], _state("q", ["review_trend", "review_sentiment"], chunks=0))
    assert r["retrieval_ok"] is False
    assert r["exact"] is False


def test_out_of_scope_must_be_refused():
    assert ev.score_routing(ITEMS[2], _state("q", in_scope=False))["exact"] is True
    answered = ev.score_routing(ITEMS[2], _state("q", in_scope=True))
    assert answered["scope_ok"] is False and answered["exact"] is False


def test_in_scope_refusal_fails_scope():
    r = ev.score_routing(ITEMS[0], _state("q", in_scope=False))
    assert r["scope_ok"] is False and r["exact"] is False


def test_test_split_is_frozen():
    with pytest.raises(PermissionError):
        ev.chosen(ITEMS, "test", final=False)
    assert [i["qid"] for i in ev.chosen(ITEMS, "test", final=True)] == ["t1"]


def test_run_records_answers_routing_and_latency(tmp_path):
    out = tmp_path / "answers.jsonl"
    tools = {"cheapest?": ["query_catalog"], "did the patch help?": ["review_trend"]}

    def run(question):
        if question == "banana bread?":
            return _state(question, in_scope=False)
        return _state(question, tools[question], chunks=2)

    records = ev.run_answers(run, ITEMS, out, split="dev", meta={"model": "m"})
    assert [r["qid"] for r in records] == ["c1", "p1", "o1"]
    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert saved == records
    first = saved[0]
    assert first["model"] == "m" and first["answer"] == "an answer [1]"
    assert first["latency_s"] >= 0 and first["error"] is None
    assert first["tool_calls"][0]["content"] == '{"rows": []}'
    assert first["documents"][0]["text"].startswith("t")
    assert saved[1]["routing"]["missing_tools"] == ["review_sentiment"]


def test_run_keeps_going_after_a_failed_question(tmp_path):
    def run(question):
        if question == "cheapest?":
            raise ConnectionError("ollama down")
        return _state(question, in_scope=question != "banana bread?")

    records = ev.run_answers(run, ITEMS, tmp_path / "a.jsonl", split="dev")
    assert records[0]["error"] == "ConnectionError: ollama down"
    assert records[0]["routing"] is None
    assert records[1]["error"] is None


def test_run_resumes_without_rerunning_done_questions(tmp_path):
    out = tmp_path / "a.jsonl"
    calls = []

    def run(question):
        calls.append(question)
        return _state(question, in_scope=question != "banana bread?")

    ev.run_answers(run, ITEMS[:1], out, split="dev")
    records = ev.run_answers(run, ITEMS, out, split="dev")
    assert calls == ["cheapest?", "did the patch help?", "banana bread?"]
    assert len(records) == 3 and len(out.read_text().splitlines()) == 3


def test_failed_questions_are_retried_on_resume(tmp_path):
    out = tmp_path / "a.jsonl"
    state = {"fail": True}

    def run(question):
        if state["fail"]:
            raise TimeoutError("slow")
        return _state(question, ["query_catalog"])

    ev.run_answers(run, ITEMS[:1], out, split="dev")
    state["fail"] = False
    records = ev.run_answers(run, ITEMS[:1], out, split="dev")
    assert len(records) == 1 and records[0]["error"] is None


def test_summary_routing_and_latency():
    ok = {"scope_ok": True, "retrieval_ok": True}
    records = [
        {
            "category": "catalog",
            "latency_s": 2.0,
            "error": None,
            "routing": {**ok, "required_recall": 1.0, "exact": True, "unexpected_tools": []},
        },
        {
            "category": "catalog",
            "latency_s": 4.0,
            "error": None,
            "routing": {
                **ok,
                "required_recall": 0.5,
                "exact": False,
                "unexpected_tools": ["predict_reception"],
            },
        },
        {"category": "patch", "latency_s": 10.0, "error": "boom", "routing": None},
    ]
    s = ev.summarize(records)
    assert s["n"] == 3 and s["failed"] == 1
    assert s["overall"]["exact"] == pytest.approx(0.5)
    assert s["overall"]["required_recall"] == pytest.approx(0.75)
    assert s["overall"]["unexpected_rate"] == pytest.approx(0.5)
    assert s["overall"]["latency_p50_s"] == pytest.approx(3.0)
    assert s["by_category"]["catalog"]["n"] == 2
    assert s["by_category"]["patch"] == {"n": 0, "failed": 1}


def test_load_questions_rejects_missing_fields(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"qid": "a", "question": "q"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        ev.load_questions(p)
