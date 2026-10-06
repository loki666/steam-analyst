import json

from steam_game_analyst.eval import synth
from steam_game_analyst.retrieval import chunking
from test_retrieval import NAMES, _news, _reviews, _wiki


def _chunks():
    return (
        chunking.review_chunks(_reviews(), NAMES)
        + chunking.news_chunks(_news(), NAMES)
        + chunking.wiki_chunks(_wiki(), NAMES)
    )


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        return self.replies.pop(0)


def test_sample_chunks_is_stratified_and_deterministic():
    a = synth.sample_chunks(_chunks(), per_type=1, min_words=3, seed=7)
    b = synth.sample_chunks(_chunks(), per_type=1, min_words=3, seed=7)
    assert [c.id for c in a] == [c.id for c in b]
    assert sorted(c.payload["doc_type"] for c in a) == ["patch_note", "review", "wiki"]


def test_sample_chunks_skips_short_chunks():
    picked = synth.sample_chunks(_chunks(), per_type=5, min_words=8, seed=1)
    assert all(len(c.payload["text"].split()) >= 8 for c in picked)


def test_generate_questions_parses_json_and_skips_bad_replies():
    chunks = synth.sample_chunks(_chunks(), per_type=1, min_words=3, seed=7)
    replies = [
        json.dumps({"question": "Q1 about it?", "answer": "A1"}),
        "not json at all",
        'Sure! {"question": "Q3?", "answer": "A3"}',
    ]
    items = synth.generate_questions(chunks, ScriptedLLM(replies), model="m")
    assert [i["question"] for i in items] == ["Q1 about it?", "Q3?"]
    first = items[0]
    assert first["chunk_id"] == chunks[0].id
    assert first["doc_type"] == chunks[0].payload["doc_type"]
    assert first["model"] == "m"
    assert set(first) >= {"qid", "reference_answer", "source_id", "appids", "game"}


def test_split_is_stable_per_question_and_order_independent():
    qids = [f"q{i}" for i in range(200)]
    a = {q: synth.split_for(q) for q in qids}
    b = {q: synth.split_for(q) for q in reversed(qids)}
    assert a == b
    assert set(a.values()) == {"dev", "test"}
    assert 70 <= sum(v == "test" for v in a.values()) <= 130  # roughly half


def test_generated_items_carry_their_split():
    chunks = synth.sample_chunks(_chunks(), per_type=1, min_words=3, seed=7)
    items = synth.generate_questions(
        chunks[:1], ScriptedLLM([json.dumps({"question": "Q?", "answer": "A"})]), model="m"
    )
    assert items[0]["split"] == synth.split_for(items[0]["qid"])
