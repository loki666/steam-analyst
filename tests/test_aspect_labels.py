import pytest

from steam_game_analyst.ml import aspect_labels as al


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(messages)
        return self.replies.pop(0)


def test_prompt_carries_the_rubric_and_the_review():
    llm = FakeLLM(['{"aspects": ["bugs"]}'])
    al.label_review("Crashes every hour.", llm, game="Hades")
    system, user = llm.prompts[0]
    assert "performance:" in system["content"] and "controls:" in system["content"]
    assert "Crashes every hour." in user["content"] and "Hades" in user["content"]


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"aspects": ["bugs", "price"]}', ["bugs", "price"]),
        ('Sure! ```json\n{"aspects": ["story"]}\n```', ["story"]),
        ('{"aspects": ["Gameplay", "graphics", "gameplay"]}', ["gameplay"]),
        ('{"aspects": []}', []),
    ],
)
def test_reply_parsing_keeps_known_aspects_once(reply, expected):
    assert al.label_review("text", FakeLLM([reply])) == expected


def test_unparseable_reply_is_retried_then_reported():
    llm = FakeLLM(["no idea", '{"aspects": ["controls"]}'])
    assert al.label_review("text", llm) == ["controls"]
    with pytest.raises(ValueError, match="no aspect list"):
        al.label_review("text", FakeLLM(["nope", "still nope"]))


def test_long_reviews_are_cut_before_prompting():
    llm = FakeLLM(['{"aspects": []}'])
    al.label_review("word " * 5000, llm)
    assert len(llm.prompts[0][1]["content"]) < al.MAX_CHARS + 500


def test_label_file_resumes_where_it_stopped(tmp_path):
    out = tmp_path / "silver.jsonl"
    rows = [{"review_id": f"r{i}", "game": "G", "text": f"text {i}"} for i in range(3)]
    al.label_to_file(rows[:2], FakeLLM(['{"aspects": ["bugs"]}'] * 2), out)
    llm = FakeLLM(['{"aspects": ["price"]}'])
    al.label_to_file(rows, llm, out)
    assert len(llm.prompts) == 1  # only r2 was new
    lines = [__import__("json").loads(x) for x in out.read_text().splitlines()]
    assert [(x["review_id"], x["aspects"]) for x in lines] == [
        ("r0", ["bugs"]),
        ("r1", ["bugs"]),
        ("r2", ["price"]),
    ]
