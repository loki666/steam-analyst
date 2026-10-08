"""UI view models over real agent states (tests/fixtures/ui: captured from `Agent.run` with the
14B model; tool_errors.json rebuilt from a trace line by re-running its tool calls)."""

import json
from pathlib import Path

from steam_game_analyst.agent.state import RouterDecision
from steam_game_analyst.ui import results

FIX = Path(__file__).parent / "fixtures" / "ui"


def load(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text())


def test_parse_content_reads_json_and_salvages_a_truncated_output():
    valheim = load("valheim_trend")["tool_results"][0]["content"]
    assert results.parse_content(valheim)["summary"]["n_reviews"] == 32183
    cut = valheim[: valheim.index('"series"') + 200] + '..."[truncated 900 chars]"'
    out = results.parse_content(cut)
    assert out["_truncated"] is True
    assert out["game"]["name"] == "Valheim" and out["data_through"] == "2024-11-01"
    assert "series" not in out  # only complete fields survive
    assert results.parse_content("not json") == {"_truncated": True}


def test_cited_ids_reads_grouped_and_adjacent_citations_in_order_once():
    text = "A [3, 2] and [T1][2]; again [3]. Range [1-6] is not a citation."
    assert results.cited_ids(text) == ["3", "2", "T1"]


def test_sources_split_cited_documents_from_the_rest():
    cited, rest = results.sources(load("hades_complaints"))
    assert [s.n for s in cited] == [1]
    assert cited[0].label.startswith("Hades II · review · 2024-05-07")
    assert len(rest) == 5 and all(s.n != 1 for s in rest)
    assert len(cited[0].snippet) <= results.SNIPPET_CHARS + 1


def test_sources_keep_urls_and_spell_out_patch_notes():
    cited, rest = results.sources(load("cyberpunk_patch"))  # the answer cites only [T2]
    assert cited == []
    patch = next(s for s in rest if s.n == 3)
    assert patch.label == "Cyberpunk 2077 · patch note · 2023-09-21 · Update 2.0"
    assert patch.url and patch.url.startswith("https://")


def test_snippet_cuts_at_a_word_and_marks_the_cut():
    assert results.snippet("one  two\nthree") == "one two three"
    s = results.snippet("word " * 100, n=22)
    assert s == "word word word word…"


def test_answer_markdown_turns_citations_into_badges():
    md = results.answer_markdown(load("hades_complaints"))
    assert ":gray-badge[T1]" in md and ":blue-badge[1]" in md
    assert "[T1]" not in md.replace(":gray-badge[T1]", "")


def test_answer_markdown_escapes_dollars_and_leaves_unknown_ids():
    state = {**load("reception"), "answer": "At $25 it leans negative [T1], see [T7]."}
    md = results.answer_markdown(state)
    assert r"\$25" in md and "[T7]" in md and ":gray-badge[T1]" in md


def test_a_paragraph_of_only_citations_joins_the_line_above():
    valheim = load("valheim_trend")["answer"]  # ends "...significant change. [T1]" already
    assert results.attach_lone_citations(valheim) == valheim
    text = "Reviews are flat.\n\n[T1]\n\n- bugs fell\n  [2, 3].\nNext line."
    assert results.attach_lone_citations(text) == (
        "Reviews are flat. [T1]\n\n- bugs fell [2, 3].\nNext line."
    )
    assert results.attach_lone_citations("[T1] first") == "[T1] first"
    md = results.answer_markdown({**load("valheim_trend"), "answer": "Flat.\n\n[T1]"})
    assert md == "Flat. :gray-badge[T1]"


def test_refusal_is_read_from_the_route_dict_or_model():
    assert results.is_refusal(load("refusal"))
    assert not results.is_refusal(load("valheim_trend"))
    assert results.is_refusal({"route": RouterDecision(in_scope=False)})
    assert not results.is_refusal({})


def test_tool_rows_list_results_then_refused_calls():
    rows = results.tool_rows(load("tool_errors"))
    assert [r.status for r in rows] == ["no data"] * 3 + ["error"] * 4
    assert rows[0].id == "T1" and "no english reviews" in rows[0].summary
    assert rows[3].tool == "query_catalog" and "BinderException" in rows[3].summary
    assert all(len(r.args) <= results.ARGS_CHARS for r in rows)


def test_tool_summaries_are_compact_and_come_from_the_output():
    valheim = results.tool_rows(load("valheim_trend"))[0]
    assert valheim.status == "ok"
    assert "32,183 reviews" in valheim.summary and "trend flat (p 0.98)" in valheim.summary
    hades = results.tool_rows(load("hades_complaints"))[0].summary
    assert "768 sampled · effective n 523.5 · 95.8% positive" in hades
    starfield = results.tool_rows(load("starfield_no_data"))[0].summary
    assert starfield == "Starfield · no dated reviews: catalog facts only"
    reception = results.tool_rows(load("reception"))[0].summary
    assert reception == "most likely negative or mixed (74.0%)"


def test_catalog_summary_counts_rows_and_flags_no_match():
    out = '{"columns":["appid","name"],"rows":[[1,"a"]],"row_count":1,"truncated":true}'
    assert results.summarize("query_catalog", out) == "1 rows (more rows cut) · appid, name"
    nm = '{"no_match":"nothing matched \'Roguelike\'","columns":[],"rows":[],"row_count":0}'
    assert results.summarize("query_catalog", nm).startswith("no match: nothing matched")
