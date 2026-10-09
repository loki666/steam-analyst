Retrieval relevance rubric v1 (the scale in eval/judged.py, written down for the test run)

You see one question (written by an LLM from one passage of a Steam game's reviews, patch
notes, news or wiki) and a list of candidate passages, each with its game, document type,
title and text. Grade every candidate on its own, for this question:

  2 = answers it: the passage contains what the question asks for (the fact, the opinion,
      the change), for the game the question is about.
  1 = relevant evidence: it partly answers, or gives directly related evidence a good
      answer would use (same topic and game, but incomplete or indirect).
  0 = does not help: off-topic, another game when the question is about a specific one, or
      only shares keywords.

When the question names no game ("the game"), judge by whether the passage would help
answer it for the game of the passage the question was most likely written from (the
candidates usually make it clear). Judge only relevance, not writing quality or length.
Return one JSON object per candidate:
{"qid": ..., "chunk_id": ..., "grade": 0|1|2, "reason": "<at most 12 words>",
 "judge": "claude-subagent", "rubric": "v1"}
