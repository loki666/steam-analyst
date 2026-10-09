Rubric v1

Grade one answer of a Steam game-analytics assistant. You see the question, the key
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
{"jid": ..., "correct": 0|1|2, "grounded": 0|1|2, "cites": 0|1, "reason": "<one sentence>"}
