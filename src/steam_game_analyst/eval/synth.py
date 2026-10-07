"""Synthetic half of the retrieval gold set: sample chunks, have the local LLM write one
question each chunk answers. Every pair is hand-checked before it counts (PRD: Evaluation)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from tqdm import tqdm

from steam_game_analyst.retrieval.chunking import Chunk, approx_token_count
from steam_game_analyst.retrieval.rag import ChatModel

console = Console()

PROMPT = """Here is a passage from a source about the PC game "{game}" ({doc_type}).

<passage>
{text}
</passage>

Write ONE question that a player or games-industry analyst could ask, which this passage \
answers specifically. Write it as someone who has never seen the passage: name the game, and \
when the question is about a patch, event or character, name that too. Never refer to the \
source itself: no "the passage", "the review", "the reviewer", "the player", "the author", \
"this update" or "the latest patch". Write the question in English whatever the passage's \
language. Reply with JSON only: {{"question": "...", "answer": "..."}} where the answer is one \
or two sentences from the passage, in English."""

_JSON = re.compile(r"\{.*\}", re.S)


def split_for(qid: str) -> str:
    """Frozen dev/test assignment from the question id alone: adding or regenerating other
    questions never moves one across. Tune on dev only; report test once (PRD: Evaluation)."""
    return "test" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 else "dev"


def sample_chunks(
    chunks: Sequence[Chunk], per_type: int, min_words: int = 40, seed: int = 13
) -> list[Chunk]:
    """Up to `per_type` chunks per doc_type, skipping short ones (CJK text counted per
    character, so it is not dropped for lacking spaces); one chunk per game per type until
    every game has had a turn, so questions spread across the catalog."""
    by_type: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if approx_token_count(c.payload["text"]) >= min_words:
            by_type[c.payload["doc_type"]].append(c)
    rng = random.Random(seed)
    picked: list[Chunk] = []
    for doc_type in sorted(by_type):
        pool = sorted(by_type[doc_type], key=lambda c: c.id)
        rng.shuffle(pool)
        first_per_game: list[Chunk] = []
        rest: list[Chunk] = []
        seen: set[str] = set()
        for c in pool:
            (rest if c.payload["game"] in seen else first_per_game).append(c)
            seen.add(c.payload["game"])
        picked += (first_per_game + rest)[:per_type]
    return picked


def _parse(reply: str) -> dict[str, str] | None:
    m = _JSON.search(reply)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q = str(obj.get("question", "")).strip()
    return {"question": q, "answer": str(obj.get("answer", "")).strip()} if q else None


def generate_questions(chunks: Sequence[Chunk], llm: ChatModel, model: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for c in tqdm(chunks, desc="synth questions"):
        p = c.payload
        prompt = PROMPT.format(game=p["game"], doc_type=p["doc_type"], text=p["text"][:2500])
        parsed = _parse(llm.chat([{"role": "user", "content": prompt}]))
        if parsed is None:
            continue
        qid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synth:{c.id}"))
        items.append(
            {
                "qid": qid,
                "split": split_for(qid),
                "question": parsed["question"],
                "reference_answer": parsed["answer"],
                "chunk_id": c.id,
                "source_id": p["source_id"],
                "doc_type": p["doc_type"],
                "appids": p["appids"],
                "game": p["game"],
                "model": model,
                "checked": False,  # flipped by hand during review (W4)
            }
        )
    console.print(f"synth: {len(items)}/{len(chunks)} questions generated")
    return items
