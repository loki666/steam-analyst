"""LLM aspect labels for reviews: which of the six aspects a review discusses.

The rubric below is the one the gold labels were made with (Claude labelers on 300 reviews of
held-out games); a local model applies it to reviews of training games to make silver labels
for the aspect classifier. The prompt is written once and not tuned on the gold set."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from steam_game_analyst.ml.aspects import ASPECTS
from steam_game_analyst.retrieval.rag import ChatModel

MAX_CHARS = 2000  # the median review is ~140 characters; long ones are cut

RUBRIC = """An aspect is present when the review says something about it - praise, complaint, or a \
neutral remark. Sentiment does not matter, only whether the aspect is discussed. A review can \
have zero, one or several aspects.
- performance: technical performance - frame rate/FPS, stutter, lag (including network \
latency), loading times, optimisation, hardware requirements or how well it runs on some \
hardware (e.g. Steam Deck).
- bugs: bugs, glitches, crashes, errors, broken or non-working features, save loss, \
server/connection failures, cheaters exploiting glitches.
- gameplay: how the game plays - mechanics, combat, controls of strategy/systems, difficulty, \
progression, level/quest/content design, game modes, multiplayer, replayability, amount of \
content, grind. A bare "fun" or "good game" with nothing about how it plays does NOT count.
- story: narrative - plot, characters, writing, dialogue, lore, setting/world-building, \
ending, voice acting.
- price: price or value for money - cost, "worth it", sales/discounts, refunds, DLC or \
microtransaction pricing, pay-to-win monetisation, free-to-play model.
- controls: input - control scheme, keybinds, mouse/keyboard, controller/gamepad support, \
camera control, input responsiveness.
Not aspects here (ignore): graphics/art, music/sound, UI/menus (unless about controls), \
community, developers' behaviour, nostalgia."""

SYSTEM = (
    "You label Steam game reviews with the aspects of the game they talk about.\n\n"
    + RUBRIC
    + '\n\nAnswer with JSON only: {"aspects": [...]}, using the names performance, bugs, '
    "gameplay, story, price, controls."
)
_OBJECT = re.compile(r"\{[^{}]*\}", re.S)


def _parse(reply: str) -> list[str] | None:
    for m in _OBJECT.finditer(reply):
        try:
            found = json.loads(m.group(0)).get("aspects")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(found, list):
            names = {str(a).strip().lower() for a in found}
            return [a for a in ASPECTS if a in names]
    return None


def label_review(text: str, llm: ChatModel, game: str | None = None) -> list[str]:
    """Aspects the review discusses, in ASPECTS order. One retry on an unparseable reply."""
    about = f"Game: {game}\n" if game else ""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{about}Review:\n{text[:MAX_CHARS]}"},
    ]
    for _ in range(2):
        parsed = _parse(llm.chat(messages))
        if parsed is not None:
            return parsed
    raise ValueError("the model returned no aspect list twice")


def label_to_file(
    rows: Sequence[Mapping[str, Any]], llm: ChatModel, out: Path, progress: bool = False
) -> int:
    """Label each {review_id, game, text} row, appending {review_id, aspects} lines to `out`.
    Rows already in `out` are skipped, so an interrupted run resumes. Returns rows labelled."""
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["review_id"] for line in out.read_text().splitlines() if line}
    todo = [r for r in rows if r["review_id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in tqdm(todo, desc="aspect labels", disable=not progress):
            aspects = label_review(r["text"], llm, game=r.get("game"))
            f.write(json.dumps({"review_id": r["review_id"], "aspects": aspects}) + "\n")
            f.flush()
    return len(todo)
