"""Citation ids in answer text, shared by the agent, `sga ask` and the UI."""

from __future__ import annotations

import re

# one bracket can hold several ids: "[3]", "[T2]", "[3, 2]", "[T1][2]"
CITE_GROUP = re.compile(r"\[((?:T?\d+)(?:\s*,\s*T?\d+)*)\]")


def cited_ids(text: str) -> list[str]:
    """Ids the text cites, in reading order, each once ("T2", "4", ...)."""
    seen: dict[str, None] = {}
    for group in CITE_GROUP.findall(text or ""):
        for ref in group.split(","):
            seen.setdefault(ref.strip(), None)
    return list(seen)
