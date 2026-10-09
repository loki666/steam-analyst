#!/usr/bin/env python
"""Held-out check of tools/events.match_patch_titles:  uv run python scripts/validate_events.py

Uses the news of games that no answer-eval question touches (named in a question or called
in its reference tool calls), samples patch notes with a fixed seed and asks one templated
question per note:
  version: "Did reviews of {game} change after patch {version}?" -> first pick should be the
           earliest note carrying that exact version (a strict rule: an announcement can be
           earlier than the release)
  named:   "How did players react to the {name} update for {game}?" -> first pick should
           carry the update's name ({name} comes from titles like "{name} Update")."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Any

import duckdb

from steam_game_analyst.agent.factory import game_index
from steam_game_analyst.config import get_settings
from steam_game_analyst.tools.events import _tokens, match_patch_titles

SEED = 13
N_CASES = 50
VERSION = re.compile(r"(?<![\w.#])(\d+\.\d+(?:\.\d+)*)(?![\w.])")
NAMED = re.compile(r"^(?:.*?[-:|]\s*)?([A-Z][\w'&]+(?:\s+[A-Z][\w'&]+){0,3})\s+[Uu]pdate\b")
GENERIC_NAMES = {"major", "new", "big", "free"}

News = dict[tuple[int, str], list[tuple[Any, str, bool]]]


def held_out_news() -> News:
    s = get_settings()
    idx = game_index(s.processed_dir)
    excluded: set[int] = set()
    for line in (s.data_dir / "eval" / "answer_questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        excluded |= {a for a, _ in idx.find(q["question"])}
        excluded |= {c.get("args", {}).get("appid") for c in q.get("reference_calls") or []}
    con = duckdb.connect(str(s.duckdb_path), read_only=True)
    rows = con.execute(
        "SELECT n.appid, g.name, CAST(n.published_at AS DATE), n.title, n.is_patch_note "
        "FROM news n JOIN games g USING (appid)"
    ).fetchall()
    news: News = defaultdict(list)
    for appid, name, day, title, is_patch in rows:
        if appid not in excluded:
            news[(appid, name)].append((day, title, is_patch))
    print(f"{len(news)} held-out games")
    return news


def check_versions(news: News, rng: random.Random) -> None:
    cases = [
        (name, items, m.group(1))
        for (_, name), items in news.items()
        for _, title, is_patch in items
        if is_patch and (m := VERSION.search(title))
    ]
    picked = rng.sample(cases, min(N_CASES, len(cases)))
    misses = []
    for name, items, version in picked:
        q = f"Did reviews of {name} change after patch {version}?"
        truth = min(d for d, t, _ in items if version in _tokens(t))
        hits = match_patch_titles(q, name, items)
        if not hits or hits[0][0] != truth:
            misses.append((q, hits[:1], truth))
    print(f"version: {len(picked) - len(misses)}/{len(picked)} first pick = earliest note")
    for miss in misses:
        print("   miss:", miss)


def check_named(news: News, rng: random.Random) -> None:
    cases = []
    for (_, name), items in news.items():
        name_tokens = set(_tokens(name)) | GENERIC_NAMES
        for _, title, is_patch in items:
            m = NAMED.search(title)
            if is_patch and m and not set(_tokens(m.group(1))) <= name_tokens:
                cases.append((name, items, m.group(1)))
    picked = rng.sample(cases, min(N_CASES, len(cases)))
    hits_ok = patch_ok = 0
    misses = []
    for name, items, update in picked:
        q = f"How did players react to the {update} update for {name}?"
        hits = match_patch_titles(q, name, items)
        if hits and set(_tokens(update)) <= set(_tokens(hits[0][1])):
            hits_ok += 1
            patch_ok += any(p for d, t, p in items if (d, t) == hits[0])
        else:
            misses.append((q, hits[:1]))
    print(f"named: {hits_ok}/{len(picked)} first pick carries the name, {patch_ok} patch notes")
    for miss in misses:
        print("   miss:", miss)


if __name__ == "__main__":
    news = held_out_news()
    rng = random.Random(SEED)
    check_versions(news, rng)
    check_named(news, rng)
