"""Metadata filters for retrieval, translated to a Qdrant payload filter."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import UTC, date, datetime, time

from qdrant_client import models as qm


@dataclass(frozen=True)
class SearchFilters:
    appids: tuple[int, ...] = ()
    doc_types: tuple[str, ...] = ()  # review | patch_note | news | wiki
    min_playtime_min: int | None = None
    max_playtime_min: int | None = None
    date_from: date | None = None
    date_to: date | None = None
    voted_up: bool | None = None
    language: str | None = None
    text_contains: str | None = None  # full-text match on the chunk body

    def merged(self, overrides: SearchFilters) -> SearchFilters:
        """Fields set in `overrides` (non-empty / non-None) win."""
        changes = {
            f.name: getattr(overrides, f.name)
            for f in fields(self)
            if getattr(overrides, f.name) not in (None, ())
        }
        return replace(self, **changes)

    def to_qdrant(self) -> qm.Filter | None:
        must: list[qm.Condition] = []
        if self.appids:
            must.append(qm.FieldCondition(key="appids", match=qm.MatchAny(any=list(self.appids))))
        if self.doc_types:
            must.append(
                qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=list(self.doc_types)))
            )
        if self.min_playtime_min is not None or self.max_playtime_min is not None:
            must.append(
                qm.FieldCondition(
                    key="playtime_min",
                    range=qm.Range(gte=self.min_playtime_min, lte=self.max_playtime_min),
                )
            )
        if self.date_from or self.date_to:
            must.append(
                qm.FieldCondition(
                    key="created_at",
                    range=qm.Range(gte=_epoch(self.date_from), lte=_epoch(self.date_to, end=True)),
                )
            )
        if self.voted_up is not None:
            must.append(qm.FieldCondition(key="voted_up", match=qm.MatchValue(value=self.voted_up)))
        if self.language:
            must.append(qm.FieldCondition(key="language", match=qm.MatchValue(value=self.language)))
        if self.text_contains:
            must.append(qm.FieldCondition(key="text", match=qm.MatchText(text=self.text_contains)))
        return qm.Filter(must=must) if must else None


def _epoch(d: date | None, end: bool = False) -> int | None:
    if d is None:
        return None
    return int(datetime.combine(d, time.max if end else time.min, tzinfo=UTC).timestamp())
