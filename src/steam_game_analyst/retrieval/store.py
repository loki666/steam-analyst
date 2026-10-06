"""Qdrant collection holding every chunk (reviews, patch notes, news, wiki) with its payload."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from steam_game_analyst.retrieval.chunking import Chunk
from steam_game_analyst.retrieval.filters import SearchFilters

DEFAULT_COLLECTION = "sga_chunks"

# payload field -> index type; these are the fields SearchFilters can filter on
_PAYLOAD_INDEXES: dict[str, Any] = {
    "appids": qm.PayloadSchemaType.INTEGER,
    "doc_type": qm.PayloadSchemaType.KEYWORD,
    "language": qm.PayloadSchemaType.KEYWORD,
    "created_at": qm.PayloadSchemaType.INTEGER,
    "playtime_min": qm.PayloadSchemaType.INTEGER,
    "voted_up": qm.PayloadSchemaType.BOOL,
    "text": qm.TextIndexParams(
        type=qm.TextIndexType.TEXT, tokenizer=qm.TokenizerType.WORD, lowercase=True
    ),
}


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str = DEFAULT_COLLECTION) -> None:
        self.client = client
        self.collection = collection

    def ensure(self, dim: int, recreate: bool = False) -> None:
        """Create the collection (and payload indexes) unless it exists; `recreate` drops it first."""
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE, on_disk=True),
            # int8 vectors in RAM, originals on disk: ~4x less memory, rescored on search
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            ),
        )
        with warnings.catch_warnings():  # the in-memory client (tests) ignores indexes
            warnings.simplefilter("ignore", UserWarning)
            for field, schema in _PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection, field, field_schema=schema)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        batch: int = 512,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for start in range(0, len(chunks), batch):
            points = [
                qm.PointStruct(
                    id=c.id, vector=v.tolist(), payload={**c.payload, **(extra_payload or {})}
                )
                for c, v in zip(
                    chunks[start : start + batch], vectors[start : start + batch], strict=True
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: np.ndarray, filters: SearchFilters, k: int = 8) -> list[Hit]:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            query_filter=filters.to_qdrant(),
            limit=k,
            with_payload=True,
        )
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def delete_stale(self, doc_types: Sequence[str], run_id: str) -> None:
        """After a successful re-index: drop chunks of these types not written by `run_id`."""
        selector = qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=list(doc_types)))],
                must_not=[qm.FieldCondition(key="run_id", match=qm.MatchValue(value=run_id))],
            )
        )
        self.client.delete(self.collection, points_selector=selector, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
