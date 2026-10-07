"""Qdrant collection holding every chunk (reviews, patch notes, news, wiki) with its payload.

Each point has a dense vector (bge-m3) and a BM25 sparse vector; search fuses the two rankings
with reciprocal rank fusion (RRF), or uses either one alone for comparison."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from steam_game_analyst.retrieval import sparse
from steam_game_analyst.retrieval.chunking import Chunk
from steam_game_analyst.retrieval.filters import SearchFilters

DEFAULT_COLLECTION = "sga_chunks"
DENSE = "dense"
BM25 = "bm25"
# candidates each leg passes to RRF, at least this many and at least the requested k
MIN_PREFETCH = 50

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
            self.check_layout()
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE, on_disk=True)
            },
            # Qdrant supplies the IDF from collection statistics at query time
            sparse_vectors_config={BM25: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            # int8 vectors in RAM, originals on disk: ~4x less memory, rescored on search
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            ),
        )
        with warnings.catch_warnings():  # the in-memory client (tests) ignores indexes
            warnings.simplefilter("ignore", UserWarning)
            for field, schema in _PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection, field, field_schema=schema)

    def check_layout(self) -> None:
        """Fail early, with the fix, on a collection built before hybrid search."""
        params = self.client.get_collection(self.collection).config.params
        dense = params.vectors if isinstance(params.vectors, dict) else {}
        if DENSE not in dense or BM25 not in (params.sparse_vectors or {}):
            raise RuntimeError(
                f"collection {self.collection!r} predates hybrid search (no {DENSE!r} + {BM25!r} "
                "vectors); rebuild it with `sga index build --recreate`"
            )

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
                    id=c.id,
                    vector={DENSE: v.tolist(), BM25: sparse.encode_document(c.text)},
                    payload={**c.payload, **(extra_payload or {})},
                )
                for c, v in zip(
                    chunks[start : start + batch], vectors[start : start + batch], strict=True
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(
        self,
        vector: np.ndarray | None,
        filters: SearchFilters,
        k: int = 8,
        text: str | None = None,
    ) -> list[Hit]:
        """Dense search with `vector`, BM25 with `text`, RRF-fused hybrid with both."""
        if vector is None and text is None:
            raise ValueError("search needs a vector, a text or both")
        flt = filters.to_qdrant()
        legs = []
        if vector is not None:
            legs.append((vector.tolist(), DENSE))
        if text is not None:
            legs.append((sparse.encode_query(text), BM25))
        if len(legs) == 1:
            query, using = legs[0]
            res = self.client.query_points(
                self.collection,
                query=query,
                using=using,
                query_filter=flt,
                limit=k,
                with_payload=True,
            )
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    qm.Prefetch(query=q, using=u, filter=flt, limit=max(k, MIN_PREFETCH))
                    for q, u in legs
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def fetch(self, ids: Sequence[str]) -> list[Hit]:
        """Chunks by id (missing ids are skipped), score 0."""
        points = self.client.retrieve(self.collection, ids=list(ids), with_payload=True)
        return [Hit(str(p.id), 0.0, dict(p.payload or {})) for p in points]

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
