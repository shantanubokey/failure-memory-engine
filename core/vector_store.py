"""
Vector Store — Qdrant-backed failure memory store.
Supports payload filtering by error_type, decay_score, cluster_id.
Falls back to in-memory store for local testing.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from core.schema import FailureCase, ErrorType


class InMemoryVectorStore:
    """Lightweight in-memory store for testing without Qdrant."""

    def __init__(self):
        self._cases: Dict[str, FailureCase] = {}

    def upsert(self, case: FailureCase):
        self._cases[case.id] = case

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.82,
        filter_error_type: Optional[ErrorType] = None,
    ) -> List[Tuple[FailureCase, float]]:
        if not self._cases:
            return []

        q = np.array(query_embedding)
        results = []

        for case in self._cases.values():
            if case.embedding is None:
                continue
            if filter_error_type and case.error_type != filter_error_type:
                continue

            v   = np.array(case.embedding)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))

            # Apply decay re-ranking
            adjusted = sim * case.decay_score * case.severity
            if sim >= min_similarity:
                results.append((case, sim, adjusted))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(c, s) for c, s, _ in results[:top_k]]

    def delete(self, case_id: str):
        self._cases.pop(case_id, None)

    def get_all(self) -> List[FailureCase]:
        return list(self._cases.values())

    def count(self) -> int:
        return len(self._cases)


class QdrantVectorStore:
    """Production Qdrant-backed store."""

    COLLECTION = "fme_failures"

    def __init__(self, host: str = "localhost", port: int = 6333):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self._client = QdrantClient(host=host, port=port)
            self._ensure_collection()
            print(f"[VectorStore] Connected to Qdrant at {host}:{port}")
        except ImportError:
            raise ImportError("pip install qdrant-client")

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION not in existing:
            self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def upsert(self, case: FailureCase):
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self.COLLECTION,
            points=[PointStruct(
                id=case.id,
                vector=case.embedding,
                payload=case.to_dict(),
            )],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.82,
        filter_error_type: Optional[ErrorType] = None,
    ) -> List[Tuple[FailureCase, float]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filter_error_type:
            query_filter = Filter(must=[
                FieldCondition(key="error_type",
                               match=MatchValue(value=filter_error_type.value))
            ])

        hits = self._client.search(
            collection_name=self.COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=min_similarity,
            query_filter=query_filter,
        )

        results = []
        for hit in hits:
            p = hit.payload
            case = FailureCase(**{k: v for k, v in p.items()
                                  if k in FailureCase.__dataclass_fields__})
            results.append((case, hit.score))
        return results
