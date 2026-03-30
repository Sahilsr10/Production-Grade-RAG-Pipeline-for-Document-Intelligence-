"""
Hybrid retriever: Dense (BGE + Qdrant ANN) + Sparse (BM25) fused via RRF.

Key change from OpenAI version:
- embed() now passes is_query=True so BGE prepends the instruction prefix
  at query time (not at index time — asymmetric encoding)
- Everything else is identical
"""

from __future__ import annotations


from qdrant_client.models import Filter, FieldCondition, MatchValue
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient

from src.ingestion.embedder import BM25Index, DenseEmbedder
from src.retrieval.query_transformer import QueryTransformer, TransformedQuery


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """
    Retrieves candidates using dense + sparse search, fused via RRF.

    Args:
        qdrant_url: Qdrant instance URL
        collection_name: Qdrant collection to search
        bm25_index_path: Path to persisted BM25 index
        embed_model: BGE model name (must match what was used during ingestion)
        top_k_dense: How many results to fetch from dense search
        top_k_sparse: How many results to fetch from BM25
        top_k_fused: How many results to return after fusion
        rrf_k: RRF constant (60 is standard)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "test_docs",
        bm25_index_path: str = "data/processed/bm25.pkl",
        embed_model: str = "BAAI/bge-base-en-v1.5",
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        top_k_fused: int = 50,
        rrf_k: int = 60,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_fused = top_k_fused
        self.rrf_k = rrf_k

        self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._embedder = DenseEmbedder(model_name=embed_model)
        self._bm25 = BM25Index()
        self._bm25.load(bm25_index_path)

    def retrieve(
        self,
        transformed_query: TransformedQuery,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        queries = transformed_query.queries_for_retrieval
        logger.debug(f"Retrieving for {len(queries)} query variants")

        all_dense: List[List[tuple[str, float]]] = []
        all_sparse: List[List[tuple[str, float]]] = []
        chunk_texts: Dict[str, tuple[str, dict]] = {}

        qdrant_filter = self._build_qdrant_filter(metadata_filter)

        for query_text in queries:
            dense_results, payloads = self._dense_search_with_payload(
                query_text, qdrant_filter
            )
            all_dense.append([(cid, score) for cid, score, _, _ in dense_results])
            all_sparse.append(self._sparse_search(query_text))

            for chunk_id, _, text, meta in dense_results:
                chunk_texts[chunk_id] = (text, meta)

        dense_flat = self._merge_ranked_lists(all_dense)
        sparse_flat = self._merge_ranked_lists(all_sparse)
        fused = self._rrf_fusion(dense_flat, sparse_flat)

        results = []
        for chunk_id, rrf_score in fused[: self.top_k_fused]:
            text, meta = chunk_texts.get(chunk_id, ("", {}))
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                rrf_score=rrf_score,
                dense_score=0.0,
                sparse_score=0.0,
            ))

        logger.debug(f"Hybrid retrieval returned {len(results)} candidates")
        return results

    def _dense_search_with_payload(
        self, query: str, filter_: Optional[Filter]
    ) -> tuple[List[tuple[str, float, str, dict]], None]:
        # is_query=True → BGE instruction prefix applied at query time
        embedding = self._embedder.embed_single(query, is_query=True)
        results = self._qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=filter_,
            limit=self.top_k_dense,
            with_payload=True
        )
        out = []
        for r in results.points:
            payload = dict(r.payload or {})
            text = payload.get("text", "") 
            chunk_id = payload.get("chunk_id", str(r.id))
            out.append((chunk_id, r.score, text, payload))
        return out, None

    def _sparse_search(self, query: str) -> List[tuple[str, float]]:
        return self._bm25.search(query, top_k=self.top_k_sparse)

    def _merge_ranked_lists(
        self, ranked_lists: List[List[tuple[str, float]]]
    ) -> List[tuple[str, float]]:
        best: Dict[str, float] = {}
        for ranked in ranked_lists:
            for chunk_id, score in ranked:
                if chunk_id not in best or score > best[chunk_id]:
                    best[chunk_id] = score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    def _rrf_fusion(
        self,
        dense_results: List[tuple[str, float]],
        sparse_results: List[tuple[str, float]],
    ) -> List[tuple[str, float]]:
        rrf_scores: Dict[str, float] = defaultdict(float)
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
        for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _build_qdrant_filter(metadata_filter: Optional[Dict]) -> Optional[Filter]:
        if not metadata_filter:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        conditions = []

        if "source" in metadata_filter:
            conditions.append(
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=metadata_filter["source"])
                )
            )
        return Filter(must=conditions)
