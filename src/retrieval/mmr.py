"""
Maximal Marginal Relevance (MMR) retrieval.

Standard retrieval returns the top-K most relevant chunks — but if your
document collection has many near-duplicate passages, you get redundant context.

MMR balances:
  - Relevance to the query (you want high relevance)
  - Diversity from already-selected chunks (you want low similarity to what's selected)

Formula:
  MMR(d) = λ * sim(d, query) - (1-λ) * max_j sim(d, selected_j)

λ=1.0 → pure relevance (same as standard retrieval)
λ=0.0 → pure diversity
λ=0.5 → balanced (recommended default)

When to use:
  - Long documents with repetitive content
  - Multi-document collections covering similar topics
  - When you notice the LLM getting the same information from multiple chunks
"""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_retriever import RetrievedChunk


class MMRReranker:
    """
    Reranks retrieved chunks using Maximal Marginal Relevance.

    Can be used as an alternative or complement to the cross-encoder reranker.
    MMR is fast (no neural inference) — good when latency is critical.

    Args:
        lambda_mult: Trade-off between relevance and diversity (0-1).
                     0.5 = balanced, 1.0 = relevance-only, 0.0 = diversity-only
        top_k: Number of chunks to select
        embed_model: Model for computing chunk-to-chunk similarity
    """

    def __init__(
        self,
        lambda_mult: float = 0.5,
        top_k: int = 5,
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.lambda_mult = lambda_mult
        self.top_k = top_k
        self._model = SentenceTransformer(embed_model)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Select diverse, relevant chunks via MMR."""
        if not chunks:
            return []

        if len(chunks) <= self.top_k:
            return chunks

        # Embed query and all chunks
        texts = [query] + [c.text for c in chunks]
        embeddings = self._model.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # Relevance scores: cosine similarity with query
        relevance = chunk_embs @ query_emb  # shape: (N,)

        selected_indices: List[int] = []
        remaining = list(range(len(chunks)))

        for _ in range(self.top_k):
            if not remaining:
                break

            if not selected_indices:
                # First selection: pick most relevant
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # MMR: balance relevance vs diversity
                selected_embs = chunk_embs[selected_indices]  # shape: (S, D)

                scores = []
                for i in remaining:
                    rel = self.lambda_mult * relevance[i]
                    # Redundancy: max similarity to any already-selected chunk
                    redundancy = (1 - self.lambda_mult) * float(
                        np.max(chunk_embs[i] @ selected_embs.T)
                    )
                    scores.append((i, rel - redundancy))

                best = max(scores, key=lambda x: x[1])[0]

            selected_indices.append(best)
            remaining.remove(best)

        result = [chunks[i] for i in selected_indices]
        logger.debug(
            f"MMR selected {len(result)} diverse chunks from {len(chunks)} candidates "
            f"(λ={self.lambda_mult})"
        )
        return result
