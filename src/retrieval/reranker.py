"""
Cross-encoder re-ranker.

The retriever uses bi-encoders (query and doc embedded separately).
This is fast but loses interaction signals between query and document.

The cross-encoder sees (query, doc) jointly — far more accurate.
We run it only on the top-50 candidates from retrieval, then keep top-5.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking
- Fast inference (L-6 = 6 transformer layers)
- Strong zero-shot generalization
"""

from __future__ import annotations

from typing import List

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.retrieval.hybrid_retriever import RetrievedChunk


class CrossEncoderReranker:
    """
    Scores (query, chunk) pairs jointly and returns top-k by score.

    Args:
        model_name: HuggingFace model identifier
        top_k: How many chunks to return after re-ranking
        batch_size: Inference batch size
        max_length: Max token length for (query + doc) pair
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading cross-encoder: {model_name} on {self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Score all (query, chunk) pairs and return top-k sorted by score.

        Args:
            query: The original user query (NOT the HyDE doc)
            chunks: Candidates from hybrid retrieval

        Returns:
            Top-k chunks sorted by cross-encoder score descending
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._score_pairs(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        top = reranked[: self.top_k]

        logger.debug(
            f"Re-ranked {len(chunks)} → {len(top)} chunks. "
            f"Top score: {top[0].rerank_score:.3f}"
        )
        return top

    def _score_pairs(self, pairs: List[tuple[str, str]]) -> List[float]:
        """Score all (query, doc) pairs in batches."""
        all_scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                queries, docs = zip(*batch)

                encoded = self._tokenizer(
                    list(queries),
                    list(docs),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                logits = self._model(**encoded).logits
                # ms-marco models output a single logit — use it directly
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1).float().cpu().tolist()
                else:
                    scores = logits[:, 1].float().cpu().tolist()

                all_scores.extend(scores)

        return all_scores
