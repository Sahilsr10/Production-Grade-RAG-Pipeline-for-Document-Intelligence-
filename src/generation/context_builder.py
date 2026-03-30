"""
Context builder: takes re-ranked chunks and assembles a clean context string
for the LLM, respecting a token budget and deduplicating overlapping chunks.
"""

from __future__ import annotations

from typing import List

import tiktoken
from loguru import logger

from src.retrieval.hybrid_retriever import RetrievedChunk


class ContextBuilder:
    """
    Builds the final context string from re-ranked chunks.

    Steps:
    1. Deduplicate chunks with high text overlap (MMR-style)
    2. Sort by re-rank score (already done by reranker)
    3. Add chunks until token budget is reached
    4. Format with source attribution markers

    Args:
        max_tokens: Hard token budget for context
        overlap_threshold: Jaccard similarity above which a chunk is dropped
        encoding_name: tiktoken encoding for token counting
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        overlap_threshold: float = 0.8,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_threshold = overlap_threshold
        self._enc = tiktoken.get_encoding(encoding_name)

    def build(self, chunks: List[RetrievedChunk]) -> "Context":
        """
        Build context from chunks.

        Returns a Context object with the assembled text and source metadata.
        """
        deduplicated = self._deduplicate(chunks)
        selected, token_count = self._fill_budget(deduplicated)

        context_parts = []
        for i, chunk in enumerate(selected, start=1):
            source = chunk.metadata.get("title") or chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "")
            source_label = f"{source}, p.{page}" if page else source
            context_parts.append(f"[{i}] {chunk.text}\n(Source: {source_label})")

        context_text = "\n\n".join(context_parts)

        logger.debug(
            f"Context: {len(selected)} chunks, ~{token_count} tokens "
            f"(from {len(chunks)} candidates)"
        )

        return Context(
            text=context_text,
            chunks=selected,
            token_count=token_count,
            total_candidates=len(chunks),
        )

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove chunks whose content heavily overlaps with already-selected chunks."""
        selected: List[RetrievedChunk] = []
        selected_sets: List[set] = []

        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            is_duplicate = False

            for existing_words in selected_sets:
                intersection = chunk_words & existing_words
                union = chunk_words | existing_words
                if union and len(intersection) / len(union) > self.overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(chunk)
                selected_sets.append(chunk_words)

        if len(selected) < len(chunks):
            logger.debug(f"Deduplication removed {len(chunks) - len(selected)} chunks")

        return selected

    def _fill_budget(
        self, chunks: List[RetrievedChunk]
    ) -> tuple[List[RetrievedChunk], int]:
        """Select chunks until token budget is exhausted."""
        selected = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(self._enc.encode(chunk.text))
            if total_tokens + chunk_tokens > self.max_tokens:
                break
            selected.append(chunk)
            total_tokens += chunk_tokens

        return selected, total_tokens


class Context:
    """Assembled context ready to inject into the LLM prompt."""

    def __init__(
        self,
        text: str,
        chunks: List[RetrievedChunk],
        token_count: int,
        total_candidates: int,
    ):
        self.text = text
        self.chunks = chunks
        self.token_count = token_count
        self.total_candidates = total_candidates

    @property
    def sources(self) -> List[dict]:
        return [
            {
                "index": i + 1,
                "chunk_id": c.chunk_id,
                "source": c.metadata.get("source", ""),
                "title": c.metadata.get("title", ""),
                "page": c.metadata.get("page"),
                "score": c.rerank_score,
            }
            for i, c in enumerate(self.chunks)
        ]
