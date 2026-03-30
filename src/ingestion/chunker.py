"""
Semantic chunker that splits documents at topic-drift boundaries
rather than arbitrary token counts.

How it works:
1. Split document into sentences using NLTK
2. Embed each sentence with a lightweight model
3. Compute cosine distance between consecutive sentence embeddings
4. Split where distance exceeds drift_threshold
5. Merge short chunks into their neighbours
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import nltk
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

# Download punkt tokenizer on first use
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    token_count: int = 0
    embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id!r}, tokens={self.token_count}, text={self.text[:60]!r}...)"


class SemanticChunker:
    """
    Splits documents at semantic topic boundaries.

    Args:
        drift_threshold: Cosine distance above which a new chunk starts.
                         Higher = fewer, larger chunks. Lower = more, smaller chunks.
        min_chunk_tokens: Chunks smaller than this get merged into neighbours.
        max_chunk_tokens: Hard cap — chunks over this get force-split.
        embed_model: Sentence transformer used for drift detection.
                     Should be fast; accuracy matters less here than speed.
    """

    def __init__(
        self,
        drift_threshold: float = 0.2,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 600,
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.drift_threshold = drift_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        logger.info(f"Loading chunker embedding model: {embed_model}")
        self._model = SentenceTransformer(embed_model)

    def chunk(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        """Split a document into semantically coherent chunks."""
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [self._make_chunk(sentences, metadata, idx=0)]

        # Embed all sentences in one batch for efficiency
        embeddings = self._model.encode(sentences, batch_size=64, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine distance between consecutive sentences
        # distance = 1 - cosine_similarity
        distances = [
            1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Find split points where topic drifts
        split_points = {
            i + 1
            for i, d in enumerate(distances)
            if d > self.drift_threshold
        }

        # Build initial groups
        groups = self._group_sentences(sentences, split_points)

        # Merge groups that are too short
        groups = self._merge_short_groups(groups)

        # Force-split groups that are too long
        groups = self._split_long_groups(groups)

        chunks = [
            self._make_chunk(group, metadata, idx=i)
            for i, group in enumerate(groups)
            if group
        ]
        logger.debug(f"Chunked document into {len(chunks)} semantic chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences and clean up whitespace."""
        text = re.sub(r"\s+", " ", text).strip()
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _group_sentences(
        self, sentences: List[str], split_points: set
    ) -> List[List[str]]:
        groups: List[List[str]] = []
        current: List[str] = []
        for i, sent in enumerate(sentences):
            if i in split_points and current:
                groups.append(current)
                current = []
            current.append(sent)
        if current:
            groups.append(current)
        return groups

    def _merge_short_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """Merge groups whose token count is below min_chunk_tokens."""
        merged: List[List[str]] = []
        for group in groups:
            tokens = self._estimate_tokens(" ".join(group))
            if merged and tokens < self.min_chunk_tokens:
                merged[-1].extend(group)
            else:
                merged.append(group)
        return merged

    def _split_long_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """Force-split groups that exceed max_chunk_tokens."""
        result: List[List[str]] = []
        for group in groups:
            if self._estimate_tokens(" ".join(group)) <= self.max_chunk_tokens:
                result.append(group)
                continue
            # Split into halves recursively
            mid = len(group) // 2
            for sub in self._split_long_groups([group[:mid], group[mid:]]):
                result.append(sub)
        return result

    def _make_chunk(self, sentences: List[str], metadata: dict, idx: int) -> Chunk:
        text = " ".join(sentences)
        doc_id = metadata.get("doc_id", "unknown")
        return Chunk(
            text=text,
            metadata={**metadata, "chunk_index": idx, "sentence_count": len(sentences)},
            chunk_id=f"{doc_id}_chunk_{idx}",
            token_count=self._estimate_tokens(text),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Fast approximation: ~1.3 tokens per word."""
        return int(len(text.split()) * 1.3)
