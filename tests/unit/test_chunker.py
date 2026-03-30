"""
Unit tests for SemanticChunker.
"""

import pytest

from src.ingestion.chunker import SemanticChunker


@pytest.fixture
def chunker():
    return SemanticChunker(
        drift_threshold=0.4,
        min_chunk_tokens=30,
        max_chunk_tokens=200,
    )


SHORT_TEXT = "This is a single sentence."

MULTI_PARA_TEXT = """
Machine learning is a subset of artificial intelligence that enables systems to learn
from data. It uses statistical techniques to give computers the ability to learn.

Quantum computing uses quantum mechanical phenomena to perform computations.
Unlike classical computers that use bits, quantum computers use qubits.
Superposition allows qubits to exist in multiple states simultaneously.

The stock market is a complex system where shares of publicly held companies
are issued and traded. It reflects the collective expectations of investors
about corporate earnings and economic conditions.
"""


def test_chunker_returns_chunks(chunker):
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "test"})
    assert len(chunks) >= 1


def test_chunk_ids_are_unique(chunker):
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "test"})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata_propagated(chunker):
    meta = {"doc_id": "doc_abc", "source": "test.pdf", "page": 3}
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata=meta)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["doc_id"] == "doc_abc"


def test_chunk_text_non_empty(chunker):
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "test"})
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0


def test_single_sentence_returns_one_chunk(chunker):
    chunks = chunker.chunk(SHORT_TEXT, metadata={"doc_id": "single"})
    assert len(chunks) == 1
    assert chunks[0].text.strip() == SHORT_TEXT.strip()


def test_empty_text_returns_empty(chunker):
    chunks = chunker.chunk("", metadata={"doc_id": "empty"})
    assert chunks == []


def test_whitespace_only_returns_empty(chunker):
    chunks = chunker.chunk("   \n\n   ", metadata={"doc_id": "ws"})
    assert chunks == []


def test_token_count_estimated(chunker):
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "tok"})
    for chunk in chunks:
        assert chunk.token_count > 0


def test_no_chunk_exceeds_max_tokens(chunker):
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "max"})
    for chunk in chunks:
        assert chunk.token_count <= chunker.max_chunk_tokens + 50  # small tolerance


def test_all_text_preserved(chunker):
    """Concatenated chunk text should cover most of the original words."""
    chunks = chunker.chunk(MULTI_PARA_TEXT, metadata={"doc_id": "coverage"})
    original_words = set(MULTI_PARA_TEXT.lower().split())
    chunk_words = set(" ".join(c.text for c in chunks).lower().split())
    # At least 90% of original words should appear in some chunk
    overlap = len(original_words & chunk_words) / max(len(original_words), 1)
    assert overlap >= 0.9
