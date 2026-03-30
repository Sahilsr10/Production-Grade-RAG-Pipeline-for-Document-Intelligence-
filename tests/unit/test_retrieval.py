"""
Unit tests for RRF fusion logic and context builder.
"""

import pytest

from src.generation.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# RRF fusion (tested via _rrf_fusion directly)
# ---------------------------------------------------------------------------

def make_retriever_stub():
    """Return a HybridRetriever without hitting any external service."""
    obj = HybridRetriever.__new__(HybridRetriever)
    obj.rrf_k = 60
    return obj


def test_rrf_fusion_combines_lists():
    retriever = make_retriever_stub()
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 10.0), ("c", 8.0), ("d", 6.0)]
    result = retriever._rrf_fusion(dense, sparse)
    ids = [r[0] for r in result]
    # "b" and "c" appear in both lists so should rank high
    assert "b" in ids[:2]
    assert "c" in ids[:3]


def test_rrf_fusion_unique_ids():
    retriever = make_retriever_stub()
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("a", 5.0), ("c", 3.0)]
    result = retriever._rrf_fusion(dense, sparse)
    ids = [r[0] for r in result]
    assert len(ids) == len(set(ids))


def test_rrf_scores_positive():
    retriever = make_retriever_stub()
    dense = [("a", 0.9)]
    sparse = [("b", 5.0)]
    result = retriever._rrf_fusion(dense, sparse)
    for _, score in result:
        assert score > 0


def test_rrf_empty_lists():
    retriever = make_retriever_stub()
    result = retriever._rrf_fusion([], [])
    assert result == []


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

def make_chunk(chunk_id, text, score=0.9):
    c = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"title": "Test Doc", "source": "test.pdf"},
        rerank_score=score,
    )
    return c


def test_context_builder_respects_token_budget():
    builder = ContextBuilder(max_tokens=100)
    # Each chunk is ~30 tokens; 4 chunks = ~120 tokens > budget
    chunks = [make_chunk(f"c{i}", f"word " * 25) for i in range(4)]
    ctx = builder.build(chunks)
    assert ctx.token_count <= 110  # small tolerance


def test_context_builder_deduplication():
    builder = ContextBuilder(max_tokens=5000, overlap_threshold=0.8)
    text = "The revenue grew significantly in the third quarter of fiscal year 2024."
    chunks = [
        make_chunk("c1", text, score=0.9),
        make_chunk("c2", text, score=0.8),  # near-duplicate
    ]
    ctx = builder.build(chunks)
    assert len(ctx.chunks) == 1  # duplicate removed


def test_context_builder_sources_have_correct_indices():
    builder = ContextBuilder(max_tokens=5000)
    chunks = [make_chunk(f"c{i}", f"Unique content for chunk {i}.") for i in range(3)]
    ctx = builder.build(chunks)
    for i, source in enumerate(ctx.sources, start=1):
        assert source["index"] == i


def test_context_text_contains_source_labels():
    builder = ContextBuilder(max_tokens=5000)
    chunks = [make_chunk("c1", "Some content about the topic.")]
    ctx = builder.build(chunks)
    assert "[1]" in ctx.text
    assert "Source:" in ctx.text


def test_context_empty_chunks():
    builder = ContextBuilder(max_tokens=5000)
    ctx = builder.build([])
    assert ctx.text == ""
    assert ctx.chunks == []
    assert ctx.token_count == 0
