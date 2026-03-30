"""
Unit tests for MMR reranker and drift monitor.
"""

import numpy as np
import pytest

from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.mmr import MMRReranker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"title": "Test"},
        rerank_score=score,
    )


DIVERSE_CHUNKS = [
    make_chunk("c1", "Machine learning enables computers to learn from data automatically.", 0.9),
    make_chunk("c2", "Machine learning lets computers learn from data without explicit programming.", 0.88),
    make_chunk("c3", "Quantum computing uses qubits and superposition for parallel computation.", 0.75),
    make_chunk("c4", "The stock market reflects investor expectations about future corporate earnings.", 0.70),
    make_chunk("c5", "Deep learning is a subset of machine learning using neural networks.", 0.85),
]


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------

def test_mmr_returns_correct_count():
    reranker = MMRReranker(top_k=3)
    result = reranker.rerank("machine learning", DIVERSE_CHUNKS)
    assert len(result) == 3


def test_mmr_returns_all_when_fewer_than_top_k():
    reranker = MMRReranker(top_k=10)
    result = reranker.rerank("machine learning", DIVERSE_CHUNKS[:2])
    assert len(result) == 2


def test_mmr_diversity_reduces_near_duplicates():
    """With lambda=0.3 (diversity-focused), c1 and c2 (near-duplicates) should not both be top-2."""
    reranker = MMRReranker(top_k=2, lambda_mult=0.3)
    result = reranker.rerank("machine learning", DIVERSE_CHUNKS)
    ids = {c.chunk_id for c in result}
    # c1 and c2 are near-duplicates — only one should be selected
    assert not ({"c1", "c2"}.issubset(ids)), "Near-duplicate chunks both selected — MMR not working"


def test_mmr_relevance_mode_picks_top_scores():
    """With lambda=1.0, MMR degenerates to pure relevance — picks highest-scored chunks."""
    reranker = MMRReranker(top_k=2, lambda_mult=1.0)
    result = reranker.rerank("machine learning", DIVERSE_CHUNKS)
    ids = [c.chunk_id for c in result]
    # c1 (0.9) and c5 (0.85) are highest scored
    assert "c1" in ids


def test_mmr_empty_input():
    reranker = MMRReranker(top_k=3)
    result = reranker.rerank("query", [])
    assert result == []


# ---------------------------------------------------------------------------
# Drift monitor tests
# ---------------------------------------------------------------------------

def test_drift_monitor_records_queries():
    from src.monitoring.drift_monitor import QueryDriftMonitor

    monitor = QueryDriftMonitor(window_size=10)
    for _ in range(5):
        emb = np.random.randn(128)
        monitor.record_query(emb)

    assert monitor._query_count == 5
    assert len(monitor._window) == 5


def test_drift_monitor_no_alert_without_baseline():
    from src.monitoring.drift_monitor import QueryDriftMonitor

    monitor = QueryDriftMonitor()  # no baseline
    emb = np.random.randn(128)
    alert = monitor.record_query(emb)
    assert alert is None


def test_retrieval_score_monitor_tracks_scores():
    from src.monitoring.drift_monitor import RetrievalScoreMonitor

    monitor = RetrievalScoreMonitor(window_size=5)
    for score in [0.9, 0.85, 0.88, 0.91, 0.87]:
        monitor.record(score)

    assert monitor.recent_mean is not None
    assert 0.8 < monitor.recent_mean < 1.0


def test_retrieval_score_monitor_sets_baseline_after_window():
    from src.monitoring.drift_monitor import RetrievalScoreMonitor

    monitor = RetrievalScoreMonitor(window_size=5)
    assert monitor._baseline_mean is None

    for score in [0.9] * 5:
        monitor.record(score)

    assert monitor._baseline_mean is not None
    assert abs(monitor._baseline_mean - 0.9) < 0.01
