"""
Unit tests for retrieval evaluation metrics.
"""

import pytest

from src.evaluation.metrics import (
    aggregate_metrics,
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_zero():
    assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_recall_partial():
    result = recall_at_k(["a", "x", "y"], ["a", "b"], k=3)
    assert result == pytest.approx(0.5)


def test_recall_k_smaller_than_list():
    # Only first 2 of retrieved list considered
    assert recall_at_k(["x", "a"], ["a"], k=1) == 0.0
    assert recall_at_k(["a", "x"], ["a"], k=1) == 1.0


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], [], k=3) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_perfect():
    score = ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_ndcg_zero():
    score = ndcg_at_k(["x", "y"], ["a"], k=2)
    assert score == 0.0


def test_ndcg_penalizes_lower_rank():
    # a at rank 1 is better than a at rank 2
    score_rank1 = ndcg_at_k(["a", "x"], ["a"], k=2)
    score_rank2 = ndcg_at_k(["x", "a"], ["a"], k=2)
    assert score_rank1 > score_rank2


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mrr_empty_retrieved():
    assert mean_reciprocal_rank([], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------

def test_compute_retrieval_metrics_keys():
    metrics = compute_retrieval_metrics(
        retrieved_ids=["a", "b", "c"],
        relevant_ids=["a"],
        k_values=[1, 3, 5],
    )
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "ndcg@5" in metrics
    assert "mrr" in metrics


def test_compute_retrieval_metrics_values_in_range():
    metrics = compute_retrieval_metrics(["a", "b", "c"], ["a", "b"], k_values=[3])
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_mean():
    result = aggregate_metrics([
        {"recall@5": 0.4, "mrr": 0.6},
        {"recall@5": 0.8, "mrr": 0.2},
    ])
    assert result["recall@5"] == pytest.approx(0.6)
    assert result["mrr"] == pytest.approx(0.4)


def test_aggregate_empty():
    assert aggregate_metrics([]) == {}
