"""
Concept drift monitor.

In production, two things drift over time:
1. Query drift: users start asking different kinds of questions
2. Document drift: the underlying corpus becomes stale

This module tracks both using embedding-space statistics.

Query drift detection:
  - Maintain a rolling window of query embeddings
  - Compare current window distribution to baseline using Maximum Mean Discrepancy (MMD)
  - Alert when MMD exceeds threshold

Document staleness:
  - Track document ingestion timestamps
  - Alert when average document age exceeds threshold
  - Detect when retrieval scores drop (may indicate stale docs)

This is what interviewers mean when they ask "how do you handle model degradation?"
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class DriftAlert:
    alert_type: str          # "query_drift" | "score_degradation" | "low_recall"
    severity: str            # "warning" | "critical"
    message: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class QueryDriftMonitor:
    """
    Detects when the distribution of incoming queries shifts significantly
    from the baseline distribution observed during system validation.

    Uses Maximum Mean Discrepancy (MMD) — a kernel-based test that measures
    the distance between two distributions without assuming any particular form.

    Args:
        baseline_path: Path to saved baseline query embeddings (numpy array)
        window_size: Rolling window of recent queries to compare against baseline
        mmd_threshold: MMD value above which drift is flagged
        alert_on_n_queries: How often to run the MMD test (every N queries)
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        window_size: int = 200,
        mmd_threshold: float = 0.05,
        alert_on_n_queries: int = 50,
    ):
        self.window_size = window_size
        self.mmd_threshold = mmd_threshold
        self.alert_on_n_queries = alert_on_n_queries
        self._window: Deque[np.ndarray] = deque(maxlen=window_size)
        self._baseline: Optional[np.ndarray] = None
        self._query_count = 0
        self._alerts: List[DriftAlert] = []

        if baseline_path and Path(baseline_path).exists():
            self._baseline = np.load(baseline_path)
            logger.info(
                f"Loaded query drift baseline: {self._baseline.shape[0]} embeddings"
            )

    def record_query(self, query_embedding: np.ndarray) -> Optional[DriftAlert]:
        """
        Record a new query embedding and optionally run drift detection.

        Returns a DriftAlert if drift is detected, otherwise None.
        """
        self._window.append(query_embedding)
        self._query_count += 1

        if (
            self._baseline is not None
            and self._query_count % self.alert_on_n_queries == 0
            and len(self._window) >= 50
        ):
            return self._check_drift()
        return None

    def set_baseline(self, embeddings: np.ndarray, save_path: Optional[str] = None) -> None:
        """Set the baseline distribution from validation query embeddings."""
        self._baseline = embeddings
        if save_path:
            np.save(save_path, embeddings)
            logger.info(f"Query drift baseline saved: {embeddings.shape[0]} queries")

    def _check_drift(self) -> Optional[DriftAlert]:
        """Compute MMD between baseline and current window."""
        current = np.array(list(self._window))
        mmd = self._mmd(self._baseline, current)

        logger.debug(f"Query drift MMD: {mmd:.4f} (threshold: {self.mmd_threshold})")

        if mmd > self.mmd_threshold:
            severity = "critical" if mmd > self.mmd_threshold * 2 else "warning"
            alert = DriftAlert(
                alert_type="query_drift",
                severity=severity,
                message=(
                    f"Query distribution has drifted from baseline. "
                    f"MMD={mmd:.4f} > threshold={self.mmd_threshold}. "
                    f"Consider re-evaluating pipeline performance on recent queries."
                ),
                metric_value=mmd,
                threshold=self.mmd_threshold,
            )
            self._alerts.append(alert)
            logger.warning(f"[DRIFT ALERT] {alert.message}")
            return alert
        return None

    @staticmethod
    def _mmd(X: np.ndarray, Y: np.ndarray, kernel_bandwidth: float = 1.0) -> float:
        """
        Maximum Mean Discrepancy with RBF kernel.

        MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

        Subsamples to keep computation O(n) for large arrays.
        """
        n = min(len(X), len(Y), 500)
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), n, replace=False)]
        Y = Y[rng.choice(len(Y), n, replace=False)]

        def rbf(A: np.ndarray, B: np.ndarray) -> float:
            dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
            return float(np.mean(np.exp(-dists / (2 * kernel_bandwidth ** 2))))

        return rbf(X, X) - 2 * rbf(X, Y) + rbf(Y, Y)


class RetrievalScoreMonitor:
    """
    Tracks retrieval quality metrics over time and alerts on degradation.

    Key signal: if average top-1 rerank scores are dropping, it likely means:
    - Documents are becoming stale (answers are no longer in the corpus)
    - Query types have changed (retrieval is less accurate)
    - Embedding model drift (rare, but possible with API model updates)
    """

    def __init__(
        self,
        window_size: int = 100,
        score_drop_threshold: float = 0.15,  # Alert if score drops by >15% of baseline
    ):
        self.window_size = window_size
        self.score_drop_threshold = score_drop_threshold
        self._scores: Deque[float] = deque(maxlen=window_size)
        self._baseline_mean: Optional[float] = None

    def record(self, top_score: float) -> Optional[DriftAlert]:
        """Record the top rerank score for a query."""
        self._scores.append(top_score)

        if len(self._scores) == self.window_size and self._baseline_mean is None:
            self._baseline_mean = float(np.mean(list(self._scores)))
            logger.info(f"Retrieval score baseline set: {self._baseline_mean:.3f}")

        if self._baseline_mean and len(self._scores) >= 20:
            recent_mean = float(np.mean(list(self._scores)[-20:]))
            drop = (self._baseline_mean - recent_mean) / self._baseline_mean

            if drop > self.score_drop_threshold:
                alert = DriftAlert(
                    alert_type="score_degradation",
                    severity="warning" if drop < 0.3 else "critical",
                    message=(
                        f"Retrieval scores degraded by {drop:.0%}. "
                        f"Baseline={self._baseline_mean:.3f}, "
                        f"Recent={recent_mean:.3f}. "
                        f"Consider re-ingesting updated documents."
                    ),
                    metric_value=recent_mean,
                    threshold=self._baseline_mean * (1 - self.score_drop_threshold),
                )
                logger.warning(f"[SCORE ALERT] {alert.message}")
                return alert
        return None

    @property
    def recent_mean(self) -> Optional[float]:
        if self._scores:
            return float(np.mean(list(self._scores)[-20:]))
        return None


class DriftMonitor:
    """
    Composite drift monitor combining query drift + retrieval score monitoring.
    Use this in production by calling .record() after every query.
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        alert_log_path: str = "data/drift_alerts.jsonl",
    ):
        self.query_monitor = QueryDriftMonitor(baseline_path=baseline_path)
        self.score_monitor = RetrievalScoreMonitor()
        self.alert_log_path = alert_log_path
        Path(alert_log_path).parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        query_embedding: np.ndarray,
        top_retrieval_score: float,
    ) -> List[DriftAlert]:
        """Record a completed query and return any triggered alerts."""
        alerts = []

        score_alert = self.score_monitor.record(top_retrieval_score)
        if score_alert:
            alerts.append(score_alert)

        drift_alert = self.query_monitor.record_query(query_embedding)
        if drift_alert:
            alerts.append(drift_alert)

        for alert in alerts:
            self._log_alert(alert)

        return alerts

    def _log_alert(self, alert: DriftAlert) -> None:
        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": alert.timestamp,
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "metric": alert.metric_value,
                "threshold": alert.threshold,
            }) + "\n")

    @property
    def stats(self) -> Dict:
        return {
            "queries_recorded": self.query_monitor._query_count,
            "recent_mean_score": self.score_monitor.recent_mean,
            "score_baseline": self.score_monitor._baseline_mean,
            "total_alerts": len(self.query_monitor._alerts),
        }
