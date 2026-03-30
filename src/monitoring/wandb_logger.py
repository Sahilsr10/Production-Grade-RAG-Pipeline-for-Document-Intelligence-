"""
W&B experiment logger.

Logs:
- Pipeline configuration (as W&B config)
- Per-query latency traces
- Evaluation metric runs
- Retrieval quality signals (RRF scores, rerank scores)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger


class WandbLogger:
    """
    Thin wrapper around W&B for experiment tracking.

    Degrades gracefully if wandb is not installed or not configured.
    """

    def __init__(
        self,
        project: str = "production-rag",
        entity: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[list] = None,
        config: Optional[Dict] = None,
    ):
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                tags=tags or [],
                config=config or {},
                reinit=True,
            )
            logger.info(f"W&B run started: {self._run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed (continuing without logging): {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception as e:
            logger.debug(f"W&B log failed: {e}")

    def log_query(
        self,
        query: str,
        answer: str,
        latency: Dict,
        retrieval_scores: list,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        self.log(
            {
                "query/length": len(query),
                "answer/length": len(answer),
                "retrieval/top_score": retrieval_scores[0] if retrieval_scores else 0,
                **latency,
            },
            step=step,
        )

    def log_eval_results(self, results: Dict[str, float]) -> None:
        if not self.enabled:
            return
        self.log({f"eval/{k}": v for k, v in results.items()})

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._wandb.finish()
