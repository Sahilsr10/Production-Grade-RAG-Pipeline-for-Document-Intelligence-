"""
Evaluation harness: runs the full pipeline against a testset and
produces a comprehensive metrics report.

Compares multiple configurations (baselines vs full pipeline) and
logs everything to W&B for experiment tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import (
    GenerationEvaluator,
    aggregate_metrics,
    compute_retrieval_metrics,
)
from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.monitoring.wandb_logger import WandbLogger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""
    name: str
    use_hyde: bool = True
    use_decomposition: bool = True
    use_reranker: bool = True
    top_k_retrieve: int = 50
    top_k_rerank: int = 5
    tags: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    question: str
    answer: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    context_text: str
    latency_ms: float
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    ground_truth: Optional[str] = None


class EvaluationHarness:
    """
    Runs end-to-end evaluation across a testset.

    Supports multiple configurations to compare baselines vs full pipeline.

    Args:
        retriever: Configured HybridRetriever
        reranker: Configured CrossEncoderReranker
        generator: Configured AnswerGenerator
        context_builder: Configured ContextBuilder
        gen_evaluator: LLM-as-judge evaluator
        wandb_logger: Optional W&B logger
        k_values: List of K values for Recall@K and NDCG@K
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: AnswerGenerator,
        context_builder: ContextBuilder,
        gen_evaluator: Optional[GenerationEvaluator] = None,
        wandb_logger: Optional[WandbLogger] = None,
        k_values: List[int] = [1, 3, 5, 10],
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.context_builder = context_builder
        self.gen_evaluator = gen_evaluator or GenerationEvaluator()
        self.wandb_logger = wandb_logger
        self.k_values = k_values

    def run(
        self,
        dataset: EvalDataset,
        config: EvalConfig,
        max_queries: Optional[int] = None,
    ) -> Dict:
        """
        Run evaluation for a single configuration.

        Args:
            dataset: EvalDataset with questions (and optionally ground truth)
            config: Pipeline configuration to evaluate
            max_queries: Limit number of queries (for quick smoke tests)

        Returns:
            Summary metrics dict
        """
        logger.info(f"Starting eval run: '{config.name}' on {len(dataset)} questions")

        transformer = QueryTransformer(
            use_hyde=config.use_hyde,
            use_decomposition=config.use_decomposition,
        )

        query_results: List[QueryResult] = []
        items = list(dataset)
        if max_queries:
            items = items[:max_queries]

        for i, item in enumerate(tqdm(items, desc=f"Eval [{config.name}]")):
            try:
                result = self._run_single(
                    item=item,
                    transformer=transformer,
                    config=config,
                )
                query_results.append(result)

                if self.wandb_logger:
                    self.wandb_logger.log(
                        {
                            f"{config.name}/latency_ms": result.latency_ms,
                            **{f"{config.name}/{k}": v for k, v in result.retrieval_metrics.items()},
                            **{f"{config.name}/{k}": v for k, v in result.generation_metrics.items()},
                        },
                        step=i,
                    )
            except Exception as e:
                logger.warning(f"Query {i} failed: {e}")

        # Aggregate
        retrieval_summary = aggregate_metrics([r.retrieval_metrics for r in query_results])
        generation_summary = aggregate_metrics([r.generation_metrics for r in query_results])
        latency_summary = self._latency_stats([r.latency_ms for r in query_results])

        summary = {
            "config": config.name,
            "num_queries": len(query_results),
            **retrieval_summary,
            **generation_summary,
            **latency_summary,
        }

        logger.success(f"Eval complete [{config.name}]: {summary}")

        if self.wandb_logger:
            self.wandb_logger.log_eval_results(summary)

        return summary

    def compare(
        self,
        dataset: EvalDataset,
        configs: List[EvalConfig],
        max_queries: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run multiple configurations and return a comparison DataFrame.

        This is what you show in your project README / interview.
        """
        results = []
        for config in configs:
            summary = self.run(dataset, config, max_queries=max_queries)
            results.append(summary)

        df = pd.DataFrame(results).set_index("config")
        logger.info(f"\n{df.to_string()}")
        return df

    def _run_single(
        self,
        item: Dict,
        transformer: QueryTransformer,
        config: EvalConfig,
    ) -> QueryResult:
        question = item["question"]
        relevant_ids = item.get("relevant_chunk_ids", [])
        ground_truth = item.get("ground_truth_answer")

        t0 = time.time()

        # 1. Transform query
        transformed = transformer.transform(question)

        # 2. Retrieve
        candidates = self.retriever.retrieve(transformed)

        # 3. Optionally rerank
        if config.use_reranker:
            final_chunks = self.reranker.rerank(question, candidates)
        else:
            final_chunks = candidates[: config.top_k_rerank]

        # 4. Build context
        context = self.context_builder.build(final_chunks)

        # 5. Generate answer
        answer = self.generator.generate(question, context)

        latency_ms = (time.time() - t0) * 1000

        # 6. Retrieval metrics
        retrieved_ids = [c.chunk_id for c in candidates]
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids, relevant_ids, self.k_values
        ) if relevant_ids else {}

        # 7. Generation metrics
        gen_metrics = self.gen_evaluator.evaluate(
            query=question,
            answer=answer.answer,
            context=context.text,
            ground_truth=ground_truth,
        )

        return QueryResult(
            question=question,
            answer=answer.answer,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            context_text=context.text,
            latency_ms=latency_ms,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=gen_metrics,
            ground_truth=ground_truth,
        )

    @staticmethod
    def _latency_stats(latencies: List[float]) -> Dict[str, float]:
        import numpy as np
        if not latencies:
            return {}
        arr = np.array(latencies)
        return {
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_p99_ms": float(np.percentile(arr, 99)),
            "latency_mean_ms": float(np.mean(arr)),
        }
