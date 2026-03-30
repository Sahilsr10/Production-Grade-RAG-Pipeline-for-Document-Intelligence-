#!/usr/bin/env python3
"""
CLI: Run the full evaluation suite and produce a comparison report.

Usage examples:
  python scripts/evaluate.py --testset data/testset.json
  python scripts/evaluate.py --testset data/testset.json --max-queries 50
  python scripts/evaluate.py --testset data/testset.json --config configs/eval.yaml
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--testset", required=True, help="Path to testset JSON")
    parser.add_argument("--collection", default="rag_documents")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries (for testing)")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--bm25-path", default="data/processed/bm25.pkl")
    parser.add_argument("--output", default="data/eval_results.csv", help="Save results CSV")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    from src.evaluation.dataset import EvalDataset
    from src.evaluation.harness import EvalConfig, EvaluationHarness
    from src.evaluation.metrics import GenerationEvaluator
    from src.generation.context_builder import ContextBuilder
    from src.generation.generator import AnswerGenerator
    from src.monitoring.wandb_logger import WandbLogger
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    logger.info(f"Loading testset: {args.testset}")
    dataset = EvalDataset(args.testset)
    logger.info(f"Testset stats: {dataset.stats()}")

    # Build pipeline components
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        bm25_index_path=args.bm25_path,
    )
    reranker = CrossEncoderReranker(top_k=5)
    generator = AnswerGenerator()
    context_builder = ContextBuilder()
    gen_evaluator = GenerationEvaluator()

    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT", "production-rag"),
        enabled=not args.no_wandb and bool(os.getenv("WANDB_API_KEY")),
        tags=["evaluation"],
        config={"testset": args.testset, "collection": args.collection},
    )

    harness = EvaluationHarness(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        context_builder=context_builder,
        gen_evaluator=gen_evaluator,
        wandb_logger=wandb_logger,
    )

    # Define configs to compare
    configs = [
        EvalConfig(
            name="naive_rag",
            use_hyde=False,
            use_decomposition=False,
            use_reranker=False,
        ),
        EvalConfig(
            name="dense_plus_hyde",
            use_hyde=True,
            use_decomposition=False,
            use_reranker=False,
        ),
        EvalConfig(
            name="full_pipeline",
            use_hyde=True,
            use_decomposition=True,
            use_reranker=True,
        ),
    ]

    # Run comparison
    results_df = harness.compare(
        dataset=dataset,
        configs=configs,
        max_queries=args.max_queries,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output)
    logger.success(f"Results saved to {args.output}")

    print("\n=== Evaluation Results ===")
    print(results_df.to_string())

    wandb_logger.finish()


if __name__ == "__main__":
    main()
