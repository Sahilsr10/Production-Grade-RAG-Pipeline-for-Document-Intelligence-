#!/usr/bin/env python3
"""
CLI: Auto-generate an evaluation testset from ingested chunks.

Samples chunks from Qdrant, generates questions using an LLM,
and saves to a JSON file for human review and annotation.

Usage:
  python scripts/build_testset.py --collection rag_documents --num-chunks 100
  python scripts/build_testset.py --collection rag_documents --output data/testset.json
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build RAG evaluation testset")
    parser.add_argument("--collection", default="rag_documents")
    parser.add_argument("--num-chunks", type=int, default=100, help="Chunks to sample")
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--output", default="data/testset.json")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from src.evaluation.dataset import TestsetBuilder

    logger.info(f"Sampling {args.num_chunks} chunks from '{args.collection}'...")
    client = QdrantClient(url=args.qdrant_url)

    # Scroll through collection to get sample chunks
    points, _ = client.scroll(
        collection_name=args.collection,
        limit=args.num_chunks,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        logger.error(f"No points found in collection '{args.collection}'")
        sys.exit(1)

    chunks = []
    for point in points:
        payload = point.payload or {}
        chunks.append({
            "text": payload.get("text", ""),
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "metadata": {
                k: v for k, v in payload.items()
                if k not in ("text", "chunk_id")
            },
        })

    logger.info(f"Sampled {len(chunks)} chunks")

    builder = TestsetBuilder(
        questions_per_chunk=args.questions_per_chunk,
    )
    dataset = builder.build_from_chunks(
        chunks=chunks,
        output_path=args.output,
        seed=args.seed,
    )

    print(f"\n=== Testset Built ===")
    print(f"  Questions generated: {len(dataset)}")
    print(f"  Saved to:           {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review {args.output} and fill in 'ground_truth_answer' for each question")
    print(f"  2. Verify 'relevant_chunk_ids' are correct")
    print(f"  3. Run: python scripts/evaluate.py --testset {args.output}")


if __name__ == "__main__":
    main()
