#!/usr/bin/env python3
"""
CLI: Ingest documents into the RAG pipeline.

Usage examples:
  python scripts/ingest.py --source data/raw/reports/
  python scripts/ingest.py --source data/raw/paper.pdf --recreate
  python scripts/ingest.py --source https://arxiv.org/abs/2312.10997
  python scripts/ingest.py --source data/raw/ --collection finance_docs
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument("--source", required=True, help="File, directory, or URL to ingest")
    parser.add_argument("--collection", default="rag_documents", help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--bm25-path", default="data/processed/bm25.pkl")
    parser.add_argument("--drift-threshold", type=float, default=0.4)
    args = parser.parse_args()

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        bm25_index_path=args.bm25_path,
        drift_threshold=args.drift_threshold,
    )

    logger.info(f"Starting ingestion: {args.source} → '{args.collection}'")
    stats = pipeline.ingest(
        source=args.source,
        recreate_collection=args.recreate,
    )

    print("\n=== Ingestion Complete ===")
    print(f"  Documents loaded:  {stats['documents']}")
    print(f"  Chunks produced:   {stats['chunks']}")
    print(f"  Vectors upserted:  {stats['vectors_upserted']}")
    print(f"  Collection:        {args.collection}")


if __name__ == "__main__":
    main()
