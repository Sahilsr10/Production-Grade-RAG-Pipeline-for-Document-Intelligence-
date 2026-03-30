"""
Ingestion pipeline: load → chunk → embed (BGE) → upsert to Qdrant + BM25
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.embedder import BM25Index, DenseEmbedder
from src.ingestion.loaders import get_loader


class IngestionPipeline:
    """
    End-to-end document ingestion:
    1. Load documents from source
    2. Semantic chunk each document
    3. Generate BGE dense embeddings (local, free)
    4. Upsert to Qdrant vector store
    5. Build / save BM25 sparse index
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_documents",
        bm25_index_path: str = "data/processed/bm25.pkl",
        embed_model: str = "BAAI/bge-base-en-v1.5",
        drift_threshold: float = 0.4,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.bm25_index_path = bm25_index_path

        self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._chunker = SemanticChunker(drift_threshold=drift_threshold)
        self._embedder = DenseEmbedder(model_name=embed_model)
        self._bm25 = BM25Index()

    def ingest(
        self,
        source: str,
        batch_size: int = 64,
        recreate_collection: bool = False,
    ) -> dict:
        # 1. Load
        logger.info(f"Loading from: {source}")
        paths = []

        source_path = Path(source)

        if source_path.is_dir():
            paths = list(source_path.glob("*"))
        else:
            paths = [source_path]

        raw_docs = []

        for path in paths:
            logger.info(f"Loading file: {path}")
            loader = get_loader(str(path))
            docs = loader.load(str(path))
            raw_docs.extend(docs)

        logger.info(f"Loaded {len(raw_docs)} documents")

        # 2. Chunk
        all_chunks: List[Chunk] = []
        for doc in tqdm(raw_docs, desc="Chunking"):
            chunks = self._chunker.chunk(doc["text"], metadata=doc["metadata"])
            all_chunks.extend(chunks)
        logger.info(f"Produced {len(all_chunks)} chunks")

        # 3. Ensure Qdrant collection
        self._ensure_collection(recreate=recreate_collection)

        # 4. Embed + upsert in batches
        # is_query=False during ingestion (no BGE instruction prefix at index time)
        vectors_upserted = 0
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & upserting"):
            batch = all_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts, is_query=False)

            points = [
                PointStruct(
                    id=self._chunk_id_to_int(c.chunk_id),
                    vector=embeddings[j].tolist(),
                    payload={
                        "text": c.text,
                        "chunk_id": c.chunk_id,
                        "token_count": c.token_count,
                        "filename": Path(c.metadata.get("source", "")).name,
                        **c.metadata,
                    },
                )
                for j, c in enumerate(batch)
            ]
            self._qdrant.upsert(collection_name=self.collection_name, points=points)
            vectors_upserted += len(points)

        # 5. Build and save BM25
        logger.info("Building BM25 index...")
        self._bm25.build(all_chunks)
        self._bm25.save(self.bm25_index_path)

        stats = {
            "documents": len(raw_docs),
            "chunks": len(all_chunks),
            "vectors_upserted": vectors_upserted,
        }
        logger.success(f"Ingestion complete: {stats}")
        return stats

    def _ensure_collection(self, recreate: bool = False) -> None:
        exists = any(
            c.name == self.collection_name
            for c in self._qdrant.get_collections().collections
        )
        if exists and recreate:
            self._qdrant.delete_collection(self.collection_name)
            exists = False

        if not exists:
            vector_size = self._embedder.embedding_dim
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' "
                f"(vector_size={vector_size})"
            )

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        import hashlib
        h = hashlib.md5(chunk_id.encode()).hexdigest()
        return int(h[:16], 16) % (2**53)
