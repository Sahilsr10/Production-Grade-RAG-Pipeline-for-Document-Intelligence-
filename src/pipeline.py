"""
RAGPipeline: high-level orchestrator — fully local, zero cost.

Stack:
  Embeddings  → BAAI/bge-base-en-v1.5 (local, sentence-transformers)
  Retrieval   → Qdrant dense + BM25 sparse, fused via RRF
  Reranker    → cross-encoder/ms-marco-MiniLM-L-6-v2 (local, HuggingFace)
  Generator   → Mistral via Ollama (local, OpenAI-compat API)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator, GeneratedAnswer
from src.monitoring.tracer import Tracer
from src.monitoring.wandb_logger import WandbLogger
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import CrossEncoderReranker

load_dotenv()


@dataclass
class PipelineConfig:
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "rag_documents"
    bm25_index_path: str = "data/processed/bm25.pkl"
    embed_model: str = "BAAI/bge-base-en-v1.5"
    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434/v1"
    use_hyde: bool = True
    use_decomposition: bool = True
    use_reranker: bool = True
    top_k_retrieve: int = 50
    top_k_rerank: int = 5
    max_context_tokens: int = 3000
    wandb_project: str = "production-rag"
    wandb_enabled: bool = False


@dataclass
class PipelineResult:
    query: str
    answer: str
    sources: List[dict]
    retrieved_chunks: List[RetrievedChunk]
    latency: Dict[str, float]
    model: str
    context_tokens: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class RAGPipeline:
    """
    Fully local production RAG pipeline. No API keys required.

    Usage:
        pipeline = RAGPipeline.from_env()
        result = pipeline.query("What was the revenue in Q3?")
        print(result.answer)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info("Initializing fully-local RAG pipeline...")

        self.retriever = HybridRetriever(
            qdrant_url=config.qdrant_url,
            collection_name=config.collection_name,
            bm25_index_path=config.bm25_index_path,
            embed_model=config.embed_model,
            top_k_dense=config.top_k_retrieve,
            top_k_sparse=config.top_k_retrieve,
            top_k_fused=config.top_k_retrieve,
        )
        self.reranker = CrossEncoderReranker(top_k=config.top_k_rerank)
        self.context_builder = ContextBuilder(max_tokens=config.max_context_tokens)
        self.generator = AnswerGenerator(
            model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )
        self.wandb = WandbLogger(
            project=config.wandb_project,
            enabled=config.wandb_enabled,
        )
        self._query_count = 0
        logger.success("Pipeline ready. Stack: BGE + Qdrant/BM25 + MiniLM + Mistral")

    @classmethod
    def from_env(cls) -> "RAGPipeline":
        config = PipelineConfig(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("COLLECTION_NAME", "rag_documents"),
            bm25_index_path=os.getenv("BM25_INDEX_PATH", "data/processed/bm25.pkl"),
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            use_hyde=os.getenv("USE_HYDE", "true").lower() == "true",
            use_decomposition=os.getenv("USE_DECOMPOSITION", "true").lower() == "true",
            use_reranker=os.getenv("USE_RERANKER", "true").lower() == "true",
            wandb_enabled=bool(os.getenv("WANDB_API_KEY")),
        )
        return cls(config)

    def query(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        use_hyde: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
    ) -> PipelineResult:
        self._query_count += 1
        tracer = Tracer()
        _use_hyde = use_hyde if use_hyde is not None else self.config.use_hyde
        _use_reranker = use_reranker if use_reranker is not None else self.config.use_reranker

        try:
            with tracer.stage("query_transform"):
                transformer = QueryTransformer(
                    use_hyde=_use_hyde,
                    use_decomposition=self.config.use_decomposition,
                    model=self.config.ollama_model,
                    ollama_base_url=self.config.ollama_base_url,
                )
                transformed = transformer.transform(query)

            with tracer.stage("retrieval"):
                candidates = self.retriever.retrieve(transformed, metadata_filter)

            if not candidates:
                return PipelineResult(
                    query=query,
                    answer="No relevant documents found for this query.",
                    sources=[], retrieved_chunks=[],
                    latency=tracer.as_flat_dict(),
                    model=self.config.ollama_model,
                    context_tokens=0, error="no_results",
                )

            with tracer.stage("rerank"):
                if _use_reranker:
                    try:
                        final_chunks = self.reranker.rerank(query, candidates)
                    except Exception as e:
                        logger.warning(f"Reranker failed, falling back: {e}")
                        final_chunks = candidates[: self.config.top_k_rerank]
                else:
                    final_chunks = candidates[: self.config.top_k_rerank]

            with tracer.stage("context_build"):
                context = self.context_builder.build(final_chunks)

            with tracer.stage("generation"):
                answer = self.generator.generate(query, context)

            return PipelineResult(
                query=query,
                answer=answer.answer,
                sources=answer.sources,
                retrieved_chunks=final_chunks,
                latency=tracer.as_flat_dict(),
                model=answer.model,
                context_tokens=context.token_count,
            )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            return PipelineResult(
                query=query,
                answer=f"Error: {str(e)}",
                sources=[], retrieved_chunks=[],
                latency=tracer.as_flat_dict(),
                model=self.config.ollama_model,
                context_tokens=0, error=str(e),
            )

    def batch_query(self, queries: List[str], **kwargs) -> List[PipelineResult]:
        return [self.query(q, **kwargs) for q in queries]
