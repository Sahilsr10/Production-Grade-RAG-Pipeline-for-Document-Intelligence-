"""
FastAPI application: production RAG serving layer (fully local stack).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.middleware import latency_middleware
from src.api.models import (
    HealthResponse, IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, SourceDoc,
)
from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring.tracer import Tracer
from src.monitoring.wandb_logger import WandbLogger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import CrossEncoderReranker

load_dotenv()

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_documents")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
BM25_PATH        = os.getenv("BM25_INDEX_PATH", "data/processed/bm25.pkl")
WANDB_ENABLED    = bool(os.getenv("WANDB_API_KEY"))

_pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing local RAG pipeline (BGE + Qdrant + MiniLM + Mistral)...")

    _pipeline["retriever"] = HybridRetriever(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        bm25_index_path=BM25_PATH,
        embed_model=EMBED_MODEL,
    )
    _pipeline["reranker"] = CrossEncoderReranker(top_k=5)
    _pipeline["context_builder"] = ContextBuilder(max_tokens=3000)
    _pipeline["generator"] = AnswerGenerator(
        model=OLLAMA_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
    )
    _pipeline["wandb"] = WandbLogger(
        project=os.getenv("WANDB_PROJECT", "production-rag"),
        enabled=WANDB_ENABLED,
    )
    logger.success("Pipeline ready.")
    yield
    if WANDB_ENABLED:
        _pipeline["wandb"].finish()


app = FastAPI(
    title="Production RAG Pipeline",
    description="Fully local: BGE embeddings · Qdrant+BM25 retrieval · MiniLM reranker · Mistral generation",
    version="2.0.0",
    lifespan=lifespan,
)
app.middleware("http")(latency_middleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        cols = client.get_collections()
        qdrant_status = f"ok ({len(cols.collections)} collections)"
    except Exception as e:
        qdrant_status = f"error: {e}"
    return HealthResponse(status="ok", qdrant=qdrant_status, version="2.0.0")


@app.get("/metrics", tags=["Ops"])
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    tracer = Tracer()
    retriever: HybridRetriever      = _pipeline["retriever"]
    reranker: CrossEncoderReranker  = _pipeline["reranker"]
    ctx_builder: ContextBuilder     = _pipeline["context_builder"]
    generator: AnswerGenerator      = _pipeline["generator"]
    wandb: WandbLogger              = _pipeline["wandb"]

    try:
        with tracer.stage("query_transform"):
            transformer = QueryTransformer(
                use_hyde=request.use_hyde,
                model=OLLAMA_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
            )
            transformed = transformer.transform(request.query)

        with tracer.stage("retrieval"):
            candidates = retriever.retrieve(transformed, request.metadata_filter)

        if not candidates:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        with tracer.stage("rerank"):
            final_chunks = (
                reranker.rerank(request.query, candidates)
                if request.use_reranker
                else candidates[: request.top_k]
            )

        with tracer.stage("context_build"):
            context = ctx_builder.build(final_chunks)

        with tracer.stage("generation"):
            answer = generator.generate(request.query, context)

        latency = {
            k: round(v["duration_ms"] if isinstance(v, dict) else v, 1)
            for k, v in {**tracer.summary()["stages"], "total_ms": tracer.total_ms}.items()
        }

        wandb.log_query(
            query=request.query,
            answer=answer.answer,
            latency=tracer.as_flat_dict(),
            retrieval_scores=[c.rerank_score for c in final_chunks],
        )

        return QueryResponse(
            query=request.query,
            answer=answer.answer,
            sources=[SourceDoc(**s) for s in answer.sources],
            latency=latency,
            model=answer.model,
            context_tokens=context.token_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(request: IngestRequest):
    try:
        pipeline = IngestionPipeline(
            qdrant_url=QDRANT_URL,
            collection_name=request.collection,
            bm25_index_path=BM25_PATH,
            embed_model=EMBED_MODEL,
        )
        stats = pipeline.ingest(
            source=request.source,
            recreate_collection=request.recreate_collection,
        )
        return IngestResponse(**stats, collection=request.collection)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
