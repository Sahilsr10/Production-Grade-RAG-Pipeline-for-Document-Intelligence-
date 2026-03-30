"""
Pydantic schemas for the FastAPI request/response layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User question")
    collection: str = Field(default="rag_documents", description="Qdrant collection to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return")
    use_hyde: bool = Field(default=True, description="Enable HyDE query expansion")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    metadata_filter: Optional[Dict] = Field(
        default=None,
        description="Optional metadata filter, e.g. {'source': 'report_2024.pdf'}",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was the revenue growth in Q3?",
                "collection": "rag_documents",
                "top_k": 5,
                "use_hyde": True,
                "use_reranker": True,
            }
        }


class SourceDoc(BaseModel):
    index: int
    chunk_id: str
    source: str
    title: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDoc]
    latency: Dict[str, float]
    model: str
    context_tokens: int

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was the revenue growth in Q3?",
                "answer": "Revenue grew by 12% YoY in Q3 [1], driven primarily...",
                "sources": [
                    {
                        "index": 1,
                        "chunk_id": "report_2024_chunk_5",
                        "source": "report_2024.pdf",
                        "title": "Q3 Report 2024",
                        "page": 12,
                        "score": 0.91,
                    }
                ],
                "latency": {
                    "total_ms": 1240.5,
                    "query_transform_ms": 420.1,
                    "retrieval_ms": 180.3,
                    "rerank_ms": 210.8,
                    "generation_ms": 429.3,
                },
                "model": "gpt-4o",
                "context_tokens": 2140,
            }
        }


class IngestRequest(BaseModel):
    source: str = Field(..., description="Path to file/directory or URL to ingest")
    collection: str = Field(default="rag_documents")
    recreate_collection: bool = Field(
        default=False, description="Drop and recreate the collection"
    )


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    vectors_upserted: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str = "1.0.0"
