"""
Integration tests for the end-to-end pipeline.

These tests require:
  - OPENAI_API_KEY in environment
  - Qdrant running at QDRANT_URL (default: http://localhost:6333)

Run with:
  pytest tests/integration/ -v -m integration

Skip if dependencies unavailable:
  pytest tests/integration/ --ignore-glob="*" (CI without API keys)
"""

import os

import pytest

pytestmark = pytest.mark.integration

# Skip entire module if no API key
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)


SAMPLE_DOCS = [
    {
        "text": (
            "Acme Corporation reported record revenues of $4.2 billion in fiscal year 2024, "
            "representing a 23% increase compared to the prior year. The growth was driven "
            "primarily by strong performance in the cloud services division, which grew 45% YoY. "
            "CEO Jane Smith attributed the success to strategic investments in AI infrastructure."
        ),
        "metadata": {
            "doc_id": "acme_2024_q4",
            "title": "Acme FY2024 Annual Report",
            "source": "acme_annual_report.pdf",
            "page": 5,
        },
    },
    {
        "text": (
            "The company's operating expenses increased by 15% to $2.1 billion, with R&D spending "
            "accounting for the largest share at $680 million. Capital expenditures totaled $320 million, "
            "focused primarily on data center expansion in three new geographic regions."
        ),
        "metadata": {
            "doc_id": "acme_2024_q4_expenses",
            "title": "Acme FY2024 Annual Report",
            "source": "acme_annual_report.pdf",
            "page": 8,
        },
    },
]

TEST_QUERY = "What was Acme Corporation's revenue in 2024?"
TEST_COLLECTION = "integration_test_collection"


@pytest.fixture(scope="module")
def ingestion_pipeline():
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=TEST_COLLECTION,
        bm25_index_path="/tmp/test_bm25.pkl",
    )
    return pipeline


@pytest.fixture(scope="module")
def retriever(ingestion_pipeline):
    """Ingest sample docs once, then return a retriever."""
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import BM25Index, DenseEmbedder
    from src.ingestion.chunker import Chunk

    # Manually chunk + ingest the sample docs for speed
    chunker = SemanticChunker(drift_threshold=0.3, min_chunk_tokens=20)
    embedder = DenseEmbedder()
    bm25 = BM25Index()

    all_chunks = []
    for doc in SAMPLE_DOCS:
        chunks = chunker.chunk(doc["text"], metadata=doc["metadata"])
        all_chunks.extend(chunks)

    ingestion_pipeline._ensure_collection(recreate=True)

    from qdrant_client.models import PointStruct
    embeddings = embedder.embed([c.text for c in all_chunks])
    points = [
        PointStruct(
            id=ingestion_pipeline._chunk_id_to_int(c.chunk_id),
            vector=embeddings[i].tolist(),
            payload={"text": c.text, "chunk_id": c.chunk_id, **c.metadata},
        )
        for i, c in enumerate(all_chunks)
    ]
    ingestion_pipeline._qdrant.upsert(collection_name=TEST_COLLECTION, points=points)

    bm25.build(all_chunks)
    bm25.save("/tmp/test_bm25.pkl")

    from src.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(
        collection_name=TEST_COLLECTION,
        bm25_index_path="/tmp/test_bm25.pkl",
    )


def test_retrieval_finds_relevant_chunks(retriever):
    from src.retrieval.query_transformer import QueryTransformer

    transformer = QueryTransformer(use_hyde=False, use_decomposition=False)
    transformed = transformer.transform(TEST_QUERY)
    candidates = retriever.retrieve(transformed)

    assert len(candidates) >= 1
    # Top result should mention revenue
    top_text = candidates[0].text.lower()
    assert "revenue" in top_text or "billion" in top_text


def test_reranker_reduces_candidates(retriever):
    from src.retrieval.query_transformer import QueryTransformer
    from src.retrieval.reranker import CrossEncoderReranker

    transformer = QueryTransformer(use_hyde=False, use_decomposition=False)
    transformed = transformer.transform(TEST_QUERY)
    candidates = retriever.retrieve(transformed)

    reranker = CrossEncoderReranker(top_k=1)
    final = reranker.rerank(TEST_QUERY, candidates)

    assert len(final) == 1
    assert final[0].rerank_score != 0.0


def test_full_pipeline_generates_answer(retriever):
    from src.generation.context_builder import ContextBuilder
    from src.generation.generator import AnswerGenerator
    from src.retrieval.query_transformer import QueryTransformer
    from src.retrieval.reranker import CrossEncoderReranker

    transformer = QueryTransformer(use_hyde=False, use_decomposition=False)
    reranker = CrossEncoderReranker(top_k=2)
    context_builder = ContextBuilder(max_tokens=1000)
    generator = AnswerGenerator(model="gpt-4o-mini", max_tokens=200)

    transformed = transformer.transform(TEST_QUERY)
    candidates = retriever.retrieve(transformed)
    final = reranker.rerank(TEST_QUERY, candidates)
    context = context_builder.build(final)
    answer = generator.generate(TEST_QUERY, context)

    assert len(answer.answer) > 20
    # Answer should mention revenue or financial figures
    assert any(
        word in answer.answer.lower()
        for word in ["revenue", "billion", "4.2", "2024", "acme"]
    )
    assert len(answer.sources) >= 1
