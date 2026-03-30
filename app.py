"""
Streamlit demo UI for the Production RAG Pipeline.

Run with:
  streamlit run app.py

Features:
- Interactive query interface with source attribution
- Pipeline configuration toggles (HyDE, reranker on/off)
- Per-stage latency breakdown
- Retrieved chunks inspector
- Evaluation metrics dashboard
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

st.set_page_config(
    page_title="Production RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #6366f1;
    }
    .source-card {
        background: #f0f4ff;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dde1f0;
    }
    .chunk-preview {
        font-family: monospace;
        font-size: 0.85rem;
        color: #444;
        background: #f5f5f5;
        padding: 0.5rem;
        border-radius: 4px;
    }
    .latency-bar {
        height: 8px;
        border-radius: 4px;
        background: #6366f1;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline initialization (cached — only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline():
    try:
        from src.pipeline import RAGPipeline
        # Use the same initialization as API to avoid config drift
        pipeline = RAGPipeline.from_env()
        return pipeline, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline settings")

    use_hyde = st.toggle("HyDE query expansion", value=True,
                          help="Generate a hypothetical answer, then embed it for retrieval")
    use_reranker = st.toggle("Cross-encoder reranker", value=True,
                              help="Re-score top-50 candidates with a cross-encoder (slower but more accurate)")
    top_k = st.slider("Sources to show", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("### 🔍 Filter by source")
    source_filter = st.text_input(
        "Source filename (optional)",
        placeholder="e.g. pbi_syll.pdf",
        help="Enter just the filename (not full path). Filtering will match documents containing this filename."
    )

    st.divider()
    st.markdown("### 📂 Upload & Ingest")

    uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt", "md", "json"])

    if uploaded_file:
        save_path = Path("data/raw") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("⚡ Ingest uploaded files"):
        with st.spinner("Running ingestion..."):
            os.system("python scripts/ingest.py --source data/raw --collection test_docs")
        st.success("Ingestion complete! You can now query your documents.")

    st.divider()
    st.markdown("### 📊 About")
    st.markdown("""
    **Stack:**
    - BGE (`BAAI/bge-base-en-v1.5`) embeddings
    - Qdrant vector store
    - BM25 + RRF hybrid retrieval
    - `ms-marco-MiniLM` cross-encoder
    - Local LLM (Mistral via Ollama)
    
    **[GitHub](https://github.com/your-repo)** · **[Docs](https://your-docs.com)**
    """)


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Production RAG Pipeline</div>', unsafe_allow_html=True)
st.caption("Hybrid retrieval · Cross-encoder re-ranking · HyDE query expansion · Cited answers")

pipeline, init_error = load_pipeline()

if init_error:
    st.error(f"⚠️ Pipeline initialization failed:\n\n```\n{init_error}\n```")
    st.info("Make sure Qdrant is running and your `.env` file has valid API keys.")
    st.stop()

# Query input
query = st.text_area(
    "Ask a question",
    placeholder="e.g. What was the company's revenue growth in Q3?",
    height=80,
    key="query_input",
)

col1, col2, col3 = st.columns([2, 1, 4])
with col1:
    submit = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

# Example queries
with st.expander("💡 Example queries"):
    examples = [
        "What are the key findings from the latest report?",
        "Summarize the main risks identified in the document.",
        "What recommendations were made for Q4?",
        "Compare the performance metrics from the two quarters.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            query = ex
            submit = True

# ---------------------------------------------------------------------------
# Session state initialization for history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if submit and query.strip():
    metadata_filter = None
    if source_filter.strip():
        # match by filename instead of full path
        metadata_filter = {"source": source_filter.strip()}

    with st.spinner("🔍 Retrieving and generating answer..."):
        result = pipeline.query(
            query=query.strip(),
            metadata_filter=metadata_filter,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
        )

    if not result.success:
        st.error(f"Pipeline error: {result.error}")
    else:
        # Apply client-side filtering (filename match)
        if metadata_filter:
            filtered_sources = []
            for src in result.sources:
                if metadata_filter["source"] in (src.get("source") or ""):
                    filtered_sources.append(src)
            result.sources = filtered_sources
        # Answer
        st.markdown("### 💬 Answer")
        st.markdown(result.answer)

        st.session_state.history.append((query, result.answer))

        st.markdown("### 🧠 Chat History")
        for q, a in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.divider()

        st.divider()

        # Metrics row
        latency_total = result.latency.get("latency/total_ms", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total latency", f"{latency_total:.0f} ms")
        col2.metric("Sources used", len(result.sources))
        col3.metric("Context tokens", result.context_tokens)
        col4.metric("Model", result.model.replace("gpt-", "GPT-"))

        # Latency breakdown
        with st.expander("⏱ Latency breakdown"):
            stage_map = {
                "latency/query_transform_ms": "Query transform",
                "latency/retrieval_ms": "Hybrid retrieval",
                "latency/rerank_ms": "Cross-encoder rerank",
                "latency/context_build_ms": "Context assembly",
                "latency/generation_ms": "LLM generation",
            }
            for key, label in stage_map.items():
                val = result.latency.get(key, 0)
                pct = int(val / max(latency_total, 1) * 100)
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{label}**")
                cols[1].markdown(f"`{val:.0f} ms`")
                st.progress(pct / 100)

        # Sources
        st.markdown("### 📚 Sources")
        for src in result.sources[:top_k]:
            with st.container():
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>[{src["index"]}] {src["title"] or src["source"]}</strong>'
                    + (f' · Page {src["page"]}' if src.get("page") else "")
                    + f' · Score: <code>{src["score"]:.3f}</code>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Retrieved chunks inspector
        with st.expander("🔬 Inspect retrieved chunks"):
            if result.retrieved_chunks:
                for i, chunk in enumerate(result.retrieved_chunks[:top_k], 1):
                    st.markdown(f"**Chunk {i}** — `{chunk.chunk_id}` — rerank score: `{chunk.rerank_score:.3f}`")
                    st.markdown(
                        f'<div class="chunk-preview">{chunk.text[:400]}{"..." if len(chunk.text) > 400 else ""}</div>',
                        unsafe_allow_html=True,
                    )
                    st.divider()
            else:
                st.info("No chunks to inspect.")

        # Pipeline config used
        with st.expander("🛠 Pipeline config for this query"):
            st.json({
                "use_hyde": use_hyde,
                "use_reranker": use_reranker,
                "metadata_filter": metadata_filter,
                "model": result.model,
                "context_tokens": result.context_tokens,
            })

elif submit and not query.strip():
    st.warning("Please enter a question.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Production RAG Pipeline · "
    "Hybrid dense+sparse retrieval · "
    "Cross-encoder re-ranking · "
    "HyDE query expansion · "
    "LLM-as-judge evaluation"
)
