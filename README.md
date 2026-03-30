# 🚀 Production-Grade RAG Pipeline for Document Intelligence

An end-to-end Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over multi-format documents (PDF, TXT, JSON) using hybrid retrieval, reranking, and LLM-based generation.

---

## ✨ Features

- 📄 Multi-format document ingestion (PDF, TXT, JSON)
- 🔍 Hybrid Retrieval (Dense + Sparse + RRF Fusion)
- 🧠 Embeddings using BGE (`BAAI/bge-base-en-v1.5`)
- 📊 BM25 Sparse Indexing
- ⚡ Cross-Encoder Reranking (MiniLM)
- 🧩 HyDE Query Expansion
- 🗄️ Vector Database with Qdrant
- 🤖 Local LLM inference using Ollama (Mistral)
- 🌐 FastAPI backend for serving queries
- 🎨 Streamlit UI for interactive querying
- 🎯 Metadata-based document filtering

---

## 🏗️ System Architecture

```
User Query
   ↓
Query Transformer (HyDE / Decomposition)
   ↓
Hybrid Retriever (Dense + BM25)
   ↓
RRF Fusion
   ↓
Cross-Encoder Reranker (MiniLM)
   ↓
Context Builder
   ↓
LLM (Mistral via Ollama)
   ↓
Final Answer + Sources
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector DB | Qdrant |
| Embeddings | BGE (BAAI/bge-base-en-v1.5) |
| Reranker | MiniLM (Cross-Encoder) |
| Sparse Retrieval | BM25 |
| LLM | Mistral via Ollama |
| Infrastructure | Docker |

---

## 📂 Project Structure

```
production-rag/
│
├── src/
│   ├── ingestion/         # Document loaders and chunking logic
│   ├── retrieval/         # Dense, sparse, hybrid retrieval modules
│   ├── generation/        # LLM prompt templates and answer generation
│   ├── evaluation/        # Evaluation metrics and test harness
│   ├── api/               # FastAPI route definitions
│   └── pipeline.py        # End-to-end pipeline orchestration
│
├── scripts/
│   ├── ingest.py          # CLI script to ingest documents
│   └── evaluate.py        # CLI script to run evaluation
│
├── data/
│   ├── raw/               # Source documents (PDF, TXT, JSON)
│   └── processed/         # Chunked and embedded documents
│
├── app.py                 # Streamlit UI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/your-username/production-rag.git
cd production-rag
```

### 2. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Start Ollama

```bash
ollama serve
ollama pull mistral
```

### 5. Ingest Documents

```bash
python scripts/ingest.py --source data/raw --collection test_docs
```

### 6. Run Backend API

```bash
uvicorn src.api.main:app --reload
```

### 7. Launch UI

```bash
streamlit run app.py
```

---

## 🧪 Example Queries

- *"What topics are covered in Unit III?"*
- *"Summarize key concepts from the uploaded document."*
- *"Compare classification and regression techniques."*

---

## 📊 Evaluation

```bash
python scripts/evaluate.py --testset data/testset.json
```

**Metrics tracked:**

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevance | Does the answer address the question? |
| Latency (p50, p95) | End-to-end response time percentiles |

---

## 🧠 Key Highlights

- Built a production-grade RAG pipeline from scratch
- Implemented hybrid retrieval (Dense + BM25 + RRF fusion)
- Integrated cross-encoder reranking for precision (MiniLM)
- Enabled metadata-aware filtering for document-specific QA
- Designed a fully local LLM system — **zero paid API costs**

---

## 📌 Future Improvements

- [ ] Multi-user support with session isolation
- [ ] Cloud deployment (AWS / GCP)
- [ ] Vector compression for scalability
- [ ] Direct file upload via Streamlit UI

---
