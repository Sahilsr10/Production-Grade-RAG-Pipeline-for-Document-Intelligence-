# Production-Grade-RAG-Pipeline-for-Document-Intelligence-


An end-to-end Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over multi-format documents (PDF, TXT, JSON) using hybrid retrieval, reranking, and LLM-based generation.

⸻

✨ Features
	•	📄 Multi-format document ingestion (PDF, TXT, JSON)
	•	🔍 Hybrid Retrieval (Dense + Sparse + RRF Fusion)
	•	🧠 Embeddings using BGE (BAAI/bge-base-en-v1.5)
	•	📊 BM25 Sparse Indexing
	•	⚡ Cross-Encoder Reranking (MiniLM)
	•	🧩 HyDE Query Expansion
	•	🗄️ Vector Database with Qdrant
	•	🤖 Local LLM inference using Ollama (Mistral)
	•	🌐 FastAPI backend for serving queries
	•	🎨 Streamlit UI for interactive querying
	•	🎯 Metadata-based document filtering

⸻

🏗️ System Architecture

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


⸻

⚙️ Tech Stack
	•	Language: Python
	•	Backend: FastAPI
	•	Frontend: Streamlit
	•	Vector DB: Qdrant
	•	Embeddings: BGE (BAAI)
	•	Reranker: MiniLM (Cross-Encoder)
	•	Sparse Retrieval: BM25
	•	LLM: Mistral via Ollama
	•	Infra: Docker

⸻

📂 Project Structure

production-rag/
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   ├── api/
│   └── pipeline.py
│
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── app.py
├── requirements.txt
└── README.md


⸻

🚀 Getting Started

1. Clone Repository

git clone https://github.com/your-username/production-rag.git
cd production-rag


⸻

2. Setup Virtual Environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


⸻

3. Start Qdrant

docker run -p 6333:6333 qdrant/qdrant


⸻

4. Start Ollama

ollama serve
ollama pull mistral


⸻

5. Ingest Documents

python scripts/ingest.py --source data/raw --collection test_docs


⸻

6. Run Backend API

uvicorn src.api.main:app --reload


⸻

7. Launch UI

streamlit run app.py


⸻

🧪 Example Queries
	•	What topics are covered in Unit III?
	•	Summarize key concepts from the uploaded document
	•	Compare classification and regression techniques

⸻

📊 Evaluation

python scripts/evaluate.py --testset data/testset.json

Metrics:
	•	Faithfulness
	•	Answer relevance
	•	Latency (p50, p95)

⸻

🧠 Key Highlights
	•	Built a production-grade RAG pipeline from scratch
	•	Implemented hybrid retrieval (Dense + BM25 + RRF)
	•	Integrated cross-encoder reranking (MiniLM)
	•	Enabled metadata-aware filtering for document-specific QA
	•	Designed a fully local LLM system (no paid APIs)

⸻

📌 Future Improvements
	•	Multi-user support
	•	Cloud deployment (AWS/GCP)
	•	Vector compression for scalability
	•	File upload directly in UI

⸻

