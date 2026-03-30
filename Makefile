.PHONY: help install test test-all lint format docker-up docker-down ingest evaluate app

help:
	@echo "Production RAG Pipeline — available commands:"
	@echo ""
	@echo "  make install       Install all dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make test-all      Run unit + integration tests"
	@echo "  make lint          Lint with ruff"
	@echo "  make format        Auto-format with ruff"
	@echo "  make docker-up     Start Qdrant via docker-compose"
	@echo "  make docker-down   Stop all services"
	@echo "  make ingest        Ingest sample data (set SOURCE= to override)"
	@echo "  make evaluate      Run evaluation suite"
	@echo "  make app           Launch Streamlit demo"
	@echo "  make api           Launch FastAPI server"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v -m "unit or integration"

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

docker-up:
	docker-compose -f docker/docker-compose.yml up -d qdrant
	@echo "Qdrant running at http://localhost:6333"

docker-down:
	docker-compose -f docker/docker-compose.yml down

ingest:
	@SOURCE ?= data/raw/
	python scripts/ingest.py --source $(or $(SOURCE),data/raw/)

build-testset:
	python scripts/build_testset.py --collection rag_documents --num-chunks 100

evaluate:
	python scripts/evaluate.py --testset data/testset.json

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

app:
	streamlit run app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml
