"""
Document loaders for various source types.
Each loader returns a list of dicts: {text, metadata}
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger


class BaseLoader:
    def load(self, source: str) -> List[Dict]:
        raise NotImplementedError

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _make_doc_id() -> str:
        return str(uuid.uuid4())[:8]


class PDFLoader(BaseLoader):
    """Load and extract text from PDF files."""

    def load(self, source: str) -> List[Dict]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Run: pip install pypdf")

        path = Path(source)
        reader = PdfReader(str(path))
        docs = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if len(text) < 50:
                continue
            docs.append({
                "text": text,
                "metadata": {
                    "source": str(path),
                    "title": path.stem,
                    "page": page_num + 1,
                    "total_pages": len(reader.pages),
                    "doc_id": f"{path.stem}_p{page_num + 1}",
                    "type": "pdf",
                },
            })

        logger.info(f"Loaded {len(docs)} pages from {path.name}")
        return docs


class URLLoader(BaseLoader):
    """Scrape and clean text from a web URL."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def load(self, source: str) -> List[Dict]:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        resp = requests.get(source, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string if soup.title else urlparse(source).netloc
        text = self._clean_text(soup.get_text(separator=" "))

        if len(text) < 100:
            logger.warning(f"Very little text extracted from {source}")

        return [{
            "text": text,
            "metadata": {
                "source": source,
                "title": title,
                "doc_id": self._make_doc_id(),
                "type": "url",
            },
        }]


class TextLoader(BaseLoader):
    """Load plain text or markdown files."""

    def load(self, source: str) -> List[Dict]:
        path = Path(source)
        text = self._clean_text(path.read_text(encoding="utf-8"))
        return [{
            "text": text,
            "metadata": {
                "source": str(path),
                "title": path.stem,
                "doc_id": path.stem,
                "type": "text",
            },
        }]


class JSONLoader(BaseLoader):
    """
    Load from a JSON file. Expects either:
    - A list of {"text": ..., "metadata": ...} dicts
    - A list of strings
    - A dict with a "documents" key
    """

    def __init__(self, text_key: str = "text", metadata_keys: Optional[List[str]] = None):
        self.text_key = text_key
        self.metadata_keys = metadata_keys or []

    def load(self, source: str) -> List[Dict]:
        data = json.loads(Path(source).read_text())

        if isinstance(data, dict) and "documents" in data:
            data = data["documents"]

        docs = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                docs.append({
                    "text": item,
                    "metadata": {"doc_id": f"doc_{i}", "source": source, "type": "json"},
                })
            elif isinstance(item, dict):
                text = item.get(self.text_key, "")
                metadata = {k: item.get(k) for k in self.metadata_keys if k in item}
                metadata.setdefault("doc_id", f"doc_{i}")
                metadata.setdefault("source", source)
                metadata.setdefault("type", "json")
                docs.append({"text": text, "metadata": metadata})

        logger.info(f"Loaded {len(docs)} documents from {source}")
        return docs


class DirectoryLoader(BaseLoader):
    """Recursively load all supported files in a directory."""

    LOADER_MAP = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".json": JSONLoader,
    }

    def load(self, source: str) -> List[Dict]:
        path = Path(source)
        all_docs = []

        for file_path in sorted(path.rglob("*")):
            suffix = file_path.suffix.lower()
            if suffix not in self.LOADER_MAP:
                continue
            loader = self.LOADER_MAP[suffix]()
            try:
                docs = loader.load(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(all_docs)} documents from directory {source}")
        return all_docs


def get_loader(source: str) -> BaseLoader:
    """Auto-detect the right loader for a source path/URL."""
    if source.startswith("http://") or source.startswith("https://"):
        return URLLoader()
    path = Path(source)
    if path.is_dir():
        return DirectoryLoader()
    ext = path.suffix.lower()
    loaders = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".json": JSONLoader,
    }
    if ext in loaders:
        return loaders[ext]()
    raise ValueError(f"No loader available for: {source}")
