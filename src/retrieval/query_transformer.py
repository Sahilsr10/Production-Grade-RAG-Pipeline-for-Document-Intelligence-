"""
Query transformation strategies using local Mistral via Ollama.

1. HyDE (Hypothetical Document Embeddings):
   - Ask Mistral to generate a fake answer to the question
   - Embed that fake answer instead of the raw query
   - Rationale: a hypothetical answer lives closer to real answers in embedding space

2. Query Decomposition:
   - For multi-hop questions, break into independent sub-queries
   - Retrieve for each sub-query, merge before re-ranking

Both use Ollama's OpenAI-compatible endpoint — zero cost, fully local.
"""

from __future__ import annotations

import json
from typing import List

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

HYDE_PROMPT = """You are a helpful assistant. A user asked the following question:

"{query}"

Write a detailed, factual paragraph that would be a good answer to this question.
Write as if you are an authoritative document on the subject.
Be specific and informative. Do NOT say you don't know — make your best attempt.
Output ONLY the answer paragraph, no preamble."""

DECOMPOSE_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.

Given this question: "{query}"

Break it into {max_subqueries} or fewer independent sub-questions that, when answered together,
would answer the original question. Each sub-question should be answerable independently.

If the original question is already simple, return it as the only sub-question.

Output ONLY a JSON array of strings. Example: ["sub-question 1", "sub-question 2"]"""


class QueryTransformer:
    """
    Transforms queries before retrieval using a local Mistral model.

    Args:
        use_hyde: Whether to generate hypothetical document embeddings
        use_decomposition: Whether to decompose multi-hop queries
        model: Ollama model name for HyDE/decomposition generation
        ollama_base_url: Ollama server URL
        max_subqueries: Maximum number of sub-queries to decompose into
    """

    def __init__(
        self,
        use_hyde: bool = True,
        use_decomposition: bool = True,
        model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434/v1",
        max_subqueries: int = 3,
    ):
        self.use_hyde = use_hyde
        self.use_decomposition = use_decomposition
        self.model = model
        self.max_subqueries = max_subqueries
        self._client = OpenAI(base_url=ollama_base_url, api_key="ollama")

    def transform(self, query: str) -> "TransformedQuery":
        result = TransformedQuery(original=query)

        if self.use_hyde:
            try:
                result.hypothetical_doc = self._generate_hyde(query)
                logger.debug(f"HyDE generated ({len(result.hypothetical_doc)} chars)")
            except Exception as e:
                logger.warning(f"HyDE failed, using original query: {e}")

        if self.use_decomposition:
            try:
                result.sub_queries = self._decompose(query)
                logger.debug(f"Decomposed into {len(result.sub_queries)} sub-queries")
            except Exception as e:
                logger.warning(f"Decomposition failed, using original query: {e}")

        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _generate_hyde(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _decompose(self, query: str) -> List[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": DECOMPOSE_PROMPT.format(
                    query=query, max_subqueries=self.max_subqueries
                ),
            }],
            max_tokens=200,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        # Find JSON array in response (Mistral sometimes adds preamble)
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list):
            return [query]
        return [q for q in sub_queries if isinstance(q, str) and q.strip()]


class TransformedQuery:
    """Container for a query and all its transformations."""

    def __init__(self, original: str):
        self.original = original
        self.hypothetical_doc: str | None = None
        self.sub_queries: List[str] = [original]

    @property
    def queries_for_retrieval(self) -> List[str]:
        """All query strings to retrieve for (deduplicated)."""
        queries = list(self.sub_queries)
        if self.hypothetical_doc:
            queries.append(self.hypothetical_doc)
        return list(dict.fromkeys(queries))

    def __repr__(self) -> str:
        return (
            f"TransformedQuery(original={self.original!r}, "
            f"sub_queries={len(self.sub_queries)}, "
            f"hyde={'yes' if self.hypothetical_doc else 'no'})"
        )
