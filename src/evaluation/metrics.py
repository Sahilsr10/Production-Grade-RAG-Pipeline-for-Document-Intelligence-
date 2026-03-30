"""
Evaluation metrics for the RAG pipeline.

Retrieval metrics (need ground-truth relevant chunk IDs):
  - Recall@K, NDCG@K, MRR

Generation metrics (LLM-as-judge via local Mistral):
  - Faithfulness: are all claims supported by context?
  - Answer Relevance: does the answer address the question?
  - Answer Correctness: semantic similarity to ground truth
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from openai import OpenAI   # points to Ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.prompts import FAITHFULNESS_JUDGE_PROMPT, RELEVANCE_JUDGE_PROMPT


# ── Retrieval metrics ──────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, cid in enumerate(top_k) if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    results["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    return results


# ── Generation metrics (local Mistral as judge) ───────────────────────────

class GenerationEvaluator:
    """
    Evaluates generated answers using local Mistral as judge.

    Uses the same Ollama endpoint as the generator — fully local, free.
    """

    def __init__(
        self,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434/v1",
        batch_size: int = 10,
    ):
        self.ollama_model = ollama_model
        self._client = OpenAI(base_url=ollama_base_url, api_key="ollama")

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        results = {}
        results["faithfulness"] = self._evaluate_faithfulness(query, answer, context)
        results["answer_relevance"] = self._evaluate_relevance(query, answer)
        if ground_truth:
            results["answer_correctness"] = self._semantic_similarity(answer, ground_truth)
        return results

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
    def _evaluate_faithfulness(self, query: str, answer: str, context: str) -> float:
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            context=context, query=query, answer=answer
        )
        response = self._client.chat.completions.create(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        try:
            raw = response.choices[0].message.content.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                return float(data.get("score", 0.0))
        except Exception:
            pass
        return 0.0

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
    def _evaluate_relevance(self, query: str, answer: str) -> float:
        prompt = RELEVANCE_JUDGE_PROMPT.format(query=query, answer=answer)
        response = self._client.chat.completions.create(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        try:
            raw = response.choices[0].message.content.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                return float(data.get("score", 0.0))
        except Exception:
            pass
        return 0.0

    def _semantic_similarity(self, answer: str, ground_truth: str) -> float:
        a_tokens = set(answer.lower().split())
        g_tokens = set(ground_truth.lower().split())
        if not g_tokens:
            return 0.0
        intersection = a_tokens & g_tokens
        precision = len(intersection) / len(a_tokens) if a_tokens else 0
        recall = len(intersection) / len(g_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {
        k: float(np.mean([m[k] for m in metric_list if k in m]))
        for k in keys
    }
