"""
Evaluation dataset management.

A testset is a JSON file with this schema:
[
  {
    "question": "What is the revenue growth rate?",
    "ground_truth_answer": "The revenue grew by 12% YoY...",
    "relevant_chunk_ids": ["doc_1_chunk_3", "doc_2_chunk_7"],
    "metadata": {"difficulty": "hard", "category": "finance"}
  },
  ...
]

TestsetBuilder uses an LLM to auto-generate questions from ingested chunks,
which you then manually verify / filter. This is how you build a realistic
eval set without writing 200 questions by hand.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


QUESTION_GEN_PROMPT = """You are building an evaluation dataset for a RAG system.

Given this document passage, generate {num_questions} realistic questions that:
1. Can be answered using ONLY the information in this passage
2. Vary in complexity (simple factual, reasoning, multi-step)
3. Sound like questions a real user would ask

Passage:
{passage}

Source: {source}

Output ONLY a JSON array of question strings. Example:
["Question 1?", "Question 2?", "Question 3?"]"""


class EvalDataset:
    """Loads and manages an evaluation testset."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._items: List[Dict] = []
        if self.path.exists():
            self.load()

    def load(self) -> None:
        self._items = json.loads(self.path.read_text())
        logger.info(f"Loaded {len(self._items)} eval items from {self.path}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, indent=2))
        logger.info(f"Saved {len(self._items)} eval items to {self.path}")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def sample(self, n: int, seed: int = 42) -> "EvalDataset":
        """Return a random subsample as a new EvalDataset."""
        rng = random.Random(seed)
        subset = EvalDataset.__new__(EvalDataset)
        subset.path = self.path
        subset._items = rng.sample(self._items, min(n, len(self._items)))
        return subset

    def filter_by_category(self, category: str) -> "EvalDataset":
        subset = EvalDataset.__new__(EvalDataset)
        subset.path = self.path
        subset._items = [
            item for item in self._items
            if item.get("metadata", {}).get("category") == category
        ]
        return subset

    def add(self, item: Dict) -> None:
        self._items.append(item)

    def stats(self) -> Dict:
        categories = {}
        difficulties = {}
        for item in self._items:
            meta = item.get("metadata", {})
            cat = meta.get("category", "unknown")
            diff = meta.get("difficulty", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        return {
            "total": len(self._items),
            "categories": categories,
            "difficulties": difficulties,
            "with_ground_truth": sum(
                1 for i in self._items if i.get("ground_truth_answer")
            ),
            "with_chunk_ids": sum(
                1 for i in self._items if i.get("relevant_chunk_ids")
            ),
        }


class TestsetBuilder:
    """
    Generates evaluation questions from ingested chunks using an LLM.

    Workflow:
    1. Sample chunks from Qdrant
    2. For each chunk, generate N questions
    3. Save to testset JSON for human review
    4. Human annotates ground_truth_answer and relevant_chunk_ids
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        questions_per_chunk: int = 2,
    ):
        self.model = model
        self.questions_per_chunk = questions_per_chunk
        self._client = OpenAI()

    def build_from_chunks(
        self,
        chunks: List[Dict],
        output_path: str,
        max_chunks: Optional[int] = None,
        seed: int = 42,
    ) -> EvalDataset:
        """
        Generate questions from a list of chunk dicts.

        Args:
            chunks: List of {"text": ..., "chunk_id": ..., "metadata": ...}
            output_path: Where to save the testset JSON
            max_chunks: Cap on number of chunks to use (None = all)

        Returns:
            EvalDataset with generated questions (no ground truth yet)
        """
        rng = random.Random(seed)
        if max_chunks:
            chunks = rng.sample(chunks, min(max_chunks, len(chunks)))

        dataset = EvalDataset(output_path)

        for chunk in chunks:
            try:
                questions = self._generate_questions(
                    chunk["text"],
                    chunk.get("metadata", {}).get("source", "Unknown"),
                )
                for q in questions:
                    dataset.add({
                        "question": q,
                        "ground_truth_answer": None,    # Fill in manually
                        "relevant_chunk_ids": [chunk["chunk_id"]],
                        "source_chunk": chunk["chunk_id"],
                        "metadata": chunk.get("metadata", {}),
                    })
                logger.debug(f"Generated {len(questions)} questions for {chunk['chunk_id']}")
            except Exception as e:
                logger.warning(f"Question gen failed for {chunk.get('chunk_id')}: {e}")

        dataset.save()
        logger.success(
            f"Built testset with {len(dataset)} questions → {output_path}"
        )
        return dataset

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _generate_questions(self, passage: str, source: str) -> List[str]:
        prompt = QUESTION_GEN_PROMPT.format(
            passage=passage[:2000],  # Truncate very long chunks
            source=source,
            num_questions=self.questions_per_chunk,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8,
        )
        raw = response.choices[0].message.content.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        questions = json.loads(raw)
        return [q for q in questions if isinstance(q, str) and q.strip()]
