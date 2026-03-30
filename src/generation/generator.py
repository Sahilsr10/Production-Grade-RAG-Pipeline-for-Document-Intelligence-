"""
Answer generator using Mistral via Ollama (fully local, zero cost).

Why Ollama + Mistral?
- Runs 100% locally — no API keys, no cost, no rate limits
- Mistral-7B-Instruct is the strongest 7B model for instruction following
- Ollama exposes an OpenAI-compatible API (same /v1/chat/completions endpoint)
  so the code is identical to the OpenAI version — just a different base_url
- Easy to swap: change OLLAMA_MODEL in .env to try mistral, llama3, gemma2, etc.

Ollama OpenAI-compat docs: https://ollama.com/blog/openai-compatibility

Interview quote:
  "I used Ollama to run Mistral locally. Ollama exposes an OpenAI-compatible
   REST API, so my generation code didn't change at all — I just pointed the
   base_url at localhost. This means I can swap to GPT-4 for production by
   changing a single environment variable."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger
from openai import OpenAI          # same client — Ollama is OpenAI-compatible
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.context_builder import Context
from src.generation.prompts import ANSWER_PROMPT, SYSTEM_PROMPT


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_token_count: int = 0


class AnswerGenerator:
    """
    Generates answers grounded in retrieved context using a local Mistral model.

    Uses the OpenAI client pointed at Ollama's local server.
    Ollama must be running: `ollama serve`
    Model must be pulled:   `ollama pull mistral`

    Args:
        model: Ollama model name. Options: mistral, llama3, gemma2, phi3, etc.
        ollama_base_url: Ollama server URL (default: http://localhost:11434/v1)
        max_tokens: Max tokens for the generated answer
        temperature: Lower = more factual/deterministic (0.1 recommended for RAG)
    """

    def __init__(
        self,
        model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434/v1",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Ollama uses OpenAI-compatible API — same client, different base_url
        # api_key="ollama" is a dummy value required by the openai client
        self._client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama",
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
    def generate(self, query: str, context: Context) -> GeneratedAnswer:
        """
        Generate a grounded, cited answer.

        Args:
            query: The original user query
            context: Assembled context from context_builder

        Returns:
            GeneratedAnswer with answer text and source attribution
        """
        user_message = ANSWER_PROMPT.format(
            context=context.text,
            query=query,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer_text = response.choices[0].message.content.strip()
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        logger.debug(
            f"Generated answer ({completion_tokens} tokens) "
            f"from {len(context.chunks)} context chunks via {self.model}"
        )

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            sources=context.sources,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_token_count=context.token_count,
        )
