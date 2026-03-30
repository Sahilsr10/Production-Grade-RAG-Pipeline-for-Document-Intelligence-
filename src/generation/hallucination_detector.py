"""
Hallucination detector: post-generation faithfulness guard.

After the LLM generates an answer, this module checks whether each
factual claim in the answer is actually supported by the retrieved context.

This is the difference between a toy RAG and a production system —
you don't trust the LLM blindly, you verify its output.

Two modes:
1. Fast mode: NLI (Natural Language Inference) model — no API call needed
2. Accurate mode: LLM-as-judge — slower but more nuanced

The NLI model checks if context "entails" the answer claim.
Labels: ENTAILMENT (supported), NEUTRAL, CONTRADICTION (hallucinated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import nltk
import torch
from loguru import logger
from transformers import pipeline as hf_pipeline

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


@dataclass
class ClaimVerification:
    claim: str
    verdict: str          # "supported", "unsupported", "contradicted"
    confidence: float
    supporting_context: Optional[str] = None


@dataclass
class HallucinationReport:
    answer: str
    context: str
    claims: List[ClaimVerification]
    overall_faithfulness: float  # fraction of claims that are supported
    is_faithful: bool            # True if faithfulness >= threshold

    def __str__(self) -> str:
        lines = [f"Faithfulness: {self.overall_faithfulness:.0%}"]
        for c in self.claims:
            icon = "✓" if c.verdict == "supported" else "✗"
            lines.append(f"  {icon} [{c.verdict}] {c.claim[:80]}")
        return "\n".join(lines)


class HallucinationDetector:
    """
    Detects unsupported claims in generated answers using NLI.

    Model: cross-encoder/nli-deberta-v3-small
    - Fast, runs on CPU
    - Strong NLI performance on diverse text

    Args:
        model_name: HuggingFace NLI model
        faithfulness_threshold: Minimum fraction of supported claims to pass
        batch_size: NLI inference batch size
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        faithfulness_threshold: float = 0.8,
        batch_size: int = 16,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.batch_size = batch_size

        logger.info(f"Loading NLI model: {model_name}")
        self._nli = hf_pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
        )

    def check(self, answer: str, context: str) -> HallucinationReport:
        """
        Check whether the answer is supported by the context.

        Args:
            answer: Generated answer text
            context: Retrieved context that was given to the LLM

        Returns:
            HallucinationReport with per-claim verdicts
        """
        # Split answer into individual claims (sentences)
        claims = self._extract_claims(answer)
        if not claims:
            return HallucinationReport(
                answer=answer,
                context=context,
                claims=[],
                overall_faithfulness=1.0,
                is_faithful=True,
            )

        # Run NLI: does context entail each claim?
        verifications = self._verify_claims(claims, context)

        supported = sum(1 for v in verifications if v.verdict == "supported")
        faithfulness = supported / len(verifications) if verifications else 1.0

        report = HallucinationReport(
            answer=answer,
            context=context,
            claims=verifications,
            overall_faithfulness=faithfulness,
            is_faithful=faithfulness >= self.faithfulness_threshold,
        )

        if not report.is_faithful:
            logger.warning(
                f"Low faithfulness detected: {faithfulness:.0%} "
                f"({supported}/{len(verifications)} claims supported)"
            )

        return report

    def _extract_claims(self, text: str) -> List[str]:
        """Split text into individual verifiable sentences."""
        sentences = nltk.sent_tokenize(text)
        # Filter out very short sentences and hedging phrases
        claims = [
            s.strip() for s in sentences
            if len(s.split()) >= 5
            and not s.lower().startswith(("i don't", "i cannot", "the document"))
        ]
        return claims

    def _verify_claims(
        self, claims: List[str], context: str
    ) -> List[ClaimVerification]:
        """Run NLI on (context, claim) pairs."""
        # Build pairs: premise = context, hypothesis = claim
        pairs = [{"text": context[:1000], "text_pair": claim} for claim in claims]

        results: List[ClaimVerification] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            outputs = self._nli(batch)

            for claim, label_scores in zip(claims[i : i + self.batch_size], outputs):
                # label_scores is a list of {"label": ..., "score": ...}
                score_map = {item["label"].upper(): item["score"] for item in label_scores}

                entail_score = score_map.get("ENTAILMENT", 0.0)
                contradict_score = score_map.get("CONTRADICTION", 0.0)

                if entail_score > 0.5:
                    verdict = "supported"
                    confidence = entail_score
                elif contradict_score > 0.4:
                    verdict = "contradicted"
                    confidence = contradict_score
                else:
                    verdict = "unsupported"
                    confidence = 1 - entail_score

                results.append(
                    ClaimVerification(
                        claim=claim,
                        verdict=verdict,
                        confidence=confidence,
                    )
                )

        return results
