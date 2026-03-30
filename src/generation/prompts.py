"""
Prompt templates for answer generation.

Design principles:
- Instruct model to cite sources inline using [N] notation
- Instruct model to say "I don't know" if context doesn't contain the answer
  (prevents hallucination)
- Chain-of-thought reasoning before final answer
- Separate system and user prompts for cleaner role boundaries
"""

SYSTEM_PROMPT = """You are a precise, factual question-answering assistant.
You answer questions strictly based on the provided context documents.

Rules:
1. Cite your sources inline using [N] notation, where N is the document number.
2. If the context does not contain enough information to answer, say:
   "The provided documents do not contain sufficient information to answer this question."
3. Do NOT invent information. Do NOT use knowledge outside the provided context.
4. Be concise but complete. Use bullet points for multi-part answers.
5. If multiple sources say different things, acknowledge the discrepancy."""

ANSWER_PROMPT = """Context documents:
{context}

---

Question: {query}

Think step-by-step about which context documents are relevant, then write your answer."""

FAITHFULNESS_JUDGE_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Evaluate:
1. Does every factual claim in the answer appear in the context? (Yes/No for each claim)
2. Overall faithfulness score from 0.0 to 1.0 (1.0 = fully supported, 0.0 = hallucinated)

Output JSON: {{"claims": [{{"claim": "...", "supported": true/false}}], "score": 0.0-1.0}}"""

RELEVANCE_JUDGE_PROMPT = """Rate how well this answer addresses the question.

Question: {query}
Answer: {answer}

Score from 0.0 to 1.0:
- 1.0: Directly and completely answers the question
- 0.7: Mostly answers but misses some aspects
- 0.4: Partially relevant
- 0.0: Does not address the question

Output JSON: {{"reasoning": "...", "score": 0.0-1.0}}"""
