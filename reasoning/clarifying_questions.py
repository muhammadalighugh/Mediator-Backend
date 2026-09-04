"""
clarifying_questions.py
-----------------------
Generates a single neutral, non-accusatory question for a ContradictionFlag.
Max 25 words. Question is addressed to both parties.
"""

from __future__ import annotations

import logging

from models.schemas import Claim, ContradictionFlag
from reasoning.llm_client import llm_client

logger = logging.getLogger(__name__)

_QUESTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
    },
    "required": ["question"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are a professional mediator. Given a contradiction between two statements in an \
argument, generate exactly ONE neutral, non-accusatory clarifying question addressed \
to BOTH parties.

Rules:
- Maximum 25 words.
- Neutral tone — do not imply either party is wrong.
- The question should invite both parties to clarify their position or find common \
ground.
- Do not repeat the claims verbatim.
- Output only the JSON object — no prose.\
"""


def _build_user_prompt(
    contradiction: ContradictionFlag,
    claim_a: Claim | None,
    claim_b: Claim | None,
) -> str:
    lines = [
        f"Contradiction: {contradiction.description}",
        "",
    ]
    if claim_a:
        lines.append(f'Party A says: "{claim_a.verbatim_quote}"')
    if claim_b:
        lines.append(f'Party B says: "{claim_b.verbatim_quote}"')
    return "\n".join(lines)


async def generate_question(
    contradiction: ContradictionFlag,
    claim_a: Claim | None = None,
    claim_b: Claim | None = None,
) -> str:
    """Return a single clarifying question string (≤25 words)."""
    result = await llm_client.complete_json(
        system=_SYSTEM_PROMPT,
        user=_build_user_prompt(contradiction, claim_a, claim_b),
        schema=_QUESTION_SCHEMA,
        max_tokens=100,
    )
    question = result.get("question", "")
    # Hard truncation safeguard: trim to 25 words if LLM overshoots
    words = question.split()
    if len(words) > 25:
        question = " ".join(words[:25])
        if not question.endswith("?"):
            question += "?"
    logger.debug("Generated clarifying question: %s", question)
    return question
