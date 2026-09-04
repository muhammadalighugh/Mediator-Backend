"""
contradiction_detector.py
--------------------------
Detects pairs of conflicting claims and identifies common ground.

Returns:
  contradictions  — list[ContradictionFlag]
  agreements      — list[str]  (common-ground summaries)
  dispute_type    — "FACTUAL" | "PRIORITIES" | "MISUNDERSTANDING" | "MIXED"
"""

from __future__ import annotations

import logging
import uuid

from models.schemas import Claim, ContradictionFlag
from reasoning.llm_client import llm_client

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# JSON schema for contradiction analysis
# ------------------------------------------------------------------

_CONTRADICTION_ITEM = {
    "type": "object",
    "properties": {
        "claim_id_a": {"type": "string"},
        "claim_id_b": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["claim_id_a", "claim_id_b", "description"],
    "additionalProperties": False,
}

ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "contradictions": {
            "type": "array",
            "items": _CONTRADICTION_ITEM,
        },
        "agreements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "dispute_type": {
            "type": "string",
            "enum": ["FACTUAL", "PRIORITIES", "MISUNDERSTANDING", "MIXED"],
        },
    },
    "required": ["contradictions", "agreements", "dispute_type"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert mediator analysing claims extracted from a heated two-person argument.

Speaker names: {speaker_map}

Your tasks:
1. CONTRADICTIONS — identify ALL pairs of claims that directly conflict with each other.
   Include both cross-speaker conflicts AND same-speaker self-contradictions.
   For each pair: reference the exact claim IDs from the provided list and write
   a neutral, one-sentence description of why they conflict using REAL speaker names
   (never "Speaker 0" / "Speaker 1").
   Rules:
   - A concession is NOT a conflict. If one party accepts the other's point ("Fine, 3
     months"), do NOT flag it as a contradiction.
   - Do not report two contradictions that represent the same underlying conflict in
     different utterance variants — group them into ONE contradiction entry.

2. AGREEMENTS — identify claims or positions that BOTH parties appear to accept \
(common ground). Express each as a single sentence beginning "Both parties agree…"

3. DISPUTE TYPE — classify the overall nature of the disagreement:
   FACTUAL       — the parties disagree about verifiable facts
   PRIORITIES    — they agree on facts but prioritise values/goals differently
   MISUNDERSTANDING — they appear to be talking past each other / using different \
definitions
   MIXED         — a combination of the above

Return only the JSON object — no prose.\
"""


def _build_user_prompt(claims: list[Claim], speakers: list | None = None) -> str:
    # Build speaker_id → display_name map for the prompt
    speaker_map: dict[str, str] = {}
    if speakers:
        for s in speakers:
            speaker_map[s.id] = s.display_name

    speaker_map_str = ", ".join(
        f"{sid}={name}" for sid, name in speaker_map.items()
    ) or "unknown"

    system = _SYSTEM_PROMPT_TEMPLATE.format(speaker_map=speaker_map_str)

    lines = ["Claims:"]
    for c in claims:
        display_name = speaker_map.get(c.speaker_id or "", c.speaker_id or "unknown")
        lines.append(
            f'  id={c.id}  speaker={display_name}'
            f'  type={c.statement_type.value}'
            f'  | "{c.verbatim_quote}"'
        )
    return system, "\n".join(lines)


async def detect_contradictions(
    claims: list[Claim],
    speakers: list | None = None,
) -> tuple[list[ContradictionFlag], list[str], str]:
    """Analyse claims for conflicts, agreements, and dispute type.

    Returns (contradictions, agreements, dispute_type).
    """
    if len(claims) < 2:
        return [], [], "MIXED"

    system, user = _build_user_prompt(claims, speakers)
    result = await llm_client.complete_json(
        system=system,
        user=user,
        schema=ANALYSIS_SCHEMA,
        max_tokens=2000,
    )

    # Build a quick lookup by claim id for validation
    claim_ids = {c.id for c in claims}

    flags: list[ContradictionFlag] = []
    for item in result.get("contradictions", []):
        a, b = item.get("claim_id_a", ""), item.get("claim_id_b", "")
        # Only include pairs where both IDs are real
        if a in claim_ids and b in claim_ids:
            flags.append(
                ContradictionFlag(
                    claim_id_a=a,
                    claim_id_b=b,
                    description=item.get("description", ""),
                    resolved=False,
                )
            )

    agreements: list[str] = result.get("agreements", [])
    dispute_type: str = result.get("dispute_type", "MIXED")

    logger.info(
        "Contradiction detection: %d conflicts, %d agreements, type=%s",
        len(flags),
        len(agreements),
        dispute_type,
    )
    return flags, agreements, dispute_type
