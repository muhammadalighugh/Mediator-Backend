"""
evidence_matcher.py
--------------------
Matches claims against evidence using a single batched LLM call per batch.

Architecture:
- Collect ALL evidence chunks from session.evidence grouped by source_name
- Build one user prompt containing every eligible claim + the full evidence corpus
- One LLM call returns a verdict for every claim id
- Claims > 12 are split into batches of 10; each batch receives the full corpus
- NO per-claim Chroma vector query (vector_store.py kept for future scaling)
- NO_MATCH → INSUFFICIENT_EVIDENCE in the final report

Eligible types: CLAIM, FACT, ASSUMPTION, EVIDENCE_REF  (OPINION only is skipped)
EVIDENCE_REF claims ("the log shows…", "check the message from…") ARE evaluated
against the corpus — they are checkable assertions, not mere opinions.
"""

from __future__ import annotations

import logging

from models.enums import StatementType, VerdictType
from models.schemas import Claim, EvidenceChunk, EvidenceLink
from reasoning.llm_client import llm_client

logger = logging.getLogger(__name__)

_ELIGIBLE_TYPES = {
    StatementType.CLAIM,
    StatementType.FACT,
    StatementType.ASSUMPTION,
    StatementType.EVIDENCE_REF,
}
_BATCH_SIZE = 10
_MAX_CLAIMS = 12   # batched above this; each batch still gets the full corpus

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You evaluate numbered claims from a two-party dispute against the COMPLETE evidence \
corpus. For each claim, compare the claim's ASSERTION against what the evidence \
ESTABLISHES — who did what, when, and how much.

Rules:
- A passage that attributes the action to a different person, date, or amount than the \
claim asserts CONTRADICTS the claim. Same topic is not support.
- The same passage may SUPPORT one speaker's claim and CONTRADICT the other's. That is \
expected in a dispute — verify payer, dates, and amounts precisely.
- Quotes must be copied CHARACTER-FOR-CHARACTER from the evidence. NEVER paraphrase. \
NEVER change names, dates, or amounts in a quote.
- If no passage bears on the claim, verdict is NO_MATCH (quote and source empty).
- Cross-reference claims against each other: if two claims conflict, the evidence \
typically resolves one in favor of the other.

Return a verdict object for EVERY claim id. None may be omitted.

STANCE COMPARISON — follow this reasoning chain for every claim:
1. Restate the claim: who did what, when, how much.
2. Restate what each passage establishes.
3. If the passage attributes the action to a different person, date, or amount than the \
claim asserts → CONTRADICTED, even though the topic matches.
4. Final check: if SUPPORTED, does the quote confirm the claim's assertion? If \
CONTRADICTED, does it show the opposite? If not, re-derive before answering.

Examples:
  Claim (Sam): "Sam paid the March rent."
  Passage: "Alex paid the March rent of $1800 on March 3, 2024."
  Claim says Sam paid; passage says Alex paid. Different payer → disproves the claim.
  Verdict: CONTRADICTED. quote: "Alex paid the March rent of $1800 on March 3, 2024."

  Claim (Sam): "Sam never agreed to cover groceries."
  Passage: "Sam: yeah I'll take groceries through June, you got the electric bill"
  Claim says no agreement; passage shows Sam agreeing.
  Verdict: CONTRADICTED. quote: "Sam: yeah I'll take groceries through June, you got the electric bill"

  The same passage may SUPPORT one speaker's claim and CONTRADICT the other's — that \
is expected in a dispute.\
"""

# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

_VERDICT_ITEM = {
    "type": "object",
    "properties": {
        "claim_id":  {"type": "string"},
        "verdict":   {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "UNCERTAIN", "NO_MATCH"]},
        "quote":     {"type": "string"},
        "source":    {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["claim_id", "verdict", "quote", "source", "reasoning"],
    "additionalProperties": False,
}

_BATCH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": _VERDICT_ITEM,
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_corpus_section(evidence: list[EvidenceChunk]) -> str:
    """Group chunks by source_name and format as labelled blocks."""
    if not evidence:
        return "EVIDENCE (complete corpus):\n[No evidence uploaded for this session]\n"

    # Group by source, preserving chunk order within each source
    sources: dict[str, list[EvidenceChunk]] = {}
    for chunk in sorted(evidence, key=lambda c: (c.source_name, c.chunk_index)):
        sources.setdefault(chunk.source_name, []).append(chunk)

    lines = ["EVIDENCE (complete corpus):"]
    for source_name, chunks in sources.items():
        lines.append(f"\n[{source_name}]")
        for chunk in chunks:
            lines.append(chunk.text)
    return "\n".join(lines)


def _build_claims_section(batch: list[Claim], speakers_by_id: dict[str, str]) -> str:
    lines = ["CLAIMS:"]
    for i, claim in enumerate(batch, 1):
        spk = speakers_by_id.get(claim.speaker_id or "", "unknown")
        lines.append(f'{i}. [{claim.id}] ({spk}): {claim.verbatim_quote}')
    return "\n".join(lines)


def _build_user_prompt(
    batch: list[Claim],
    evidence: list[EvidenceChunk],
    speakers_by_id: dict[str, str],
) -> str:
    return (
        _build_claims_section(batch, speakers_by_id)
        + "\n\n"
        + _build_corpus_section(evidence)
    )

# ---------------------------------------------------------------------------
# Verdict → VerdictType mapping
# ---------------------------------------------------------------------------

_VERDICT_MAP: dict[str, VerdictType] = {
    "SUPPORTED":    VerdictType.SUPPORTED,
    "CONTRADICTED": VerdictType.CONTRADICTED,
    "UNCERTAIN":    VerdictType.UNCERTAIN,
    "NO_MATCH":     VerdictType.INSUFFICIENT_EVIDENCE,
}


def _parse_verdict(raw: str) -> VerdictType:
    return _VERDICT_MAP.get(raw.upper(), VerdictType.INSUFFICIENT_EVIDENCE)

# ---------------------------------------------------------------------------
# Post-check: validate that quoted text exists verbatim in corpus
# ---------------------------------------------------------------------------

def validate_verdicts(verdicts: list[dict], chunks: list[EvidenceChunk]) -> list[dict]:
    """
    Downgrade any SUPPORTED/CONTRADICTED verdict whose quote cannot be found
    verbatim (with collapsed whitespace) in the evidence corpus.
    """
    corpus = " ".join(c.text for c in chunks)
    corpus = " ".join(corpus.split())  # collapse whitespace
    for v in verdicts:
        if v["verdict"] in ("SUPPORTED", "CONTRADICTED"):
            q = " ".join(v["quote"].strip().split())
            if not q or q not in corpus:
                v["quote"] = ""
                v["source"] = ""
                v["verdict"] = "UNCERTAIN"
                v["reasoning"] = "Auto-downgraded: quoted text not found verbatim in evidence."
    return verdicts

# ---------------------------------------------------------------------------
# Single-batch LLM call
# ---------------------------------------------------------------------------

async def _run_batch(
    batch: list[Claim],
    evidence: list[EvidenceChunk],
    speakers_by_id: dict[str, str],
) -> list[EvidenceLink]:
    """Run one LLM call for a batch of claims + full corpus. Returns EvidenceLinks."""

    user_prompt = _build_user_prompt(batch, evidence, speakers_by_id)

    try:
        result = await llm_client.complete_json(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=_BATCH_SCHEMA,
            max_tokens=4000,
        )
    except Exception as exc:
        logger.warning("Batch evidence matching failed: %s", exc)
        # Return INSUFFICIENT_EVIDENCE for the whole batch on LLM failure
        return [
            EvidenceLink(
                claim_id=c.id,
                evidence_chunk_id="",
                quote="",
                verdict=VerdictType.INSUFFICIENT_EVIDENCE,
                reasoning=f"Evidence matching unavailable: {exc}",
            )
            for c in batch
        ]

    # Post-check: reject fabricated quotes not found verbatim in corpus
    raw_verdicts = result.get("verdicts", [])
    raw_verdicts = validate_verdicts(raw_verdicts, evidence)

    # Build claim_id → EvidenceChunk lookup for resolving evidence_chunk_id
    # (best effort: match quote substring against stored chunks)
    chunk_by_source: dict[str, list[EvidenceChunk]] = {}
    for chunk in evidence:
        chunk_by_source.setdefault(chunk.source_name, []).append(chunk)

    def _find_chunk_id(source: str, quote: str) -> str:
        candidates = chunk_by_source.get(source, [])
        if not candidates:
            # source name may differ slightly — search all chunks
            candidates = evidence
        for chunk in candidates:
            if quote and quote[:60] in chunk.text:
                return chunk.id
        return candidates[0].id if candidates else ""

    # Index batch by claim_id for O(1) lookup
    batch_ids = {c.id for c in batch}

    links: list[EvidenceLink] = []
    seen_ids: set[str] = set()

    for item in raw_verdicts:
        claim_id = item.get("claim_id", "")
        if claim_id not in batch_ids:
            continue   # hallucinated id — skip
        seen_ids.add(claim_id)

        verdict = _parse_verdict(item.get("verdict", "NO_MATCH"))
        quote = item.get("quote", "")
        source = item.get("source", "")
        reasoning = item.get("reasoning", "")

        evidence_chunk_id = _find_chunk_id(source, quote) if quote else ""

        links.append(EvidenceLink(
            claim_id=claim_id,
            evidence_chunk_id=evidence_chunk_id,
            quote=quote,
            verdict=verdict,
            reasoning=reasoning,
        ))

    # Fill in any claim ids the LLM omitted
    for claim in batch:
        if claim.id not in seen_ids:
            logger.warning("LLM omitted verdict for claim %s — defaulting to INSUFFICIENT_EVIDENCE", claim.id)
            links.append(EvidenceLink(
                claim_id=claim.id,
                evidence_chunk_id="",
                quote="",
                verdict=VerdictType.INSUFFICIENT_EVIDENCE,
                reasoning="Verdict not returned by LLM.",
            ))

    return links

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def match_claims(
    session_id: str,
    claims: list[Claim],
    evidence: list[EvidenceChunk] | None = None,
    speakers: list | None = None,        # list[Speaker] — avoids circular import
) -> list[EvidenceLink]:
    """
    Match all eligible claims against the full evidence corpus in one (or more)
    batched LLM calls.

    Args:
        session_id: used only for logging
        claims:     all claims for the session (ineligible types are filtered out)
        evidence:   EvidenceChunk list; if None, returns INSUFFICIENT_EVIDENCE for all
        speakers:   list of Speaker objects for display names in the prompt
    """
    eligible = [c for c in claims if c.statement_type in _ELIGIBLE_TYPES]

    if not eligible:
        logger.info("No eligible claims to match for session %s", session_id)
        return []

    if not evidence:
        logger.info("No evidence uploaded for session %s — all claims get INSUFFICIENT_EVIDENCE", session_id)
        return [
            EvidenceLink(
                claim_id=c.id,
                evidence_chunk_id="",
                quote="",
                verdict=VerdictType.INSUFFICIENT_EVIDENCE,
                reasoning="No evidence documents were uploaded for this session.",
            )
            for c in eligible
        ]

    # speaker_id → display_name for the prompt (falls back to speaker_id if unknown)
    speakers_by_id: dict[str, str] = (
        {s.id: s.display_name for s in speakers} if speakers else {}
    )

    # Split into batches of _BATCH_SIZE; each batch gets the FULL corpus
    batches = [
        eligible[i : i + _BATCH_SIZE]
        for i in range(0, len(eligible), _BATCH_SIZE)
    ]

    logger.info(
        "Evidence matching: %d eligible claims → %d batch(es) for session %s",
        len(eligible), len(batches), session_id,
    )

    all_links: list[EvidenceLink] = []
    for batch_num, batch in enumerate(batches, 1):
        logger.info("Running batch %d/%d (%d claims)", batch_num, len(batches), len(batch))
        links = await _run_batch(batch, evidence, speakers_by_id)
        all_links.extend(links)

    logger.info(
        "Evidence matching complete: %d links for session %s",
        len(all_links), session_id,
    )
    return all_links
