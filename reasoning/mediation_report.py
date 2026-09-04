"""
mediation_report.py
--------------------
Assembles the final MediationReport at end_session time.

Pipeline:
  (a) Async diarization  — transcribe sessions/{id}/audio.wav with AssemblyAI
                           batch API + speaker_labels; backfill any utterances
                           where speaker_id is None.
  (b) Claim extraction   — re-run on the canonical (diarized) transcript.
  (c) Contradiction det. — contradictions, agreements, dispute_type.
  (d) Evidence matching  — EvidenceLink list.
  (e) Summary generation — 3-5 sentence executive summary refusing to
                           declare a winner when evidence is insufficient.
  (f) Assemble & store   — MediationReport written to session; GET-able via
                           /report/{session_id}.

The report is also stored in-memory on the Session object as session.report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import assemblyai as aai

from core.config import settings
from core.session import Session
from evidence.evidence_matcher import match_claims
from models.enums import VerdictType
from models.schemas import MediationReport, Utterance
from reasoning.claim_extractor import extract_claims
from reasoning.contradiction_detector import detect_contradictions
from reasoning.llm_client import llm_client

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# (a) Async diarization
# ------------------------------------------------------------------

async def _diarize_wav(wav_path: str) -> list[tuple[str, str, int, int]]:
    """
    Transcribe a local WAV with AssemblyAI batch API (speaker_labels=True).
    Returns list of (speaker_label, text, start_ms, end_ms).
    Runs the blocking SDK call in a thread executor.
    """
    def _blocking() -> list[tuple[str, str, int, int]]:
        aai.settings.api_key = settings.assemblyai_api_key
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            punctuate=True,
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(wav_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI diarization error: {transcript.error}")

        results = []
        if transcript.utterances:
            for utt in transcript.utterances:
                results.append(
                    (
                        utt.speaker,            # e.g. "A", "B"
                        utt.text,
                        utt.start,              # ms
                        utt.end,                # ms
                    )
                )
        return results

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _blocking)


async def _backfill_speakers(
    session: Session,
    diarized: list[tuple[str, str, int, int]],
) -> list[Utterance]:
    """
    Build canonical Utterance list from diarization output.
    Map AssemblyAI speaker labels to session.speakers by position.
    """
    # Build label → speaker_id map from existing speakers (A→speaker_0, B→speaker_1 …)
    label_map: dict[str, str] = {}
    for i, spk in enumerate(session.speakers):
        label = chr(ord("A") + i)
        label_map[label] = spk.id

    canonical: list[Utterance] = []
    for label, text, start_ms, end_ms in diarized:
        speaker_id = label_map.get(label)
        if speaker_id is None and label:
            # Extra speaker not anticipated at session start
            idx = len(label_map)
            speaker_id = f"speaker_{idx}"
            label_map[label] = speaker_id

        canonical.append(
            Utterance(
                id=str(uuid.uuid4()),
                session_id=session.id,
                speaker_id=speaker_id,
                text=text,
                is_final=True,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return canonical


# ------------------------------------------------------------------
# (e) Executive summary
# ------------------------------------------------------------------

_SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

_SUMMARY_SYSTEM = """\
You are a professional mediator writing an impartial executive summary of a \
dispute analysis. Write 3–5 sentences covering:
  1. The core disagreement and its type (factual/priorities/misunderstanding/mixed).
  2. Any common ground found.
  3. Key unresolved tensions.

Critical rule: If the evidence is insufficient to determine who is correct, \
explicitly state that — do NOT declare or imply a winner. Maintain strict \
neutrality at all times.

Output only the JSON object — no prose.\
"""


def _build_summary_prompt(
    report: MediationReport,
    speakers: list,
) -> str:
    spk_names = {s.id: s.display_name for s in speakers}

    lines = [
        f"Dispute type: {report.dispute_type}",
        f"Agreements: {'; '.join(report.agreements) or 'None identified'}",
        f"Contradictions: {len(report.contradictions)} pair(s) found",
        "",
        "Claims and verdicts:",
    ]
    # Build claim id → claim map
    claim_map = {c.id: c for c in report.claims}
    verdict_map = {v.claim_id: v for v in report.verdicts}
    for claim in report.claims:
        verdict = verdict_map.get(claim.id)
        verdict_str = verdict.verdict.value if verdict else "unmatched"
        spk = spk_names.get(claim.speaker_id or "", claim.speaker_id or "unknown")
        lines.append(f'  [{spk}] ({verdict_str}) "{claim.verbatim_quote}"')

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

async def build_report(session: Session) -> MediationReport:
    """Full mediation pipeline. Stores result on session.report."""

    # ----------------------------------------------------------------
    # (a) Diarization
    # ----------------------------------------------------------------
    wav_path = os.path.join("sessions", session.id, "audio.wav")
    canonical_utterances: list[Utterance] = []

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:  # > WAV header
        try:
            logger.info("Starting diarization for session %s …", session.id)
            diarized = await _diarize_wav(wav_path)
            canonical_utterances = await _backfill_speakers(session, diarized)
            logger.info(
                "Diarization produced %d utterances", len(canonical_utterances)
            )
        except Exception as exc:
            logger.warning(
                "Diarization failed (%s) — falling back to live utterances", exc
            )
            canonical_utterances = list(session.utterances)
    else:
        logger.info(
            "No WAV found or WAV is empty — using live utterances for session %s",
            session.id,
        )
        canonical_utterances = list(session.utterances)

    # ----------------------------------------------------------------
    # (b) Claim extraction on canonical transcript
    # ----------------------------------------------------------------
    # Clear any live-extracted claims so the report uses ONLY the fresh
    # post-diarization extraction (live claims are UI-only).
    async with session._lock:
        session.claims.clear()

    if canonical_utterances:
        new_claims = await extract_claims(canonical_utterances, session)
        logger.info("Post-diarization extraction: %d claims", len(new_claims))
    else:
        new_claims = []

    claims = list(session.claims)

    # ----------------------------------------------------------------
    # (c) Contradictions, agreements, dispute type
    # ----------------------------------------------------------------
    contradictions, agreements, dispute_type = await detect_contradictions(
        claims, speakers=list(session.speakers)
    )

    # ----------------------------------------------------------------
    # (d) Evidence matching
    # ----------------------------------------------------------------
    verdicts = await match_claims(
        session.id,
        claims,
        evidence=list(session.evidence),
        speakers=list(session.speakers),
    )

    # ----------------------------------------------------------------
    # (e) Executive summary
    # ----------------------------------------------------------------
    # Assemble partial report for summary prompt
    partial = MediationReport(
        session_id=session.id,
        claims=claims,
        verdicts=verdicts,
        contradictions=contradictions,
        agreements=agreements,
        dispute_type=dispute_type,
        summary="",  # placeholder
    )

    try:
        summary_result = await llm_client.complete_json(
            system=_SUMMARY_SYSTEM,
            user=_build_summary_prompt(partial, session.speakers),
            schema=_SUMMARY_SCHEMA,
            max_tokens=400,
        )
        summary = summary_result.get("summary", "")
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        summary = (
            f"Analysis identified {len(claims)} claims, "
            f"{len(contradictions)} contradictions, and "
            f"{len(agreements)} areas of agreement. "
            "Insufficient evidence to determine a winner. "
            "Both parties should review the evidence and consider further mediation."
        )

    # ----------------------------------------------------------------
    # (f) Assemble final report
    # ----------------------------------------------------------------
    report = MediationReport(
        session_id=session.id,
        claims=claims,
        verdicts=verdicts,
        contradictions=contradictions,
        agreements=agreements,
        dispute_type=dispute_type,
        summary=summary,
    )

    # Store on session for GET /report/{session_id}
    session.report = report  # type: ignore[attr-defined]

    logger.info("MediationReport assembled for session %s", session.id)
    return report
