"""
mediation_report.py
--------------------
Assembles the final MediationReport at end_session time.

Pipeline:
  (a)   Async diarization  — transcribe sessions/{id}/audio.wav with AssemblyAI
                             batch API + speaker_labels; backfill any utterances
                             where speaker_id is None.
  (a.1) Speaker consolid.  — collapse excess diarized labels down to the declared
                             roster (top-N by total word count); guarantee every
                             surviving utterance has a non-blank speaker.
  (a.2) Injected-speech    — one LLM pass over the canonical transcript to drop
        filter               non-participant / system-injected utterances before
                             they can become claims.
  (b)   Claim extraction   — re-run on the canonical (diarized, consolidated,
                             filtered) transcript.
  (c)   Contradiction det. — contradictions, agreements, dispute_type.
  (d)   Evidence matching  — EvidenceLink list.
  (e)   Summary generation — 3–5 sentence executive summary; names the account
                             the evidence favors when verdicts are one-sided,
                             refuses to declare a winner when evidence is absent.
  (f)   Assemble & store   — MediationReport written to session; GET-able via
                             /report/{session_id}.
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
from models.enums import StatementType, VerdictType
from models.schemas import MediationReport, SpeakerAssessment, Utterance
from reasoning.claim_extractor import extract_claims
from reasoning.contradiction_detector import detect_contradictions
from reasoning.llm_client import llm_client
from transcription.speaker_mapper import consolidate_speakers
from voice.voice_id import VoiceID

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

    # FIX 4: get_event_loop() is deprecated inside a running coroutine;
    # get_running_loop() is the correct call.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking)


async def _backfill_speakers(
    session: Session,
    diarized: list[tuple[str, str, int, int]],
) -> list[Utterance]:
    """
    Build canonical Utterance list from diarization output.

    Step 0 — Enrollment label binding (when enrollment records exist):
        Each EnrollmentRecord stores the realtime label that was active when
        a speaker said their name, plus the timestamp of that utterance.
        We find which diarized label covers that timestamp (largest overlap)
        and build a diarized_label → speaker_id map directly from enrollment.
        This is the most accurate strategy and takes priority over positional.
        Falls back to positional mapping if two records resolve to the same
        diarized label (merged voices) — in that case time-overlap
        reconciliation in Step 2 will still fix attribution downstream.
        Logged as "enrollment label binding".

    Step 1 — Positional label mapping:
        AssemblyAI batch-diarization label "A" → speaker_0, "B" → speaker_1.
        Used when no enrollment records exist.
        Logged as "positional label mapping".

    Step 2 — Live-attribution reconciliation:
        The live realtime transcript is ground truth for speaker identity.
        Each canonical utterance's speaker_id is replaced by the live
        speaker_id with the largest time overlap. Fallback: Step 0/1 result.
        Skipped if no live utterance has a speaker_id.

    Step 3 — Log canonical speaker distribution (always).
    """
    # ---- Step 0: enrollment label binding (if records exist) ---------------
    enrollment_label_map: dict[str, str] = {}   # diarized_label → speaker_id
    if session.enrollment_records:
        # Build a lookup: for each enrollment record find which diarized label
        # overlaps its [start_ms, end_ms] the most.
        for rec in session.enrollment_records:
            # Find the matching speaker by name in session.speakers
            matched_spk = next(
                (s for s in session.speakers if s.display_name == rec.speaker_name),
                None,
            )
            if matched_spk is None:
                logger.warning(
                    "Enrollment record name %r not found in session.speakers — skipping",
                    rec.speaker_name,
                )
                continue

            best_label: str | None = None
            best_overlap = 0
            for d_label, d_text, d_start, d_end in diarized:
                overlap = min(rec.end_ms, d_end) - max(rec.start_ms, d_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = d_label

            if best_label is None:
                logger.warning(
                    "Enrollment record for %r found no overlapping diarized utterance — skipping",
                    rec.speaker_name,
                )
                continue

            # Detect voice-merge: two records mapping to the same diarized label
            if best_label in enrollment_label_map:
                logger.warning(
                    "Enrollment: two speakers mapped to the same diarized label %r "
                    "(voices merged) — falling back to positional mapping",
                    best_label,
                )
                enrollment_label_map = {}   # discard — fall back entirely
                break

            enrollment_label_map[best_label] = matched_spk.id
            logger.info(
                "Enrollment label binding: diarized label %r → %s (%s)",
                best_label, matched_spk.id, matched_spk.display_name,
            )

    use_enrollment = bool(enrollment_label_map)

    # ---- Step 1: positional label map (fallback or no enrollment) ----------
    label_map: dict[str, str] = {}
    for i, spk in enumerate(session.speakers):
        label = chr(ord("A") + i)
        label_map[label] = spk.id
    if use_enrollment:
        # Override positional entries with enrollment-derived bindings
        label_map.update(enrollment_label_map)
        logger.info("_backfill_speakers: strategy = enrollment label binding")
    else:
        logger.info("_backfill_speakers: strategy = positional label mapping")

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

    # ---- Step 2: reconcile with live attribution ----------------------------
    # Collect live utterances that have a resolved speaker_id.
    live_attributed = [u for u in session.utterances if u.speaker_id is not None]

    if live_attributed:
        # Build per-speaker sorted intervals for efficient overlap computation.
        # Structure: {speaker_id: [(start_ms, end_ms), ...]}
        live_intervals: dict[str, list[tuple[int, int]]] = {}
        for u in live_attributed:
            live_intervals.setdefault(u.speaker_id, []).append(  # type: ignore[arg-type]
                (u.start_ms, u.end_ms)
            )

        reconciled: list[Utterance] = []
        for utt in canonical:
            c_start, c_end = utt.start_ms, utt.end_ms

            # Sum overlap_ms per live speaker_id
            overlap_by_spk: dict[str, int] = {}
            for spk_id, intervals in live_intervals.items():
                total = 0
                for ls, le in intervals:
                    overlap = min(c_end, le) - max(c_start, ls)
                    if overlap > 0:
                        total += overlap
                if total > 0:
                    overlap_by_spk[spk_id] = total

            if overlap_by_spk:
                best_spk = max(overlap_by_spk, key=overlap_by_spk.__getitem__)
                utt = utt.model_copy(update={"speaker_id": best_spk})

            reconciled.append(utt)

        canonical = reconciled
        logger.info("Speaker reconciliation applied using %d live attributed utterances.", len(live_attributed))
    else:
        logger.info("No live attributed utterances — keeping positional speaker mapping.")

    # ---- Step 3: log canonical speaker distribution -------------------------
    counts: dict[str, int] = {}
    for utt in canonical:
        sid = utt.speaker_id or "None"
        counts[sid] = counts.get(sid, 0) + 1
    logger.info(
        "Canonical speakers: %s",
        ", ".join(f"{sid}={n}" for sid, n in sorted(counts.items())),
    )

    return canonical


# ------------------------------------------------------------------
# (a.2) Injected-speech filter
# ------------------------------------------------------------------
#
# Runs after diarization/backfill/consolidation and before claim extraction.
# This is a content-based filter (an LLM judgment on WHAT was said) and is
# separate from — and complements — the short/numeric garbage filter that
# already runs in stream_handler.py (which is about HOW SHORT/malformed the
# ASR output is). Neither replaces the other.

_INJECTED_SPEECH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "utterance_ids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "keep": {"type": "boolean"},
                },
                "required": ["id", "keep"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["utterance_ids"],
    "additionalProperties": False,
}

_INJECTED_SPEECH_SYSTEM = """\
You are filtering a transcript of a two-person dispute. Some transcripts are \
machine-generated and contain injected non-participant speech. Classify each \
utterance KEEP or DROP.

DROP if the utterance is:
- System, assistant, or recording meta-speech — talking about the system, \
commands, recording, playback, or the conversation itself \
("When you say start, the system runs the command", "Recording begins after \
the tone", "I will now repeat your last message")
- An announcement or instruction addressed to a listener rather than an argument \
turn between the two parties
- Completely unrelated to the dispute topic

KEEP everything the actual participants say — including heated, rude, sarcastic, \
or fragmented speech. When unsure, KEEP: wrongly dropping a real argument turn is \
worse than keeping a stray line.

Output only the JSON object — no prose.\
"""


def _build_injected_speech_prompt(utterances: list[Utterance], speakers: list) -> str:
    spk_names = {s.id: s.display_name for s in speakers}
    lines = ["Transcript:"]
    for u in utterances:
        spk = spk_names.get(u.speaker_id or "", u.speaker_id or "unknown")
        lines.append(f'  id={u.id} [{spk}]: "{u.text}"')
    return "\n".join(lines)


async def _filter_injected_speech(
    utterances: list[Utterance],
    speakers: list,
) -> list[Utterance]:
    """
    One LLM call over the full speaker-attributed transcript to drop
    non-participant / system-injected utterances before they can become
    claims. Every utterance id is expected back exactly once with a
    keep/drop decision.

    Fails OPEN on any error — LLM failure, malformed response, or an id the
    response didn't cover all default to KEEP. A filtering bug should never
    silently erase real dispute content; that mirrors the "when unsure,
    KEEP" instruction given to the model itself.
    """
    if not utterances:
        return utterances

    try:
        result = await llm_client.complete_json(
            system=_INJECTED_SPEECH_SYSTEM,
            user=_build_injected_speech_prompt(utterances, speakers),
            schema=_INJECTED_SPEECH_SCHEMA,
            max_tokens=max(500, len(utterances) * 30),
        )
        decisions: dict[str, bool] = {
            item["id"]: bool(item["keep"])
            for item in result.get("utterance_ids", [])
            if "id" in item
        }
    except Exception as exc:
        logger.warning(
            "Injected-speech filter failed (%s) — keeping all %d utterance(s)",
            exc,
            len(utterances),
        )
        return utterances

    kept: list[Utterance] = []
    dropped: list[Utterance] = []
    missing_ids: list[str] = []
    for u in utterances:
        if u.id not in decisions:
            missing_ids.append(u.id)
        keep = decisions.get(u.id, True)  # fail open: unknown id → KEEP
        (kept if keep else dropped).append(u)

    if missing_ids:
        logger.warning(
            "Injected-speech filter response omitted %d utterance id(s) — "
            "kept by default: %s",
            len(missing_ids),
            missing_ids,
        )

    if dropped:
        logger.info(
            "Injected-speech filter: dropped %d of %d utterance(s): %s",
            len(dropped),
            len(utterances),
            "; ".join(f'{u.id}: "{u.text}"' for u in dropped),
        )

    return kept


# ------------------------------------------------------------------
# (a.3) Post-diarization voice identification
# ------------------------------------------------------------------

def _build_voice_id_from_wav(
    wav_path: str,
    session: Session,
) -> VoiceID | None:
    """
    Reconstruct a VoiceID instance from EnrollmentRecord timestamps by reading
    the saved WAV file (same format as the live buffer: 16kHz 16-bit mono PCM).

    Returns None if enrollment records are absent, resemblyzer is unavailable,
    or the WAV file cannot be read.
    """
    if not session.enrollment_records:
        return None

    import wave as _wave

    try:
        with _wave.open(wav_path, "rb") as wf:
            raw_pcm = wf.readframes(wf.getnframes())
    except Exception as exc:
        logger.warning("VoiceID: could not read WAV for enrollment: %s", exc)
        return None

    # Byte rate: 16000 samples/s × 2 bytes/sample = 32000 bytes/s
    byte_rate = 16000 * 2  # must match Session._sample_rate * _BYTES_PER_SAMPLE

    vid = VoiceID()
    for rec in session.enrollment_records:
        start_byte = int(rec.start_ms / 1000 * byte_rate)
        end_byte = min(int(rec.end_ms / 1000 * byte_rate), len(raw_pcm))
        pcm_slice = raw_pcm[start_byte:end_byte]
        enrolled = vid.enroll(rec.speaker_name, pcm_slice)
        if not enrolled:
            logger.warning(
                "VoiceID (report): enrollment skipped for %r (clip too short or "
                "resemblyzer unavailable)",
                rec.speaker_name,
            )

    return vid if vid.enrolled_names else None


def _apply_voice_id_to_canonical(
    utterances: list[Utterance],
    vid: VoiceID,
    session: Session,
    wav_path: str,
) -> list[Utterance]:
    """
    Re-attribute each canonical utterance using voice identification against
    the enrolled voiceprints. Identified name wins over the existing speaker_id;
    returns (None, 0.0) cases unchanged (existing label kept as fallback).
    """
    import wave as _wave

    try:
        with _wave.open(wav_path, "rb") as wf:
            raw_pcm = wf.readframes(wf.getnframes())
    except Exception as exc:
        logger.warning("VoiceID (report): cannot read WAV for identification: %s", exc)
        return utterances

    byte_rate = 16000 * 2

    # Build name → speaker_id map from session.speakers
    name_to_id = {s.display_name: s.id for s in session.speakers}

    overrides = 0
    result: list[Utterance] = []
    for utt in utterances:
        start_byte = int(utt.start_ms / 1000 * byte_rate)
        end_byte = min(int(utt.end_ms / 1000 * byte_rate), len(raw_pcm))
        pcm_slice = raw_pcm[start_byte:end_byte]

        identified_name, sim = vid.identify(pcm_slice)
        if identified_name is not None:
            new_spk_id = name_to_id.get(identified_name)
            if new_spk_id is not None and new_spk_id != utt.speaker_id:
                logger.debug(
                    "VoiceID (report): utt [%d–%d ms] %r → %r (sim=%.3f)",
                    utt.start_ms, utt.end_ms, utt.speaker_id, new_spk_id, sim,
                )
                utt = utt.model_copy(update={"speaker_id": new_spk_id})
                overrides += 1

        result.append(utt)

    if overrides:
        logger.info(
            "VoiceID (report): overrode speaker attribution on %d / %d utterances",
            overrides, len(utterances),
        )
    else:
        logger.info("VoiceID (report): no attribution overrides (all fallback labels kept)")

    return result


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
  1. Who the evidence favors (using the tallies below — verbatim).
  2. Any common ground found.
  3. Key unresolved tensions.

Rules — follow in order:
  • Begin the summary with one sentence naming who the evidence favors, using \
the tallies. Example: "The evidence favors Alex: 3 of his 3 checkable claims \
are supported, while 3 of Sam's 3 are contradicted." Use the exact numbers \
from the tallies — do not round, omit, or contradict them.
  • If checkable totals are low (fewer than 2 per speaker) or mixed (neither \
party clearly ahead), say the evidence is insufficient to favor either party \
— never force a conclusion the tallies do not support.
  • Only withhold a judgment when the tallies are genuinely balanced or \
evidence is absent, and then say what evidence would resolve the dispute.
  • Never invent a winner the tallies do not show.

Output only the JSON object — no prose.\
"""


def _compute_assessments(
    report: MediationReport,
    speakers: list,
) -> list[SpeakerAssessment]:
    """
    Deterministically tally per-speaker verdict counts from the EvidenceLink list.

    - supported:      VerdictType.SUPPORTED
    - contradicted:   VerdictType.CONTRADICTED
    - uncertain:      UNCERTAIN + INSUFFICIENT_EVIDENCE
    - checkable_total: supported + contradicted only (uncertain excluded)

    Claims with no matching verdict are not counted (they carry no evidence signal).
    Claims whose speaker_id is None are attributed to "unknown" and not shown.
    """
    claim_map = {c.id: c for c in report.claims}
    verdict_map = {v.claim_id: v for v in report.verdicts}

    # { speaker_id → {supported, contradicted, uncertain} }
    tallies: dict[str, dict[str, int]] = {}
    for spk in speakers:
        tallies[spk.id] = {"supported": 0, "contradicted": 0, "uncertain": 0}

    for verdict_link in report.verdicts:
        claim = claim_map.get(verdict_link.claim_id)
        if claim is None or claim.speaker_id is None:
            continue
        sid = claim.speaker_id
        if sid not in tallies:
            tallies[sid] = {"supported": 0, "contradicted": 0, "uncertain": 0}

        v = verdict_link.verdict
        if v == VerdictType.SUPPORTED:
            tallies[sid]["supported"] += 1
        elif v == VerdictType.CONTRADICTED:
            tallies[sid]["contradicted"] += 1
        else:  # UNCERTAIN or INSUFFICIENT_EVIDENCE
            tallies[sid]["uncertain"] += 1

    assessments = []
    for spk in speakers:
        t = tallies[spk.id]
        assessments.append(
            SpeakerAssessment(
                speaker_id=spk.id,
                supported=t["supported"],
                contradicted=t["contradicted"],
                uncertain=t["uncertain"],
                checkable_total=t["supported"] + t["contradicted"],
            )
        )
    return assessments


def _build_summary_prompt(
    report: MediationReport,
    speakers: list,
    assessments: list[SpeakerAssessment],
) -> str:
    spk_names = {s.id: s.display_name for s in speakers}

    # Hard-fact tally block the LLM must use verbatim
    tally_lines = ["Speaker tallies (must be used verbatim in the summary):"]
    for a in assessments:
        name = spk_names.get(a.speaker_id, a.speaker_id)
        tally_lines.append(
            f"  {name}: {a.supported} supported, {a.contradicted} contradicted, "
            f"{a.uncertain} uncertain (of {a.checkable_total} checkable claims)."
        )

    verdict_map = {v.claim_id: v for v in report.verdicts}
    claim_lines = ["Claims and verdicts:"]
    for claim in report.claims:
        verdict = verdict_map.get(claim.id)
        verdict_str = verdict.verdict.value if verdict else "unmatched"
        spk = spk_names.get(claim.speaker_id or "", claim.speaker_id or "unknown")
        claim_lines.append(f'  [{spk}] ({verdict_str}) "{claim.verbatim_quote}"')

    lines = [
        *tally_lines,
        "",
        f"Dispute type: {report.dispute_type}",
        f"Agreements: {'; '.join(report.agreements) or 'None identified'}",
        f"Contradictions: {len(report.contradictions)} pair(s) found",
        "",
        *claim_lines,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

async def build_report(session: Session) -> MediationReport:
    """Full mediation pipeline. Stores result on session.report."""
    sid = session.id

    # ----------------------------------------------------------------
    # (a) Diarization
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: diarization", sid)
    wav_path = os.path.join("sessions", sid, "audio.wav")
    canonical_utterances: list[Utterance] = []

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:  # > WAV header
        try:
            logger.info("[REPORT] [%s] diarizing WAV: %s", sid, wav_path)
            diarized = await _diarize_wav(wav_path)
            logger.info(
                "[REPORT] [%s] diarization produced %d utterances; labels: %s",
                sid, len(diarized),
                sorted({label for label, *_ in diarized if label}) or ["<none>"],
            )
            canonical_utterances = await _backfill_speakers(session, diarized)
        except Exception as exc:
            logger.warning(
                "[REPORT] [%s] diarization failed (%s) — falling back to live utterances",
                sid, exc,
            )
            canonical_utterances = list(session.utterances)
    else:
        logger.info(
            "[REPORT] [%s] no WAV or WAV empty — using %d live utterance(s)",
            sid, len(session.utterances),
        )
        canonical_utterances = list(session.utterances)

    # ----------------------------------------------------------------
    # (a.0) Enrollment utterance filter
    # Strip any utterance whose audio falls within the enrollment window
    # (i.e. end_ms <= session.enrollment_end_ms).  These are "I am Maya"
    # / "I am Daniel" turns that must never become claims.
    # ----------------------------------------------------------------
    if session.enrollment_end_ms > 0:
        before = len(canonical_utterances)
        canonical_utterances = [
            u for u in canonical_utterances
            if u.end_ms > session.enrollment_end_ms
        ]
        dropped = before - len(canonical_utterances)
        logger.info(
            "[REPORT] [%s] enrollment filter: removed %d utterance(s) "
            "(end_ms <= %d ms), %d remaining",
            sid, dropped, session.enrollment_end_ms, len(canonical_utterances),
        )

    # ----------------------------------------------------------------
    # (a.1) Speaker consolidation
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: consolidation (%d utterances)", sid, len(canonical_utterances))
    canonical_utterances = consolidate_speakers(
        canonical_utterances, list(session.speakers)
    )

    # ----------------------------------------------------------------
    # (a.2) Injected-speech filter
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: filter (%d utterances)", sid, len(canonical_utterances))
    if canonical_utterances:
        canonical_utterances = await _filter_injected_speech(
            canonical_utterances, list(session.speakers)
        )
        logger.info("[REPORT] [%s] after filter: %d utterance(s)", sid, len(canonical_utterances))

    # ----------------------------------------------------------------
    # (a.3) Voice-ID attribution
    # ----------------------------------------------------------------
    if os.path.exists(wav_path):
        vid = _build_voice_id_from_wav(wav_path, session)
        if vid is not None:
            canonical_utterances = _apply_voice_id_to_canonical(
                canonical_utterances, vid, session, wav_path
            )
            logger.info(
                "[REPORT] [%s] voice-ID ran against %d enrolled speaker(s): %s",
                sid, len(vid.enrolled_names), vid.enrolled_names,
            )
        else:
            logger.info("[REPORT] [%s] voice-ID skipped (no enrollment or resemblyzer absent)", sid)

    # ----------------------------------------------------------------
    # (b) Claim extraction
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: extraction", sid)
    async with session._lock:
        session.claims.clear()

    if canonical_utterances:
        new_claims = await extract_claims(canonical_utterances, session)
    else:
        new_claims = []
        logger.info("[REPORT] [%s] no canonical utterances — 0 claims", sid)

    claims = list(session.claims)

    # ----------------------------------------------------------------
    # (b.1) Trivial-session detection — short-circuit before heavy LLM steps
    #
    # A session is "trivial" (just conversation, nothing to mediate) when:
    #   • 0 claims were extracted, OR
    #   • fewer than 4 utterances AND no CLAIM-type statements in the transcript
    #
    # For trivial sessions we skip steps (c)–(e) entirely and emit a
    # lightweight "conversation" report instead.
    # ----------------------------------------------------------------
    n_utterances = len(canonical_utterances)
    has_claim_type = any(
        c.statement_type == StatementType.CLAIM for c in claims
    )
    is_trivial = (len(claims) == 0) or (n_utterances < 4 and not has_claim_type)

    if is_trivial:
        # Build a human-readable name list from the session speakers.
        spk_names = [s.display_name for s in session.speakers]
        if spk_names:
            names_str = " and ".join(spk_names) if len(spk_names) <= 2 else ", ".join(spk_names)
        else:
            names_str = "the participants"

        friendly_summary = (
            f"No dispute detected — {names_str} talked, but no checkable claims were made. "
            "Nothing to mediate. Start a new session when there's something to sort out."
        )
        logger.info(
            "[REPORT] trivial session (%d utterances, %d claims) — friendly mode",
            n_utterances, len(claims),
        )

        report = MediationReport(
            session_id=sid,
            claims=[],
            verdicts=[],
            contradictions=[],
            agreements=[],
            dispute_type="conversation",
            summary=friendly_summary,
            assessments=[],
            report_kind="conversation",
        )

        session.report = report  # type: ignore[attr-defined]
        logger.info("[REPORT] [%s] done (trivial): conversation kind, 0 claims", sid)
        return report

    # ----------------------------------------------------------------
    # (c) Contradictions, agreements, dispute type
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: contradictions (%d claims)", sid, len(claims))
    contradictions, agreements, dispute_type = await detect_contradictions(
        claims, speakers=list(session.speakers)
    )
    logger.info(
        "[REPORT] [%s] contradictions: %d pair(s), %d agreement(s), type=%s",
        sid, len(contradictions), len(agreements), dispute_type,
    )

    # ----------------------------------------------------------------
    # (d) Evidence matching
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: matching", sid)
    verdicts = await match_claims(
        sid,
        claims,
        evidence=list(session.evidence),
        speakers=list(session.speakers),
    )

    # ----------------------------------------------------------------
    # (e) Summary
    # ----------------------------------------------------------------
    logger.info("[REPORT] [%s] pipeline step reached: summary", sid)
    partial = MediationReport(
        session_id=sid,
        claims=claims,
        verdicts=verdicts,
        contradictions=contradictions,
        agreements=agreements,
        dispute_type=dispute_type,
        summary="",
    )

    assessments = _compute_assessments(partial, list(session.speakers))
    logger.info(
        "[REPORT] [%s] speaker assessments: %s",
        sid,
        "; ".join(
            f"{a.speaker_id} sup={a.supported} con={a.contradicted} unc={a.uncertain}"
            for a in assessments
        ),
    )

    try:
        summary_result = await llm_client.complete_json(
            system=_SUMMARY_SYSTEM,
            user=_build_summary_prompt(partial, session.speakers, assessments),
            schema=_SUMMARY_SCHEMA,
            max_tokens=400,
        )
        summary = summary_result.get("summary", "")
    except Exception as exc:
        logger.warning("[REPORT] [%s] summary generation failed: %s", sid, exc)
        summary = (
            f"Analysis identified {len(claims)} claims, "
            f"{len(contradictions)} contradictions, and "
            f"{len(agreements)} areas of agreement. "
            "Insufficient evidence to determine a winner. "
            "Both parties should review the evidence and consider further mediation."
        )

    # ----------------------------------------------------------------
    # (f) Assemble
    # ----------------------------------------------------------------
    report = MediationReport(
        session_id=sid,
        claims=claims,
        verdicts=verdicts,
        contradictions=contradictions,
        agreements=agreements,
        dispute_type=dispute_type,
        summary=summary,
        assessments=assessments,
        report_kind="dispute",
    )

    # Store on session for GET /report/{session_id}
    session.report = report  # type: ignore[attr-defined]

    logger.info(
        "[REPORT] [%s] done: %d claim(s), %d verdict(s)",
        sid, len(claims), len(verdicts),
    )
    return report