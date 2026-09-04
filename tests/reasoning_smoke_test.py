"""
reasoning_smoke_test.py
------------------------
End-to-end smoke test that:
  1. Creates a session in-process (no audio — injects utterances directly).
  2. Ingests a small evidence document.
  3. Runs the full report pipeline (claim extraction → contradictions → evidence
     matching → summary).
  4. Asserts at least one SUPPORTED and one CONTRADICTED claim.
  5. Prints the full report JSON.

Usage:
  cd backend
  python tests/reasoning_smoke_test.py

  # Optional: provide a custom script file (one "Speaker: text" line per turn)
  python tests/reasoning_smoke_test.py --script path/to/script.txt

  # Optional: provide an evidence file
  python tests/reasoning_smoke_test.py --evidence path/to/evidence.txt

Environment:
  .env must have LLM_PROVIDER, LLM_API_KEY, and ASSEMBLYAI_API_KEY set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
import uuid

# Make sure we can import the backend packages regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Default 15-utterance scripted argument about workplace policy
# ---------------------------------------------------------------------------

DEFAULT_SCRIPT = """\
Alex: Everyone knows remote work kills collaboration. Teams need to be in the office.
Sam: That's not true. Studies show remote workers are actually more productive.
Alex: Those studies are funded by the remote work industry — they're biased.
Sam: The Stanford study by Nicholas Bloom found a 13% productivity increase. That's peer-reviewed.
Alex: Productivity numbers don't capture culture and spontaneous innovation.
Sam: Google and Microsoft both report higher employee satisfaction with hybrid work.
Alex: Employee satisfaction is not the same as company performance.
Sam: Microsoft reported a 40% reduction in attrition in teams with hybrid options.
Alex: Lower attrition could be due to other factors — benefits, management, market conditions.
Sam: You keep moving the goalposts. First you said productivity, now it's culture, now attrition.
Alex: I'm saying in-office work builds the kind of trust you simply cannot replicate over Zoom.
Sam: Teams at Automattic and GitLab are fully remote and ship world-class software.
Alex: Those are outliers — extreme culture-fit companies. Most companies aren't like that.
Sam: The data doesn't support mandatory office work as superior to flexible arrangements.
Alex: The data supports whatever conclusion you want to reach. We need people in the office.\
"""

DEFAULT_EVIDENCE = """\
Remote Work Productivity Research Summary

Stanford University Study (Bloom et al., 2015):
A randomised controlled trial at a Chinese travel company found that remote workers \
showed a 13% performance increase compared to office workers. The improvement was \
attributed to fewer interruptions, quieter work environments, and no commute fatigue.

Microsoft WorkLab 2022 Report:
Microsoft's annual Work Trend Index found that 73% of employees want flexible remote \
work options to stay, and companies offering hybrid arrangements saw a 40% reduction \
in voluntary attrition compared to fully in-office mandates.

Harvard Business Review — Collaboration Study (2021):
A study of 61,000 Microsoft employees found that remote work made communication \
networks more siloed — remote teams had fewer cross-team connections. The authors \
concluded that remote work can reduce spontaneous collaboration and weak-tie connections \
that drive innovation.

Automattic/GitLab Case Studies:
Both companies operate as fully distributed organisations with no physical offices. \
Automattic (WordPress.com) employs ~2,000 people in 90 countries. GitLab has ~2,000 \
team members across 60+ countries. Both companies report successful software delivery \
and strong engineering culture despite being fully remote.

Nicholas Bloom, Stanford Graduate School of Business:
"Hybrid work — two to three days in the office — appears to be the optimal arrangement \
for most knowledge workers. Fully remote reduces collaboration; fully in-office \
reduces satisfaction and increases attrition without a corresponding productivity gain."\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_script(text: str, speakers: list[str]) -> list[tuple[str, str]]:
    """Parse 'Speaker: text' lines. Returns [(speaker_name, text), ...]."""
    utterances = []
    for line in text.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        name, _, content = line.partition(":")
        name = name.strip()
        content = content.strip()
        if name and content:
            utterances.append((name, content))
    return utterances


async def run_smoke_test(script_text: str, evidence_text: str) -> None:
    from core.session import Session, session_manager
    from evidence.ingest import ingest_document
    from models.schemas import Utterance
    from reasoning.mediation_report import build_report
    from models.enums import VerdictType

    session_id = f"smoke-{uuid.uuid4().hex[:8]}"
    print(f"\n{'='*60}")
    print(f"Smoke test  session_id={session_id}")
    print(f"{'='*60}\n")

    # ---- 1. Create session ------------------------------------------------
    session = await session_manager.create(session_id)

    # ---- 2. Parse script and build speakers --------------------------------
    from models.schemas import Speaker
    from transcription.speaker_mapper import SpeakerMapper

    raw_turns = parse_script(script_text, [])
    speaker_names = []
    for name, _ in raw_turns:
        if name not in speaker_names:
            speaker_names.append(name)

    mapper = SpeakerMapper(speaker_names)
    session.speakers = mapper.speakers
    name_to_id = {
        s.display_name: s.id for s in session.speakers
    }

    print(f"Speakers: {[s.display_name for s in session.speakers]}")
    print(f"Script has {len(raw_turns)} turns\n")

    # ---- 3. Inject utterances (simulate transcription) ---------------------
    for i, (name, text) in enumerate(raw_turns):
        spk_id = name_to_id.get(name)
        utt = Utterance(
            id=str(uuid.uuid4()),
            session_id=session_id,
            speaker_id=spk_id,
            text=text,
            is_final=True,
            start_ms=i * 5000,
            end_ms=(i + 1) * 5000 - 100,
        )
        await session.add_utterance(utt)

    print(f"Injected {len(session.utterances)} utterances into session.\n")

    # ---- 4. Ingest evidence ------------------------------------------------
    chunks = ingest_document(
        session_id, "evidence.txt", evidence_text.encode("utf-8")
    )
    for c in chunks:
        async with session._lock:
            session.evidence.append(c)
    print(f"Ingested evidence document → {len(chunks)} chunks.\n")

    # ---- 5. Finalize session (write WAV — will be near-empty) --------------
    # We write a tiny silent WAV so finalize() doesn't error,
    # but skip diarization since there's no real audio.
    import wave, struct
    out_dir = os.path.join("sessions", session_id)
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "audio.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 16000)  # 1 second of silence

    session.status = __import__("models.enums", fromlist=["SessionStatus"]).SessionStatus.ENDED

    # ---- 6. Build report (skip diarization — WAV is silent/fake) -----------
    print("Running report pipeline …\n")
    report = await build_report(session)

    # ---- 7. Print summary --------------------------------------------------
    print("\n" + "="*60)
    print("MEDIATION REPORT")
    print("="*60)
    print(json.dumps(report.model_dump(), indent=2))

    # ---- 8. Assertions -----------------------------------------------------
    print("\n" + "="*60)
    print("ASSERTIONS")
    print("="*60)

    verdict_types = {v.verdict for v in report.verdicts}
    has_supported = VerdictType.SUPPORTED in verdict_types
    has_contradicted = VerdictType.CONTRADICTED in verdict_types
    has_evidence_quotes = any(v.quote.strip() for v in report.verdicts
                              if v.verdict in {VerdictType.SUPPORTED, VerdictType.CONTRADICTED})

    print(f"  Claims extracted      : {len(report.claims)}")
    print(f"  Verdicts              : {len(report.verdicts)}")
    print(f"  Verdict types present : {[v.value for v in verdict_types]}")
    print(f"  Contradictions found  : {len(report.contradictions)}")
    print(f"  Agreements found      : {len(report.agreements)}")
    print(f"  Dispute type          : {report.dispute_type}")
    print(f"  Has SUPPORTED claim   : {has_supported}")
    print(f"  Has CONTRADICTED claim: {has_contradicted}")
    print(f"  Has evidence quotes   : {has_evidence_quotes}")
    print(f"  Report stored on sess : {hasattr(session, 'report') and session.report is not None}")
    print()

    failures = []
    if not report.claims:
        failures.append("No claims were extracted")
    if not has_supported:
        failures.append("No SUPPORTED claim found")
    if not has_contradicted:
        failures.append("No CONTRADICTED claim found")
    if not has_evidence_quotes:
        failures.append("No verbatim evidence quote attached to SUPPORTED/CONTRADICTED verdict")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
        print()
        sys.exit(1)
    else:
        print("All assertions PASSED ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Argument Mediator reasoning smoke test")
    parser.add_argument(
        "--script",
        default=None,
        help="Path to a plain-text script (one 'Speaker: text' line per utterance). "
             "Defaults to the built-in 15-turn remote-work argument.",
    )
    parser.add_argument(
        "--evidence",
        default=None,
        help="Path to an evidence text file (.txt, .md, .csv, .pdf). "
             "Defaults to the built-in research summary.",
    )
    args = parser.parse_args()

    script_text = DEFAULT_SCRIPT
    if args.script:
        with open(args.script, encoding="utf-8") as f:
            script_text = f.read()

    evidence_text = DEFAULT_EVIDENCE
    if args.evidence:
        with open(args.evidence, "rb") as f:
            evidence_text = f.read().decode("utf-8", errors="replace")

    asyncio.run(run_smoke_test(script_text, evidence_text))


if __name__ == "__main__":
    main()
