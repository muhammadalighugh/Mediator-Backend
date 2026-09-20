"""
ws_routes.py
------------
WebSocket endpoint: /ws/session/{session_id}

Critical path (the ONLY thing that must work for live transcript):
  connect → start_session → audio_chunk → TurnEvent → transcript_final → broadcast

Client → server frames
  {"type":"start_session","speakers":["Alex","Sam"]}
  {"type":"audio_chunk","data":"<base64-encoded PCM>"}
  {"type":"analyze"}          ← triggers contradiction detection on-demand
  {"type":"end_session"}

  Optional enrollment frames (processed before start_session):
  {"type":"begin_enrollment", "slot": 1}
  {"type":"enrollment_result_ack"}   ← ignored

Server → client frames
  {"type":"enrollment_result", "slot":1, "name":"Sam"}
  {"type":"enrollment_result", "slot":1, "name":null}
  {"type":"enrollment_result", "slot":2, "name":null, "duplicate_of":"Sam"}
  {"type":"enrollment_result", "slot":2, "name":null, "give_up":true}
  {"type":"transcript_partial", "speaker_id":..., "text":..., "start_ms":..., "end_ms":...}
  {"type":"transcript_final",   "speaker_id":..., "text":..., "start_ms":..., "end_ms":..., "utterance_id":...}
  {"type":"claims_updated",     "claims":[...]}
  {"type":"contradictions",     "contradictions":[...], "agreements":[...], "dispute_type":...}
  {"type":"clarifying_question","question":..., "contradiction":...}
  {"type":"report_ready",       "report":{...}}
  {"type":"session_ended",      "wav_path":...}
  {"type":"error",              "detail":...}

Enrollment flow (optional, before start_session):
  1. Client sends begin_enrollment(slot=N).
  2. Server creates Session + AssemblyAIClient; audio begins flowing.
  3. Client streams audio_chunk frames while the speaker says their name.
  4. First valid final transcript is handed to reasoning.enrollment.extract_name().
  5. Server replies enrollment_result(slot, name).
  6. Client repeats for slot=2, then sends start_session with the collected names.
     The server calls mapper.bind_labels() to wire realtime labels to those names.

No-enrollment flow:
  Client sends start_session directly. Positional mapping A→speaker_0, B→speaker_1.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from core.session import session_manager
from models.schemas import EnrollmentRecord
from reasoning.enrollment import extract_name
from transcription.assemblyai_client import AssemblyAIClient
from transcription.speaker_mapper import SpeakerMapper
from transcription.stream_handler import StreamHandler
from reasoning.claim_extractor import ClaimExtractionCoordinator
from reasoning.contradiction_detector import detect_contradictions
from reasoning.clarifying_questions import generate_question

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: run contradiction analysis and send results over WS
# ---------------------------------------------------------------------------

async def _run_analysis(session, websocket: WebSocket) -> None:
    claims = list(session.claims)
    if not claims:
        await websocket.send_text(
            json.dumps({"type": "error", "detail": "No claims to analyze yet"})
        )
        return

    contradictions, agreements, dispute_type = await detect_contradictions(claims)
    claim_map = {c.id: c for c in claims}

    await websocket.send_text(json.dumps({
        "type": "contradictions",
        "contradictions": [c.model_dump() for c in contradictions],
        "agreements": agreements,
        "dispute_type": dispute_type,
    }))

    for flag in contradictions:
        try:
            question = await generate_question(
                flag,
                claim_a=claim_map.get(flag.claim_id_a),
                claim_b=claim_map.get(flag.claim_id_b),
            )
            await websocket.send_text(json.dumps({
                "type": "clarifying_question",
                "question": question,
                "contradiction": flag.model_dump(),
            }))
        except Exception as exc:
            logger.warning("Clarifying question generation failed: %s", exc)


# ---------------------------------------------------------------------------
# Main WebSocket handler
# ---------------------------------------------------------------------------

@router.websocket("/ws/session/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("[WS] connection opened: %s", session_id)

    session = None
    transcriber: AssemblyAIClient | None = None
    handler: StreamHandler | None = None
    coordinator: ClaimExtractionCoordinator | None = None

    # ------------------------------------------------------------------
    # turn_handler_ref — single mutable slot for the active turn handler.
    # The AssemblyAI SDK accumulates listeners; we register ONE lambda at
    # connect time that delegates to whichever callable is in [0].
    # Swapping [0] atomically changes the active handler without re-registering
    # with the SDK (which would fire both old and new handlers).
    # ------------------------------------------------------------------
    turn_handler_ref: list = [None]   # [0]: Callable(TurnEvent) | None

    # One-shot Future resolved by _enrollment_turn_handler when a valid
    # name utterance arrives.
    enrollment_future: asyncio.Future | None = None

    # Per-slot duplicate-rejection counter.
    _MAX_DUPLICATE_REJECTIONS = 3
    enrollment_duplicate_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def _cleanup() -> None:
        nonlocal transcriber, handler, coordinator
        if handler:
            await handler.remove_client(websocket)
        if coordinator:
            coordinator.unregister_ws(websocket)
        if transcriber:
            try:
                await transcriber.disconnect()
            except Exception:
                pass
            transcriber = None
        if session and session.status.value == "live":
            try:
                wav_path = await session.finalize()
                await websocket.send_text(
                    json.dumps({"type": "session_ended", "wav_path": wav_path})
                )
            except Exception as exc:
                logger.exception("Error during session finalize: %s", exc)
                return
            asyncio.create_task(
                _build_and_broadcast_report(session, websocket),
                name=f"report_{session_id}",
            )

    async def _build_and_broadcast_report(session, ws: WebSocket) -> None:
        from reasoning.mediation_report import build_report
        from starlette.websockets import WebSocketDisconnect as _WSD
        try:
            report = await build_report(session)
            logger.info("Report ready for session %s", session_id)
            try:
                await ws.send_text(json.dumps({
                    "type": "report_ready",
                    "report": report.model_dump(),
                }))
            except (_WSD, Exception):
                logger.info(
                    "WS closed before report delivery for %s — "
                    "client can poll GET /report/%s", session_id, session_id,
                )
        except Exception as exc:
            logger.exception("Report build failed for %s: %s", session_id, exc)
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": f"Report failed: {exc}"}))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Enrollment turn handler — active only between begin_enrollment and
    # the enrollment result being sent. Resolves enrollment_future with
    # the first final transcript that looks like a name utterance.
    # ------------------------------------------------------------------
    def _enrollment_turn_handler(turn) -> None:
        nonlocal enrollment_future
        words = turn.words or []
        start_ms = words[0].start if words else 0
        end_ms = words[-1].end if words else 0
        label = getattr(turn, "speaker_label", None) or ""

        if not turn.end_of_turn:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(websocket.send_text(json.dumps({
                    "type": "transcript_partial",
                    "speaker_id": None,
                    "text": turn.transcript,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })))
            except Exception:
                pass
            return

        text = (turn.transcript or "").strip()
        looks_like_name = len(text.split()) >= 2 or (text and text[0].isupper())
        if not looks_like_name:
            logger.debug("Enrollment: skipping %r (too short / uncapitalised)", text)
            return

        fut = enrollment_future
        if fut is not None and not fut.done():
            fut.set_result((label, text, start_ms, end_ms))

    # ------------------------------------------------------------------
    # Ensure session + transcriber exist (idempotent).
    # Creates them the first time; subsequent calls are no-ops.
    # ------------------------------------------------------------------
    async def _ensure_session_and_transcriber() -> None:
        nonlocal session, transcriber

        if session is None:
            try:
                session = await session_manager.create(
                    session_id=session_id,
                    sample_rate=settings.sample_rate,
                )
            except ValueError:
                session = await session_manager.get_or_raise(session_id)

        if transcriber is None:
            transcriber = AssemblyAIClient(api_key=settings.assemblyai_api_key)
            transcriber.on_error(lambda err: logger.error("AssemblyAI error: %s", err))
            # ONE delegating lambda — swap turn_handler_ref[0] to change handler
            transcriber.on_turn(
                lambda turn: turn_handler_ref[0](turn) if turn_handler_ref[0] else None
            )
            await transcriber.connect(sample_rate=settings.sample_rate, session_id=session_id)

    # ------------------------------------------------------------------
    # Message loop
    # ------------------------------------------------------------------
    try:
        async for raw in websocket.iter_text():
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
                continue

            msg_type = frame.get("type")
            if msg_type == "audio_chunk":
                _data = frame.get("data", "")
                # byte length ≈ len(base64) * 3/4 — cheap approximation, no decode needed
                _approx_bytes = int(len(_data) * 0.75)
                logger.info(
                    "[WS] [%s] received message type: audio_chunk (~%d bytes)",
                    session_id, _approx_bytes,
                )
            else:
                logger.info("[WS] [%s] received message type: %s", session_id, msg_type)

            # ------------------------------------------------------------
            # begin_enrollment
            # ------------------------------------------------------------
            if msg_type == "begin_enrollment":
                slot: int = int(frame.get("slot", 1))
                logger.info("Enrollment slot %d starting, session %s", slot, session_id)

                await _ensure_session_and_transcriber()

                loop = asyncio.get_running_loop()
                enrollment_future = loop.create_future()
                turn_handler_ref[0] = _enrollment_turn_handler

                try:
                    label, text, start_ms, end_ms = await asyncio.wait_for(
                        asyncio.shield(enrollment_future), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Enrollment slot %d timed out, session %s", slot, session_id)
                    await websocket.send_text(json.dumps({
                        "type": "enrollment_result", "slot": slot, "name": None
                    }))
                    turn_handler_ref[0] = None
                    enrollment_future = None
                    continue
                except asyncio.CancelledError:
                    turn_handler_ref[0] = None
                    enrollment_future = None
                    return

                turn_handler_ref[0] = None
                enrollment_future = None

                name = await extract_name(text)

                # Duplicate detection for slot 2+
                if name and slot > 1 and session.enrollment_records:
                    norm_name = name.strip().lower()
                    dup_of: str | None = None
                    for prior in session.enrollment_records:
                        label_match = (
                            label and prior.realtime_label and
                            label.strip().upper() == prior.realtime_label.strip().upper()
                        )
                        name_match = norm_name == prior.speaker_name.strip().lower()
                        if label_match or name_match:
                            dup_of = prior.speaker_name
                            break

                    if dup_of is not None:
                        count = enrollment_duplicate_counts.get(slot, 0) + 1
                        enrollment_duplicate_counts[slot] = count
                        logger.info(
                            "Enrollment duplicate: slot=%d label=%r dup_of=%r attempt=%d",
                            slot, label, dup_of, count,
                        )
                        if count >= _MAX_DUPLICATE_REJECTIONS:
                            await websocket.send_text(json.dumps({
                                "type": "enrollment_result", "slot": slot,
                                "name": None, "give_up": True,
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "enrollment_result", "slot": slot,
                                "name": None, "duplicate_of": dup_of,
                            }))
                        continue

                if name:
                    record = EnrollmentRecord(
                        speaker_name=name,
                        realtime_label=label,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                    session.enrollment_records.append(record)
                    # Advance the enrollment window cutoff so build_report can
                    # exclude enrollment utterances from claim extraction.
                    if end_ms > session.enrollment_end_ms:
                        session.enrollment_end_ms = end_ms
                    logger.info(
                        "Enrollment slot %d accepted: name=%r label=%r end_ms=%d session=%s",
                        slot, name, label, end_ms, session_id,
                    )

                await websocket.send_text(json.dumps({
                    "type": "enrollment_result", "slot": slot, "name": name,
                }))

            # ------------------------------------------------------------
            # start_session  ← CRITICAL PATH ENTRY POINT
            # ------------------------------------------------------------
            elif msg_type == "start_session":
                raw_names: list[str] = frame.get("speakers", [])
                speaker_names: list[str] = raw_names if raw_names else ["Unknown", "Unknown"]

                await _ensure_session_and_transcriber()

                mapper = SpeakerMapper(speaker_names)

                if session.enrollment_records:
                    mapper.bind_labels(session.enrollment_records)
                    logger.info(
                        "start_session: enrollment binding — session=%s names=%s",
                        session_id, [r.speaker_name for r in session.enrollment_records],
                    )
                else:
                    mapper._log_label_map("start_session positional")
                    logger.info(
                        "start_session: positional mapping — session=%s names=%s",
                        session_id, speaker_names,
                    )

                session.speakers = mapper.speakers
                logger.info(
                    "[SESSION] [%s] speakers set: %s",
                    session_id,
                    [s.display_name for s in mapper.speakers],
                )

                handler = StreamHandler(session=session, speaker_mapper=mapper)
                await handler.add_client(websocket)

                coordinator = ClaimExtractionCoordinator(session=session)
                coordinator.register_ws(websocket)

                # Patch add_utterance to also notify the claim coordinator.
                # This is the ONLY patch — no voice_id intercept here.
                _orig_add_utterance = session.add_utterance

                async def _add_utterance_and_notify(utt, _orig=_orig_add_utterance):
                    await _orig(utt)
                    if coordinator:
                        await coordinator.on_new_utterance()

                session.add_utterance = _add_utterance_and_notify  # type: ignore[method-assign]

                # Route all SDK turns to StreamHandler
                turn_handler_ref[0] = handler.handle_turn

                logger.info(
                    "Session %s started — speakers: %s", session_id, speaker_names
                )

            # ------------------------------------------------------------
            # audio_chunk  ← CRITICAL PATH: forward to AssemblyAI + buffer
            # ------------------------------------------------------------
            elif msg_type == "audio_chunk":
                if session is None or transcriber is None:
                    await websocket.send_text(json.dumps({
                        "type": "error", "detail": "Session not started"
                    }))
                    continue

                encoded: str = frame.get("data", "")
                try:
                    pcm_bytes = base64.b64decode(encoded)
                except Exception:
                    await websocket.send_text(json.dumps({
                        "type": "error", "detail": "Invalid base64 audio data"
                    }))
                    continue

                await transcriber.stream(pcm_bytes)
                await session.append_audio(pcm_bytes)

            # ------------------------------------------------------------
            # analyze  (on-demand contradiction detection)
            # ------------------------------------------------------------
            elif msg_type == "analyze":
                if session is None:
                    await websocket.send_text(json.dumps({
                        "type": "error", "detail": "Session not started"
                    }))
                    continue
                await _run_analysis(session, websocket)

            # ------------------------------------------------------------
            # end_session
            # ------------------------------------------------------------
            elif msg_type == "end_session":
                await _cleanup()
                break

            # ------------------------------------------------------------
            # Ignored / unknown
            # ------------------------------------------------------------
            elif msg_type in {"enrollment_result_ack"}:
                pass
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", "detail": f"Unknown message type: {msg_type!r}"
                }))

    except WebSocketDisconnect:
        logger.info("[WS] connection closed: %s (WebSocketDisconnect)", session_id)
        await _cleanup()
    except Exception as exc:
        logger.exception("[WS] connection closed: %s (unhandled error: %s)", session_id, exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            pass
        await _cleanup()
    else:
        logger.info("[WS] connection closed: %s (end_session or loop exhausted)", session_id)
