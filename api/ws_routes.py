"""
ws_routes.py
------------
WebSocket endpoint: /ws/session/{session_id}

Client → server frames
  {"type":"start_session","speakers":["Alex","Sam"]}
  {"type":"audio_chunk","data":"<base64-encoded PCM>"}
  {"type":"analyze"}          ← triggers contradiction detection on-demand
  {"type":"end_session"}

Server → client frames
  {"type":"transcript_partial", "speaker_id":..., "text":..., "start_ms":..., "end_ms":...}
  {"type":"transcript_final",   "speaker_id":..., "text":..., "start_ms":..., "end_ms":..., "utterance_id":...}
  {"type":"claims_updated",     "claims":[...]}
  {"type":"contradictions",     "contradictions":[...], "agreements":[...], "dispute_type":...}
  {"type":"clarifying_question","question":..., "contradiction":...}
  {"type":"report_ready",       "report":{...}}
  {"type":"session_ended",      "wav_path":...}
  {"type":"error",              "detail":...}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from core.session import session_manager
from transcription.assemblyai_client import AssemblyAIClient
from transcription.speaker_mapper import SpeakerMapper
from transcription.stream_handler import StreamHandler
from reasoning.claim_extractor import ClaimExtractionCoordinator
from reasoning.contradiction_detector import detect_contradictions
from reasoning.clarifying_questions import generate_question

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: send a contradiction analysis result to a single WS client
# ---------------------------------------------------------------------------

async def _run_analysis(session, websocket: WebSocket) -> None:
    """Run contradiction detection and broadcast results + clarifying questions."""
    claims = list(session.claims)
    if not claims:
        await websocket.send_text(
            json.dumps({"type": "error", "detail": "No claims to analyze yet"})
        )
        return

    contradictions, agreements, dispute_type = await detect_contradictions(claims)

    claim_map = {c.id: c for c in claims}

    await websocket.send_text(
        json.dumps({
            "type": "contradictions",
            "contradictions": [c.model_dump() for c in contradictions],
            "agreements": agreements,
            "dispute_type": dispute_type,
        })
    )

    # Generate and broadcast a clarifying question for each contradiction
    for flag in contradictions:
        try:
            question = await generate_question(
                flag,
                claim_a=claim_map.get(flag.claim_id_a),
                claim_b=claim_map.get(flag.claim_id_b),
            )
            await websocket.send_text(
                json.dumps({
                    "type": "clarifying_question",
                    "question": question,
                    "contradiction": flag.model_dump(),
                })
            )
        except Exception as exc:
            logger.warning("Clarifying question generation failed: %s", exc)


# ---------------------------------------------------------------------------
# Main WebSocket handler
# ---------------------------------------------------------------------------

@router.websocket("/ws/session/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    session = None
    transcriber: AssemblyAIClient | None = None
    handler: StreamHandler | None = None
    coordinator: ClaimExtractionCoordinator | None = None

    async def _cleanup() -> None:
        """Tear down transcriber, finalize session, run report pipeline."""
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

            # Launch report pipeline as a background task so WS isn't blocked
            asyncio.create_task(
                _build_and_broadcast_report(session, websocket),
                name=f"report_{session_id}",
            )

    async def _build_and_broadcast_report(session, ws: WebSocket) -> None:
        from reasoning.mediation_report import build_report
        from starlette.websockets import WebSocketDisconnect
        try:
            report = await build_report(session)
            logger.info("Report ready for session %s — attempting WS delivery", session_id)
            # Try to push report over WS. If the client has already navigated
            # away the socket will be closed — that's fine, the report is still
            # stored on session and retrievable via GET /report/{session_id}.
            try:
                await ws.send_text(
                    json.dumps({
                        "type": "report_ready",
                        "report": report.model_dump(),
                    })
                )
            except (WebSocketDisconnect, Exception):
                logger.info(
                    "WS closed before report delivery for session %s — "
                    "client can fetch via GET /report/%s",
                    session_id, session_id,
                )
        except Exception as exc:
            logger.exception("Report build failed for session %s: %s", session_id, exc)
            try:
                await ws.send_text(
                    json.dumps({"type": "error", "detail": f"Report failed: {exc}"})
                )
            except Exception:
                pass

    try:
        async for raw in websocket.iter_text():
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": "Invalid JSON"})
                )
                continue

            msg_type = frame.get("type")

            # ----------------------------------------------------------------
            # start_session
            # ----------------------------------------------------------------
            if msg_type == "start_session":
                speaker_names: list[str] = frame.get("speakers", ["Speaker A", "Speaker B"])

                try:
                    session = await session_manager.create(
                        session_id=session_id,
                        sample_rate=settings.sample_rate,
                    )
                except ValueError:
                    session = await session_manager.get_or_raise(session_id)

                mapper = SpeakerMapper(speaker_names)
                session.speakers = mapper.speakers

                handler = StreamHandler(session=session, speaker_mapper=mapper)
                await handler.add_client(websocket)

                coordinator = ClaimExtractionCoordinator(session=session)
                coordinator.register_ws(websocket)

                # Patch StreamHandler to also notify coordinator after each utterance
                _orig_add_utterance = session.add_utterance

                async def _add_utterance_and_extract(utt, _orig=_orig_add_utterance):
                    await _orig(utt)
                    if coordinator:
                        await coordinator.on_new_utterance()

                session.add_utterance = _add_utterance_and_extract  # type: ignore[method-assign]

                transcriber = AssemblyAIClient(api_key=settings.assemblyai_api_key)
                transcriber.on_turn(handler.handle_turn)
                transcriber.on_error(
                    lambda err: logger.error("AssemblyAI error: %s", err)
                )
                await transcriber.connect(sample_rate=settings.sample_rate)

                logger.info(
                    "Session %s started with speakers: %s", session_id, speaker_names
                )

            # ----------------------------------------------------------------
            # audio_chunk
            # ----------------------------------------------------------------
            elif msg_type == "audio_chunk":
                if session is None or transcriber is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "Session not started"})
                    )
                    continue

                encoded: str = frame.get("data", "")
                try:
                    pcm_bytes = base64.b64decode(encoded)
                except Exception:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "Invalid base64 audio data"})
                    )
                    continue

                await transcriber.stream(pcm_bytes)
                await session.append_audio(pcm_bytes)

            # ----------------------------------------------------------------
            # analyze  (on-demand contradiction detection)
            # ----------------------------------------------------------------
            elif msg_type == "analyze":
                if session is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "Session not started"})
                    )
                    continue
                await _run_analysis(session, websocket)

            # ----------------------------------------------------------------
            # end_session
            # ----------------------------------------------------------------
            elif msg_type == "end_session":
                await _cleanup()
                break

            else:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "detail": f"Unknown message type: {msg_type!r}"}
                    )
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
        await _cleanup()
    except Exception as exc:
        logger.exception("Unhandled error in session %s: %s", session_id, exc)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "detail": str(exc)})
            )
        except Exception:
            pass
        await _cleanup()
