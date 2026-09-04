"""
manual_test_client.py
---------------------
Standalone asyncio script that:
  1. Connects to the backend WebSocket at ws://localhost:8000/ws/session/<id>
  2. Sends start_session
  3. Reads a local 16kHz mono 16-bit PCM WAV and streams it in 100ms chunks
  4. Sends end_session
  5. Prints every server message

Usage:
  python tests/manual_test_client.py --wav path/to/audio.wav

  # Convert any MP3 to required format (16kHz, mono, 16-bit PCM WAV):
  # ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output_16k.wav

Requirements: websockets, already in requirements.txt
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave

import websockets


DEFAULT_HOST = "ws://localhost:8000"
CHUNK_MS = 100          # stream in 100ms windows
SESSION_ID = "test-session-001"
SPEAKERS = ["Alex", "Sam"]


def read_wav_chunks(wav_path: str, sample_rate: int = 16000, chunk_ms: int = CHUNK_MS):
    """Yield raw PCM byte chunks from a WAV file.

    Validates that the file is 16kHz mono 16-bit PCM; raises ValueError otherwise.
    """
    with wave.open(wav_path, "rb") as wf:
        actual_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if actual_rate != sample_rate:
            raise ValueError(
                f"Expected sample rate {sample_rate} Hz, got {actual_rate} Hz.\n"
                f"Convert with:  ffmpeg -i input.mp3 -ar {sample_rate} -ac 1 -sample_fmt s16 output.wav"
            )
        if channels != 1:
            raise ValueError(
                f"Expected mono (1 channel), got {channels} channels.\n"
                f"Convert with:  ffmpeg -i input.mp3 -ar {sample_rate} -ac 1 -sample_fmt s16 output.wav"
            )
        if sampwidth != 2:
            raise ValueError(
                f"Expected 16-bit PCM (sampwidth=2), got sampwidth={sampwidth}.\n"
                f"Convert with:  ffmpeg -i input.mp3 -ar {sample_rate} -ac 1 -sample_fmt s16 output.wav"
            )

        frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield data


async def listen(ws) -> None:
    """Background task: print every message from the server."""
    async for raw in ws:
        try:
            msg = json.loads(raw)
            msg_type = msg.get("type", "unknown")
            print(f"\n[SERVER] {msg_type}: ", end="")
            if msg_type in ("transcript_partial", "transcript_final"):
                spk = msg.get("speaker_id") or "unknown"
                print(f'[{spk}] "{msg.get("text")}"', end="")
                if msg_type == "transcript_final":
                    print(f'  (id={msg.get("utterance_id")})', end="")
            elif msg_type == "session_ended":
                print(f'WAV saved → {msg.get("wav_path")}', end="")
            elif msg_type == "error":
                print(f'ERROR: {msg.get("detail")}', end="")
            else:
                print(json.dumps(msg, indent=2), end="")
            print()
        except Exception as exc:
            print(f"[CLIENT] Parse error: {exc} | raw={raw!r}")


async def stream_wav(ws, wav_path: str, sample_rate: int) -> None:
    """Send audio_chunk frames and then end_session."""
    print(f"[CLIENT] Streaming {wav_path!r} in {CHUNK_MS}ms chunks…")
    for chunk in read_wav_chunks(wav_path, sample_rate=sample_rate):
        encoded = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({"type": "audio_chunk", "data": encoded}))
        # Simulate real-time pacing
        await asyncio.sleep(CHUNK_MS / 1000)

    print("[CLIENT] Done streaming — sending end_session")
    await ws.send(json.dumps({"type": "end_session"}))


async def run(wav_path: str, host: str, sample_rate: int) -> None:
    uri = f"{host}/ws/session/{SESSION_ID}"
    print(f"[CLIENT] Connecting to {uri}")
    async with websockets.connect(uri) as ws:
        # Start background listener
        listener = asyncio.create_task(listen(ws))

        # Handshake: start session
        start_msg = json.dumps({"type": "start_session", "speakers": SPEAKERS})
        await ws.send(start_msg)
        print(f"[CLIENT] Sent start_session with speakers={SPEAKERS}")

        # Stream audio
        await stream_wav(ws, wav_path, sample_rate)

        # Wait up to 30 s for server to send session_ended and close
        try:
            await asyncio.wait_for(listener, timeout=30)
        except asyncio.TimeoutError:
            print("[CLIENT] Timeout waiting for session_ended")
        except websockets.exceptions.ConnectionClosedOK:
            pass

    print("[CLIENT] Connection closed — done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Argument Mediator manual test client")
    parser.add_argument(
        "--wav",
        required=True,
        help="Path to a 16kHz mono 16-bit PCM WAV file",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Backend host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Expected sample rate in Hz (default: 16000)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.wav, args.host, args.sample_rate))


if __name__ == "__main__":
    main()
