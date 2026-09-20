"""
enrollment.py
-------------
Name extraction for the voice enrollment flow.

Public API:
    extract_name(transcript_text: str) -> str | None

One small LLM call per enrollment utterance. Returns the extracted name (e.g.
"Sam"), or None if no clear name was found. The caller (ws_routes.py) decides
whether to retry or fall back to typed input.

Language note: transcripts may arrive in any language (Urdu, Hindi, English,
etc.).  The system prompt is intentionally minimal — no language assumption —
so names in any script pass through unchanged.
"""

from __future__ import annotations

import logging

from reasoning.llm_client import llm_client

logger = logging.getLogger(__name__)

_EXTRACT_NAME_SYSTEM = """\
Extract the person's name from a self-introduction. Strip filler phrases such \
as "my name is", "I am", "I'm", "this is", "hi", "hey", "hello", "it's", \
"they call me", "you can call me". Return ONE name — exactly as the person \
said it, preserving capitalisation. If no clear name is present, return an \
empty string.

Examples:
  "I'm Jordan"               → "Jordan"
  "Hey, my name is Taylor"   → "Taylor"
  "Yeah so"                  → ""
  "This is Morgan speaking"  → "Morgan"
  "Hello"                    → ""

Output only the JSON object — no prose.\
"""

_EXTRACT_NAME_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
    },
    "required": ["name"],
    "additionalProperties": False,
}


async def extract_name(transcript_text: str) -> str | None:
    """
    Call the LLM to extract a person's name from a self-introduction.

    Returns the extracted name string, or None if:
      - The transcript contains no recognisable name.
      - The LLM call fails (logged at WARNING; caller retries or falls back).

    The returned string is already stripped of leading/trailing whitespace.
    Empty string from the model is treated as "no name found" and returns None.
    """
    text = (transcript_text or "").strip()
    if not text:
        return None

    try:
        result = await llm_client.complete_json(
            system=_EXTRACT_NAME_SYSTEM,
            user=text,
            schema=_EXTRACT_NAME_SCHEMA,
            max_tokens=50,
        )
    except Exception as exc:
        logger.warning("extract_name LLM call failed for %r: %s", text[:60], exc)
        return None

    name = result.get("name", "").strip()
    return name if name else None
