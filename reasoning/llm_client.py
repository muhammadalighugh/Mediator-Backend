"""
llm_client.py
-------------
Provider-agnostic LLM wrapper for structured JSON output.

Supported providers (controlled by LLM_PROVIDER in .env):
  anthropic  — forces JSON via tool-use with input_schema
  openai     — forces JSON via response_format json_schema (strict=True)

Public API:
  client = LLMClient()
  result: dict = await client.complete_json(system, user, schema, max_tokens)

Both paths validate the response against the schema and retry once on
invalid output before raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model fallbacks when LLM_MODEL is not set in .env
# ---------------------------------------------------------------------------
_DEFAULT_MODELS = {
    "anthropic": "claude-3-5-haiku-20241022",
    "openai": "gpt-4o-mini",
}


def _get_model() -> str:
    return settings.llm_model or _DEFAULT_MODELS.get(settings.llm_provider, "")


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

async def _anthropic_complete_json(
    system: str, user: str, schema: dict, max_tokens: int
) -> dict:
    import anthropic  # lazy import — only needed when provider=anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
    model = _get_model()

    # Force JSON by wrapping the schema as a single tool the model MUST call.
    tool_name = "structured_output"
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": "Return the structured JSON output for the request.",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )

    # Extract the tool-use input block
    for block in message.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input  # already a dict

    raise ValueError("Anthropic response did not contain the expected tool_use block")


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

async def _openai_complete_json(
    system: str, user: str, schema: dict, max_tokens: int
) -> dict:
    import openai as openai_sdk  # lazy import

    client = openai_sdk.AsyncOpenAI(api_key=settings.llm_api_key)
    model = _get_model()

    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            },
        },
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("OpenAI returned an empty response")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(result: Any, schema: dict) -> None:
    """Minimal structural validation: check required top-level keys exist."""
    required = schema.get("required", [])
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result).__name__}")
    missing = [k for k in required if k not in result]
    if missing:
        raise ValueError(f"Response missing required keys: {missing}")


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class LLMClient:
    """Provider-agnostic JSON-completion client. Thread- and task-safe."""

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 2000,
    ) -> dict:
        """Call the configured LLM and return a validated dict.

        Retries once on schema-validation failure before re-raising.
        """
        provider = settings.llm_provider.lower()
        if provider == "anthropic":
            _call = _anthropic_complete_json
        elif provider == "openai":
            _call = _openai_complete_json
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER {provider!r}. "
                "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai in .env."
            )

        last_exc: Exception | None = None
        for attempt in range(2):  # one retry
            try:
                result = await _call(system, user, schema, max_tokens)
                _validate(result, schema)
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM attempt %d/%d failed: %s", attempt + 1, 2, exc
                )

        raise RuntimeError(
            f"LLM call failed after 2 attempts: {last_exc}"
        ) from last_exc


# Module-level singleton
llm_client = LLMClient()
