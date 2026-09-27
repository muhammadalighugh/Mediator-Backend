import sys

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load .env first; fall back to .env.example so the app works even
        # when the user hasn't copied .env.example → .env yet.
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    assemblyai_api_key: str = ""

    # LLM — reasoning layer
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = ""

    sample_rate: int = 16000

    # MongoDB Atlas — optional persistence
    mongodb_uri: str = ""

    # Auth — secret used to sign session tokens (HMAC-SHA256)
    jwt_secret: str = "change-me-in-production"


settings = Settings()

# Warn loudly at startup if keys are missing or placeholder — prevents silent
# runtime failures where transcription works but claims/reports never appear.
if not settings.assemblyai_api_key:
    print(
        "\n"
        "  ⚠️  WARNING: ASSEMBLYAI_API_KEY is not set in .env\n"
        "  Streaming sessions will fail with HTTP 1008 Unauthorized.\n"
        "  Get your key at https://www.assemblyai.com/dashboard\n",
        file=sys.stderr,
    )

if not settings.llm_api_key:
    print(
        "\n"
        "  ⚠️  WARNING: LLM_API_KEY is not set in .env\n"
        "  Claim extraction and report generation will silently fail.\n"
        "  Claims will never appear on the ClaimBoard during live sessions.\n"
        f"  Provider: {settings.llm_provider}\n"
        "  • OpenAI:    https://platform.openai.com/api-keys\n"
        "  • Anthropic: https://console.anthropic.com/\n",
        file=sys.stderr,
    )
