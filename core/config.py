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


settings = Settings()

# Warn loudly at startup if the key is missing — prevents the cryptic
# "Unauthorized" error from AssemblyAI that only surfaces at connect time.
if not settings.assemblyai_api_key:
    print(
        "\n"
        "  ⚠️  WARNING: ASSEMBLYAI_API_KEY is not set in .env\n"
        "  Streaming sessions will fail with HTTP 1008 Unauthorized.\n"
        "  Get your key at https://www.assemblyai.com/dashboard\n",
        file=sys.stderr,
    )
