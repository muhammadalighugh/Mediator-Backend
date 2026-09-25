from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ws_routes import router as ws_router
from api.http_routes import router as http_router
from api.auth_routes import router as auth_router
from api.session_routes import router as session_router
from core.config import settings
from core import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # ── startup ──────────────────────────────────────────────────────────
    logger.info(
        "[BOOT] backend ready — llm_provider=%s model=%r sample_rate=%d",
        settings.llm_provider,
        settings.llm_model or "(default)",
        settings.sample_rate,
    )
    logger.info(
        "[BOOT] assemblyai_api_key=%s  llm_api_key=%s",
        "SET" if settings.assemblyai_api_key else "MISSING",
        "SET" if settings.llm_api_key else "MISSING",
    )
    await database.connect(settings.mongodb_uri)
    yield
    # ── shutdown (nothing to do for now) ─────────────────────────────────


app = FastAPI(title="MediFact API", version="0.3.0", lifespan=lifespan)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(http_router)
app.include_router(auth_router)
app.include_router(session_router)


@app.get("/")
async def health_check() -> dict:
    return {"status": "ok", "mongo": "connected" if database.db is not None else "disabled"}
