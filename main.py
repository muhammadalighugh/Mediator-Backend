from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ws_routes import router as ws_router
from api.http_routes import router as http_router

app = FastAPI(title="Argument Mediator API", version="0.2.0")

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


@app.get("/")
async def health_check() -> dict:
    return {"status": "ok"}
