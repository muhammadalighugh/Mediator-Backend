"""
database.py
-----------
Async MongoDB client via Motor.

The database is OPTIONAL — the app boots and serves live sessions whether
or not MONGODB_URI is set.  Any function in this module that touches Mongo
checks `db` for None first and returns a sentinel so callers can degrade
gracefully without try/except at every call site.

Public surface
--------------
db          — AsyncIOMotorDatabase | None  (None when Mongo is unavailable)
connect()   — called once from the FastAPI lifespan; logs outcome
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# Module-level handle.  Set by connect(); None until then (or if unavailable).
db: "AsyncIOMotorDatabase | None" = None


async def connect(mongodb_uri: str) -> None:
    """
    Attempt to connect to Atlas and confirm reachability with a ping.

    Sets the module-level `db` on success; logs a warning and leaves `db`
    as None on any failure so the rest of the app degrades gracefully.
    """
    global db

    if not mongodb_uri:
        logger.warning(
            "[BOOT] MONGODB_URI not set — persistence disabled. "
            "Set MONGODB_URI in .env to enable Atlas storage."
        )
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client: AsyncIOMotorClient = AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5_000,   # fail fast if Atlas is unreachable
        )
        # Confirm reachability — raises ServerSelectionTimeoutError if down
        await client.admin.command("ping")

        db = client["argument_mediator"]
        logger.info("[BOOT] MongoDB connected — database='argument_mediator'")

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[BOOT] MongoDB unavailable (%s) — persistence disabled. "
            "Live sessions will still work.",
            exc,
        )
        db = None
