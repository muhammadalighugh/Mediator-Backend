"""
security.py
-----------
Password hashing (bcrypt directly) and JWT helpers (python-jose).

passlib's bcrypt wrapper has a bug with bcrypt>=4.x on Python 3.14 —
we call bcrypt directly instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from core.config import settings

_TOKEN_EXPIRE_DAYS = 30
_ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def create_token(user_id: str) -> str:
    expire = datetime.now(tz=timezone.utc) + timedelta(days=_TOKEN_EXPIRE_DAYS)
    payload: dict[str, Any] = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
        if payload.get("sub") is None:
            return None
        return payload
    except JWTError:
        return None
