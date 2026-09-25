"""
auth_routes.py
--------------
Real auth endpoints — email + password, JWT tokens, MongoDB Atlas storage.

POST /auth/register  — create account (email, password, name)
POST /auth/login     — verify password, return JWT
GET  /auth/me        — return current user (requires Bearer token)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from core import database
from core.security import hash_password, verify_password, create_token, decode_token
from models.schemas import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_COLLECTION = "users"
_bearer = HTTPBearer()


# ---------------------------------------------------------------------------
# Request / response bodies
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "database unavailable"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> User:
    """Dependency — validates Bearer JWT and returns the User."""
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if database.db is None:
        raise _unavailable()

    col = database.db[_COLLECTION]
    doc = await col.find_one({"id": payload["sub"]})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return User(id=doc["id"], name=doc["name"], email=doc["email"], created_at=doc["created_at"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registration disabled — endpoint commented out, login only
# ---------------------------------------------------------------------------
# @router.post(
#     "/register",
#     response_model=AuthResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="Create a new account",
# )
# async def register(body: RegisterRequest) -> AuthResponse:
#     if database.db is None:
#         raise _unavailable()
#
#     email = body.email.strip().lower()
#     name = body.name.strip()
#     password = body.password
#
#     if len(name) < 1:
#         raise HTTPException(status_code=422, detail="Name is required")
#     if len(password) < 6:
#         raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
#
#     col = database.db[_COLLECTION]
#
#     # Check for existing account
#     existing = await col.find_one({"email": email})
#     if existing:
#         raise HTTPException(status_code=409, detail="An account with this email already exists")
#
#     now = datetime.now(tz=timezone.utc)
#     user_id = str(uuid.uuid4())
#     hashed = hash_password(password)
#
#     await col.insert_one({
#         "id": user_id,
#         "name": name,
#         "email": email,
#         "password_hash": hashed,
#         "created_at": now,
#     })
#
#     logger.info("[AUTH] registered user email=%r id=%s", email, user_id)
#
#     user = User(id=user_id, name=name, email=email, created_at=now)
#     token = create_token(user_id)
#     return AuthResponse(token=token, user=user)


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Sign in with email and password",
)
async def login(body: LoginRequest) -> AuthResponse:
    if database.db is None:
        raise _unavailable()

    col = database.db[_COLLECTION]
    email = body.email.strip().lower()

    doc = await col.find_one({"email": email})

    WRONG = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
    )

    if doc is None:
        raise WRONG

    if not verify_password(body.password, doc.get("password_hash", "")):
        raise WRONG

    logger.info("[AUTH] login email=%r id=%s", email, doc["id"])

    user = User(id=doc["id"], name=doc["name"], email=doc["email"], created_at=doc["created_at"])
    token = create_token(doc["id"])
    return AuthResponse(token=token, user=user)


@router.get(
    "/me",
    response_model=User,
    summary="Return the current authenticated user",
)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
