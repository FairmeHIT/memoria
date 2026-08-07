"""Authentication compatible with the Leaderboard's supported schemes."""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from memoria.config import Settings


def require_authorization(request: Request, settings: Settings) -> None:
    """Raise a safe HTTP error unless the configured credential matches."""

    if settings.auth_scheme == "none":
        return

    expected = settings.api_key
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"reason": "authentication is not configured"},
        )

    if settings.auth_scheme == "x_api_key":
        supplied = request.headers.get("X-Api-Key")
    else:
        authorization = request.headers.get("Authorization", "")
        prefix = "Token " if settings.auth_scheme == "token" else "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else None

    if supplied is None or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "authentication failed"},
        )

