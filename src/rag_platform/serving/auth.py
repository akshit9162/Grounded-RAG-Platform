"""
Minimal auth + rate limiting for the API layer. Deliberately simple —
swap for OAuth2/JWT + a Redis-backed token bucket if you need multi-tenant
auth in production; the interface (a FastAPI dependency) doesn't change.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, status

from rag_platform.config import settings

_request_log: dict[str, deque] = defaultdict(deque)


def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")

    now = time.time()
    window_start = now - 60
    log = _request_log[x_api_key]
    while log and log[0] < window_start:
        log.popleft()

    if len(log) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")

    log.append(now)
    return x_api_key
