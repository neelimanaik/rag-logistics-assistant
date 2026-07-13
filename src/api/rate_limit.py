import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException

from src.api.auth import get_current_user
from src.config.settings import settings

# Module-level config (read from settings, but overridable in tests).
MAX_REQUESTS = settings.rate_limit_max
WINDOW_SECONDS = settings.rate_limit_window_seconds

# In-memory sliding window: key -> timestamps of recent requests.
# NOTE: this is per-process. For a multi-instance deployment you'd back it with
# a shared store (e.g. Redis) so the limit is enforced across all instances.
_hits: defaultdict[str, deque] = defaultdict(deque)


def reset():
    """Clear all counters (used by tests)."""
    _hits.clear()


def _allow(key):
    now = time.monotonic()
    window = _hits[key]
    # Drop timestamps outside the window.
    while window and window[0] <= now - WINDOW_SECONDS:
        window.popleft()
    if len(window) >= MAX_REQUESTS:
        return False
    window.append(now)
    return True


def rate_limit(user: str = Depends(get_current_user)):
    """Dependency: allow up to MAX_REQUESTS per WINDOW_SECONDS per user.

    Keyed by the authenticated user, so one noisy client can't starve others.
    Returns 429 when the limit is exceeded.
    """
    if not _allow(user):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and try again shortly.",
        )
    return user
