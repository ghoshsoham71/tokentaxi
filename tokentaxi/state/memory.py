# tokentaxi /state/memory.py
"""
In-process, in-memory state backend.

Uses asyncio.Lock for safe concurrent access within a single event loop.
All state is lost when the process exits — appropriate for single-instance
deployments and development/testing.

Architecture note
-----------------
Each provider maintains two deques of (timestamp, value) tuples:
  - rpm_window: (timestamp,) — one entry per request
  - tpm_window: (timestamp, token_count) — one entry per request

On every read the deques are purged of entries older than window_seconds,
so RPM = len(rpm_window) and TPM = sum of token_count values.

Session affinity is stored in a plain dict with per-session expiry times.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any


from .base import AbstractStateBackend


class InMemoryStateBackend(AbstractStateBackend):
    """In-process sliding window state backend (default, zero deps)."""

    def __init__(self) -> None:
        # provider → deque of (timestamp, token_count)
        self._windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._lock = asyncio.Lock()

        # session_id → (provider, expiry_timestamp)
        self._sessions: dict[str, tuple[str, float]] = {}
        self._session_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    async def record_request(
        self,
        provider: str,
        tokens: int,
        window_seconds: int,
    ) -> None:
        now = time.time()
        async with self._lock:
            self._windows[provider].append((now, tokens))
            self._purge(provider, now, window_seconds)

    async def get_usage(
        self,
        provider: str,
        window_seconds: int,
    ) -> tuple[int, int]:
        now = time.time()
        async with self._lock:
            self._purge(provider, now, window_seconds)
            window = self._windows[provider]
            rpm = len(window)
            tpm = sum(t for _, t in window)
        return rpm, tpm

    def _purge(self, provider: str, now: float, window_seconds: int) -> None:
        """Remove entries older than window_seconds from provider's window.
        Must be called while holding self._lock."""
        cutoff = now - window_seconds
        window = self._windows[provider]
        while window and window[0][0] < cutoff:
            window.popleft()

    # ------------------------------------------------------------------
    # Session affinity
    # ------------------------------------------------------------------

    async def get_session_provider(self, session_id: str) -> str | None:
        now = time.time()
        async with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            provider, expiry = entry
            if now > expiry:
                del self._sessions[session_id]
                return None
            return provider

    async def set_session_provider(
        self,
        session_id: str,
        provider: str,
        ttl_seconds: int = 3600,
    ) -> None:
        expiry = time.time() + ttl_seconds
        async with self._session_lock:
            self._sessions[session_id] = (provider, expiry)

    # ------------------------------------------------------------------
    # Registry Synchronization
    # ------------------------------------------------------------------

    async def get_registered_providers(self) -> list[dict[str, Any]]:
        async with self._lock:
            return getattr(self, "_registry_cache", [])

    async def set_registered_provider(self, name: str, config_dict: dict[str, Any]) -> None:
        async with self._lock:
            if not hasattr(self, "_registry_cache"):
                self._registry_cache = []
            # Update or add
            self._registry_cache = [p for p in self._registry_cache if p["name"] != name]
            self._registry_cache.append(config_dict)
