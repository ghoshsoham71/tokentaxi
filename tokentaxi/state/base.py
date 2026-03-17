# tokentaxi /state/base.py
"""
Abstract interface that every state backend must implement.

The state backend is responsible for:
  - Tracking per-provider RPM (request count in rolling window)
  - Tracking per-provider TPM (token count in rolling window)
  - Storing and looking up session → provider affinity (sticky routing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractStateBackend(ABC):
    """Interface contract for all state backend implementations."""

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    @abstractmethod
    async def record_request(
        self,
        provider: str,
        tokens: int,
        window_seconds: int,
    ) -> None:
        """
        Record a completed request for *provider*.

        Parameters
        ----------
        provider:
            Provider identifier string.
        tokens:
            Number of tokens consumed by this request (input + output).
        window_seconds:
            Rolling window duration. Entries older than this must be discarded.
        """

    @abstractmethod
    async def get_usage(
        self,
        provider: str,
        window_seconds: int,
    ) -> tuple[int, int]:
        """
        Return current rolling usage for *provider*.

        Returns
        -------
        (rpm, tpm)
            rpm: request count in the current window.
            tpm: token count in the current window.
        """

    # ------------------------------------------------------------------
    # Session affinity
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_session_provider(self, session_id: str) -> str | None:
        """Return the pinned provider name for *session_id*, or None."""

    @abstractmethod
    async def set_session_provider(
        self,
        session_id: str,
        provider: str,
        ttl_seconds: int = 3600,
    ) -> None:
        """Pin *provider* for *session_id*. Expires after *ttl_seconds*."""

    # ------------------------------------------------------------------
    # Registry Synchronization
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_registered_providers(self) -> list[dict[str, Any]]:
        """Return all registered providers from the shared state."""

    @abstractmethod
    async def set_registered_provider(self, name: str, config_dict: dict[str, Any]) -> None:
        """Persist a provider registration to the shared state."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release any resources held by this backend (e.g. Redis connections)."""
