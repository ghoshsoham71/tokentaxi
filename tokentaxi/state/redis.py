# tokentaxi /state/redis.py
"""
Redis-backed state backend for multi-instance deployments.

Uses atomic ZADD + ZRANGEBYSCORE + ZREMRANGEBYSCORE pipelines so all
router instances share the same accurate picture of provider usage.

Session affinity is stored as Redis string keys with TTL.
Circuit breaker state is handled separately in breaker/circuit.py using
Redis keys with TTL so instances don't need a background job to re-admit
providers.

Requirements
------------
  pip install "redis[asyncio]>=5.0"
"""

from __future__ import annotations

import time

from .base import AbstractStateBackend
from ..constants import (
    REDIS_RPM_KEY_TMPL,
    REDIS_SESSION_KEY_TMPL,
    REDIS_TPM_KEY_TMPL,
)


class RedisStateBackend(AbstractStateBackend):
    """
    Redis-backed sliding window state backend.

    Parameters
    ----------
    redis_url:
        Connection URL, e.g. ``"redis://localhost:6379"`` or
        ``"rediss://user:pass@host:6380/0"`` for TLS.
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "redis[asyncio] is required for RedisStateBackend. "
                "Install it with: pip install 'redis[asyncio]>=5.0'"
            ) from exc

        self._client = aioredis.from_url(redis_url, decode_responses=True)

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
        cutoff = now - window_seconds

        rpm_key = REDIS_RPM_KEY_TMPL.format(provider=provider)
        tpm_key = REDIS_TPM_KEY_TMPL.format(provider=provider)

        async with self._client.pipeline(transaction=True) as pipe:
            # RPM: scored set member = unique timestamp+uuid
            pipe.zadd(rpm_key, {str(now): now})
            pipe.zremrangebyscore(rpm_key, "-inf", cutoff)
            pipe.expire(rpm_key, window_seconds * 2)

            # TPM: scored set member = "timestamp:token_count"
            tpm_member = f"{now}:{tokens}"
            pipe.zadd(tpm_key, {tpm_member: now})
            pipe.zremrangebyscore(tpm_key, "-inf", cutoff)
            pipe.expire(tpm_key, window_seconds * 2)

            await pipe.execute()

    async def get_usage(
        self,
        provider: str,
        window_seconds: int,
    ) -> tuple[int, int]:
        now = time.time()
        cutoff = now - window_seconds

        rpm_key = REDIS_RPM_KEY_TMPL.format(provider=provider)
        tpm_key = REDIS_TPM_KEY_TMPL.format(provider=provider)

        async with self._client.pipeline(transaction=True) as pipe:
            pipe.zrangebyscore(rpm_key, cutoff, "+inf")
            pipe.zrangebyscore(tpm_key, cutoff, "+inf")
            rpm_members, tpm_members = await pipe.execute()

        rpm = len(rpm_members)
        tpm = 0
        for member in tpm_members:
            try:
                _, token_str = member.rsplit(":", 1)
                tpm += int(token_str)
            except (ValueError, AttributeError):
                pass  # malformed member; skip

        return rpm, tpm

    # ------------------------------------------------------------------
    # Session affinity
    # ------------------------------------------------------------------

    async def get_session_provider(self, session_id: str) -> str | None:
        key = REDIS_SESSION_KEY_TMPL.format(session_id=session_id)
        value = await self._client.get(key)
        return value  # None if key doesn't exist

    async def set_session_provider(
        self,
        session_id: str,
        provider: str,
        ttl_seconds: int = 3600,
    ) -> None:
        key = REDIS_SESSION_KEY_TMPL.format(session_id=session_id)
        await self._client.set(key, provider, ex=ttl_seconds)

    # ------------------------------------------------------------------
    # Registry Synchronization
    # ------------------------------------------------------------------

    async def get_registered_providers(self) -> list[dict[str, Any]]:
        import json
        from ..constants import REDIS_REGISTRY_KEY
        
        raw_data = await self._client.hgetall(REDIS_REGISTRY_KEY)
        results = []
        for val in raw_data.values():
            try:
                results.append(json.loads(val))
            except json.JSONDecodeError:
                pass
        return results

    async def set_registered_provider(self, name: str, config_dict: dict[str, Any]) -> None:
        import json
        from ..constants import REDIS_REGISTRY_KEY
        
        await self._client.hset(REDIS_REGISTRY_KEY, name, json.dumps(config_dict))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.aclose()
