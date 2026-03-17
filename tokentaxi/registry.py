# tokentaxi/registry.py
"""
ProviderRegistry — thread-safe container for all registered provider clients.

Redesigned to be provider-agnostic. It stores the client, model name,
and its metadata (RPM, TPM limits, weight, enabled status).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .adapter import UniversalAdapter
from .models import ProviderConfig

class RegisteredProvider:
    """Combines a client adapter with its configuration."""
    
    def __init__(self, name: str, adapter: UniversalAdapter, config: ProviderConfig):
        self.name = name
        self.adapter = adapter
        self.config = config
        self.enabled = config.enabled

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def rpm_limit(self) -> int:
        return self.config.rpm_limit

    @property
    def tpm_limit(self) -> int:
        return self.config.tpm_limit

    @property
    def weight(self) -> float:
        return self.config.weight

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        return await self.adapter.chat(self.model, *args, **kwargs)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        return await self.adapter.stream(self.model, *args, **kwargs)

    async def close(self) -> None:
        """Attempt to close the underlying client if it has a close method."""
        client = self.adapter._client
        if hasattr(client, "close") and callable(client.close):
            if asyncio.iscoroutinefunction(client.close):
                await client.close()
            else:
                client.close()

class ProviderRegistry:
    """Thread-safe registry for LLM providers."""

    def __init__(self, state: AbstractStateBackend | None = None):
        self._providers: Dict[str, RegisteredProvider] = {}
        self._lock = asyncio.Lock()
        self._state = state

    async def register(
        self,
        name: str,
        client: Any,
        model: str,
        rpm: int,
        tpm: int,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Register a new provider with an arbitrary client."""
        async with self._lock:
            adapter = UniversalAdapter(client)
            config = ProviderConfig(
                name=name,
                model=model,
                api_key="BYOC", # Placeholder since client is already built
                rpm_limit=rpm,
                tpm_limit=tpm,
                weight=weight,
                enabled=enabled
            )
            self._providers[name] = RegisteredProvider(name, adapter, config)

            # Persist to state for cross-process discovery (e.g. dashboard)
            if self._state:
                await self._state.set_registered_provider(
                    name,
                    config.model_dump()
                )

    async def refresh_from_state(self) -> None:
        """Discover providers registered in other processes via shared state."""
        if not self._state:
            return
            
        configs = await self._state.get_registered_providers()
        async with self._lock:
            for cfg_dict in configs:
                name = cfg_dict["name"]
                if name not in self._providers:
                    # Create a "Skeleton" provider for status reporting
                    # Note: This provider cannot perform chat() calls as it has no client,
                    # but it allows the dashboard to show headroom/circuits.
                    config = ProviderConfig(**cfg_dict)
                    # Use None for client — UniversalAdapter will handle it (gracefully failing if called)
                    adapter = UniversalAdapter(None) 
                    self._providers[name] = RegisteredProvider(name, adapter, config)

    async def register_from_config(self, config: ProviderConfig) -> None:
        """
        Legacy support for registering from a ProviderConfig.
        NOTE: This now requires the client to be handled by the adapter logic
        or provided via BYOC. For now, we'll assume the client must be 
        injected via register().
        """
        # In a real scenario, we might need a way to instantiate clients from config
        # but for a provider-agnostic library, BYOC is the primary path.
        pass

    async def get_all(self) -> List[RegisteredProvider]:
        """Return all enabled providers."""
        async with self._lock:
            return [p for p in self._providers.values() if p.enabled]

    async def get(self, name: str) -> Optional[RegisteredProvider]:
        """Get a provider by name."""
        async with self._lock:
            return self._providers.get(name)

    async def names(self) -> List[str]:
        """Return all registered names."""
        async with self._lock:
            return list(self._providers.keys())

    async def close_all(self) -> None:
        """Close all registered clients."""
        async with self._lock:
            for p in self._providers.values():
                await p.close()
