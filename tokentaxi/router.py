# tokentaxi /router.py
"""
LLMRouter — the primary class the developer interacts with.

Orchestrates the full routing pipeline:
  1. Estimate tokens for the request.
  2. Resolve session affinity (sticky routing).
  3. Score and rank all available providers.
  4. Iterate the ranked list, calling each provider until one succeeds.
  5. Record usage, update latency EMA, fire the on_route callback.
  6. Return a RouterResponse to the caller.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from .breaker.circuit import CircuitBreaker
from .config import RouterConfig
from .engine.estimator import estimate_tokens
from .engine.predictor import ExhaustionPredictor
from .engine.pricing import PricingEngine
from .engine.scorer import LatencyTracker, ProviderScore, Scorer
from .exceptions import AllProvidersFailed, CircuitOpenError, NoProvidersConfigured
from .models import RouteEvent, RouterRequest, RouterResponse
from .registry import ProviderRegistry
from .state.base import AbstractStateBackend
from .state.memory import InMemoryStateBackend


class LLMRouter:
    """
    Adaptive, rate-limit-aware LLM router.

    Parameters
    ----------
    config:
        Full router configuration. Use one of the factory class methods
        (from_dict, from_yaml, from_env) for convenient construction.
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._registry = ProviderRegistry()
        self._scorer = Scorer()
        self._latency = LatencyTracker()
        self._predictor = ExhaustionPredictor(window_seconds=config.window_seconds)
        self._pricing = PricingEngine()
        self._breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker.failure_threshold,
            cooldown_seconds=config.circuit_breaker.cooldown_seconds,
        )
        self._state: AbstractStateBackend = InMemoryStateBackend()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def pricing(self) -> PricingEngine:
        """Access the internal pricing engine (primarily for testing/inspection)."""
        return self._pricing

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "LLMRouter":
        """Construct from a plain Python dictionary."""
        from .config import RouterConfig

        on_route = kwargs.pop("on_route", None)
        cfg = RouterConfig.from_dict(data)
        if on_route is not None:
            cfg = cfg.model_copy(update={"on_route": on_route})
        return cls(cfg)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: Any) -> "LLMRouter":
        """Construct from a YAML config file."""
        from .config import RouterConfig

        on_route = kwargs.pop("on_route", None)
        cfg = RouterConfig.from_yaml(path)
        if on_route is not None:
            cfg = cfg.model_copy(update={"on_route": on_route})
        return cls(cfg)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "LLMRouter":
        """Construct from environment variables."""
        from .config import RouterConfig

        on_route = kwargs.pop("on_route", None)
        cfg = RouterConfig.from_env()
        if on_route is not None:
            cfg = cfg.model_copy(update={"on_route": on_route})
        return cls(cfg)

    # ------------------------------------------------------------------
    # Lazy async initialisation
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Lazy-initialise providers and state backend on first call."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            # State backend
            if self._config.redis_url:
                from .state.redis import RedisStateBackend

                self._state = RedisStateBackend(self._config.redis_url)

            # Link state to registry for syncing
            self._registry._state = self._state

            # Fetch pricing
            await self._pricing.fetch_pricing()

            self._initialized = True

    # ------------------------------------------------------------------
    # BYOC — Bring Your Own Client (runtime registration)
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        client: Any,
        model: str,
        rpm: int,
        tpm: int,
        weight: float = 1.0,
    ) -> None:
        """
        Register a pre-configured provider client at runtime.

        This is the BYOC interface. The developer keeps their existing,
        fully configured SDK client and the router wraps it.

        Parameters
        ----------
        name:
            Provider identifier. Must be one of: openai, anthropic, gemini, groq.
            For custom providers, subclass BaseProvider and call
            registry.register_adapter() directly.
        client:
            A pre-configured async SDK client instance.
        model:
            Model string to use with this client.
        rpm:
            Requests-per-minute limit for this provider key.
        tpm:
            Tokens-per-minute limit for this provider key.
        weight:
            Static preference weight (0.0–1.0).
        """
        # Queue the registration — will be applied on first chat() call
        # so we can stay synchronous here.
        self._pending_byoc = getattr(self, "_pending_byoc", [])
        self._pending_byoc.append(
            dict(name=name, client=client, model=model, rpm=rpm, tpm=tpm, weight=weight)
        )

    async def _flush_pending_registrations(self) -> None:
        """Apply any queued BYOC registrations."""
        pending = getattr(self, "_pending_byoc", [])
        for kwargs in pending:
            await self._registry.register(**kwargs)
        self._pending_byoc = []

    # ------------------------------------------------------------------
    # Core routing logic
    # ------------------------------------------------------------------

    async def _get_ranked_providers(
        self,
        estimated_tokens: int,
        priority: str,
        optimization_strategy: str,
        force_provider: str | None,
        session_id: str | None,
    ) -> list[Any]:
        """
        Return providers sorted by score (highest first).

        Applies:
          - Session affinity (sticky routing) — pinned provider goes first.
          - Force provider — put pinned provider at front of list.
          - Circuit breaker — skip open circuits.
          - Scoring — rank remaining providers.
        """
        all_providers = await self._registry.get_all()
        if not all_providers:
            raise NoProvidersConfigured(
                "No providers are registered. Use router.register() or provide "
                "providers in RouterConfig."
            )

        # Resolve session affinity
        pinned_name: str | None = None
        if session_id:
            pinned_name = await self._state.get_session_provider(session_id)

        # Force provider overrides session affinity
        if force_provider:
            pinned_name = force_provider

        scored: list[ProviderScore] = []
        unscored_fallback: list[Any] = []

        for provider in all_providers:
            # Check circuit breaker
            if await self._breaker.is_open(provider.name):
                continue

            rpm_used, tpm_used = await self._state.get_usage(
                provider.name, self._config.window_seconds
            )

            is_at_risk = self._predictor.is_at_risk(
                provider.name,
                rpm_used=rpm_used,
                rpm_limit=provider.rpm_limit,
                tpm_used=tpm_used,
                tpm_limit=provider.tpm_limit,
            )

            # Get pricing for this model
            costs = self._pricing.get_cost(provider.model) or {"prompt": 0.0, "completion": 0.0}
            
            ps = self._scorer.score_provider(
                name=provider.name,
                rpm_used=rpm_used,
                rpm_limit=provider.rpm_limit,
                tpm_used=tpm_used,
                tpm_limit=provider.tpm_limit,
                estimated_tokens=estimated_tokens,
                latency_ema_ms=self._latency.get(provider.name),
                static_weight=provider.weight,
                priority=priority,
                optimization_strategy=optimization_strategy,
                cost_per_1k_tokens=costs["prompt"] * 1000,
                is_at_risk=is_at_risk,
                high_priority_reserve_pct=self._config.high_priority_reserve_pct,
            )
            if ps is not None:
                scored.append(ps)
            else:
                # Provider has no capacity; keep as last-resort fallback
                unscored_fallback.append(provider)

        ranked_scores = self._scorer.rank(scored)

        # Build final ordered provider list
        provider_map = {p.name: p for p in all_providers}
        ranked: list[Any] = [
            provider_map[ps.name] for ps in ranked_scores if ps.name in provider_map
        ]

        # Promote pinned provider to front (session affinity / force)
        if pinned_name and pinned_name in provider_map:
            ranked = [p for p in ranked if p.name != pinned_name]
            pinned = provider_map[pinned_name]
            if not await self._breaker.is_open(pinned_name):
                ranked.insert(0, pinned)

        # Append capacity-exhausted providers as last resort
        for p in unscored_fallback:
            if p not in ranked:
                ranked.append(p)

        return ranked

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(self, request: RouterRequest) -> RouterResponse:
        """
        Route a chat completion request to the best available provider.
        """
        await self._ensure_initialized()
        await self._flush_pending_registrations()

        estimated_tokens = estimate_tokens(request.messages)

        ranked = await self._get_ranked_providers(
            estimated_tokens=estimated_tokens,
            priority=request.priority,
            optimization_strategy=request.optimization_strategy,
            force_provider=request.force_provider,
            session_id=request.session_id,
        )

        if not ranked:
            raise NoProvidersConfigured(
                "No providers are available (all circuits open or no capacity)."
            )

        errors: list[Exception] = []
        for attempt_number, provider in enumerate(ranked, start=1):
            try:
                await self._breaker.guard(provider.name)
            except CircuitOpenError:
                continue

            t0 = time.monotonic()
            try:
                content, input_tokens, output_tokens = await provider.chat(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            except Exception as exc:
                errors.append(exc)
                await self._breaker.record_failure(provider.name)
                continue

            latency_ms = (time.monotonic() - t0) * 1000

            # Record success
            total_tokens = input_tokens + output_tokens
            await self._state.record_request(
                provider.name, total_tokens, self._config.window_seconds
            )
            self._latency.update(provider.name, latency_ms)
            self._predictor.record(provider.name, total_tokens)
            await self._breaker.record_success(provider.name)
            
            # Calculate cost
            cost_usd = self._pricing.calculate_request_cost(provider.model, input_tokens, output_tokens)

            # Session affinity — pin provider for future requests
            if request.session_id:
                await self._state.set_session_provider(request.session_id, provider.name)

            # Calculate headroom for RouteEvent
            rpm_used, tpm_used = await self._state.get_usage(
                provider.name, self._config.window_seconds
            )
            headroom_pct = min(
                (1 - (rpm_used / provider.rpm_limit if provider.rpm_limit > 0 else 1.0)) * 100,
                (1 - (tpm_used / provider.tpm_limit if provider.tpm_limit > 0 else 1.0)) * 100,
            )

            # Fire callback
            if self._config.on_route:
                event = RouteEvent(
                    provider=provider.name,
                    model=provider.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    headroom_pct=headroom_pct,
                    circuit_open=False,
                    attempt_number=attempt_number,
                    session_id=request.session_id,
                    priority=request.priority,
                    cost_usd=cost_usd,
                )
                try:
                    await self._config.on_route(event)
                except Exception:
                    pass  # Callback errors must not affect routing

            return RouterResponse(
                content=content,
                provider=provider.name,
                model=provider.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                attempts=attempt_number,
                cost_usd=cost_usd,
            )

        raise AllProvidersFailed(
            f"All {len(errors)} provider(s) failed.",
            attempts=len(errors),
            errors=errors,
        )

    async def stream(self, request: RouterRequest) -> AsyncIterator[str]:
        """
        Route a streaming chat completion request.
        """
        await self._ensure_initialized()
        await self._flush_pending_registrations()

        estimated_tokens = estimate_tokens(request.messages)
        ranked = await self._get_ranked_providers(
            estimated_tokens=estimated_tokens,
            priority=request.priority,
            optimization_strategy=request.optimization_strategy,
            force_provider=request.force_provider,
            session_id=request.session_id,
        )

        if not ranked:
            raise NoProvidersConfigured(
                "No providers are available (all circuits open or no capacity)."
            )

        errors: list[Exception] = []
        for provider in ranked:
            try:
                await self._breaker.guard(provider.name)
            except CircuitOpenError:
                continue

            t0 = time.monotonic()
            try:
                stream = await provider.stream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                async for chunk in stream:
                    yield chunk
                # Approximate post-stream record (token count is estimated)
                latency_ms = (time.monotonic() - t0) * 1000
                await self._state.record_request(
                    provider.name, estimated_tokens, self._config.window_seconds
                )
                self._latency.update(provider.name, latency_ms)
                self._predictor.record(provider.name, estimated_tokens)
                await self._breaker.record_success(provider.name)
                if request.session_id:
                    await self._state.set_session_provider(request.session_id, provider.name)
                return
            except Exception as exc:
                errors.append(exc)
                await self._breaker.record_failure(provider.name)
                continue

        raise AllProvidersFailed(
            f"All {len(errors)} provider(s) failed during streaming.",
            attempts=len(errors),
            errors=errors,
        )

    async def status(self) -> dict[str, Any]:
        """
        Return the current status of the registry.
        """
        await self._ensure_initialized()
        await self._flush_pending_registrations()
        
        # Discover providers from shared state (e.g. registered in other processes)
        await self._registry.refresh_from_state()

        result: dict[str, Any] = {}
        providers = await self._registry.get_all()

        for provider in providers:
            rpm_used, tpm_used = await self._state.get_usage(
                provider.name, self._config.window_seconds
            )
            circuit_status = await self._breaker.get_status(provider.name)

            rpm_headroom_pct = max(0.0, (1 - rpm_used / provider.rpm_limit if provider.rpm_limit > 0 else 1.0) * 100)
            tpm_headroom_pct = max(0.0, (1 - tpm_used / provider.tpm_limit if provider.tpm_limit > 0 else 1.0) * 100)
            headroom_pct = min(rpm_headroom_pct, tpm_headroom_pct)

            result[provider.name] = {
                "rpm_used": rpm_used,
                "rpm_limit": provider.rpm_limit,
                "tpm_used": tpm_used,
                "tpm_limit": provider.tpm_limit,
                "headroom_pct": round(headroom_pct, 1),
                "circuit_open": circuit_status["circuit_open"],
                "avg_latency_ms": round(self._latency.get(provider.name), 1),
            }

        return result

    async def close(self) -> None:
        """Release all resources."""
        await self._registry.close_all()
        await self._state.close()

    async def __aenter__(self) -> "LLMRouter":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
