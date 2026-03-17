# tokentaxi /config.py
"""
RouterConfig and related sub-configs.

Supports construction from:
  - Python dict   → RouterConfig.from_dict(data)
  - YAML file     → RouterConfig.from_yaml("router.yaml")
  - Environment   → RouterConfig.from_env()
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from .constants import (
    CIRCUIT_COOLDOWN_SECONDS,
    CIRCUIT_FAILURE_THRESHOLD,
    HIGH_PRIORITY_RESERVE_PCT,
    W_CAPACITY_NORMAL,
    W_LATENCY_NORMAL,
    W_STATIC_NORMAL,
    WINDOW_SECONDS,
)
from .models import ProviderConfig


class RoutingWeights(BaseModel):
    """Scoring weight coefficients for the default (normal) priority tier."""

    capacity: float = Field(default=W_CAPACITY_NORMAL, ge=0.0, le=1.0)
    latency: float = Field(default=W_LATENCY_NORMAL, ge=0.0, le=1.0)
    static: float = Field(default=W_STATIC_NORMAL, ge=0.0, le=1.0)


class CircuitBreakerConfig(BaseModel):
    """Configuration for the per-provider circuit breaker."""

    failure_threshold: int = Field(
        default=CIRCUIT_FAILURE_THRESHOLD,
        gt=0,
        description="Consecutive failures required to trip the circuit.",
    )
    cooldown_seconds: int = Field(
        default=CIRCUIT_COOLDOWN_SECONDS,
        gt=0,
        description="Seconds the circuit stays open before the provider is re-admitted.",
    )


class RouterConfig(BaseModel):
    """
    Top-level configuration for the LLM Router.

    Instantiate directly or use one of the factory class methods:
      RouterConfig.from_dict(data)
      RouterConfig.from_yaml(path)
      RouterConfig.from_env()
    """

    model_config = {"arbitrary_types_allowed": True}

    providers: list[ProviderConfig] = Field(default_factory=list)
    weights: RoutingWeights = Field(default_factory=RoutingWeights)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL. If set, uses Redis state backend for multi-instance deployments.",
    )
    window_seconds: int = Field(
        default=WINDOW_SECONDS,
        gt=0,
        description="Sliding window duration in seconds.",
    )
    high_priority_reserve_pct: float = Field(
        default=HIGH_PRIORITY_RESERVE_PCT,
        ge=0.0,
        le=1.0,
        description="Fraction of capacity reserved exclusively for high-priority requests.",
    )
    on_route: Callable | None = Field(
        default=None,
        description="Optional async callback fired after every routing decision. Receives a RouteEvent.",
        exclude=True,
    )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "RouterConfig":
        """Build config from a plain Python dictionary."""
        merged = {**data, **kwargs}
        return cls.model_validate(merged)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: Any) -> "RouterConfig":
        """
        Build config from a YAML file.

        Environment variable interpolation is supported:
          api_key: "${OPENAI_API_KEY}"
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for from_yaml(). Install it with: pip install pyyaml"
            ) from exc

        with open(path) as f:
            raw = f.read()

        # Interpolate ${ENV_VAR} placeholders
        def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
            var = match.group(1)
            value = os.environ.get(var)
            if value is None:
                raise EnvironmentError(
                    f"Environment variable '{var}' referenced in '{path}' is not set."
                )
            return value

        raw = re.sub(r"\$\{([^}]+)\}", _replace, raw)
        data = yaml.safe_load(raw)
        return cls.from_dict(data, **kwargs)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RouterConfig":
        """
        Build a minimal config from environment variables.

        Reads the following variables to auto-configure known providers:
          OPENAI_API_KEY   → registers an OpenAI provider
          ANTHROPIC_API_KEY → registers an Anthropic provider
          GEMINI_API_KEY   → registers a Gemini provider
          GROQ_API_KEY     → registers a Groq provider

        Optional overrides:
          tokentaxi _REDIS_URL      → redis_url
          tokentaxi _WINDOW_SECONDS → window_seconds
        """
        from .constants import (
            DEFAULT_ANTHROPIC_MODEL,
            DEFAULT_GEMINI_MODEL,
            DEFAULT_GROQ_MODEL,
            DEFAULT_OPENAI_MODEL,
        )

        providers: list[dict[str, Any]] = []

        _known = [
            ("OPENAI_API_KEY", "openai", DEFAULT_OPENAI_MODEL, 500, 200_000),
            ("ANTHROPIC_API_KEY", "anthropic", DEFAULT_ANTHROPIC_MODEL, 50, 200_000),
            ("GEMINI_API_KEY", "gemini", DEFAULT_GEMINI_MODEL, 60, 100_000),
            ("GROQ_API_KEY", "groq", DEFAULT_GROQ_MODEL, 30, 100_000),
        ]

        for env_var, name, model, rpm, tpm in _known:
            api_key = os.environ.get(env_var)
            if api_key:
                providers.append(
                    {
                        "name": name,
                        "api_key": api_key,
                        "model": model,
                        "rpm_limit": rpm,
                        "tpm_limit": tpm,
                    }
                )

        data: dict[str, Any] = {"providers": providers}

        redis_url = os.environ.get("REDIS_URL") or os.environ.get("TOKENTAXI_REDIS_URL") or os.environ.get("tokentaxi_REDIS_URL")
        if redis_url:
            data["redis_url"] = redis_url

        window = os.environ.get("tokentaxi _WINDOW_SECONDS")
        if window:
            data["window_seconds"] = int(window)

        data.update(kwargs)
        return cls.from_dict(data)
