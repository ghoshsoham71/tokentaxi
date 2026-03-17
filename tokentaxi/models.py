# tokentaxi /models.py
"""
Pydantic v2 data models used throughout tokentaxi .

These are part of the public API surface — changes here require a major
version bump once the library reaches 1.0.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .constants import VALID_PRIORITIES


class ProviderConfig(BaseModel):
    """
    Configuration for a single LLM provider.

    Used when constructing a RouterConfig via from_dict / from_yaml / from_env.
    Developers using BYOC (register()) don't need to instantiate this directly —
    the router creates it from the keyword arguments passed to register().
    """

    name: str = Field(..., description="Unique identifier, e.g. 'openai', 'anthropic'.")
    model: str = Field(..., description="Model string, e.g. 'gpt-4o', 'claude-sonnet-4-5'.")
    api_key: str = Field(..., description="Provider API key.")
    rpm_limit: int = Field(..., gt=0, description="Max requests per minute for this provider/key.")
    tpm_limit: int = Field(..., gt=0, description="Max tokens per minute for this provider/key.")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Static preference weight (0.0–1.0). Acts as tie-breaker.",
    )
    enabled: bool = Field(default=True, description="Toggle without removing from config.")


class RouterRequest(BaseModel):
    """
    A routing request submitted by the developer's application.
    """

    messages: list[dict[str, Any]] = Field(..., description="Chat messages in OpenAI format.")
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    priority: str = Field(
        default="normal",
        description="Request priority: 'low' | 'normal' | 'high'.",
    )
    session_id: str | None = Field(
        default=None,
        description="Enables sticky routing — all requests with the same session_id go to the same provider.",
    )
    force_provider: str | None = Field(
        default=None,
        description="Pin this request to a specific provider. Fallback still applies if it fails.",
    )
    optimization_strategy: str = Field(
        default="latency",
        description="Routing strategy: 'latency' (default) | 'cost' | 'balanced'.",
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(VALID_PRIORITIES)}, got '{v}'")
        return v


class RouterResponse(BaseModel):
    """
    The result returned to the developer after a successful routing decision.
    """

    content: str = Field(..., description="The LLM completion text.")
    provider: str = Field(..., description="Name of the provider that served the request.")
    model: str = Field(..., description="Model string used.")
    input_tokens: int = Field(..., description="Prompt tokens consumed.")
    output_tokens: int = Field(..., description="Completion tokens produced.")
    latency_ms: float = Field(..., description="End-to-end request latency in milliseconds.")
    attempts: int = Field(
        ...,
        description="Number of providers tried before a successful response (1 = no fallback needed).",
    )
    cost_usd: float | None = Field(default=None, description="Estimated cost of the request in USD.")


class RouteEvent(BaseModel):
    """
    Fired after every routing decision via the optional on_route callback.
    Developers can forward this to Datadog, Sentry, Slack, or any internal system.
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    headroom_pct: float = Field(description="Remaining capacity percentage at the time of routing.")
    circuit_open: bool
    timestamp: float = Field(default_factory=time.time)
    attempt_number: int
    session_id: str | None
    priority: str
    cost_usd: float | None = None
