# tokentaxi /engine/scorer.py
"""
Provider scoring engine.

Ranks available providers on every routing decision using a weighted
combination of:
  - capacity_score: remaining RPM/TPM headroom (bottleneck of the two)
  - latency_score:  inverse of exponential moving average latency
  - static_score:   developer-configured preference weight

Score formula (normal priority)
--------------------------------
  score = (capacity_score x w_capacity)
        + (latency_score  x w_latency)
        + (static_score   x w_static)

Priority adjustments shift the weights so that:
  high  → capacity + latency matter most (best provider for user-facing traffic)
  low   → static weight matters most (cheapest/preferred for batch jobs)

The scorer does not make any I/O calls. It receives all state as arguments
so it can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import (
    EMA_ALPHA,
    INITIAL_LATENCY_MS,
    LATENCY_CEILING_MS,
    W_CAPACITY_HIGH,
    W_CAPACITY_LOW,
    W_CAPACITY_NORMAL,
    W_LATENCY_HIGH,
    W_LATENCY_LOW,
    W_LATENCY_NORMAL,
    W_STATIC_HIGH,
    W_STATIC_LOW,
    W_STATIC_NORMAL,
)


@dataclass
class ProviderScore:
    """Scoring result for a single provider."""

    name: str
    score: float
    capacity_score: float
    latency_score: float
    static_score: float
    rpm_headroom: float
    tpm_headroom: float
    cost_score: float = 0.0
    is_at_risk: bool = False


_PRIORITY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "high": (W_CAPACITY_HIGH, W_LATENCY_HIGH, W_STATIC_HIGH),
    "normal": (W_CAPACITY_NORMAL, W_LATENCY_NORMAL, W_STATIC_NORMAL),
    "low": (W_CAPACITY_LOW, W_LATENCY_LOW, W_STATIC_LOW),
}


class LatencyTracker:
    """
    Per-provider EMA latency tracker.

    Maintains an in-process EMA (alpha=EMA_ALPHA) of observed latencies.
    Intentionally not shared across instances — slight latency inconsistency
    is acceptable; it avoids Redis writes on every completed request.
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        self._alpha = alpha
        # provider → EMA latency in ms
        self._ema: dict[str, float] = {}

    def update(self, provider: str, latency_ms: float) -> None:
        """Update the EMA with a new observation."""
        current = self._ema.get(provider, INITIAL_LATENCY_MS)
        self._ema[provider] = self._alpha * latency_ms + (1 - self._alpha) * current

    def get(self, provider: str) -> float:
        """Return the current EMA for *provider*."""
        return self._ema.get(provider, INITIAL_LATENCY_MS)


class Scorer:
    """
    Stateless scoring engine.

    All mutable state (latency EMA, usage) is passed in as arguments
    so the scorer can be unit-tested without any I/O.
    """

    def score_provider(
        self,
        *,
        name: str,
        rpm_used: int,
        rpm_limit: int,
        tpm_used: int,
        tpm_limit: int,
        estimated_tokens: int,
        latency_ema_ms: float,
        static_weight: float,
        priority: str,
        optimization_strategy: str = "latency",
        cost_per_1k_tokens: float = 0.0,
        is_at_risk: bool = False,
        high_priority_reserve_pct: float = 0.0,
    ) -> ProviderScore | None:
        """
        Score a single provider.

        Returns None if the provider has no capacity (headroom = 0) or is
        at risk under the current consumption rate (for non-high-priority
        requests).

        Parameters
        ----------
        name:
            Provider identifier.
        rpm_used / rpm_limit:
            Current rolling window usage and configured limit.
        tpm_used / tpm_limit:
            Current rolling window token usage and configured limit.
        estimated_tokens:
            Estimated tokens for the pending request (from Estimator).
        latency_ema_ms:
            Current EMA latency for this provider.
        static_weight:
            Developer-configured preference weight (0.0–1.0).
        priority:
            Request priority: "low" | "normal" | "high".
        is_at_risk:
            True if the Predictor has flagged this provider as at risk of
            quota exhaustion.
        high_priority_reserve_pct:
            Fraction of capacity reserved for high-priority requests. Low/
            normal requests treat the provider as full when remaining headroom
            is within this fraction.
        """
        # --- Capacity headroom -------------------------------------------
        rpm_headroom = 1.0 - (rpm_used / rpm_limit) if rpm_limit > 0 else 0.0
        tpm_effective_used = tpm_used + estimated_tokens
        tpm_headroom = 1.0 - (tpm_effective_used / tpm_limit) if tpm_limit > 0 else 0.0

        # Clamp to [0, 1]
        rpm_headroom = max(0.0, min(1.0, rpm_headroom))
        tpm_headroom = max(0.0, min(1.0, tpm_headroom))

        # Enforce high-priority reserve for non-high requests
        if priority != "high":
            effective_reserve = high_priority_reserve_pct
            if rpm_headroom <= effective_reserve or tpm_headroom <= effective_reserve:
                return None  # reserved — skip for low/normal requests

        # Skip if either dimension is exhausted
        if rpm_headroom <= 0.0 or tpm_headroom <= 0.0:
            return None

        # Skip at-risk providers for non-high-priority requests
        if is_at_risk and priority != "high":
            return None

        # --- Composite capacity score (bottleneck) -----------------------
        capacity_score = min(rpm_headroom, tpm_headroom)

        # --- Latency score -----------------------------------------------
        latency_score = max(0.0, 1.0 - latency_ema_ms / LATENCY_CEILING_MS)

        # --- Static score ------------------------------------------------
        static_score = max(0.0, min(1.0, static_weight))

        # --- Cost score --------------------------------------------------
        # Normalize cost. 0.1 USD per 1k = 100 per 1M (very expensive)
        # 0.0001 USD per 1k = 0.1 per 1M (very cheap)
        cost_score = max(0.0, 1.0 - (cost_per_1k_tokens / 0.1))

        # --- Weighted sum ------------------------------------------------
        w_cap, w_lat, w_sta = _PRIORITY_WEIGHTS.get(priority, _PRIORITY_WEIGHTS["normal"])
        
        # Adjust weights based on optimization strategy
        if optimization_strategy == "cost":
            w_cap, w_lat, w_sta, w_cost = 0.2, 0.1, 0.1, 0.6
        elif optimization_strategy == "balanced":
            w_cap, w_lat, w_sta, w_cost = 0.3, 0.2, 0.1, 0.4
        else: # latency
            w_cap, w_lat, w_sta, w_cost = w_cap * 0.7, w_lat * 0.7, w_sta * 0.7, 0.3

        score = (
            capacity_score * w_cap
            + latency_score * w_lat
            + static_score * w_sta
            + cost_score * w_cost
        )

        return ProviderScore(
            name=name,
            score=score,
            capacity_score=capacity_score,
            latency_score=latency_score,
            static_score=static_score,
            rpm_headroom=rpm_headroom,
            tpm_headroom=tpm_headroom,
            cost_score=cost_score,
            is_at_risk=is_at_risk,
        )

    def rank(self, scores: list[ProviderScore]) -> list[ProviderScore]:
        """Return providers sorted by score descending."""
        return sorted(scores, key=lambda s: s.score, reverse=True)
