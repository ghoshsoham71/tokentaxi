# tokentaxi /constants.py
"""
Default constants for the LLM Router.
All tunable values are centralised here so they can be overridden via RouterConfig
without touching internal logic.
"""

# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------
WINDOW_SECONDS: int = 60
"""Duration of the rolling usage window in seconds."""

# ---------------------------------------------------------------------------
# Scoring weights — normal priority
# ---------------------------------------------------------------------------
W_CAPACITY_NORMAL: float = 0.5
W_LATENCY_NORMAL: float = 0.3
W_STATIC_NORMAL: float = 0.2

# Scoring weights — high priority
W_CAPACITY_HIGH: float = 0.5
W_LATENCY_HIGH: float = 0.4
W_STATIC_HIGH: float = 0.1

# Scoring weights — low priority
W_CAPACITY_LOW: float = 0.3
W_LATENCY_LOW: float = 0.1
W_STATIC_LOW: float = 0.6

# ---------------------------------------------------------------------------
# Latency scoring
# ---------------------------------------------------------------------------
EMA_ALPHA: float = 0.2
"""Exponential moving average smoothing factor for latency tracking."""

LATENCY_CEILING_MS: float = 3_000.0
"""Latency value (ms) at which latency_score reaches 0."""

INITIAL_LATENCY_MS: float = 500.0
"""Assumed latency for a provider with no history yet."""

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
CIRCUIT_FAILURE_THRESHOLD: int = 5
"""Number of consecutive failures before tripping the circuit breaker."""

CIRCUIT_COOLDOWN_SECONDS: int = 30
"""Seconds the circuit remains open (provider blocked) after tripping."""

# ---------------------------------------------------------------------------
# Quota exhaustion prediction
# ---------------------------------------------------------------------------
PREDICTION_WINDOW_SECONDS: int = 120
"""Look-ahead window for quota exhaustion prediction (2 minutes)."""

PREDICTION_CONSUMPTION_MULTIPLIER: float = 3.0
"""If consumption rate is this many times above average, shift load away proactively."""

# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------
HIGH_PRIORITY_RESERVE_PCT: float = 0.2
"""
Fraction of capacity reserved exclusively for high-priority requests.
Low/normal-priority requests treat a provider as full when it has only
this fraction of headroom remaining.
"""

VALID_PRIORITIES = frozenset({"low", "normal", "high"})

# ---------------------------------------------------------------------------
# Default model strings (used when adapter creates its own client)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL: str = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-5"
DEFAULT_GEMINI_MODEL: str = "gemini-1.5-pro"
DEFAULT_GROQ_MODEL: str = "llama-3.1-70b-versatile"

# ---------------------------------------------------------------------------
# Redis key prefixes
# ---------------------------------------------------------------------------
REDIS_PREFIX: str = "tokentaxi"
REDIS_RPM_KEY_TMPL: str = REDIS_PREFIX + ":rpm:{provider}"
REDIS_TPM_KEY_TMPL: str = REDIS_PREFIX + ":tpm:{provider}"
REDIS_CIRCUIT_KEY_TMPL: str = REDIS_PREFIX + ":circuit:{provider}"
REDIS_SESSION_KEY_TMPL: str = REDIS_PREFIX + ":session:{session_id}"
REDIS_REGISTRY_KEY: str = REDIS_PREFIX + ":registry"
