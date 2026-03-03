# tokentaxi

**Adaptive rate-limit-aware LLM routing. Bring your own clients.**

`tokentaxi` is a lightweight Python library that sits between your application and your LLM providers. It intelligently routes every request based on real-time provider health, rate-limit headroom, latency, and request priority — with zero network hops and no external dependencies beyond an optional Redis connection.

---

## The core idea — Bring Your Own Client (BYOC)

Every other routing solution asks you to replace your LLM SDK. `tokentaxi` doesn't. You keep your existing, fully configured clients. The router wraps them and adds routing intelligence on top.

```python
# What you'd write today — fragile, manual, scattered
try:
    response = openai_client.chat(...)
except RateLimitError:
    try:
        response = anthropic_client.chat(...)
    except RateLimitError:
        response = gemini_client.chat(...)

# What you write with tokentaxi — once, tested, intelligent
router = LLMRouter.from_dict({"providers": [...]})
response = await router.chat(RouterRequest(messages=messages))
```

---

## Features

| Feature | Description |
|---|---|
| **Adaptive rate-limit-aware routing** | Tracks RPM and TPM in a rolling 60-second window. Routes to the provider with the most headroom. |
| **Automatic fallback** | Transparently retries with the next-ranked provider on any failure. |
| **Circuit breaker** | Trips per-provider circuits after N failures. Auto-recovers after cooldown. Redis-backed for multi-instance. |
| **Latency-aware scoring (EMA)** | Tracks latency per-provider using an exponential moving average. Slower providers get lower scores. |
| **Quota exhaustion prediction** | Proactively shifts load before a provider hits its hard limit. |
| **Session affinity** | Pass a `session_id` to pin all requests in a conversation to the same provider. |
| **Priority lanes** | Tag requests `"high"`, `"normal"`, or `"low"`. High-priority traffic gets the best available provider. |
| **Provider pinning** | Override the router for a specific call via `force_provider`. |
| **Static preference weights** | Express a preference for one provider over others via a `weight` parameter. |
| **BYOC** | Register your own pre-configured SDK clients. The router wraps them, not the other way around. |

---

## Installation

```bash
# Core library — in-memory state, no extra deps
pip install tokentaxi

# Multi-instance deployments (Redis-backed state)
pip install "tokentaxi[redis]"

# Local real-time dashboard
pip install "tokentaxi[dashboard]"

# CLI (status command, watch mode)
pip install "tokentaxi[cli]"

# YAML config support
pip install "tokentaxi[yaml]"

# Everything
pip install "tokentaxi[all]"
```

---

## Quick Start

### From a dictionary

```python
from tokentaxi import LLMRouter, RouterRequest

router = LLMRouter.from_dict({
    "providers": [
        {"name": "openai",    "api_key": "sk-...",    "model": "gpt-4o",            "rpm_limit": 500, "tpm_limit": 200_000},
        {"name": "anthropic", "api_key": "sk-ant-...", "model": "claude-sonnet-4-5", "rpm_limit": 50,  "tpm_limit": 200_000},
        {"name": "groq",      "api_key": "gsk-...",   "model": "llama-3.1-70b",     "rpm_limit": 30,  "tpm_limit": 100_000},
    ]
})

response = await router.chat(RouterRequest(
    messages=[{"role": "user", "content": "Summarize this article..."}],
    priority="normal",
))

print(response.content)
print(response.provider)    # "anthropic"
print(response.latency_ms)  # 310.4
print(response.attempts)    # 1
```

### BYOC — Bring Your Own Client

```python
import openai
import anthropic
from tokentaxi import LLMRouter

openai_client    = openai.AsyncOpenAI(api_key="sk-...", timeout=30, max_retries=0)
anthropic_client = anthropic.AsyncAnthropic(api_key="sk-ant-...", timeout=30)

router = LLMRouter.from_dict({"providers": []})
router.register("openai",    client=openai_client,    model="gpt-4o",            rpm=500, tpm=200_000)
router.register("anthropic", client=anthropic_client, model="claude-sonnet-4-5", rpm=50,  tpm=200_000)

response = await router.chat(RouterRequest(messages=[{"role": "user", "content": "Hello"}]))
```

### From a YAML file

```python
router = LLMRouter.from_yaml("router.yaml")
```

### From environment variables

```python
# Reads OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
router = LLMRouter.from_env()
```

---

## Priority Lanes

```python
# User-facing — use best available provider
response = await router.chat(RouterRequest(messages=messages, priority="high"))

# Background batch job — don't burn premium quota
response = await router.chat(RouterRequest(messages=messages, priority="low"))
```

## Session Affinity

```python
# All requests with the same session_id go to the same provider
response = await router.chat(RouterRequest(
    messages=conversation_history,
    session_id="user-session-abc123",
))
```

## Provider Pinning

```python
# Force a specific provider — fallback still applies if it fails
response = await router.chat(RouterRequest(
    messages=messages,
    force_provider="anthropic",
))
```

## Streaming

```python
async for chunk in router.stream(RouterRequest(messages=messages)):
    print(chunk, end="", flush=True)
```

## Callbacks

```python
async def on_route(event: RouteEvent):
    print(f"Routed to {event.provider} | latency: {event.latency_ms}ms")
    # Send to Datadog, Sentry, Slack, etc.

router = LLMRouter.from_yaml("router.yaml", on_route=on_route)
```

---

## Provider Status

### In code

```python
status = await router.status()
# {
#   "openai":    {"rpm_used": 423, "rpm_limit": 500, "headroom_pct": 15.4, "circuit_open": False, "avg_latency_ms": 312},
#   "anthropic": {"rpm_used": 12,  "rpm_limit": 50,  "headroom_pct": 76.0, "circuit_open": False, "avg_latency_ms": 410},
# }
```

### FastAPI integration

```python
@app.get("/llm/status")
async def llm_status():
    return await router.status()
```

### CLI

```bash
tokentaxi status --config router.yaml
tokentaxi status --watch --interval 3    # live-updating like htop
```

### Dashboard

```bash
pip install "tokentaxi[dashboard]"
tokentaxi dashboard --config router.yaml
# → open http://localhost:8501
```

---

## Scoring Formula

```
score = (capacity_score × w_capacity) + (latency_score × w_latency) + (static_score × w_static)

capacity_score = min(rpm_headroom, tpm_headroom)
rpm_headroom   = 1 - (rpm_used / rpm_limit)
tpm_headroom   = 1 - ((tpm_used + estimated_tokens) / tpm_limit)
latency_score  = max(0, 1 - (latency_ema_ms / 3000))
static_score   = provider.weight

# Default weights (normal priority)
w_capacity = 0.5  |  w_latency = 0.3  |  w_static = 0.2

# High priority
w_capacity = 0.5  |  w_latency = 0.4  |  w_static = 0.1

# Low priority
w_capacity = 0.3  |  w_latency = 0.1  |  w_static = 0.6
```

---

## Multi-Instance Deployments

```bash
pip install "tokentaxi[redis]"
```

```yaml
# router.yaml
redis_url: "redis://localhost:6379"
```

With Redis, all router instances share the same accurate picture of provider state — sliding window usage, circuit breaker status, and session affinity. Scale horizontally without coordination.

---

## Project Structure

```
tokentaxi/
├── __init__.py          # Public API exports
├── router.py            # LLMRouter — main class
├── config.py            # RouterConfig, RoutingWeights, CircuitBreakerConfig
├── models.py            # RouterRequest, RouterResponse, ProviderConfig, RouteEvent
├── exceptions.py        # AllProvidersFailed, NoProvidersConfigured, TokenLimitExceeded
├── constants.py         # Default weights, thresholds, window sizes
├── cli.py               # typer CLI (status, dashboard commands)
├── _dashboard.py        # Streamlit dashboard
├── engine/
│   ├── scorer.py        # Provider scoring (capacity + latency + static weight)
│   ├── estimator.py     # Pre-flight token count estimation (tiktoken)
│   └── predictor.py     # Quota exhaustion prediction
├── providers/
│   ├── base.py          # BaseProvider abstract class
│   ├── registry.py      # ProviderRegistry (thread-safe)
│   ├── openai.py        # OpenAI adapter
│   ├── anthropic.py     # Anthropic adapter
│   ├── gemini.py        # Gemini adapter
│   └── groq.py          # Groq adapter
├── state/
│   ├── base.py          # AbstractStateBackend interface
│   ├── memory.py        # InMemoryStateBackend (default, zero deps)
│   └── redis.py         # RedisStateBackend (multi-instance)
└── breaker/
    └── circuit.py       # CircuitBreaker (per-provider)

tests/
├── conftest.py          # Shared fixtures
├── test_scorer.py       # Scorer unit tests
├── test_circuit_breaker.py
├── test_state_memory.py
├── test_predictor.py
└── test_router.py       # Integration tests (mocked providers)

examples/
├── quickstart.py        # Dict config quickstart
├── byoc.py              # BYOC example
├── streaming.py         # Streaming example
└── router.yaml          # YAML config example
```

---

## Running Tests

```bash
pip install "tokentaxi[dev]"
pytest
```

---

## Publishing

```bash
pip install hatch twine
hatch build
twine upload dist/*
```

---

## Licence

MIT
