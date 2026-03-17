# tokentaxi/engine/pricing.py
"""
PricingEngine — fetches and caches LLM pricing from OpenRouter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

class PricingEngine:
    """Fetches real-time pricing from OpenRouter and caches it."""
    
    def __init__(self, cache_ttl_seconds: int = 3600):
        self._cache_ttl = cache_ttl_seconds
        self._pricing_cache: Dict[str, Dict[str, float]] = {}
        self._last_fetch_time: float = 0
        self._lock = asyncio.Lock()

    async def fetch_pricing(self) -> None:
        """Fetch model pricing from OpenRouter API."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_fetch_time < self._cache_ttl and self._pricing_cache:
                return

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(OPENROUTER_MODELS_URL)
                    response.raise_for_status()
                    data = response.json()
                    
                    new_cache = {}
                    for model in data.get("data", []):
                        model_id = model.get("id")
                        pricing = model.get("pricing", {})
                        # Costs are in USD per 1 token in the API, we'll keep it consistent
                        # OpenRouter returns pricing as strings sometimes or floats
                        prompt_cost = float(pricing.get("prompt", 0))
                        completion_cost = float(pricing.get("completion", 0))
                        
                        new_cache[model_id] = {
                            "prompt": prompt_cost,
                            "completion": completion_cost
                        }
                    
                    self._pricing_cache = new_cache
                    self._last_fetch_time = current_time
                    logger.info(f"Fetched pricing for {len(self._pricing_cache)} models from OpenRouter")
            except Exception as e:
                logger.error(f"Failed to fetch pricing from OpenRouter: {e}")
                # Don't clear cache if fetch fails, keep old data

    def get_cost(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get prompt and completion cost for a specific model."""
        # Check for common variants (e.g., openai/gpt-4o vs gpt-4o)
        if model_id in self._pricing_cache:
            return self._pricing_cache[model_id]
        
        # Try to find a partial match if the exact ID isn't found
        for cached_id, costs in self._pricing_cache.items():
            if model_id in cached_id:
                return costs
                
        return None

    def calculate_request_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the USD cost of a request."""
        costs = self.get_cost(model_id)
        if not costs:
            return 0.0
        
        return (input_tokens * costs["prompt"]) + (output_tokens * costs["completion"])
