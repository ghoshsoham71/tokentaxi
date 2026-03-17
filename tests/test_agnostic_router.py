# tests/test_agnostic_router.py
import asyncio
import pytest
from typing import Any, AsyncIterator
from tokentaxi.router import LLMRouter
from tokentaxi.models import RouterRequest
from tokentaxi.config import RouterConfig

class MockClient:
    def __init__(self):
        # UniversalAdapter does hasattr(client, "chat") -> client.chat.completions.create(...)
        class MockCompletions:
            async def create(self, *args, **kwargs):
                class MockResponse:
                    class Choice:
                        class Message:
                            content = "Hello from mock"
                        message = Message()
                    choices = [Choice()]
                    class Usage:
                        prompt_tokens = 10
                        completion_tokens = 20
                    usage = Usage()
                return MockResponse()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        self.chat = MockChat()

@pytest.mark.asyncio
async def test_agnostic_registration_and_cost():
    config = RouterConfig(providers=[])
    router = LLMRouter(config)
    await router._ensure_initialized()
    
    mock_client = MockClient()
    # Mock OpenRouter fetch to avoid network call in tests
    router.pricing._pricing_cache = {
        "openai/gpt-4o": {"prompt": 0.000005, "completion": 0.000015}
    }
    router.pricing._last_fetch_time = asyncio.get_event_loop().time()
    
    # Register via BYOC
    router.register(
        name="mock-openai",
        client=mock_client,
        model="openai/gpt-4o",
        rpm=100,
        tpm=10000
    )
    
    request = RouterRequest(
        messages=[{"role": "user", "content": "hi"}],
        optimization_strategy="cost"
    )
    
    response = await router.chat(request)
    
    assert response.provider == "mock-openai"
    assert response.cost_usd > 0
    assert response.cost_usd == (10 * 0.000005) + (20 * 0.000015)
    print(f"Verified cost: {response.cost_usd}")

if __name__ == "__main__":
    asyncio.run(test_agnostic_registration_and_cost())
