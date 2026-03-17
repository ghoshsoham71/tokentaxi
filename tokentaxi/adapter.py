# tokentaxi/adapter.py
"""
UniversalAdapter — a provider-agnostic wrapper for any LLM client.

This class uses dynamic feature detection to interact with different SDKs
(OpenAI-compatible, Anthropic, Gemini, etc.) through a uniform interface.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class AsyncChatClient(Protocol):
    """Protocol for clients that support async chat completions."""
    async def chat(self, *args: Any, **kwargs: Any) -> Any: ...

class BaseAdapter(ABC):
    """Abstract base for all internal adapter implementations."""
    
    @abstractmethod
    async def chat(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        """Send a non-streaming chat request."""

    @abstractmethod
    async def stream(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat request."""

class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible clients (vLLM, Groq, Ollama)."""
    
    async def chat(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return content, input_tokens, output_tokens

    async def stream(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content"):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic clients."""
    
    async def chat(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        system = None
        filtered_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content")
            else:
                filtered_messages.append(m)
        
        call_params = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            call_params["system"] = system
            
        response = await client.messages.create(**call_params)
        content = response.content[0].text if response.content else ""
        return content, response.usage.input_tokens, response.usage.output_tokens

    async def stream(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        system = None
        filtered_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content")
            else:
                filtered_messages.append(m)
        
        call_params = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            call_params["system"] = system
            
        async with client.messages.stream(**call_params) as stream:
            async for text in stream.text_stream:
                yield text

class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini clients."""
    
    async def chat(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        from google.generativeai.types import GenerationConfig, content_types
        
        history = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            history.append(content_types.to_content({"role": role, "parts": [m["content"]]}))
            
        prompt = history.pop() # Last one is prompt
        
        config = GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        chat = client.start_chat(history=history)
        response = await chat.send_message_async(prompt, generation_config=config)
        
        content = response.text
        usage = response.usage_metadata
        return content, usage.prompt_token_count, usage.candidates_token_count

    async def stream(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        from google.generativeai.types import GenerationConfig, content_types
        
        history = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            history.append(content_types.to_content({"role": role, "parts": [m["content"]]}))
            
        prompt = history.pop()
        
        config = GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        chat = client.start_chat(history=history)
        response = await chat.send_message_async(prompt, generation_config=config, stream=True)
        async for chunk in response:
            yield chunk.text

class GoogleGenAIAdapter(BaseAdapter):
    """Adapter for the newer google-genai (v1) SDK."""
    
    async def chat(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        from google.genai import types
        
        # Convert messages to google-genai format
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
            
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Use the async client (aio) if available, otherwise fallback to sync models
        # genai.Client has an 'aio' property for async operations
        if hasattr(client, "aio"):
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        else:
            # Fallback for sync client (blocks)
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        
        content = response.text or ""
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        candidate_tokens = response.usage_metadata.candidates_token_count or 0
        
        return content, prompt_tokens, candidate_tokens

    async def stream(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        from google.genai import types
        
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
            
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        if hasattr(client, "aio"):
            async for chunk in await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    yield chunk.text
        else:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    yield chunk.text

class MistralAdapter(BaseAdapter):
    """Adapter for Mistral AI native SDK."""
    async def chat(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> tuple[str, int, int]:
        response = await client.chat.complete(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    async def stream(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> AsyncIterator[str]:
        async for chunk in await client.chat.stream(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs):
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content

class CohereAdapter(BaseAdapter):
    """Adapter for Cohere SDK (v2)."""
    async def chat(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> tuple[str, int, int]:
        # Cohere v2 uses 'chat' with simple message format
        response = client.chat(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return response.message.content[0].text, response.usage.tokens.input_tokens, response.usage.tokens.output_tokens

    async def stream(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> AsyncIterator[str]:
        for event in client.chat_stream(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs):
            if event.type == "content-delta":
                yield event.delta.message.content.text

class BedrockAdapter(BaseAdapter):
    """Adapter for AWS Bedrock (Boto3)."""
    async def chat(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> tuple[str, int, int]:
        import json
        # Simplified Bedrock Converse API
        response = client.converse(modelId=model, messages=messages, inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}, **kwargs)
        text = response['output']['message']['content'][0]['text']
        usage = response['usage']
        return text, usage['inputTokens'], usage['outputTokens']

    async def stream(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> AsyncIterator[str]:
        response = client.converse_stream(modelId=model, messages=messages, inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}, **kwargs)
        for event in response['stream']:
            if 'contentBlockDelta' in event:
                yield event['contentBlockDelta']['delta']['text']

class CustomAdapter(BaseAdapter):
    """Adapter for arbitrary callables (Functional BYOC)."""
    async def chat(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> tuple[str, int, int]:
        # client is expected to be a callable for custom implementations
        if asyncio.iscoroutinefunction(client):
            return await client(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False, **kwargs)
        return client(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False, **kwargs)

    async def stream(self, client: Any, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, **kwargs: Any) -> AsyncIterator[str]:
        # client is expected to return an async iterator
        if asyncio.iscoroutinefunction(client):
            stream = await client(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True, **kwargs)
        else:
            stream = client(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True, **kwargs)
        async for chunk in stream:
            yield chunk

class UniversalAdapter:
    """Detects and delegates to the appropriate specialized adapter."""
    
    def __init__(self, client: Any):
        self._client = client
        self._impl = self._get_adapter(client)
        
    def _get_adapter(self, client: Any) -> BaseAdapter:
        client_type = str(type(client)).lower()
        if "openai" in client_type or hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return OpenAIAdapter()
        if "anthropic" in client_type:
            return AnthropicAdapter()
        if "google.generativeai" in client_type or "generativemodel" in client_type:
            return GeminiAdapter()
        if "google.genai" in client_type or "genai.client" in client_type:
            return GoogleGenAIAdapter()
        if "mistral" in client_type:
            return MistralAdapter()
        if "cohere" in client_type:
            return CohereAdapter()
        if "bedrock" in client_type:
            return BedrockAdapter()
        if callable(client):
            return CustomAdapter()
        
        if hasattr(client, "chat") and not hasattr(client, "models"):
            return OpenAIAdapter()
            
        if client is None:
            # Placeholder for discovered providers that have no local client instance
            return CustomAdapter()

        raise ValueError(f"Unsupported client type: {type(client)}. "
                         "Client must be OpenAI-compatible, Anthropic, Gemini, or Google GenAI.")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        return await self._impl.chat(self._client, model, messages, max_tokens, temperature, **kwargs)

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        return self._impl.stream(self._client, model, messages, max_tokens, temperature, **kwargs)
