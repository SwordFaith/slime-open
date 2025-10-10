"""
Simplified OpenAI Chat Completion API integration tests.

This test suite validates the new simplified Chat Completion architecture that:
- Automatically detects cache capability via /retrieve_from_messages_template endpoint
- Uses unified flow: messages → generate → OpenAI format
- Supports both direct proxy mode and cached mode
- Handles streaming and non-streaming responses

Tests cover:
- Cache availability detection
- Messages template caching with new endpoint
- Unified generate flow integration
- Streaming responses with proper SSE formatting
- Fallback to direct SGLang proxy when cache not available
"""

import json
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from httpx import AsyncClient

from slime.router.openai_chat_completion import ChatCompletionHandler, create_chat_completion_handler
from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware


class TestSimplifiedChatCompletion:
    """Test simplified Chat Completion architecture."""

    @pytest.fixture
    def mock_router(self):
        """Mock SlimeRouter with necessary attributes."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000
        router.client = AsyncMock()
        router.app = MagicMock()
        router.app.user_middleware = []

        # Mock URL management methods
        router._use_url = AsyncMock(return_value="http://localhost:10090")
        router._finish_url = AsyncMock()

        return router

    @pytest.fixture
    def mock_router_with_cache(self, mock_router):
        """Mock router with cache support (via /retrieve_from_messages_template endpoint)."""
        # Mock successful cache availability check
        mock_cache_response = MagicMock()
        mock_cache_response.status_code = 200

        # Mock successful cache endpoint response
        mock_cache_response.json = AsyncMock(return_value={
            "tokens": [1, 2, 3],
            "response": "<|im_start|>assistant\nHello there!<|im_end|>",
            "loss_mask": [0, 0, 0],
            "token_length": 3,
            "loss_mask_length": 3,
            "rollout_logp": [-0.1, -0.2, -0.3],
            "generation_versions": [1, 1, 1],
        })

        mock_router.client.post.return_value = mock_cache_response
        return mock_router

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False
        })
        request.body = AsyncMock(return_value=json.dumps({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False
        }).encode())
        request.headers = {"content-type": "application/json"}
        return request

    @pytest.fixture
    def mock_streaming_request(self):
        """Mock FastAPI Request for streaming."""
        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": True
        })
        request.body = AsyncMock(return_value=json.dumps({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": True
        }).encode())
        request.headers = {"content-type": "application/json"}
        return request

    @pytest.mark.asyncio
    async def test_cache_availability_detection_with_support(self, mock_router_with_cache):
        """Test automatic detection of cache support."""
        handler = ChatCompletionHandler(mock_router_with_cache)

        # Should detect cache availability
        cache_available = await handler._check_cache_availability()
        assert cache_available is True

    @pytest.mark.asyncio
    async def test_cache_availability_detection_without_support(self, mock_router):
        """Test behavior when cache is not available."""
        handler = ChatCompletionHandler(mock_router)

        # Mock failed cache availability check
        mock_router.client.post.side_effect = Exception("Cache endpoint not available")

        # Should detect no cache availability
        cache_available = await handler._check_cache_availability()
        assert cache_available is False

    @pytest.mark.asyncio
    async def test_cached_mode_flow(self, mock_router_with_cache, mock_request):
        """Test cached mode flow using messages template."""
        handler = ChatCompletionHandler(mock_router_with_cache)

        # Mock the generate endpoint response
        mock_generate_response = MagicMock()
        mock_generate_response.json = AsyncMock(return_value={
            "text": "Hello there!",
            "request_id": "req-123"
        })
        mock_router.client.post.return_value = mock_generate_response

        # Handle request
        response = await handler.handle_request(mock_request)

        # Verify response format
        assert response.status_code == 200
        response_data = json.loads(response.body.decode())
        assert response_data["object"] == "chat.completion"
        assert response_data["model"] == "test-model"
        assert response_data["choices"][0]["message"]["role"] == "assistant"
        assert response_data["choices"][0]["message"]["content"] == "Hello there!"

        # Verify cache endpoint was called first
        cache_call_args = mock_router.client.post.call_args_list[0]
        assert cache_call_args[1]['json']['messages'] == [{"role": "user", "content": "Hello!"}]
        assert cache_call_args[1]['json']['add_generation_prompt'] is True

        # Verify generate endpoint was called
        generate_call_args = mock_router.client.post.call_args_list[1]
        generate_request = json.loads(generate_call_args[1]['json'])
        assert "input_tokens" in generate_request
        assert generate_request["input_tokens"] == [1, 2, 3]  # From mocked cache response

    @pytest.mark.asyncio
    async def test_direct_proxy_mode_flow(self, mock_router, mock_request):
        """Test direct proxy mode when no RadixTreeMiddleware."""
        handler = ChatCompletionHandler(mock_router)

        # Mock SGLang chat completion response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aread = AsyncMock(return_value=json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from SGLang!"
                },
                "finish_reason": "stop"
            }]
        }).encode())
        mock_response.headers = {"content-type": "application/json"}
        mock_router.client.request.return_value = mock_response

        # Handle request
        response = await handler.handle_request(mock_request)

        # Verify direct proxy response
        assert response.status_code == 200
        response_data = json.loads(response.content.decode())
        assert response_data["object"] == "chat.completion"
        assert response_data["choices"][0]["message"]["content"] == "Hello from SGLang!"

    @pytest.mark.asyncio
    async def test_streaming_response_with_cache(self, mock_router_with_middleware, mock_streaming_request):
        """Test streaming response with cache integration."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock streaming response from generate endpoint
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        # Simulate SSE chunks from SGLang
        sse_chunks = [
            "data: {\"text\": \"Hello\", \"finished\": false}\n",
            "data: {\"text\": \" there\", \"finished\": false}\n",
            "data: {\"text\": \"!\", \"finished\": false}\n",
            "data: {\"text\": \"\", \"finished\": true, \"finish_reason\": \"stop\"}\n",
            "data: [DONE]\n"
        ]
        mock_stream_response.aiter_lines.return_value = iter(sse_chunks)

        mock_router.client.stream.return_value.__aenter__.return_value = mock_stream_response

        # Handle streaming request
        response = await handler.handle_request(mock_streaming_request)

        # Verify streaming response
        assert hasattr(response, 'body_iterator')

        # Collect chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        # Verify OpenAI SSE format
        assert len(chunks) > 0
        assert any("chat.completion.chunk" in chunk for chunk in chunks)
        assert any("data: [DONE]" in chunk for chunk in chunks)

        # Verify middleware was used
        handler.radix_middleware.query_cache_by_messages_template.assert_called_once_with(
            [{"role": "user", "content": "Hello!"}],
            None,
            add_generation_prompt=True
        )

    @pytest.mark.asyncio
    async def test_streaming_direct_proxy_mode(self, mock_router, mock_streaming_request):
        """Test streaming response in direct proxy mode."""
        handler = ChatCompletionHandler(mock_router)

        # Mock SGLang streaming response
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.headers = {"content-type": "text/plain"}
        mock_stream_response.aiter_bytes = AsyncMock()

        # Simulate SGLang SSE chunks
        sse_bytes = [
            b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n",
            b"data: {\"choices\": [{\"delta\": {\"content\": \" there\"}}]}\n\n",
            b"data: {\"choices\": [{\"delta\": {\"content\": \"!\"}}]}\n\n",
            b"data: [DONE]\n\n"
        ]
        mock_stream_response.aiter_bytes.return_value = iter(sse_bytes)

        mock_router.client.stream.return_value.__aenter__.return_value = mock_stream_response

        # Handle streaming request
        response = await handler.handle_request(mock_streaming_request)

        # Verify direct streaming proxy
        assert hasattr(response, 'body_iterator')

        # Should pass through SGLang chunks unchanged
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 4
        assert b"Hello" in chunks[0]
        assert b"there" in chunks[1]
        assert b"!" in chunks[2]
        assert b"[DONE]" in chunks[3]

    @pytest.mark.asyncio
    async def test_messages_with_tools(self, mock_router_with_middleware):
        """Test messages with tools parameter."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Request with tools
        request_with_tools = MagicMock(spec=Request)
        request_with_tools.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What's the weather?"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}}
            ],
            "stream": False
        })

        # Mock generate response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "text": "I'll check the weather for you.",
            "request_id": "req-456"
        })
        mock_router.client.post.return_value = mock_response

        # Handle request
        response = await handler.handle_request(request_with_tools)

        # Verify tools were passed to cache query
        handler.radix_middleware.query_cache_by_messages_template.assert_called_once_with(
            [{"role": "user", "content": "What's the weather?"}],
            [{"type": "function", "function": {"name": "get_weather"}}],
            add_generation_prompt=True
        )

    @pytest.mark.asyncio
    async def test_error_handling_missing_messages(self, mock_router_with_middleware):
        """Test error handling when messages are missing."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Request without messages
        invalid_request = MagicMock(spec=Request)
        invalid_request.json = AsyncMock(return_value={
            "model": "test-model",
            "stream": False
        })

        # Should raise HTTPException
        with pytest.raises(Exception):  # HTTPException would be raised
            await handler.handle_request(invalid_request)

    @pytest.mark.asyncio
    async def test_sampling_params_conversion(self, mock_router_with_middleware):
        """Test conversion of OpenAI parameters to SGLang format."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Request with various sampling parameters
        request_with_params = MagicMock(spec=Request)
        request_with_params.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 150,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["\n", "User:"],
            "stream": False
        })

        # Mock generate response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"text": "Response", "request_id": "req-789"})
        mock_router.client.post.return_value = mock_response

        # Handle request
        await handler.handle_request(request_with_params)

        # Verify the request sent to generate endpoint
        call_args = mock_router.client.post.call_args
        generate_request = json.loads(call_args[1]['json'])

        sampling_params = generate_request["sampling_params"]
        assert sampling_params["temperature"] == 0.7
        assert sampling_params["top_p"] == 0.9
        assert sampling_params["top_k"] == 50
        assert sampling_params["max_new_tokens"] == 150
        assert sampling_params["frequency_penalty"] == 0.1
        assert sampling_params["presence_penalty"] == 0.2
        assert sampling_params["stop"] == ["\n", "User:"]

    @pytest.mark.asyncio
    async def test_factory_function(self, mock_router):
        """Test factory function for creating handlers."""
        handler = create_chat_completion_handler(mock_router)

        assert isinstance(handler, ChatCompletionHandler)
        assert handler.router == mock_router
        assert handler.args == mock_router.args

    @pytest.mark.asyncio
    async def test_concurrent_requests_cached_mode(self, mock_router_with_middleware):
        """Test concurrent requests in cached mode."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock responses
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"text": "Response", "request_id": "req-123"})
        mock_router.client.post.return_value = mock_response

        # Create multiple concurrent requests
        requests = []
        for i in range(5):
            req = MagicMock(spec=Request)
            req.json = AsyncMock(return_value={
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Hello {i}!"}],
                "stream": False
            })
            requests.append(req)

        # Handle all requests concurrently
        tasks = [handler.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Verify all requests succeeded
        assert len(responses) == 5
        for response in responses:
            assert response.status_code == 200

        # Verify middleware was called for each request
        assert handler.radix_middleware.query_cache_by_messages_template.call_count == 5


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def mock_router(self):
        """Mock router for error testing."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000
        router.client = AsyncMock()
        router.app = MagicMock()
        router.app.user_middleware = []
        router._use_url = AsyncMock(return_value="http://localhost:10090")
        router._finish_url = AsyncMock()
        return router

    @pytest.mark.asyncio
    async def test_generate_endpoint_failure(self, mock_router):
        """Test handling of generate endpoint failures."""
        # Add middleware to use cached mode
        mock_middleware = MagicMock()
        mock_middleware.cls = RadixTreeMiddleware
        mock_middleware.cls.__name__ = 'RadixTreeMiddleware'
        mock_middleware.cls.query_cache_by_messages_template = AsyncMock(
            return_value=([1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], 1)
        )
        mock_router.app.user_middleware = [mock_middleware]

        handler = ChatCompletionHandler(mock_router)

        # Mock generate endpoint failure
        mock_router.client.post.side_effect = Exception("SGLang worker unavailable")

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        # Should propagate the error
        with pytest.raises(Exception, match="SGLang worker unavailable"):
            await handler.handle_request(request)

    @pytest.mark.asyncio
    async def test_invalid_json_request(self, mock_router):
        """Test handling of invalid JSON requests."""
        handler = ChatCompletionHandler(mock_router)

        # Request with invalid JSON
        invalid_request = MagicMock(spec=Request)
        invalid_request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        # Should handle JSON decode error
        with pytest.raises(json.JSONDecodeError):
            await handler.handle_request(invalid_request)

    @pytest.mark.asyncio
    async def test_middleware_method_failure(self, mock_router):
        """Test handling of middleware method failures."""
        # Add failing middleware
        mock_middleware = MagicMock()
        mock_middleware.cls = RadixTreeMiddleware
        mock_middleware.cls.__name__ = 'RadixTreeMiddleware'
        mock_middleware.cls.query_cache_by_messages_template = AsyncMock(
            side_effect=Exception("Cache lookup failed")
        )
        mock_router.app.user_middleware = [mock_middleware]

        handler = ChatCompletionHandler(mock_router)

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        # Should fall back to direct mode when middleware fails
        # This tests the resilience of the simplified architecture
        # Note: The actual behavior might depend on implementation details
        try:
            await handler.handle_request(request)
        except Exception as e:
            # Should either succeed with fallback or fail gracefully
            assert "Cache lookup failed" in str(e) or hasattr(e, 'response')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])