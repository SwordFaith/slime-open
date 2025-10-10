"""
Streaming response and messages cache integration tests.

This test suite specifically validates:
- Streaming responses with proper SSE formatting
- Messages template caching with query_cache_by_messages_template
- Cache behavior across multiple turns
- Concurrent streaming with cache access
- Integration with RadixTreeMiddleware

These tests validate the core functionality that was missing in the original
Chat Completion implementation.
"""

import json
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import Request

from slime.router.openai_chat_completion import ChatCompletionHandler
from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware


class TestStreamingResponses:
    """Test streaming response functionality."""

    @pytest.fixture
    def mock_router_with_middleware(self):
        """Mock router with RadixTreeMiddleware."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000
        router.client = AsyncMock()
        router.app = MagicMock()
        router.app.user_middleware = []

        # Mock middleware
        mock_middleware = MagicMock()
        mock_middleware.cls = RadixTreeMiddleware
        mock_middleware.cls.__name__ = 'RadixTreeMiddleware'
        mock_middleware.cls.query_cache_by_messages_template = AsyncMock(
            return_value=([1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], 1)
        )
        router.app.user_middleware = [mock_middleware]

        return router

    @pytest.fixture
    def mock_streaming_request(self):
        """Mock streaming request."""
        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True,
            "max_tokens": 50
        })
        return request

    @pytest.mark.asyncio
    async def test_streaming_sse_format(self, mock_router_with_middleware, mock_streaming_request):
        """Test proper Server-Sent Events formatting for streaming."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock streaming response from generate endpoint
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        # Simulate SGLang SSE chunks
        sglang_chunks = [
            "data: {\"text\": \"Once\", \"finished\": false}\n",
            "data: {\"text\": \" upon\", \"finished\": false}\n",
            "data: {\"text\": \" a\", \"finished\": false}\n",
            "data: {\"text\": \" time\", \"finished\": false}\n",
            "data: {\"text\": \"...\", \"finished\": false}\n",
            "data: {\"text\": \"\", \"finished\": true, \"finish_reason\": \"stop\"}\n",
            "data: [DONE]\n"
        ]
        mock_stream_response.aiter_lines.return_value = iter(sglang_chunks)

        mock_router_with_middleware.client.stream.return_value.__aenter__.return_value = mock_stream_response

        # Handle streaming request
        response = await handler.handle_request(mock_streaming_request)

        # Verify response is streaming
        assert hasattr(response, 'body_iterator')

        # Collect and verify OpenAI-formatted chunks
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk.decode('utf-8'))
            else:
                chunks.append(str(chunk))

        # Should have content chunks + finish chunk + [DONE]
        assert len(chunks) >= 2

        # Verify OpenAI format
        content_chunks = [c for c in chunks if "chat.completion.chunk" in c]
        assert len(content_chunks) >= 1

        # Parse a content chunk
        first_chunk_data = json.loads(content_chunks[0].split("data: ")[1])
        assert first_chunk_data["object"] == "chat.completion.chunk"
        assert first_chunk_data["model"] == "test-model"
        assert len(first_chunk_data["choices"]) == 1
        assert "delta" in first_chunk_data["choices"][0]
        assert "content" in first_chunk_data["choices"][0]["delta"]

        # Verify final chunk
        done_chunks = [c for c in chunks if "[DONE]" in c]
        assert len(done_chunks) == 1

    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(self, mock_router_with_middleware, mock_streaming_request):
        """Test streaming with empty response chunks."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock streaming with empty chunks
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        # Simulate response with empty chunks
        sglang_chunks = [
            "data: {\"text\": \"\", \"finished\": false}\n",  # Empty text
            "data: {\"text\": \"Hello\", \"finished\": false}\n",
            "data: {\"text\": \"\", \"finished\": false}\n",  # Another empty
            "data: {\"text\": \"!\", \"finished\": false}\n",
            "data: {\"text\": \"\", \"finished\": true, \"finish_reason\": \"stop\"}\n",
            "data: [DONE]\n"
        ]
        mock_stream_response.aiter_lines.return_value = iter(sglang_chunks)

        mock_router_with_middleware.client.stream.return_value.__aenter__.return_value = mock_stream_response

        # Handle streaming request
        response = await handler.handle_request(mock_streaming_request)

        # Collect chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        # Should filter out empty content chunks but keep structure
        content_chunks = [c for c in chunks if "Hello" in c or "!" in c]
        assert len(content_chunks) == 2

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, mock_router_with_middleware, mock_streaming_request):
        """Test error handling during streaming."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock streaming that fails mid-stream
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        async def failing_chunks():
            yield "data: {\"text\": \"Hello\", \"finished\": false}\n"
            raise Exception("Stream interrupted")

        mock_stream_response.aiter_lines.return_value = failing_chunks()

        mock_router_with_middleware.client.stream.return_value.__aenter__.return_value = mock_stream_response

        # Should handle stream errors gracefully
        with pytest.raises(Exception, match="Stream interrupted"):
            response = await handler.handle_request(mock_streaming_request)
            async for chunk in response.body_iterator:
                pass  # Iterate to trigger the error


class TestMessagesTemplateCache:
    """Test messages template caching functionality."""

    @pytest.fixture
    def mock_router_with_middleware(self):
        """Mock router with configurable middleware."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000
        router.client = AsyncMock()
        router.app = MagicMock()
        router.app.user_middleware = []

        # Mock middleware
        mock_middleware = MagicMock()
        mock_middleware.cls = RadixTreeMiddleware
        mock_middleware.cls.__name__ = 'RadixTreeMiddleware'
        mock_middleware.cls.query_cache_by_messages_template = AsyncMock()
        router.app.user_middleware = [mock_middleware]

        return router, mock_middleware

    @pytest.mark.asyncio
    async def test_simple_messages_cache_query(self, mock_router_with_middleware):
        """Test basic messages cache query."""
        router, mock_middleware = mock_router_with_middleware
        handler = ChatCompletionHandler(router)

        # Mock cache response
        mock_middleware.cls.query_cache_by_messages_template.return_value = (
            [1, 2, 3, 4, 5],  # tokens
            [-0.1, -0.2, -0.1, -0.3, -0.2],  # logp
            [0, 0, 0, 0, 0],  # loss_mask
            1  # weight_version
        )

        # Mock generate response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"text": "Hello there!", "request_id": "req-123"})
        router.client.post.return_value = mock_response

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False
        })

        # Handle request
        await handler.handle_request(request)

        # Verify cache was queried with correct parameters
        mock_middleware.cls.query_cache_by_messages_template.assert_called_once_with(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            None,  # no tools
            add_generation_prompt=True
        )

    @pytest.mark.asyncio
    async def test_messages_with_tools_cache_query(self, mock_router_with_middleware):
        """Test messages with tools in cache query."""
        router, mock_middleware = mock_router_with_middleware
        handler = ChatCompletionHandler(router)

        # Mock cache response
        mock_middleware.cls.query_cache_by_messages_template.return_value = (
            [10, 11, 12], [-0.1, -0.2, -0.3], [0, 0, 0], 1
        )

        # Mock generate response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"text": "I'll help with that!", "request_id": "req-456"})
        router.client.post.return_value = mock_response

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What's the weather in Boston?"}
            ],
            "tools": tools,
            "stream": False
        })

        # Handle request
        await handler.handle_request(request)

        # Verify cache was queried with tools
        mock_middleware.cls.query_cache_by_messages_template.assert_called_once_with(
            [{"role": "user", "content": "What's the weather in Boston?"}],
            tools,
            add_generation_prompt=True
        )

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_cache(self, mock_router_with_middleware):
        """Test cache behavior across multiple conversation turns."""
        router, mock_middleware = mock_router_with_middleware
        handler = ChatCompletionHandler(router)

        # Mock progressive cache responses
        cache_responses = [
            # First turn - cache miss
            ([], [], [], 1),
            # Second turn - partial cache hit
            ([1, 2, 3, 4, 5, 6, 7, 8], [-0.1] * 8, [0] * 8, 1),
            # Third turn - longer cache hit
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [-0.1] * 11, [0] * 11, 1)
        ]
        mock_middleware.cls.query_cache_by_messages_template.side_effect = cache_responses

        # Mock generate responses
        generate_responses = [
            {"text": "Hello! How can I help you today?", "request_id": "req-1"},
            {"text": "I'm doing well, thank you!", "request_id": "req-2"},
            {"text": "That sounds great!", "request_id": "req-3"}
        ]

        mock_response = MagicMock()
        mock_response.json = AsyncMock(side_effect=generate_responses)
        router.client.post.return_value = mock_response

        # First turn
        request1 = MagicMock(spec=Request)
        request1.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })
        await handler.handle_request(request1)

        # Second turn
        request2 = MagicMock(spec=Request)
        request2.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
                {"role": "user", "content": "How are you?"}
            ],
            "stream": False
        })
        await handler.handle_request(request2)

        # Third turn
        request3 = MagicMock(spec=Request)
        request3.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "user", "content": "Great to hear!"}
            ],
            "stream": False
        })
        await handler.handle_request(request3)

        # Verify cache was queried with progressively longer conversations
        assert mock_middleware.cls.query_cache_by_messages_template.call_count == 3

        # Check the third call (longest conversation)
        third_call_args = mock_middleware.cls.query_cache_by_messages_template.call_args_list[2]
        third_messages = third_call_args[0][0]
        assert len(third_messages) == 5  # Full conversation history

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, mock_router_with_middleware):
        """Test concurrent access to messages cache."""
        router, mock_middleware = mock_router_with_middleware
        handler = ChatCompletionHandler(router)

        # Mock cache response
        mock_middleware.cls.query_cache_by_messages_template.return_value = (
            [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], 1
        )

        # Mock generate response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"text": "Response", "request_id": "req-123"})
        router.client.post.return_value = mock_response

        # Create multiple concurrent requests with different messages
        requests = []
        for i in range(5):
            req = MagicMock(spec=Request)
            req.json = AsyncMock(return_value={
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Message {i}"}],
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

        # Verify cache was accessed concurrently
        assert mock_middleware.cls.query_cache_by_messages_template.call_count == 5

        # Verify different messages were queried
        call_args_list = mock_middleware.cls.query_cache_by_messages_template.call_args_list
        queried_messages = [call[0][0] for call in call_args_list]
        assert len(set(str(msg[0]["content"]) for msg in queried_messages)) == 5  # All unique

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, mock_router_with_middleware):
        """Test error handling in cache operations."""
        router, mock_middleware = mock_router_with_middleware
        handler = ChatCompletionHandler(router)

        # Mock cache failure
        mock_middleware.cls.query_cache_by_messages_template.side_effect = Exception("Cache service unavailable")

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        # Should handle cache errors gracefully
        # Note: The actual behavior depends on implementation
        # It might fall back to direct mode or fail gracefully
        try:
            await handler.handle_request(request)
        except Exception as e:
            # Should either succeed with fallback or fail with clear error
            assert "Cache service unavailable" in str(e)


class TestStreamingWithCacheIntegration:
    """Test integration of streaming responses with cache functionality."""

    @pytest.fixture
    def mock_router_with_middleware(self):
        """Mock router for streaming+cache tests."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000
        router.client = AsyncMock()
        router.app = MagicMock()
        router.app.user_middleware = []

        # Mock middleware
        mock_middleware = MagicMock()
        mock_middleware.cls = RadixTreeMiddleware
        mock_middleware.cls.__name__ = 'RadixTreeMiddleware'
        mock_middleware.cls.query_cache_by_messages_template = AsyncMock(
            return_value=([1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], 1)
        )
        router.app.user_middleware = [mock_middleware]

        return router

    @pytest.mark.asyncio
    async def test_streaming_with_cache_hit(self, mock_router_with_middleware):
        """Test streaming when cache hit occurs."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock streaming response
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        sglang_chunks = [
            "data: {\"text\": \"Cached\", \"finished\": false}\n",
            "data: {\"text\": \" response\", \"finished\": false}\n",
            "data: {\"text\": \"!\", \"finished\": false}\n",
            "data: {\"text\": \"\", \"finished\": true, \"finish_reason\": \"stop\"}\n",
            "data: [DONE]\n"
        ]
        mock_stream_response.aiter_lines.return_value = iter(sglang_chunks)

        mock_router_with_middleware.client.stream.return_value.__aenter__.return_value = mock_stream_response

        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Continue"}
            ],
            "stream": True
        })

        # Handle streaming request
        response = await handler.handle_request(request)

        # Verify cache was queried for multi-turn conversation
        mock_middleware = mock_router_with_middleware.app.user_middleware[0]
        mock_middleware.cls.query_cache_by_messages_template.assert_called_once_with(
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Continue"}
            ],
            None,
            add_generation_prompt=True
        )

        # Verify streaming response format
        assert hasattr(response, 'body_iterator')

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        # Should have properly formatted OpenAI streaming chunks
        assert any("Cached" in chunk for chunk in chunks)
        assert any("chat.completion.chunk" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_concurrent_streaming_with_cache(self, mock_router_with_middleware):
        """Test multiple concurrent streaming requests with cache."""
        handler = ChatCompletionHandler(mock_router_with_middleware)

        # Mock middleware to return different cache results for different requests
        def cache_side_effect(messages, tools, add_generation_prompt):
            # Simulate different cache hits based on message content
            if "story" in str(messages):
                return ([1, 2], [-0.1, -0.2], [0, 0], 1)
            elif "joke" in str(messages):
                return ([3, 4, 5], [-0.1, -0.1, -0.1], [0, 0, 0], 1)
            else:
                return ([], [], [], 1)

        mock_middleware = mock_router_with_middleware.app.user_middleware[0]
        mock_middleware.cls.query_cache_by_messages_template.side_effect = cache_side_effect

        # Mock streaming responses
        async def mock_stream_response():
            chunks = [
                "data: {\"text\": \"Response\", \"finished\": false}\n",
                "data: {\"text\": \"!\", \"finished\": false}\n",
                "data: {\"text\": \"\", \"finished\": true, \"finish_reason\": \"stop\"}\n",
                "data: [DONE]\n"
            ]
            for chunk in chunks:
                yield chunk

        # Create multiple concurrent streaming requests
        requests = []
        for i, content in enumerate(["Tell me a story", "Tell me a joke", "How are you?"]):
            req = MagicMock(spec=Request)
            req.json = AsyncMock(return_value={
                "model": "test-model",
                "messages": [{"role": "user", "content": content}],
                "stream": True
            })
            requests.append(req)

        # Mock the streaming endpoint
        mock_stream_obj = MagicMock()
        mock_stream_obj.aiter_lines.return_value = mock_stream_response()
        mock_router_with_middleware.client.stream.return_value.__aenter__.return_value = mock_stream_obj

        # Handle all streaming requests concurrently
        tasks = [handler.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Verify all streaming responses were created
        assert len(responses) == 3
        for response in responses:
            assert hasattr(response, 'body_iterator')

        # Verify cache was accessed for each request
        assert mock_middleware.cls.query_cache_by_messages_template.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])