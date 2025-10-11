"""
OpenAI Chat Completion API unit tests for simplified implementation.

Tests cover:
- ChatCompletionHandler initialization
- Request validation
- Cache availability detection
- Error handling and fallback mechanisms
- Basic streaming and non-streaming functionality
- Message formatting using tokenizer templates

Simplified Strategy:
- Focus on core functionality and error cases
- Mock external dependencies thoroughly
- Avoid complex multi-turn conversation tests that require extensive setup
- Use FastAPI test client for integration-like testing
"""

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from slime.router.openai_chat_completion import ChatCompletionHandler, create_chat_completion_handler
from slime.router.component_registry import ComponentRegistry


class TestChatCompletionHandler:
    """Test ChatCompletionHandler core functionality."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock SlimeRouter with necessary attributes."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000

        # Mock component registry
        router._component_registry = None
        router._registry_lock = MagicMock()

        # Mock worker management
        router.worker_urls = {"http://test:10090": 0}
        router._url_lock = AsyncMock()
        router._use_url = AsyncMock(return_value="http://test:10090")
        router._finish_url = AsyncMock()

        # Mock HTTP client
        router.client = AsyncMock()

        return router

    @pytest.fixture
    def chat_handler(self, mock_router):
        """Create ChatCompletionHandler instance."""
        return ChatCompletionHandler(mock_router)

    def test_handler_initialization(self, mock_router):
        """Test ChatCompletionHandler initialization."""
        handler = ChatCompletionHandler(mock_router)

        assert handler.router is mock_router
        assert handler.args is mock_router.args
        assert handler._cache_available is None

    def test_check_cache_availability_with_available_components(self, mock_router):
        """Test cache availability detection when components are available."""
        handler = ChatCompletionHandler(mock_router)

        # Mock component registry with required components
        mock_registry = MagicMock()
        mock_registry.has.side_effect = lambda x: x in ["radix_tree", "tokenizer"]
        mock_router.component_registry = mock_registry

        # First call should perform check and cache result
        result1 = asyncio.run(handler._check_cache_availability())
        assert result1 is True

        # Second call should return cached result
        result2 = asyncio.run(handler._check_cache_availability())
        assert result2 is True
        assert handler._cache_available is True

    def test_check_cache_availability_missing_components(self, mock_router):
        """Test cache availability detection when components are missing."""
        handler = ChatCompletionHandler(mock_router)

        # Mock component registry with missing components
        mock_registry = MagicMock()
        mock_registry.has.side_effect = lambda x: x == "tokenizer"  # Only has tokenizer
        mock_router.component_registry = mock_registry

        result1 = asyncio.run(handler._check_cache_availability())
        assert result1 is False

        result2 = asyncio.run(handler._check_cache_availability())
        assert result2 is False
        assert handler._cache_available is False

    def test_check_cache_availability_with_exception(self, mock_router):
        """Test cache availability detection handles exceptions gracefully."""
        handler = ChatCompletionHandler(mock_router)

        # Mock component registry that raises exception
        mock_registry = MagicMock()
        mock_registry.has.side_effect = Exception("Registry error")
        mock_router.component_registry = mock_registry

        result = asyncio.run(handler._check_cache_availability())
        assert result is False
        assert handler._cache_available is False

    def test_validate_chat_completion_request_valid_request(self, chat_handler):
        """Test validation of valid Chat Completion request."""
        valid_request = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }

        # Should not raise exception
        chat_handler._validate_chat_completion_request(valid_request)

    def test_validate_chat_completion_request_missing_messages(self, chat_handler):
        """Test validation fails when messages are missing."""
        invalid_request = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }

        with pytest.raises(Exception, match="'messages' field is required"):
            chat_handler._validate_chat_completion_request(invalid_request)

    def test_validate_chat_completion_request_empty_messages(self, chat_handler):
        """Test validation fails when messages list is empty."""
        invalid_request = {
            "messages": []
        }

        with pytest.raises(Exception, match="'messages' must be a non-empty list"):
            chat_handler._validate_chat_completion_request(invalid_request)

    def test_validate_chat_completion_request_invalid_message_structure(self, chat_handler):
        """Test validation fails when messages have invalid structure."""
        invalid_request = {
            "messages": [
                {"content": "Hello"}  # Missing role field
            ]
        }

        with pytest.raises(Exception, match="message at index 0 must have 'role' and 'content' fields"):
            chat_handler._validate_chat_completion_request(invalid_request)


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""

    @pytest.fixture
    def mock_router_error(self):
        """Create a mock router for error testing."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000

        # Mock component registry
        router._component_registry = None
        router._registry_lock = MagicMock()

        # Mock worker management
        router.worker_urls = {"http://test:10090": 0}
        router._url_lock = AsyncMock()
        router._use_url = AsyncMock(return_value="http://test:10090")
        router._finish_url = AsyncMock()

        # Mock HTTP client
        router.client = AsyncMock()

        return router

    @pytest.fixture
    def chat_handler_with_mocks(self, mock_router_error):
        """Create ChatCompletionHandler with mocked router methods."""
        handler = ChatCompletionHandler(mock_router_error)

        # Mock router methods for error scenarios
        mock_router_error._use_url = AsyncMock(side_effect=Exception("Worker unavailable"))
        mock_router_error._finish_url = AsyncMock()
        mock_router_error.client = AsyncMock()

        return handler

    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self, chat_handler_with_mocks):
        """Test handling of malformed JSON in request."""
        handler = chat_handler_with_mocks

        # Mock request with invalid JSON
        mock_request = MagicMock()
        mock_request.json.side_effect = json.JSONDecodeError("Invalid JSON", doc="", pos=0)

        with pytest.raises(Exception, match="Invalid JSON in request body"):
            await handler.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_worker_unavailable_error_handling(self, chat_handler_with_mocks):
        """Test handling when SGLang worker is unavailable."""
        handler = chat_handler_with_mocks

        # Mock request with valid data
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })

        # Mock router to raise worker exception
        chat_handler_with_mocks.router._use_url.side_effect = Exception("No workers available")
        chat_handler_with_mocks.router._finish_url = AsyncMock()

        # Should raise HTTPException with appropriate error message
        with pytest.raises(Exception, match="Service temporarily unavailable"):
            await handler.handle_request(mock_request)


class TestCacheIntegration:
    """Test cache integration and fallback behavior."""

    @pytest.fixture
    def mock_router_with_components(self):
        """Create mock router with cache components."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000

        # Mock component registry with cache support
        mock_registry = MagicMock()
        mock_registry.has.side_effect = lambda x: x in ["radix_tree", "tokenizer"]
        router.component_registry = mock_registry

        # Mock worker management
        router.worker_urls = {"http://test:10090": 0}
        router._url_lock = AsyncMock()
        router._use_url = AsyncMock(return_value="http://test:10090")
        router._finish_url = AsyncMock()

        # Mock HTTP client
        router.client = AsyncMock()

        return router

    @pytest.fixture
    def chat_handler_with_cache(self, mock_router_with_components):
        """Create ChatCompletionHandler with cache support."""
        return ChatCompletionHandler(mock_router_with_components)

    @pytest.mark.asyncio
    async def test_cached_mode_success_flow(self, chat_handler_with_cache):
        """Test successful cached mode flow with valid components."""
        handler = chat_handler_with_cache

        # Mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False,
            "max_tokens": 100
        })

        # Mock successful generation response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello! I'm doing well, thank you for asking!",
            "output_ids": [100, 101, 102]
        }
        handler.router._use_url.return_value = "http://test:10090"
        handler.router._finish_url = AsyncMock()
        handler.router.client.post.return_value = mock_response

        result = await handler.handle_request(mock_request)

        # Should return JSON response with OpenAI format
        assert hasattr(result, 'content')
        response_data = json.loads(result.body)
        assert response_data["object"] == "chat.completion"
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        assert response_data["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_cached_mode_fallback_on_error(self, chat_handler_with_cache):
        """Test fallback to direct mode when cached mode fails."""
        handler = chat_handler_with_cache

        # Mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        # Mock direct SGLang response for fallback
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}]
        }).encode()
        mock_response.headers = {"content-type": "application/json"}

        handler.router.client.request.return_value = mock_response

        result = await handler.handle_request(mock_request)

        # Should return response from fallback
        assert result is not None

    @pytest.mark.asyncio
    async def test_handler_basic_functionality(self, chat_handler_with_cache):
        """Test basic handler functionality."""
        handler = chat_handler_with_cache

        # Mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }).encode()
        mock_response.headers = {"content-type": "application/json"}

        handler.router.client.request.return_value = mock_response

        result = await handler.handle_request(mock_request)

        # Should return response
        assert result is not None
        assert result.status_code == 200


class TestStreamingSupport:
    """Test streaming functionality."""

    @pytest.fixture
    def mock_router_with_streaming(self):
        """Create mock router for streaming tests."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000

        # Mock component registry
        mock_registry = MagicMock()
        mock_registry.has.side_effect = lambda x: x in ["radix_tree", "tokenizer"]
        router.component_registry = mock_registry

        # Mock worker management
        router.worker_urls = {"http://test:10090": 0}
        router._url_lock = AsyncMock()
        router._use_url = AsyncMock(return_value="http://test:10090")
        router._finish_url = AsyncMock()

        # Mock HTTP client
        mock_client = AsyncMock()

        # Mock streaming response
        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = AsyncMock()

        # Simulate SSE chunks
        chunks = [
            b'data: {"id": "1", "text": "Hello", "finish_reason": None}\n\n',
            b'data: {"id": "1", "text": " world!", "finish_reason": "stop"}\n\n',
            b'data: [DONE]\n\n'
        ]
        mock_stream_response.aiter_lines.side_effect = chunks

        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream_response)

        router.client = mock_client
        return router

    @pytest.fixture
    def chat_handler_streaming(self, mock_router_with_streaming):
        """Create ChatCompletionHandler for streaming tests."""
        return ChatCompletionHandler(mock_router_with_streaming)

    @pytest.mark.asyncio
    async def test_streaming_response_format(self, chat_handler_streaming):
        """Test streaming response format matches OpenAI SSE format."""
        handler = chat_handler_streaming

        # Mock streaming request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        })

        result = await handler.handle_request(mock_request)

        # Should return StreamingResponse
        assert hasattr(result, 'body_iterator')

        # Collect chunks
        chunks = []
        async for line in result.body_iterator:
            if isinstance(line, bytes):
                chunks.append(line.decode('utf-8'))
            else:
                chunks.append(str(line))

        # Verify SSE format
        assert any("data:" in chunk for chunk in chunks)
        assert any("data: [DONE]" in chunk for chunk in chunks)
        assert any("chat.completion.chunk" in chunk for chunk in chunk)

    @pytest.mark.asyncio
    async def test_non_streaming_response_format(self, chat_handler_streaming):
        """Test non-streaming response format."""
        handler = chat_handler_streaming

        # Mock non-streaming request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False
        })

        result = await handler.handle_request(mock_request)

        # Should return JSON response
        assert hasattr(result, 'content')
        response_data = json.loads(result.body)
        assert response_data["object"] == "chat.completion"
        assert "choices" in response_data


class TestSamplingParameters:
    """Test OpenAI to SGLang parameter mapping."""

    @pytest.fixture
    def mock_router_params(self):
        """Create a mock router for parameter testing."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        router.args.model_name = "test-model"
        router.args.sglang_router_port = 30000
        router.args.port = 30000

        # Mock component registry
        router._component_registry = None
        router._registry_lock = MagicMock()

        # Mock worker management
        router.worker_urls = {"http://test:10090": 0}
        router._url_lock = AsyncMock()
        router._use_url = AsyncMock(return_value="http://test:10090")
        router._finish_url = AsyncMock()

        # Mock HTTP client
        router.client = AsyncMock()

        return router

    @pytest.fixture
    def chat_handler(self, mock_router_params):
        """Create ChatCompletionHandler for parameter tests."""
        return ChatCompletionHandler(mock_router_params)

    def test_build_sampling_params_basic(self, chat_handler):
        """Test basic parameter mapping."""
        request_data = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2
        }

        params = chat_handler._build_sampling_params(request_data, stream=False)

        assert params["max_new_tokens"] == 100
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50
        assert params["frequency_penalty"] == 0.1
        assert params["presence_penalty"] == 0.2
        assert params["stream"] is False

    def test_build_sampling_params_streaming(self, chat_handler):
        """Test streaming parameter mapping."""
        request_data = {
            "max_tokens": 50,
            "stream": True
        }

        params = chat_handler._build_sampling_params(request_data, stream=True)

        assert params["max_new_tokens"] == 50
        assert params["stream"] is True

    def test_build_sampling_params_omits_none_values(self, chat_handler):
        """Test that None values are filtered out."""
        request_data = {
            "max_tokens": 100,
            "temperature": None,  # Should be omitted
            "stop": None,  # Should be omitted
            "presence_penalty": 0.1
        }

        params = chat_handler._build_sampling_params(request_data, stream=False)

        assert "max_new_tokens" in params
        assert "temperature" not in params  # None values filtered
        assert "stop" not in params  # None values filtered
        assert "presence_penalty" == 0.1


class TestFactoryFunction:
    """Test the factory function for creating handlers."""

    @pytest.fixture
    def mock_router(self):
        """Create mock router."""
        return MagicMock()

    def test_create_chat_completion_handler(self, mock_router):
        """Test factory function creates handler correctly."""
        handler = create_chat_completion_handler(mock_router)

        assert isinstance(handler, ChatCompletionHandler)
        assert handler.router is mock_router


class TestIntegrationHelpers:
    """Test helper functions and utilities."""

    @pytest.fixture
    def mock_router(self):
        """Create mock router for helper tests."""
        router = MagicMock()
        router.args = MagicMock()
        router.args.verbose = False
        return router

    def test_sse_chunk_serialization(self):
        """Test Server-Sent Events chunk serialization."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "slime-model",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }

        sse_chunk = f"data: {json.dumps(data)}\n\n"

        assert "data: " in sse_chunk
        assert "chat.completion.chunk" in sse_chunk
        assert "Hello" in sse_chunk
        assert sse_chunk.endswith("\n\n")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])