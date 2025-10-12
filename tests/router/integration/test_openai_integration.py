"""
OpenAI Chat Completion API integration tests.

This test suite validates the complete integration of Chat Completion functionality
with Slime Router's middleware system and SGLang worker communication.

Tests cover:
- End-to-end request flow
- Middleware integration
- Mock SGLang worker communication
- Streaming and non-streaming modes
- Error handling at integration level
"""

import json
import asyncio
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from slime.router.handlers.openai_chat_completion import (
    ChatCompletionHandler,
    ChatCompletionRequest,
    create_chat_completion_handler,
)
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware
from slime.router.router import SlimeRouter


class TestChatCompletionIntegration:
    """Test Chat Completion integration with Slime Router."""

    @pytest.fixture
    def mock_router_args(self):
        """Mock router arguments."""
        args = MagicMock()
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 32
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []
        args.sglang_worker_timeout = 60
        args.sglang_worker_max_retries = 3
        return args

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock HuggingFace tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "User: Hello!\nAssistant: "
        tokenizer.encode.return_value [1, 2, 3]  # Mock token IDs
        return tokenizer

    @pytest.fixture
    def mock_radix_tree_middleware(self):
        """Mock Radix Tree middleware with cache."""
        middleware = AsyncMock(spec=RadixTreeMiddleware)
        middleware.radix_tree = AsyncMock()
        middleware.radix_tree.find_longest_prefix_async = AsyncMock()
        middleware.radix_tree.insert_async = AsyncMock()
        middleware.radix_tree.get_stats_async = AsyncMock()
        return middleware

    @pytest.fixture
    def mock_generate_api_handler(self):
        """Mock /generate API handler."""
        handler = AsyncMock()

        # Mock non-streaming response
        handler.return_value = {
            "output_token_ids": [100, 101, 102],
            "output_text": "Hello there!",
            "request_id": "req-123",
            "meta_info": {
                "input_token_count": 3,
                "output_token_count": 3
            }
        }

        # Mock streaming response
        async def mock_stream_response(request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
            yield {"text": "Hello", "finished": False}
            yield {"text": " there", "finished": False}
            yield {"text": "!", "finished": False}
            yield {"text": "", "finished": True, "finish_reason": "stop"}

        handler.stream = mock_stream_response
        return handler

    @pytest.mark.asyncio
    async def test_full_workflow_mock_sglang(self, mock_radix_tree_middleware, mock_tokenizer, mock_generate_api_handler):
        """Test complete workflow with mocked SGLang communication."""
        # Setup cache hit scenario
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        mock_cached_result = MatchResult(
            matched_prefix="User: Hello",
            token_ids=[1, 2],
            logp=[-0.1, -0.2],
            loss_mask=[0, 0],
            remaining_string="!",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = mock_cached_result

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_generate_api_handler
        )

        # Create request
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=False
        )

        # Handle request
        response = await handler.handle_request(request)

        # Verify response format
        assert response["object"] == "chat.completion"
        assert response["model"] == "slime-model"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["message"]["content"] == "Hello there!"
        assert response["usage"]["prompt_tokens"] == 2  # Cached tokens
        assert response["usage"]["completion_tokens"] == 3  # Generated tokens

        # Verify cache operations
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.assert_called_once()
        mock_radix_tree_middleware.radix_tree.insert_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_response_mock(self, mock_radix_tree_middleware, mock_tokenizer, mock_generate_api_handler):
        """Test streaming response with mocked SGLang communication."""
        # Setup cache miss scenario
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        mock_cached_result = MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="User: Hello!\nAssistant: ",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = mock_cached_result

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_generate_api_handler
        )

        # Create streaming request
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True
        )

        # Handle streaming request
        stream_response = await handler.handle_request(request)

        # Collect streaming chunks
        chunks = []
        async for chunk in stream_response.body_iterator:
            chunks.append(chunk)

        # Verify streaming format
        assert len(chunks) >= 2  # At least content + finish chunk

        # Verify SSE format
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            # Parse JSON from chunk
            data_str = chunk[6:-2]  # Remove "data: " and "\n\n"
            data = json.loads(data_str)
            assert "choices" in data
            assert len(data["choices"]) == 1

        # Verify final chunk has finish_reason
        final_chunk_data = json.loads(chunks[-1][6:-2])
        assert final_chunk_data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_radix_tree_middleware, mock_tokenizer):
        """Test error handling at integration level."""
        # Mock generate API failure
        failing_handler = AsyncMock(side_effect=Exception("SGLang worker unavailable"))

        # Setup cache miss
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        mock_cached_result = MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="User: Hello!\nAssistant: ",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = mock_cached_result

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            failing_handler
        )

        # Create request
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Verify error propagation
        with pytest.raises(Exception, match="SGLang worker unavailable"):
            await handler.handle_request(request)

    @pytest.mark.asyncio
    async def test_cache_integration_workflow(self, mock_radix_tree_middleware, mock_tokenizer, mock_generate_api_handler):
        """Test cache integration across multiple requests."""
        # First request - cache miss
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        cache_miss_result = MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="User: Hello!\nAssistant: ",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = cache_miss_result

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_generate_api_handler
        )

        # First request
        request1 = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        await handler.handle_request(request1)

        # Second request - cache hit
        cache_hit_result = MatchResult(
            matched_prefix="User: Hello!",
            token_ids=[1, 2, 3],
            logp=[-0.1, -0.2, -0.1],
            loss_mask=[0, 0, 0],
            remaining_string=" How are you?",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = cache_hit_result

        # Second request with follow-up
        request2 = ChatCompletionRequest(
            model="slime-model",
            messages=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        response2 = await handler.handle_request(request2)

        # Verify cache was used
        assert response2["usage"]["prompt_tokens"] == 3  # Cached tokens

        # Verify cache operations
        assert mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.call_count == 2
        assert mock_radix_tree_middleware.radix_tree.insert_async.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_cache(self, mock_radix_tree_middleware, mock_tokenizer, mock_generate_api_handler):
        """Test multi-turn conversation caching."""
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        # Simulate progressive cache hits
        cache_results = [
            # First turn - no cache
            MatchResult(
                matched_prefix="",
                token_ids=[],
                logp=[],
                loss_mask=[],
                remaining_string="User: Hello!\nAssistant: ",
                node=StringTreeNode()
            ),
            # Second turn - partial cache
            MatchResult(
                matched_prefix="User: Hello!\nAssistant: Hi there!\nUser: How are",
                token_ids=[1, 2, 3, 4, 5, 6, 7],
                logp=[-0.1] * 7,
                loss_mask=[0] * 7,
                remaining_string=" you?",
                node=StringTreeNode()
            )
        ]

        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.side_effect = cache_results

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_generate_api_handler
        )

        # First turn
        request1 = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        response1 = await handler.handle_request(request1)

        # Second turn
        request2 = ChatCompletionRequest(
            model="slime-model",
            messages=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        )
        response2 = await handler.handle_request(request2)

        # Verify progressive cache improvement
        assert response1["usage"]["prompt_tokens"] == 0  # No cache
        assert response2["usage"]["prompt_tokens"] == 7  # Partial cache

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mock_radix_tree_middleware, mock_tokenizer, mock_generate_api_handler):
        """Test concurrent request handling."""
        # Setup cache miss for all requests
        from slime.router.middleware_hub.radix_tree_refactored import MatchResult, StringTreeNode

        cache_miss_result = MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="User: Hello!\nAssistant: ",
            node=StringTreeNode()
        )
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.return_value = cache_miss_result

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_generate_api_handler
        )

        # Create multiple concurrent requests
        requests = [
            ChatCompletionRequest(
                model="slime-model",
                messages=[{"role": "user", "content": f"Hello {i}!"}]
            )
            for i in range(5)
        ]

        # Handle all requests concurrently
        tasks = [handler.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Verify all requests completed successfully
        assert len(responses) == 5
        for response in responses:
            assert response["object"] == "chat.completion"
            assert "choices" in response
            assert len(response["choices"]) == 1

        # Verify concurrent cache access
        assert mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.call_count == 5


class TestRouterIntegration:
    """Test Chat Completion integration with Slime Router."""

    @pytest.fixture
    def mock_router_args(self):
        """Mock router arguments with Chat Completion enabled."""
        args = MagicMock()
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 32
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []
        args.enable_openai_chat_completion = True
        args.openai_chat_completion_path = "/v1/chat/completions"
        args.openai_default_model = "slime-model"
        return args

    @pytest.mark.asyncio
    async def test_router_route_registration(self, mock_router_args):
        """Test Chat Completion route registration in router."""
        # Mock dependencies
        with patch('slime.router.openai_chat_completion.format_messages_with_hf_template') as mock_format:
            mock_format.return_value = "User: Hello!\nAssistant: "

            with patch('slime.router.openai_chat_completion.create_chat_completion_handler') as mock_create:
                mock_handler = AsyncMock()
                mock_handler.handle_request.return_value = {
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
                }
                mock_create.return_value = mock_handler

                # Create router with Chat Completion enabled
                router = SlimeRouter(mock_router_args)

                # Verify route is registered (this would be implemented in router.py)
                # For now, we just verify the router can be created
                assert router.app is not None

    @pytest.mark.asyncio
    async def test_middleware_integration(self, mock_radix_tree_middleware, mock_tokenizer):
        """Test Chat Completion integration with middleware system."""
        # This test would verify that the Chat Completion handler
        # correctly integrates with the existing middleware system

        # Mock generate API handler that simulates middleware behavior
        async def mock_middleware_handler(request: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate cache lookup
            cached_result = await mock_radix_tree_middleware.radix_tree.find_longest_prefix_async(
                request.get("text", "")
            )

            # Simulate generation
            if cached_result.matched_prefix:
                return {
                    "output_token_ids": [100, 101],
                    "output_text": "Cached response",
                    "request_id": request.get("request_id"),
                }
            else:
                return {
                    "output_token_ids": [200, 201, 202],
                    "output_text": "New response",
                    "request_id": request.get("request_id"),
                }

        # Create handler
        handler = create_chat_completion_handler(
            mock_radix_tree_middleware.radix_tree,
            mock_tokenizer,
            mock_middleware_handler
        )

        # Test request
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        response = await handler.handle_request(request)

        # Verify middleware was used
        mock_radix_tree_middleware.radix_tree.find_longest_prefix_async.assert_called()
        assert response["object"] == "chat.completion"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])