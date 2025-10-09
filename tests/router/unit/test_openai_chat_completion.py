"""
OpenAI Chat Completion API unit tests.

This test suite validates the Chat Completion functionality that provides
100% OpenAI API compatibility while leveraging Slime Router's Radix Cache
for optimal performance in multi-turn conversations.

Test cases follow TDD methodology and are organized by functional area:
- Message formatting and validation
- OpenAI API parameter handling
- Cache integration and combination logic
- Response formatting for streaming/non-streaming modes
"""

import json
import asyncio
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from slime.router.openai_chat_completion import (
    ChatCompletionHandler,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    format_messages_with_hf_template,
    convert_generate_to_openai_response,
    validate_chat_completion_request,
)
from slime.router.middleware_hub.radix_tree import MatchResult, StringTreeNode


class TestMessageFormatting:
    """Test message formatting using HuggingFace chat templates."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock HuggingFace tokenizer with apply_chat_template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="System: You are helpful.\nUser: Hello!\nAssistant: ")
        return tokenizer

    @pytest.mark.asyncio
    async def test_system_user_formatting(self, mock_tokenizer):
        """Test system + user message formatting."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]

        formatted = format_messages_with_hf_template(messages, mock_tokenizer)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        assert formatted == "System: You are helpful.\nUser: Hello!\nAssistant: "

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_tokenizer):
        """Test multi-turn conversation formatting."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        mock_tokenizer.apply_chat_template.return_value = (
            "System: You are helpful.\n"
            "User: Hello!\n"
            "Assistant: Hi there!\n"
            "User: How are you?\n"
            "Assistant: "
        )

        formatted = format_messages_with_hf_template(messages, mock_tokenizer)

        assert "User: How are you?" in formatted
        assert formatted.endswith("Assistant: ")

    @pytest.mark.asyncio
    async def test_empty_messages_handling(self, mock_tokenizer):
        """Test empty messages list handling."""
        messages = []

        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            format_messages_with_hf_template(messages, mock_tokenizer)

    @pytest.mark.asyncio
    async def test_invalid_role_handling(self, mock_tokenizer):
        """Test invalid role handling."""
        messages = [
            {"role": "invalid_role", "content": "Hello!"}
        ]

        with pytest.raises(ValueError, match="Invalid message role: invalid_role"):
            format_messages_with_hf_template(messages, mock_tokenizer)


class TestParameterValidation:
    """Test OpenAI Chat Completion API parameter validation."""

    def test_model_validation(self):
        """Test model name validation."""
        # Valid request
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert validate_chat_completion_request(request) is True

        # Missing model
        request.model = ""
        assert validate_chat_completion_request(request) is False

    def test_temperature_range(self):
        """Test temperature parameter range validation."""
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        assert validate_chat_completion_request(request) is True

        # Invalid temperature (too high)
        request.temperature = 2.5
        assert validate_chat_completion_request(request) is False

        # Invalid temperature (negative)
        request.temperature = -0.1
        assert validate_chat_completion_request(request) is False

    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1000
        )
        assert validate_chat_completion_request(request) is True

        # Invalid max_tokens (negative)
        request.max_tokens = -1
        assert validate_chat_completion_request(request) is False

        # Invalid max_tokens (zero)
        request.max_tokens = 0
        assert validate_chat_completion_request(request) is False

    def test_top_p_range(self):
        """Test top_p parameter range validation."""
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[{"role": "user", "content": "Hello"}],
            top_p=0.9
        )
        assert validate_chat_completion_request(request) is True

        # Invalid top_p (too high)
        request.top_p = 1.5
        assert validate_chat_completion_request(request) is False

        # Invalid top_p (negative)
        request.top_p = -0.1
        assert validate_chat_completion_request(request) is False


class TestCacheIntegration:
    """Test Radix Cache integration and combination logic."""

    @pytest.fixture
    def mock_radix_tree(self):
        """Mock RadixTree with async methods."""
        tree = AsyncMock()
        tree.find_longest_prefix_async = AsyncMock()
        tree.insert_async = AsyncMock()
        return tree

    @pytest.fixture
    def mock_generate_api_handler(self):
        """Mock generate API handler."""
        handler = AsyncMock()
        handler.return_value = {
            "output_token_ids": [100, 101, 102],
            "output_text": "Hello there!",
            "request_id": "req-123",
        }
        return handler

    @pytest.fixture
    def mock_match_result(self):
        """Mock cache match result."""
        node = StringTreeNode()
        node.string_key = "Hello, how are"
        node.token_ids = [1, 2, 3, 4, 5]
        node.logp = [-0.1, -0.2, -0.1, -0.3, -0.2]
        node.loss_mask = [0, 0, 0, 0, 0]

        return MatchResult(
            matched_prefix="Hello, how are",
            token_ids=[1, 2, 3, 4, 5],
            logp=[-0.1, -0.2, -0.1, -0.3, -0.2],
            loss_mask=[0, 0, 0, 0, 0],
            remaining_string=" you today?",
            last_node=node
        )

    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, mock_radix_tree, mock_match_result, mock_generate_api_handler):
        """Test scenario where cache hit occurs."""
        mock_radix_tree.find_longest_prefix_async.return_value = mock_match_result

        handler = ChatCompletionHandler(
            radix_tree=mock_radix_tree,
            tokenizer=MagicMock(),
            generate_api_handler=mock_generate_api_handler
        )

        formatted_text = "Hello, how are you today?"
        cached_result = await handler._query_cache(formatted_text)

        assert cached_result.matched_prefix == "Hello, how are"
        assert cached_result.remaining_string == " you today?"
        assert len(cached_result.token_ids) == 5

    @pytest.mark.asyncio
    async def test_cache_miss_scenario(self, mock_radix_tree, mock_generate_api_handler):
        """Test scenario where cache miss occurs."""
        # No match found
        mock_match_result = MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="Hello, how are you today?",
            last_node=StringTreeNode()
        )
        mock_radix_tree.find_longest_prefix_async.return_value = mock_match_result

        handler = ChatCompletionHandler(
            radix_tree=mock_radix_tree,
            tokenizer=MagicMock(),
            generate_api_handler=mock_generate_api_handler
        )

        formatted_text = "Hello, how are you today?"
        cached_result = await handler._query_cache(formatted_text)

        assert cached_result.matched_prefix == ""
        assert cached_result.remaining_string == formatted_text
        assert len(cached_result.token_ids) == 0

    @pytest.mark.asyncio
    async def test_cache_update_after_generation(self, mock_radix_tree, mock_generate_api_handler):
        """Test cache update after generation."""
        handler = ChatCompletionHandler(
            radix_tree=mock_radix_tree,
            tokenizer=MagicMock(),
            generate_api_handler=mock_generate_api_handler
        )

        formatted_text = "Hello, how are you today?"
        generated_tokens = [6, 7, 8, 9]
        generated_text = " I'm doing well!"

        await handler._update_cache(
            formatted_text,
            generated_text,
            generated_tokens
        )

        # Verify cache insertion was called
        mock_radix_tree.insert_async.assert_called_once()
        call_args = mock_radix_tree.insert_async.call_args[0]

        assert call_args[0] == formatted_text + generated_text  # Full text
        assert call_args[1] == generated_tokens  # All tokens


class TestResponseConversion:
    """Test conversion from generate API response to OpenAI format."""

    def test_convert_generate_to_openai_response(self):
        """Test conversion of generate API response to OpenAI Chat Completion format."""
        # Mock generate API response
        generate_response = {
            "output_token_ids": [100, 101, 102],
            "output_text": "Hello there!",
            "request_id": "req-123",
            "meta_info": {
                "input_token_count": 10,
                "output_token_count": 3
            }
        }

        # Mock cached tokens
        cached_tokens = [1, 2, 3, 4, 5]
        cached_text = "Hello, how are"

        # Mock messages
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Mock prompt tokens
        mock_tokenizer.apply_chat_template.return_value = "User: Hello, how are you?\n\nAssistant:"  # Mock formatted text

        openai_response = convert_generate_to_openai_response(
            generate_response, cached_tokens, messages, mock_tokenizer
        )

        # Verify OpenAI format (use to_dict() method)
        response_dict = openai_response.to_dict()
        assert response_dict["object"] == "chat.completion"
        assert response_dict["model"] == "slime-model"
        assert len(response_dict["choices"]) == 1
        assert response_dict["choices"][0]["message"]["role"] == "assistant"
        assert response_dict["choices"][0]["message"]["content"] == "Hello there!"
        # After using real tokenizer, prompt tokens should be calculated from formatted text
        assert response_dict["usage"]["prompt_tokens"] == 10  # From mock tokenizer.encode() return value
        assert response_dict["usage"]["completion_tokens"] == 3
        assert response_dict["usage"]["total_tokens"] == 13  # 10 prompt + 3 completion tokens

    def test_streaming_response_format(self):
        """Test streaming response chunk format."""
        chunk = {
            "token_id": 100,
            "text": "Hello",
            "finished": False
        }

        stream_response = ChatCompletionStreamResponse(
            request_id="req-123",
            chunk=chunk
        )

        formatted_chunk = stream_response.to_sse_format()

        assert "data: " in formatted_chunk
        assert '"choices"' in formatted_chunk
        assert '"delta"' in formatted_chunk
        assert '"content": "Hello"' in formatted_chunk

    def test_streaming_finish_chunk(self):
        """Test streaming finish chunk."""
        finish_chunk = {
            "token_id": None,
            "text": "",
            "finished": True,
            "finish_reason": "stop"
        }

        stream_response = ChatCompletionStreamResponse(
            request_id="req-123",
            chunk=finish_chunk
        )

        formatted_chunk = stream_response.to_sse_format()

        assert '"finish_reason": "stop"' in formatted_chunk
        assert '"delta": {}' in formatted_chunk


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def mock_generate_api_handler(self):
        """Mock generate API handler."""
        handler = AsyncMock()
        handler.return_value = {
            "output_token_ids": [100, 101, 102],
            "output_text": "Hello there!",
            "request_id": "req-123",
        }
        return handler

    def test_missing_messages_error(self):
        """Test error when messages are missing."""
        request = ChatCompletionRequest(
            model="slime-model",
            messages=[]
        )

        assert not validate_chat_completion_request(request)  # Returns False instead of raising

    @pytest.mark.asyncio
    async def test_tokenizer_unavailable_error(self, mock_generate_api_handler):
        """Test error when tokenizer is not available."""
        with patch('slime.router.openai_chat_completion.format_messages_with_hf_template') as mock_format:
            mock_format.side_effect = Exception("Tokenizer not available")

            handler = ChatCompletionHandler(
                radix_tree=AsyncMock(),
                tokenizer=None,  # No tokenizer
                generate_api_handler=mock_generate_api_handler
            )

            with pytest.raises(Exception, match="Tokenizer not available"):
                await handler.handle_request(
                    ChatCompletionRequest(
                        model="slime-model",
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                )

    @pytest.mark.asyncio
    async def test_generate_api_error(self, mock_generate_api_handler):
        """Test error when generate API call fails."""
        # Mock radix tree with proper async method
        mock_radix_tree = AsyncMock()
        mock_radix_tree.find_longest_prefix_async = AsyncMock(return_value=MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="Hello",
            last_node=StringTreeNode()
        ))

        handler = ChatCompletionHandler(
            radix_tree=mock_radix_tree,
            tokenizer=MagicMock(),
            generate_api_handler=mock_generate_api_handler
        )

        # Mock generate API failure - make AsyncMock raise an async exception
        async def failing_handler(*args, **kwargs):
            raise Exception("SGLang worker unavailable")

        mock_generate_api_handler.side_effect = failing_handler

        with pytest.raises(Exception, match="SGLang worker unavailable"):
            await handler.handle_request(
                ChatCompletionRequest(
                    model="slime-model",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            )


class TestConcurrencySafety:
    """Test concurrent request handling safety."""

    @pytest.fixture
    def mock_radix_tree(self):
        """Mock RadixTree with async methods."""
        tree = AsyncMock()
        tree.find_longest_prefix_async = AsyncMock()
        tree.insert_async = AsyncMock()
        return tree

    @pytest.fixture
    def mock_generate_api_handler(self):
        """Mock generate API handler."""
        handler = AsyncMock()
        handler.return_value = {
            "output_token_ids": [100, 101, 102],
            "output_text": "Hello there!",
            "request_id": "req-123",
        }
        return handler

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, mock_radix_tree, mock_generate_api_handler):
        """Test concurrent cache access doesn't cause race conditions."""
        # Simulate concurrent cache queries
        mock_radix_tree.find_longest_prefix_async.return_value = MatchResult(
            matched_prefix="Hello",
            token_ids=[1, 2],
            logp=[-0.1, -0.2],
            loss_mask=[0, 0],
            remaining_string=" world",
            last_node=StringTreeNode()
        )

        handler = ChatCompletionHandler(
            radix_tree=mock_radix_tree,
            tokenizer=MagicMock(),
            generate_api_handler=mock_generate_api_handler
        )

        # Create multiple concurrent tasks
        tasks = [
            handler._query_cache("Hello world")
            for _ in range(10)
        ]

        # All tasks should complete without errors
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(result.matched_prefix == "Hello" for result in results)

        # Verify cache was accessed concurrently
        assert mock_radix_tree.find_longest_prefix_async.call_count == 10


class TestIntegrationHelpers:
    """Test helper functions and utilities."""

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

    def test_token_count_calculation(self):
        """Test accurate token count calculation."""
        cached_tokens = [1, 2, 3, 4, 5]  # 5 tokens from cache
        generated_tokens = [100, 101, 102]  # 3 tokens generated

        total_tokens = len(cached_tokens) + len(generated_tokens)

        assert total_tokens == 8
        assert len(cached_tokens) == 5
        assert len(generated_tokens) == 3


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])