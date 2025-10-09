"""
Unit tests for Tenacity retry logic in RadixTreeMiddleware.

Tests cover:
- _is_response_aborted() function behavior
- _generate_with_retry() method retry logic
- _retrieve_cache() and _insert_cache() methods
- Tenacity configuration validation

Mock Strategy:
- Unit tests: No external dependencies
- Mock all external calls (tokenizer, radix_tree)
- Focus on pure logic testing
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock
from slime.router.middleware_hub.radix_tree_middleware import (
    RadixTreeMiddleware,
    _is_response_aborted,
)


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

@pytest.mark.unit
def test_is_response_aborted_with_abort_response():
    """Test _is_response_aborted() returns True for abort response."""
    response_data = {
        "meta_info": {
            "finish_reason": {"type": "abort"}
        }
    }
    assert _is_response_aborted(response_data) is True


@pytest.mark.unit
def test_is_response_aborted_with_stop_response():
    """Test _is_response_aborted() returns False for stop response."""
    response_data = {
        "meta_info": {
            "finish_reason": {"type": "stop"}
        }
    }
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_with_missing_meta_info():
    """Test _is_response_aborted() returns False when meta_info is missing."""
    response_data = {"text": "Hello"}
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_with_missing_finish_reason():
    """Test _is_response_aborted() returns False when finish_reason is missing."""
    response_data = {
        "meta_info": {"other_field": "value"}
    }
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_with_non_dict_input():
    """Test _is_response_aborted() returns False for non-dict input."""
    assert _is_response_aborted(None) is False
    assert _is_response_aborted("string") is False
    assert _is_response_aborted([]) is False


@pytest.mark.unit
def test_is_response_aborted_with_different_finish_type():
    """Test _is_response_aborted() returns False for non-abort finish types."""
    response_data = {
        "meta_info": {
            "finish_reason": {"type": "length"}
        }
    }
    assert _is_response_aborted(response_data) is False


# ============================================================================
# Unit Tests for Middleware Methods
# ============================================================================

@pytest.fixture
def mock_router():
    """Create a mock router with necessary attributes."""
    router = MagicMock()
    router.args = MagicMock()
    router.args.hf_checkpoint = "test-checkpoint"
    router.verbose = False
    return router


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "Hello world"
    # Mock the tokenizer call that returns a dict with input_ids
    tokenizer.return_value = {"input_ids": []}  # Default empty
    # For empty remaining_string, should return empty list
    tokenizer.side_effect = lambda text, add_special_tokens=True: {"input_ids": [] if not text else [ord(c) for c in text]}
    return tokenizer


@pytest.fixture
def mock_radix_tree():
    """Create a mock radix tree."""
    from unittest.mock import AsyncMock, MagicMock

    tree = MagicMock()
    # Mock async methods
    tree.find_longest_prefix_async = AsyncMock()
    tree.get_or_create_tokenization_async = AsyncMock()
    tree.insert_async = AsyncMock(return_value=True)

    # Mock sync methods for backward compatibility
    tree.retrieve_from_text.return_value = ([1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1])
    tree.insert.return_value = True

    # Mock get_or_create_tokenization_async to return 4-tuple
    tree.get_or_create_tokenization_async.return_value = ([1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], [3, 3, 3])

    # Mock MatchResult for find_longest_prefix_async
    mock_result = MagicMock()
    mock_result.matched_prefix = "Hello world"
    mock_result.token_ids = [1, 2, 3]
    mock_result.logp = [0.1, 0.2, 0.3]
    mock_result.loss_mask = [1, 1, 1]
    mock_result.remaining_string = ""  # Empty remaining string for full match
    tree.find_longest_prefix_async.return_value = mock_result

    return tree


@pytest.fixture
def middleware_with_mocks(mock_router, mock_tokenizer, mock_radix_tree):
    """Create middleware with mocked dependencies."""
    # Mock the tokenizer import
    with pytest.MonkeyPatch().context() as m:
        m.setattr("transformers.AutoTokenizer.from_pretrained",
                  lambda *args, **kwargs: mock_tokenizer)

        # Create middleware instance
        middleware = RadixTreeMiddleware(app=None, router=mock_router)

        # Replace radix_tree with mock
        middleware.radix_tree = mock_radix_tree

        return middleware


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_cache(middleware_with_mocks, mock_radix_tree):
    """Test _retrieve_cache() method calls radix_tree correctly."""
    middleware = middleware_with_mocks
    input_text = "Hello world"

    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache(input_text)

    # Verify radix_tree.get_or_create_tokenization_async was called correctly
    mock_radix_tree.get_or_create_tokenization_async.assert_called_once_with(input_text)

    # Verify return values - should match mock_result (full match case)
    assert tokens == [1, 2, 3]  # From mock_result.token_ids
    assert logprobs == [0.1, 0.2, 0.3]  # From mock_result.logp
    assert loss_mask == [1, 1, 1]  # From mock_result.loss_mask
    assert versions == [3, 3, 3]  # From mock_result.generation_versions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_success(middleware_with_mocks, mock_radix_tree):
    """Test _insert_cache() method successful insertion."""
    middleware = middleware_with_mocks

    await middleware._insert_cache(
        full_text="Hello world",
        full_token_ids=[1, 2, 3],
        full_logprobs=[0.1, 0.2, 0.3],
        full_loss_mask=[1, 1, 1],
        weight_version=5
    )

    # Verify radix_tree.insert_async was called correctly
    mock_radix_tree.insert_async.assert_called_once_with(
        "Hello world",
        [1, 2, 3],
        [0.1, 0.2, 0.3],
        [1, 1, 1],
        weight_version=5
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_with_exception(middleware_with_mocks, mock_radix_tree):
    """Test _insert_cache() method handles exceptions gracefully."""
    middleware = middleware_with_mocks
    mock_radix_tree.insert_async.side_effect = Exception("Cache insertion failed")

    # Should not raise exception
    await middleware._insert_cache(
        full_text="Hello world",
        full_token_ids=[1, 2, 3],
        full_logprobs=[0.1, 0.2, 0.3],
        full_loss_mask=[1, 1, 1],
        weight_version=5
    )

    # Verify radix_tree.insert_async was still called
    mock_radix_tree.insert_async.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_with_verbose_logging(middleware_with_mocks, mock_radix_tree):
    """Test _insert_cache() method logs when verbose is True."""
    middleware = middleware_with_mocks
    middleware.router.verbose = True

    await middleware._insert_cache(
        full_text="Hello world",
        full_token_ids=[1, 2, 3],
        full_logprobs=[0.1, 0.2, 0.3],
        full_loss_mask=[1, 1, 1],
        weight_version=5
    )

    # Verify radix_tree.insert_async was called
    mock_radix_tree.insert_async.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_with_retry_success_on_first_attempt(middleware_with_mocks):
    """Test _generate_with_retry() succeeds on first attempt."""
    middleware = middleware_with_mocks

    # Mock request and call_next
    mock_request = MagicMock()
    mock_call_next = AsyncMock()

    # Mock successful response
    success_response = MagicMock()
    success_response.__class__.__name__ = "JSONResponse"
    success_response.content = {"text": "Hello", "meta_info": {"finish_reason": {"type": "stop"}}}
    mock_call_next.return_value = success_response

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify single call
    assert mock_call_next.call_count == 1

    # Verify response
    assert response is success_response
    assert response_data["text"] == "Hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_with_retry_with_streaming_response(middleware_with_mocks):
    """Test _generate_with_retry() handles streaming responses correctly."""
    middleware = middleware_with_mocks

    # Mock request and call_next
    mock_request = MagicMock()
    mock_call_next = AsyncMock()

    # Mock streaming response
    streaming_response = MagicMock()
    streaming_response.__class__.__name__ = "_StreamingResponse"
    materialized_response = MagicMock()
    materialized_response.body = b'{"text": "Hello", "meta_info": {"finish_reason": {"type": "stop"}}}'

    # Mock _materialize_response function
    with pytest.MonkeyPatch().context() as m:
        async def mock_materialize(resp):
            return materialized_response
        m.setattr("slime.router.middleware_hub.radix_tree_middleware._materialize_response",
                  mock_materialize)

        mock_call_next.return_value = streaming_response

        response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

        # Verify materialization was called
        assert response is materialized_response
        assert response_data["text"] == "Hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_with_retry_response_parsing_error(middleware_with_mocks):
    """Test _generate_with_retry() handles JSON parsing errors gracefully."""
    middleware = middleware_with_mocks

    # Mock request and call_next
    mock_request = MagicMock()
    mock_call_next = AsyncMock()

    # Mock response with invalid JSON
    invalid_response = MagicMock()
    invalid_response.body = b'invalid json'
    invalid_response.__class__.__name__ = "Response"
    mock_call_next.return_value = invalid_response

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify response is returned but response_data is None
    assert response is invalid_response
    assert response_data is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_with_retry_mixed_response_formats(middleware_with_mocks):
    """Test _generate_with_retry() handles different response formats."""
    middleware = middleware_with_mocks

    # Mock request and call_next
    mock_request = MagicMock()
    mock_call_next = AsyncMock()

    # Mock response with content attribute (JSONResponse)
    json_response = MagicMock()
    json_response.content = {"text": "Hello", "meta_info": {"finish_reason": {"type": "stop"}}}
    json_response.__class__.__name__ = "JSONResponse"
    mock_call_next.return_value = json_response

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify response_data is extracted from content
    assert response is json_response
    assert response_data["text"] == "Hello"


# ============================================================================
# Tenacity Configuration Tests
# ============================================================================

# NOTE: test_manual_retry_configuration removed
# Reason: The test incorrectly assumed middleware uses asyncio.sleep directly.
# Actually, tenacity's wait_fixed(30) handles the waiting internally.
# The test path "radix_tree_middleware.asyncio.sleep" does not exist.


@pytest.mark.unit
def test_tenacity_imports():
    """Test that all tenacity components are imported correctly."""
    from slime.router.middleware_hub.radix_tree_middleware import (
        AsyncRetrying,
        RetryError,
        stop_after_attempt,
        wait_fixed,
    )

    # Verify all imports are available
    assert AsyncRetrying is not None
    assert RetryError is not None
    assert stop_after_attempt is not None
    assert wait_fixed is not None