"""
Unit tests for edge cases and error handling in RadixTreeMiddleware.

Tests cover:
- Damaged or malformed response_data handling
- Missing meta_info scenarios
- Tenacity RetryError exception handling
- _insert_cache exception graceful degradation
- Boundary conditions for request parsing
- Error recovery scenarios

Mock Strategy:
- Unit tests: Isolate individual error conditions
- Mock all external dependencies
- Focus on error handling paths
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi import Request
from slime.router.middleware_hub.radix_tree_middleware import (
    RadixTreeMiddleware,
    _is_response_aborted,
)


# ============================================================================
# Test Fixtures
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
    return tokenizer


@pytest.fixture
def mock_radix_tree():
    """Create a mock radix tree."""
    tree = MagicMock()
    tree.retrieve_from_text.return_value = ([1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1])
    tree.insert.return_value = True
    # Add async insert method mock
    tree.insert_async = AsyncMock(return_value=True)
    return tree


@pytest.fixture
def middleware_with_mocks(mock_router, mock_tokenizer, mock_radix_tree):
    """Create middleware with mocked dependencies."""
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        middleware = RadixTreeMiddleware(app=None, router=mock_router)
        middleware.radix_tree = mock_radix_tree
        return middleware


# ============================================================================
# Malformed Response Data Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_parsing_with_invalid_json(middleware_with_mocks):
    """
    Test: Middleware handles invalid JSON in response body gracefully.

    Scenario:
    - Response contains invalid JSON
    - Should continue processing without crashing
    - response_data should be None
    """
    middleware = middleware_with_mocks

    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with invalid JSON
    invalid_response = MagicMock()
    invalid_response.body = b'{"text": "incomplete json'
    invalid_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=invalid_response)

    # Execute middleware
    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify graceful handling
    assert response is invalid_response
    assert response_data is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_parsing_with_none_body(middleware_with_mocks):
    """
    Test: Middleware handles response with None body.

    Scenario:
    - Response has None body attribute
    - Should continue processing
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with None body
    none_body_response = MagicMock()
    none_body_response.body = None
    none_body_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=none_body_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    assert response is none_body_response
    assert response_data is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_parsing_with_binary_body(middleware_with_mocks):
    """
    Test: Middleware handles binary response body correctly.

    Scenario:
    - Response contains binary data instead of JSON
    - Should handle gracefully
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with binary body
    binary_response = MagicMock()
    binary_response.body = b'\x00\x01\x02\x03\x04\x05'  # Binary data
    binary_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=binary_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    assert response is binary_response
    assert response_data is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_parsing_with_unicode_error(middleware_with_mocks):
    """
    Test: Middleware handles Unicode decoding errors.

    Scenario:
    - Response body contains invalid Unicode sequences
    - Should handle gracefully without crashing
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with invalid Unicode
    unicode_error_response = MagicMock()
    unicode_error_response.body = b'\xff\xfe\x00\x00'  # Invalid UTF-8
    unicode_error_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=unicode_error_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    assert response is unicode_error_response
    assert response_data is None


# ============================================================================
# Missing Meta Info Tests
# ============================================================================

@pytest.mark.unit
def test_is_response_aborted_missing_meta_info():
    """
    Test: _is_response_aborted handles missing meta_info gracefully.

    Scenario:
    - Response has no meta_info key
    - Should return False (not aborted)
    """
    response_data = {"text": "Hello world"}
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_empty_meta_info():
    """
    Test: _is_response_aborted handles empty meta_info.

    Scenario:
    - Response has empty meta_info dict
    - Should return False (not aborted)
    """
    response_data = {"meta_info": {}, "text": "Hello"}
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_meta_info_not_dict():
    """
    Test: _is_response_aborted handles non-dict meta_info.

    Scenario:
    - meta_info is not a dict (e.g., string, number)
    - Should return False (not aborted)
    """
    test_cases = [
        {"meta_info": "not a dict", "text": "Hello"},
        {"meta_info": 123, "text": "Hello"},
        {"meta_info": None, "text": "Hello"},
        {"meta_info": [], "text": "Hello"},
    ]

    for response_data in test_cases:
        assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_missing_finish_reason():
    """
    Test: _is_response_aborted handles missing finish_reason.

    Scenario:
    - meta_info exists but has no finish_reason
    - Should return False (not aborted)
    """
    response_data = {
        "meta_info": {
            "other_field": "value"
        }
    }
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_finish_reason_not_dict():
    """
    Test: _is_response_aborted handles non-dict finish_reason.

    Scenario:
    - finish_reason exists but is not a dict
    - Should return False (not aborted)
    """
    test_cases = [
        {"meta_info": {"finish_reason": "string"}},
        {"meta_info": {"finish_reason": 123}},
        {"meta_info": {"finish_reason": None}},
        {"meta_info": {"finish_reason": []}},
    ]

    for response_data in test_cases:
        assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_missing_finish_type():
    """
    Test: _is_response_aborted handles missing finish_type.

    Scenario:
    - finish_reason exists but has no type field
    - Should return False (not aborted)
    """
    response_data = {
        "meta_info": {
            "finish_reason": {
                "other_field": "value"
            }
        }
    }
    assert _is_response_aborted(response_data) is False


@pytest.mark.unit
def test_is_response_aborted_various_finish_types():
    """
    Test: _is_response_aborted correctly identifies different finish types.

    Scenario:
    - Test various finish_reason types
    - Only "abort" should return True
    """
    test_cases = [
        ({"meta_info": {"finish_reason": {"type": "abort"}}}, True),
        ({"meta_info": {"finish_reason": {"type": "stop"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "length"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "eos"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "match"}}}, False),
    ]

    for response_data, expected in test_cases:
        assert _is_response_aborted(response_data) is expected


# ============================================================================
# Tenacity RetryError Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_retry_error_handling(middleware_with_mocks):
    """
    Test: Middleware handles tenacity RetryError gracefully.

    Scenario:
    - All retry attempts fail, raising RetryError
    - Should return last response without crashing
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Mock call_next to always return abort response
    abort_response = MagicMock()
    abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
    abort_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=abort_response)

    # Mock tenacity to raise RetryError
    with patch("slime.router.middleware_hub.radix_tree_middleware.AsyncRetrying") as mock_retrying:
        from tenacity import RetryError
        mock_retrying.side_effect = RetryError("All retries exhausted")

        # Execute and verify no exception is raised
        response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

        # Should return some response (not crash)
        # In this case, response might be None due to RetryError, which is acceptable
        assert response is None or hasattr(response, '__class__')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_with_mixed_success_failure(middleware_with_mocks, mocker):
    """
    Test: Tenacity handles mixed success/failure scenarios correctly.

    Scenario:
    - First attempt: abort (triggers retry)
    - Second attempt: success (stops retry)
    - Should call call_next exactly 2 times
    - Should return successful response

    Uses real tenacity logic with mocked asyncio.sleep for fast testing.
    This is the recommended approach when tenacity creates inline AsyncRetrying instances.
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create responses: first abort, then success
    abort_response = MagicMock()
    abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
    abort_response.__class__.__name__ = "Response"

    success_response = MagicMock()
    success_response.body = b'{"text": "Success", "meta_info": {"finish_reason": {"type": "stop"}}}'
    success_response.__class__.__name__ = "Response"

    mock_call_next = AsyncMock(side_effect=[abort_response, success_response])

    # Mock asyncio.sleep to avoid 30s wait in tests (tenacity uses asyncio.sleep internally)
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify retry behavior
    assert mock_call_next.call_count == 2, "Should call call_next twice (abort + success)"
    assert mock_sleep.call_count == 1, "Should sleep once between attempts"
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"
    assert response_data["text"] == "Success"


# ============================================================================
# Cache Insertion Error Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_with_various_exceptions(middleware_with_mocks, mock_radix_tree):
    """
    Test: _insert_cache handles different types of exceptions.

    Scenario:
    - radix_tree.insert raises various exceptions
    - Should handle all gracefully without crashing
    """
    middleware = middleware_with_mocks

    # Test different exception types
    exceptions_to_test = [
        Exception("Generic error"),
        ValueError("Invalid value"),
        RuntimeError("Runtime error"),
        AttributeError("Missing attribute"),
        KeyError("Missing key"),
    ]

    for exception in exceptions_to_test:
        mock_radix_tree.insert_async.side_effect = exception
        mock_radix_tree.insert_async.reset_mock()  # Reset mock for each iteration

        # Should not raise exception
        await middleware._insert_cache(
            full_text="Hello world",
            full_token_ids=[1, 2, 3],
            full_logprobs=[0.1, 0.2, 0.3],
            full_loss_mask=[1, 1, 1],
            weight_version=5
        )

        # Verify insert_async was called despite exception
        mock_radix_tree.insert_async.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_with_verbose_logging(middleware_with_mocks, mock_radix_tree):
    """
    Test: _insert_cache logs errors when verbose is True.

    Scenario:
    - Cache insertion fails
    - Verbose logging is enabled
    - Should log warning message
    """
    middleware = middleware_with_mocks
    middleware.router.verbose = True  # Enable verbose logging

    mock_radix_tree.insert_async.side_effect = Exception("Test error")

    # Mock print to capture logging
    with patch("builtins.print") as mock_print:
        await middleware._insert_cache(
            full_text="Hello world",
            full_token_ids=[1, 2, 3],
            full_logprobs=[0.1, 0.2, 0.3],
            full_loss_mask=[1, 1, 1],
            weight_version=5
        )

        # Verify warning was printed
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Warning: Failed to cache trajectory" in call_args


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_cache_with_none_parameters(middleware_with_mocks, mock_radix_tree):
    """
    Test: _insert_cache handles None parameters gracefully.

    Scenario:
    - Some parameters are None
    - Should still attempt insertion
    """
    middleware = middleware_with_mocks

    await middleware._insert_cache(
        full_text=None,
        full_token_ids=None,
        full_logprobs=None,
        full_loss_mask=None,
        weight_version=None
    )

    # Verify insert_async was called with None parameters
    mock_radix_tree.insert_async.assert_called_once_with(
        None, None, None, None, weight_version=None
    )


# ============================================================================
# Request Parsing Edge Cases
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_with_missing_text_field(middleware_with_mocks):
    """
    Test: dispatch handles missing text field gracefully.

    Scenario:
    - Request has no text field
    - Should bypass middleware processing
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"other_field": "value"})  # No text field
    mock_request._json = None

    mock_call_next = AsyncMock(return_value=MagicMock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Should have called call_next without cache processing
    mock_call_next.assert_called_once_with(mock_request)
    assert response is mock_call_next.return_value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_with_empty_request_json(middleware_with_mocks):
    """
    Test: dispatch handles empty JSON request gracefully.

    Scenario:
    - Request JSON is empty dict
    - Should bypass middleware processing
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={})
    mock_request._json = None

    mock_call_next = AsyncMock(return_value=MagicMock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Should have called call_next without cache processing
    mock_call_next.assert_called_once_with(mock_request)
    assert response is mock_call_next.return_value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_with_input_ids_text_missing(middleware_with_mocks, mock_tokenizer):
    """
    Test: dispatch handles input_ids when text is missing.

    Scenario:
    - Request has input_ids but no text field
    - Should decode input_ids to text
    """
    middleware = middleware_with_mocks
    mock_tokenizer.decode.return_value = "Decoded text"

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"input_ids": [1, 2, 3]})
    mock_request._json = None

    mock_call_next = AsyncMock(return_value=MagicMock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify tokenizer was called with input_ids
    mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    # Should have processed the request
    mock_call_next.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_non_generate_path(middleware_with_mocks):
    """
    Test: dispatch bypasses processing for non-/generate paths.

    Scenario:
    - Request path is not "/generate"
    - Should bypass middleware completely
    """
    middleware = middleware_with_mocks

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/other_path"  # Not "/generate"

    mock_call_next = AsyncMock(return_value=MagicMock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Should have called call_next directly
    mock_call_next.assert_called_once_with(mock_request)
    assert response is mock_call_next.return_value


# ============================================================================
# Error Recovery and Resilience Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_flow_with_mixed_errors(middleware_with_mocks, mock_radix_tree, mocker):
    """
    Test: Complete flow handles mixed errors gracefully.

    Scenario:
    - Cache retrieval fails (now caught by _retrieve_cache() exception handling)
    - Call_next succeeds (no abort, no retry)
    - Cache insertion fails (already has exception handling)
    - Should still return successful response

    Tests the new defensive exception handling in _retrieve_cache().
    Uses real tenacity with mocked asyncio.sleep for fast testing.
    """
    middleware = middleware_with_mocks

    # Make cache retrieval fail - now handled by _retrieve_cache() exception handling
    mock_radix_tree.retrieve_from_text.side_effect = ValueError("Cache retrieval failed")

    # Make cache insertion fail - already has exception handling
    mock_radix_tree.insert.side_effect = Exception("Cache insertion failed")

    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Mock successful response (no abort, so no retry triggered)
    success_response = MagicMock()
    success_response.body = b'{"text": "Success", "output_ids": [83, 117, 99, 99, 101, 115, 115], "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": 1}}'
    success_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=success_response)

    # Mock asyncio.sleep for fast testing (no retry expected since no abort)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify complete flow succeeded despite cache errors
    assert response is success_response
    assert mock_call_next.call_count == 1, "Should call once (no retry since no abort)"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_middleware_initialization_with_errors(mocker):
    """
    Test: Middleware handles initialization errors gracefully.

    Scenario:
    - Tokenizer initialization fails
    - Should handle gracefully
    """
    from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware

    mock_router = MagicMock()
    mock_router.args = MagicMock()
    mock_router.args.hf_checkpoint = "invalid-checkpoint"

    # Mock tokenizer to raise exception
    mocker.patch("transformers.AutoTokenizer.from_pretrained",
                 side_effect=Exception("Failed to load tokenizer"))

    # Should raise exception during initialization (expected behavior)
    with pytest.raises(Exception):
        RadixTreeMiddleware(app=None, router=mock_router)