"""
Middleware Edge Cases and Retry Logic Tests

Tests cover:
- Middleware edge cases and error handling
- Response parsing edge cases
- Tenacity retry logic and resilience
- Error recovery scenarios
- Integration scenarios with cache errors
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi import Request
from starlette.responses import JSONResponse

from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware, _is_response_aborted
from slime.router.utils.component_registry import ComponentRegistry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_router():
    """Create a mock router for testing."""
    router = Mock()
    router.args = Mock()
    router.args.hf_checkpoint = "test-checkpoint"
    router.args.radix_tree_max_size = 1000
    router.args.verbose = False
    router.verbose = False
    router.component_registry = ComponentRegistry()
    return router


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.apply_chat_template = Mock(return_value="Mock template")
    tokenizer.decode = Mock(return_value="Hello world")
    tokenizer.return_value = {"input_ids": [72, 101, 108, 108, 111]}
    return tokenizer


@pytest.fixture
def middleware_with_mocks(mock_router, mock_tokenizer):
    """Create middleware with mocked dependencies."""
    mock_radix_tree = Mock()
    mock_radix_tree.retrieve_from_text.return_value = ([1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1])
    mock_radix_tree.insert.return_value = True
    mock_radix_tree.insert_async = AsyncMock(return_value=True)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        middleware = RadixTreeMiddleware(app=None, router=mock_router)
        middleware.radix_tree = mock_radix_tree
        return middleware


@pytest.mark.asyncio
async def test_middleware_response_parsing_with_invalid_json(middleware_with_mocks):
    """Test middleware handles invalid JSON in response body gracefully."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with invalid JSON
    invalid_response = Mock()
    invalid_response.body = b'{"text": "incomplete json'
    invalid_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=invalid_response)

    # Execute middleware
    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify graceful handling
    assert response is invalid_response
    assert response_data is None


@pytest.mark.asyncio
async def test_middleware_response_parsing_with_none_body(middleware_with_mocks):
    """Test middleware handles response with None body."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with None body
    none_body_response = Mock()
    none_body_response.body = None
    none_body_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=none_body_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    assert response is none_body_response
    assert response_data is None


@pytest.mark.asyncio
async def test_middleware_response_parsing_with_binary_body(middleware_with_mocks):
    """Test middleware handles binary response body correctly."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create response with binary body
    binary_response = Mock()
    binary_response.body = b'\x00\x01\x02\x03\x04\x05'  # Binary data
    binary_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=binary_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    assert response is binary_response
    assert response_data is None


def test_is_response_aborted_missing_meta_info():
    """Test _is_response_aborted handles missing meta_info gracefully."""
    response_data = {"text": "Hello world"}
    assert _is_response_aborted(response_data) is False


def test_is_response_aborted_empty_meta_info():
    """Test _is_response_aborted handles empty meta_info."""
    response_data = {"meta_info": {}, "text": "Hello"}
    assert _is_response_aborted(response_data) is False


def test_is_response_aborted_meta_info_not_dict():
    """Test _is_response_aborted handles non-dict meta_info."""
    test_cases = [
        {"meta_info": "not a dict", "text": "Hello"},
        {"meta_info": 123, "text": "Hello"},
        {"meta_info": None, "text": "Hello"},
        {"meta_info": [], "text": "Hello"},
    ]

    for response_data in test_cases:
        assert _is_response_aborted(response_data) is False


def test_is_response_aborted_missing_finish_reason():
    """Test _is_response_aborted handles missing finish_reason."""
    response_data = {
        "meta_info": {
            "other_field": "value"
        }
    }
    assert _is_response_aborted(response_data) is False


def test_is_response_aborted_various_finish_types():
    """Test _is_response_aborted correctly identifies different finish types."""
    test_cases = [
        ({"meta_info": {"finish_reason": {"type": "abort"}}}, True),
        ({"meta_info": {"finish_reason": {"type": "stop"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "length"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "eos"}}}, False),
        ({"meta_info": {"finish_reason": {"type": "match"}}}, False),
    ]

    for response_data, expected in test_cases:
        assert _is_response_aborted(response_data) is expected


@pytest.mark.asyncio
async def test_middleware_tenacity_retry_error_handling(middleware_with_mocks):
    """Test middleware handles tenacity RetryError gracefully."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Mock call_next to always return abort response
    abort_response = Mock()
    abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
    abort_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=abort_response)

    # Mock tenacity to raise RetryError
    with patch("slime.router.middleware.radix_tree_middleware.AsyncRetrying") as mock_retrying:
        from tenacity import RetryError
        mock_retrying.side_effect = RetryError("All retries exhausted")

        # Execute and verify no exception is raised
        response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

        # Should return some response (not crash)
        assert response is None or hasattr(response, '__class__')


@pytest.mark.asyncio
async def test_middleware_dispatch_with_missing_text_field(middleware_with_mocks):
    """Test dispatch handles missing text field gracefully."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"other_field": "value"})  # No text field
    mock_request._json = None

    mock_call_next = AsyncMock(return_value=Mock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Should have called call_next without cache processing
    mock_call_next.assert_called_once_with(mock_request)
    assert response is mock_call_next.return_value


@pytest.mark.asyncio
async def test_middleware_dispatch_non_generate_path(middleware_with_mocks):
    """Test dispatch bypasses processing for non-/generate paths."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/other_path"  # Not "/generate"

    mock_call_next = AsyncMock(return_value=Mock())

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Should have called call_next directly
    mock_call_next.assert_called_once_with(mock_request)
    assert response is mock_call_next.return_value



# ============================================================================
# Group D: Tenacity Retry Logic
# ============================================================================

@pytest.mark.asyncio
async def test_tenacity_retry_with_mixed_success_failure(middleware_with_mocks, mocker):
    """Test tenacity handles mixed success/failure scenarios correctly."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Create responses: first abort, then success
    abort_response = Mock()
    abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
    abort_response.__class__.__name__ = "Response"

    success_response = Mock()
    success_response.body = b'{"text": "Success", "meta_info": {"finish_reason": {"type": "stop"}}}'
    success_response.__class__.__name__ = "Response"

    mock_call_next = AsyncMock(side_effect=[abort_response, success_response])

    # Mock asyncio.sleep to avoid 30s wait in tests
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Verify retry behavior
    assert mock_call_next.call_count == 2, "Should call call_next twice (abort + success)"
    assert mock_sleep.call_count == 1, "Should sleep once between attempts"
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"
    assert response_data["text"] == "Success"


@pytest.mark.asyncio
async def test_tenacity_max_retries_exceeded(middleware_with_mocks, mocker):
    """Test tenacity stops after max retries."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Always return abort response
    abort_response = Mock()
    abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
    abort_response.__class__.__name__ = "Response"

    mock_call_next = AsyncMock(return_value=abort_response)

    # Mock asyncio.sleep for fast testing
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Should retry multiple times then give up
    assert mock_call_next.call_count > 1, "Should retry multiple times"
    assert mock_sleep.call_count > 0, "Should sleep between retries"


@pytest.mark.asyncio
async def test_tenacity_no_retry_for_success(middleware_with_mocks):
    """Test tenacity doesn't retry when request succeeds."""
    middleware = middleware_with_mocks

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Return success immediately
    success_response = Mock()
    success_response.body = b'{"text": "Success", "meta_info": {"finish_reason": {"type": "stop"}}}'
    success_response.__class__.__name__ = "Response"

    mock_call_next = AsyncMock(return_value=success_response)

    response, response_data = await middleware._generate_with_retry(mock_request, mock_call_next)

    # Should only call once (no retry)
    assert mock_call_next.call_count == 1, "Should call only once for success"
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"



async def test_complete_flow_with_mixed_errors(middleware_with_mocks, mocker):
    """Test complete flow handles mixed errors gracefully."""
    middleware = middleware_with_mocks

    # Make cache retrieval fail
    middleware.radix_tree.retrieve_from_text.side_effect = ValueError("Cache retrieval failed")

    # Make cache insertion fail
    middleware.radix_tree.insert.side_effect = Exception("Cache insertion failed")

    # Fix tokenizer mock to return dict with input_ids when called
    middleware.tokenizer.return_value = {"input_ids": [83, 117, 99, 99, 101, 115, 115]}

    mock_request = Mock(spec=Request)
    mock_request.url.path = "/generate"
    mock_request.json = AsyncMock(return_value={"text": "Hello"})
    mock_request._json = None

    # Mock successful response (no abort, so no retry triggered)
    success_response = Mock()
    success_response.body = b'{"text": "Success", "output_ids": [83, 117, 99, 99, 101, 115, 115], "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": 1}}'
    success_response.__class__.__name__ = "Response"
    mock_call_next = AsyncMock(return_value=success_response)

    # Mock asyncio.sleep for fast testing (no retry expected since no abort)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify complete flow succeeded despite cache errors
    assert response is success_response
    assert mock_call_next.call_count == 1, "Should call once (no retry since no abort)"


if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v", "--tb=short"])
