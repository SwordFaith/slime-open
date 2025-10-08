"""
Unit tests for tenacity-based retry logic in RadixTreeMiddleware.

Tests verify:
1. Tenacity correctly retries on abort responses
2. Maximum 5 retry attempts enforced
3. Fixed 30-second wait between retries
4. RetryError handling (reraise=False)
5. _parse_response() method correctness
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, Mock
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_router():
    """Mock SlimeRouter instance."""
    router = MagicMock()
    router.args = MagicMock()
    router.args.hf_checkpoint = "test_models/Qwen3-0.6B"
    router.verbose = False
    return router


@pytest.fixture
def mock_app():
    """Mock FastAPI application."""
    return FastAPI()


@pytest.fixture
def middleware(mock_app, mock_router, mocker):
    """Create RadixTreeMiddleware with mocked dependencies."""
    # Mock AutoTokenizer to avoid loading real model
    mock_tokenizer_class = mocker.patch(
        "slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer"
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "User: Hello"
    mock_tokenizer.return_value = {"input_ids": [72, 101, 108, 108, 111]}
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # Mock StringRadixTrie
    mock_trie_class = mocker.patch(
        "slime.router.middleware_hub.radix_tree_middleware.StringRadixTrie"
    )
    mock_trie = MagicMock()
    mock_trie.retrieve_from_text.return_value = (
        [72, 101, 108, 108, 111],  # input_tokens
        [0.1, 0.1, 0.1, 0.1, 0.1],  # input_logprobs
        [0, 0, 0, 0, 0],  # input_loss_mask
    )
    mock_trie.insert.return_value = True
    mock_trie_class.return_value = mock_trie

    # Import after patching
    from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware

    middleware_instance = RadixTreeMiddleware(mock_app, router=mock_router)
    return middleware_instance


@pytest.fixture
def mock_request():
    """Create mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.url.path = "/generate"

    async def json_func():
        return {"text": "User: Hello", "stream": True}

    request.json = json_func
    request._json = None
    return request


def create_sglang_response(
    text: str, finish_reason: str = "stop", weight_version: int = 1
) -> JSONResponse:
    """Helper to create mock SGLang response."""
    response_data = {
        "text": text,
        "output_ids": [ord(c) for c in text],  # Simple char → ASCII mapping
        "meta_info": {
            "finish_reason": {"type": finish_reason},
            "output_token_logprobs": [
                [0.1, ord(c)] for c in text
            ],  # [(logp, token_id), ...]
            "weight_version": weight_version,
        },
    }
    return JSONResponse(content=response_data, status_code=200)


# ============================================================================
# Test Scenarios
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_retry_on_abort(middleware, mock_request, mocker):
    """
    Test: Abort once → tenacity retries → success.

    Scenario:
    - 1st call: abort
    - 2nd call: success
    - Should call asyncio.sleep(30) exactly once
    - Should call call_next() exactly twice
    """
    # Mock call_next to return abort then success
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    success_response = create_sglang_response(" Hi there!", finish_reason="stop")
    mock_call_next = AsyncMock(side_effect=[abort_response, success_response])

    # Mock asyncio.sleep to avoid real waiting
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response, response_data = await middleware._generate_with_retry(
        mock_request, mock_call_next
    )

    # Verify retry behavior
    assert mock_call_next.call_count == 2, "Should call call_next twice (abort + retry)"
    mock_sleep.assert_called_once_with(30)

    # Verify final response is success
    assert response.status_code == 200
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"
    assert " Hi there!" in response_data["text"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_stop_after_5_attempts(middleware, mock_request, mocker):
    """
    Test: 5 consecutive aborts → tenacity stops → return last response.

    Scenario:
    - All 5 calls return abort
    - Should call call_next() exactly 5 times
    - Should call asyncio.sleep(30) exactly 4 times (no sleep after last attempt)
    - Should return the last abort response
    """
    # Mock call_next to always return abort
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    mock_call_next = AsyncMock(return_value=abort_response)

    # Mock asyncio.sleep
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response, response_data = await middleware._generate_with_retry(
        mock_request, mock_call_next
    )

    # Verify retry exhaustion
    assert mock_call_next.call_count == 5, "Should retry exactly 5 times"
    assert (
        mock_sleep.call_count == 4
    ), "Should sleep 4 times (after attempts 1-4, not after 5th)"

    # Verify final response is still abort (no more retries)
    assert response.status_code == 200
    assert response_data["meta_info"]["finish_reason"]["type"] == "abort"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_wait_fixed_30s(middleware, mock_request, mocker):
    """
    Test: Verify each retry waits exactly 30 seconds.

    Scenario:
    - 2 aborts + 1 success
    - Should call asyncio.sleep(30) exactly twice
    - All sleep calls should use 30 seconds (fixed wait)
    """
    # Mock call_next to return abort twice then success
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    success_response = create_sglang_response(" Hi there!", finish_reason="stop")
    mock_call_next = AsyncMock(
        side_effect=[abort_response, abort_response, success_response]
    )

    # Mock asyncio.sleep
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response, response_data = await middleware._generate_with_retry(
        mock_request, mock_call_next
    )

    # Verify all sleep calls are exactly 30 seconds
    assert mock_sleep.call_count == 2, "Should sleep twice (2 aborts)"
    assert all(
        call.args[0] == 30 for call in mock_sleep.call_args_list
    ), "All sleep calls should be 30 seconds"

    # Verify final success
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenacity_reraise_false(middleware, mock_request, mocker):
    """
    Test: RetryError should be caught, not propagated.

    Scenario:
    - All 5 attempts return abort
    - tenacity should catch RetryError internally
    - Should return last abort response, not raise exception
    """
    # Mock call_next to always return abort
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    mock_call_next = AsyncMock(return_value=abort_response)

    # Mock asyncio.sleep
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute - should NOT raise RetryError
    try:
        response, response_data = await middleware._generate_with_retry(
            mock_request, mock_call_next
        )
        # Should return last abort response
        assert response.status_code == 200
        assert response_data["meta_info"]["finish_reason"]["type"] == "abort"
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.unit
def test_parse_response_extraction_json_response(middleware):
    """
    Test: Verify _parse_response() correctly extracts from JSONResponse.content.

    Scenario:
    - Mock response with .content attribute (dict)
    - Should return the dict directly
    """
    # Mock JSONResponse with .content (dict)
    mock_response = MagicMock()
    mock_response.content = {"text": "Hello", "meta_info": {"finish_reason": {"type": "stop"}}}
    delattr(mock_response, "body")  # Ensure body doesn't exist

    result = middleware._parse_response(mock_response)
    assert result == {"text": "Hello", "meta_info": {"finish_reason": {"type": "stop"}}}


@pytest.mark.unit
def test_parse_response_extraction_response_body(middleware):
    """
    Test: Verify _parse_response() correctly extracts from Response.body.

    Scenario:
    - Mock response with .body attribute (bytes)
    - Should parse JSON from bytes
    """
    # Mock Response with .body (bytes)
    mock_response = MagicMock()
    mock_response.body = b'{"text": "World", "meta_info": {}}'
    delattr(mock_response, "content")  # Ensure content doesn't exist

    result = middleware._parse_response(mock_response)
    assert result == {"text": "World", "meta_info": {}}


@pytest.mark.unit
def test_parse_response_extraction_malformed_json(middleware):
    """
    Test: Verify _parse_response() handles malformed JSON gracefully.

    Scenario:
    - Mock response with invalid JSON in body
    - Should return None instead of raising exception
    """
    # Mock Response with invalid JSON
    mock_response = MagicMock()
    mock_response.body = b"invalid json {{"
    delattr(mock_response, "content")

    result = middleware._parse_response(mock_response)
    assert result is None


@pytest.mark.unit
def test_parse_response_extraction_none_response(middleware):
    """
    Test: Verify _parse_response() handles None response gracefully.

    Scenario:
    - Pass None as response
    - Should return None without exception
    """
    result = middleware._parse_response(None)
    assert result is None


# ============================================================================
# Integration-like Test: Full Retry Flow
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_retry_flow_with_tenacity(middleware, mock_request, mocker):
    """
    Test: Complete retry flow using tenacity.

    Scenario:
    - Abort → Abort → Success
    - Should retry 3 times total
    - Should wait 2 times (30s each)
    - Should return final success response
    """
    # Mock call_next sequence
    abort_1 = create_sglang_response(" Abort1", finish_reason="abort")
    abort_2 = create_sglang_response(" Abort2", finish_reason="abort")
    success = create_sglang_response(" Success", finish_reason="stop")
    mock_call_next = AsyncMock(side_effect=[abort_1, abort_2, success])

    # Mock asyncio.sleep
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response, response_data = await middleware._generate_with_retry(
        mock_request, mock_call_next
    )

    # Verify complete flow
    assert mock_call_next.call_count == 3, "Should attempt 3 times"
    assert mock_sleep.call_count == 2, "Should sleep twice (after abort 1 and 2)"
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"
    assert " Success" in response_data["text"]
