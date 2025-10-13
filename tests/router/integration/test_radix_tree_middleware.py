"""
Integration tests for RadixTreeMiddleware retry logic.

Tests cover:
- Successful generation without retry
- Abort retry with async sleep
- Retry exhaustion after 5 attempts
- Cache insertion after successful generation

Mock Strategy:
- Mock call_next() to return abort/success responses
- Mock asyncio.sleep() to avoid real waiting
- Mock StringRadixTrie.insert() to verify cache insertion
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware
from slime.router.core.radix_tree import StringRadixTrie


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

    # Mock component_registry to return False for has() calls
    # This ensures middleware creates new instances using our patched StringRadixTrie
    mock_registry = MagicMock()
    mock_registry.has.return_value = False
    router.get_component_registry.return_value = mock_registry

    return router


@pytest.fixture
def mock_app():
    """Mock FastAPI application."""
    return FastAPI()


@pytest.fixture
def middleware(mock_app, mock_router, mocker):
    """Create RadixTreeMiddleware with mocked dependencies."""
    # Mock AutoTokenizer to avoid loading real model
    mock_tokenizer_class = mocker.patch("slime.router.middleware.radix_tree_middleware.AutoTokenizer")
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "User: Hello"
    mock_tokenizer.return_value = {"input_ids": [72, 101, 108, 108, 111]}
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # Mock StringRadixTrie
    mock_trie_class = mocker.patch("slime.router.middleware.radix_tree_middleware.StringRadixTrie")
    mock_trie = MagicMock(spec=StringRadixTrie)
    # Mock get_or_create_tokenization_async (returns 4 values: tokens, logprobs, loss_mask, versions)
    mock_trie.get_or_create_tokenization_async = AsyncMock(return_value=(
        [72, 101, 108, 108, 111],  # input_tokens
        [0.1, 0.1, 0.1, 0.1, 0.1],  # input_logprobs
        [0, 0, 0, 0, 0],  # input_loss_mask
        [-1, -1, -1, -1, -1],  # generation_versions (non-AI-generated)
    ))
    mock_trie.insert_async = AsyncMock(return_value=True)
    mock_trie_class.return_value = mock_trie

    middleware = RadixTreeMiddleware(mock_app, router=mock_router)
    return middleware


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


def create_sglang_response(text: str, finish_reason: str = "stop", weight_version: int = 1) -> JSONResponse:
    """Helper to create mock SGLang response."""
    response_data = {
        "text": text,
        "output_ids": [ord(c) for c in text],  # Simple char → ASCII mapping
        "meta_info": {
            "finish_reason": {"type": finish_reason},
            "output_token_logprobs": [[0.1, ord(c)] for c in text],  # [(logp, token_id), ...]
            "weight_version": weight_version,
        },
    }
    return JSONResponse(content=response_data, status_code=200)


# ============================================================================
# Test Scenarios
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_successful_generation_no_retry(middleware, mock_request, mocker):
    """
    Test: Successful generation should NOT trigger retry.

    Scenario:
    - call_next() returns success response (finish_reason="stop")
    - Should NOT call asyncio.sleep()
    - Should call call_next() exactly once
    """
    # Mock call_next to return success response
    mock_call_next = AsyncMock(return_value=create_sglang_response(" Hi there!", finish_reason="stop"))

    # Mock asyncio.sleep to verify it's NOT called
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify
    assert mock_call_next.call_count == 1, "Should call call_next exactly once for successful generation"
    mock_sleep.assert_not_called()
    assert response.status_code == 200

    # Verify response content
    response_data = json.loads(response.body.decode("utf-8"))
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_abort_retry_with_async_sleep(middleware, mock_request, mocker):
    """
    Test: Abort once → should async wait 30s → retry → success.

    Scenario:
    - 1st call: abort
    - 2nd call: success
    - Should call asyncio.sleep(30) exactly once
    - Should call call_next() exactly twice

    NOTE: This test verifies the FIXED code behavior (async sleep).
    """
    # Mock call_next to return abort then success
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    success_response = create_sglang_response(" Hi there!", finish_reason="stop")
    mock_call_next = AsyncMock(side_effect=[abort_response, success_response])

    # Mock asyncio.sleep to avoid real waiting
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Execute
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify retry behavior
    assert mock_call_next.call_count == 2, "Should call call_next twice (abort + retry)"
    mock_sleep.assert_called_once_with(30)

    # Verify final response is success
    assert response.status_code == 200
    response_data = json.loads(response.body.decode("utf-8"))
    assert response_data["meta_info"]["finish_reason"]["type"] == "stop"
    assert " Hi there!" in response_data["text"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_exhaustion(middleware, mock_request, mocker):
    """
    Test: 5 consecutive aborts → return last response (retry exhausted).

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
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify retry exhaustion
    assert mock_call_next.call_count == 5, "Should retry exactly 5 times"
    assert mock_sleep.call_count == 4, "Should sleep 4 times (after attempts 1-4, not after 5th)"

    # Verify final response is still abort (no more retries)
    assert response.status_code == 200
    response_data = json.loads(response.body.decode("utf-8"))
    assert response_data["meta_info"]["finish_reason"]["type"] == "abort"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_insertion_after_generation(middleware, mock_request, mocker):
    """
    Test: Successful generation → should insert into cache.

    Scenario:
    - call_next() returns success with output_token_logprobs
    - Should call radix_tree.insert() with correct parameters
    - Verify insert called with: full_text, full_token_ids, full_logprobs, full_loss_mask, weight_version
    """
    # Mock call_next to return success with token logprobs
    generated_text = " Hi there!"
    success_response = create_sglang_response(generated_text, finish_reason="stop", weight_version=5)
    mock_call_next = AsyncMock(return_value=success_response)

    # Spy on radix_tree.insert_async to verify it's called (middleware uses async version)
    insert_spy = mocker.spy(middleware.radix_tree, "insert_async")

    # Execute
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify response success
    assert response.status_code == 200

    # Verify cache insertion
    insert_spy.assert_called_once()
    call_args = insert_spy.call_args

    # Extract call arguments
    full_text_arg = call_args[0][0]
    full_token_ids_arg = call_args[0][1]
    full_logprobs_arg = call_args[0][2]
    full_loss_mask_arg = call_args[0][3]
    weight_version_arg = call_args[1]["weight_version"]

    # Verify arguments
    assert "User: Hello" in full_text_arg, "Full text should contain input text"
    assert generated_text in full_text_arg, "Full text should contain generated text"

    # Verify full_token_ids = input_tokens + generated_token_ids
    expected_generated_tokens = [ord(c) for c in generated_text]
    assert full_token_ids_arg[-len(expected_generated_tokens) :] == expected_generated_tokens

    # Verify full_logprobs = input_logprobs + generated_logprobs
    assert full_logprobs_arg is not None
    assert len(full_logprobs_arg) == len(full_token_ids_arg)

    # Verify full_loss_mask = input_loss_mask (all 0s) + generated_loss_mask (all 1s)
    input_loss_mask_len = 5  # From mock retrieve_from_text
    generated_loss_mask_len = len(expected_generated_tokens)
    assert len(full_loss_mask_arg) == input_loss_mask_len + generated_loss_mask_len

    # Verify input part is all 0s (prompt)
    assert all(m == 0 for m in full_loss_mask_arg[:input_loss_mask_len]), "Input loss_mask should be all 0s"

    # Verify generated part is all 1s (response)
    assert all(m == 1 for m in full_loss_mask_arg[input_loss_mask_len:]), "Generated loss_mask should be all 1s"

    # Verify weight_version
    assert weight_version_arg == 5, "Weight version should match response meta_info"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_generate_path_bypass(middleware, mock_request, mocker):
    """
    Test: Non-/generate paths should bypass middleware logic.

    Scenario:
    - Request path is /health
    - Should directly call call_next without any processing
    - Should NOT call radix_tree methods
    """
    # Change request path
    mock_request.url.path = "/health"

    # Mock call_next
    health_response = JSONResponse(content={"status": "ok"}, status_code=200)
    mock_call_next = AsyncMock(return_value=health_response)

    # Spy on radix_tree methods
    retrieve_spy = mocker.spy(middleware.radix_tree, "retrieve_from_text")

    # Execute
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify bypass
    assert response.status_code == 200
    mock_call_next.assert_called_once()
    retrieve_spy.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_text_in_request(middleware, mocker):
    """
    Test: Missing 'text' field → should bypass cache logic.

    Scenario:
    - Request has no 'text' or 'input_ids'
    - Should directly call call_next without cache retrieval
    """
    # Create request without text
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/generate"

    async def json_func():
        return {"stream": False}  # No text or input_ids

    mock_request.json = json_func

    # Mock call_next
    mock_response = JSONResponse(content={"text": "response"}, status_code=200)
    mock_call_next = AsyncMock(return_value=mock_response)

    # Spy on radix_tree.retrieve_from_text
    retrieve_spy = mocker.spy(middleware.radix_tree, "retrieve_from_text")

    # Execute
    response = await middleware.dispatch(mock_request, mock_call_next)

    # Verify bypass
    assert response.status_code == 200
    retrieve_spy.assert_not_called()
