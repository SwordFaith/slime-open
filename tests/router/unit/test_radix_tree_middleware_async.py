"""Test async RadixTreeMiddleware implementation."""
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware


@pytest.fixture
def mock_router():
    """Create a mock router for testing."""
    router = Mock()
    router.args = Mock()
    router.args.hf_checkpoint = "test-checkpoint"
    router.verbose = False
    return router


@pytest.fixture
async def middleware(mock_router):
    """Create middleware instance with mocked dependencies."""
    with patch('slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer_instance = Mock()

        def mock_tokenizer_call(text, add_special_tokens=True):
            if text == "":
                return {"input_ids": []}
            elif text == "Hello world":
                return {"input_ids": [1, 2, 3]}
            elif text == "Hello":
                return {"input_ids": [1, 2]}
            elif text == " world":
                return {"input_ids": [3]}
            elif text == "Test insertion":
                return {"input_ids": [4, 5, 6]}
            elif text == "Test":
                return {"input_ids": [100, 101]}
            elif text == " 0":
                return {"input_ids": [102]}
            elif text == " 1":
                return {"input_ids": [103]}
            elif text == " 2":
                return {"input_ids": [104]}
            elif text == "Unknown text":
                return {"input_ids": [7, 8, 9]}
            elif text == " generated":
                return {"input_ids": [10]}
            else:
                # Default behavior for other texts
                return {"input_ids": [99] * len(text.split())}

        mock_tokenizer_instance.side_effect = mock_tokenizer_call
        mock_tokenizer_instance.decode.return_value = "Hello world"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        app = Mock()
        middleware = RadixTreeMiddleware(app, router=mock_router)

        yield middleware


@pytest.mark.asyncio
async def test_middleware_async_cache_retrieval(middleware):
    """Test that middleware uses async cache retrieval correctly."""
    # First insert test data
    await middleware.radix_tree.insert_async(
        "Hello world", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1
    )

    # Test with cached content
    tokens, logprobs, loss_mask = await middleware._retrieve_cache("Hello world")

    assert tokens == [1, 2, 3]
    assert len(logprobs) == 3
    assert len(loss_mask) == 3


@pytest.mark.asyncio
async def test_middleware_async_cache_retrieval_partial_match(middleware):
    """Test middleware with partial cache match."""
    # Add "Hello" to cache
    await middleware.radix_tree.insert_async(
        "Hello", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1
    )

    # Test with "Hello world" (partial match)
    tokens, logprobs, loss_mask = await middleware._retrieve_cache("Hello world")

    # Should get cached "Hello" + tokenized " world"
    assert len(tokens) >= 2  # At least the cached "Hello" tokens
    assert len(logprobs) == len(tokens)
    assert len(loss_mask) == len(tokens)


@pytest.mark.asyncio
async def test_middleware_async_cache_retrieval_no_match(middleware):
    """Test middleware with no cache match."""
    tokens, logprobs, loss_mask = await middleware._retrieve_cache("Unknown text")

    # Should return tokenized version of "Unknown text"
    assert len(tokens) > 0  # Should be tokenized
    assert len(logprobs) == len(tokens)
    assert len(loss_mask) == len(tokens)


@pytest.mark.asyncio
async def test_middleware_async_cache_insertion(middleware):
    """Test that middleware uses async cache insertion correctly."""
    result = await middleware._insert_cache(
        "Test insertion", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1], weight_version=2
    )

    # Verify insertion was successful
    assert result is True

    # Verify insertion worked by retrieving it
    tokens, logprobs, loss_mask = await middleware._retrieve_cache("Test insertion")

    assert tokens == [4, 5, 6]
    assert logprobs == [-0.4, -0.5, -0.6]
    assert loss_mask == [1, 1, 1]


@pytest.mark.asyncio
async def test_middleware_dispatch_with_cache_hit(mock_router):
    """Test middleware dispatch with cache hit."""
    with patch('slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer_instance = Mock()

        def mock_tokenizer_call(text, add_special_tokens=True):
            if text == "":
                return {"input_ids": []}
            elif text == "Hello world":
                return {"input_ids": [1, 2, 3]}
            else:
                return {"input_ids": [99] * len(text.split())}

        mock_tokenizer_instance.side_effect = mock_tokenizer_call
        mock_tokenizer_instance.decode.return_value = "Hello world"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        app = Mock()
        middleware = RadixTreeMiddleware(app, router=mock_router)

        # Pre-populate cache
        await middleware.radix_tree.insert_async(
            "Hello world", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1
        )

        # Mock request
        request = Mock(spec=Request)
        request.url.path = "/generate"
        request.json = AsyncMock(return_value={"text": "Hello world"})

        # Mock response from next middleware
        mock_response = JSONResponse(content={
            "text": "",
            "output_ids": [],
            "meta_info": {"weight_version": 1}
        })
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        result = await middleware.dispatch(request, call_next)

        # Verify cache was used (request was modified with cached tokens)
        assert hasattr(request, '_json')
        assert request._json["input_tokens"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_middleware_dispatch_with_cache_miss(mock_router):
    """Test middleware dispatch with cache miss."""
    with patch('slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": [7, 8, 9]}
        mock_tokenizer_instance.decode.return_value = "New text"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        app = Mock()
        middleware = RadixTreeMiddleware(app, router=mock_router)

        # Mock request with uncached text
        request = Mock(spec=Request)
        request.url.path = "/generate"
        request.json = AsyncMock(return_value={"text": "New text"})

        # Mock response from next middleware
        mock_response = JSONResponse(content={
            "text": " generated",
            "output_ids": [10],
            "meta_info": {
                "weight_version": 2,
                "output_token_logprobs": [[-0.5, 10]]
            }
        })
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        result = await middleware.dispatch(request, call_next)

        # Verify request was modified with tokenized input
        assert hasattr(request, '_json')
        assert len(request._json["input_tokens"]) > 0


@pytest.mark.asyncio
async def test_middleware_concurrent_operations(middleware):
    """Test that middleware handles concurrent operations correctly."""
    # Test concurrent cache retrievals
    async def retrieve_text(text: str):
        return await middleware._retrieve_cache(text)

    # Test concurrent cache insertions
    async def insert_text(text: str, tokens: list):
        return await middleware._insert_cache(text, tokens, [-0.1] * len(tokens), [1] * len(tokens), weight_version=1)

    # Run concurrent operations
    tasks = []

    # Concurrent retrievals
    for i in range(5):
        tasks.append(retrieve_text("Hello world"))

    # Concurrent insertions
    for i in range(3):
        tasks.append(insert_text(f"Test {i}", [100 + i, 101 + i]))

    # Wait for all operations to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify no exceptions occurred
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"

    # Verify some operations succeeded
    assert len(results) == 8


@pytest.mark.asyncio
async def test_middleware_error_handling(middleware):
    """Test middleware error handling with async operations."""
    # Test with empty text (should not raise exception)
    tokens, logprobs, loss_mask = await middleware._retrieve_cache("")
    assert tokens == []
    assert logprobs == []
    assert loss_mask == []

    # Test insertion with empty text (should return False)
    result = await middleware._insert_cache("", [], [], [], weight_version=1)
    assert result is False


if __name__ == "__main__":
    # Run tests manually for debugging
    asyncio.run(test_middleware_async_cache_retrieval(None))
    asyncio.run(test_middleware_async_cache_insertion(None))
    asyncio.run(test_middleware_concurrent_operations(None))
    print("All middleware async tests passed!")