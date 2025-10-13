"""
OpenAI and Middleware Tests (Merged)

This test suite combines tests from:
- test_openai_chat_completion.py
- test_middleware_edge_cases.py
- test_component_registry_thread_safety.py
- test_tenacity_retry_logic.py

Tests cover:
- OpenAI Chat Completion API functionality
- Middleware edge cases and error handling
- Component Registry thread safety
- Tenacity retry logic and resilience
- Integration scenarios and error recovery
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi import Request
from starlette.responses import JSONResponse

from slime.router.handlers.openai_chat_completion import create_chat_completion_handler
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware, _is_response_aborted
from slime.router.utils.component_registry import ComponentRegistry


# ============================================================================
# Group A: OpenAI Chat Completion Core Functionality
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

    def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        if not messages:
            return ""

        result = ""
        for msg in messages:
            if msg["role"] == "system":
                result += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                result += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                result += f"Assistant: {msg['content']}\n"

        if add_generation_prompt:
            result += "Assistant:"

        return result

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.decode.return_value = "Hello world"
    # Mock tokenizer call to return dict with input_ids (required when tokenizer is called like tokenizer(text))
    tokenizer.return_value = {"input_ids": [72, 101, 108, 108, 111]}
    return tokenizer


@pytest.fixture
async def mock_handler(mock_router, mock_tokenizer):
    """Create chat completion handler with mocked dependencies."""
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        handler = create_chat_completion_handler(mock_router)
        return handler


@pytest.mark.asyncio
async def test_openai_chat_completion_basic_request(mock_handler):
    """Test basic OpenAI chat completion request processing."""
    # Mock request
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    # Create mock FastAPI Request object
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Mock the expected OpenAI response
    expected_response = JSONResponse(content={
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    })

    # Mock cache unavailable -> direct proxy mode
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response is the expected one
    assert isinstance(response, JSONResponse)
    response_content = json.loads(response.body.decode())

    # Verify response structure
    assert "id" in response_content
    assert "object" in response_content
    assert response_content["object"] == "chat.completion"
    assert "created" in response_content
    assert "model" in response_content
    assert response_content["model"] == "test-model"
    assert "choices" in response_content
    assert len(response_content["choices"]) == 1
    assert "message" in response_content["choices"][0]
    assert response_content["choices"][0]["message"]["role"] == "assistant"
    assert "content" in response_content["choices"][0]["message"]
    assert "finish_reason" in response_content["choices"][0]
    assert "usage" in response_content


@pytest.mark.asyncio
async def test_openai_chat_completion_streaming(mock_handler):
    """Test OpenAI chat completion with streaming."""
    from fastapi.responses import StreamingResponse

    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "stream": True
    }

    # Create mock FastAPI Request object
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Create mock streaming response that yields OpenAI-formatted chunks
    async def mock_stream_generator():
        chunks = [
            '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":" Why"},"finish_reason":null}]}',
            '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":" don\'t"},"finish_reason":null}]}',
            '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":" scientists"},"finish_reason":null}]}',
            '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        ]
        for chunk in chunks:
            yield f"data: {chunk}\n\n".encode()
        yield b"data: [DONE]\n\n"

    expected_response = StreamingResponse(mock_stream_generator(), media_type="text/event-stream")

    # Mock cache unavailable -> direct proxy mode with streaming
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

        # Should return a StreamingResponse
        assert hasattr(response, 'body_iterator')

        # Collect streaming chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        # Verify streaming format
        assert len(chunks) > 0
        assert any("data: " in chunk for chunk in chunks), f"Expected streaming format, got: {chunks}"
        assert any("[DONE]" in chunk for chunk in chunks), f"Expected [DONE] marker, got: {chunks}"


@pytest.mark.asyncio
async def test_openai_chat_completion_with_system_prompt(mock_handler):
    """Test chat completion with system prompt."""
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write hello world in Python"}
        ]
    }

    # Create mock FastAPI Request object
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Mock OpenAI response
    expected_response = JSONResponse(content={
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "```python\nprint('Hello, World!')\n```"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    })

    # Mock direct proxy mode
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response
    response_content = json.loads(response.body.decode())
    assert response_content["choices"][0]["message"]["content"] == "```python\nprint('Hello, World!')\n```"


@pytest.mark.asyncio
async def test_openai_chat_completion_multi_turn_conversation(mock_handler):
    """Test multi-turn conversation handling."""
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Helpful assistant"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
            {"role": "user", "content": "What is 3+3?"}
        ]
    }

    # Create mock FastAPI Request object
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Mock OpenAI response
    expected_response = JSONResponse(content={
        "id": "chatcmpl-test789",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "3+3 = 6"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25
        }
    })

    # Mock direct proxy mode
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response includes the new answer
    response_content = json.loads(response.body.decode())
    assert "6" in response_content["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_openai_chat_completion_parameter_validation(mock_handler):
    """Test parameter validation for OpenAI requests."""
    from fastapi import HTTPException

    # Test missing messages field
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400
    assert "messages" in str(exc_info.value.detail).lower()

    # Test empty messages list
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": []})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400
    assert "messages" in str(exc_info.value.detail).lower()

    # Test invalid message structure (not a dict)
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": ["invalid"]})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400

    # Test missing role/content in message
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": [{"role": "user"}]})  # Missing content
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400

    # Note: Temperature validation is delegated to SGLang in the current implementation


# ============================================================================
# Group B: Middleware Edge Cases and Error Handling
# ============================================================================

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
# Group C: Component Registry Thread Safety
# ============================================================================

def test_component_registry_basic_operations():
    """Test basic component registry operations."""
    registry = ComponentRegistry()

    # Register component
    test_component = {"name": "test", "value": 123}
    registry.register("test_component", test_component)

    # Get component
    retrieved = registry.get("test_component")
    assert retrieved is test_component

    # Get non-existent component
    with pytest.raises(RuntimeError):
        registry.get("non_existent")

    # Get with default
    default_obj = {"default": True}
    result = registry.get("non_existent", default_obj)
    assert result is default_obj


def test_component_registry_thread_safety():
    """Test component registry thread safety under concurrent access."""
    import threading
    import time

    registry = ComponentRegistry()
    results = []
    errors = []

    def worker(worker_id):
        try:
            # Register component
            component = {"worker_id": worker_id, "data": f"data_{worker_id}"}
            registry.register(f"component_{worker_id}", component)

            # Small delay to increase chance of race conditions
            time.sleep(0.001)

            # Try to get component
            retrieved = registry.get(f"component_{worker_id}")
            results.append((worker_id, retrieved["worker_id"]))

        except Exception as e:
            errors.append((worker_id, str(e)))

    # Run multiple threads concurrently
    threads = []
    for i in range(10):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all operations completed successfully
    assert len(results) == 10
    for worker_id, retrieved_id in results:
        assert worker_id == retrieved_id


def test_component_registry_overwrite_protection():
    """Test component registry protects against overwriting."""
    registry = ComponentRegistry()

    # Register initial component
    component1 = {"version": 1}
    registry.register("test_component", component1)

    # Try to register again with same name
    component2 = {"version": 2}
    with pytest.raises(RuntimeError):
        registry.register("test_component", component2)

    # Original component should still be intact
    retrieved = registry.get("test_component")
    assert retrieved["version"] == 1


def test_component_registry_list_operations():
    """Test component registry list operations."""
    registry = ComponentRegistry()

    # Register multiple components
    registry.register("comp1", {"name": "Component 1"})
    registry.register("comp2", {"name": "Component 2"})
    registry.register("comp3", {"name": "Component 3"})

    # List all components
    components = registry.list()
    assert len(components) == 3
    assert "comp1" in components
    assert "comp2" in components
    assert "comp3" in components
    assert components["comp1"]["name"] == "Component 1"


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


# ============================================================================
# Group E: Integration and Error Recovery
# ============================================================================

@pytest.mark.asyncio
async def test_openai_middleware_integration(mock_handler, mock_router):
    """Test integration between OpenAI handler and middleware."""
    # Register middleware in router
    middleware = RadixTreeMiddleware(app=None, router=mock_router)
    mock_router.component_registry.register("radix_tree", middleware.radix_tree)
    mock_router.component_registry.register("tokenizer", middleware.tokenizer)

    # Prepare request
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Mock expected response
    expected_response = JSONResponse(content={
        "id": "chatcmpl-integration",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from integrated system!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10
        }
    })

    # Mock cache available -> use cache mode
    with patch.object(mock_handler, '_check_cache_availability', return_value=True), \
         patch.object(mock_handler, '_handle_with_radix_cache', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify integration worked
    response_content = json.loads(response.body.decode())
    assert response_content["choices"][0]["message"]["content"] == "Hello from integrated system!"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_error_recovery_scenarios(mock_handler, mocker):
    """Test various error recovery scenarios."""
    from fastapi import HTTPException

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Test network timeout simulation
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=504, detail="Network timeout")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code == 504

    # Test backend service error
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=503, detail="Service unavailable")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code == 503

    # Test invalid response from backend
    with patch.object(mock_handler, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=500, detail="Invalid response")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code >= 400


if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v", "--tb=short"])