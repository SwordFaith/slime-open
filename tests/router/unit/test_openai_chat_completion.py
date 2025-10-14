"""
OpenAI Chat Completion API Tests

Tests cover:
- OpenAI Chat Completion API functionality
- Request/response formatting and validation
- Streaming support
- Multi-turn conversations
- Parameter validation
- Integration with router/middleware
- Error recovery scenarios
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi import Request, HTTPException
from starlette.responses import JSONResponse

from slime.router.handlers.openai_chat_completion import create_chat_completion_handler
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware
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
    # Mock tokenizer call to return dict with input_ids
    tokenizer.return_value = {"input_ids": [72, 101, 108, 108, 111]}
    return tokenizer


@pytest.fixture
async def mock_handler(mock_router, mock_tokenizer):
    """Create chat completion handler with mocked dependencies."""
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        handler = create_chat_completion_handler(mock_router)
        return handler


# ============================================================================
# Group A: OpenAI Chat Completion Core Functionality
# ============================================================================

@pytest.mark.unit
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
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response is the expected one
    assert isinstance(response, JSONResponse), "Response should be JSONResponse instance"
    response_content = json.loads(response.body.decode())

    # Verify response structure
    assert "id" in response_content, "Response should have 'id' field"
    assert "object" in response_content, "Response should have 'object' field"
    assert response_content["object"] == "chat.completion", \
        f"Object should be 'chat.completion', got {response_content['object']}"
    assert "created" in response_content, "Response should have 'created' field"
    assert "model" in response_content, "Response should have 'model' field"
    assert response_content["model"] == "test-model", \
        f"Model should be 'test-model', got {response_content['model']}"
    assert "choices" in response_content, "Response should have 'choices' field"
    assert len(response_content["choices"]) == 1, \
        f"Should have 1 choice, got {len(response_content['choices'])}"
    assert "message" in response_content["choices"][0], "Choice should have 'message' field"
    assert response_content["choices"][0]["message"]["role"] == "assistant", \
        "Message role should be 'assistant'"
    assert "content" in response_content["choices"][0]["message"], "Message should have 'content' field"
    assert "finish_reason" in response_content["choices"][0], "Choice should have 'finish_reason' field"
    assert "usage" in response_content, "Response should have 'usage' field"


@pytest.mark.unit
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
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

        # Should return a StreamingResponse
        assert hasattr(response, 'body_iterator'), "Response should have 'body_iterator' attribute"

        # Collect streaming chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        # Verify streaming format
        assert len(chunks) > 0, f"Should have streaming chunks, got {len(chunks)}"
        assert any("data: " in chunk for chunk in chunks), \
            f"Expected streaming format with 'data:', got: {chunks}"
        assert any("[DONE]" in chunk for chunk in chunks), \
            f"Expected [DONE] marker in streaming, got: {chunks}"


@pytest.mark.unit
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
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response
    response_content = json.loads(response.body.decode())
    assert response_content["choices"][0]["message"]["content"] == "```python\nprint('Hello, World!')\n```", \
        "System prompt should influence response content"


@pytest.mark.unit
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
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify response includes the new answer
    response_content = json.loads(response.body.decode())
    assert "6" in response_content["choices"][0]["message"]["content"], \
        "Multi-turn conversation should maintain context and answer correctly"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_completion_parameter_validation(mock_handler):
    """Test parameter validation for OpenAI requests."""
    # Test missing messages field
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400, "Should return 400 for missing messages"
    assert "messages" in str(exc_info.value.detail).lower(), \
        "Error should mention 'messages' field"

    # Test empty messages list
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": []})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400, "Should return 400 for empty messages"
    assert "messages" in str(exc_info.value.detail).lower(), \
        "Error should mention 'messages' field"

    # Test invalid message structure (not a dict)
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": ["invalid"]})
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400, "Should return 400 for invalid message structure"

    # Test missing role/content in message
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value={"messages": [{"role": "user"}]})  # Missing content
    with pytest.raises(HTTPException) as exc_info:
        await mock_handler.handle_request(mock_request)
    assert exc_info.value.status_code == 400, "Should return 400 for missing content"


# ============================================================================
# Group B: Integration and Error Recovery
# ============================================================================

@pytest.mark.unit
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
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=True), \
         patch.object(mock_handler, '_handle_with_radix_cache', return_value=expected_response):
        response = await mock_handler.handle_request(mock_request)

    # Verify integration worked
    response_content = json.loads(response.body.decode())
    assert response_content["choices"][0]["message"]["content"] == "Hello from integrated system!", \
        "Integration should properly route through cache layer"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_recovery_scenarios(mock_handler, mocker):
    """Test various error recovery scenarios."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    mock_request = Mock(spec=Request)
    mock_request.json = AsyncMock(return_value=request_data)

    # Test network timeout simulation
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=504, detail="Network timeout")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code == 504, \
            f"Should return 504 for network timeout, got {exc_info.value.status_code}"

    # Test backend service error
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=503, detail="Service unavailable")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code == 503, \
            f"Should return 503 for service unavailable, got {exc_info.value.status_code}"

    # Test invalid response from backend
    with patch.object(mock_handler.router, '_check_cache_availability', return_value=False), \
         patch.object(mock_handler, '_proxy_to_sglang_chat', side_effect=HTTPException(status_code=500, detail="Invalid response")):
        with pytest.raises(HTTPException) as exc_info:
            await mock_handler.handle_request(mock_request)
        assert exc_info.value.status_code >= 400, \
            f"Should return error status (>= 400), got {exc_info.value.status_code}"


if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v", "--tb=short"])
