"""
OpenAI Chat Completion API E2E tests.

This test suite validates the complete end-to-end functionality of
Chat Completion API with real router integration.

Tests cover:
- Full HTTP request/response cycle
- OpenAI SDK compatibility
- Streaming and non-streaming modes
- Error handling at HTTP level
- Performance benchmarking
"""

import json
import asyncio
from typing import Dict, Any
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
import httpx

from slime.router.router import SlimeRouter


class TestOpenAICompatibility:
    """Test OpenAI SDK compatibility."""

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

    @pytest.fixture
    def router_client(self, mock_router_args):
        """Create FastAPI test client with router."""
        router = SlimeRouter(mock_router_args)
        return TestClient(router.app)

    def test_openai_sdk_compatibility(self, router_client):
        """Test compatibility with OpenAI SDK format."""
        # Simulate OpenAI SDK request
        request_data = {
            "model": "slime-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        response = router_client.post("/v1/chat/completions", json=request_data)

        # Verify response format matches OpenAI API
        assert response.status_code == 200

        response_data = response.json()
        assert "id" in response_data
        assert response_data["object"] == "chat.completion"
        assert "created" in response_data
        assert response_data["model"] == "slime-model"
        assert "choices" in response_data
        assert len(response_data["choices"]) == 1

        choice = response_data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert "finish_reason" in choice

        assert "usage" in response_data
        usage = response_data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_streaming_sdk_compatibility(self, router_client):
        """Test streaming response compatibility with OpenAI SDK."""
        request_data = {
            "model": "slime-model",
            "messages": [
                {"role": "user", "content": "Tell me a story"}
            ],
            "stream": True,
            "max_tokens": 50
        }

        response = router_client.post("/v1/chat/completions", json=request_data)

        # Verify streaming response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Parse streaming chunks
        content = response.content.decode()
        chunks = content.split("\n\n")

        # Should have multiple chunks
        assert len(chunks) > 1

        # Verify SSE format
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                assert chunk.startswith("data: ")
                chunk_data = chunk[6:]  # Remove "data: "

                if chunk_data != "[DONE]":  # Skip end marker if present
                    data = json.loads(chunk_data)
                    assert "id" in data
                    assert data["object"] == "chat.completion.chunk"
                    assert "choices" in data
                    assert len(data["choices"]) == 1

    def test_error_handling(self, router_client):
        """Test error handling with invalid requests."""
        # Missing required fields
        invalid_request = {
            "messages": []  # Empty messages
        }

        response = router_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code in [400, 422]  # Bad Request or Validation Error

        # Invalid parameters
        invalid_request = {
            "model": "slime-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 5.0  # Invalid temperature range
        }

        response = router_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 400

    def test_concurrent_requests(self, router_client):
        """Test handling of concurrent requests."""
        import threading
        import time

        results = []
        errors = []

        def make_request(request_id):
            try:
                request_data = {
                    "model": "slime-model",
                    "messages": [{"role": "user", "content": f"Hello {request_id}!"}]
                }
                response = router_client.post("/v1/chat/completions", json=request_data)
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    errors.append(f"Request {request_id} failed: {response.status_code}")
            except Exception as e:
                errors.append(f"Request {request_id} error: {e}")

        # Create multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # Verify all requests succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify responses are unique
        response_ids = [r["id"] for r in results]
        assert len(set(response_ids)) == 5  # All IDs should be unique


class TestPerformanceBenchmark:
    """Test performance benchmarks."""

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
        args.enable_openai_chat_completion = True
        return args

    @pytest.fixture
    def router_client(self, mock_router_args):
        """Create FastAPI test client."""
        router = SlimeRouter(mock_router_args)
        return TestClient(router.app)

    @pytest.mark.performance
    def test_response_time_benchmark(self, router_client):
        """Test response time is within acceptable limits."""
        import time

        request_data = {
            "model": "slime-model",
            "messages": [{"role": "user", "content": "Hello!"}]
        }

        start_time = time.time()
        response = router_client.post("/v1/chat/completions", json=request_data)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within 1 second (generous for test environment)
        assert response_time < 1.0, f"Response time {response_time}s exceeds limit"
        assert response.status_code == 200

    @pytest.mark.performance
    def test_throughput_benchmark(self, router_client):
        """Test throughput with multiple requests."""
        import time
        import threading

        request_count = 10
        completed_requests = []
        start_time = time.time()

        def make_request():
            request_data = {
                "model": "slime-model",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
            response = router_client.post("/v1/chat/completions", json=request_data)
            if response.status_code == 200:
                completed_requests.append(response.json())

        # Create concurrent requests
        threads = []
        for _ in range(request_count):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all requests completed
        assert len(completed_requests) == request_count

        # Calculate throughput
        throughput = request_count / total_time
        print(f"Throughput: {throughput:.2f} requests/second")

        # Should handle at least 2 requests per second (conservative)
        assert throughput > 2.0

    @pytest.mark.performance
    def test_memory_usage_benchmark(self, router_client):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make multiple requests
        for i in range(50):
            request_data = {
                "model": "slime-model",
                "messages": [{"role": "user", "content": f"Request {i}"}]
            }
            response = router_client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase {memory_increase/1024/1024:.2f}MB exceeds limit"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

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
        args.enable_openai_chat_completion = True
        return args

    @pytest.fixture
    def router_client(self, mock_router_args):
        """Create FastAPI test client."""
        router = SlimeRouter(mock_router_args)
        return TestClient(router.app)

    def test_empty_request_body(self, router_client):
        """Test handling of empty request body."""
        response = router_client.post("/v1/chat/completions", data="")
        assert response.status_code in [400, 422]

    def test_malformed_json(self, router_client):
        """Test handling of malformed JSON."""
        response = router_client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_very_long_message(self, router_client):
        """Test handling of very long messages."""
        long_content = "Hello " * 10000  # Long message

        request_data = {
            "model": "slime-model",
            "messages": [{"role": "user", "content": long_content}]
        }

        response = router_client.post("/v1/chat/completions", json=request_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 413, 429]  # 413 Payload Too Large or 429 Too Many Requests

    def test_unicode_content(self, router_client):
        """Test handling of Unicode content."""
        unicode_content = "Hello 世界 🌍 Test with émojis"

        request_data = {
            "model": "slime-model",
            "messages": [{"role": "user", "content": unicode_content}]
        }

        response = router_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        response_data = response.json()
        assert "choices" in response_data
        assert len(response_data["choices"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])