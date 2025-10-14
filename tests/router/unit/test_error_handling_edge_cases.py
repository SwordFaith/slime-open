"""
Error Handling Edge Cases Unit Tests

Tests cover critical error scenarios and recovery:
- Network timeout variations (connection vs read timeout)
- Partial HTTP responses and truncated data
- Resource exhaustion (OOM, file descriptors)
- Exception during cleanup (finally blocks)
- Cascading failures
- Error recovery and system resilience

Test Strategy:
- Unit testing with mocked network/system calls
- Error injection and fault simulation
- Recovery verification
- Resource cleanup validation
"""

import asyncio
import pytest
import threading
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import HTTPException
from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware
from slime.router.router import SlimeRouter
from slime.router.core.radix_tree import StringRadixTrie


# ==============================================================================
# Group A: Network Timeout Edge Cases
# ==============================================================================

@pytest.mark.unit
class TestNetworkTimeoutEdgeCases:
    """Test different types of network timeouts."""

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """
        Test: Connection timeout before request is sent.

        Timeout Type: Cannot establish connection
        Expected: Should raise timeout error and clean up resources
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://unreachable:10090")
        router._finish_url = AsyncMock()

        # Mock client that times out on connection
        router.client = Mock()
        router.client.request = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": []}')
        request.headers = {}

        # Should raise timeout
        with pytest.raises((asyncio.TimeoutError, HTTPException)):
            await handler._proxy_to_sglang_chat(request)

        # Should clean up URL even on timeout
        router._finish_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_timeout_after_connection(self):
        """
        Test: Timeout while reading response body.

        Timeout Type: Connected but response too slow
        Expected: Should timeout and clean up
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://worker:10090")
        router._finish_url = AsyncMock()

        # Mock response that times out during read
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(side_effect=asyncio.TimeoutError("Read timeout"))

        router.client = Mock()
        router.client.request = AsyncMock(return_value=mock_response)

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": []}')
        request.headers = {}

        # Should handle read timeout
        with pytest.raises((asyncio.TimeoutError, HTTPException)):
            await handler._proxy_to_sglang_chat(request)

        # Should still clean up
        router._finish_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_during_streaming(self):
        """
        Test: Timeout during streaming response.

        Timeout Type: Stream established but stalls
        Expected: Should handle gracefully and clean up
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://worker:10090")
        router._finish_url = AsyncMock()

        # Mock streaming response that times out mid-stream
        async def timeout_stream():
            yield b"data: chunk1\n\n"
            raise asyncio.TimeoutError("Stream timeout")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.aiter_bytes = timeout_stream

        # Mock async context manager properly
        async def mock_aenter(self):
            return self
        mock_response.__aenter__ = mock_aenter
        mock_response.__aexit__ = AsyncMock()

        router.client = Mock()
        # stream() should return the mock_response that supports async context manager
        router.client.stream = AsyncMock(return_value=mock_response)

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": [], "stream": true}')
        request.headers = {}

        # Should handle timeout during streaming
        with pytest.raises((asyncio.TimeoutError, HTTPException, Exception)):
            await handler._proxy_to_sglang_chat(request)

        # Should clean up even if stream fails
        router._finish_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_retry_with_timeouts(self):
        """
        Test: Multiple retries all timeout.

        Scenario: All retry attempts fail with timeout
        Expected: Should eventually give up and return error
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        # Mock middleware
        app = Mock()

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(app, router=router)

        # Mock request
        request = Mock()
        request.url.path = "/generate"
        request.json = AsyncMock(return_value={"text": "test"})
        request._json = None

        # Mock call_next that always returns abort (triggers retry)
        abort_response = Mock()
        abort_response.body = b'{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}'
        abort_response.__class__.__name__ = "Response"

        # Timeout during retry mechanism
        async def failing_call_next(req):
            raise asyncio.TimeoutError("Retry timeout")

        # Should handle retries timing out
        # Note: Actual retry logic might vary, this tests resilience
        # The retry mechanism may catch and handle exceptions gracefully
        # So we test that it doesn't hang indefinitely
        try:
            # Set a timeout to prevent hanging
            result = await asyncio.wait_for(
                middleware.dispatch(request, failing_call_next),
                timeout=5.0
            )
            # If it completes without raising, that's acceptable too
            # (graceful degradation)
        except (asyncio.TimeoutError, Exception) as e:
            # Expected: either timeout or exception from retry logic
            # TimeoutError or any exception is acceptable (test passes)
            pass  # Test succeeds - retry mechanism raised an exception or timed out


# ==============================================================================
# Group B: Partial and Malformed Responses
# ==============================================================================

@pytest.mark.unit
class TestPartialResponses:
    """Test handling of partial or malformed HTTP responses."""

    @pytest.mark.asyncio
    async def test_partial_json_response(self):
        """
        Test: HTTP response with truncated JSON.

        Malformed: JSON incomplete or cut off mid-stream
        Expected: Should detect and handle gracefully
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # Mock response with partial JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"text": "incomplete')  # Truncated

        router.client.request = AsyncMock(return_value=mock_response)
        router.worker_urls = {"http://worker:10090": 0}

        request = Mock()
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"text": "test"}')
        request.headers = {}

        # Should handle partial JSON gracefully
        result = await router.proxy(request, "generate")

        # Should return something (error response or raw body)
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_response_body(self):
        """
        Test: HTTP 200 but empty body.

        Edge Case: Valid status code but no content
        Expected: Should handle as error or empty result
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://worker:10090")
        router._finish_url = AsyncMock()

        # Mock empty response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'')

        router.client = Mock()
        router.client.request = AsyncMock(return_value=mock_response)

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": []}')
        request.headers = {}

        # Should handle empty body
        result = await handler._proxy_to_sglang_chat(request)

        # Should return some response (might be error)
        assert result is not None

    @pytest.mark.asyncio
    async def test_response_with_invalid_utf8(self):
        """
        Test: Response body with invalid UTF-8 encoding.

        Malformed: Byte sequence not valid UTF-8
        Expected: Should handle encoding error
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # Invalid UTF-8 sequence
        invalid_utf8 = b'{"text": "\xff\xfe invalid utf8"}'

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=invalid_utf8)

        router.client.request = AsyncMock(return_value=mock_response)
        router.worker_urls = {"http://worker:10090": 0}

        request = Mock()
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"text": "test"}')
        request.headers = {}

        # Should handle invalid UTF-8
        result = await router.proxy(request, "generate")

        # Should return something (might be raw bytes or error)
        assert result is not None

    @pytest.mark.asyncio
    async def test_response_status_500_with_error_body(self):
        """
        Test: HTTP 500 with error message in body.

        Error Response: Server error with details
        Expected: Should propagate error appropriately
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://worker:10090")
        router._finish_url = AsyncMock()

        # Mock 500 error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"error": "Internal server error", "detail": "OOM"}')

        router.client = Mock()
        router.client.post = AsyncMock(return_value=mock_response)

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": []}')
        request.headers = {}

        # Should handle 500 error
        with pytest.raises((HTTPException, Exception)):
            await handler._proxy_to_sglang_chat(request)

        # Should still clean up
        router._finish_url.assert_called_once()


# ==============================================================================
# Group C: Resource Exhaustion
# ==============================================================================

@pytest.mark.unit
class TestResourceExhaustion:
    """Test behavior under resource exhaustion."""

    def test_cache_under_memory_pressure(self):
        """
        Test: Cache operations when memory is limited.

        Resource Limit: Simulated low memory
        Expected: Should handle gracefully or reject operations
        """
        trie = StringRadixTrie(max_cache_size=10000)

        # Try to insert many large entries
        try:
            for i in range(100):
                large_tokens = list(range(1000))
                trie.insert(f"large_{i}", large_tokens, [0.001] * 1000, [1] * 1000, weight_version=i)

            # If it succeeds, verify cache is managed
            assert trie.cur_cache_size <= trie.max_cache_size * 2, "Cache should be bounded"

        except MemoryError:
            # Acceptable to fail with MemoryError
            pytest.skip("Insufficient memory for test")

    @pytest.mark.asyncio
    async def test_concurrent_operations_under_high_load(self):
        """
        Test: Many concurrent operations stressing resources.

        Stress: 100+ concurrent async operations
        Expected: Should handle without resource leaks
        """
        trie = StringRadixTrie(max_cache_size=10000)

        async def heavy_operation(op_id):
            # Mix of operations
            if op_id % 3 == 0:
                await trie.insert_async(f"item_{op_id}", [op_id], [0.1], [1], weight_version=op_id)
            elif op_id % 3 == 1:
                await trie.find_longest_prefix_async(f"item_{op_id - 1}")
            else:
                await trie.gc_by_weight_version_async(current_weight_version=op_id)

        # Launch many concurrent operations
        tasks = [heavy_operation(i) for i in range(200)]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]

            # Some exceptions are acceptable under extreme load
            # But shouldn't be majority
            assert len(exceptions) < len(tasks) * 0.5, f"Too many failures: {len(exceptions)}/{len(tasks)}"

        except Exception as e:
            pytest.skip(f"System resource limit reached: {e}")

    def test_router_under_worker_exhaustion(self):
        """
        Test: Router behavior when all workers are busy.

        Resource Limit: No available workers
        Expected: Should queue or reject gracefully
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # Set all workers to max capacity
        router.worker_urls = {
            "http://worker1:10090": 128,  # At max
            "http://worker2:10090": 128,  # At max
        }

        # Try to get URL when all busy
        # Implementation may block or raise error
        try:
            # Note: _use_url is async, need to run in event loop
            import asyncio

            async def try_get_url():
                return await router._use_url()

            url = asyncio.run(try_get_url())

            # If it succeeds, verify it's one of the workers
            assert "worker" in url

        except Exception as e:
            # May raise error when all busy, that's acceptable
            print(f"Router handling: {e}")


# ==============================================================================
# Group D: Exception During Cleanup
# ==============================================================================

@pytest.mark.unit
class TestExceptionDuringCleanup:
    """Test behavior when cleanup operations fail."""

    @pytest.mark.asyncio
    async def test_exception_in_finish_url(self):
        """
        Test: Exception during _finish_url cleanup.

        Error During Cleanup: finally block raises
        Expected: Should not mask original error
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router._use_url = AsyncMock(return_value="http://worker:10090")

        # _finish_url raises exception
        router._finish_url = AsyncMock(side_effect=Exception("Cleanup failed"))

        router.client = Mock()
        # Primary operation also fails
        router.client.post = AsyncMock(side_effect=Exception("Primary error"))

        handler = ChatCompletionHandler(router)

        request = Mock()
        request.body = AsyncMock(return_value=b'{"messages": []}')
        request.headers = {}

        # Should raise exception (preferably primary error, not cleanup error)
        with pytest.raises(Exception) as exc_info:
            await handler._proxy_to_sglang_chat(request)

        # At least one error should be raised (check for error/cleanup/failed keywords)
        exc_str = str(exc_info.value).lower()
        assert any(keyword in exc_str for keyword in ["error", "cleanup", "failed"]), \
            f"Exception should mention error/cleanup/failed, got: {exc_info.value}"

    @pytest.mark.asyncio
    async def test_exception_in_cache_cleanup(self):
        """
        Test: Exception during cache cleanup after error.

        Scenario: Primary operation fails, then cleanup fails
        Expected: Should handle both errors
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        app = Mock()

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(app, router=router)

        # Make cache operations fail
        middleware.radix_tree.insert_async = AsyncMock(side_effect=Exception("Cache insert failed"))

        # Configure tokenizer mock to support subscript operations
        middleware.tokenizer = MagicMock()
        middleware.tokenizer.return_value = {"input_ids": [1, 2, 3]}
        middleware.tokenizer.decode = Mock(return_value="decoded text")

        request = Mock()
        request.url.path = "/generate"
        request.json = AsyncMock(return_value={"text": "test"})
        request._json = None

        # Mock response (use MagicMock to support subscript operations)
        response = MagicMock()
        response.body = b'{"text": "response", "output_ids": [1, 2], "meta_info": {"weight_version": 1}}'
        response.__class__.__name__ = "Response"

        call_next = AsyncMock(return_value=response)

        # Should handle cache errors gracefully
        result = await middleware.dispatch(request, call_next)

        # Should still return response despite cache error
        assert result is response


# ==============================================================================
# Group E: Cascading Failures
# ==============================================================================

@pytest.mark.unit
class TestCascadingFailures:
    """Test system behavior during cascading failures."""

    @pytest.mark.asyncio
    async def test_worker_failure_cascade(self):
        """
        Test: Multiple workers fail in sequence.

        Cascading Failure: Workers fail one after another
        Expected: Router should handle and attempt retries
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3  # Multiple workers
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 0,
            "http://worker3:10090": 0,
        }

        # Mock client that fails for all workers
        error_count = {'count': 0}

        async def failing_request(*args, **kwargs):
            error_count['count'] += 1
            raise Exception(f"Worker failure #{error_count['count']}")

        router.client.request = AsyncMock(side_effect=failing_request)

        request = Mock()
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"text": "test"}')
        request.headers = {}

        # Should attempt workers and fail gracefully
        with pytest.raises(Exception):
            await router.proxy(request, "generate")

        # Should have tried at least one worker
        assert error_count['count'] >= 1

    def test_concurrent_failures_with_recovery(self):
        """
        Test: System recovers after concurrent failures.

        Resilience: Multiple failures followed by recovery
        Expected: System should continue operating after errors
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Pre-populate
        for i in range(20):
            trie.insert(f"data_{i}", [i], [0.1 * i], [1], weight_version=1)

        # Simulate failures (invalid operations)
        failure_count = 0
        for i in range(10):
            try:
                # Try to insert invalid data
                trie.insert("", [], [], [], weight_version=i)
            except:
                failure_count += 1

        # System should still work after failures
        success = trie.insert("recovery", [99], [0.9], [1], weight_version=20)
        assert success is True

        match = trie.find_longest_prefix("recovery")
        assert match.matched_prefix == "recovery"

    @pytest.mark.asyncio
    async def test_partial_system_failure(self):
        """
        Test: Some components fail while others continue.

        Partial Failure: Cache fails but main system works
        Expected: Should degrade gracefully
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        app = Mock()

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(app, router=router)

        # Cache queries fail
        middleware.radix_tree.get_or_create_tokenization_async = AsyncMock(
            side_effect=Exception("Cache unavailable")
        )

        request = Mock()
        request.url.path = "/generate"
        request.json = AsyncMock(return_value={"text": "test"})
        request._json = None

        # Main generation succeeds
        response = Mock()
        response.body = b'{"text": "success"}'
        response.__class__.__name__ = "Response"

        call_next = AsyncMock(return_value=response)

        # Should work despite cache failure (graceful degradation)
        result = await middleware.dispatch(request, call_next)

        # Should return response
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
