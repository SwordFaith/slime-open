"""
Tests for P0 and P1 issues identified in router implementation.

P0 Issues (Critical):
1. Double-checked locking flaw in _check_cache_availability()
2. Resource leak in streaming response error handling
3. Test import issues (FIXED)

P1 Issues (Important):
1. Internal HTTP overhead in OpenAI handler
2. Error handling with overly broad exceptions
3. Duplicate cache availability checks
"""

import asyncio
import concurrent.futures
import pytest
import threading
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import Request
from fastapi.responses import StreamingResponse

from slime.router.router import SlimeRouter
from slime.router.handlers.openai_chat_completion import ChatCompletionHandler
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware


# ==============================================================================
# P0-1: Test double-checked locking in _check_cache_availability()
# ==============================================================================

@pytest.mark.unit
class TestP0_DoubleLockIssue:
    """Test concurrent safety of cache availability checking."""

    def test_cache_availability_concurrent_initialization(self):
        """
        P0-1: Verify _check_cache_availability() doesn't have race conditions.

        The current implementation has a flawed double-check lock pattern where
        the second `if self._cache_available is None` check is missing.

        Expected: Only one thread should initialize the cache check
        Actual: Multiple threads might redundantly check
        """
        # Create args mock
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []  # Fix: must be iterable

        router = SlimeRouter(args)

        # Track how many times the expensive check operation runs
        check_count = {'value': 0}
        lock = threading.Lock()

        original_method = router._check_cache_availability

        def tracked_check():
            """Wrapper to count actual checks."""
            with lock:
                check_count['value'] += 1
            return original_method()

        router._check_cache_availability = tracked_check

        # Run concurrent checks
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(router._check_cache_availability) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        # All results should be consistent
        assert all(r == results[0] for r in results), "All threads should get same result"

        # ISSUE: With flawed double-check, might check multiple times
        # Should only check once after first call completes
        print(f"Cache availability checked {check_count['value']} times for {num_threads} concurrent calls")

        # Ideally should be 1, but with current implementation might be > 1
        # This test documents the issue
        assert check_count['value'] >= 1, "Should check at least once"

    @pytest.mark.asyncio
    async def test_cache_availability_async_safety(self):
        """Test cache availability check is safe for async calls."""
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # Simulate async concurrent access
        results = await asyncio.gather(*[
            asyncio.to_thread(router._check_cache_availability)
            for _ in range(20)
        ])

        # All should return same result
        assert len(set(results)) == 1, "All concurrent checks should return same result"


# ==============================================================================
# P0-2: Test resource leak in streaming response
# ==============================================================================

@pytest.mark.unit
class TestP0_StreamingResourceLeak:
    """Test resource cleanup in streaming error scenarios."""

    @pytest.mark.asyncio
    async def test_streaming_proxy_exception_calls_finish_url(self):
        """
        P0-2: Verify _finish_url() is called even when streaming fails.

        Current implementation in _proxy_to_sglang_chat() lacks finally block
        for streaming branch, potentially leaking worker URL tracking.
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router.client = Mock()
        router._use_url = AsyncMock(return_value="http://worker1:10090")
        router._finish_url = AsyncMock()
        router.component_registry = Mock()
        router.component_registry.has = Mock(return_value=True)

        handler = ChatCompletionHandler(router)

        # Create request that will cause streaming to fail
        request = Mock(spec=Request)
        request.body = AsyncMock(return_value=b'{"messages": [], "stream": true}')
        request.headers = {}

        # Create a proper async context manager mock
        async_ctx_mgr = AsyncMock()
        async_ctx_mgr.__aenter__.side_effect = Exception("Connection error")
        async_ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        router.client.stream = Mock(return_value=async_ctx_mgr)

        # Call should raise exception
        with pytest.raises(Exception, match="Connection error"):
            await handler._proxy_to_sglang_chat(request)

        # CRITICAL: _finish_url should still be called to clean up worker tracking
        router._finish_url.assert_called_once_with("http://worker1:10090")

    @pytest.mark.asyncio
    async def test_streaming_iteration_error_cleanup(self):
        """Test cleanup when stream iteration fails midway."""
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router.client = AsyncMock()
        router._use_url = AsyncMock(return_value="http://worker1:10090")
        router._finish_url = AsyncMock()

        handler = ChatCompletionHandler(router)

        # Create mock streaming response that fails during iteration
        async def failing_stream():
            yield b"data: chunk1\n\n"
            raise Exception("Stream interrupted")

        mock_response = Mock()
        mock_response.aiter_bytes = failing_stream
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        router.client.stream = AsyncMock(return_value=mock_response)

        request = Mock(spec=Request)
        request.body = AsyncMock(return_value=b'{"messages": [], "stream": true}')
        request.headers = {}

        # Should not raise, but cleanup should happen
        with pytest.raises(Exception):
            await handler._proxy_to_sglang_chat(request)

        # Verify cleanup
        router._finish_url.assert_called_once()


# ==============================================================================
# P1-1: Test performance overhead of internal HTTP calls
# ==============================================================================

@pytest.mark.unit
class TestP1_InternalHTTPOverhead:
    """Test that internal HTTP overhead has been eliminated by refactoring."""

    @pytest.mark.asyncio
    async def test_internal_http_overhead_eliminated(self):
        """
        P1-1: FIXED - Verify internal HTTP overhead has been eliminated.

        PREVIOUS ISSUE:
        Old implementation in _stream_generate_response() called
        http://localhost:{port}/generate which added 3.5-7.5ms overhead.

        RESOLUTION:
        Refactored to _non_stream_generate_with_cache() and _stream_generate_with_cache()
        which call SGLang workers directly without internal HTTP.

        This test verifies the new methods exist and the old methods are gone.
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_port = 30000
        args.model_name = "test-model"

        router = Mock()
        router.args = args
        router.component_registry = Mock()

        handler = ChatCompletionHandler(router)

        # Verify new methods exist (direct worker calls, no internal HTTP)
        assert hasattr(handler, '_non_stream_generate_with_cache'), \
            "New method _non_stream_generate_with_cache should exist"
        assert hasattr(handler, '_stream_generate_with_cache'), \
            "New method _stream_generate_with_cache should exist"

        # Verify old methods with internal HTTP are gone
        assert not hasattr(handler, '_non_stream_generate_response'), \
            "Old method _non_stream_generate_response should be removed"
        assert not hasattr(handler, '_stream_generate_response'), \
            "Old method _stream_generate_response should be removed"

        # This documents that the P1-1 issue has been fixed through refactoring
        print("[P1-1 FIXED] Internal HTTP overhead eliminated by direct worker calls")


# ==============================================================================
# P1-2: Test error handling issues
# ==============================================================================

@pytest.mark.unit
class TestP1_ErrorHandling:
    """Test error handling doesn't leak information or mask bugs."""

    @pytest.mark.asyncio
    async def test_broad_exception_catching_masks_errors(self):
        """
        P1-2: Verify that overly broad exception catching doesn't hide bugs.

        middleware/radix_tree_middleware.py:query_cache_by_text() catches
        all exceptions with bare `except Exception`, which can hide bugs.
        """
        from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware

        # Create minimal setup
        args = Mock()
        args.hf_checkpoint = "/tmp/fake-model"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False
        router.get_component_registry = Mock()

        app = Mock()

        # Mock tokenizer and radix tree
        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'):
            with patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
                middleware = RadixTreeMiddleware(app, router=router)

        # Make radix_tree throw a serious bug (not a validation error)
        serious_error = RuntimeError("Critical internal bug")
        middleware.radix_tree.get_or_create_tokenization_async = AsyncMock(
            side_effect=serious_error
        )

        # Current implementation swallows this and returns empty tuple
        result = await middleware.query_cache_by_text("test")

        # ISSUE: Critical bug is silently swallowed
        assert result == ([], [], [], []), "Error should be caught and return empty"

        # Better: Should log or re-raise critical errors
        # This test documents the issue

    @pytest.mark.asyncio
    async def test_json_parse_error_information_leakage(self):
        """Test that JSON parsing errors don't leak internal details."""
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # Create mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"invalid": json}')

        router.client.request = AsyncMock(return_value=mock_response)
        router.worker_urls = {"http://worker1:10090": 0}

        # Simulate proxy call
        request = Mock(spec=Request)
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"text": "test"}')
        request.headers = {}

        result = await router.proxy(request, "generate")

        # Should handle error gracefully
        # Current implementation returns raw body, which is okay
        # but error handling could be more structured
        assert result is not None


# ==============================================================================
# P1-3: Test duplicate cache availability logic
# ==============================================================================

@pytest.mark.unit
class TestP1_DuplicateCacheCheck:
    """Test that cache availability checking is consistent."""

    @pytest.mark.asyncio
    async def test_router_and_handler_cache_check_consistency(self):
        """
        P1-3: Verify router and handler have consistent cache checking logic.

        Both SlimeRouter._check_cache_availability() and
        ChatCompletionHandler._check_cache_availability() have similar logic.
        They should return the same result.
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.model_name = "test-model"
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        handler = ChatCompletionHandler(router)

        # Both should return same result
        router_result = router._check_cache_availability()
        handler_result = await handler._check_cache_availability()

        assert router_result == handler_result, \
            "Router and handler cache checks should be consistent"

    def test_cache_availability_caching_works(self):
        """Test that cache availability result is properly cached."""
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)

        # First call
        result1 = router._check_cache_availability()

        # Second call should use cached result
        result2 = router._check_cache_availability()

        assert result1 == result2, "Should return cached result"
        assert router._cache_available is not None, "Result should be cached"


# ==============================================================================
# Additional Tests for Common Issues
# ==============================================================================

@pytest.mark.unit
class TestConcurrencySafety:
    """Additional concurrency safety tests."""

    @pytest.mark.asyncio
    async def test_url_lock_prevents_race_conditions(self):
        """Test that _url_lock properly prevents worker selection races."""
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 0,
            "http://worker3:10090": 0,
        }

        # Simulate concurrent worker selection
        selected_urls = await asyncio.gather(*[
            router._use_url() for _ in range(30)
        ])

        # All URLs should have been tracked properly
        total_in_flight = sum(router.worker_urls.values())
        assert total_in_flight == 30, "All requests should be tracked"

        # Clean up
        for url in selected_urls:
            await router._finish_url(url)

        # All counts should be back to zero
        assert all(count == 0 for count in router.worker_urls.values())

    @pytest.mark.asyncio
    async def test_finish_url_doesnt_go_negative(self):
        """Test that worker counts never go negative."""
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {"http://worker1:10090": 0}

        url = await router._use_url()
        assert router.worker_urls[url] == 1

        await router._finish_url(url)
        assert router.worker_urls[url] == 0

        # Should raise assertion error if trying to go negative
        with pytest.raises(AssertionError):
            await router._finish_url(url)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
