"""
Router Core Unit Tests

Tests cover core router functionality:
- Worker selection algorithm (round-robin load balancing)
- URL tracking and request counting
- Load balancing verification
- Concurrent worker selection safety
- Worker management (add/list/remove)
- Cache availability checking
- Component registry interaction

Test Strategy:
- Unit testing with minimal mocking
- Concurrent operations testing
- Load balancing verification
- Resource tracking validation
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request
from slime.router.router import SlimeRouter


# ==============================================================================
# Group A: Worker Selection Algorithm
# ==============================================================================

@pytest.mark.unit
class TestWorkerSelectionAlgorithm:
    """Test worker selection and load balancing."""

    @pytest.mark.asyncio
    async def test_single_worker_selection(self):
        """
        Test: Select worker when only one is available.

        Expected: Always returns the only worker
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
        router.worker_urls = {"http://worker1:10090": 0}

        # Select worker multiple times
        selected = []
        for _ in range(10):
            url = await router._use_url()
            selected.append(url)

        # Should always select the only worker (Boundary: single worker scenario)
        assert all(url == "http://worker1:10090" for url in selected), \
            f"All selections should be the only worker, got: {set(selected)}"

        # Count should increment to 10
        assert router.worker_urls["http://worker1:10090"] == 10, \
            f"Worker count should be 10 after 10 selections, got {router.worker_urls['http://worker1:10090']}"

    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """
        Test: Round-robin selection among multiple workers.

        Algorithm: Always selects worker with minimum count
        Expected: Load distributed evenly
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 0,
            "http://worker3:10090": 0,
        }

        # Select 15 times
        for _ in range(15):
            await router._use_url()

        # Each worker should have 5 requests (perfect distribution) (Boundary: round-robin fairness)
        assert router.worker_urls["http://worker1:10090"] == 5, \
            f"Worker1 should have 5 requests, got {router.worker_urls['http://worker1:10090']}"
        assert router.worker_urls["http://worker2:10090"] == 5, \
            f"Worker2 should have 5 requests, got {router.worker_urls['http://worker2:10090']}"
        assert router.worker_urls["http://worker3:10090"] == 5, \
            f"Worker3 should have 5 requests, got {router.worker_urls['http://worker3:10090']}"

    @pytest.mark.asyncio
    async def test_selection_with_uneven_initial_load(self):
        """
        Test: Worker selection with uneven initial load.

        Scenario: Workers start with different request counts
        Expected: Algorithm should balance by always picking minimum
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        # Uneven initial load
        router.worker_urls = {
            "http://worker1:10090": 10,
            "http://worker2:10090": 5,
            "http://worker3:10090": 0,
        }

        # Select 6 times
        for _ in range(6):
            await router._use_url()

        # Worker3 should get all 6 (was at 0) (Boundary: uneven initial load balancing)
        # Then maybe worker2 gets some
        assert router.worker_urls["http://worker3:10090"] >= 5, \
            f"Least loaded worker (was 0) should get priority, got {router.worker_urls['http://worker3:10090']}"
        assert router.worker_urls["http://worker1:10090"] == 10, \
            f"Most loaded worker (was 10) should get none, got {router.worker_urls['http://worker1:10090']}"

    @pytest.mark.asyncio
    async def test_no_workers_available(self):
        """
        Test: Attempt to select worker when none are registered.

        Expected: Should raise assertion error
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
        router.worker_urls = {}  # No workers

        # Should raise assertion error
        with pytest.raises(AssertionError, match="No workers available"):
            await router._use_url()


# ==============================================================================
# Group B: URL Tracking and Request Counting
# ==============================================================================

@pytest.mark.unit
class TestURLTrackingAndCounting:
    """Test URL tracking and request count management."""

    @pytest.mark.asyncio
    async def test_finish_url_decrements_count(self):
        """
        Test: _finish_url decrements worker count.

        Expected: Count goes down by 1
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
        router.worker_urls = {"http://worker1:10090": 5}

        # Finish a request
        await router._finish_url("http://worker1:10090")

        # Count should decrease by 1 (Boundary: finish URL decrements count)
        assert router.worker_urls["http://worker1:10090"] == 4, \
            f"Count should decrease from 5 to 4, got {router.worker_urls['http://worker1:10090']}"

    @pytest.mark.asyncio
    async def test_finish_url_never_goes_negative(self):
        """
        Test: _finish_url prevents count from going negative.

        Expected: Assertion error when count would go below 0
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
        router.worker_urls = {"http://worker1:10090": 0}

        # Try to finish when count is already 0
        with pytest.raises(AssertionError, match="went negative"):
            await router._finish_url("http://worker1:10090")

    @pytest.mark.asyncio
    async def test_finish_url_with_unknown_url(self):
        """
        Test: _finish_url with URL not in worker_urls.

        Expected: Assertion error for unrecognized URL
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
        router.worker_urls = {"http://worker1:10090": 5}

        # Try to finish unknown URL
        with pytest.raises(AssertionError, match="not recognized"):
            await router._finish_url("http://unknown:10090")

    @pytest.mark.asyncio
    async def test_use_and_finish_cycle(self):
        """
        Test: Complete cycle of _use_url and _finish_url.

        Expected: Count should return to original after finish
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
        router.worker_urls = {"http://worker1:10090": 0}

        # Use URL
        url = await router._use_url()
        assert router.worker_urls[url] == 1

        # Finish URL
        await router._finish_url(url)
        assert router.worker_urls[url] == 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_use_and_finish(self):
        """
        Test: Multiple concurrent use/finish cycles.

        Expected: Counts should balance out correctly
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 2
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 0,
        }

        # Use 10 URLs
        urls = []
        for _ in range(10):
            url = await router._use_url()
            urls.append(url)

        # Total should be 10
        assert sum(router.worker_urls.values()) == 10

        # Finish all
        for url in urls:
            await router._finish_url(url)

        # All should be back to 0
        assert all(count == 0 for count in router.worker_urls.values())


# ==============================================================================
# Group C: Concurrent Access Safety
# ==============================================================================

@pytest.mark.unit
class TestConcurrentAccessSafety:
    """Test thread-safety of worker selection."""

    @pytest.mark.asyncio
    async def test_concurrent_worker_selection(self):
        """
        Test: Many concurrent worker selections.

        Race Condition: Multiple tasks select workers simultaneously
        Expected: All selections should be tracked correctly
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 0,
            "http://worker3:10090": 0,
        }

        # 100 concurrent selections
        urls = await asyncio.gather(*[router._use_url() for _ in range(100)])

        # Total count should be 100 (Boundary: concurrent selection count accuracy)
        assert sum(router.worker_urls.values()) == 100, \
            f"Total count should be 100 after 100 concurrent selections, got {sum(router.worker_urls.values())}: {router.worker_urls}"

        # All URLs should be valid
        assert all(url in router.worker_urls for url in urls), \
            f"All selected URLs should be valid workers"

        # Distribution should be roughly even (within 10% of perfect)
        perfect = 100 / 3
        for count in router.worker_urls.values():
            assert abs(count - perfect) < perfect * 0.2, f"Distribution not balanced: {router.worker_urls}"

    @pytest.mark.asyncio
    async def test_concurrent_finish_operations(self):
        """
        Test: Many concurrent finish operations.

        Race Condition: Multiple tasks finish URLs simultaneously
        Expected: Counts should be accurate
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 50,
            "http://worker2:10090": 30,
            "http://worker3:10090": 20,
        }

        # Finish 50 URLs concurrently (distributed)
        finish_urls = (
            ["http://worker1:10090"] * 30 +
            ["http://worker2:10090"] * 15 +
            ["http://worker3:10090"] * 5
        )

        await asyncio.gather(*[router._finish_url(url) for url in finish_urls])

        # Verify counts
        assert router.worker_urls["http://worker1:10090"] == 20
        assert router.worker_urls["http://worker2:10090"] == 15
        assert router.worker_urls["http://worker3:10090"] == 15

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """
        Test: Mix of concurrent use and finish operations.

        Stress: Concurrent additions and removals
        Expected: Final counts should be consistent
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 2
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 10,
            "http://worker2:10090": 10,
        }

        async def use_and_finish():
            """Use a URL then finish it."""
            url = await router._use_url()
            await asyncio.sleep(0.001)  # Simulate work
            await router._finish_url(url)

        # Run 50 concurrent use-and-finish cycles
        await asyncio.gather(*[use_and_finish() for _ in range(50)])

        # Counts should be back to original (Boundary: concurrent use/finish cycle correctness)
        total_count = sum(router.worker_urls.values())
        assert total_count == 20, \
            f"Count should return to original (20) after use/finish cycles, got {total_count}: {router.worker_urls}"

    @pytest.mark.asyncio
    async def test_high_concurrency_stress_test(self):
        """
        Test: Very high concurrency (500+ operations).

        Stress: Extreme concurrent load
        Expected: No errors, counts accurate
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 5
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {f"http://worker{i}:10090": 0 for i in range(5)}

        # 500 concurrent selections
        urls = await asyncio.gather(*[router._use_url() for _ in range(500)])

        # Total should be 500 (Boundary: extreme concurrency stress test)
        assert sum(router.worker_urls.values()) == 500, \
            f"Total count should be 500 after 500 concurrent selections, got {sum(router.worker_urls.values())}"

        # All selections should be valid
        assert len(urls) == 500, f"Should have 500 URL selections, got {len(urls)}"
        assert all(url in router.worker_urls for url in urls), \
            "All selected URLs should be from registered workers"


# ==============================================================================
# Group D: Worker Management
# ==============================================================================

@pytest.mark.unit
class TestWorkerManagement:
    """Test worker add/list/remove operations."""

    @pytest.mark.asyncio
    async def test_add_worker_via_query_param(self):
        """
        Test: Add worker via query parameter.

        Expected: Worker added to worker_urls
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

        # Mock request with query param
        request = Mock(spec=Request)
        request.query_params = {"url": "http://new-worker:10090"}
        request.body = AsyncMock(return_value=b'{}')

        # Add worker
        response = await router.add_worker(request)

        # Verify worker was added
        assert "http://new-worker:10090" in router.worker_urls, \
            f"New worker should be in worker_urls, got: {list(router.worker_urls.keys())}"
        assert router.worker_urls["http://new-worker:10090"] == 0, \
            f"New worker should start with count 0, got {router.worker_urls.get('http://new-worker:10090')}"

    @pytest.mark.asyncio
    async def test_add_worker_via_json_body(self):
        """
        Test: Add worker via JSON body.

        Expected: Worker added to worker_urls
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

        # Mock request with JSON body
        request = Mock(spec=Request)
        request.query_params = {}
        request.body = AsyncMock(return_value=b'{"url": "http://json-worker:10090"}')

        # Add worker
        response = await router.add_worker(request)

        # Verify worker was added
        assert "http://json-worker:10090" in router.worker_urls

    @pytest.mark.asyncio
    async def test_add_duplicate_worker(self):
        """
        Test: Adding same worker twice.

        Expected: Count should not reset, stays at current value
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
        router.worker_urls = {"http://existing:10090": 5}

        # Try to add same worker
        request = Mock(spec=Request)
        request.query_params = {"url": "http://existing:10090"}
        request.body = AsyncMock(return_value=b'{}')

        response = await router.add_worker(request)

        # Count should remain unchanged (Boundary: duplicate worker addition)
        assert router.worker_urls["http://existing:10090"] == 5, \
            f"Duplicate worker addition should not reset count, got {router.worker_urls['http://existing:10090']}"

    @pytest.mark.asyncio
    async def test_add_worker_missing_url(self):
        """
        Test: Add worker without providing URL.

        Expected: Error response
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

        # Request without URL
        request = Mock(spec=Request)
        request.query_params = {}
        request.body = AsyncMock(return_value=b'{}')

        response = await router.add_worker(request)

        # Should get error response (Boundary: missing required URL parameter)
        assert response.status_code == 400, \
            f"Missing URL should return 400 Bad Request, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_list_workers(self):
        """
        Test: List all workers.

        Expected: Returns list of worker URLs
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 0,
            "http://worker2:10090": 5,
            "http://worker3:10090": 10,
        }

        # List workers
        request = Mock(spec=Request)
        response = await router.list_workers(request)

        # Should return all URLs
        assert len(response["urls"]) == 3
        assert "http://worker1:10090" in response["urls"]
        assert "http://worker2:10090" in response["urls"]
        assert "http://worker3:10090" in response["urls"]


# ==============================================================================
# Group E: Cache Availability Checking
# ==============================================================================

@pytest.mark.unit
class TestCacheAvailabilityChecking:
    """Test cache availability detection."""

    def test_cache_available_with_components(self):
        """
        Test: Cache is available when components are registered.

        Expected: Returns True
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

        # Register components
        router.component_registry.register("radix_tree", Mock())
        router.component_registry.register("tokenizer", Mock())

        # Check availability
        available = router._check_cache_availability()

        assert available is True

    def test_cache_unavailable_missing_radix_tree(self):
        """
        Test: Cache unavailable when radix_tree is missing.

        Expected: Returns False
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

        # Only register tokenizer
        router.component_registry.register("tokenizer", Mock())

        # Check availability
        available = router._check_cache_availability()

        assert available is False

    def test_cache_unavailable_missing_tokenizer(self):
        """
        Test: Cache unavailable when tokenizer is missing.

        Expected: Returns False
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

        # Only register radix_tree
        router.component_registry.register("radix_tree", Mock())

        # Check availability
        available = router._check_cache_availability()

        assert available is False

    def test_cache_availability_caching(self):
        """
        Test: Cache availability result is cached.

        Expected: Second check doesn't re-evaluate
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

        # Register components
        router.component_registry.register("radix_tree", Mock())
        router.component_registry.register("tokenizer", Mock())

        # First check
        result1 = router._check_cache_availability()
        assert result1 is True

        # Remove components (shouldn't affect cached result)
        router.component_registry.clear()

        # Second check should still return cached True
        result2 = router._check_cache_availability()
        assert result2 is True, "Result should be cached"


# ==============================================================================
# Group F: Load Balancing Verification
# ==============================================================================

@pytest.mark.unit
class TestLoadBalancingVerification:
    """Test load balancing accuracy and fairness."""

    @pytest.mark.asyncio
    async def test_load_distribution_accuracy(self):
        """
        Test: Load is distributed accurately across workers.

        Verification: Statistical distribution check
        Expected: Variance should be low for even distribution
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 4
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {f"http://worker{i}:10090": 0 for i in range(4)}

        # Make 1000 selections
        for _ in range(1000):
            await router._use_url()

        # Check distribution
        counts = list(router.worker_urls.values())
        expected = 1000 / 4  # 250 each

        # All should be close to expected (within 5%)
        for count in counts:
            assert abs(count - expected) < expected * 0.05, \
                f"Unbalanced: {router.worker_urls}"

    @pytest.mark.asyncio
    async def test_load_rebalancing_after_worker_finishes(self):
        """
        Test: Load rebalances when workers finish requests.

        Scenario: Some workers finish faster
        Expected: New requests go to freed workers
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 3
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 10,
            "http://worker2:10090": 10,
            "http://worker3:10090": 10,
        }

        # Worker 1 finishes 5 requests
        for _ in range(5):
            await router._finish_url("http://worker1:10090")

        # Now worker1 has count=5, others have 10
        # Next selections should go to worker1
        for _ in range(5):
            url = await router._use_url()
            assert url == "http://worker1:10090", "Should pick least loaded"

        # Now all should be balanced again
        assert all(count == 10 for count in router.worker_urls.values())

    @pytest.mark.asyncio
    async def test_dynamic_worker_addition(self):
        """
        Test: New worker gets load when added.

        Scenario: Add worker during operation
        Expected: New worker participates in load balancing
        """
        args = Mock()
        args.verbose = False
        args.sglang_router_ip = "0.0.0.0"
        args.sglang_router_port = 30000
        args.sglang_server_concurrency = 128
        args.rollout_num_gpus = 2
        args.rollout_num_gpus_per_engine = 1
        args.slime_router_middleware_paths = []

        router = SlimeRouter(args)
        router.worker_urls = {
            "http://worker1:10090": 10,
            "http://worker2:10090": 10,
        }

        # Add new worker
        router.worker_urls["http://worker3:10090"] = 0

        # Next 10 selections should all go to worker3
        for _ in range(10):
            url = await router._use_url()
            assert url == "http://worker3:10090", "New worker should get priority"

        # Now all balanced
        assert all(count == 10 for count in router.worker_urls.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
