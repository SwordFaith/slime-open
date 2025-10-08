"""
Integration tests for Router /metrics API endpoint.

Tests cover:
- Basic metrics endpoint response validation
- Router statistics accuracy (active workers, load distribution)
- Cache statistics integration (when radix_tree is available)
- Memory usage calculation accuracy
- Metrics behavior without cache

Mock Strategy:
- Mock Request objects for FastAPI
- Mock radix_tree.get_stats() when needed
- Integration with actual SlimeRouter class
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request
from slime.router.router import SlimeRouter
from slime.router.middleware_hub.radix_tree import StringRadixTrie


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_args():
    """Create mock arguments for SlimeRouter."""
    args = MagicMock()
    args.slime_router_middleware_paths = []
    args.sglang_server_concurrency = 4
    args.rollout_num_gpus = 2
    args.rollout_num_gpus_per_engine = 1
    return args


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/metrics"
    return request


@pytest.fixture
def router_without_cache(mock_args):
    """Create SlimeRouter without cache for testing."""
    return SlimeRouter(args=mock_args)


@pytest.fixture
def mock_radix_tree():
    """Create a mock radix tree with sample stats."""
    tree = MagicMock(spec=StringRadixTrie)
    tree.get_stats.return_value = {
        "total_entries": 150,
        "cache_hits": 1200,
        "cache_misses": 300,
        "hit_rate": 0.8,
        "cur_cache_size": 8500,
        "max_cache_size": 10000,
    }
    return tree


@pytest.fixture
def router_with_cache(mock_args, mock_radix_tree):
    """Create SlimeRouter with cache for testing."""
    router = SlimeRouter(args=mock_args)
    router.radix_tree = mock_radix_tree
    return router


# ============================================================================
# Basic Metrics Endpoint Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_basic_response(router_without_cache, mock_request):
    """
    Test: Basic /metrics endpoint returns proper response structure.

    Scenario:
    - Call /metrics with no workers and no cache
    - Should return valid JSON with router section only
    - All router fields should be present
    """
    response = await router_without_cache.get_metrics(mock_request)

    # Verify response structure
    assert response.status_code == 200
    assert hasattr(response, 'body')

    # Parse JSON response
    response_data = json.loads(response.body.decode("utf-8"))

    # Verify top-level structure
    assert "router" in response_data
    assert "cache" not in response_data  # No cache available

    # Verify router section
    router_section = response_data["router"]
    assert "active_workers" in router_section
    assert "worker_loads" in router_section
    assert "total_in_flight" in router_section

    # Verify router values (empty router)
    assert router_section["active_workers"] == 0
    assert router_section["worker_loads"] == {}
    assert router_section["total_in_flight"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_with_workers(router_without_cache, mock_request):
    """
    Test: /metrics endpoint correctly reports worker statistics.

    Scenario:
    - Add workers to router
    - Should return correct worker counts and loads
    - Verify load calculation accuracy
    """
    router = router_without_cache

    # Add some workers
    router.worker_urls = {
        "http://worker1:10090": 2,
        "http://worker2:10090": 1,
        "http://worker3:10090": 3,
    }

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    # Verify router statistics
    router_section = response_data["router"]
    assert router_section["active_workers"] == 3
    assert router_section["worker_loads"] == {
        "http://worker1:10090": 2,
        "http://worker2:10090": 1,
        "http://worker3:10090": 3,
    }
    assert router_section["total_in_flight"] == 6  # 2 + 1 + 3


# ============================================================================
# Cache Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_with_cache(router_with_cache, mock_request):
    """
    Test: /metrics endpoint includes cache statistics when available.

    Scenario:
    - Router has radix_tree attached
    - Should include cache section with memory calculation
    - Verify memory usage calculation (16 bytes per token)
    """
    response = await router_with_cache.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    # Verify both sections present
    assert "router" in response_data
    assert "cache" in response_data

    # Verify cache section
    cache_section = response_data["cache"]
    expected_cache_stats = {
        "total_entries": 150,
        "cache_hits": 1200,
        "cache_misses": 300,
        "hit_rate": 0.8,
        "cur_cache_size": 8500,
        "max_cache_size": 10000,
        "cache_size_mb": pytest.approx(8500 * 16 / 1024 / 1024, rel=1e-9),  # ~0.13MB
    }

    for key, expected_value in expected_cache_stats.items():
        assert key in cache_section
        if key == "cache_size_mb":
            # Special handling for float comparison
            assert abs(cache_section[key] - expected_value) < 0.001
        else:
            assert cache_section[key] == expected_value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_memory_usage_calculation(router_with_cache, mock_request):
    """
    Test: Memory usage calculation is accurate.

    Scenario:
    - Different cache sizes should produce correct MB values
    - Formula: cache_size_mb = cur_cache_size * 16 / 1024 / 1024
    """
    router = router_with_cache

    # Test different cache sizes
    test_sizes = [0, 1024, 65536, 1048576]  # 0, 1K, 64K, 1M tokens

    for size in test_sizes:
        router.radix_tree.get_stats.return_value = {
            "cur_cache_size": size,
            "total_entries": size // 10,
        }

        response = await router.get_metrics(mock_request)
        response_data = json.loads(response.body.decode("utf-8"))

        expected_mb = size * 16 / 1024 / 1024
        actual_mb = response_data["cache"]["cache_size_mb"]

        assert abs(actual_mb - expected_mb) < 0.001


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_empty_workers(router_without_cache, mock_request):
    """
    Test: /metrics endpoint handles empty worker list gracefully.

    Scenario:
    - Router has no workers
    - Should return valid JSON with zero counts
    """
    router = router_without_cache
    router.worker_urls = {}

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    router_section = response_data["router"]
    assert router_section["active_workers"] == 0
    assert router_section["worker_loads"] == {}
    assert router_section["total_in_flight"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_single_worker(router_without_cache, mock_request):
    """
    Test: /metrics endpoint handles single worker correctly.

    Scenario:
    - Router has only one worker
    - Should report correct statistics for single worker
    """
    router = router_without_cache
    router.worker_urls = {"http://single-worker:10090": 5}

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    router_section = response_data["router"]
    assert router_section["active_workers"] == 1
    assert router_section["worker_loads"] == {"http://single-worker:10090": 5}
    assert router_section["total_in_flight"] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_high_worker_loads(router_without_cache, mock_request):
    """
    Test: /metrics endpoint handles high worker loads correctly.

    Scenario:
    - Workers with high load counts
    - Should handle large integers without overflow
    """
    router = router_without_cache
    router.worker_urls = {
        "http://worker1:10090": 999999,
        "http://worker2:10090": 1000000,
    }

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    router_section = response_data["router"]
    assert router_section["active_workers"] == 2
    assert router_section["total_in_flight"] == 1999999  # 999999 + 1000000


# ============================================================================
# Cache Statistics Edge Cases
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_cache_with_zero_entries(router_with_cache, mock_request):
    """
    Test: /metrics endpoint handles empty cache correctly.

    Scenario:
    - Cache exists but has no entries
    - Should report zero values correctly
    """
    router = router_with_cache
    router.radix_tree.get_stats.return_value = {
        "total_entries": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "hit_rate": 0.0,
        "cur_cache_size": 0,
        "max_cache_size": 10000,
    }

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    cache_section = response_data["cache"]
    assert cache_section["cache_size_mb"] == 0.0
    assert cache_section["total_entries"] == 0
    assert cache_section["hit_rate"] == 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_cache_max_memory_calculation(router_with_cache, mock_request):
    """
    Test: /metrics endpoint calculates maximum possible memory usage.

    Scenario:
    - Cache at maximum capacity
    - Should calculate memory usage for max_cache_size
    """
    router = router_with_cache
    router.radix_tree.get_stats.return_value = {
        "total_entries": 1000,
        "cache_hits": 5000,
        "cache_misses": 1000,
        "hit_rate": 0.833,
        "cur_cache_size": 10000,
        "max_cache_size": 10000,
    }

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    cache_section = response_data["cache"]
    max_possible_mb = 10000 * 16 / 1024 / 1024  # ~0.153MB
    assert abs(cache_section["cache_size_mb"] - max_possible_mb) < 0.001


# ============================================================================
# JSON Response Format Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_response_content_type(router_without_cache, mock_request):
    """
    Test: /metrics endpoint returns correct content type.

    Scenario:
    - Verify response has proper JSON content type
    - Check headers are set correctly
    """
    response = await router_without_cache.get_metrics(mock_request)

    assert response.status_code == 200
    assert "application/json" in response.media_type


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_response_serialization(router_with_cache, mock_request):
    """
    Test: /metrics endpoint response is properly serializable.

    Scenario:
    - All values in response should be JSON serializable
    - No datetime objects or other non-serializable types
    """
    response = await router_with_cache.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    # Verify all values are basic JSON types
    def check_json_serializable(obj):
        if isinstance(obj, dict):
            return all(check_json_serializable(v) for v in obj.values())
        elif isinstance(obj, list):
            return all(check_json_serializable(v) for v in obj)
        else:
            return isinstance(obj, (str, int, float, bool, type(None)))

    assert check_json_serializable(response_data)


# ============================================================================
# Performance and Consistency Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_consistency(router_with_cache, mock_request):
    """
    Test: Multiple calls to /metrics return consistent data.

    Scenario:
    - Call metrics endpoint multiple times
    - Should return consistent data for unchanged state
    """
    responses = []
    for _ in range(5):
        response = await router_with_cache.get_metrics(mock_request)
        response_data = json.loads(response.body.decode("utf-8"))
        responses.append(response_data)

    # All responses should be identical
    first_response = responses[0]
    for i, response in enumerate(responses[1:], 1):
        assert response == first_response, f"Response {i} differs from first response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_with_dynamic_changes(router_without_cache, mock_request):
    """
    Test: /metrics endpoint reflects dynamic changes.

    Scenario:
    - Change worker loads and verify metrics update
    - Test real-time monitoring capability
    """
    router = router_without_cache

    # Initial state
    router.worker_urls = {"http://worker1:10090": 1}
    response1 = await router.get_metrics(mock_request)
    data1 = json.loads(response1.body.decode("utf-8"))
    assert data1["router"]["total_in_flight"] == 1

    # Update state
    router.worker_urls = {"http://worker1:10090": 2, "http://worker2:10090": 3}
    response2 = await router.get_metrics(mock_request)
    data2 = json.loads(response2.body.decode("utf-8"))
    assert data2["router"]["total_in_flight"] == 5
    assert data2["router"]["active_workers"] == 2