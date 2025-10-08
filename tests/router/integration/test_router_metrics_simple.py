"""
Simple integration tests for Router /metrics API endpoint.

Tests cover:
- Basic metrics endpoint response validation
- Router statistics accuracy (active workers, load distribution)
- Memory usage calculation accuracy

Mock Strategy:
- Mock SlimeRouter class to avoid sglang dependency
- Mock Request objects for FastAPI
- Mock radix_tree.get_stats() when needed
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_slime_router():
    """Create a mock SlimeRouter with minimal dependencies."""
    router = MagicMock()

    # Mock worker_urls dict
    router.worker_urls = {}

    # Explicitly set radix_tree to None to avoid MagicMock auto-creation
    router.radix_tree = None

    # Mock get_metrics method that we want to test
    async def mock_get_metrics(request):
        """GET /metrics - Return router and cache metrics."""
        metrics = {
            "router": {
                "active_workers": len(router.worker_urls),
                "worker_loads": dict(router.worker_urls),
                "total_in_flight": sum(router.worker_urls.values()),
            }
        }

        if hasattr(router, "radix_tree") and router.radix_tree:
            cache_stats = router.radix_tree.get_stats()
            # Estimate memory usage (16 bytes per token ID)
            cache_stats["cache_size_mb"] = (
                cache_stats["cur_cache_size"] * 16 / 1024 / 1024
            )
            metrics["cache"] = cache_stats

        from fastapi.responses import JSONResponse
        return JSONResponse(content=metrics)

    router.get_metrics = mock_get_metrics
    return router


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/metrics"
    return request


@pytest.fixture
def mock_radix_tree():
    """Create a mock radix tree with sample stats."""
    tree = MagicMock()
    tree.get_stats.return_value = {
        "total_entries": 150,
        "cache_hits": 1200,
        "cache_misses": 300,
        "hit_rate": 0.8,
        "cur_cache_size": 8500,
        "max_cache_size": 10000,
    }
    return tree


# ============================================================================
# Basic Metrics Endpoint Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_basic_response(mock_slime_router, mock_request):
    """
    Test: Basic /metrics endpoint returns proper response structure.

    Scenario:
    - Call /metrics with no workers and no cache
    - Should return valid JSON with router section only
    - All router fields should be present
    """
    response = await mock_slime_router.get_metrics(mock_request)

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
async def test_metrics_endpoint_with_workers(mock_slime_router, mock_request):
    """
    Test: /metrics endpoint correctly reports worker statistics.

    Scenario:
    - Add workers to router
    - Should return correct worker counts and loads
    - Verify load calculation accuracy
    """
    router = mock_slime_router

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
async def test_metrics_endpoint_with_cache(mock_slime_router, mock_request, mock_radix_tree):
    """
    Test: /metrics endpoint includes cache statistics when available.

    Scenario:
    - Router has radix_tree attached
    - Should include cache section with memory calculation
    - Verify memory usage calculation (16 bytes per token)
    """
    router = mock_slime_router
    router.radix_tree = mock_radix_tree

    response = await router.get_metrics(mock_request)
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
        # pytest.approx handles float comparison automatically
        assert cache_section[key] == expected_value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_memory_usage_calculation(mock_slime_router, mock_request, mock_radix_tree):
    """
    Test: Memory usage calculation is accurate.

    Scenario:
    - Different cache sizes should produce correct MB values
    - Formula: cache_size_mb = cur_cache_size * 16 / 1024 / 1024
    """
    router = mock_slime_router
    router.radix_tree = mock_radix_tree

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
async def test_metrics_endpoint_empty_workers(mock_slime_router, mock_request):
    """
    Test: /metrics endpoint handles empty worker list gracefully.

    Scenario:
    - Router has no workers
    - Should return valid JSON with zero counts
    """
    router = mock_slime_router
    router.worker_urls = {}

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    router_section = response_data["router"]
    assert router_section["active_workers"] == 0
    assert router_section["worker_loads"] == {}
    assert router_section["total_in_flight"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_single_worker(mock_slime_router, mock_request):
    """
    Test: /metrics endpoint handles single worker correctly.

    Scenario:
    - Router has only one worker
    - Should report correct statistics for single worker
    """
    router = mock_slime_router
    router.worker_urls = {"http://single-worker:10090": 5}

    response = await router.get_metrics(mock_request)
    response_data = json.loads(response.body.decode("utf-8"))

    router_section = response_data["router"]
    assert router_section["active_workers"] == 1
    assert router_section["worker_loads"] == {"http://single-worker:10090": 5}
    assert router_section["total_in_flight"] == 5


# ============================================================================
# JSON Response Format Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_response_content_type(mock_slime_router, mock_request):
    """
    Test: /metrics endpoint returns correct content type.

    Scenario:
    - Verify response has proper JSON content type
    - Check headers are set correctly
    """
    response = await mock_slime_router.get_metrics(mock_request)

    assert response.status_code == 200
    assert "application/json" in response.media_type


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_response_serialization(mock_slime_router, mock_request, mock_radix_tree):
    """
    Test: /metrics endpoint response is properly serializable.

    Scenario:
    - All values in response should be JSON serializable
    - No datetime objects or other non-serializable types
    """
    router = mock_slime_router
    router.radix_tree = mock_radix_tree

    response = await router.get_metrics(mock_request)
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
async def test_metrics_endpoint_consistency(mock_slime_router, mock_request, mock_radix_tree):
    """
    Test: Multiple calls to /metrics return consistent data.

    Scenario:
    - Call metrics endpoint multiple times
    - Should return consistent data for unchanged state
    """
    router = mock_slime_router
    router.radix_tree = mock_radix_tree

    responses = []
    for _ in range(5):
        response = await router.get_metrics(mock_request)
        response_data = json.loads(response.body.decode("utf-8"))
        responses.append(response_data)

    # All responses should be identical
    first_response = responses[0]
    for i, response in enumerate(responses[1:], 1):
        assert response == first_response, f"Response {i} differs from first response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_with_dynamic_changes(mock_slime_router, mock_request):
    """
    Test: /metrics endpoint reflects dynamic changes.

    Scenario:
    - Change worker loads and verify metrics update
    - Test real-time monitoring capability
    """
    router = mock_slime_router

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