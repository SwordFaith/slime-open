"""
Concurrency safety tests for SlimeRouter URL selection logic.

Tests verify that _use_url() and _finish_url() are thread-safe under high concurrency.
Without proper locking, race conditions can lead to:
- Unbalanced worker load distribution
- Negative worker counts
- Deadlocks on exceptions
"""

import asyncio
import sys
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from collections import Counter

# Mock sglang module structure to avoid import errors
sglang_mock = MagicMock()
sglang_srt_mock = MagicMock()
sglang_srt_mock.ServerArgs = MagicMock()
sglang_mock.srt = sglang_srt_mock
sys.modules["sglang"] = sglang_mock
sys.modules["sglang.srt"] = sglang_srt_mock
sys.modules["sglang.srt.server_args"] = MagicMock()


@pytest.fixture
def mock_router():
    """
    Create SlimeRouter instance with mocked dependencies.

    Bypasses FastAPI initialization by using __new__() and manually setting attributes.
    """
    from slime.router.router import SlimeRouter

    # Create minimal args mock
    args = Mock()
    args.sglang_server_concurrency = 10
    args.rollout_num_gpus = 4
    args.rollout_num_gpus_per_engine = 1
    args.slime_router_middleware_paths = []

    # Bypass __init__ to avoid FastAPI setup
    router = SlimeRouter.__new__(SlimeRouter)
    router.args = args
    router.verbose = False
    router.worker_urls = {}
    router.client = AsyncMock()  # Mock httpx.AsyncClient

    # Add _url_lock (should exist in fixed version)
    router._url_lock = asyncio.Lock()

    return router


@pytest.mark.asyncio
async def test_concurrent_url_selection_load_balance(mock_router):
    """
    Test 1: Verify load balancing correctness under concurrent requests.

    Scenario:
    - 4 workers with initial count = 0
    - 100 concurrent requests calling _use_url() + _finish_url()
    - Expected: Each worker gets ~25 requests (variance < 20%)

    Without lock:
    - Race condition causes uneven distribution
    - Multiple coroutines call min() simultaneously, getting same result
    - Some workers get 40+ requests, others get <10

    With lock:
    - Round-robin correctly distributes load
    - Variance stays within acceptable range
    """
    # Setup: Add 4 workers
    mock_router.worker_urls = {
        "http://worker1:10090": 0,
        "http://worker2:10090": 0,
        "http://worker3:10090": 0,
        "http://worker4:10090": 0,
    }

    # Track which worker was selected by each request
    selected_workers = []
    selection_lock = asyncio.Lock()  # Protect list append

    # Track max concurrent count for each worker
    max_concurrent_counts = {url: 0 for url in mock_router.worker_urls}
    count_lock = asyncio.Lock()

    async def simulate_request():
        """Simulate a single request: select worker -> do work -> release worker"""
        url = mock_router._use_url()
        if asyncio.iscoroutine(url):  # Fixed version returns coroutine
            url = await url

        async with selection_lock:
            selected_workers.append(url)

        # Track max concurrent count (this exposes race condition)
        async with count_lock:
            max_concurrent_counts[url] = max(max_concurrent_counts[url], mock_router.worker_urls[url])

        # Simulate some async work (CRITICAL: add yield point to expose race)
        await asyncio.sleep(0)  # Force context switch

        result = mock_router._finish_url(url)
        if asyncio.iscoroutine(result):  # Fixed version is async
            await result

    # Execute 100 concurrent requests
    await asyncio.gather(*[simulate_request() for _ in range(100)])

    # Verify: All 4 workers should be used
    counts = Counter(selected_workers)
    assert len(counts) == 4, f"Expected 4 workers used, got {len(counts)}: {dict(counts)}"

    # Verify: Load should be balanced (each worker gets ~25 requests)
    mean_requests = 25.0
    variance = sum((count - mean_requests) ** 2 for count in counts.values()) / 4
    std_dev = variance**0.5

    # Allow 20% deviation from mean (5 requests)
    max_std_dev = mean_requests * 0.2

    print(f"\n[Load Balance Test] Worker distribution: {dict(counts)}")
    print(f"[Load Balance Test] Max concurrent per worker: {max_concurrent_counts}")
    print(f"[Load Balance Test] Standard deviation: {std_dev:.2f} (max: {max_std_dev:.2f})")

    assert std_dev < max_std_dev, (
        f"Load imbalance detected!\n"
        f"Worker distribution: {dict(counts)}\n"
        f"Standard deviation: {std_dev:.2f} (max allowed: {max_std_dev:.2f})\n"
        f"This indicates a race condition in _use_url()"
    )

    # Verify: All workers are back to count=0 (all requests finished)
    assert all(
        count == 0 for count in mock_router.worker_urls.values()
    ), f"Worker counts not back to 0: {mock_router.worker_urls}"


@pytest.mark.asyncio
async def test_concurrent_no_negative_counts(mock_router):
    """
    Test 2: Verify worker counts never go negative under high concurrency.

    Scenario:
    - 2 workers with initial count = 0
    - 1000 concurrent requests (stress test)
    - Expected: All worker counts >= 0 throughout execution

    Without lock:
    - Race condition in _finish_url() can cause count < 0
    - Example: Two coroutines both decrement at same time

    With lock:
    - Atomic decrement ensures count never goes negative
    """
    # Setup: Add 2 workers
    mock_router.worker_urls = {
        "http://worker1:10090": 0,
        "http://worker2:10090": 0,
    }

    # Track minimum count observed during execution
    min_counts_observed = {"http://worker1:10090": 0, "http://worker2:10090": 0}
    observation_lock = asyncio.Lock()

    async def simulate_request():
        """Simulate request and observe counts"""
        url = mock_router._use_url()
        if asyncio.iscoroutine(url):
            url = await url

        # Observe counts after _use_url
        async with observation_lock:
            for worker_url, count in mock_router.worker_urls.items():
                min_counts_observed[worker_url] = min(min_counts_observed[worker_url], count)

        await asyncio.sleep(0.0001)  # Minimal work

        result = mock_router._finish_url(url)
        if asyncio.iscoroutine(result):
            await result

        # Observe counts after _finish_url
        async with observation_lock:
            for worker_url, count in mock_router.worker_urls.items():
                min_counts_observed[worker_url] = min(min_counts_observed[worker_url], count)

    # Execute 1000 concurrent requests (high stress)
    await asyncio.gather(*[simulate_request() for _ in range(1000)])

    # Verify: No negative counts observed
    for worker_url, min_count in min_counts_observed.items():
        assert min_count >= 0, (
            f"Negative count detected for {worker_url}: {min_count}\n"
            f"This indicates a race condition in _finish_url()"
        )

    # Verify: All workers back to count=0
    assert all(
        count == 0 for count in mock_router.worker_urls.values()
    ), f"Worker counts not back to 0: {mock_router.worker_urls}"


@pytest.mark.asyncio
async def test_lock_deadlock_prevention(mock_router):
    """
    Test 3: Verify lock is correctly released when exceptions occur.

    Scenario:
    - 1 worker
    - Mock client.request() to raise exception
    - Call proxy() which should trigger finally block
    - Expected: _finish_url() is called, subsequent requests can acquire lock

    Without proper try/finally:
    - Lock may not be released on exception
    - Deadlock occurs (subsequent requests hang)

    With proper try/finally:
    - Lock is always released via _finish_url() in finally block
    - No deadlock, count correctly decrements
    """
    from slime.router.router import SlimeRouter
    from fastapi import Request

    # Setup: Add 1 worker
    mock_router.worker_urls = {"http://worker1:10090": 0}

    # Mock client to raise exception
    mock_router.client.request = AsyncMock(side_effect=Exception("Worker timeout"))

    # Create mock Request
    mock_request = AsyncMock(spec=Request)
    mock_request.method = "POST"
    mock_request.body = AsyncMock(return_value=b'{"text": "Hello"}')
    mock_request.headers = {}

    # Attempt to call proxy (should fail but not deadlock)
    with pytest.raises(Exception, match="Worker timeout"):
        await mock_router.proxy(mock_request, "generate")

    # Verify: Worker count is back to 0 (finally block executed)
    assert mock_router.worker_urls["http://worker1:10090"] == 0, "Worker count not decremented in finally block"

    # Verify: Lock can be acquired again (no deadlock)
    # Simulate a successful request after the failed one
    mock_router.client.request = AsyncMock(
        return_value=AsyncMock(
            aread=AsyncMock(return_value=b'{"status": "ok"}'),
            status_code=200,
            headers={"content-type": "application/json"},
        )
    )

    # This should not hang (proves lock was released)
    try:
        result = await asyncio.wait_for(mock_router.proxy(mock_request, "generate"), timeout=1.0)
        # If we get here, lock was properly released
        assert True, "Lock correctly released after exception"
    except asyncio.TimeoutError:
        pytest.fail("Deadlock detected: subsequent request timed out")

    # Verify: Worker count is back to 0 again
    assert mock_router.worker_urls["http://worker1:10090"] == 0


@pytest.mark.asyncio
async def test_concurrent_add_worker_no_interference(mock_router):
    """
    Bonus Test 4: Verify add_worker() doesn't interfere with concurrent requests.

    This is a regression test for potential issues when workers are added
    during active request processing.
    """
    mock_router.worker_urls = {"http://worker1:10090": 0}

    async def continuous_requests():
        """Keep making requests"""
        for _ in range(50):
            url = mock_router._use_url()
            if asyncio.iscoroutine(url):
                url = await url
            await asyncio.sleep(0.001)
            result = mock_router._finish_url(url)
            if asyncio.iscoroutine(result):
                await result

    async def add_workers_midway():
        """Add workers while requests are in flight"""
        await asyncio.sleep(0.05)  # Let some requests start

        # Add workers (this modifies worker_urls dict)
        if hasattr(mock_router, "_url_lock"):
            async with mock_router._url_lock:
                mock_router.worker_urls["http://worker2:10090"] = 0
        else:
            mock_router.worker_urls["http://worker2:10090"] = 0

        await asyncio.sleep(0.05)

        if hasattr(mock_router, "_url_lock"):
            async with mock_router._url_lock:
                mock_router.worker_urls["http://worker3:10090"] = 0
        else:
            mock_router.worker_urls["http://worker3:10090"] = 0

    # Run both tasks concurrently
    await asyncio.gather(continuous_requests(), add_workers_midway())

    # Verify: All workers back to count=0
    assert all(
        count == 0 for count in mock_router.worker_urls.values()
    ), f"Worker counts not consistent: {mock_router.worker_urls}"
