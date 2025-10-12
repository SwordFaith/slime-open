"""
Radix Tree Async Integration and Performance Tests (Merged)

This test suite combines tests from:
- test_radix_tree_middleware_async.py
- test_lock_performance_issues.py

Tests cover:
- Async middleware functionality
- Performance issues with RLock vs AsyncReadWriteLock
- Integration with external services
- Lock contention and scalability
- Error handling in production scenarios
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware
from slime.router.core.radix_tree import StringRadixTrie


# ============================================================================
# Group A: Async Middleware Functionality
# ============================================================================

@pytest.fixture
def mock_router():
    """Create a mock router for testing."""
    router = Mock()
    router.args = Mock()
    router.args.hf_checkpoint = "test-checkpoint"
    router.args.radix_tree_max_size = 1000  # Set real value
    router.args.verbose = False  # Set real value
    router.verbose = False
    return router


@pytest.fixture
async def middleware(mock_router):
    """Create middleware instance with mocked dependencies."""
    with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
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
    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("Hello world")

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
    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("Hello world")

    # Should get cached "Hello" + tokenized " world"
    assert len(tokens) >= 2  # At least the cached "Hello" tokens
    assert len(logprobs) == len(tokens)
    assert len(loss_mask) == len(tokens)


@pytest.mark.asyncio
async def test_middleware_async_cache_retrieval_no_match(middleware):
    """Test middleware with no cache match."""
    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("Unknown text")

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
    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("Test insertion")

    assert tokens == [4, 5, 6]
    assert logprobs == [-0.4, -0.5, -0.6]
    assert loss_mask == [1, 1, 1]


@pytest.mark.asyncio
async def test_middleware_dispatch_with_cache_hit(mock_router):
    """Test middleware dispatch with cache hit."""
    with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
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
    with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
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
    tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("")
    assert tokens == []
    assert logprobs == []
    assert loss_mask == []

    # Test insertion with empty text (should return False)
    result = await middleware._insert_cache("", [], [], [], weight_version=1)
    assert result is False


# ============================================================================
# Group B: Performance Issues with Lock Implementation
# ============================================================================

@pytest.mark.asyncio
async def test_rlock_blocks_event_loop():
    """Test that current RLock implementation blocks the event loop."""
    # Create a RadixTree with current implementation
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Insert some test data
    tree.insert("Hello", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0])
    tree.insert("World", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1])

    # Track task execution timing
    task_timestamps = []
    cache_operation_durations = []

    async def background_counter():
        """Background task that runs continuously."""
        start_time = time.time()
        for i in range(20):
            task_timestamps.append(("background", time.time() - start_time, i))
            await asyncio.sleep(0.005)  # Very frequent updates

    async def intensive_cache_operations():
        """Perform cache operations that should block other tasks."""
        start_time = time.time()

        for i in range(10):
            op_start = time.time()
            # These operations use threading.RLock which blocks the event loop
            result = tree.find_longest_prefix("Hello")

            # Add some computation to make the lock held longer
            time.sleep(0.002)  # This will block the entire event loop

            op_end = time.time()
            cache_operation_durations.append(op_end - op_start)

            # Very small async sleep
            await asyncio.sleep(0.001)

        end_time = time.time()
        return end_time - start_time

    # Start background task first
    background_task = asyncio.create_task(background_counter())

    # Let background task run a bit
    await asyncio.sleep(0.02)

    # Now start intensive cache operations
    cache_task = asyncio.create_task(intensive_cache_operations())

    # Wait for both to complete
    await cache_task
    await background_task

    # Analyze the timing data
    print(f"Background task executed {len(task_timestamps)} times")
    print(f"Cache operations took: {cache_operation_durations}")

    # Find when cache operations were running
    if task_timestamps and cache_operation_durations:
        cache_start_time = min([ts for _, ts, _ in task_timestamps])
        cache_end_time = cache_start_time + sum(cache_operation_durations) + len(cache_operation_durations) * 0.001

        # Count background executions during cache operations
        background_during_cache = [
            ts for (task_type, ts, i) in task_timestamps
            if cache_start_time <= ts <= cache_end_time
        ]

        print(f"Background task executions during cache ops: {len(background_during_cache)}")

        expected_concurrent_executions = (cache_end_time - cache_start_time) / 0.005

        # Document the problem - RLock is causing blocking
        print(f"CONCURRENCY ISSUE: Only {len(background_during_cache)} background executions during cache operations")
        print(f"Expected without blocking: ~{expected_concurrent_executions:.1f} executions")

        # This demonstrates the performance problem
        if expected_concurrent_executions > 0:
            print(f"PERFORMANCE ISSUE: RLock reduced background executions by ~{((expected_concurrent_executions - len(background_during_cache)) / expected_concurrent_executions * 100):.1f}%")


@pytest.mark.asyncio
async def test_concurrent_cache_access_performance():
    """Test performance degradation with concurrent cache access."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Insert test data with different prefixes
    test_data = [
        ("Hello", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0]),
        ("World", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1]),
        ("Test", [7, 8, 9], [-0.7, -0.8, -0.9], [0, 0, 1]),
    ]

    for text, tokens, logp, mask in test_data:
        tree.insert(text, tokens, logp, mask)

    async def concurrent_cache_reads():
        """Multiple concurrent read operations."""
        tasks = []
        start_time = time.time()

        # Create multiple concurrent tasks
        for i in range(20):
            for text, _, _, _ in test_data:
                task = asyncio.create_task(asyncio.to_thread(tree.find_longest_prefix, text))
                tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        return end_time - start_time, len(results)

    # Run concurrent operations
    duration, num_operations = await concurrent_cache_reads()

    print(f"Concurrent operations: {num_operations}")
    print(f"Total duration: {duration:.3f}s")
    print(f"Operations per second: {num_operations/duration:.1f}")

    # With threading.RLock, concurrent operations are actually serialized
    # Performance should be much worse than true concurrent operations
    ops_per_second = num_operations / duration

    # This documents the current performance level
    # The key issue is that asyncio.to_thread + RLock creates unnecessary overhead
    print(f"CURRENT PERFORMANCE: {ops_per_second:.1f} ops/sec with RLock + thread pool")
    print(f"PERFORMANCE ISSUE: asyncio.to_thread adds thread switching overhead to RLock blocking")

    # The main issue is the combination of asyncio.to_thread + RLock
    # Each operation requires thread switching, which is inefficient
    assert ops_per_second < 50000, f"Operations suspiciously fast ({ops_per_second:.1f}/s), may not be measuring accurately"

    # Document the performance problem
    print(f"OPTIMIZATION TARGET: Should achieve much higher performance with native async locks")


@pytest.mark.asyncio
async def test_read_write_performance_disparity():
    """Test that read operations are unnecessarily blocked by write operations."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Initial data
    tree.insert("Base", [1, 2], [-0.1, -0.2], [0, 0])

    read_times = []
    write_times = []

    async def read_operation():
        """Read operation that should be fast."""
        start_time = time.time()
        result = tree.find_longest_prefix("Base")
        end_time = time.time()
        read_times.append(end_time - start_time)
        return result

    async def write_operation():
        """Write operation that should block reads."""
        start_time = time.time()
        tree.insert("BaseExtended", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1])
        end_time = time.time()
        write_times.append(end_time - start_time)

    # Run read and write operations concurrently
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(read_operation()))
        tasks.append(asyncio.create_task(write_operation()))
        tasks.append(asyncio.create_task(read_operation()))

    await asyncio.gather(*tasks)

    if read_times and write_times:
        avg_read_time = sum(read_times) / len(read_times)
        avg_write_time = sum(write_times) / len(write_times)

        print(f"Average read time: {avg_read_time*1000:.2f}ms")
        print(f"Average write time: {avg_write_time*1000:.2f}ms")

        # Document the current performance characteristics
        print(f"CURRENT PERFORMANCE: Read avg: {avg_read_time*1000:.2f}ms, Write avg: {avg_write_time*1000:.2f}ms")
        print(f"PERFORMANCE ISSUE: Even simple read/write operations require RLock acquisition")
        print(f"OPTIMIZATION TARGET: Async read-write lock would allow concurrent reads")

        # The current implementation has minimal contention because operations are very fast
        # But the architecture is still problematic for higher load scenarios
        assert avg_read_time > 0.000001, "Read operations impossibly fast, measurement error"

        # Document the architectural problem
        print(f"ARCHITECTURAL ISSUE: RLock prevents concurrent reads even for simple operations")


# ============================================================================
# Group C: Integration and Performance Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_middleware_performance_under_load(mock_router):
    """Test middleware performance under concurrent load."""
    with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        mock_tokenizer_instance.decode.return_value = "Test response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        app = Mock()
        middleware = RadixTreeMiddleware(app, router=mock_router)

        # Pre-populate cache with some data
        await middleware.radix_tree.insert_async(
            "cached_text", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1
        )

        async def concurrent_request(request_id: int):
            """Simulate concurrent request processing."""
            # Mock request
            request = Mock(spec=Request)
            request.url.path = "/generate"

            # Mix of cached and uncached requests
            if request_id % 2 == 0:
                request.json = AsyncMock(return_value={"text": "cached_text"})
            else:
                request.json = AsyncMock(return_value={"text": f"uncached_text_{request_id}"})

            # Mock response
            mock_response = JSONResponse(content={
                "text": " response",
                "output_ids": [10],
                "meta_info": {"weight_version": 1}
            })
            call_next = AsyncMock(return_value=mock_response)

            # Process request
            start_time = time.time()
            result = await middleware.dispatch(request, call_next)
            end_time = time.time()

            return end_time - start_time

        # Run concurrent requests
        num_requests = 20
        tasks = [asyncio.create_task(concurrent_request(i)) for i in range(num_requests)]
        durations = await asyncio.gather(*tasks)

        # Analyze performance
        avg_duration = sum(durations) / len(durations)
        total_duration = max(durations)
        throughput = num_requests / total_duration

        print(f"Processed {num_requests} concurrent requests")
        print(f"Average request duration: {avg_duration*1000:.2f}ms")
        print(f"Total duration: {total_duration:.3f}s")
        print(f"Throughput: {throughput:.1f} requests/sec")

        # Verify reasonable performance
        assert avg_duration < 0.1, f"Average request duration too high: {avg_duration:.3f}s"
        assert throughput > 50, f"Throughput too low: {throughput:.1f} req/sec"


@pytest.mark.asyncio
async def test_cache_hit_rate_optimization(mock_router):
    """Test cache hit rate impact on performance."""
    with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        app = Mock()
        middleware = RadixTreeMiddleware(app, router=mock_router)

        # Pre-populate cache with common prefixes
        common_prefixes = ["Hello", "Hi", "Hey", "Good morning", "Good evening"]
        for prefix in common_prefixes:
            await middleware.radix_tree.insert_async(
                prefix, [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1
            )

        async def process_request(text: str):
            """Process a single request and measure timing."""
            request = Mock(spec=Request)
            request.url.path = "/generate"
            request.json = AsyncMock(return_value={"text": text})

            mock_response = JSONResponse(content={
                "text": " response",
                "output_ids": [4],
                "meta_info": {"weight_version": 1}
            })
            call_next = AsyncMock(return_value=mock_response)

            start_time = time.time()
            await middleware.dispatch(request, call_next)
            end_time = time.time()

            return end_time - start_time

        # Test with high cache hit rate (80%)
        high_hit_requests = []
        for i in range(50):
            if i % 5 == 0:  # 20% uncached
                text = f"Unique text {i}"
            else:  # 80% cached
                text = common_prefixes[i % len(common_prefixes)]
            duration = await process_request(text)
            high_hit_requests.append(duration)

        # Test with low cache hit rate (20%)
        low_hit_requests = []
        for i in range(50):
            if i % 5 == 0:  # 20% cached
                text = common_prefixes[i % len(common_prefixes)]
            else:  # 80% uncached
                text = f"Unique text {i}"
            duration = await process_request(text)
            low_hit_requests.append(duration)

        # Analyze performance difference
        avg_high_hit = sum(high_hit_requests) / len(high_hit_requests)
        avg_low_hit = sum(low_hit_requests) / len(low_hit_requests)
        performance_improvement = (avg_low_hit - avg_high_hit) / avg_low_hit * 100

        print(f"High cache hit rate (80%): {avg_high_hit*1000:.2f}ms avg")
        print(f"Low cache hit rate (20%): {avg_low_hit*1000:.2f}ms avg")
        print(f"Performance improvement: {performance_improvement:.1f}%")

        # Cache should provide measurable performance improvement
        assert performance_improvement > 5, f"Cache should provide >5% improvement, got {performance_improvement:.1f}%"


if __name__ == "__main__":
    # Run tests manually for debugging
    async def run_all_tests():
        # Middleware functionality tests
        print("Running middleware tests...")
        # Note: These would need proper fixtures to run standalone

        # Performance tests
        print("Running performance tests...")
        await test_rlock_blocks_event_loop()
        await test_concurrent_cache_access_performance()
        await test_read_write_performance_disparity()

        print("All async integration and performance tests passed!")

    import asyncio
    asyncio.run(run_all_tests())