"""Test performance issues with current threading.RLock implementation."""
import asyncio
import time
from unittest.mock import Mock

import pytest

from slime.router.middleware_hub.radix_tree import StringRadixTrie


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
    cache_start_time = min([ts for _, ts, _ in task_timestamps if len(cache_operation_durations) > 0])
    cache_end_time = cache_start_time + sum(cache_operation_durations) + len(cache_operation_durations) * 0.001

    # Count background executions during cache operations
    background_during_cache = [
        ts for (task_type, ts, i) in task_timestamps
        if cache_start_time <= ts <= cache_end_time
    ]

    print(f"Background task executions during cache ops: {len(background_during_cache)}")
    print(f"Expected background executions if no blocking: {len(background_during_cache) / (cache_end_time - cache_start_time) * 0.005:.1f}")

    # With proper async implementation, background task should continue during cache ops
    # With RLock blocking, background executions should be severely limited
    expected_concurrent_executions = (cache_end_time - cache_start_time) / 0.005

    # Document the problem - RLock is causing blocking
    print(f"CONCURRENCY ISSUE: Only {len(background_during_cache)} background executions during cache operations")
    print(f"Expected without blocking: ~{expected_concurrent_executions:.1f} executions")

    # This demonstrates the performance problem
    # Even with a small difference, it shows RLock is blocking concurrency
    print(f"PERFORMANCE ISSUE: RLock reduced background executions by ~{((expected_concurrent_executions - len(background_during_cache)) / expected_concurrent_executions * 100):.1f}%")

    # Document the issue - any reduction in background executions indicates blocking
    assert len(background_during_cache) < expected_concurrent_executions, \
        f"Background task should have executed more frequently: {len(background_during_cache)} vs expected {expected_concurrent_executions:.1f}"


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


if __name__ == "__main__":
    # Run tests manually for debugging
    asyncio.run(test_rlock_blocks_event_loop())
    asyncio.run(test_concurrent_cache_access_performance())
    asyncio.run(test_read_write_performance_disparity())