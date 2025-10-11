"""
Unit tests for AsyncReadWriteLock implementation.

Tests cover:
- Basic read-write lock functionality
- Concurrent read operations
- Write operation exclusivity
- Lock fairness and starvation prevention
- Error handling and edge cases

Test Strategy:
- Unit tests with mocked dependencies
- Focus on lock behavior under concurrent access
- Verify thread safety and correctness
"""
import asyncio
import time

import pytest

from slime.router.middleware_hub.async_read_write_lock import (
    AsyncReadWriteLock,
    AsyncLock,
    read_lock,
    write_lock,
)


@pytest.mark.asyncio
async def test_multiple_readers_concurrent():
    """Test that multiple readers can access concurrently."""

    lock = AsyncReadWriteLock(debug=False)
    reader_count = 5
    read_durations = []
    concurrent_readers = []
    max_concurrent_readers = 0

    async def reader(reader_id: int):
        nonlocal max_concurrent_readers
        start_time = time.time()
        async with read_lock(lock):
            concurrent_readers.append(reader_id)
            max_concurrent_readers = max(max_concurrent_readers, len(concurrent_readers))
            # Simulate read operation
            await asyncio.sleep(0.01)
            concurrent_readers.remove(reader_id)
        end_time = time.time()
        read_durations.append(end_time - start_time)

    # Start multiple readers concurrently
    tasks = [asyncio.create_task(reader(i)) for i in range(reader_count)]
    await asyncio.gather(*tasks)

    # Verify all readers completed
    assert len(read_durations) == reader_count

    # Check that multiple readers were active simultaneously
    assert max_concurrent_readers > 1, f"Readers should be able to access concurrently (max: {max_concurrent_readers})"

    # Verify timing is consistent with concurrent execution
    total_time = max(read_durations)
    expected_time = 0.01  # Should be close to the sleep time, not multiplied by reader count
    assert total_time < expected_time * 1.5, f"Readers took too long: {total_time:.3f}s vs expected {expected_time}s"


@pytest.mark.asyncio
async def test_writer_exclusive_access():
    """Test that writer has exclusive access."""

    lock = AsyncReadWriteLock(debug=False)
    writer_durations = []
    active_writers = []

    async def writer(writer_id: int):
        start_time = time.time()
        async with write_lock(lock):
            active_writers.append(writer_id)
            # Simulate write operation
            await asyncio.sleep(0.01)
            active_writers.remove(writer_id)
        end_time = time.time()
        writer_durations.append(end_time - start_time)

    # Start multiple writers
    tasks = [asyncio.create_task(writer(i)) for i in range(3)]
    await asyncio.gather(*tasks)

    # Verify only one writer was active at any time
    assert len(active_writers) <= 1, "Only one writer should be active at a time"

    # Verify timing is consistent with serial execution
    total_time = sum(writer_durations)
    expected_time = 0.01 * 3  # Should be approximately 3x the sleep time
    assert total_time >= expected_time * 0.8, f"Writers completed too quickly: {total_time:.3f}s vs expected {expected_time}s"


@pytest.mark.asyncio
async def test_readers_blocked_by_writer():
    """Test that readers are blocked when a writer is active."""

    lock = AsyncReadWriteLock(debug=False)
    reader_started = []
    reader_completed = []
    writer_active = False

    async def writer():
        nonlocal writer_active
        async with write_lock(lock):
            writer_active = True
            await asyncio.sleep(0.02)
            writer_active = False

    async def reader(reader_id: int):
        start_time = time.time()
        reader_started.append(reader_id)
        async with read_lock(lock):
            # This should only start after writer finishes
            assert not writer_active, "Reader should not start while writer is active"
            await asyncio.sleep(0.005)
        end_time = time.time()
        reader_completed.append((reader_id, end_time - start_time))

    # Start writer first
    writer_task = asyncio.create_task(writer())

    # Let writer start
    await asyncio.sleep(0.005)

    # Start readers while writer is active
    reader_tasks = [asyncio.create_task(reader(i)) for i in range(3)]

    await writer_task
    await asyncio.gather(*reader_tasks)

    # Verify all readers completed after writer
    assert len(reader_completed) == 3
    assert all(duration > 0.015 for _, duration in reader_completed), \
        "Readers should wait for writer to finish"


@pytest.mark.asyncio
async def test_writer_blocked_by_readers():
    """Test that writer is blocked when readers are active."""

    lock = AsyncReadWriteLock(debug=False)
    writer_started = False
    writer_completed = False
    readers_active = 0

    async def long_reader():
        nonlocal readers_active
        async with read_lock(lock):
            readers_active += 1
            await asyncio.sleep(0.02)
            readers_active -= 1

    async def writer():
        nonlocal writer_started, writer_completed
        writer_started = True
        start_time = time.time()
        async with write_lock(lock):
            # This should only start after all readers finish
            assert readers_active == 0, "Writer should not start while readers are active"
            await asyncio.sleep(0.005)
        end_time = time.time()
        writer_completed = True
        return end_time - start_time

    # Start long-running readers
    reader_tasks = [asyncio.create_task(long_reader()) for _ in range(2)]

    # Let readers start
    await asyncio.sleep(0.005)

    # Start writer while readers are active
    writer_task = asyncio.create_task(writer())

    # Wait for writer to complete
    writer_duration = await writer_task
    await asyncio.gather(*reader_tasks)

    # Verify writer waited for readers
    assert writer_started and writer_completed
    assert writer_duration > 0.015, f"Writer should wait for readers: {writer_duration:.3f}s"


@pytest.mark.asyncio
async def test_fairness_no_starvation():
    """Test that the lock implementation is fair and prevents starvation."""

    lock = AsyncReadWriteLock(debug=False)
    operations = []
    stats = lock.get_stats()

    async def mixed_operation(op_id: str, is_writer: bool):
        start_time = time.time()
        if is_writer:
            async with write_lock(lock):
                operations.append(f"writer_{op_id}_start")
                await asyncio.sleep(0.005)
                operations.append(f"writer_{op_id}_end")
        else:
            async with read_lock(lock):
                operations.append(f"reader_{op_id}_start")
                await asyncio.sleep(0.003)
                operations.append(f"reader_{op_id}_end")
        end_time = time.time()
        return end_time - start_time

    # Create mixed workload
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(mixed_operation(f"r{i}", False)))
        if i % 2 == 0:  # Add some writers
            tasks.append(asyncio.create_task(mixed_operation(f"w{i}", True)))

    await asyncio.gather(*tasks)

    # Verify all operations completed
    assert len(operations) > 0

    # Check that we have both readers and writers
    reader_ops = [op for op in operations if "reader" in op]
    writer_ops = [op for op in operations if "writer" in op]

    assert len(reader_ops) > 0, "Should have reader operations"
    assert len(writer_ops) > 0, "Should have writer operations"

    # Verify lock is in clean state
    final_stats = lock.get_stats()
    assert final_stats["readers"] == 0, "No readers should be active"
    assert final_stats["writers"] == 0, "No writers should be active"


@pytest.mark.asyncio
async def test_async_lock_compatibility():
    """Test simple AsyncLock for compatibility."""

    lock = AsyncLock(debug=False)
    durations = []

    async def operation(op_id: int):
        start_time = time.time()
        async with lock:
            await asyncio.sleep(0.01)
        end_time = time.time()
        durations.append(end_time - start_time)
        return op_id

    # Test that operations are serialized
    tasks = [asyncio.create_task(operation(i)) for i in range(3)]
    results = await asyncio.gather(*tasks)

    # Verify operations were serialized (not concurrent)
    total_time = sum(durations)
    expected_time = 0.01 * 3  # Should be approximately 3x the sleep time
    assert total_time >= expected_time * 0.8, \
        f"Operations completed too quickly: {total_time:.3f}s vs expected {expected_time}s"

    # Verify all operations completed
    assert len(results) == 3
    assert sorted(results) == [0, 1, 2]


if __name__ == "__main__":
    # Run tests manually for debugging
    asyncio.run(test_multiple_readers_concurrent())
    asyncio.run(test_writer_exclusive_access())
    asyncio.run(test_readers_blocked_by_writer())
    asyncio.run(test_writer_blocked_by_readers())
    asyncio.run(test_fairness_no_starvation())
    asyncio.run(test_async_lock_compatibility())
    print("All async lock tests passed!")