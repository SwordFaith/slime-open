"""
Unit tests for async RadixTree implementation.

Tests cover:
- Async insert and retrieval operations
- Concurrent access safety
- Performance under async workloads
- Integration with async middleware
- Error handling in async context

Test Strategy:
- Unit tests with real async operations
- Mock external dependencies when needed
- Focus on async correctness and performance
"""
import asyncio
import time

import pytest

from slime.router.middleware_hub.radix_tree import StringRadixTrie


@pytest.mark.asyncio
async def test_async_find_longest_prefix():
    """Test async find_longest_prefix method."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Insert test data
    await tree.insert_async("Hello", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], weight_version=1)
    await tree.insert_async("World", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1], weight_version=2)

    # Test finding exact match
    result = await tree.find_longest_prefix_async("Hello")
    assert result.matched_prefix == "Hello"
    assert result.token_ids == [1, 2, 3]
    assert result.remaining_string == ""

    # Test finding partial match
    result = await tree.find_longest_prefix_async("HelloWorld")
    assert result.matched_prefix == "Hello"
    assert result.token_ids == [1, 2, 3]
    assert result.remaining_string == "World"

    # Test no match
    result = await tree.find_longest_prefix_async("Unknown")
    assert result.matched_prefix == ""
    assert result.token_ids == []
    assert result.remaining_string == "Unknown"


@pytest.mark.asyncio
async def test_async_insert():
    """Test async insert method."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Test successful insertion
    result = await tree.insert_async("Test", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 1, 1], weight_version=1)
    assert result is True

    # Verify insertion worked
    find_result = await tree.find_longest_prefix_async("Test")
    assert find_result.matched_prefix == "Test"
    assert find_result.token_ids == [1, 2, 3]

    # Test failed insertion (empty text)
    result = await tree.insert_async("", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 1, 1])
    assert result is False


@pytest.mark.asyncio
async def test_concurrent_async_reads():
    """Test that multiple async reads can happen concurrently."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Insert test data
    await tree.insert_async("Hello", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], weight_version=1)
    await tree.insert_async("World", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1], weight_version=2)

    read_count = 5
    results = []

    async def reader(reader_id: int):
        start_time = time.time()
        result = await tree.find_longest_prefix_async("Hello")
        end_time = time.time()
        results.append({
            "reader_id": reader_id,
            "result": result,
            "duration": end_time - start_time
        })

    # Start multiple readers concurrently
    tasks = [asyncio.create_task(reader(i)) for i in range(read_count)]
    await asyncio.gather(*tasks)

    # Verify all readers got the same result
    assert len(results) == read_count
    for result in results:
        assert result["result"].matched_prefix == "Hello"
        assert result["result"].token_ids == [1, 2, 3]

    # Verify concurrent execution (total time should be close to single read time)
    total_time = max(result["duration"] for result in results)
    # With proper async read-write lock, reads should be concurrent
    # This is a rough check - exact timing depends on system load
    assert total_time < 0.1, f"Reads took too long: {total_time:.3f}s"


@pytest.mark.asyncio
async def test_concurrent_read_write():
    """Test concurrent reads and writes with proper synchronization."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Initial data
    await tree.insert_async("Base", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1)

    read_results = []
    write_results = []

    async def reader(reader_id: int):
        start_time = time.time()
        result = await tree.find_longest_prefix_async("Base")
        end_time = time.time()
        read_results.append({
            "reader_id": reader_id,
            "result": result,
            "duration": end_time - start_time
        })

    async def writer(writer_id: int):
        start_time = time.time()
        result = await tree.insert_async(f"Data{writer_id}", [10 + writer_id], [-0.5], [1], weight_version=2)
        end_time = time.time()
        write_results.append({
            "writer_id": writer_id,
            "result": result,
            "duration": end_time - start_time
        })

    # Start mixed operations
    tasks = []
    for i in range(3):
        tasks.append(asyncio.create_task(reader(i)))
        tasks.append(asyncio.create_task(writer(i)))

    await asyncio.gather(*tasks)

    # Verify all operations completed
    assert len(read_results) == 3
    assert len(write_results) == 3

    # Verify reads were successful
    for read_result in read_results:
        assert read_result["result"].matched_prefix == "Base"
        assert read_result["result"].token_ids == [1, 2]

    # Verify writes were successful
    for write_result in write_results:
        assert write_result["result"] is True


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test that sync methods still work after adding async methods."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Test sync insert
    result = tree.insert("Hello", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], weight_version=1)
    assert result is True

    # Test sync find
    sync_result = tree.find_longest_prefix("Hello")
    assert sync_result.matched_prefix == "Hello"
    assert sync_result.token_ids == [1, 2, 3]

    # Test async insert on same tree
    await tree.insert_async("World", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1], weight_version=2)

    # Test async find
    async_result = await tree.find_longest_prefix_async("World")
    assert async_result.matched_prefix == "World"
    assert async_result.token_ids == [4, 5, 6]

    # Verify both can access the same data
    sync_result2 = tree.find_longest_prefix("World")
    assert sync_result2.token_ids == [4, 5, 6]

    async_result2 = await tree.find_longest_prefix_async("Hello")
    assert async_result2.token_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_mixed_sync_async_usage():
    """Test mixing sync and async usage patterns."""
    tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Mix sync and async operations
    sync_result = tree.insert("Sync", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1)
    assert sync_result is True

    async_result = await tree.insert_async("Async", [3, 4], [-0.3, -0.4], [1, 1], weight_version=2)
    assert async_result is True

    # Both methods should find both insertions
    sync_find = tree.find_longest_prefix("Sync")
    assert sync_find.matched_prefix == "Sync"

    async_find = await tree.find_longest_prefix_async("Async")
    assert async_find.matched_prefix == "Async"

    # Cross-access should also work
    async_find_sync = await tree.find_longest_prefix_async("Sync")
    assert async_find_sync.matched_prefix == "Sync"

    sync_find_async = tree.find_longest_prefix("Async")
    assert sync_find_async.matched_prefix == "Async"


if __name__ == "__main__":
    # Run tests manually for debugging
    asyncio.run(test_async_find_longest_prefix())
    asyncio.run(test_async_insert())
    asyncio.run(test_concurrent_async_reads())
    asyncio.run(test_concurrent_read_write())
    asyncio.run(test_backward_compatibility())
    asyncio.run(test_mixed_sync_async_usage())
    print("All async RadixTree tests passed!")