"""Performance comparison between RLock and async read-write lock implementations."""
import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from slime.router.middleware_hub.async_read_write_lock import AsyncReadWriteLock
from slime.router.middleware_hub.radix_tree import StringRadixTrie


class TestPerformanceComparison:
    """Compare performance between RLock and async implementations."""

    @pytest.mark.asyncio
    async def test_concurrent_read_performance(self):
        """Test that async implementation provides better concurrent read performance."""

        # Create both sync and async versions
        sync_tree = StringRadixTrie(max_cache_size=1000, verbose=False)
        async_tree = StringRadixTrie(max_cache_size=1000, verbose=False)

        # Pre-populate both trees with the same data
        test_data = [
            ("Hello world", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1]),
            ("Test case", [4, 5, 6], [-0.4, -0.5, -0.6], [1, 1, 1]),
            ("Another test", [7, 8, 9], [-0.7, -0.8, -0.9], [0, 1, 0]),
        ]

        for text, tokens, logp, mask in test_data:
            sync_tree.insert(text, tokens, logp, mask, weight_version=1)
            await async_tree.insert_async(text, tokens, logp, mask, weight_version=1)

        # Test concurrent reads
        num_concurrent_reads = 20
        sync_times = []
        async_times = []

        # Measure sync performance (using asyncio.to_thread to simulate concurrent access)
        async def sync_read():
            start_time = time.time()
            result = await asyncio.to_thread(sync_tree.find_longest_prefix, "Hello world")
            end_time = time.time()
            sync_times.append(end_time - start_time)
            return result

        # Measure async performance
        async def async_read():
            start_time = time.time()
            result = await async_tree.find_longest_prefix_async("Hello world")
            end_time = time.time()
            async_times.append(end_time - start_time)
            return result

        # Run sync reads
        sync_tasks = [asyncio.create_task(sync_read()) for _ in range(num_concurrent_reads)]
        sync_results = await asyncio.gather(*sync_tasks)

        # Run async reads
        async_tasks = [asyncio.create_task(async_read()) for _ in range(num_concurrent_reads)]
        async_results = await asyncio.gather(*async_tasks)

        # Verify correctness - both should return the same results
        for sync_result, async_result in zip(sync_results, async_results):
            assert sync_result.token_ids == async_result.token_ids
            assert sync_result.matched_prefix == async_result.matched_prefix

        # Performance analysis
        avg_sync_time = sum(sync_times) / len(sync_times)
        avg_async_time = sum(async_times) / len(async_times)
        max_sync_time = max(sync_times)
        max_async_time = max(async_times)

        print(f"\nConcurrent Read Performance ({num_concurrent_reads} reads):")
        print(f"Sync RLock - Avg: {avg_sync_time*1000:.2f}ms, Max: {max_sync_time*1000:.2f}ms")
        print(f"Async RWLock - Avg: {avg_async_time*1000:.2f}ms, Max: {max_async_time*1000:.2f}ms")
        print(f"Performance improvement: {((max_sync_time - max_async_time) / max_sync_time * 100):.1f}%")

        # Async should be faster for concurrent reads due to better concurrency
        # The exact improvement depends on system load, but we expect some benefit
        assert avg_async_time <= avg_sync_time * 1.1, "Async should be competitive or better"

        # Both should return correct results
        assert all(result.token_ids == [1, 2, 3] for result in async_results)
        assert all(result.matched_prefix == "Hello world" for result in async_results)

    @pytest.mark.asyncio
    async def test_mixed_read_write_performance(self):
        """Test performance with mixed read/write operations."""

        async_tree = StringRadixTrie(max_cache_size=1000, verbose=False)

        # Initial data
        await async_tree.insert_async("Base", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1)

        operation_times = []

        async def reader(reader_id: int):
            start_time = time.time()
            result = await async_tree.find_longest_prefix_async("Base")
            end_time = time.time()
            operation_times.append(("read", reader_id, end_time - start_time))
            return result

        async def writer(writer_id: int):
            start_time = time.time()
            result = await async_tree.insert_async(f"Data{writer_id}", [10 + writer_id], [-0.5], [1], weight_version=2)
            end_time = time.time()
            operation_times.append(("write", writer_id, end_time - start_time))
            return result

        # Run mixed operations
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(reader(i)))
            tasks.append(asyncio.create_task(writer(i)))

        results = await asyncio.gather(*tasks)

        # Analyze performance
        read_times = [t for op, _, t in operation_times if op == "read"]
        write_times = [t for op, _, t in operation_times if op == "write"]

        avg_read_time = sum(read_times) / len(read_times)
        avg_write_time = sum(write_times) / len(write_times)

        print(f"\nMixed Read/Write Performance:")
        print(f"Read operations - Avg: {avg_read_time*1000:.2f}ms, Count: {len(read_times)}")
        print(f"Write operations - Avg: {avg_write_time*1000:.2f}ms, Count: {len(write_times)}")

        # Verify all operations completed successfully
        read_results = results[::2]  # Every other result (readers first)
        write_results = results[1::2]  # Every other result (writers second)

        assert all(r.token_ids == [1, 2] for r in read_results if r is not None)
        assert all(r is True for r in write_results)

    @pytest.mark.asyncio
    async def test_middleware_performance_comparison(self):
        """Test middleware performance with async cache operations."""
        from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware

        # Create middleware with mocked dependencies
        mock_router = Mock()
        mock_router.args = Mock()
        mock_router.args.hf_checkpoint = "test-checkpoint"
        mock_router.args.radix_tree_max_size = 1000  # Set real value
        mock_router.args.verbose = False  # Set real value
        mock_router.verbose = False

        with patch('slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
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

            # Test concurrent cache retrievals
            num_retrievals = 50
            retrieval_times = []

            async def retrieve_cache():
                start_time = time.time()
                tokens, logprobs, loss_mask, versions = await middleware._retrieve_cache("Hello world")
                end_time = time.time()
                retrieval_times.append(end_time - start_time)
                return tokens

            # Run concurrent retrievals
            tasks = [asyncio.create_task(retrieve_cache()) for _ in range(num_retrievals)]
            results = await asyncio.gather(*tasks)

            # Performance analysis
            avg_time = sum(retrieval_times) / len(retrieval_times)
            max_time = max(retrieval_times)
            min_time = min(retrieval_times)
            total_time = sum(retrieval_times)

            print(f"\nMiddleware Cache Retrieval Performance ({num_retrievals} retrievals):")
            print(f"Average: {avg_time*1000:.2f}ms")
            print(f"Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
            print(f"Total: {total_time*1000:.2f}ms")
            print(f"Throughput: {num_retrievals/total_time:.1f} retrievals/second")

            # Verify correctness
            assert all(result == [1, 2, 3] for result in results)

            # Performance should be reasonable for concurrent operations
            # Async operations should allow good concurrency
            assert avg_time < 0.01, f"Average retrieval time too high: {avg_time*1000:.2f}ms"
            assert max_time < 0.05, f"Maximum retrieval time too high: {max_time*1000:.2f}ms"

    @pytest.mark.asyncio
    async def test_event_loop_non_blocking(self):
        """Test that async operations don't block the event loop."""

        async_tree = StringRadixTrie(max_cache_size=1000, verbose=False)
        await async_tree.insert_async("Test", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1)

        # Background task that should continue running
        background_counter = 0
        background_timestamps = []

        async def background_task():
            nonlocal background_counter
            for i in range(100):
                background_counter += 1
                background_timestamps.append(time.time())
                await asyncio.sleep(0.001)  # Small delay

        # Cache operations that should not block background task
        async def cache_operations():
            for i in range(10):
                result = await async_tree.find_longest_prefix_async("Test")
                await asyncio.sleep(0.002)  # Simulate some processing

        # Start both tasks
        bg_task = asyncio.create_task(background_task())
        cache_task = asyncio.create_task(cache_operations())

        # Wait for completion
        await asyncio.gather(bg_task, cache_task)

        # Verify background task continued running
        assert background_counter == 100, f"Background task should have run 100 times, got {background_counter}"

        # Analyze timing distribution
        if len(background_timestamps) >= 2:
            intervals = [background_timestamps[i+1] - background_timestamps[i]
                        for i in range(len(background_timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            max_interval = max(intervals)

            print(f"\nEvent Loop Non-Blocking Test:")
            print(f"Background task ran {background_counter} times")
            print(f"Average interval: {avg_interval*1000:.2f}ms")
            print(f"Maximum interval: {max_interval*1000:.2f}ms")

            # Intervals should be relatively consistent (no major blocking)
            # Allow some variance due to system scheduling
            assert max_interval < avg_interval * 3, f"Event loop appears to be blocked: max interval {max_interval*1000:.2f}ms vs avg {avg_interval*1000:.2f}ms"


if __name__ == "__main__":
    # Run performance tests manually
    asyncio.run(TestPerformanceComparison().test_concurrent_read_performance())
    asyncio.run(TestPerformanceComparison().test_mixed_read_write_performance())
    asyncio.run(TestPerformanceComparison().test_middleware_performance_comparison())
    asyncio.run(TestPerformanceComparison().test_event_loop_non_blocking())
    print("\nAll performance comparison tests completed!")