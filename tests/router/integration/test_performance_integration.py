#!/usr/bin/env python3
"""
Performance integration tests for radix tree.
性能集成测试，比较同步和异步版本的性能差异
"""

import asyncio
import time
import random
import pytest
from concurrent.futures import ThreadPoolExecutor
from slime.router.core.radix_tree import StringRadixTrie


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text: str, add_special_tokens: bool = True):
        tokens = [ord(c) % 1000 for c in text]
        return {"input_ids": tokens}


class TestPerformanceIntegration:
    """Test performance integration between sync and async versions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = MockTokenizer()

    def generate_test_data(self, num_items: int):
        """Generate test data."""
        data = []
        for i in range(num_items):
            text = f"test_{i}_{random.randint(1000, 9999)}"
            tokens = [random.randint(1, 999) for _ in range(random.randint(1, 5))]
            logp = [random.random() for _ in tokens]
            loss_mask = [1] * len(tokens)
            weight_version = random.randint(1, 3)
            data.append((text, tokens, logp, loss_mask, weight_version))
        return data

    @pytest.mark.asyncio
    async def test_insert_performance_comparison(self):
        """Test insert performance comparison between sync and async."""
        print("🚀 测试insert性能对比")

        num_operations = 500  # Reduced for integration test
        test_data = self.generate_test_data(num_operations)

        # Sync version test
        sync_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)

        start_time = time.time()
        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
        sync_time = time.time() - start_time

        assert sync_trie.total_entries > 0, "Sync trie should have entries after insertion"
        assert sync_trie.cur_cache_size > 0, "Sync trie should have cache size > 0"

        # Async version test
        async_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)

        start_time = time.time()
        for text, tokens, logp, loss_mask, weight_version in test_data:
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
        async_time = time.time() - start_time

        assert async_trie.total_entries > 0, "Async trie should have entries after insertion"
        assert async_trie.cur_cache_size > 0, "Async trie should have cache size > 0"

        # Verify consistency
        assert sync_trie.total_entries == async_trie.total_entries, "Total entries should match"
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes should match"

        # Performance should be reasonable (within 10x for integration test)
        performance_ratio = max(async_time, sync_time) / min(async_time, sync_time)
        assert performance_ratio < 10.0, f"Performance ratio too high: {performance_ratio:.2f}x"

        print(f"   Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s")
        print(f"   Performance ratio: {performance_ratio:.2f}x")

    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self):
        """Test concurrent access performance."""
        print("🚀 测试并发访问性能")

        num_threads = 5  # Reduced for integration test
        operations_per_thread = 50
        test_data_sets = [self.generate_test_data(operations_per_thread) for _ in range(num_threads)]

        # Sync version concurrent test
        sync_trie = StringRadixTrie(max_cache_size=500, tokenizer=self.tokenizer, verbose=False)

        def sync_worker(data_set):
            """Sync worker thread."""
            for text, tokens, logp, loss_mask, weight_version in data_set:
                sync_trie.insert(text, tokens, logp, loss_mask, weight_version)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(sync_worker, data_set) for data_set in test_data_sets]
            for future in futures:
                future.result()
        sync_concurrent_time = time.time() - start_time

        assert sync_trie.total_entries > 0, "Sync trie should have entries after concurrent insertion"

        # Async version concurrent test
        async_trie = StringRadixTrie(max_cache_size=500, tokenizer=self.tokenizer, verbose=False)

        async def async_worker(data_set):
            """Async worker coroutine."""
            for text, tokens, logp, loss_mask, weight_version in data_set:
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        start_time = time.time()
        tasks = [async_worker(data_set) for data_set in test_data_sets]
        await asyncio.gather(*tasks)
        async_concurrent_time = time.time() - start_time

        assert async_trie.total_entries > 0, "Async trie should have entries after concurrent insertion"

        # Verify consistency
        assert sync_trie.total_entries == async_trie.total_entries, "Total entries should match after concurrent operations"

        # Performance should be reasonable
        performance_ratio = max(async_concurrent_time, sync_concurrent_time) / min(async_concurrent_time, sync_concurrent_time)
        assert performance_ratio < 15.0, f"Concurrent performance ratio too high: {performance_ratio:.2f}x"

        print(f"   Sync concurrent time: {sync_concurrent_time:.4f}s")
        print(f"   Async concurrent time: {async_concurrent_time:.4f}s")
        print(f"   Concurrent performance ratio: {performance_ratio:.2f}x")

    @pytest.mark.asyncio
    async def test_mixed_operations_performance(self):
        """Test mixed operations performance."""
        print("🚀 测试混合操作性能")

        num_operations = 300  # Reduced for integration test
        test_data = self.generate_test_data(num_operations)

        # Sync version mixed operations
        sync_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)

        start_time = time.time()
        for i, (text, tokens, logp, loss_mask, weight_version) in enumerate(test_data):
            if i % 4 == 0:
                # Insert operation
                sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            elif i % 4 == 1:
                # Find operation
                sync_trie.find_longest_prefix(text)
            elif i % 4 == 2:
                # Get tokenization
                try:
                    sync_trie.get_or_create_tokenization(text)
                except Exception:
                    pass  # May fail for invalid data
            else:
                # Stats operation
                sync_trie.get_stats()
        sync_mixed_time = time.time() - start_time

        # Async version mixed operations
        async_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)

        start_time = time.time()
        for i, (text, tokens, logp, loss_mask, weight_version) in enumerate(test_data):
            if i % 4 == 0:
                # Insert operation
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
            elif i % 4 == 1:
                # Find operation
                await async_trie.find_longest_prefix_async(text)
            elif i % 4 == 2:
                # Get tokenization
                try:
                    await async_trie.get_or_create_tokenization_async(text)
                except Exception:
                    pass  # May fail for invalid data
            else:
                # Stats operation
                await async_trie.get_stats_async()
        async_mixed_time = time.time() - start_time

        # Verify consistency
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes should match after mixed operations"

        # Performance should be reasonable
        performance_ratio = max(async_mixed_time, sync_mixed_time) / min(async_mixed_time, sync_mixed_time)
        assert performance_ratio < 12.0, f"Mixed operations performance ratio too high: {performance_ratio:.2f}x"

        print(f"   Sync mixed time: {sync_mixed_time:.4f}s, Async mixed time: {async_mixed_time:.4f}s")
        print(f"   Mixed operations performance ratio: {performance_ratio:.2f}x")

    @pytest.mark.asyncio
    async def test_memory_usage_consistency(self):
        """Test memory usage consistency between sync and async versions."""
        print("🧠 测试内存使用一致性")

        test_data = self.generate_test_data(200)

        # Sync version
        sync_trie = StringRadixTrie(max_cache_size=100, tokenizer=self.tokenizer, verbose=False)
        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)

        sync_stats = sync_trie.get_stats()

        # Async version
        async_trie = StringRadixTrie(max_cache_size=100, tokenizer=self.tokenizer, verbose=False)
        for text, tokens, logp, loss_mask, weight_version in test_data:
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        async_stats = await async_trie.get_stats_async()

        # Verify memory usage consistency
        assert sync_stats['total_entries'] == async_stats['total_entries'], "Total entries should match"
        assert sync_stats['cur_cache_size'] == async_stats['cur_cache_size'], "Cache sizes should match"

        # Memory usage should be reasonable (within reasonable bounds)
        assert sync_stats['cur_cache_size'] <= sync_trie.max_cache_size * 2, "Sync cache size should be reasonable"
        assert async_stats['cur_cache_size'] <= async_trie.max_cache_size * 2, "Async cache size should be reasonable"

        print(f"   Sync memory: {sync_stats['cur_cache_size']} cache, {sync_stats['total_entries']} entries")
        print(f"   Async memory: {async_stats['cur_cache_size']} cache, {async_stats['total_entries']} entries")

    @pytest.mark.asyncio
    async def test_gc_performance_consistency(self):
        """Test GC performance consistency."""
        print("🗑️ 测试GC性能一致性")

        # Insert data with different versions
        test_data = []
        for i in range(100):
            text = f"gc_test_{i}"
            tokens = [i, i+1]
            logp = [0.1, 0.2]
            loss_mask = [1, 1]
            weight_version = i % 5 + 1  # Versions 1-5
            test_data.append((text, tokens, logp, loss_mask, weight_version))

        # Sync version GC test
        sync_trie = StringRadixTrie(max_cache_size=50, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)

        initial_sync_entries = sync_trie.total_entries

        start_time = time.time()
        sync_removed = sync_trie.gc_by_weight_version(4)
        sync_gc_time = time.time() - start_time

        final_sync_entries = sync_trie.total_entries

        # Async version GC test
        async_trie = StringRadixTrie(max_cache_size=50, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        for text, tokens, logp, loss_mask, weight_version in test_data:
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        initial_async_entries = async_trie.total_entries

        start_time = time.time()
        async_removed = await async_trie.gc_by_weight_version_async(4)
        async_gc_time = time.time() - start_time

        final_async_entries = async_trie.total_entries

        # Verify GC consistency
        assert initial_sync_entries == initial_async_entries, "Initial entries should match"
        assert final_sync_entries == final_async_entries, "Final entries should match after GC"
        assert sync_removed == async_removed, "GC removal count should match"

        # GC performance should be reasonable
        assert sync_gc_time < 1.0, f"Sync GC too slow: {sync_gc_time:.4f}s"
        assert async_gc_time < 1.0, f"Async GC too slow: {async_gc_time:.4f}s"

        # Performance ratio should be reasonable
        gc_performance_ratio = max(async_gc_time, sync_gc_time) / min(async_gc_time, sync_gc_time)
        assert gc_performance_ratio < 10.0, f"GC performance ratio too high: {gc_performance_ratio:.2f}x"

        print(f"   Sync GC: {sync_gc_time:.4f}s, removed {sync_removed} entries")
        print(f"   Async GC: {async_gc_time:.4f}s, removed {async_removed} entries")
        print(f"   GC performance ratio: {gc_performance_ratio:.2f}x")

    @pytest.mark.asyncio
    async def test_scalability_comparison(self):
        """Test scalability comparison between sync and async versions."""
        print("📈 测试可扩展性对比")

        # Test with increasing data sizes
        data_sizes = [50, 100, 200]
        sync_times = []
        async_times = []

        for size in data_sizes:
            test_data = self.generate_test_data(size)

            # Sync version
            sync_trie = StringRadixTrie(max_cache_size=size * 2, tokenizer=self.tokenizer, verbose=False)
            start_time = time.time()
            for text, tokens, logp, loss_mask, weight_version in test_data:
                sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            sync_time = time.time() - start_time
            sync_times.append(sync_time)

            # Async version
            async_trie = StringRadixTrie(max_cache_size=size * 2, tokenizer=self.tokenizer, verbose=False)
            start_time = time.time()
            for text, tokens, logp, loss_mask, weight_version in test_data:
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
            async_time = time.time() - start_time
            async_times.append(async_time)

            # Verify consistency at each size
            assert sync_trie.total_entries == async_trie.total_entries, f"Entries mismatch for size {size}"

        # Check scalability trend (time should increase roughly linearly)
        for i in range(1, len(data_sizes)):
            sync_growth_ratio = sync_times[i] / sync_times[i-1] if sync_times[i-1] > 0 else 1
            async_growth_ratio = async_times[i] / async_times[i-1] if async_times[i-1] > 0 else 1

            # Growth should be reasonable (less than 3x for 2x data increase)
            assert sync_growth_ratio < 3.0, f"Sync growth ratio too high: {sync_growth_ratio:.2f}x"
            assert async_growth_ratio < 3.0, f"Async growth ratio too high: {async_growth_ratio:.2f}x"

        print(f"   Data sizes: {data_sizes}")
        print(f"   Sync times: {[f'{t:.4f}s' for t in sync_times]}")
        print(f"   Async times: {[f'{t:.4f}s' for t in async_times]}")

    @pytest.mark.asyncio
    async def test_lock_contention_performance(self):
        """Test performance under lock contention scenarios."""
        print("🔒 测试锁竞争性能")

        num_tasks = 8  # Reduced for integration test
        operations_per_task = 30
        shared_data = self.generate_test_data(operations_per_task)

        # Test with shared trie (high contention)
        sync_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=200, tokenizer=self.tokenizer, verbose=False)

        # Sync version with contention
        def sync_contentious_worker(task_id):
            for text, tokens, logp, loss_mask, weight_version in shared_data:
                # Add some variety to increase contention
                modified_text = f"{text}_task{task_id}"
                sync_trie.insert(modified_text, tokens, logp, loss_mask, weight_version)
                # Add some read operations
                sync_trie.find_longest_prefix(modified_text)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [executor.submit(sync_contentious_worker, i) for i in range(num_tasks)]
            for future in futures:
                future.result()
        sync_contention_time = time.time() - start_time

        # Async version with contention
        async def async_contentious_worker(task_id):
            for text, tokens, logp, loss_mask, weight_version in shared_data:
                modified_text = f"{text}_task{task_id}"
                await async_trie.insert_async(modified_text, tokens, logp, loss_mask, weight_version)
                await async_trie.find_longest_prefix_async(modified_text)

        start_time = time.time()
        tasks = [async_contentious_worker(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        async_contention_time = time.time() - start_time

        # Both should complete successfully
        assert sync_trie.total_entries > 0, "Sync trie should have entries after contention test"
        assert async_trie.total_entries > 0, "Async trie should have entries after contention test"

        # Performance under contention should be reasonable
        contention_ratio = max(async_contention_time, sync_contention_time) / min(async_contention_time, sync_contention_time)
        assert contention_ratio < 20.0, f"Contention performance ratio too high: {contention_ratio:.2f}x"

        print(f"   Sync contention time: {sync_contention_time:.4f}s")
        print(f"   Async contention time: {async_contention_time:.4f}s")
        print(f"   Contention performance ratio: {contention_ratio:.2f}x")


if __name__ == "__main__":
    # Run standalone test
    async def run_all_tests():
        test_instance = TestPerformanceIntegration()
        test_instance.setup_method()

        try:
            await test_instance.test_insert_performance_comparison()
            await test_instance.test_concurrent_access_performance()
            await test_instance.test_mixed_operations_performance()
            await test_instance.test_memory_usage_consistency()
            await test_instance.test_gc_performance_consistency()
            await test_instance.test_scalability_comparison()
            await test_instance.test_lock_contention_performance()
            print("\n🎉 All performance integration tests passed!")
            return True
        except Exception as e:
            print(f"\n💥 Performance integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)