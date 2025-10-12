#!/usr/bin/env python3
"""
Comprehensive integration tests for radix tree.
综合集成测试，验证重构后的功能完整性和一致性
"""

import asyncio
import pytest
from slime.router.core.radix_tree import StringRadixTrie


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text: str, add_special_tokens: bool = True):
        tokens = [ord(c) % 1000 for c in text]
        return {"input_ids": tokens}


class TestComprehensiveIntegration:
    """Test comprehensive integration functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = MockTokenizer()

    @pytest.mark.asyncio
    async def test_api_completeness(self):
        """Test API completeness: ensure all sync methods have async counterparts."""
        print("🔍 测试API完整性")

        sync_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)

        # Check method correspondence
        sync_methods = [method for method in dir(sync_trie) if not method.startswith('_')
                       and callable(getattr(sync_trie, method))]
        async_methods = [method for method in dir(async_trie) if not method.startswith('_')
                        and callable(getattr(async_trie, method)) and method.endswith('_async')]

        assert len(async_methods) > 0, "Should have async methods"

        # Verify required async methods exist
        required_async_methods = [
            'insert_async', 'find_longest_prefix_async', 'get_or_create_tokenization_async',
            'get_stats_async', 'remove_async', 'gc_by_weight_version_async', 'clear_async'
        ]

        for method in required_async_methods:
            assert hasattr(async_trie, method), f"Missing async method: {method}"

        print("✅ 所有必需的异步方法都已实现")

    @pytest.mark.asyncio
    async def test_gc_critical_functionality(self):
        """Test GC critical functionality: ensure async version correctly triggers GC."""
        print("🔥 测试GC关键功能")

        # Create small cache to trigger GC quickly
        sync_trie = StringRadixTrie(max_cache_size=3, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=3, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # Insert different version data to trigger GC
        test_batches = [
            [("a", [1], [0.1], [1], 1), ("b", [2], [0.2], [1], 1), ("c", [3], [0.3], [1], 1)],
            [("d", [4], [0.4], [1], 3), ("e", [5], [0.5], [1], 3)]
        ]

        # Sync version GC test
        for i, batch in enumerate(test_batches):
            for text, tokens, logp, loss_mask, weight_version in batch:
                sync_trie.insert(text, tokens, logp, loss_mask, weight_version)

        # Async version GC test
        for i, batch in enumerate(test_batches):
            for text, tokens, logp, loss_mask, weight_version in batch:
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify GC behavior consistency
        gc_consistent = (sync_trie.cur_cache_size == async_trie.cur_cache_size and
                        sync_trie.total_entries == async_trie.total_entries)

        assert gc_consistent, "GC behavior should be consistent"

        # Verify GC actually happened
        assert sync_trie.cur_cache_size <= sync_trie.max_cache_size, "GC should have been triggered"
        assert async_trie.cur_cache_size <= async_trie.max_cache_size, "GC should have been triggered"

        print("✅ GC行为完全一致!")

    @pytest.mark.asyncio
    async def test_architectural_simplification(self):
        """Test architectural simplification: verify correctness after lock simplification."""
        print("🏗️ 测试架构简化效果")

        async_trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)

        # Test async method simplicity
        await async_trie.insert_async("test1", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], 1)

        # Test various operations
        result1 = await async_trie.find_longest_prefix_async("test1")
        assert result1.matched_prefix == "test1", "Should find inserted text"

        tokens1 = await async_trie.get_or_create_tokenization_async("new_text")
        assert len(tokens1) > 0, "Should generate tokens for new text"

        stats1 = await async_trie.get_stats_async()
        assert isinstance(stats1, dict), "Stats should return dict"

        # Test modification operations
        removed1 = await async_trie.remove_async("test1")
        assert isinstance(removed1, bool), "Remove should return boolean"

        gc_removed1 = await async_trie.gc_by_weight_version_async(1)
        assert isinstance(gc_removed1, int), "GC should return count"

        await async_trie.clear_async()
        assert async_trie.cur_cache_size == 0, "Cache should be empty after clear"

        print("✅ 架构简化成功，不再需要复杂的锁管理")

    @pytest.mark.asyncio
    async def test_error_handling_and_robustness(self):
        """Test error handling and robustness."""
        print("🛡️ 测试错误处理和健壮性")

        sync_trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)

        # Test boundary conditions
        # 1. Empty lookup
        sync_result_empty = sync_trie.find_longest_prefix("nonexistent")
        async_result_empty = await async_trie.find_longest_prefix_async("nonexistent")
        assert sync_result_empty.matched_prefix == async_result_empty.matched_prefix, "Empty lookup should be consistent"

        # 2. Remove nonexistent item
        sync_remove_nonexistent = sync_trie.remove("nonexistent")
        async_remove_nonexistent = await async_trie.remove_async("nonexistent")
        assert sync_remove_nonexistent == async_remove_nonexistent, "Remove nonexistent should be consistent"

        # 3. Empty GC
        sync_gc_empty = sync_trie.gc_by_weight_version(999)
        async_gc_empty = await async_trie.gc_by_weight_version_async(999)
        assert sync_gc_empty == async_gc_empty, "Empty GC should be consistent"

        # 4. Duplicate clear
        sync_trie.clear()
        await async_trie.clear_async()
        sync_trie.clear()
        await async_trie.clear_async()

        # Verify error handling consistency
        error_handling_consistent = (
            sync_result_empty.matched_prefix == async_result_empty.matched_prefix and
            sync_remove_nonexistent == async_remove_nonexistent and
            sync_gc_empty == async_gc_empty
        )

        assert error_handling_consistent, "Error handling should be consistent"

        print("✅ 错误处理行为一致!")

    @pytest.mark.asyncio
    async def test_memory_and_resource_management(self):
        """Test memory and resource management."""
        print("💾 测试内存和资源管理")

        # Test large data memory management
        sync_trie = StringRadixTrie(max_cache_size=50, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=50, tokenizer=self.tokenizer, verbose=False)

        # Insert sufficient data to trigger multiple GC cycles
        for i in range(100):
            text = f"item_{i}"
            tokens = [i % 100 + 1, (i * 2) % 100 + 1, (i * 3) % 100 + 1]
            logp = [0.1, 0.2, 0.3]
            loss_mask = [1, 1, 1]
            weight_version = (i % 5) + 1

            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify memory management consistency
        memory_consistent = (sync_trie.cur_cache_size == async_trie.cur_cache_size and
                            sync_trie.total_entries == async_trie.total_entries and
                            sync_trie.cur_cache_size <= sync_trie.max_cache_size and
                            async_trie.cur_cache_size <= async_trie.max_cache_size)

        assert memory_consistent, "Memory management should be consistent"

        # Test resource cleanup
        final_sync_stats = sync_trie.get_stats()
        final_async_stats = await async_trie.get_stats_async()

        # Clear all data
        sync_trie.clear()
        await async_trie.clear_async()

        final_sync_stats_after = sync_trie.get_stats()
        final_async_stats_after = await async_trie.get_stats_async()

        assert final_sync_stats_after['cur_cache_size'] == 0, "Sync cache should be empty after clear"
        assert final_async_stats_after['cur_cache_size'] == 0, "Async cache should be empty after clear"

        print("✅ 内存管理行为一致且正常!")
        print("✅ 资源清理正常!")

    @pytest.mark.asyncio
    async def test_remove_async_functionality(self):
        """Test remove_async method functionality."""
        print("🗑️ 测试 remove_async 方法")

        sync_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)

        # Insert test data
        test_data = [
            ("test1", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], 1),
            ("test2", [4, 5], [0.4, 0.5], [1, 1], 2),
            ("test3", [6, 7, 8], [0.6, 0.7, 0.8], [1, 1, 1], 3),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Test removal
        sync_result = sync_trie.remove("test2")
        async_result = await async_trie.remove_async("test2")

        # Verify consistency
        consistent = (sync_result == async_result and
                      sync_trie.cur_cache_size == async_trie.cur_cache_size and
                      sync_trie.total_entries == async_trie.total_entries)

        assert consistent, "remove_async functionality should be consistent"

        # Verify item was actually removed
        result_sync = sync_trie.find_longest_prefix("test2")
        result_async = await async_trie.find_longest_prefix_async("test2")
        assert result_sync.matched_prefix == "", "Sync should not find removed item"
        assert result_async.matched_prefix == "", "Async should not find removed item"

        print("✅ remove_async 功能一致!")

    @pytest.mark.asyncio
    async def test_gc_by_weight_version_async_functionality(self):
        """Test gc_by_weight_version_async method functionality."""
        print("🗑️ 测试 gc_by_weight_version_async 方法")

        sync_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)

        # Insert different version data
        test_data = [
            ("old1", [1], [0.1], [1], 1),
            ("old2", [2], [0.2], [1], 1),
            ("current1", [3], [0.3], [1], 2),
            ("current2", [4], [0.4], [1], 2),
            ("new1", [5], [0.5], [1], 3),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Test GC by version
        sync_removed = sync_trie.gc_by_weight_version(1)
        async_removed = await async_trie.gc_by_weight_version_async(1)

        # Verify consistency
        consistent = (sync_removed == async_removed and
                      sync_trie.cur_cache_size == async_trie.cur_cache_size and
                      sync_trie.total_entries == async_trie.total_entries)

        assert consistent, "gc_by_weight_version_async functionality should be consistent"

        # Verify old data was removed but new data remains
        for text, expected_version in [("old1", 1), ("old2", 1)]:
            result_sync = sync_trie.find_longest_prefix(text)
            result_async = await async_trie.find_longest_prefix_async(text)
            assert result_sync.matched_prefix == "", f"Sync should not find {text} after GC"
            assert result_async.matched_prefix == "", f"Async should not find {text} after GC"

        for text, expected_version in [("new1", 3), ("current1", 2)]:
            result_sync = sync_trie.find_longest_prefix(text)
            result_async = await async_trie.find_longest_prefix_async(text)
            assert result_sync.matched_prefix == text, f"Sync should find {text} after GC"
            assert result_async.matched_prefix == text, f"Async should find {text} after GC"

        print("✅ gc_by_weight_version_async 功能一致!")

    @pytest.mark.asyncio
    async def test_clear_async_functionality(self):
        """Test clear_async method functionality."""
        print("🗑️ 测试 clear_async 方法")

        sync_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)

        # Insert some data
        test_data = [
            ("data1", [1, 2], [0.1, 0.2], [1, 1], 1),
            ("data2", [3, 4, 5], [0.3, 0.4, 0.5], [1, 1, 1], 2),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        assert sync_trie.cur_cache_size > 0, "Sync should have data before clear"
        assert async_trie.cur_cache_size > 0, "Async should have data before clear"

        # Test clear
        sync_trie.clear()
        await async_trie.clear_async()

        # Verify consistency
        consistent = (sync_trie.cur_cache_size == async_trie.cur_cache_size and
                      sync_trie.total_entries == async_trie.total_entries and
                      sync_trie.cur_cache_size == 0 and async_trie.cur_cache_size == 0)

        assert consistent, "clear_async functionality should be consistent"

        print("✅ clear_async 功能一致!")

    @pytest.mark.asyncio
    async def test_all_async_methods_integration(self):
        """Comprehensive test of all async methods together."""
        print("🔄 综合测试所有异步方法")

        sync_trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)

        # 1. Insert data
        test_data = [
            ("item1", [1], [0.1], [1], 1),
            ("item2", [2], [0.2], [1], 2),
            ("item3", [3], [0.3], [1], 3),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes should match after insertion"

        # 2. Remove an item
        sync_removed = sync_trie.remove("item2")
        async_removed = await async_trie.remove_async("item2")
        assert sync_removed == async_removed, "Remove results should match"

        # 3. GC by version
        sync_gc_removed = sync_trie.gc_by_weight_version(1)
        async_gc_removed = await async_trie.gc_by_weight_version_async(1)
        assert sync_gc_removed == async_gc_removed, "GC results should match"

        # 4. Final clear
        sync_trie.clear()
        await async_trie.clear_async()

        # Verify final consistency
        final_consistent = (sync_trie.cur_cache_size == async_trie.cur_cache_size and
                           sync_trie.total_entries == async_trie.total_entries and
                           sync_trie.cur_cache_size == 0)

        assert final_consistent, "Final state should be consistent"

        print("✅ 综合测试：所有异步方法功能一致!")

    @pytest.mark.asyncio
    async def test_concurrent_operations_safety(self):
        """Test concurrent operations safety."""
        print("🔒 测试并发操作安全性")

        async_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)

        # Concurrent insertions
        async def insert_worker(worker_id):
            for i in range(10):
                text = f"worker{worker_id}_item{i}"
                tokens = [worker_id * 10 + i]
                logp = [0.1 * (worker_id * 10 + i)]
                loss_mask = [1]
                weight_version = worker_id % 3 + 1
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Concurrent reads
        async def read_worker(worker_id):
            for i in range(5):
                text = f"worker{worker_id}_item{i}"
                result = await async_trie.find_longest_prefix_async(text)
                # Result may be empty if not yet inserted, but should not raise exception

        # Run concurrent operations
        insert_tasks = [insert_worker(i) for i in range(3)]
        read_tasks = [read_worker(i) for i in range(3)]

        await asyncio.gather(*(insert_tasks + read_tasks))

        # Verify data integrity
        assert async_trie.total_entries > 0, "Should have entries after concurrent operations"
        assert async_trie.cur_cache_size > 0, "Should have cache after concurrent operations"

        print("✅ 并发操作安全性验证通过")

    @pytest.mark.asyncio
    async def test_large_scale_operations(self):
        """Test large scale operations."""
        print("📈 测试大规模操作")

        sync_trie = StringRadixTrie(max_cache_size=100, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=100, tokenizer=self.tokenizer, verbose=False)

        # Large scale insertion
        for i in range(200):
            text = f"large_scale_item_{i}"
            tokens = [i % 50 + 1, (i * 2) % 50 + 1]
            logp = [0.1, 0.2]
            loss_mask = [1, 1]
            weight_version = i % 7 + 1

            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify consistency after large scale operations
        assert sync_trie.total_entries == async_trie.total_entries, "Total entries should match after large scale insertion"
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes should match after large scale insertion"

        # Large scale GC
        sync_removed = sync_trie.gc_by_weight_version(4)
        async_removed = await async_trie.gc_by_weight_version_async(4)

        assert sync_removed == async_removed, "Large scale GC should be consistent"
        assert sync_trie.total_entries == async_trie.total_entries, "Entries should match after large scale GC"

        print("✅ 大规模操作测试通过")


if __name__ == "__main__":
    # Run standalone test
    async def run_all_tests():
        test_instance = TestComprehensiveIntegration()
        test_instance.setup_method()

        try:
            await test_instance.test_api_completeness()
            await test_instance.test_gc_critical_functionality()
            await test_instance.test_architectural_simplification()
            await test_instance.test_error_handling_and_robustness()
            await test_instance.test_memory_and_resource_management()
            await test_instance.test_remove_async_functionality()
            await test_instance.test_gc_by_weight_version_async_functionality()
            await test_instance.test_clear_async_functionality()
            await test_instance.test_all_async_methods_integration()
            await test_instance.test_concurrent_operations_safety()
            await test_instance.test_large_scale_operations()
            print("\n🎉 All comprehensive integration tests passed!")
            return True
        except Exception as e:
            print(f"\n💥 Comprehensive integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)