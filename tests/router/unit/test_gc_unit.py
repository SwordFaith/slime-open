#!/usr/bin/env python3
"""
GC unit tests for radix tree.
测试垃圾回收机制的单元测试
"""

import asyncio
import gc
import pytest
from slime.router.core.radix_tree import StringRadixTrie


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Simple mock: each character as a token ID
        tokens = [ord(c) % 1000 for c in text]
        return {"input_ids": tokens}


class TestGarbageCollection:
    """Test garbage collection functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = MockTokenizer()

    async def test_aggressive_gc_trigger(self):
        """Test aggressive GC triggering with small cache size."""
        print("🔥 激进GC测试 - 确认GC实际触发")

        # Create very small cache to trigger GC quickly
        sync_trie = StringRadixTrie(max_cache_size=3, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=3, tokenizer=self.tokenizer, verbose=False)

        # Test data - each insertion will exceed cache limit
        test_data = [
            ("a", [1], [0.1], [1], 1),
            ("b", [2], [0.2], [1], 2),
            ("c", [3], [0.3], [1], 3),
            ("d", [4], [0.4], [1], 4),
            ("e", [5], [0.5], [1], 5),
        ]

        # Test sync version
        sync_cache_sizes = []
        for text, tokens, logp, loss_mask, weight_version in test_data:
            initial_size = sync_trie.cur_cache_size
            result = sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            final_size = sync_trie.cur_cache_size
            sync_cache_sizes.append(final_size)
            assert final_size <= sync_trie.max_cache_size, f"Cache size exceeded limit: {final_size}"

        # Test async version
        async_cache_sizes = []
        for text, tokens, logp, loss_mask, weight_version in test_data:
            initial_size = async_trie.cur_cache_size
            result = await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
            final_size = async_trie.cur_cache_size
            async_cache_sizes.append(final_size)
            assert final_size <= async_trie.max_cache_size, f"Cache size exceeded limit: {final_size}"

        # Compare results
        assert sync_cache_sizes == async_cache_sizes, f"GC behavior inconsistent: sync={sync_cache_sizes}, async={async_cache_sizes}"
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, f"Final cache size inconsistent: {sync_trie.cur_cache_size} vs {async_trie.cur_cache_size}"
        assert sync_trie.total_entries == async_trie.total_entries, f"Total entries inconsistent: {sync_trie.total_entries} vs {async_trie.total_entries}"

    async def test_real_gc_trigger_with_weight_version(self):
        """Test real GC triggering using weight version differences."""
        print("🔥 真实GC触发测试")

        # Create small cache and small GC threshold to ensure GC triggers
        sync_trie = StringRadixTrie(max_cache_size=3, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=3, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # First batch - weight_version=1
        batch1 = [("a", [1], [0.1], [1], 1), ("b", [2], [0.2], [1], 1), ("c", [3], [0.3], [1], 1)]

        for text, tokens, logp, loss_mask, weight_version in batch1:
            sync_result = sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            async_result = await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
            # Track intermediate states for debugging
            assert sync_trie.cur_cache_size == async_trie.cur_cache_size, f"Cache size mismatch after inserting '{text}'"

        # Second batch - weight_version=3, should trigger GC removing weight_version=1 nodes
        batch2 = [("d", [4], [0.4], [1], 3), ("e", [5], [0.5], [1], 3)]

        for text, tokens, logp, loss_mask, weight_version in batch2:
            sync_result = sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            async_result = await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)
            # Track intermediate states
            assert sync_trie.cur_cache_size == async_trie.cur_cache_size, f"Cache size mismatch after inserting '{text}'"
            assert sync_trie.total_entries == async_trie.total_entries, f"Total entries mismatch after inserting '{text}'"

        # Verify consistency
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Final cache size inconsistent"
        assert sync_trie.total_entries == async_trie.total_entries, "Final total entries inconsistent"

        # Verify GC actually happened
        assert sync_trie.cur_cache_size <= sync_trie.max_cache_size, "GC was not triggered effectively"
        assert async_trie.cur_cache_size <= async_trie.max_cache_size, "GC was not triggered effectively"

    async def test_manual_gc_calls(self):
        """Test manual GC calls."""
        print("🔧 手动GC测试")

        sync_trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # Insert some test data
        test_data = [
            ("old1", [1], [0.1], [1], 1),
            ("old2", [2], [0.2], [1], 1),
            ("old3", [3], [0.3], [1], 1),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify initial state
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Initial cache sizes don't match"
        assert sync_trie.total_entries == async_trie.total_entries, "Initial total entries don't match"

        # Manual GC trigger
        sync_removed = sync_trie.gc_by_weight_version(3)
        async_removed = async_trie._gc_by_weight_version_internal(3)

        # Verify GC results
        assert sync_removed == async_removed, f"GC removal count mismatch: sync={sync_removed}, async={async_removed}"
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes don't match after GC"
        assert sync_trie.total_entries == async_trie.total_entries, "Total entries don't match after GC"

    async def test_gc_with_different_thresholds(self):
        """Test GC behavior with different threshold settings."""
        print("🎚️ GC阈值测试")

        # Test with different GC thresholds
        thresholds = [0.5, 1, 2]

        for threshold in thresholds:
            sync_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=threshold, tokenizer=self.tokenizer, verbose=False)
            async_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=threshold, tokenizer=self.tokenizer, verbose=False)

            # Insert data that should trigger GC
            for i in range(10):
                text = f"item{i}"
                tokens = [i]
                logp = [0.1 * i]
                loss_mask = [1]
                weight_version = i % 3 + 1  # Rotate weight versions

                sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

            # Verify consistency
            assert sync_trie.cur_cache_size == async_trie.cur_cache_size, f"Cache size mismatch for threshold {threshold}"
            assert sync_trie.total_entries == async_trie.total_entries, f"Total entries mismatch for threshold {threshold}"

            # Verify GC kept cache size in check
            assert sync_trie.cur_cache_size <= sync_trie.max_cache_size * (1 + threshold), f"GC didn't work effectively for threshold {threshold}"
            assert async_trie.cur_cache_size <= async_trie.max_cache_size * (1 + threshold), f"GC didn't work effectively for threshold {threshold}"

    async def test_gc_memory_cleanup(self):
        """Test that GC properly cleans up memory."""
        print("🧹 GC内存清理测试")

        sync_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # Insert many items to trigger GC multiple times
        for i in range(20):
            text = f"memory_test_item_{i}_with_long_name_to_consume_memory"
            tokens = list(range(10))  # Longer token lists
            logp = [0.1] * 10
            loss_mask = [1] * 10
            weight_version = i // 5 + 1

            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Force Python GC to ensure any weak references are cleaned
        gc.collect()

        # Verify final state consistency
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Final cache sizes don't match after memory test"
        assert sync_trie.total_entries == async_trie.total_entries, "Final total entries don't match after memory test"

        # Verify cache is properly maintained
        assert sync_trie.cur_cache_size <= sync_trie.max_cache_size * 2, "Cache grew too large despite GC"
        assert async_trie.cur_cache_size <= async_trie.max_cache_size * 2, "Cache grew too large despite GC"

    async def test_gc_preserves_recent_data(self):
        """Test that GC preserves recent data while cleaning old data."""
        print("🛡️ GC保留近期数据测试")

        sync_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=5, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # Insert old data with weight_version=1
        old_data = [("old1", [1], [0.1], [1], 1), ("old2", [2], [0.2], [1], 1)]

        for text, tokens, logp, loss_mask, weight_version in old_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Insert new data with weight_version=3 (should trigger GC of old data)
        new_data = [("new1", [3], [0.3], [1], 3), ("new2", [4], [0.4], [1], 3)]

        for text, tokens, logp, loss_mask, weight_version in new_data:
            sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify new data is preserved
        for text, tokens, logp, loss_mask, weight_version in new_data:
            sync_result = sync_trie.find_longest_prefix(text)
            async_result = await async_trie.find_longest_prefix_async(text)

            assert sync_result.matched_prefix == text, f"New data '{text}' not found in sync trie"
            assert async_result.matched_prefix == text, f"New data '{text}' not found in async trie"
            assert sync_result.token_ids == tokens, f"Token mismatch for '{text}' in sync trie"
            assert async_result.token_ids == tokens, f"Token mismatch for '{text}' in async trie"

        # Verify consistency between tries
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Cache sizes don't match after selective GC"
        assert sync_trie.total_entries == async_trie.total_entries, "Total entries don't match after selective GC"


if __name__ == "__main__":
    # Run standalone test
    async def run_all_tests():
        test_instance = TestGarbageCollection()
        test_instance.setup_method()

        try:
            await test_instance.test_aggressive_gc_trigger()
            await test_instance.test_real_gc_trigger_with_weight_version()
            await test_instance.test_manual_gc_calls()
            await test_instance.test_gc_with_different_thresholds()
            await test_instance.test_gc_memory_cleanup()
            await test_instance.test_gc_preserves_recent_data()
            print("\n🎉 All GC tests passed!")
            return True
        except Exception as e:
            print(f"\n💥 GC test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)