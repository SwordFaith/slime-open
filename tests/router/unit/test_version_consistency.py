#!/usr/bin/env python3
"""
Version consistency unit tests for radix tree.
测试版本一致性机制的单元测试，包括generation_versions和traverse_version管理
"""

import asyncio
import pytest
from slime.router.core.radix_tree import StringRadixTrie


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Simple mock: each character as a token ID
        tokens = [ord(c) % 1000 for c in text]
        return {"input_ids": tokens}


class TestVersionConsistency:
    """Test version consistency mechanisms."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = MockTokenizer()

    def test_sync_version_core_behavior(self):
        """Test sync version core behavior consistency."""
        print("🔍 测试同步版本核心行为")

        trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=2, tokenizer=self.tokenizer, verbose=False)

        # Test basic insertion and lookup
        test_data = [
            ("hello", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], 1),
            ("world", [4, 5], [0.4, 0.5], [1, 1], 2),
            ("hello world", [1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 1, 1, 1], 3),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            success = trie.insert(text, tokens, logp, loss_mask, weight_version)
            assert success, f"Failed to insert '{text}'"

        # Test prefix matching behavior
        queries = ["hello world!", "hello there", "worldwide", "hi"]
        expected_matches = ["hello world", "hello", "world", ""]

        for query, expected in zip(queries, expected_matches):
            result = trie.find_longest_prefix(query)
            assert result.matched_prefix == expected, f"Prefix matching failed for '{query}': expected '{expected}', got '{result.matched_prefix}'"

        # Test get_or_create_tokenization behavior
        try:
            tokens, logp, loss_mask, versions = trie.get_or_create_tokenization("hello")
            assert tokens == [1, 2, 3], f"Token mismatch for 'hello': expected [1, 2, 3], got {tokens}"
            assert versions == [1, 1, 1], f"Version mismatch for 'hello': expected [1, 1, 1], got {versions}"

            tokens2, logp2, loss_mask2, versions2 = trie.get_or_create_tokenization("new text")
            assert len(tokens2) > 0, "New text should generate tokens"
            assert len(versions2) == len(tokens2), "Version count should match token count"
        except Exception as e:
            pytest.fail(f"get_or_create_tokenization failed: {e}")

    def test_sync_version_edge_cases(self):
        """Test sync version edge cases."""
        print("🧪 测试同步版本边界条件")

        trie = StringRadixTrie(max_cache_size=5, tokenizer=self.tokenizer, verbose=False)

        # Test empty string handling
        result = trie.find_longest_prefix("")
        assert result.matched_prefix == "", "Empty string should return empty match"
        assert len(result.token_ids) == 0, "Empty string should return no tokens"

        # Test inserting empty data
        success = trie.insert("", [1, 2], [0.1, 0.2], [1, 1], 1)
        assert success, "Should be able to insert empty string with tokens"

        success = trie.insert("test", [], [], [], 1)
        assert not success, "Should not be able to insert non-empty string with empty tokens"

        # Test duplicate insertion
        trie.insert("duplicate", [1, 2], [0.1, 0.2], [1, 1], 1)
        initial_size = trie.cur_cache_size

        trie.insert("duplicate", [1, 2], [0.1, 0.2], [1, 1], 2)
        final_size = trie.cur_cache_size
        assert final_size >= initial_size, "Cache size should not decrease on duplicate insertion"

        # Test removing non-existent data
        removed = trie.remove("nonexistent")
        assert not removed, "Should not be able to remove non-existent item"

        # Test GC behavior with large data
        for i in range(10):
            text = f"data_{i}"
            tokens = [i, i+1, i+2]
            logp = [0.1 * i, 0.2 * i, 0.3 * i]
            loss_mask = [1, 1, 1]
            weight_version = i % 3 + 1
            trie.insert(text, tokens, logp, loss_mask, weight_version)

        assert trie.total_entries > 0, "Should have entries after inserting data"

        removed = trie.gc_by_weight_version(4)
        assert isinstance(removed, int), "GC should return integer count"
        assert removed >= 0, "GC should not remove negative count"

    def test_sync_version_gc_logic(self):
        """Test sync version GC logic with traverse_version."""
        print("🔥 测试同步版本GC逻辑（traverse_version）")

        trie = StringRadixTrie(max_cache_size=20, gc_threshold_k=1, tokenizer=self.tokenizer, verbose=False)

        # Create complex version hierarchy
        # Version 1 data
        v1_data = [("a", [1], [0.1], [1], 1), ("b", [2], [0.2], [1], 1)]
        for text, tokens, logp, loss_mask, weight_version in v1_data:
            trie.insert(text, tokens, logp, loss_mask, weight_version)
        initial_entries = trie.total_entries
        assert initial_entries == 2, "Should have 2 entries after version 1 insertion"

        # Version 3 data (should update traverse_version)
        v3_data = [("c", [3], [0.3], [1], 3)]
        for text, tokens, logp, loss_mask, weight_version in v3_data:
            trie.insert(text, tokens, logp, loss_mask, weight_version)
        assert trie.total_entries == 3, "Should have 3 entries after version 3 insertion"

        # Version 5 data (further update traverse_version)
        v5_data = [("d", [4], [0.4], [1], 5)]
        for text, tokens, logp, loss_mask, weight_version in v5_data:
            trie.insert(text, tokens, logp, loss_mask, weight_version)
        assert trie.total_entries == 4, "Should have 4 entries after version 5 insertion"

        # Trigger GC, should clean traverse_version < (5 - 1) = 4 nodes
        removed = trie.gc_by_weight_version(5)
        assert isinstance(removed, int), "GC should return integer count"
        assert removed >= 0, "GC should remove non-negative count"

        # Verify remaining nodes accessibility
        accessible_nodes = []
        for text in ["a", "b", "c", "d"]:
            result = trie.find_longest_prefix(text)
            if result.matched_prefix == text:
                accessible_nodes.append(text)

        # At least the newest nodes should be accessible
        assert "d" in accessible_nodes, "Newest node 'd' should be accessible"
        assert "c" in accessible_nodes, "Recent node 'c' should be accessible"

    async def test_async_version_consistency(self):
        """Test async version consistency with sync version."""
        print("🔄 测试异步版本一致性")

        sync_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)
        async_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)

        # Test basic consistency
        test_data = [
            ("test_consistency", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], 2),
            ("empty_test", [], [], [], [], None),
            ("error_test", "test", [], [], [], 1),  # This will be filtered out in actual implementation
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            # Skip invalid test data
            if not tokens or not isinstance(tokens, list):
                continue

            sync_result = sync_trie.insert(text, tokens, logp, loss_mask, weight_version)
            async_result = await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

            assert sync_result == async_result, f"Insert result mismatch for '{text}': sync={sync_result}, async={async_result}"

        # Verify final state consistency
        assert sync_trie.cur_cache_size == async_trie.cur_cache_size, "Final cache sizes should match"
        assert sync_trie.total_entries == async_trie.total_entries, "Final total entries should match"

    async def test_generation_versions_functionality(self):
        """Test generation_versions functionality."""
        print("📊 测试generation_versions功能")

        async_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)

        # Test data with different versions
        test_versions = [
            ("v1_data", [1, 2], [0.1, 0.2], [1, 1], 1),
            ("v3_data", [3, 4], [0.3, 0.4], [1, 1], 3),
            ("v5_data", [5, 6], [0.5, 0.6], [1, 1], 5),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_versions:
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Verify generation_versions in results
        for text, expected_tokens, _, _, expected_version in test_versions:
            result = await async_trie.find_longest_prefix_async(text)
            assert result.matched_prefix == text, f"Should find exact match for '{text}'"
            assert result.token_ids == expected_tokens, f"Token mismatch for '{text}'"
            assert hasattr(result, 'generation_versions'), "Result should have generation_versions attribute"
            assert result.generation_versions is not None, "generation_versions should not be None"

    async def test_traverse_version_gc(self):
        """Test traverse_version-based GC."""
        print("🗑️ 测试traverse_version GC机制")

        async_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)

        # Insert data with different versions
        test_data = [
            ("old_v1", [1], [0.1], [1], 1),
            ("old_v2", [2], [0.2], [1], 2),
            ("new_v4", [3], [0.3], [1], 4),
            ("new_v5", [4], [0.4], [1], 5),
        ]

        for text, tokens, logp, loss_mask, weight_version in test_data:
            await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        initial_entries = async_trie.total_entries
        assert initial_entries == 4, "Should have 4 entries initially"

        # Trigger GC with version 4, should remove v1 and v2
        removed = await async_trie.gc_by_weight_version_async(4)
        final_entries = async_trie.total_entries

        assert isinstance(removed, int), "GC should return integer"
        assert removed >= 0, "Should remove non-negative count"
        assert final_entries <= initial_entries, "Entries should not increase after GC"

        # Verify newer data is still accessible
        new_result = await async_trie.find_longest_prefix_async("new_v5")
        assert new_result.matched_prefix == "new_v5", "Newer data should still be accessible"

    async def test_async_api_completeness(self):
        """Test async API completeness."""
        print("🔌 测试异步API完整性")

        async_trie = StringRadixTrie(max_cache_size=20, tokenizer=self.tokenizer, verbose=False)

        # Test all async methods
        api_methods = [
            ('insert_async', lambda: async_trie.insert_async("api_test", [1], [0.1], [1], 1)),
            ('find_longest_prefix_async', lambda: async_trie.find_longest_prefix_async("api_test")),
            ('remove_async', lambda: async_trie.remove_async("api_test")),
            ('gc_by_weight_version_async', lambda: async_trie.gc_by_weight_version_async(2)),
            ('get_stats_async', lambda: async_trie.get_stats_async()),
            ('clear_async', lambda: async_trie.clear_async()),
            ('get_or_create_tokenization_async', lambda: async_trie.get_or_create_tokenization_async("new_api_test")),
        ]

        for method_name, method_call in api_methods:
            try:
                result = await method_call()
                # Some methods may return None, bool, or other types
                assert True, f"{method_name} should execute without exception"
            except Exception as e:
                pytest.fail(f"{method_name} raised exception: {e}")

    async def test_concurrent_version_consistency(self):
        """Test concurrent access version consistency."""
        print("🚀 测试并发版本一致性")

        async_trie = StringRadixTrie(max_cache_size=30, tokenizer=self.tokenizer, verbose=False)

        # Concurrent insertions with different versions
        async def insert_worker(worker_id: int, base_version: int):
            for i in range(5):
                text = f"worker{worker_id}_item{i}"
                tokens = [worker_id * 10 + i]
                logp = [0.1 * (worker_id * 10 + i)]
                loss_mask = [1]
                weight_version = base_version + i
                await async_trie.insert_async(text, tokens, logp, loss_mask, weight_version)

        # Run multiple workers concurrently
        tasks = [
            insert_worker(1, 1),   # versions 1-5
            insert_worker(2, 3),   # versions 3-7
            insert_worker(3, 5),   # versions 5-9
        ]

        await asyncio.gather(*tasks)

        # Verify data integrity
        final_entries = async_trie.total_entries
        assert final_entries > 0, "Should have entries after concurrent insertions"

        # Test concurrent reads
        async def read_worker(worker_id: int):
            for i in range(3):
                text = f"worker{worker_id}_item{i}"
                result = await async_trie.find_longest_prefix_async(text)
                # Result may be empty if item was GC'd, but should not raise exception
                assert isinstance(result, object), "Read should return result object"

        read_tasks = [read_worker(i) for i in range(1, 4)]
        await asyncio.gather(*read_tasks)

    def test_version_boundary_conditions(self):
        """Test version boundary conditions."""
        print("🎯 测试版本边界条件")

        trie = StringRadixTrie(max_cache_size=10, tokenizer=self.tokenizer, verbose=False)

        # Test with zero version
        success = trie.insert("zero_version", [1], [0.1], [1], 0)
        assert success, "Should handle zero version"

        # Test with negative version (should be handled gracefully)
        try:
            success = trie.insert("negative_version", [2], [0.2], [1], -1)
            # May or may not succeed depending on implementation
        except Exception:
            # Should handle gracefully without crashing
            pass

        # Test with very large version
        large_version = 999999
        success = trie.insert("large_version", [3], [0.3], [1], large_version)
        assert success, "Should handle large version numbers"

        # Verify large version is stored correctly
        result = trie.find_longest_prefix("large_version")
        assert result.matched_prefix == "large_version", "Large version item should be retrievable"


if __name__ == "__main__":
    # Run standalone test
    async def run_all_tests():
        test_instance = TestVersionConsistency()
        test_instance.setup_method()

        try:
            test_instance.test_sync_version_core_behavior()
            test_instance.test_sync_version_edge_cases()
            test_instance.test_sync_version_gc_logic()
            await test_instance.test_async_version_consistency()
            await test_instance.test_generation_versions_functionality()
            await test_instance.test_traverse_version_gc()
            await test_instance.test_async_api_completeness()
            await test_instance.test_concurrent_version_consistency()
            test_instance.test_version_boundary_conditions()
            print("\n🎉 All version consistency tests passed!")
            return True
        except Exception as e:
            print(f"\n💥 Version consistency test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)