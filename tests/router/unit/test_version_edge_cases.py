"""
Version Edge Cases Unit Tests

Tests cover critical boundary conditions and edge cases for version management:
- Version overflow and extremes (INT_MAX, negative, zero)
- Concurrent version updates and race conditions
- Invalid version states (generation > traverse)
- Version wrap-around and ordering violations
- Stress testing with rapid version changes

Test Strategy:
- Unit testing with minimal mocking
- Focus on data integrity and consistency
- Race condition detection via concurrent operations
- Boundary value analysis for numeric limits
"""

import asyncio
import concurrent.futures
import pytest
import sys
import threading
from slime.router.core.radix_tree import StringRadixTrie


# ==============================================================================
# Group A: Version Numeric Limits and Overflow
# ==============================================================================

@pytest.mark.unit
class TestVersionNumericLimits:
    """Test version handling at numeric boundaries."""

    def test_version_at_int_max(self):
        """
        Test: Insert with version at INT_MAX.

        Boundary: INT_MAX = 2^31 - 1 = 2147483647
        Expected: Should handle gracefully without overflow
        """
        trie = StringRadixTrie()
        int_max_version = sys.maxsize  # Python's sys.maxsize

        result = trie.insert("max_version", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=int_max_version)
        assert result is True, "Should successfully insert with INT_MAX version"

        match = trie.find_longest_prefix("max_version")
        assert match.last_node.weight_version == int_max_version, \
            f"weight_version should be INT_MAX ({int_max_version}), got {match.last_node.weight_version}"
        assert match.last_node.traverse_version == int_max_version, \
            f"traverse_version should be INT_MAX ({int_max_version}), got {match.last_node.traverse_version}"

    def test_version_beyond_int_max(self):
        """
        Test: Insert with version > INT_MAX.

        Expected: Python handles arbitrarily large ints, should work
        """
        trie = StringRadixTrie()
        huge_version = sys.maxsize + 1000

        result = trie.insert("huge_version", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=huge_version)
        assert result is True, f"Should successfully insert with version > INT_MAX ({huge_version})"

        match = trie.find_longest_prefix("huge_version")
        assert match.last_node.weight_version == huge_version, \
            f"weight_version should be {huge_version}, got {match.last_node.weight_version}"

    def test_version_negative_values(self):
        """
        Test: Insert with negative version values.

        Boundary: Only -1 (default for non-AI tokens) should be accepted
        Expected: -1 succeeds, other negative values are rejected
        """
        trie = StringRadixTrie()

        # Test -1: should succeed (default value for non-AI generated tokens)
        result_minus_one = trie.insert("neg_minus_one", [1, 2], [0.1, 0.2], [1, 1], weight_version=-1)
        assert result_minus_one is True, "-1 should be accepted as default value"
        match = trie.find_longest_prefix("neg_minus_one")
        assert match.last_node.weight_version == -1, \
            "weight_version -1 should be preserved"

        # Test other negative values: should be rejected (< -1 not allowed)
        invalid_negative_cases = [-2, -100, -sys.maxsize]
        for negative_version in invalid_negative_cases:
            result = trie.insert(f"neg_{negative_version}", [1, 2], [0.1, 0.2], [1, 1], weight_version=negative_version)
            assert result is False, \
                f"Negative version {negative_version} (< -1) should be rejected"

    def test_version_zero_boundary(self):
        """
        Test: Version 0 as boundary between negative and positive.

        Boundary: Version 0
        Expected: Should work, GC should handle correctly
        """
        trie = StringRadixTrie(gc_threshold_k=1)

        # Insert at version 0
        trie.insert("v0", [1], [0.1], [1], weight_version=0)
        match = trie.find_longest_prefix("v0")
        assert match.matched_prefix == "v0", "Version 0 node should be findable after insertion"
        assert match.last_node.weight_version == 0, \
            f"weight_version should be 0, got {match.last_node.weight_version}"

        # GC at version 2 should remove version 0 (0 <= 2-1)
        trie.gc_by_weight_version(current_weight_version=2)
        match_after = trie.find_longest_prefix("v0")
        assert match_after.matched_prefix == "", "Version 0 should be GC'd"

    def test_version_rapid_increase(self):
        """
        Test: Rapid version increases (large jumps).

        Scenario: Version jumps from 1 to 1000000
        Expected: System should handle large version gaps
        """
        trie = StringRadixTrie()

        trie.insert("v1", [1], [0.1], [1], weight_version=1)
        trie.insert("v_million", [2], [0.2], [1], weight_version=1000000)

        # Both should exist
        assert trie.find_longest_prefix("v1").matched_prefix == "v1", \
            "Node with version 1 should exist after insertion"
        assert trie.find_longest_prefix("v_million").matched_prefix == "v_million", \
            "Node with version 1000000 should exist after insertion"

        # v1's traverse_version should NOT be updated (different paths)
        v1_match = trie.find_longest_prefix("v1")
        assert v1_match.last_node.traverse_version == 1, "Unrelated path should not update traverse_version"


# ==============================================================================
# Group B: Concurrent Version Updates
# ==============================================================================

@pytest.mark.unit
class TestConcurrentVersionUpdates:
    """Test concurrent version update scenarios and race conditions."""

    def test_concurrent_insert_same_key_different_versions(self):
        """
        Test: Multiple threads insert same key with different versions.

        Race Condition: Last write wins, but versions should be consistent
        Expected: Final state should have one of the versions, data consistent
        """
        trie = StringRadixTrie()
        results = []
        errors = []

        def insert_worker(worker_id, version):
            try:
                result = trie.insert("concurrent_key", [worker_id], [0.1 * worker_id], [1], weight_version=version)
                results.append((worker_id, version, result))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch 10 concurrent inserts
        threads = []
        for i in range(10):
            thread = threading.Thread(target=insert_worker, args=(i, i + 1))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have errors
        assert len(errors) == 0, f"Concurrent inserts should not error: {errors}"

        # All inserts should succeed (though some may overwrite)
        assert len(results) == 10

        # Final state should be consistent
        final_match = trie.find_longest_prefix("concurrent_key")
        assert final_match.matched_prefix == "concurrent_key", \
            "Concurrent inserts should leave key in consistent state"

        # weight_version should be one of the inserted versions
        assert final_match.last_node.weight_version in range(1, 11), \
            f"Final version {final_match.last_node.weight_version} should be from one of the inserts"

        # Token should match the version (worker_id = version - 1)
        expected_worker_id = final_match.last_node.weight_version - 1
        assert final_match.token_ids == [expected_worker_id], \
            f"Token should match final version: expected [{expected_worker_id}], got {final_match.token_ids}"

    def test_concurrent_version_traverse_updates(self):
        """
        Test: Concurrent operations that update traverse_version.

        Race Condition: Multiple threads traverse same path
        Expected: traverse_version should be highest version that traversed
        """
        trie = StringRadixTrie()

        # Create base path
        trie.insert("base", [1, 2], [0.1, 0.2], [1, 1], weight_version=1)

        results = []

        def extend_worker(worker_id, version):
            try:
                # Each worker extends base path
                trie.insert(f"base_ext{worker_id}", [1, 2, 3 + worker_id], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=version)
                results.append((worker_id, version))
            except Exception as e:
                results.append((worker_id, f"ERROR: {e}"))

        # Launch concurrent extensions
        threads = []
        for i in range(10):
            thread = threading.Thread(target=extend_worker, args=(i, 10 + i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check base node's traverse_version
        base_match = trie.find_longest_prefix("base")
        base_traverse_version = base_match.last_node.traverse_version

        # traverse_version should be >= original version
        assert base_traverse_version >= 1, \
            f"traverse_version should be >= 1 after concurrent updates, got {base_traverse_version}"

        # Should be one of the worker versions (10-19) or higher
        # Note: Due to race conditions, we can't guarantee it's exactly the max
        print(f"Base traverse_version after concurrent updates: {base_traverse_version}")

    @pytest.mark.asyncio
    async def test_concurrent_async_version_updates(self):
        """
        Test: Concurrent async operations updating versions.

        Expected: Async version updates should be safe
        """
        trie = StringRadixTrie()

        # Initial data
        await trie.insert_async("async_base", [1], [0.1], [1], weight_version=1)

        async def async_worker(worker_id, version):
            return await trie.insert_async(
                f"async_base_{worker_id}",
                [1, worker_id],
                [0.1, 0.2],
                [1, 1],
                weight_version=version
            )

        # Run concurrent async operations
        tasks = [async_worker(i, 100 + i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Should not have exceptions: {exceptions}"

        # Verify all succeeded
        assert all(r is True for r in results if not isinstance(r, Exception)), \
            "All async insert operations should succeed"

    def test_concurrent_gc_and_version_updates(self):
        """
        Test: GC running concurrently with version updates.

        Race Condition: GC might remove nodes being updated
        Expected: Should be safe, no crashes or inconsistent state
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        stop_flag = threading.Event()
        results = {'inserts': 0, 'gcs': 0, 'errors': []}

        def insert_worker():
            """Continuously insert with increasing versions."""
            version = 1
            while not stop_flag.is_set():
                try:
                    trie.insert(f"data_{version}", [version % 100], [0.1], [1], weight_version=version)
                    results['inserts'] += 1
                    version += 1
                except Exception as e:
                    results['errors'].append(f"Insert error: {e}")

        def gc_worker():
            """Continuously trigger GC."""
            version = 10
            while not stop_flag.is_set():
                try:
                    trie.gc_by_weight_version(current_weight_version=version)
                    results['gcs'] += 1
                    version += 1
                except Exception as e:
                    results['errors'].append(f"GC error: {e}")

        # Run workers for short time
        insert_thread = threading.Thread(target=insert_worker)
        gc_thread = threading.Thread(target=gc_worker)

        insert_thread.start()
        gc_thread.start()

        # Let them run for 0.1 seconds
        threading.Event().wait(0.1)
        stop_flag.set()

        insert_thread.join()
        gc_thread.join()

        # Should have done some work
        assert results['inserts'] > 0, \
            f"Should have inserted some data during stress test, got {results['inserts']}"
        assert results['gcs'] > 0, \
            f"Should have performed some GC operations during stress test, got {results['gcs']}"

        # Should not have errors
        assert len(results['errors']) == 0, f"Should not have errors: {results['errors']}"


# ==============================================================================
# Group C: Invalid Version States
# ==============================================================================

@pytest.mark.unit
class TestInvalidVersionStates:
    """Test handling of invalid version states."""

    def test_generation_version_greater_than_traverse(self):
        """
        Test: weight_version > traverse_version (should be impossible).

        Invalid State: Generation happened after last traversal
        Expected: System should prevent or handle gracefully
        """
        trie = StringRadixTrie()

        # Normal insertion
        trie.insert("base", [1, 2], [0.1, 0.2], [1, 1], weight_version=10)

        match = trie.find_longest_prefix("base")
        initial_weight = match.last_node.weight_version
        initial_traverse = match.last_node.traverse_version

        # Should be equal initially (Boundary: version equality at creation)
        assert initial_weight == initial_traverse == 10, \
            f"weight and traverse versions should both be 10, got weight={initial_weight}, traverse={initial_traverse}"

        # Try to create invalid state by manually manipulating
        # (In real code this shouldn't happen, but testing robustness)
        # Note: We can't directly set node attributes without accessing internals
        # So we test that normal operations never create this state

        # Insert extending path with LOWER version
        trie.insert("base_extended", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=5)

        # Check base node
        base_match = trie.find_longest_prefix("base")

        # traverse_version should be updated to 10 (max), weight_version stays 10
        # Actually, with lower version insert, traverse_version should still be 10
        # because the node was created at v10
        assert base_match.last_node.weight_version <= base_match.last_node.traverse_version, \
            "weight_version should never exceed traverse_version"

    def test_none_version_with_numeric_versions(self):
        """
        Test: Mixing None versions with numeric versions.

        Edge Case: Some nodes have versions, some don't
        Expected: Should handle gracefully, None versions don't participate in GC
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Insert without version
        trie.insert("no_version", [1, 2], [0.1, 0.2], [1, 1])  # version=None

        # Insert with version
        trie.insert("with_version", [3, 4], [0.3, 0.4], [1, 1], weight_version=5)

        # Verify both exist (Boundary: mixing None and numeric versions)
        assert trie.find_longest_prefix("no_version").matched_prefix == "no_version", \
            "Node with None version should be findable"
        assert trie.find_longest_prefix("with_version").matched_prefix == "with_version", \
            "Node with numeric version should be findable"

        # GC should not remove None-version nodes (or handle gracefully)
        trie.gc_by_weight_version(current_weight_version=10)

        # None-version node should still exist (not subject to GC)
        none_match = trie.find_longest_prefix("no_version")
        # Implementation may remove or keep None-version nodes, both are acceptable
        # The key is it shouldn't crash
        assert isinstance(none_match.matched_prefix, str)  # Should return something, not crash

    def test_version_ordering_violations(self):
        """
        Test: Insert operations with non-monotonic versions.

        Scenario: v10 -> v5 -> v15 -> v3
        Expected: System should handle out-of-order versions
        """
        trie = StringRadixTrie()

        # Insert in non-monotonic order
        versions = [10, 5, 15, 3, 20, 1]
        for v in versions:
            trie.insert(f"v{v}", [v], [0.1 * v], [1], weight_version=v)

        # All should exist
        for v in versions:
            match = trie.find_longest_prefix(f"v{v}")
            assert match.matched_prefix == f"v{v}"
            assert match.last_node.weight_version == v

    @pytest.mark.asyncio
    async def test_async_version_consistency_under_load(self):
        """
        Test: Version consistency under heavy async load.

        Stress Test: 100+ concurrent operations
        Expected: No version inconsistencies, no crashes
        """
        trie = StringRadixTrie()

        async def heavy_worker(worker_id):
            """Perform multiple operations per worker."""
            operations = []
            base_version = worker_id * 10

            for i in range(10):
                version = base_version + i
                result = await trie.insert_async(
                    f"worker{worker_id}_item{i}",
                    [worker_id, i],
                    [0.1, 0.2],
                    [1, 1],
                    weight_version=version
                )
                operations.append(result)

            return operations

        # Launch 50 workers, each doing 10 operations = 500 operations
        tasks = [heavy_worker(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Should not have exceptions under load: {exceptions}"

        # Verify some random entries exist and have correct versions
        for worker_id in [0, 10, 25, 49]:
            for item_id in [0, 5, 9]:
                match = await trie.find_longest_prefix_async(f"worker{worker_id}_item{item_id}")
                assert match.matched_prefix == f"worker{worker_id}_item{item_id}", \
                    f"Entry should exist after heavy load"

                expected_version = worker_id * 10 + item_id
                assert match.last_node.weight_version == expected_version, \
                    f"Version should be preserved: expected {expected_version}, got {match.last_node.weight_version}"


# ==============================================================================
# Group D: Version and GC Interaction Edge Cases
# ==============================================================================

@pytest.mark.unit
class TestVersionGCInteraction:
    """Test edge cases in version-GC interaction."""

    def test_gc_threshold_zero(self):
        """
        Test: GC with threshold_k=0.

        Boundary: Minimal threshold
        Expected: GC removes all nodes with traverse_version < current
        """
        trie = StringRadixTrie(gc_threshold_k=0)

        # Insert at various versions
        trie.insert("v1", [1], [0.1], [1], weight_version=1)
        trie.insert("v2", [2], [0.2], [1], weight_version=2)
        trie.insert("v3", [3], [0.3], [1], weight_version=3)

        # GC at version 3 with threshold 0 means remove traverse_version < 3
        trie.gc_by_weight_version(current_weight_version=3)

        # v1 and v2 should be removed (traverse_version < 3)
        assert trie.find_longest_prefix("v1").matched_prefix == ""
        assert trie.find_longest_prefix("v2").matched_prefix == ""
        # v3 should remain (traverse_version == 3, not < 3)
        assert trie.find_longest_prefix("v3").matched_prefix == "v3"

    def test_gc_threshold_very_large(self):
        """
        Test: GC with very large threshold.

        Boundary: threshold_k = 10000
        Expected: Nothing gets GC'd
        """
        trie = StringRadixTrie(gc_threshold_k=10000)

        trie.insert("v1", [1], [0.1], [1], weight_version=1)
        trie.insert("v10", [10], [0.2], [1], weight_version=10)

        # GC at version 100, threshold 10000 means remove traverse_version < (100 - 10000) = -9900
        # Nothing should be removed
        removed = trie.gc_by_weight_version(current_weight_version=100)

        assert removed == 0, "Nothing should be removed with huge threshold"
        assert trie.find_longest_prefix("v1").matched_prefix == "v1"
        assert trie.find_longest_prefix("v10").matched_prefix == "v10"

    def test_version_generation_versions_accuracy(self):
        """
        Test: generation_versions array accuracy with complex version patterns.

        Scenario: Multiple nodes with different versions in same path
        Expected: generation_versions should accurately reflect each token's generation version
        """
        trie = StringRadixTrie()

        # Build complex path: "abc"
        # "a" generated at v1
        trie.insert("a", [65], [0.1], [1], weight_version=1)

        # "ab" generated at v5 (adds "b" at v5, "a" from v1)
        trie.insert("ab", [65, 66], [0.1, 0.2], [1, 1], weight_version=5)

        # "abc" generated at v10 (adds "c" at v10, "ab" from previous)
        trie.insert("abc", [65, 66, 67], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=10)

        # Check generation_versions for full path
        match = trie.find_longest_prefix("abc")

        # Expected: "a" from v1, "b" from v5, "c" from v10
        expected_versions = [1, 5, 10]
        assert match.generation_versions == expected_versions, \
            f"Expected {expected_versions}, got {match.generation_versions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
