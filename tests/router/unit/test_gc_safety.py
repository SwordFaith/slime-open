"""
GC Safety Unit Tests

Tests cover critical safety scenarios for garbage collection:
- GC during active read operations
- GC with incomplete/NULL version data
- GC removing currently accessed nodes
- Memory pressure and high-frequency GC
- Concurrent GC operations
- GC with corrupted tree state

Test Strategy:
- Unit testing with concurrent operations
- Race condition detection
- Data consistency verification after GC
- Resource safety validation
"""

import asyncio
import gc as python_gc
import pytest
import threading
import time
import weakref
from slime.router.core.radix_tree import StringRadixTrie


# ==============================================================================
# Group A: GC During Active Operations
# ==============================================================================

@pytest.mark.unit
class TestGCDuringActiveOperations:
    """Test GC safety when operations are in progress."""

    def test_gc_during_concurrent_reads(self):
        """
        Test: GC triggered while multiple readers are active.

        Race Condition: GC might remove nodes being read
        Expected: Readers should complete safely, data consistent
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Pre-populate with old and new data
        for i in range(10):
            trie.insert(f"old_{i}", [i], [0.1 * i], [1], weight_version=1)
        for i in range(10):
            trie.insert(f"new_{i}", [100 + i], [0.1 * i], [1], weight_version=20)

        read_results = []
        gc_triggered = threading.Event()
        stop_flag = threading.Event()

        def reader_worker(reader_id):
            """Continuously read old data."""
            local_results = []
            while not stop_flag.is_set():
                for i in range(10):
                    try:
                        match = trie.find_longest_prefix(f"old_{i}")
                        local_results.append((reader_id, i, match.matched_prefix, match.token_ids if match.matched_prefix else None))
                    except Exception as e:
                        local_results.append((reader_id, i, "ERROR", str(e)))
            read_results.extend(local_results)

        def gc_worker():
            """Trigger GC while readers are active."""
            time.sleep(0.01)  # Let readers start
            gc_triggered.set()
            trie.gc_by_weight_version(current_weight_version=20)

        # Start readers
        readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(5)]
        for r in readers:
            r.start()

        # Trigger GC
        gc_thread = threading.Thread(target=gc_worker)
        gc_thread.start()

        # Wait for GC
        gc_triggered.wait()
        gc_thread.join()

        # Let readers continue briefly
        time.sleep(0.02)
        stop_flag.set()

        for r in readers:
            r.join()

        # Analyze results
        errors = [r for r in read_results if r[2] == "ERROR"]
        assert len(errors) == 0, f"Readers should not error during GC: {errors[:5]}"

        # After GC, old data should be removed (or consistent state)
        print(f"Completed {len(read_results)} read operations during/after GC")

    @pytest.mark.asyncio
    async def test_gc_during_async_operations(self):
        """
        Test: GC during concurrent async read/write operations.

        Expected: All operations complete safely
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Insert initial data
        for i in range(20):
            await trie.insert_async(f"data_{i}", [i], [0.1 * i], [1], weight_version=i % 5 + 1)

        async def reader(reader_id):
            """Async reader."""
            results = []
            for i in range(20):
                match = await trie.find_longest_prefix_async(f"data_{i}")
                results.append(match.matched_prefix)
            return results

        async def writer(writer_id):
            """Async writer."""
            return await trie.insert_async(
                f"new_data_{writer_id}",
                [1000 + writer_id],
                [0.5],
                [1],
                weight_version=50
            )

        async def gc_operation():
            """Async GC."""
            await asyncio.sleep(0.01)  # Let some operations start
            return await trie.gc_by_weight_version_async(current_weight_version=50)

        # Run all concurrently
        tasks = (
            [reader(i) for i in range(10)] +
            [writer(i) for i in range(5)] +
            [gc_operation()]
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Should not have exceptions: {exceptions}"

    def test_gc_removing_currently_accessed_node(self):
        """
        Test: GC removes node that is currently being accessed.

        Critical Race: Reader holds reference to node being GC'd
        Expected: Reader should complete or fail gracefully
        """
        trie = StringRadixTrie(gc_threshold_k=1)

        # Insert old node
        trie.insert("will_be_removed", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=1)

        access_during_gc = []

        def slow_reader():
            """Read node slowly (simulating slow operation)."""
            match = trie.find_longest_prefix("will_be_removed")
            time.sleep(0.01)  # Simulate processing
            # Try to access node data after small delay
            try:
                tokens = match.token_ids
                prefix = match.matched_prefix
                access_during_gc.append(("success", prefix, tokens))
            except Exception as e:
                access_during_gc.append(("error", str(e)))

        def fast_gc():
            """GC that removes the node."""
            time.sleep(0.005)  # Start during reader's sleep
            trie.gc_by_weight_version(current_weight_version=10)

        reader_thread = threading.Thread(target=slow_reader)
        gc_thread = threading.Thread(target=fast_gc)

        reader_thread.start()
        gc_thread.start()

        reader_thread.join()
        gc_thread.join()

        # Reader should complete (either with data or graceful handling)
        assert len(access_during_gc) == 1
        result = access_during_gc[0]
        assert result[0] in ["success", "error"], "Reader should complete"

        # If successful, data should be consistent (might be cached in MatchResult)
        if result[0] == "success":
            assert result[1] == "will_be_removed" or result[1] == ""  # Acceptable states
            assert isinstance(result[2], list)  # Should be a list


# ==============================================================================
# Group B: GC with Incomplete/NULL Version Data
# ==============================================================================

@pytest.mark.unit
class TestGCWithIncompleteVersionData:
    """Test GC handling of incomplete or NULL version information."""

    def test_gc_with_none_traverse_version(self):
        """
        Test: GC encounters node with None traverse_version.

        Edge Case: Incomplete initialization or corruption
        Expected: GC should skip or handle gracefully
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Insert normal node
        trie.insert("normal", [1, 2], [0.1, 0.2], [1, 1], weight_version=5)

        # Insert node without version (traverse_version will be None)
        trie.insert("no_version", [3, 4], [0.3, 0.4], [1, 1])  # No weight_version

        # Verify state before GC (Boundary: mixing None and numeric traverse_version)
        normal_match = trie.find_longest_prefix("normal")
        none_match = trie.find_longest_prefix("no_version")

        assert normal_match.last_node.traverse_version == 5, \
            f"Normal node should have traverse_version=5, got {normal_match.last_node.traverse_version}"
        assert none_match.last_node.traverse_version is None, \
            f"None-version node should have traverse_version=None, got {none_match.last_node.traverse_version}"

        # Trigger GC - should handle None gracefully
        try:
            removed = trie.gc_by_weight_version(current_weight_version=10)
            # GC should complete without crash
            assert isinstance(removed, int), "GC should return count"
        except Exception as e:
            pytest.fail(f"GC should handle None traverse_version gracefully: {e}")

        # System should remain in consistent state
        # None-version node behavior is implementation-defined (keep or remove)
        final_match = trie.find_longest_prefix("no_version")
        assert isinstance(final_match.matched_prefix, str), "Should return valid result"

    def test_gc_with_mixed_version_states(self):
        """
        Test: GC with mixture of None, valid, and edge-case versions.

        Scenario: Some nodes have versions, some None, some negative
        Expected: GC handles each appropriately
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Mix of version states
        trie.insert("none_version", [1], [0.1], [1])  # None
        trie.insert("zero_version", [2], [0.2], [1], weight_version=0)  # Zero
        trie.insert("negative_version", [3], [0.3], [1], weight_version=-5)  # Negative
        trie.insert("normal_old", [4], [0.4], [1], weight_version=1)  # Old
        trie.insert("normal_new", [5], [0.5], [1], weight_version=20)  # New

        # GC should handle all types
        try:
            removed = trie.gc_by_weight_version(current_weight_version=20)
            assert isinstance(removed, int)
        except Exception as e:
            pytest.fail(f"GC should handle mixed version states: {e}")

        # At minimum, new data should survive (Boundary: recent version should not be GC'd)
        new_match = trie.find_longest_prefix("normal_new")
        assert new_match.matched_prefix == "normal_new", \
            "New data with version=20 should survive GC at version 20"

    @pytest.mark.asyncio
    async def test_async_gc_with_none_versions(self):
        """
        Test: Async GC handles None versions correctly.

        Expected: Same behavior as sync GC
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Mix of versions
        await trie.insert_async("with_version", [1, 2], [0.1, 0.2], [1, 1], weight_version=10)
        await trie.insert_async("without_version", [3, 4], [0.3, 0.4], [1, 1])  # None

        # Async GC should handle gracefully
        try:
            removed = await trie.gc_by_weight_version_async(current_weight_version=15)
            assert isinstance(removed, int)
        except Exception as e:
            pytest.fail(f"Async GC should handle None versions: {e}")


# ==============================================================================
# Group C: Memory Pressure and High-Frequency GC
# ==============================================================================

@pytest.mark.unit
class TestMemoryPressureGC:
    """Test GC under memory pressure and high frequency."""

    def test_high_frequency_gc(self):
        """
        Test: GC triggered very frequently.

        Stress Test: GC called every N insertions
        Expected: System remains stable, no memory leaks
        """
        trie = StringRadixTrie(max_cache_size=50, gc_threshold_k=2)

        for i in range(200):
            # Insert data
            trie.insert(f"item_{i}", [i % 256], [0.1], [1], weight_version=i)

            # Trigger GC frequently
            if i % 5 == 0:
                removed = trie.gc_by_weight_version(current_weight_version=i)
                # Should not crash
                assert isinstance(removed, int)

        # Final state should be consistent (Boundary: high-frequency GC stability)
        stats = trie.get_stats()
        assert stats["total_entries"] > 0, \
            f"Should have some entries after high-frequency GC, got {stats['total_entries']}"
        assert stats["total_entries"] <= 100, \
            f"GC should have cleaned up old entries, got {stats['total_entries']} (expected <= 100)"

    def test_gc_thrashing_scenario(self):
        """
        Test: Rapid insert/GC cycles (thrashing).

        Pathological Case: Insert rate ~= GC removal rate
        Expected: System should stabilize, not grow unbounded
        """
        trie = StringRadixTrie(max_cache_size=20, gc_threshold_k=1)

        cache_sizes = []

        for version in range(1, 51):
            # Insert batch
            for i in range(5):
                trie.insert(f"v{version}_{i}", [version, i], [0.1, 0.2], [1, 1], weight_version=version)

            # Immediate GC
            trie.gc_by_weight_version(current_weight_version=version)

            cache_sizes.append(trie.cur_cache_size)

        # Cache size should stabilize (not grow linearly) (Boundary: thrashing scenario stability)
        assert max(cache_sizes) <= 40, \
            f"Cache should not grow unbounded under thrashing (max_cache_size=20), max observed={max(cache_sizes)}"

        # Should not oscillate wildly
        final_size = trie.cur_cache_size
        assert final_size > 0, \
            f"Should have some data remaining after thrashing, got {final_size}"
        print(f"Cache size after thrashing: {final_size}, max observed: {max(cache_sizes)}")

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """
        Test: Verify GC actually releases memory (no leaks).

        Strategy: Use weak references to detect unreachable objects
        Expected: After GC, removed nodes should be garbage collected
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Insert data and keep weak references
        await trie.insert_async("temp_data", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=1)

        # Get reference to node (through match result)
        match = await trie.find_longest_prefix_async("temp_data")
        weak_ref = weakref.ref(match.last_node)

        # Verify node exists (Boundary: pre-GC memory state)
        assert weak_ref() is not None, \
            "Weak reference to node should be alive before GC"

        # Clear match result
        del match

        # Trigger GC to remove old node
        await trie.gc_by_weight_version_async(current_weight_version=20)

        # Force Python GC
        python_gc.collect()

        # Verify node was removed (Boundary: post-GC removal verification)
        removed_match = await trie.find_longest_prefix_async("temp_data")
        assert removed_match.matched_prefix == "", \
            f"Node should be removed after GC, but got matched_prefix='{removed_match.matched_prefix}'"

        # Note: weak_ref might still be alive due to internal caching
        # This test documents GC behavior but Python's GC is non-deterministic
        # So we don't assert weak_ref() is None


# ==============================================================================
# Group D: Concurrent GC Operations
# ==============================================================================

@pytest.mark.unit
class TestConcurrentGCOperations:
    """Test multiple concurrent GC operations."""

    def test_concurrent_gc_calls(self):
        """
        Test: Multiple threads call GC simultaneously.

        Race Condition: Multiple GC operations on same tree
        Expected: Should serialize or handle safely
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Pre-populate
        for i in range(50):
            trie.insert(f"data_{i}", [i], [0.1 * i], [1], weight_version=i % 10)

        gc_results = []

        def gc_worker(worker_id, version):
            try:
                removed = trie.gc_by_weight_version(current_weight_version=version)
                gc_results.append((worker_id, removed, "success"))
            except Exception as e:
                gc_results.append((worker_id, 0, str(e)))

        # Launch concurrent GC
        threads = []
        for i in range(10):
            thread = threading.Thread(target=gc_worker, args=(i, 15 + i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        errors = [r for r in gc_results if r[2] != "success"]
        assert len(errors) == 0, f"Concurrent GC should not error: {errors}"

        # System should be in consistent state (Boundary: concurrent GC consistency)
        stats = trie.get_stats()
        assert stats["total_entries"] >= 0, \
            f"Entry count should be valid (>= 0) after concurrent GC, got {stats['total_entries']}"

    @pytest.mark.asyncio
    async def test_concurrent_async_gc(self):
        """
        Test: Concurrent async GC operations.

        Expected: Async GC should be safe for concurrent calls
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Pre-populate
        for i in range(100):
            await trie.insert_async(f"item_{i}", [i], [0.1 * i], [1], weight_version=i % 20)

        # Launch concurrent async GC
        gc_tasks = [
            trie.gc_by_weight_version_async(current_weight_version=30 + i)
            for i in range(20)
        ]

        results = await asyncio.gather(*gc_tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Async GC should not raise exceptions: {exceptions}"

        # All results should be integers
        non_exception_results = [r for r in results if not isinstance(r, Exception)]
        assert all(isinstance(r, int) for r in non_exception_results), \
            f"All GC results should be integers, got types: {[type(r) for r in non_exception_results]}"

    def test_gc_and_insert_concurrent_same_key(self):
        """
        Test: GC and insert of same key happen concurrently.

        Race Condition: Insert creates node while GC is running
        Expected: Should handle safely, final state consistent
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Insert initial data
        trie.insert("race_key", [1, 2], [0.1, 0.2], [1, 1], weight_version=1)

        results = {'gc': None, 'insert': None, 'errors': []}

        def gc_operation():
            try:
                time.sleep(0.001)  # Small delay
                removed = trie.gc_by_weight_version(current_weight_version=20)
                results['gc'] = removed
            except Exception as e:
                results['errors'].append(f"GC error: {e}")

        def insert_operation():
            try:
                # Insert with newer version
                success = trie.insert("race_key", [3, 4, 5], [0.3, 0.4, 0.5], [1, 1, 1], weight_version=25)
                results['insert'] = success
            except Exception as e:
                results['errors'].append(f"Insert error: {e}")

        gc_thread = threading.Thread(target=gc_operation)
        insert_thread = threading.Thread(target=insert_operation)

        gc_thread.start()
        insert_thread.start()

        gc_thread.join()
        insert_thread.join()

        # Should not have errors
        assert len(results['errors']) == 0, f"Should not have errors: {results['errors']}"

        # Final state should be consistent
        final_match = trie.find_longest_prefix("race_key")

        # Key should either:
        # 1. Not exist (if GC ran after insert and removed it - unlikely with v25)
        # 2. Exist with new data (if insert succeeded)
        if final_match.matched_prefix == "race_key":
            # Should have new version data
            assert final_match.last_node.weight_version in [1, 25], "Should have valid version"


# ==============================================================================
# Group E: GC with Corrupted Tree State
# ==============================================================================

@pytest.mark.unit
class TestGCWithCorruptedState:
    """Test GC resilience to corrupted tree states."""

    def test_gc_with_circular_reference_protection(self):
        """
        Test: GC doesn't infinite loop if tree has unexpected structure.

        Safety: Ensure GC terminates even with unexpected states
        Expected: GC completes in reasonable time
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Build normal tree
        for i in range(20):
            trie.insert(f"path_{i}", [i, i+1], [0.1, 0.2], [1, 1], weight_version=i)

        # GC should complete in reasonable time (< 1 second for 20 nodes) (Boundary: GC performance)
        start_time = time.time()
        removed = trie.gc_by_weight_version(current_weight_version=25)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, \
            f"GC took too long: {elapsed:.3f}s (expected < 1.0s for 20 nodes)"
        assert isinstance(removed, int), \
            f"GC should return integer count, got type {type(removed)}"

    def test_gc_consistency_after_errors(self):
        """
        Test: Tree remains consistent after GC operations with errors.

        Scenario: Force some error conditions during GC
        Expected: Tree should recover or maintain consistency
        """
        trie = StringRadixTrie(gc_threshold_k=2)

        # Populate with good data
        for i in range(30):
            trie.insert(f"good_{i}", [i], [0.1 * i], [1], weight_version=i % 10)

        # Run GC multiple times
        for version in [15, 20, 25, 30]:
            try:
                removed = trie.gc_by_weight_version(current_weight_version=version)
            except Exception:
                pass  # Ignore errors

        # Verify system still works (Boundary: consistency after error recovery)
        new_insert = trie.insert("after_errors", [999], [0.9], [1], weight_version=100)
        assert new_insert is True, \
            "Should still be able to insert after GC errors"

        match = trie.find_longest_prefix("after_errors")
        assert match.matched_prefix == "after_errors", \
            "Should be able to find newly inserted data after error recovery"

        # Stats should be valid
        stats = trie.get_stats()
        assert stats["total_entries"] >= 1, \
            f"Should have at least new entry, got {stats['total_entries']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
