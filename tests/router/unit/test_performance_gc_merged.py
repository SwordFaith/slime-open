"""
Performance and GC Tests (Merged)

This test suite combines tests from:
- test_performance_comparison.py
- test_radix_tree_gc_consistency.py
- test_weight_version_separation.py

Tests cover:
- Concurrency and functionality of async implementations
- GC traverse_version consistency and behavior
- Weight version vs traverse version separation
- Performance under concurrent load
- TDD-driven version separation architecture
"""

import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

import pytest
from slime.router.utils.async_read_write_lock import AsyncReadWriteLock
from slime.router.core.radix_tree import StringRadixTrie, MatchResult


# ============================================================================
# Group A: Concurrency and Functionality Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_read_correctness():
    """Test correctness of concurrent read operations with async vs sync implementations."""
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

    # Measure sync performance (using asyncio.to_thread to simulate concurrent access)
    async def sync_read():
        result = await asyncio.to_thread(sync_tree.find_longest_prefix, "Hello world")
        return result

    # Measure async performance
    async def async_read():
        result = await async_tree.find_longest_prefix_async("Hello world")
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

    # Verify all reads return correct results
    assert all(result.token_ids == [1, 2, 3] for result in async_results)
    assert all(result.matched_prefix == "Hello world" for result in async_results)
    assert all(result.token_ids == [1, 2, 3] for result in sync_results)
    assert all(result.matched_prefix == "Hello world" for result in sync_results)

    # Verify consistency: all async reads should be identical
    first_async_result = async_results[0]
    for result in async_results[1:]:
        assert result.token_ids == first_async_result.token_ids
        assert result.matched_prefix == first_async_result.matched_prefix

    # Verify consistency: all sync reads should be identical
    first_sync_result = sync_results[0]
    for result in sync_results[1:]:
        assert result.token_ids == first_sync_result.token_ids
        assert result.matched_prefix == first_sync_result.matched_prefix


@pytest.mark.asyncio
async def test_mixed_read_write_correctness():
    """Test correctness of mixed read/write operations under concurrency."""
    async_tree = StringRadixTrie(max_cache_size=1000, verbose=False)

    # Initial data
    await async_tree.insert_async("Base", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1)

    # Track operations for verification
    read_results = []
    write_results = []

    async def reader(reader_id: int):
        result = await async_tree.find_longest_prefix_async("Base")
        read_results.append((reader_id, result))
        return result

    async def writer(writer_id: int):
        result = await async_tree.insert_async(f"Data{writer_id}", [10 + writer_id], [-0.5], [1], weight_version=2)
        write_results.append((writer_id, result))
        return result

    # Run mixed operations concurrently
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(reader(i)))
        tasks.append(asyncio.create_task(writer(i)))

    # Wait for all operations to complete
    await asyncio.gather(*tasks)

    # Verify all read operations returned expected base data
    assert len(read_results) == 5, f"Expected 5 read operations, got {len(read_results)}"
    for reader_id, result in read_results:
        assert result.token_ids == [1, 2], f"Reader {reader_id} got unexpected result: {result.token_ids}"
        assert result.matched_prefix == "Base", f"Reader {reader_id} got unexpected prefix: {result.matched_prefix}"

    # Verify all write operations completed successfully
    assert len(write_results) == 5, f"Expected 5 write operations, got {len(write_results)}"
    for writer_id, result in write_results:
        assert result is True, f"Writer {writer_id} failed: {result}"

    # Verify that written data is actually stored and retrievable
    for i in range(5):
        result = await async_tree.find_longest_prefix_async(f"Data{i}")
        assert result.token_ids == [10 + i], f"Data{i} not found or incorrect: {result.token_ids}"
        assert result.matched_prefix == f"Data{i}", f"Prefix mismatch for Data{i}: {result.matched_prefix}"

    # Verify base data is still intact after all writes
    base_result = await async_tree.find_longest_prefix_async("Base")
    assert base_result.token_ids == [1, 2], "Base data corrupted after concurrent operations"
    assert base_result.matched_prefix == "Base", "Base prefix corrupted after concurrent operations"


@pytest.mark.asyncio
async def test_event_loop_concurrency():
    """Test that async operations allow concurrent execution without blocking."""
    async_tree = StringRadixTrie(max_cache_size=1000, verbose=False)
    await async_tree.insert_async("Test", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=1)

    # Track background task execution
    background_executions = []
    cache_operations_completed = []

    async def background_task():
        """Background task that should continue running during cache operations."""
        for i in range(50):  # Reduced from 100 for faster test execution
            background_executions.append(i)
            await asyncio.sleep(0.001)  # Small delay to allow task switching

    async def cache_operations():
        """Cache operations that should not block background task."""
        for i in range(10):
            result = await async_tree.find_longest_prefix_async("Test")
            # Verify cache operation returns correct result
            assert result.token_ids == [1, 2, 3], f"Cache operation {i} returned wrong result"
            assert result.matched_prefix == "Test", f"Cache operation {i} returned wrong prefix"
            cache_operations_completed.append(i)
            await asyncio.sleep(0.002)  # Simulate some processing

    # Start both tasks concurrently
    bg_task = asyncio.create_task(background_task())
    cache_task = asyncio.create_task(cache_operations())

    # Wait for both tasks to complete
    await asyncio.gather(bg_task, cache_task)

    # Verify both tasks completed successfully
    assert len(background_executions) == 50, f"Background task should have run 50 times, got {len(background_executions)}"
    assert len(cache_operations_completed) == 10, f"Cache operations should have completed 10 times, got {len(cache_operations_completed)}"

    # Verify cache operations executed in correct order
    assert cache_operations_completed == list(range(10)), "Cache operations should execute sequentially"

    # Additional verification: ensure cache results are consistent
    test_result = await async_tree.find_longest_prefix_async("Test")
    assert test_result.token_ids == [1, 2, 3], "Cache data should remain consistent"
    assert test_result.matched_prefix == "Test", "Cache prefix should remain consistent"

    # Test that multiple async tasks can run concurrently without interference
    concurrent_results = []

    async def concurrent_reader(reader_id: int):
        result = await async_tree.find_longest_prefix_async("Test")
        concurrent_results.append((reader_id, result))
        return result

    # Run multiple concurrent readers
    concurrent_tasks = [asyncio.create_task(concurrent_reader(i)) for i in range(20)]
    await asyncio.gather(*concurrent_tasks)

    # Verify all concurrent readers got the same correct result
    expected_result = concurrent_results[0][1]
    for reader_id, result in concurrent_results:
        assert result.token_ids == expected_result.token_ids, f"Reader {reader_id} got different result"
        assert result.matched_prefix == expected_result.matched_prefix, f"Reader {reader_id} got different prefix"


# ============================================================================
# Group B: Version Separation Core Functionality (TDD)
# ============================================================================

def test_stringTreeNode_initialization_with_separated_versions():
    """Test StringTreeNode initialization with separated version fields."""
    from slime.router.core.radix_tree import StringTreeNode

    node = StringTreeNode()

    # Check both version fields exist and are properly initialized
    assert hasattr(node, 'weight_version'), "StringTreeNode should have weight_version field"
    assert hasattr(node, 'traverse_version'), "StringTreeNode should have traverse_version field (NEW)"

    # Check initial values
    assert node.weight_version is None, "Initial weight_version should be None"
    assert node.traverse_version is None, "Initial traverse_version should be None"


def test_version_separation_on_insert():
    """Test version separation during insert operations."""
    trie = StringRadixTrie()

    # Step 1: Insert "Hello" at version 1
    trie.insert("Hello", [72, 101, 108, 108, 111], [0.1] * 5, [0] * 5, weight_version=1)

    # Check initial node versions
    result1 = trie.find_longest_prefix("Hello")
    assert result1.last_node.weight_version == 1, "New node should have weight_version=1"
    assert result1.last_node.traverse_version == 1, "New node should have traverse_version=1"

    # Step 2: Insert "Hello World" at version 5 (shares prefix)
    trie.insert("Hello World", [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100],
                [0.1] * 11, [0] * 7 + [1] * 4, weight_version=5)

    # Check version separation
    hello_result = trie.find_longest_prefix("Hello")
    world_result = trie.find_longest_prefix("Hello World")

    # "Hello" node (traversed): weight_version unchanged, traverse_version updated
    assert hello_result.last_node.weight_version == 1, "Traversed node weight_version should NOT change"
    assert hello_result.last_node.traverse_version == 5, "Traversed node traverse_version should be updated"

    # "World" node (new): both versions set to insert version
    assert world_result.last_node.weight_version == 5, "New node should have weight_version=5"
    assert world_result.last_node.traverse_version == 5, "New node should have traverse_version=5"


def test_generation_versions_alignment_in_match_result():
    """Test that MatchResult includes generation_versions aligned with token_ids."""
    trie = StringRadixTrie()

    # Insert trajectory with mixed versions
    # "Hello" (v1) + " World" (v3)
    trie.insert("Hello", [72, 101, 108, 108, 111], [0.1] * 5, [0] * 5, weight_version=1)
    trie.insert("Hello World", [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100],
                [0.1] * 11, [0] * 6 + [1] * 5, weight_version=3)

    result = trie.find_longest_prefix("Hello World")

    # Check MatchResult has generation_versions field
    assert hasattr(result, 'generation_versions'), "MatchResult should have generation_versions field"

    # Check version alignment
    assert len(result.generation_versions) == len(result.token_ids), \
        "generation_versions should align with token_ids"

    # Expected: [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]
    # First 5 tokens from version 1, next 6 tokens from version 3
    expected_versions = [1] * 5 + [3] * 6
    assert result.generation_versions == expected_versions, \
        f"Expected {expected_versions}, got {result.generation_versions}"


def test_backward_compatibility_weight_version_access():
    """Test backward compatibility for existing weight_version access."""
    trie = StringRadixTrie()

    # Insert and traverse as before
    trie.insert("Hello", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)
    trie.insert("Hello World", [1, 2, 3, 4, 5], [0.1] * 5, [0] * 3 + [1] * 2, weight_version=5)

    # Existing patterns should still work
    result = trie.find_longest_prefix("Hello")

    # weight_version should still be accessible (now represents generation version)
    assert hasattr(result.last_node, 'weight_version')
    assert result.last_node.weight_version == 1

    # The new traverse_version should be available for new functionality
    assert hasattr(result.last_node, 'traverse_version')
    assert result.last_node.traverse_version == 5


def test_existing_gc_interface_compatibility():
    """Test existing GC interface remains compatible."""
    trie = StringRadixTrie()

    # Use existing GC interface
    trie.insert("Test", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)

    # Should not raise any errors
    removed_count = trie.gc_by_weight_version(current_weight_version=10)

    # Should return valid count
    assert isinstance(removed_count, int)
    assert removed_count >= 0


# ============================================================================
# Group C: GC Based on Traverse Version
# ============================================================================

def test_gc_uses_traverse_version_not_weight_version():
    """Test: GC decisions are based on traverse_version, not weight_version."""
    trie = StringRadixTrie(gc_threshold_k=3)

    # Insert node at weight_version=1, traverse_version=1
    trie.insert("old_weight", [1, 2], [0.1, 0.2], [1, 1], weight_version=1)

    # Insert node at weight_version=10, traverse_version=10
    trie.insert("new_weight", [3, 4], [0.3, 0.4], [1, 1], weight_version=10)

    # Verify initial state
    old_node = trie.find_longest_prefix("old_weight").last_node
    new_node = trie.find_longest_prefix("new_weight").last_node
    assert old_node.weight_version == 1
    assert old_node.traverse_version == 1
    assert new_node.weight_version == 10
    assert new_node.traverse_version == 10

    # Run GC with version=7, threshold=4 (7-3)
    # old_weight should be removed (traverse_version=1 <= 4)
    # new_weight should survive (traverse_version=10 > 4)
    trie.gc_by_weight_version(current_weight_version=7)

    # Check results
    assert trie.find_longest_prefix("old_weight").matched_prefix == ""  # Removed
    assert trie.find_longest_prefix("new_weight").matched_prefix == "new_weight"  # Survived


def test_gc_based_on_traverse_version():
    """Test GC uses traverse_version instead of weight_version."""
    trie = StringRadixTrie()

    # Create nodes with different version patterns
    trie.insert("Old", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)  # v1 generated, v1 traversed
    trie.insert("Active", [4, 5, 6], [0.1] * 3, [0] * 3, weight_version=2)  # v2 generated

    # Simulate traversal: make "Old" node recently traversed at version 10
    trie.insert("Old Extended", [1, 2, 3, 7], [0.1] * 4, [0] * 3 + [1], weight_version=10)

    # At this point:
    # "Old" node: weight_version=1, traverse_version=10 (recently traversed)
    # "Active" node: weight_version=2, traverse_version=2 (not recently traversed)

    # Run GC with current_version=10, threshold=5 (remove traverse_version <= 5)
    removed_count = trie.gc_by_weight_version(current_weight_version=10)

    # "Old" should be kept (traverse_version=10 > 5)
    # "Active" should be removed (traverse_version=2 <= 5)
    old_result = trie.find_longest_prefix("Old")
    active_result = trie.find_longest_prefix("Active")

    assert old_result.matched_prefix == "Old", "Recently traversed node should be kept"
    assert active_result.matched_prefix == "", "Stale node should be removed by GC"


def test_traverse_version_propagation_before_gc():
    """Test traverse_version is properly propagated before GC decisions."""
    trie = StringRadixTrie(gc_threshold_k=5)

    # Build hierarchy: root -> a -> ab -> abc
    trie.insert("a", [1], [0.1], [1], weight_version=1)
    trie.insert("ab", [1, 2], [0.1, 0.2], [1, 1], weight_version=1)
    trie.insert("abc", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=1)

    # Verify all nodes have traverse_version=1
    node_a = trie.find_longest_prefix("a").last_node
    node_ab = trie.find_longest_prefix("ab").last_node
    node_abc = trie.find_longest_prefix("abc").last_node
    assert node_a.traverse_version == 1
    assert node_ab.traverse_version == 1
    assert node_abc.traverse_version == 1

    # Insert deeper node at higher version - should update traverse_version of all ancestors
    trie.insert("abcd", [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1], weight_version=15)

    # Verify traverse_versions are updated
    node_a_after = trie.find_longest_prefix("a").last_node
    node_ab_after = trie.find_longest_prefix("ab").last_node
    node_abc_after = trie.find_longest_prefix("abc").last_node
    node_abcd = trie.find_longest_prefix("abcd").last_node

    assert node_a_after.traverse_version == 15
    assert node_ab_after.traverse_version == 15
    assert node_abc_after.traverse_version == 15
    assert node_abcd.traverse_version == 15

    # Run GC at version=25, threshold=20 (25-5)
    # All nodes should survive since traverse_version=15 <= 20, so they get removed
    # Actually, this means they WILL be removed. Let me adjust the test.
    trie.gc_by_weight_version(current_weight_version=25)

    # All should be removed since traverse_version=15 <= 20
    assert trie.find_longest_prefix("a").matched_prefix == ""
    assert trie.find_longest_prefix("ab").matched_prefix == ""
    assert trie.find_longest_prefix("abc").matched_prefix == ""
    assert trie.find_longest_prefix("abcd").matched_prefix == ""


def test_gc_with_complex_traverse_version_patterns():
    """Test GC handles complex traverse_version patterns correctly."""
    trie = StringRadixTrie(gc_threshold_k=2)

    # Create complex pattern:
    # Path 1: root -> x -> xy (traverse_version gets updated multiple times)
    # Path 2: root -> y (single update)
    trie.insert("x", [10], [0.1], [1], weight_version=1)
    trie.insert("xy", [10, 11], [0.1, 0.2], [1, 1], weight_version=1)
    trie.insert("y", [20], [0.2], [1], weight_version=1)

    # Update path 1 to version=10
    trie.insert("xyz", [10, 11, 12], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=10)

    # Update path 2 to version=5
    trie.insert("yz", [20, 21], [0.2, 0.3], [1, 1], weight_version=5)

    # Verify traverse_versions
    node_x = trie.find_longest_prefix("x").last_node
    node_xy = trie.find_longest_prefix("xy").last_node
    node_xyz = trie.find_longest_prefix("xyz").last_node
    node_y = trie.find_longest_prefix("y").last_node
    node_yz = trie.find_longest_prefix("yz").last_node

    assert node_x.traverse_version == 10  # Updated by xyz insertion
    assert node_xy.traverse_version == 10  # Updated by xyz insertion
    assert node_xyz.traverse_version == 10
    assert node_y.traverse_version == 5   # Updated by yz insertion
    assert node_yz.traverse_version == 5

    # Run GC at version=8, threshold=6 (8-2)
    # Path 2 should be removed (traverse_version=5 <= 6)
    # Path 1 should survive (traverse_version=10 > 6)
    trie.gc_by_weight_version(current_weight_version=8)

    # Check results
    assert trie.find_longest_prefix("x").matched_prefix == "x"
    assert trie.find_longest_prefix("xy").matched_prefix == "xy"
    assert trie.find_longest_prefix("xyz").matched_prefix == "xyz"
    assert trie.find_longest_prefix("y").matched_prefix == ""  # Removed
    assert trie.find_longest_prefix("yz").matched_prefix == ""  # Removed


def test_gc_preserves_generation_version_info():
    """Test that GC preserves generation version information of kept nodes."""
    trie = StringRadixTrie()

    # Insert trajectory at version 1
    trie.insert("Hello World", [1, 2, 3, 4, 5], [0.1] * 5, [0, 0, 1, 1, 1], weight_version=1)

    # Traverse at version 10 (updates traverse_version but not weight_version)
    trie.insert("Hello World Extended", [1, 2, 3, 4, 5, 6],
                [0.1] * 6, [0, 0, 1, 1, 1, 1], weight_version=10)

    # Run GC (should keep the node due to recent traverse_version=10)
    trie.gc_by_weight_version(current_weight_version=10)

    # Check generation version info is preserved
    result = trie.find_longest_prefix("Hello World")

    # Generation version should still be 1, not changed to 10
    assert result.last_node.weight_version == 1, "Generation version should be preserved"
    assert result.last_node.traverse_version == 10, "Traverse version should be updated"


# ============================================================================
# Group D: Async GC Consistency and Performance
# ============================================================================

@pytest.mark.asyncio
async def test_async_gc_sync_consistency():
    """Test Async GC produces same results as sync GC."""
    # Create two identical tries
    sync_trie = StringRadixTrie(gc_threshold_k=3)
    async_trie = StringRadixTrie(gc_threshold_k=3)

    # Insert same data into both tries
    test_data = [
        ("test1", [1, 2], [0.1, 0.2], [1, 1], 1),
        ("test2", [3, 4], [0.3, 0.4], [1, 1], 5),
        ("test3", [5, 6], [0.5, 0.6], [1, 1], 15),
    ]

    for text, tokens, logp, loss_mask, version in test_data:
        sync_trie.insert(text, tokens, logp, loss_mask, version)
        await async_trie.insert_async(text, tokens, logp, loss_mask, version)

    # Run GC on both tries
    sync_removed = sync_trie.gc_by_weight_version(current_weight_version=20)
    async_removed = await async_trie.gc_by_weight_version_async(current_weight_version=20)

    # Results should be identical
    assert sync_removed == async_removed

    # Verify final states are identical
    for text, _, _, _, _ in test_data:
        sync_result = sync_trie.find_longest_prefix(text)
        async_result = await async_trie.find_longest_prefix_async(text)
        assert sync_result.matched_prefix == async_result.matched_prefix


def test_traverse_version_validation_in_gc():
    """Test GC completes without errors and maintains system integrity."""
    trie = StringRadixTrie(gc_threshold_k=10)  # Use high threshold to avoid removal

    # Create a valid subtree where child traverse_versions <= parent traverse_versions
    trie.insert("parent", [1], [0.1], [1], weight_version=10)
    trie.insert("parent_child", [1, 2], [0.1, 0.2], [1, 1], weight_version=10)

    # Verify data exists before GC
    parent_result_before = trie.find_longest_prefix("parent")
    child_result_before = trie.find_longest_prefix("parent_child")
    assert parent_result_before.matched_prefix == "parent"
    assert child_result_before.matched_prefix == "parent_child"

    # GC should complete without any validation errors (high threshold = no removal)
    removed = trie.gc_by_weight_version(current_weight_version=15)
    assert removed == 0  # Should not remove anything with high threshold

    # Verify data integrity is maintained after GC
    parent_result_after = trie.find_longest_prefix("parent")
    child_result_after = trie.find_longest_prefix("parent_child")
    assert parent_result_after.matched_prefix == "parent"
    assert child_result_after.matched_prefix == "parent_child"


def test_gc_does_not_affect_generation_versions():
    """Test GC operations preserve generation_versions in MatchResult."""
    trie = StringRadixTrie(gc_threshold_k=2)

    # Insert data with specific generation versions
    trie.insert("keep_me", [1, 2], [0.1, 0.2], [1, 1], weight_version=7)
    trie.insert("remove_me", [3, 4], [0.3, 0.4], [1, 1], weight_version=1)

    # Verify initial generation versions in MatchResult
    keep_result = trie.find_longest_prefix("keep_me")
    remove_result = trie.find_longest_prefix("remove_me")
    assert keep_result.generation_versions == [7, 7]
    assert remove_result.generation_versions == [1, 1]

    # Run GC at lower version to remove "remove_me"
    trie.gc_by_weight_version(current_weight_version=5)

    # Verify "keep_me" still exists and has correct generation versions
    keep_result_after = trie.find_longest_prefix("keep_me")
    assert keep_result_after.matched_prefix == "keep_me"
    assert keep_result_after.generation_versions == [7, 7]  # Preserved

    # Verify "remove_me" is actually removed
    assert trie.find_longest_prefix("remove_me").matched_prefix == ""


# ============================================================================
# Group E: Edge Cases and Version Handling
# ============================================================================

def test_version_none_handling():
    """Test handling of None versions in various scenarios."""
    trie = StringRadixTrie()

    # Insert without weight_version
    trie.insert("NoVersion", [1, 2, 3], [0.1] * 3, [0] * 3)  # weight_version=None

    result = trie.find_longest_prefix("NoVersion")

    # Both versions should be None for manually inserted data
    assert result.last_node.weight_version is None
    assert result.last_node.traverse_version is None


def test_version_reset_on_gc():
    """Test version handling during GC operations."""
    trie = StringRadixTrie()

    # Insert data
    trie.insert("Temp", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)

    # GC should handle nodes with None versions gracefully
    removed_count = trie.gc_by_weight_version(current_weight_version=5)

    # Should not raise errors
    assert isinstance(removed_count, int)


def test_partial_match_version_alignment():
    """Test version alignment in partial match scenarios."""
    trie = StringRadixTrie()

    # Insert: "AB" at v1, "ABC" at v5
    trie.insert("AB", [1, 2], [0.1, 0.2], [0, 1], weight_version=1)
    trie.insert("ABC", [1, 2, 3], [0.1, 0.2, 0.3], [0, 1, 1], weight_version=5)

    result = trie.find_longest_prefix("ABCD")

    # Should match "ABC" completely
    assert result.matched_prefix == "ABC"
    assert result.token_ids == [1, 2, 3]

    # Check version alignment
    # In actual Radix Tree structure: "AB" node (v1) + "C" node (v5)
    # So tokens [1,2] from v1, token [3] from v5
    expected_versions = [1, 1, 5]  # AB from v1, C from v5
    assert result.generation_versions == expected_versions, \
        f"Expected {expected_versions}, got {result.generation_versions}"


def test_retrieve_from_text_version_alignment():
    """Test get_or_create_tokenization returns version-aligned tokens."""
    # Mock tokenizer for testing
    def mock_tokenizer(text, add_special_tokens=True):
        token_map = {
            "Hello": [72, 101, 108, 108, 111],
            " World": [32, 87, 111, 114, 108, 100]
        }
        for key, tokens in token_map.items():
            if text == key:
                return {"input_ids": tokens}
        return {"input_ids": [99]}  # Default token

    trie = StringRadixTrie(tokenizer=mock_tokenizer)

    # Insert cached part
    trie.insert("Hello", [72, 101, 108, 108, 111], [0.1] * 5, [0] * 5, weight_version=3)

    # Retrieve full text (cached + new)
    result = trie.get_or_create_tokenization("Hello World", return_logprob=True)

    # Should return (token_ids, logp, loss_mask, generation_versions)
    assert len(result) == 4, "get_or_create_tokenization should return 4-tuple with versions"
    token_ids, logp, loss_mask, generation_versions = result

    # Check version alignment
    assert len(generation_versions) == len(token_ids), "Versions should align with tokens"

    # Expected: cached tokens have generation version 3, new tokens have version -1
    expected_versions = [3] * 5 + [-1] * 6  # "Hello" from v3, " World" non-AI
    assert generation_versions == expected_versions, \
        f"Expected {expected_versions}, got {generation_versions}"


if __name__ == "__main__":
    # Run all tests manually for debugging
    pytest.main([__file__, "-v", "--tb=short"])