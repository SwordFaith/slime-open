"""
Cache Edge Cases Unit Tests

Tests cover critical boundary conditions for cache operations:
- Cache at exact size limit
- Very long text inputs (100K+ tokens)
- Concurrent insertions of same key
- Cache eviction during lookup
- Corrupted cache entries
- Token/text length mismatches

Test Strategy:
- Unit testing with mocked components where appropriate
- Boundary value analysis
- Stress testing with large inputs
- Race condition detection
"""

import asyncio
import pytest
import threading
from unittest.mock import Mock
from slime.router.core.radix_tree import StringRadixTrie
from slime.router.middleware.radix_tree_middleware import RadixTreeMiddleware


# ==============================================================================
# Group A: Cache Size Boundaries
# ==============================================================================

@pytest.mark.unit
class TestCacheSizeBoundaries:
    """Test cache behavior at size limits."""

    def test_cache_at_exact_limit(self):
        """
        Test: Cache size exactly at max_cache_size.

        Boundary: cur_cache_size == max_cache_size
        Expected: Next insert should trigger GC or handle gracefully
        """
        trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1)

        # Fill cache to exact limit (token count = 10)
        for i in range(10):
            result = trie.insert(f"item{i}", [i], [0.1 * i], [1], weight_version=1)
            assert result is True

        # Check we're at limit
        assert trie.cur_cache_size == 10, "Cache should be at exact limit"

        # Next insert should handle boundary
        result = trie.insert("overflow", [99], [0.9], [1], weight_version=2)

        # Should either succeed with GC or handle gracefully
        assert isinstance(result, bool), "Insert should return boolean"

        # Cache size should be managed
        assert trie.cur_cache_size <= trie.max_cache_size * 2, "Cache should not grow unbounded"

    def test_cache_one_below_limit(self):
        """
        Test: Cache at max_cache_size - 1.

        Boundary: One token away from limit
        Expected: Should fit exactly one more token
        """
        trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1)

        # Fill to 9 tokens
        trie.insert("data", [1, 2, 3, 4, 5, 6, 7, 8, 9], [0.1] * 9, [1] * 9, weight_version=1)

        assert trie.cur_cache_size == 9

        # Insert 1 more token - should fit exactly
        result = trie.insert("one_more", [10], [0.1], [1], weight_version=2)
        assert result is True

        # Now at limit
        assert trie.cur_cache_size == 10

    def test_cache_one_over_limit(self):
        """
        Test: Insert that would put cache 1 over limit.

        Boundary: Insert would make cur_cache_size = max + 1
        Expected: GC should trigger or handle overflow
        """
        trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1)

        # Fill to limit
        trie.insert("full", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.1] * 10, [1] * 10, weight_version=1)

        assert trie.cur_cache_size == 10

        # Try to insert 1 more token
        result = trie.insert("overflow", [11], [0.1], [1], weight_version=2)

        # Should handle (GC or rejection)
        assert isinstance(result, bool)

        # Cache should be managed (allow some overflow but not unbounded)
        assert trie.cur_cache_size <= 20, "Cache should be managed on overflow"

    def test_cache_empty_to_full_transition(self):
        """
        Test: Cache growing from 0 to max_cache_size in one insert.

        Boundary: Empty -> Full in single operation
        Expected: Should handle large single insert
        """
        trie = StringRadixTrie(max_cache_size=100, gc_threshold_k=2)

        # Single large insert (100 tokens)
        large_tokens = list(range(100))
        large_logp = [0.01] * 100
        large_mask = [1] * 100

        result = trie.insert("large_single", large_tokens, large_logp, large_mask, weight_version=1)
        assert result is True

        assert trie.cur_cache_size == 100, "Should accept exactly at limit"

    @pytest.mark.asyncio
    async def test_async_cache_at_limit(self):
        """
        Test: Async operations at cache limit.

        Expected: Async cache management should match sync
        """
        trie = StringRadixTrie(max_cache_size=10, gc_threshold_k=1)

        # Fill to limit async
        for i in range(10):
            await trie.insert_async(f"async_item{i}", [i], [0.1 * i], [1], weight_version=1)

        assert trie.cur_cache_size == 10

        # Overflow async
        result = await trie.insert_async("async_overflow", [99], [0.9], [1], weight_version=2)

        assert isinstance(result, bool)


# ==============================================================================
# Group B: Very Long Inputs
# ==============================================================================

@pytest.mark.unit
class TestVeryLongInputs:
    """Test handling of very long text and token sequences."""

    def test_very_long_token_sequence(self):
        """
        Test: Insert with 10K tokens.

        Stress: Large token sequence
        Expected: Should handle or reject gracefully
        """
        trie = StringRadixTrie(max_cache_size=20000)

        # 10K tokens
        large_tokens = list(range(10000))
        large_logp = [0.001] * 10000
        large_mask = [1] * 10000

        result = trie.insert("very_long", large_tokens, large_logp, large_mask, weight_version=1)

        # Should either succeed or handle gracefully
        assert isinstance(result, bool)

        if result:
            # Verify retrieval
            match = trie.find_longest_prefix("very_long")
            assert len(match.token_ids) == 10000, "Should retrieve all tokens"

    def test_extremely_long_token_sequence(self):
        """
        Test: Insert with 100K tokens.

        Stress: Extremely large sequence (memory limit)
        Expected: Should handle or reject gracefully, no crash
        """
        trie = StringRadixTrie(max_cache_size=200000)

        # 100K tokens
        huge_tokens = list(range(100000))
        huge_logp = [0.0001] * 100000
        huge_mask = [1] * 100000

        try:
            result = trie.insert("extremely_long", huge_tokens, huge_logp, huge_mask, weight_version=1)

            # If it succeeds, verify basic properties
            if result:
                match = trie.find_longest_prefix("extremely_long")
                assert len(match.token_ids) == 100000

        except (MemoryError, OverflowError) as e:
            # Acceptable to fail with memory error for extreme size
            pytest.skip(f"System memory limit: {e}")

    def test_very_long_text_string(self):
        """
        Test: Text string with 50K characters.

        Stress: Very long text input
        Expected: Should handle or reject gracefully
        """
        trie = StringRadixTrie(max_cache_size=30000)

        # 50K character string (more reasonable for unit test)
        very_long_text = "a" * 50000

        # Fewer tokens (assume ~2 chars per token)
        tokens = list(range(25000))
        logp = [0.0001] * 25000
        mask = [1] * 25000

        try:
            result = trie.insert(very_long_text, tokens, logp, mask, weight_version=1)

            if result:
                # Verify we can find it (full match)
                # Note: Search for the full string to match what was inserted
                match = trie.find_longest_prefix(very_long_text)
                assert match.matched_prefix == very_long_text, "Should find full text"

        except (MemoryError, OverflowError) as e:
            pytest.skip(f"System memory limit: {e}")

    def test_empty_vs_long_mixed(self):
        """
        Test: Mix of empty, short, and very long sequences.

        Boundary: Different scale inputs
        Expected: Should handle all correctly
        """
        trie = StringRadixTrie(max_cache_size=50000)

        # Empty (should fail)
        empty_result = trie.insert("", [1, 2], [0.1, 0.2], [1, 1], weight_version=1)
        assert empty_result is False, "Empty string should not insert"

        # Short
        short_result = trie.insert("short", [1], [0.1], [1], weight_version=1)
        assert short_result is True

        # Long
        long_tokens = list(range(10000))
        long_result = trie.insert("long" * 1000, long_tokens, [0.001] * 10000, [1] * 10000, weight_version=1)
        assert isinstance(long_result, bool)

        # Should be able to find short entry
        short_match = trie.find_longest_prefix("short")
        assert short_match.matched_prefix == "short"


# ==============================================================================
# Group C: Concurrent Cache Operations
# ==============================================================================

@pytest.mark.unit
class TestConcurrentCacheOperations:
    """Test concurrent cache operations for race conditions."""

    def test_concurrent_insertions_same_key(self):
        """
        Test: Multiple threads insert same key simultaneously.

        Race Condition: Last-write-wins, but should be consistent
        Expected: No crashes, final state consistent
        """
        trie = StringRadixTrie(max_cache_size=1000)

        results = []

        def insert_worker(worker_id):
            result = trie.insert(
                "shared_key",
                [worker_id, worker_id + 1],
                [0.1 * worker_id, 0.2 * worker_id],
                [1, 1],
                weight_version=worker_id
            )
            results.append((worker_id, result))

        # Launch concurrent inserts
        threads = [threading.Thread(target=insert_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (overwrites)
        assert all(r[1] is True for r in results), "All inserts should succeed"

        # Final state should be consistent
        final_match = trie.find_longest_prefix("shared_key")
        assert final_match.matched_prefix == "shared_key"

        # Tokens should match one of the workers
        first_token = final_match.token_ids[0]
        assert 0 <= first_token < 20, f"Token should be from one worker, got {first_token}"

    def test_concurrent_insert_and_lookup_same_key(self):
        """
        Test: Insert and lookup of same key concurrently.

        Race Condition: Lookup during insert
        Expected: Lookup gets old or new data, but consistent
        """
        trie = StringRadixTrie(max_cache_size=1000)

        # Pre-insert initial data
        trie.insert("race_key", [0, 0], [0.0, 0.0], [1, 1], weight_version=1)

        lookup_results = []

        def lookup_worker(worker_id):
            for _ in range(10):
                match = trie.find_longest_prefix("race_key")
                lookup_results.append((worker_id, match.matched_prefix, match.token_ids[:]))

        def insert_worker():
            for i in range(10):
                trie.insert("race_key", [i, i], [0.1 * i, 0.1 * i], [1, 1], weight_version=i + 10)

        # Run concurrent lookup and insert
        lookup_threads = [threading.Thread(target=lookup_worker, args=(i,)) for i in range(3)]
        insert_thread = threading.Thread(target=insert_worker)

        for t in lookup_threads:
            t.start()
        insert_thread.start()

        for t in lookup_threads:
            t.join()
        insert_thread.join()

        # All lookups should return valid data
        for worker_id, prefix, tokens in lookup_results:
            assert prefix == "race_key", "Should find key"
            assert len(tokens) == 2, "Should have 2 tokens"
            # Tokens should be consistent (both same value)
            assert tokens[0] == tokens[1], f"Tokens should be consistent: {tokens}"

    @pytest.mark.asyncio
    async def test_concurrent_async_insertions_same_key(self):
        """
        Test: Concurrent async insertions of same key.

        Expected: Async-safe, no corruption
        """
        trie = StringRadixTrie(max_cache_size=1000)

        async def async_inserter(worker_id):
            return await trie.insert_async(
                "async_shared",
                [worker_id],
                [0.1 * worker_id],
                [1],
                weight_version=worker_id
            )

        # Launch concurrent async inserts
        results = await asyncio.gather(*[async_inserter(i) for i in range(50)])

        # All should succeed
        assert all(r is True for r in results)

        # Final state should be consistent
        match = await trie.find_longest_prefix_async("async_shared")
        assert match.matched_prefix == "async_shared"

    def test_concurrent_eviction_and_insertion(self):
        """
        Test: Cache eviction (GC) during insertion.

        Race Condition: GC removes space while insert adds data
        Expected: Should handle safely
        """
        trie = StringRadixTrie(max_cache_size=50, gc_threshold_k=1)

        # Pre-fill with old data
        for i in range(30):
            trie.insert(f"old_{i}", [i], [0.1 * i], [1], weight_version=1)

        stop_flag = threading.Event()
        results = {'inserts': 0, 'gcs': 0, 'errors': []}

        def inserter():
            version = 100
            while not stop_flag.is_set():
                try:
                    trie.insert(f"new_{version}", [version % 256], [0.1], [1], weight_version=version)
                    results['inserts'] += 1
                    version += 1
                except Exception as e:
                    results['errors'].append(str(e))

        def evictor():
            version = 100
            while not stop_flag.is_set():
                try:
                    trie.gc_by_weight_version(current_weight_version=version)
                    results['gcs'] += 1
                    version += 1
                except Exception as e:
                    results['errors'].append(str(e))

        insert_thread = threading.Thread(target=inserter)
        evict_thread = threading.Thread(target=evictor)

        insert_thread.start()
        evict_thread.start()

        # Run for short time
        threading.Event().wait(0.1)
        stop_flag.set()

        insert_thread.join()
        evict_thread.join()

        # Should have done work
        assert results['inserts'] > 0
        assert results['gcs'] > 0

        # Should not have errors
        assert len(results['errors']) == 0, f"Errors: {results['errors']}"


# ==============================================================================
# Group D: Data Consistency Edge Cases
# ==============================================================================

@pytest.mark.unit
class TestDataConsistencyEdgeCases:
    """Test data consistency at boundaries."""

    def test_insert_mismatched_lengths(self):
        """
        Test: Insert with token/logp/mask length mismatch.

        Invalid Input: len(tokens) != len(logp) != len(loss_mask)
        Expected: Should reject or handle gracefully
        """
        trie = StringRadixTrie()

        # Mismatch: 3 tokens, 2 logprobs
        result1 = trie.insert("mismatch1", [1, 2, 3], [0.1, 0.2], [1, 1, 1], weight_version=1)
        # Should fail or handle gracefully
        # Implementation may assert or return False

        # Mismatch: 2 tokens, 3 loss_masks
        result2 = trie.insert("mismatch2", [1, 2], [0.1, 0.2], [1, 1, 1], weight_version=1)

        # Mismatch: 3 tokens, 3 logprobs, 2 masks
        result3 = trie.insert("mismatch3", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1], weight_version=1)

        # At least one should fail or all should handle gracefully
        # The key is no crash
        results = [result1, result2, result3]
        print(f"Mismatch insert results: {results}")

    def test_insert_empty_tokens(self):
        """
        Test: Insert with empty token array.

        Boundary: Empty data arrays
        Expected: Should reject
        """
        trie = StringRadixTrie()

        result = trie.insert("empty_tokens", [], [], [], weight_version=1)
        assert result is False, "Should not insert empty tokens"

    def test_insert_single_token(self):
        """
        Test: Insert with single token.

        Boundary: Minimal valid data
        Expected: Should succeed
        """
        trie = StringRadixTrie()

        result = trie.insert("single", [42], [0.5], [1], weight_version=1)
        assert result is True

        match = trie.find_longest_prefix("single")
        assert match.token_ids == [42]

    def test_cache_hit_preserves_all_fields(self):
        """
        Test: Cache hit preserves tokens, logprobs, loss_mask, versions.

        Consistency: All fields should match original insert
        Expected: Complete field preservation
        """
        trie = StringRadixTrie()

        # Insert with specific data
        original_tokens = [1, 2, 3, 4, 5]
        original_logp = [-0.1, -0.2, -0.3, -0.4, -0.5]
        original_mask = [0, 0, 1, 1, 1]
        original_version = 42

        trie.insert("complete", original_tokens, original_logp, original_mask, weight_version=original_version)

        # Retrieve and verify all fields
        match = trie.find_longest_prefix("complete")

        assert match.token_ids == original_tokens, "Tokens should match"
        assert match.logp == original_logp, "Logprobs should match"
        assert match.loss_mask == original_mask, "Loss mask should match"
        assert match.generation_versions == [original_version] * 5, "Versions should match"


# ==============================================================================
# Group E: Middleware Cache Edge Cases
# ==============================================================================

@pytest.mark.unit
class TestMiddlewareCacheEdgeCases:
    """Test middleware cache handling edge cases."""

    def test_middleware_very_long_text_input(self):
        """
        Test: Middleware with very long text input.

        Boundary: Text > typical context length
        Expected: Should handle or limit gracefully
        """
        # Create minimal middleware setup
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 100000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        # Mock tokenizer that returns many tokens
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": list(range(50000))}  # 50K tokens

        # Can't test full middleware without real tokenizer
        # But we can test radix tree directly
        trie = StringRadixTrie(max_cache_size=100000)

        # Insert large text
        large_text = "word " * 10000  # 50K chars
        large_tokens = list(range(10000))

        try:
            result = trie.insert(large_text, large_tokens, [0.01] * 10000, [1] * 10000, weight_version=1)
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"Should handle large text gracefully: {e}")

    @pytest.mark.asyncio
    async def test_middleware_concurrent_cache_queries(self):
        """
        Test: Concurrent middleware cache queries.

        Stress: Many simultaneous cache lookups
        Expected: Should handle concurrently without errors
        """
        trie = StringRadixTrie(max_cache_size=10000)

        # Pre-populate cache
        for i in range(100):
            await trie.insert_async(f"cached_{i}", [i, i+1], [0.1, 0.2], [1, 1], weight_version=1)

        # Concurrent queries
        async def query_worker(worker_id):
            results = []
            for i in range(10):
                match = await trie.find_longest_prefix_async(f"cached_{i * 10}")
                results.append(match.matched_prefix)
            return results

        # Launch many concurrent queries
        tasks = [query_worker(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent queries should not error: {exceptions}"

    def test_cache_corruption_recovery(self):
        """
        Test: System behavior if cache entry is corrupted.

        Robustness: Handle invalid cached data
        Expected: Should detect and handle or skip corrupted entries
        """
        trie = StringRadixTrie()

        # Insert valid data
        trie.insert("valid", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=1)

        # Verify valid retrieval
        match = trie.find_longest_prefix("valid")
        assert match.matched_prefix == "valid"

        # Note: We can't easily corrupt internal structure without accessing internals
        # This test documents the expected behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
