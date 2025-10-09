"""
TDD Tests for Radix Tree Weight Version Separation.

This test suite follows Test-Driven Development methodology:
1. Write failing tests first (Phase 1)
2. Implement functionality to make tests pass (Phase 2)
3. Refactor and optimize (Phase 3)

Test Coverage:
- StringTreeNode version separation (weight_version vs traverse_version)
- Version alignment with token_ids during retrieval
- GC based on traverse_version
- Backward compatibility
"""

import pytest
from slime.router.middleware_hub.radix_tree import StringRadixTrie, MatchResult


# ============================================================================
# Scenario Group A: Version Separation Core Functionality
# ============================================================================

@pytest.mark.unit
def test_stringTreeNode_initialization_with_separated_versions():
    """
    TDD Phase 1: Test StringTreeNode initialization with separated version fields.

    This test will FAIL until we implement traverse_version field.
    """
    from slime.router.middleware_hub.radix_tree import StringTreeNode

    node = StringTreeNode()

    # Check both version fields exist and are properly initialized
    assert hasattr(node, 'weight_version'), "StringTreeNode should have weight_version field"
    assert hasattr(node, 'traverse_version'), "StringTreeNode should have traverse_version field (NEW)"

    # Check initial values
    assert node.weight_version is None, "Initial weight_version should be None"
    assert node.traverse_version is None, "Initial traverse_version should be None"


@pytest.mark.unit
def test_version_separation_on_insert():
    """
    TDD Phase 1: Test version separation during insert operations.

    Expected behavior:
    - New nodes: weight_version = insert_version, traverse_version = insert_version
    - Traversed nodes: weight_version unchanged, traverse_version = insert_version

    This test will FAIL until we implement version separation logic.
    """
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


@pytest.mark.unit
def test_generation_versions_alignment_in_match_result():
    """
    TDD Phase 1: Test that MatchResult includes generation_versions aligned with token_ids.

    This test will FAIL until we add generation_versions field to MatchResult
    and implement version collection logic.
    """
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


# ============================================================================
# Scenario Group B: GC Based on Traverse Version
# ============================================================================

@pytest.mark.unit
def test_gc_based_on_traverse_version():
    """
    TDD Phase 1: Test GC uses traverse_version instead of weight_version.

    Expected behavior:
    - GC should remove nodes based on traverse_version <= threshold
    - weight_version should NOT affect GC decisions

    This test will FAIL until we update GC logic to use traverse_version.
    """
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


@pytest.mark.unit
def test_gc_preserves_generation_version_info():
    """
    TDD Phase 1: Test that GC preserves generation version information of kept nodes.

    This test will FAIL until we implement proper version separation.
    """
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
# Scenario Group C: Version Alignment in Retrieval
# ============================================================================

@pytest.mark.unit
def test_retrieve_from_text_version_alignment():
    """
    Test get_or_create_tokenization returns version-aligned tokens.

    Expected behavior:
    - Return (token_ids, logp, loss_mask, generation_versions)
    - generation_versions align with token_ids
    - Non-AI-generated tokens marked with version -1
    """
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


@pytest.mark.unit
def test_partial_match_version_alignment():
    """
    TDD Phase 1: Test version alignment in partial match scenarios.

    This test will FAIL until we implement proper version collection.
    """
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


# ============================================================================
# Scenario Group D: Backward Compatibility
# ============================================================================

@pytest.mark.unit
def test_backward_compatibility_weight_version_access():
    """
    TDD Phase 1: Test backward compatibility for existing weight_version access.

    Existing code should continue to work accessing weight_version,
    even though we now have separate traverse_version.
    """
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


@pytest.mark.unit
def test_existing_gc_interface_compatibility():
    """
    TDD Phase 1: Test existing GC interface remains compatible.

    gc_by_weight_version should work as before, but use traverse_version internally.
    """
    trie = StringRadixTrie()

    # Use existing GC interface
    trie.insert("Test", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)

    # Should not raise any errors
    removed_count = trie.gc_by_weight_version(current_weight_version=10)

    # Should return valid count
    assert isinstance(removed_count, int)
    assert removed_count >= 0


# ============================================================================
# Scenario Group E: Edge Cases
# ============================================================================

@pytest.mark.unit
def test_version_none_handling():
    """
    TDD Phase 1: Test handling of None versions in various scenarios.
    """
    trie = StringRadixTrie()

    # Insert without weight_version
    trie.insert("NoVersion", [1, 2, 3], [0.1] * 3, [0] * 3)  # weight_version=None

    result = trie.find_longest_prefix("NoVersion")

    # Both versions should be None for manually inserted data
    assert result.last_node.weight_version is None
    assert result.last_node.traverse_version is None


@pytest.mark.unit
def test_version_reset_on_gc():
    """
    TDD Phase 1: Test version handling during GC operations.
    """
    trie = StringRadixTrie()

    # Insert data
    trie.insert("Temp", [1, 2, 3], [0.1] * 3, [0] * 3, weight_version=1)

    # GC should handle nodes with None versions gracefully
    removed_count = trie.gc_by_weight_version(current_weight_version=5)

    # Should not raise errors
    assert isinstance(removed_count, int)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])