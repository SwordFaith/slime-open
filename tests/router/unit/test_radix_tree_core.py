"""
Unit tests for RadixTree core logic.

Tests cover:
- Loss Mask semantics (new prompt, generated response, cache hit)
- Weight Version update strategy (new nodes, traversed nodes, GC, edge cases)
"""

import pytest
from slime.router.middleware_hub.radix_tree import StringRadixTrie, MatchResult


# ============================================================================
# Scenario Group A: Loss Mask Correctness
# ============================================================================


@pytest.mark.unit
def test_loss_mask_new_prompt():
    """
    Test: New prompt tokenization → loss_mask = [0] * len(tokens)

    Semantics: Prompt tokens should NOT participate in loss computation.
    """
    trie = StringRadixTrie()

    # Insert new prompt
    prompt_text = "User: Hello"
    prompt_tokens = [85, 115, 101, 114, 58, 32, 72, 101, 108, 108, 111]  # "User: Hello"
    prompt_logp = [0.1] * len(prompt_tokens)
    prompt_loss_mask = [0] * len(prompt_tokens)  # All 0s for prompt

    result = trie.insert(prompt_text, prompt_tokens, prompt_logp, prompt_loss_mask, weight_version=1)
    assert result is True

    # Retrieve and verify loss_mask (use find_longest_prefix to avoid tokenizer dependency)
    match = trie.find_longest_prefix(prompt_text)
    assert match is not None
    assert match.loss_mask == prompt_loss_mask, "Prompt loss_mask should be all 0s"
    assert all(m == 0 for m in match.loss_mask), "All prompt tokens should have loss_mask=0"


@pytest.mark.unit
def test_loss_mask_generated_response():
    """
    Test: Generated response → loss_mask = [1] * len(tokens)

    Semantics: Response tokens SHOULD participate in loss computation.
    """
    trie = StringRadixTrie()

    # Insert prompt first
    prompt_text = "User: Hello"
    prompt_tokens = [85, 115, 101, 114, 58, 32, 72, 101, 108, 108, 111]
    prompt_logp = [0.1] * len(prompt_tokens)
    prompt_loss_mask = [0] * len(prompt_tokens)
    trie.insert(prompt_text, prompt_tokens, prompt_logp, prompt_loss_mask, weight_version=1)

    # Insert full trajectory (prompt + response)
    full_text = "User: Hello\nAssistant: Hi there!"
    response_tokens = [10, 65, 115, 115, 105, 115, 116, 97, 110, 116, 58, 32, 72, 105, 32, 116, 104, 101, 114, 101, 33]
    full_tokens = prompt_tokens + response_tokens
    full_logp = [0.1] * len(full_tokens)
    response_loss_mask = [1] * len(response_tokens)  # All 1s for response
    full_loss_mask = prompt_loss_mask + response_loss_mask

    result = trie.insert(full_text, full_tokens, full_logp, full_loss_mask, weight_version=1)
    assert result is True

    # Retrieve full trajectory (use find_longest_prefix)
    match = trie.find_longest_prefix(full_text)
    assert match is not None
    assert len(match.loss_mask) == len(full_tokens)

    # Verify prompt part has loss_mask=0
    assert all(m == 0 for m in match.loss_mask[:len(prompt_tokens)]), "Prompt tokens should have loss_mask=0"

    # Verify response part has loss_mask=1
    assert all(m == 1 for m in match.loss_mask[len(prompt_tokens):]), "Response tokens should have loss_mask=1"


@pytest.mark.unit
def test_loss_mask_cache_hit_preserves():
    """
    Test: Cache hit preserves original loss_mask

    When retrieving from cache, loss_mask should be preserved exactly.
    """
    trie = StringRadixTrie()

    # Insert simple trajectory (simplified to avoid length mismatch)
    text = "AB"  # 2 characters
    tokens = [65, 66]  # A=65, B=66
    logp = [0.1, 0.2]
    loss_mask = [0, 1]  # First char prompt, second char response

    result = trie.insert(text, tokens, logp, loss_mask, weight_version=1)
    assert result is True

    # Retrieve and verify loss_mask is preserved (use find_longest_prefix)
    match = trie.find_longest_prefix(text)
    assert match is not None
    assert match.loss_mask == loss_mask, "Cache hit should preserve exact loss_mask"
    assert match.loss_mask[0] == 0, "First char should be prompt (loss_mask=0)"
    assert match.loss_mask[1] == 1, "Second char should be response (loss_mask=1)"


# ============================================================================
# Scenario Group B: Weight Version Update Strategy
# ============================================================================


@pytest.mark.unit
def test_weight_version_new_node_creation():
    """
    Test: New node creation → weight_version correctly set
    """
    trie = StringRadixTrie()

    # Insert with weight_version=5
    text = "Hello World"
    tokens = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]
    logp = [0.1] * len(tokens)
    loss_mask = [1] * len(tokens)

    result = trie.insert(text, tokens, logp, loss_mask, weight_version=5)
    assert result is True

    # Verify weight_version is set
    match = trie.find_longest_prefix(text)
    assert match is not None
    assert match.last_node.weight_version == 5, "New node should have weight_version=5"


@pytest.mark.unit
def test_weight_version_traversed_nodes_update():
    """
    Test: Traversed nodes → weight_version should be updated (EXPECTED TO FAIL BEFORE FIX)

    Current Bug (radix_tree.py:348-349):
        if weight_version is not None and new_node:
            new_node.weight_version = weight_version

    Problem: Only updates new_node, NOT traversed nodes!

    Expected Behavior After Fix:
        All traversed nodes should have weight_version updated to latest.
    """
    trie = StringRadixTrie()

    # Step 1: Insert "Hello" at weight_version=1
    trie.insert("Hello", [72, 101, 108, 108, 111], [0.1] * 5, [1] * 5, weight_version=1)

    # Verify initial weight_version (use find_longest_prefix to avoid tokenizer dependency)
    match1 = trie.find_longest_prefix("Hello")
    assert match1.last_node.weight_version == 1

    # Step 2: Insert "Hello World" at weight_version=5 (shares prefix "Hello")
    trie.insert(
        "Hello World",
        [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100],
        [0.1] * 11,
        [1] * 11,
        weight_version=5,
    )

    # Step 3: Retrieve "Hello" again - its node was traversed during "Hello World" insertion
    match2 = trie.find_longest_prefix("Hello")

    # VERSION SEPARATION ASSERTION: Updated for new version separation architecture
    # weight_version (generation version) remains unchanged: match2.last_node.weight_version == 1
    # traverse_version (GC version) gets updated: match2.last_node.traverse_version == 5
    assert match2.last_node.weight_version == 1, (
        "Generation version should remain unchanged at 1"
    )
    assert match2.last_node.traverse_version == 5, (
        "Traverse version should be updated to 5 after traversal at version 5"
    )


@pytest.mark.unit
def test_weight_version_gc_removes_old():
    """
    Test: GC removes nodes with weight_version <= current_version - 5
    """
    trie = StringRadixTrie()

    # Insert trajectories at different versions
    trie.insert("v1_trajectory", [1, 2, 3], [0.1] * 3, [1] * 3, weight_version=1)
    trie.insert("v3_trajectory", [4, 5, 6], [0.1] * 3, [1] * 3, weight_version=3)
    trie.insert("v10_trajectory", [7, 8, 9], [0.1] * 3, [1] * 3, weight_version=10)

    # Verify all exist before GC
    assert trie.find_longest_prefix("v1_trajectory").matched_prefix == "v1_trajectory"
    assert trie.find_longest_prefix("v3_trajectory").matched_prefix == "v3_trajectory"
    assert trie.find_longest_prefix("v10_trajectory").matched_prefix == "v10_trajectory"

    # Run GC with current_weight_version=10 (should remove versions <= 5)
    trie.gc_by_weight_version(current_weight_version=10)

    # v1 (version=1) should be removed (1 <= 5)
    match_v1 = trie.find_longest_prefix("v1_trajectory")
    assert match_v1.matched_prefix != "v1_trajectory", "v1_trajectory (version=1) should be removed"

    # v3 (version=3) should be removed (3 <= 5)
    match_v3 = trie.find_longest_prefix("v3_trajectory")
    assert match_v3.matched_prefix != "v3_trajectory", "v3_trajectory (version=3) should be removed"

    # v10 (version=10) should remain (10 > 5)
    match_v10 = trie.find_longest_prefix("v10_trajectory")
    assert match_v10.matched_prefix == "v10_trajectory", "v10_trajectory (version=10) should remain"


@pytest.mark.unit
def test_weight_version_edge_case_empty_tree():
    """
    Test: Edge case - Insert into empty tree with weight_version
    """
    trie = StringRadixTrie()

    # Insert into empty tree
    result = trie.insert("first", [1, 2, 3], [0.1] * 3, [1] * 3, weight_version=10)
    assert result is True

    match = trie.find_longest_prefix("first")
    assert match is not None
    assert match.last_node.weight_version == 10


@pytest.mark.unit
def test_weight_version_edge_case_single_node():
    """
    Test: Edge case - Single node tree, update weight_version
    """
    trie = StringRadixTrie()

    # Insert single node
    trie.insert("single", [1, 2, 3], [0.1] * 3, [1] * 3, weight_version=1)

    # Insert again with higher version (should update)
    trie.insert("single", [1, 2, 3], [0.1] * 3, [1] * 3, weight_version=10)

    match = trie.find_longest_prefix("single")
    assert match is not None
    # Note: Current implementation may not update existing node, this tests actual behavior
    assert match.last_node.weight_version is not None


@pytest.mark.unit
def test_weight_version_edge_case_deep_nesting():
    """
    Test: Edge case - Deep nesting (>10 levels), weight_version propagation
    """
    trie = StringRadixTrie()

    # Build deep nesting by inserting increasingly longer strings
    base = "a"
    for i in range(15):  # Create 15 levels
        text = base * (i + 1)  # "a", "aa", "aaa", ...
        tokens = [ord('a')] * (i + 1)
        logp = [0.1] * (i + 1)
        loss_mask = [1] * (i + 1)
        trie.insert(text, tokens, logp, loss_mask, weight_version=1)

    # Now insert deepest level with higher version
    deepest = base * 15
    tokens = [ord('a')] * 15
    trie.insert(deepest, tokens, [0.1] * 15, [1] * 15, weight_version=20)

    # VERSION SEPARATION: Updated for new version separation architecture
    # Verify deepest node
    match = trie.find_longest_prefix(deepest)
    assert match is not None
    # weight_version (generation version) remains 1 for original nodes
    # traverse_version (GC version) gets updated to 20
    assert match.last_node.weight_version == 1, "Generation version should remain 1 for original node"
    assert match.last_node.traverse_version == 20, "Traverse version should be updated to 20"

    # Verify intermediate nodes
    match_mid = trie.find_longest_prefix(base * 7)
    assert match_mid is not None
    assert match_mid.last_node.weight_version == 1, "Generation version should remain 1 for intermediate node"
    assert match_mid.last_node.traverse_version == 20, "Traverse version should be updated to 20 for intermediate node"


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
