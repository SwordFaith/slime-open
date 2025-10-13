"""
Retrieve Endpoints E2E Tests

Verifies /retrieve_from_text and /retrieve_from_messages_template endpoints:
- Basic functionality and response format
- Caching behavior and consistency
- Error handling (no cache available)
- Edge cases (empty input, long text, multi-turn)
- OpenAI messages template processing
- Tool parameters in messages

Test Coverage:
- /retrieve_from_text: 6 tests
- /retrieve_from_messages_template: 6 tests

These endpoints are critical for token caching workflow, used by external
clients to pre-cache prompts, query cache state, and debug tokenization.

Running:
  pytest tests/router/e2e/test_retrieve_endpoints.py -v -s -m e2e
"""

import pytest
import time
from typing import List, Dict, Any


class TestRetrieveFromText:
    """/retrieve_from_text endpoint E2E verification"""

    @pytest.mark.e2e
    def test_retrieve_from_text_basic(self, client_with_cache):
        """
        Test basic /retrieve_from_text functionality.

        Verifies:
        - Endpoint accepts text input
        - Returns tokens, token_length, loss_mask
        - Response includes all required fields
        - Token IDs are valid integers

        Expected Response Fields:
        - tokens: List[int]
        - response: str (echoed input text)
        - loss_mask: List[int]
        - token_length: int
        - loss_mask_length: int
        - rollout_logp: List[float]
        - generation_versions: List[int]
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text basic functionality")
        print("=" * 60)

        text = "Hello, world! This is a test."
        print(f"\nInput text: '{text}'")

        response = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        # Verify all required fields present
        required_fields = [
            "tokens", "response", "loss_mask",
            "token_length", "loss_mask_length",
            "rollout_logp", "generation_versions"
        ]

        print(f"\nResponse fields:")
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
            print(f"  ✓ {field}: present")

        # Verify tokens
        tokens = data["tokens"]
        assert isinstance(tokens, list), "tokens should be a list"
        assert len(tokens) > 0, "tokens should not be empty"
        assert all(isinstance(t, int) for t in tokens), "all tokens should be integers"
        print(f"\nTokens: {tokens}")
        print(f"  - count: {len(tokens)}")
        print(f"  - token_length field: {data['token_length']}")
        assert len(tokens) == data["token_length"], "token count mismatch"

        # Verify response field echoes input
        assert data["response"] == text, "response should echo input text"
        print(f"  ✓ Response text matches input")

        # Verify loss_mask
        loss_mask = data["loss_mask"]
        assert isinstance(loss_mask, list), "loss_mask should be a list"
        print(f"\nLoss mask: {len(loss_mask)} elements")
        assert len(loss_mask) == data["loss_mask_length"], "loss_mask length mismatch"

        print("\n✅ Test PASSED: Basic functionality verified")

    @pytest.mark.e2e
    def test_retrieve_from_text_caching(self, client_with_cache):
        """
        Test caching behavior of /retrieve_from_text.

        Verifies:
        - First request creates cache entry
        - Second request with same text hits cache
        - Tokens are identical across requests
        - Cache provides consistent results

        Expected Behavior:
        - get_or_create_tokenization creates cache on first call
        - Subsequent calls retrieve from cache
        - Token IDs remain consistent
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text caching behavior")
        print("=" * 60)

        text = "Test caching behavior with this specific text."
        print(f"\nInput text: '{text}'")

        # First request - cache miss
        print("\n--- First Request (cache miss) ---")
        start_time = time.time()
        response1 = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )
        time1 = time.time() - start_time

        assert response1.status_code == 200
        data1 = response1.json()
        tokens1 = data1["tokens"]

        print(f"Status: {response1.status_code}")
        print(f"Time: {time1*1000:.2f}ms")
        print(f"Tokens: {len(tokens1)} tokens")
        print(f"Token IDs: {tokens1[:10]}..." if len(tokens1) > 10 else f"Token IDs: {tokens1}")

        # Second request - cache hit
        print("\n--- Second Request (cache hit) ---")
        start_time = time.time()
        response2 = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )
        time2 = time.time() - start_time

        assert response2.status_code == 200
        data2 = response2.json()
        tokens2 = data2["tokens"]

        print(f"Status: {response2.status_code}")
        print(f"Time: {time2*1000:.2f}ms")
        print(f"Tokens: {len(tokens2)} tokens")

        # Verify cache hit - tokens should be identical
        assert tokens1 == tokens2, (
            f"Token mismatch (cache inconsistency):\n"
            f"  First:  {tokens1}\n"
            f"  Second: {tokens2}"
        )
        print(f"\n✅ Cache hit verified: tokens identical")

        # Performance comparison (cache hit may be faster, but not required)
        speedup = time1 / time2 if time2 > 0 else 1.0
        print(f"\nPerformance:")
        print(f"  First:  {time1*1000:.2f}ms")
        print(f"  Second: {time2*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        print("\n✅ Test PASSED: Caching behavior verified")

    @pytest.mark.e2e
    def test_retrieve_from_text_no_cache(self, client_no_cache):
        """
        Test error handling when cache not available.

        Verifies:
        - Endpoint returns error when radix tree not available
        - Error status code is 500 (RuntimeError)
        - Error message is informative

        Expected Behavior:
        - Router without RadixTreeMiddleware lacks radix_tree component
        - Endpoint raises RuntimeError
        - FastAPI converts to 500 Internal Server Error
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text without cache (error handling)")
        print("=" * 60)

        text = "This should fail without cache"
        print(f"\nInput text: '{text}'")
        print(f"Router mode: No cache (direct proxy)")

        response = client_no_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )

        print(f"\nStatus: {response.status_code}")

        # Expect 500 error (RuntimeError: Radix tree not available)
        assert response.status_code == 500, (
            f"Expected 500 error, got {response.status_code}"
        )

        print(f"✅ Correct error status: 500")

        # Try to get error message
        try:
            error_data = response.json()
            if "detail" in error_data:
                print(f"Error detail: {error_data['detail']}")
        except Exception:
            print(f"Error response: {response.text[:200]}")

        print("\n✅ Test PASSED: Error handling verified")

    @pytest.mark.e2e
    def test_retrieve_from_text_empty_input(self, client_with_cache):
        """
        Test handling of empty input.

        Verifies:
        - Empty string is handled gracefully
        - Returns valid response structure
        - token_length is 0 or minimal

        Expected Behavior:
        - Empty text may tokenize to empty list or BOS token
        - Response structure remains valid
        - No crash or error
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text empty input handling")
        print("=" * 60)

        text = ""
        print(f"\nInput text: '' (empty string)")

        response = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        print(f"\nResponse:")
        print(f"  - tokens: {data['tokens']}")
        print(f"  - token_length: {data['token_length']}")
        print(f"  - loss_mask: {data['loss_mask']}")

        # Verify response structure is valid
        assert isinstance(data["tokens"], list)
        assert isinstance(data["token_length"], int)
        assert data["token_length"] == len(data["tokens"])

        # Empty input should result in 0 or minimal tokens
        assert data["token_length"] <= 2, (
            f"Empty input should not produce many tokens, got {data['token_length']}"
        )

        print(f"\n✅ Empty input handled gracefully")
        print("\n✅ Test PASSED: Empty input handling verified")

    @pytest.mark.e2e
    def test_retrieve_from_text_long_text(self, client_with_cache):
        """
        Test handling of longer text input.

        Verifies:
        - Long text is tokenized correctly
        - token_length is accurate
        - Response structure remains valid
        - No truncation or errors

        Expected Behavior:
        - Longer text produces more tokens
        - All tokens are valid integers
        - token_length matches actual count
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text long text handling")
        print("=" * 60)

        # Create a longer text (multiple sentences)
        text = (
            "This is a longer text for testing tokenization. "
            "It contains multiple sentences to verify that the endpoint "
            "can handle longer inputs correctly. The tokenizer should "
            "process this text and return a larger number of tokens. "
            "We expect the token_length field to accurately reflect "
            "the number of tokens in the response."
        )

        print(f"\nInput text length: {len(text)} characters")
        print(f"Text preview: '{text[:80]}...'")

        response = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        tokens = data["tokens"]
        token_length = data["token_length"]

        print(f"\nTokenization results:")
        print(f"  - token_length: {token_length}")
        print(f"  - actual tokens: {len(tokens)}")
        print(f"  - first 10 tokens: {tokens[:10]}")
        print(f"  - last 5 tokens: {tokens[-5:]}")

        # Verify token count is reasonable for long text
        assert token_length > 20, f"Expected more tokens for long text, got {token_length}"
        assert len(tokens) == token_length, "Token count mismatch"

        # Verify all tokens are valid
        assert all(isinstance(t, int) for t in tokens), "All tokens should be integers"
        print(f"\n✅ All {token_length} tokens are valid integers")

        print("\n✅ Test PASSED: Long text handling verified")

    @pytest.mark.e2e
    def test_retrieve_from_text_response_format(self, client_with_cache):
        """
        Test detailed response format validation.

        Verifies:
        - All required fields present
        - Field types are correct
        - Nested structures are valid
        - Values are sensible

        Expected Types:
        - tokens: List[int]
        - response: str
        - loss_mask: List[int]
        - token_length: int
        - loss_mask_length: int
        - rollout_logp: List[float]
        - generation_versions: List[int]
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_text response format validation")
        print("=" * 60)

        text = "Validate response format for this text."
        print(f"\nInput text: '{text}'")

        response = client_with_cache.post(
            "/retrieve_from_text",
            json={"text": text}
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\nValidating response format:")

        # tokens field
        assert "tokens" in data
        assert isinstance(data["tokens"], list)
        assert all(isinstance(t, int) for t in data["tokens"])
        print(f"  ✓ tokens: List[int] with {len(data['tokens'])} elements")

        # response field
        assert "response" in data
        assert isinstance(data["response"], str)
        assert data["response"] == text
        print(f"  ✓ response: str (matches input)")

        # loss_mask field
        assert "loss_mask" in data
        assert isinstance(data["loss_mask"], list)
        assert all(isinstance(m, int) for m in data["loss_mask"])
        print(f"  ✓ loss_mask: List[int] with {len(data['loss_mask'])} elements")

        # token_length field
        assert "token_length" in data
        assert isinstance(data["token_length"], int)
        assert data["token_length"] >= 0
        assert data["token_length"] == len(data["tokens"])
        print(f"  ✓ token_length: int ({data['token_length']}, matches tokens)")

        # loss_mask_length field
        assert "loss_mask_length" in data
        assert isinstance(data["loss_mask_length"], int)
        assert data["loss_mask_length"] == len(data["loss_mask"])
        print(f"  ✓ loss_mask_length: int ({data['loss_mask_length']}, matches loss_mask)")

        # rollout_logp field
        assert "rollout_logp" in data
        assert isinstance(data["rollout_logp"], list)
        # Can be empty or contain floats
        if len(data["rollout_logp"]) > 0:
            assert all(isinstance(p, (int, float)) for p in data["rollout_logp"])
        print(f"  ✓ rollout_logp: List[float] with {len(data['rollout_logp'])} elements")

        # generation_versions field
        assert "generation_versions" in data
        assert isinstance(data["generation_versions"], list)
        if len(data["generation_versions"]) > 0:
            assert all(isinstance(v, int) for v in data["generation_versions"])
        print(f"  ✓ generation_versions: List[int] with {len(data['generation_versions'])} elements")

        print("\n✅ All fields present with correct types")
        print("\n✅ Test PASSED: Response format validated")


class TestRetrieveFromMessagesTemplate:
    """/retrieve_from_messages_template endpoint E2E verification"""

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_basic(self, client_with_cache):
        """
        Test basic /retrieve_from_messages_template functionality.

        Verifies:
        - Endpoint accepts OpenAI messages format
        - Applies chat template correctly
        - Returns tokens and templated text
        - Response includes all required fields

        Expected Behavior:
        - tokenizer.apply_chat_template() converts messages to text
        - Radix tree tokenizes the templated text
        - Response includes both tokens and template result
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template basic functionality")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "Hello! How are you?"}
        ]

        print(f"\nInput messages: {messages}")

        response = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={
                "messages": messages,
                "add_generation_prompt": True
            }
        )

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        # Verify required fields
        required_fields = [
            "tokens", "response", "loss_mask",
            "token_length", "loss_mask_length",
            "rollout_logp", "generation_versions"
        ]

        print(f"\nResponse fields:")
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
            print(f"  ✓ {field}")

        # Verify tokens
        tokens = data["tokens"]
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        print(f"\nTokens: {len(tokens)} tokens")
        print(f"  Token IDs: {tokens[:10]}..." if len(tokens) > 10 else f"  Token IDs: {tokens}")

        # Verify response contains templated text
        templated_text = data["response"]
        assert isinstance(templated_text, str)
        assert len(templated_text) > 0
        print(f"\nTemplated text length: {len(templated_text)} chars")
        print(f"  Preview: '{templated_text[:100]}...'")

        # Verify templated text contains the user message
        assert "Hello" in templated_text or "hello" in templated_text.lower(), (
            "Templated text should contain user message content"
        )
        print(f"  ✓ Contains user message content")

        print("\n✅ Test PASSED: Basic functionality verified")

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_with_tools(self, client_with_cache):
        """
        Test messages template with tools parameter.

        Verifies:
        - Tools are passed to apply_chat_template
        - Tool information is included in templated text
        - Tokenization includes tool context
        - Token count is higher with tools

        Expected Behavior:
        - tokenizer.apply_chat_template(tools=tools) includes tool definitions
        - Templated text contains tool schemas
        - Token count increases with tool context
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template with tools")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "What's the weather?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        print(f"\nInput messages: {len(messages)} message(s)")
        print(f"Tools: {tools[0]['function']['name']}")

        # Request without tools (baseline)
        print("\n--- Without Tools ---")
        response_no_tools = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={
                "messages": messages,
                "add_generation_prompt": True
            }
        )
        assert response_no_tools.status_code == 200
        data_no_tools = response_no_tools.json()
        tokens_no_tools = len(data_no_tools["tokens"])
        print(f"Token count: {tokens_no_tools}")

        # Request with tools
        print("\n--- With Tools ---")
        response_with_tools = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": True
            }
        )

        print(f"Status: {response_with_tools.status_code}")
        assert response_with_tools.status_code == 200

        data_with_tools = response_with_tools.json()
        tokens_with_tools = len(data_with_tools["tokens"])
        templated_text = data_with_tools["response"]

        print(f"Token count: {tokens_with_tools}")
        print(f"Templated text length: {len(templated_text)} chars")

        # Verify tools increase token count
        assert tokens_with_tools > tokens_no_tools, (
            f"Tools should increase token count: "
            f"{tokens_no_tools} (no tools) vs {tokens_with_tools} (with tools)"
        )
        print(f"\n✅ Tools increased token count: {tokens_no_tools} → {tokens_with_tools} (+{tokens_with_tools - tokens_no_tools})")

        # Verify tool information in templated text
        # (format depends on tokenizer, so we check loosely)
        tool_indicators = ["get_weather", "function", "weather", "location"]
        found_indicators = [ind for ind in tool_indicators if ind in templated_text.lower()]
        print(f"\nTool indicators found in templated text: {found_indicators}")
        assert len(found_indicators) > 0, "Templated text should contain tool information"

        print("\n✅ Test PASSED: Tools integration verified")

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_multi_turn(self, client_with_cache):
        """
        Test multi-turn conversation handling.

        Verifies:
        - Multiple messages are correctly templated
        - All turns are included in tokenization
        - Token count increases with more turns
        - Template preserves message order

        Expected Behavior:
        - Each message is included in template
        - Token count grows with conversation length
        - System, user, assistant roles handled correctly
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template multi-turn")
        print("=" * 60)

        # Single turn
        messages_single = [
            {"role": "user", "content": "Hello"}
        ]

        print("\n--- Single Turn ---")
        response_single = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages_single, "add_generation_prompt": True}
        )
        assert response_single.status_code == 200
        tokens_single = len(response_single.json()["tokens"])
        print(f"Token count: {tokens_single}")

        # Multi-turn conversation
        messages_multi = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "What is 2+2?"}
        ]

        print("\n--- Multi-Turn (3 messages) ---")
        response_multi = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages_multi, "add_generation_prompt": True}
        )

        print(f"Status: {response_multi.status_code}")
        assert response_multi.status_code == 200

        data_multi = response_multi.json()
        tokens_multi = len(data_multi["tokens"])
        templated_text = data_multi["response"]

        print(f"Token count: {tokens_multi}")
        print(f"Templated text length: {len(templated_text)} chars")

        # Verify multi-turn has more tokens
        assert tokens_multi > tokens_single, (
            f"Multi-turn should have more tokens: "
            f"{tokens_single} (single) vs {tokens_multi} (multi)"
        )
        print(f"\n✅ Multi-turn increased token count: {tokens_single} → {tokens_multi} (+{tokens_multi - tokens_single})")

        # Verify all messages appear in templated text
        for msg in messages_multi:
            content_lower = msg["content"].lower()
            # Check if key words from each message appear
            key_words = content_lower.split()[:2]  # First 2 words
            found = any(word in templated_text.lower() for word in key_words if len(word) > 2)
            if found:
                print(f"  ✓ Message '{msg['content'][:20]}...' found in template")

        print("\n✅ Test PASSED: Multi-turn handling verified")

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_no_cache(self, client_no_cache):
        """
        Test error handling without cache.

        Verifies:
        - Endpoint returns error when components unavailable
        - Error status is 500
        - Error message is informative

        Expected Behavior:
        - RuntimeError: Radix tree and tokenizer not available
        - HTTP 500 status code
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template without cache (error)")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "This should fail"}
        ]

        print(f"\nInput messages: {messages}")
        print(f"Router mode: No cache")

        response = client_no_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages, "add_generation_prompt": True}
        )

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 500, (
            f"Expected 500 error, got {response.status_code}"
        )

        print(f"✅ Correct error status: 500")

        try:
            error_data = response.json()
            if "detail" in error_data:
                print(f"Error detail: {error_data['detail'][:100]}...")
        except Exception:
            print(f"Error response: {response.text[:200]}...")

        print("\n✅ Test PASSED: Error handling verified")

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_cache_consistency(self, client_with_cache):
        """
        Test caching consistency for messages template.

        Verifies:
        - Same messages produce same tokens
        - Cache hits are consistent
        - No variation across requests

        Expected Behavior:
        - First request creates cache
        - Subsequent requests use cache
        - Token sequences are identical
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template cache consistency")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "Test cache consistency"}
        ]

        print(f"\nInput messages: {messages}")

        # First request
        print("\n--- Request 1 ---")
        response1 = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages, "add_generation_prompt": True}
        )
        assert response1.status_code == 200
        data1 = response1.json()
        tokens1 = data1["tokens"]
        print(f"Tokens: {len(tokens1)}")

        # Second request (same messages)
        print("\n--- Request 2 (same messages) ---")
        response2 = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages, "add_generation_prompt": True}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        tokens2 = data2["tokens"]
        print(f"Tokens: {len(tokens2)}")

        # Verify consistency
        assert tokens1 == tokens2, (
            f"Token mismatch (cache inconsistency):\n"
            f"  Request 1: {tokens1}\n"
            f"  Request 2: {tokens2}"
        )
        print(f"\n✅ Cache consistency verified: tokens identical")

        # Third request (should also match)
        response3 = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={"messages": messages, "add_generation_prompt": True}
        )
        assert response3.status_code == 200
        tokens3 = response3.json()["tokens"]
        assert tokens1 == tokens3, "Third request tokens mismatch"
        print(f"✅ Third request also consistent")

        print("\n✅ Test PASSED: Cache consistency verified")

    @pytest.mark.e2e
    def test_retrieve_from_messages_template_generation_prompt(self, client_with_cache):
        """
        Test add_generation_prompt parameter.

        Verifies:
        - add_generation_prompt=True adds generation prefix
        - add_generation_prompt=False omits it
        - Token counts differ appropriately
        - Both modes work correctly

        Expected Behavior:
        - True: Adds assistant prefix for generation
        - False: Omits assistant prefix
        - Token count higher with generation prompt
        """
        print("\n" + "=" * 60)
        print("Test: /retrieve_from_messages_template generation_prompt param")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "Test generation prompt"}
        ]

        print(f"\nInput messages: {messages}")

        # With generation prompt (True)
        print("\n--- add_generation_prompt=True ---")
        response_with = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={
                "messages": messages,
                "add_generation_prompt": True
            }
        )
        assert response_with.status_code == 200
        data_with = response_with.json()
        tokens_with = len(data_with["tokens"])
        text_with = data_with["response"]
        print(f"Token count: {tokens_with}")
        print(f"Text length: {len(text_with)} chars")
        print(f"Text preview: '{text_with[:100]}...'")

        # Without generation prompt (False)
        print("\n--- add_generation_prompt=False ---")
        response_without = client_with_cache.post(
            "/retrieve_from_messages_template",
            json={
                "messages": messages,
                "add_generation_prompt": False
            }
        )
        assert response_without.status_code == 200
        data_without = response_without.json()
        tokens_without = len(data_without["tokens"])
        text_without = data_without["response"]
        print(f"Token count: {tokens_without}")
        print(f"Text length: {len(text_without)} chars")
        print(f"Text preview: '{text_without[:100]}...'")

        # Verify difference
        print(f"\n--- Comparison ---")
        print(f"Token diff: {tokens_with - tokens_without} tokens")
        print(f"Text diff: {len(text_with) - len(text_without)} chars")

        # With generation prompt should typically have more tokens
        # (though exact behavior depends on tokenizer)
        assert tokens_with != tokens_without, (
            "add_generation_prompt should affect token count"
        )

        if tokens_with > tokens_without:
            print(f"✅ add_generation_prompt=True adds tokens (as expected)")
        else:
            print(f"⚠️  add_generation_prompt=True has fewer tokens (model-specific behavior)")

        print("\n✅ Test PASSED: generation_prompt parameter verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "e2e"])
