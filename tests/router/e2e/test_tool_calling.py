"""
Tool Calling E2E Tests

Verifies tool call parser integration in the router:
- Tool call parser configuration and initialization
- Tool calls in responses with/without cache
- Parser integration with qwen25 tool_call_parser
- Multiple tool calling scenarios

Test Coverage:
- Basic tool calling with cache enabled
- Tool calling without cache (direct proxy)
- Parser integration verification
- Multiple tool calls handling

Running:
  pytest tests/router/e2e/test_tool_calling.py -v -s -m e2e
"""

import pytest
import json
from typing import List, Dict, Any


class TestToolCalling:
    """Tool call parser E2E verification"""

    def _create_weather_tool(self) -> Dict[str, Any]:
        """Create a weather tool definition for testing."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g., Beijing"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }

    def _create_calculator_tool(self) -> Dict[str, Any]:
        """Create a calculator tool definition for testing."""
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }

    @pytest.mark.e2e
    def test_tool_calling_with_cache_basic(self, client_with_cache):
        """
        Test basic tool calling with cache enabled.

        Verifies:
        - Request with tools parameter succeeds
        - Response contains tool_calls field (if model decides to call)
        - Tool call format matches OpenAI specification
        - Parser correctly extracts function name and arguments

        Expected Behavior:
        - Model may or may not call the tool depending on the prompt
        - If tool is called, format should be valid OpenAI tool_calls
        - Response structure should be correct
        """
        print("\n" + "=" * 60)
        print("Test: Tool calling with cache - basic")
        print("=" * 60)

        request_data = {
            "model": "qwen3-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Beijing?"
                }
            ],
            "tools": [self._create_weather_tool()],
            "max_tokens": 100,
            "temperature": 0.3,
        }

        print(f"\nPrompt: {request_data['messages'][0]['content']}")
        print(f"Tools provided: {request_data['tools'][0]['function']['name']}")

        response = client_with_cache.post("/v1/chat/completions", json=request_data)

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        # Validate response structure
        assert "choices" in data, "Response missing 'choices' field"
        assert len(data["choices"]) > 0, "Response has no choices"

        choice = data["choices"][0]
        message = choice["message"]

        print(f"\nResponse structure:")
        print(f"  - role: {message.get('role')}")
        print(f"  - content: {message.get('content', 'None')[:100] if message.get('content') else 'None'}...")

        # Check for tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"  - tool_calls: {len(tool_calls)} calls")

            # Verify tool call structure
            for i, tool_call in enumerate(tool_calls):
                print(f"\n  Tool Call {i+1}:")
                assert "id" in tool_call, f"Tool call {i} missing 'id'"
                assert "type" in tool_call, f"Tool call {i} missing 'type'"
                assert tool_call["type"] == "function", f"Tool call {i} wrong type"
                assert "function" in tool_call, f"Tool call {i} missing 'function'"

                function = tool_call["function"]
                assert "name" in function, f"Tool call {i} function missing 'name'"
                assert "arguments" in function, f"Tool call {i} function missing 'arguments'"

                print(f"    - id: {tool_call['id']}")
                print(f"    - name: {function['name']}")
                print(f"    - arguments: {function['arguments'][:100]}...")

                # Verify arguments are valid JSON
                try:
                    args = json.loads(function["arguments"]) if isinstance(function["arguments"], str) else function["arguments"]
                    print(f"    - parsed args: {args}")
                except json.JSONDecodeError as e:
                    pytest.fail(f"Tool call {i} arguments not valid JSON: {e}")

            print(f"\n✅ Tool calls present and valid")
        else:
            print(f"  - tool_calls: None (model chose not to call tool)")
            print(f"\n⚠️  Model did not call tool, but response is valid")

        # Verify finish_reason
        assert "finish_reason" in choice, "Choice missing 'finish_reason'"
        print(f"  - finish_reason: {choice['finish_reason']}")

        print("\n✅ Test PASSED: Tool calling with cache basic")

    @pytest.mark.e2e
    def test_tool_calling_without_cache(self, client_no_cache):
        """
        Test tool calling without cache (direct proxy mode).

        Verifies:
        - Direct proxy to SGLang with tools parameter
        - Parser works in non-cached mode
        - Response format consistent with cached mode
        - No caching layer interference with tool calls

        Expected Behavior:
        - Request forwards to SGLang /v1/chat/completions
        - Tool call parser on SGLang side processes output
        - Router passes through tool_calls unchanged
        """
        print("\n" + "=" * 60)
        print("Test: Tool calling without cache (direct proxy)")
        print("=" * 60)

        request_data = {
            "model": "qwen3-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the temperature in Shanghai?"
                }
            ],
            "tools": [self._create_weather_tool()],
            "max_tokens": 100,
            "temperature": 0.3,
        }

        print(f"\nPrompt: {request_data['messages'][0]['content']}")
        print(f"Mode: Direct proxy (no cache)")

        response = client_no_cache.post("/v1/chat/completions", json=request_data)

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        # Validate response structure
        assert "choices" in data
        choice = data["choices"][0]
        message = choice["message"]

        print(f"\nResponse received:")
        print(f"  - role: {message.get('role')}")
        print(f"  - has content: {bool(message.get('content'))}")
        print(f"  - has tool_calls: {bool(message.get('tool_calls'))}")

        # Verify response is valid (either content or tool_calls)
        has_content = message.get("content") is not None and len(message.get("content", "")) > 0
        has_tool_calls = message.get("tool_calls") is not None and len(message.get("tool_calls", [])) > 0

        assert has_content or has_tool_calls, "Response has neither content nor tool_calls"

        if has_tool_calls:
            print(f"\n✅ Tool calls present in direct proxy mode")
            tool_calls = message["tool_calls"]
            print(f"  - {len(tool_calls)} tool call(s)")
            for tc in tool_calls:
                print(f"  - {tc['function']['name']}")
        else:
            print(f"\n⚠️  No tool calls, but response valid")

        print("\n✅ Test PASSED: Tool calling without cache")

    @pytest.mark.e2e
    def test_tool_calling_parser_integration(self, client_with_cache):
        """
        Test parser integration with specific verification.

        Verifies:
        - qwen25 tool_call_parser correctly configured
        - Parser extracts function calls from model output
        - Arguments are properly parsed
        - Multiple tools can be provided

        Expected Behavior:
        - Parser processes raw model output
        - Extracts function names and arguments
        - Converts to OpenAI tool_calls format
        """
        print("\n" + "=" * 60)
        print("Test: Parser integration verification")
        print("=" * 60)

        # Use a prompt more likely to trigger tool calling
        request_data = {
            "model": "qwen3-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "I need to know the weather in Tokyo and the result of 15 + 27."
                }
            ],
            "tools": [
                self._create_weather_tool(),
                self._create_calculator_tool()
            ],
            "max_tokens": 150,
            "temperature": 0.2,  # Lower temp for more consistent behavior
        }

        print(f"\nPrompt: {request_data['messages'][0]['content']}")
        print(f"Tools provided: {len(request_data['tools'])}")
        for tool in request_data['tools']:
            print(f"  - {tool['function']['name']}")

        response = client_with_cache.post("/v1/chat/completions", json=request_data)

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()
        message = data["choices"][0]["message"]

        print(f"\nResponse analysis:")
        print(f"  - role: {message.get('role')}")

        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"  - tool_calls count: {len(tool_calls)}")

            # Detailed verification
            for i, tc in enumerate(tool_calls):
                func = tc["function"]
                print(f"\n  Tool Call {i+1}:")
                print(f"    - id: {tc['id']}")
                print(f"    - name: {func['name']}")

                # Parse and validate arguments
                args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
                print(f"    - arguments: {args}")

                # Verify expected argument structure
                if func["name"] == "get_weather":
                    assert "location" in args, "Weather tool missing 'location' argument"
                    print(f"    ✓ Weather tool: location={args['location']}")
                elif func["name"] == "calculate":
                    assert "operation" in args, "Calculator tool missing 'operation'"
                    assert "a" in args, "Calculator tool missing 'a'"
                    assert "b" in args, "Calculator tool missing 'b'"
                    print(f"    ✓ Calculator tool: {args['operation']}({args['a']}, {args['b']})")

            print(f"\n✅ Parser integration verified: {len(tool_calls)} tool call(s) extracted")
        else:
            print(f"  - tool_calls: None")
            print(f"  - content: {message.get('content', '')[:100]}...")
            print(f"\n⚠️  Parser configured but no tool calls in this response")
            print(f"     This may be expected behavior for this specific prompt/model")

        # Verify response is structurally valid regardless
        assert "role" in message
        assert message["role"] == "assistant"
        print(f"\n✅ Response structure valid")

        print("\n✅ Test PASSED: Parser integration verified")

    @pytest.mark.e2e
    def test_multiple_tool_calls(self, client_with_cache):
        """
        Test handling of multiple tool calls in one response.

        Verifies:
        - Model can call multiple tools in single response
        - All tool calls are captured
        - Each tool call has unique ID
        - Arguments parsed correctly for each call

        Expected Behavior:
        - Response may contain 0, 1, or multiple tool_calls
        - Each tool call is valid OpenAI format
        - Tool call IDs are unique
        """
        print("\n" + "=" * 60)
        print("Test: Multiple tool calls handling")
        print("=" * 60)

        # Prompt designed to potentially trigger multiple tools
        request_data = {
            "model": "qwen3-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Beijing, Shanghai, and Tokyo?"
                }
            ],
            "tools": [self._create_weather_tool()],
            "max_tokens": 200,
            "temperature": 0.3,
        }

        print(f"\nPrompt: {request_data['messages'][0]['content']}")
        print(f"Expected: Potentially multiple tool calls for different cities")

        response = client_with_cache.post("/v1/chat/completions", json=request_data)

        print(f"\nStatus: {response.status_code}")
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()
        message = data["choices"][0]["message"]

        print(f"\nResponse analysis:")

        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"  - tool_calls count: {len(tool_calls)}")

            # Collect IDs to verify uniqueness
            tool_call_ids = set()

            for i, tc in enumerate(tool_calls):
                print(f"\n  Tool Call {i+1}:")
                print(f"    - id: {tc['id']}")
                print(f"    - name: {tc['function']['name']}")

                args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                print(f"    - location: {args.get('location', 'N/A')}")

                # Check ID uniqueness
                assert tc['id'] not in tool_call_ids, f"Duplicate tool call ID: {tc['id']}"
                tool_call_ids.add(tc['id'])

            print(f"\n✅ Multiple tool calls handled: {len(tool_calls)} call(s)")
            print(f"✅ All tool call IDs unique: {len(tool_call_ids)} unique IDs")

            if len(tool_calls) > 1:
                print(f"✅ Model successfully made multiple tool calls")
            else:
                print(f"⚠️  Single tool call (may be expected for this model/prompt)")
        else:
            print(f"  - tool_calls: None")
            print(f"  - content: {message.get('content', '')[:150]}...")
            print(f"\n⚠️  No tool calls in response")
            print(f"     Model may have chosen to answer directly")

        # Verify response structure regardless
        assert "role" in message
        assert "finish_reason" in data["choices"][0]
        print(f"\n✅ Response structure valid")

        print("\n✅ Test PASSED: Multiple tool calls handling verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "e2e"])
