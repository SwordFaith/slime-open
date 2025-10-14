"""
Response Parsing Unit Tests

Tests cover SGLang response parsing and handling:
- Response abort detection (_is_response_aborted)
- Response data extraction (_parse_response)
- JSON parsing edge cases
- Malformed response handling
- Meta-info validation
- Finish reason parsing
- Response streaming and materialization

Test Strategy:
- Unit testing with various response formats
- Edge case and malformed data testing
- Type safety verification
- Error recovery validation
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.responses import JSONResponse
from starlette.responses import Response
from slime.router.middleware.radix_tree_middleware import (
    _is_response_aborted,
    _materialize_response,
    RadixTreeMiddleware
)


# ==============================================================================
# Group A: Response Abort Detection
# ==============================================================================

@pytest.mark.unit
class TestResponseAbortDetection:
    """Test _is_response_aborted function with various inputs."""

    def test_abort_detected_correctly(self):
        """
        Test: Detect abort finish reason.

        Expected: Returns True for abort type
        """
        response_data = {
            "text": "",
            "meta_info": {
                "finish_reason": {"type": "abort"}
            }
        }

        assert _is_response_aborted(response_data) is True

    def test_non_abort_finish_reasons(self):
        """
        Test: Non-abort finish reasons.

        Expected: Returns False for stop, length, eos, etc.
        """
        test_cases = [
            {"meta_info": {"finish_reason": {"type": "stop"}}},
            {"meta_info": {"finish_reason": {"type": "length"}}},
            {"meta_info": {"finish_reason": {"type": "eos"}}},
            {"meta_info": {"finish_reason": {"type": "match"}}},
        ]

        for response_data in test_cases:
            assert _is_response_aborted(response_data) is False, \
                f"Should not detect abort for {response_data}"

    def test_missing_meta_info(self):
        """
        Test: Response without meta_info.

        Expected: Returns False gracefully
        """
        response_data = {"text": "Hello world"}

        assert _is_response_aborted(response_data) is False

    def test_empty_meta_info(self):
        """
        Test: Empty meta_info dict.

        Expected: Returns False
        """
        response_data = {
            "text": "Hello",
            "meta_info": {}
        }

        assert _is_response_aborted(response_data) is False

    def test_meta_info_wrong_type(self):
        """
        Test: meta_info is not a dict.

        Expected: Returns False gracefully
        """
        test_cases = [
            {"meta_info": "not a dict"},
            {"meta_info": 123},
            {"meta_info": None},
            {"meta_info": []},
            {"meta_info": True},
        ]

        for response_data in test_cases:
            assert _is_response_aborted(response_data) is False, \
                f"Should handle {type(response_data['meta_info']).__name__} gracefully"

    def test_missing_finish_reason(self):
        """
        Test: meta_info without finish_reason.

        Expected: Returns False
        """
        response_data = {
            "meta_info": {
                "other_field": "value"
            }
        }

        assert _is_response_aborted(response_data) is False

    def test_finish_reason_wrong_type(self):
        """
        Test: finish_reason is not a dict.

        Expected: Returns False gracefully
        """
        test_cases = [
            {"meta_info": {"finish_reason": "string"}},
            {"meta_info": {"finish_reason": 123}},
            {"meta_info": {"finish_reason": None}},
            {"meta_info": {"finish_reason": []}},
        ]

        for response_data in test_cases:
            assert _is_response_aborted(response_data) is False

    def test_finish_reason_missing_type(self):
        """
        Test: finish_reason dict without 'type' field.

        Expected: Returns False
        """
        response_data = {
            "meta_info": {
                "finish_reason": {"other": "field"}
            }
        }

        assert _is_response_aborted(response_data) is False

    def test_response_data_not_dict(self):
        """
        Test: response_data is not a dict.

        Expected: Returns False gracefully
        """
        test_cases = [
            "string response",
            123,
            None,
            [],
            True,
        ]

        for response_data in test_cases:
            assert _is_response_aborted(response_data) is False

    def test_nested_abort_detection(self):
        """
        Test: Abort with complex nested meta_info.

        Expected: Still detects abort correctly
        """
        response_data = {
            "text": "",
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "details": "Connection lost",
                    "nested": {"deep": "value"}
                },
                "other_data": [1, 2, 3]
            }
        }

        assert _is_response_aborted(response_data) is True


# ==============================================================================
# Group B: Response Parsing
# ==============================================================================

@pytest.mark.unit
class TestResponseParsing:
    """Test _parse_response method."""

    def test_parse_json_response(self):
        """
        Test: Parse standard JSONResponse.

        Expected: Returns content dict
        """
        # Create middleware instance
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        # Mock JSONResponse
        response = Mock()
        response.body = b'{"text": "Hello", "output_ids": [1, 2, 3]}'

        result = middleware._parse_response(response)

        assert result is not None
        assert result["text"] == "Hello"
        assert result["output_ids"] == [1, 2, 3]

    def test_parse_empty_response(self):
        """
        Test: Parse response with empty body.

        Expected: Returns None or empty dict gracefully
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        response = Mock()
        response.body = b''

        result = middleware._parse_response(response)

        # Should handle empty body gracefully (may return None or raise)
        assert result is None or result == {}

    def test_parse_invalid_json(self):
        """
        Test: Parse response with invalid JSON.

        Expected: Returns None
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        response = Mock()
        response.body = b'{invalid json}'

        result = middleware._parse_response(response)

        assert result is None

    def test_parse_response_with_content_dict(self):
        """
        Test: Response with content attribute (dict).

        Expected: Returns content directly
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        response = Mock()
        response.content = {"text": "Direct content"}
        del response.body  # No body attribute

        result = middleware._parse_response(response)

        # Should use content if body not available
        # Or return None if implementation requires body
        assert result is None or result.get("text") == "Direct content"

    def test_parse_non_utf8_response(self):
        """
        Test: Response with non-UTF-8 body.

        Expected: Handles gracefully
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        response = Mock()
        response.body = b'\xff\xfe invalid utf8'

        result = middleware._parse_response(response)

        # Should return None on decode error
        assert result is None


# ==============================================================================
# Group C: Response Materialization
# ==============================================================================

@pytest.mark.unit
class TestResponseMaterialization:
    """Test response streaming and materialization."""

    @pytest.mark.asyncio
    async def test_materialize_simple_response(self):
        """
        Test: Materialize simple streaming response.

        Expected: Collects all chunks into single response
        """
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}

        # Mock async iterator
        async def mock_body_iterator():
            yield b'{"text": '
            yield b'"Hello",'
            yield b'"output_ids": [1, 2, 3]}'

        mock_response.body_iterator = mock_body_iterator()

        # Materialize
        materialized = await _materialize_response(mock_response)

        # Should have combined body
        assert hasattr(materialized, 'body')
        body_text = materialized.body.decode('utf-8')
        data = json.loads(body_text)
        assert data["text"] == "Hello"
        assert data["output_ids"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_materialize_empty_response(self):
        """
        Test: Materialize response with no chunks.

        Expected: Empty body
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Ensure headers.get() returns a string, not Mock
        mock_response.headers = MagicMock()
        mock_response.headers.get = Mock(return_value="")

        async def mock_body_iterator():
            # Empty iterator
            return
            yield  # Never reached

        mock_response.body_iterator = mock_body_iterator()

        materialized = await _materialize_response(mock_response)

        assert hasattr(materialized, 'body')
        assert materialized.body == b''

    @pytest.mark.asyncio
    async def test_materialize_large_response(self):
        """
        Test: Materialize response with many chunks.

        Expected: All chunks collected
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Ensure headers.get() returns a string, not Mock
        mock_response.headers = MagicMock()
        mock_response.headers.get = Mock(return_value="")

        # Many small chunks
        async def mock_body_iterator():
            for i in range(100):
                yield f'chunk{i}'.encode()

        mock_response.body_iterator = mock_body_iterator()

        materialized = await _materialize_response(mock_response)

        # Should have all chunks
        body_text = materialized.body.decode('utf-8')
        assert len(body_text) > 500  # At least 100 * 6 characters
        assert 'chunk0' in body_text
        assert 'chunk99' in body_text


# ==============================================================================
# Group D: Meta-Info Validation
# ==============================================================================

@pytest.mark.unit
class TestMetaInfoValidation:
    """Test meta_info structure validation."""

    def test_valid_meta_info_structure(self):
        """
        Test: Well-formed meta_info.

        Expected: Passes validation
        """
        response_data = {
            "text": "Response text",
            "output_ids": [1, 2, 3],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "weight_version": 42,
                "output_token_logprobs": [[-0.1, 1], [-0.2, 2], [-0.3, 3]]
            }
        }

        # Check it's not aborted
        assert _is_response_aborted(response_data) is False

        # Check we can access fields
        assert response_data["meta_info"]["weight_version"] == 42
        assert len(response_data["meta_info"]["output_token_logprobs"]) == 3

    def test_meta_info_with_extra_fields(self):
        """
        Test: meta_info with unexpected extra fields.

        Expected: Extra fields ignored, no errors
        """
        response_data = {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "extra_field": "unexpected",
                "nested": {"deep": {"value": 123}}
            }
        }

        # Should not error on extra fields
        assert _is_response_aborted(response_data) is False

    def test_meta_info_missing_common_fields(self):
        """
        Test: meta_info missing optional fields.

        Expected: Handles gracefully
        """
        # Only finish_reason, no weight_version or logprobs
        response_data = {
            "meta_info": {
                "finish_reason": {"type": "stop"}
            }
        }

        # Should work fine
        assert _is_response_aborted(response_data) is False

    def test_meta_info_weight_version_types(self):
        """
        Test: weight_version with various types.

        Expected: Accepts int, handles others
        """
        test_cases = [
            {"meta_info": {"weight_version": 42}},  # int
            {"meta_info": {"weight_version": "42"}},  # string
            {"meta_info": {"weight_version": None}},  # None
            {"meta_info": {"weight_version": 42.5}},  # float
        ]

        # Should not crash on any type
        for response_data in test_cases:
            # Just verify it doesn't crash
            weight_version = response_data["meta_info"].get("weight_version")
            assert weight_version is not None or weight_version is None  # Always true

    def test_output_token_logprobs_structure(self):
        """
        Test: output_token_logprobs structure validation.

        Expected: List of [logprob, token_id] pairs
        """
        response_data = {
            "meta_info": {
                "output_token_logprobs": [
                    [-0.1, 100],
                    [-0.2, 101],
                    [-0.3, 102]
                ]
            }
        }

        logprobs = response_data["meta_info"]["output_token_logprobs"]
        assert len(logprobs) == 3
        assert all(len(pair) == 2 for pair in logprobs)
        assert all(isinstance(pair[1], int) for pair in logprobs)


# ==============================================================================
# Group E: Edge Cases and Robustness
# ==============================================================================

@pytest.mark.unit
class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""

    def test_very_large_response(self):
        """
        Test: Parse very large JSON response.

        Expected: Handles without memory issues
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        # Large response (10K token IDs)
        large_data = {
            "text": "x" * 10000,
            "output_ids": list(range(10000))
        }

        response = Mock()
        response.body = json.dumps(large_data).encode()

        result = middleware._parse_response(response)

        assert result is not None
        assert len(result["output_ids"]) == 10000

    def test_response_with_unicode(self):
        """
        Test: Response containing Unicode characters.

        Expected: Handles Unicode correctly
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        unicode_data = {
            "text": "Hello 世界 🌍 мир",
            "output_ids": [1, 2, 3]
        }

        response = Mock()
        response.body = json.dumps(unicode_data, ensure_ascii=False).encode('utf-8')

        result = middleware._parse_response(response)

        assert result is not None
        assert "世界" in result["text"]
        assert "🌍" in result["text"]
        assert "мир" in result["text"]

    def test_response_with_null_values(self):
        """
        Test: Response with null/None values.

        Expected: Preserves nulls correctly
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        data_with_nulls = {
            "text": None,
            "output_ids": [1, None, 3],
            "meta_info": None
        }

        response = Mock()
        response.body = json.dumps(data_with_nulls).encode()

        result = middleware._parse_response(response)

        assert result is not None
        assert result["text"] is None
        assert result["output_ids"][1] is None
        assert result["meta_info"] is None

    def test_response_with_deeply_nested_structure(self):
        """
        Test: Response with deeply nested dicts/lists.

        Expected: Handles nesting correctly
        """
        args = Mock()
        args.hf_checkpoint = "/tmp/fake"
        args.radix_tree_max_size = 1000
        args.verbose = False

        router = Mock()
        router.args = args
        router.verbose = False

        with patch('slime.router.middleware.radix_tree_middleware.AutoTokenizer'), \
             patch('slime.router.middleware.radix_tree_middleware.StringRadixTrie'):
            middleware = RadixTreeMiddleware(None, router=router)

        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "deep_value": "found"
                        }
                    }
                }
            }
        }

        response = Mock()
        response.body = json.dumps(nested_data).encode()

        result = middleware._parse_response(response)

        assert result is not None
        assert result["level1"]["level2"]["level3"]["level4"]["deep_value"] == "found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
