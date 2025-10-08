"""
Router-specific pytest configuration and fixtures.

This conftest provides fixtures exclusively for router module testing,
including mock tokenizers, SGLang response mocks, and radix tree fixtures.
"""

from typing import Any, Dict, List

import pytest


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable token IDs."""

    class MockTokenizer:
        def encode(self, text: str) -> List[int]:
            # Simple mock: each character → ASCII code
            return [ord(c) for c in text]

        def decode(self, token_ids: List[int]) -> str:
            return "".join(chr(tid) for tid in token_ids)

    return MockTokenizer()


@pytest.fixture
def mock_sglang_response():
    """Mock SGLang API response."""

    def _response(text: str, finish_reason: str = "stop") -> Dict[str, Any]:
        return {
            "text": text,
            "output_ids": [ord(c) for c in text],
            "meta_info": {"finish_reason": {"type": finish_reason}},
        }

    return _response


@pytest.fixture
def sample_radix_tree():
    """Create a sample radix tree for testing."""
    from slime.router.middleware_hub.radix_tree import StringRadixTrie

    tree = StringRadixTrie()
    # Pre-populate with some test data
    tree.insert("Hello", [72, 101, 108, 108, 111], [0.1] * 5, [1] * 5, weight_version=1)
    tree.insert(
        "Hello World",
        [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100],
        [0.1] * 11,
        [1] * 11,
        weight_version=1,
    )
    return tree


@pytest.fixture
def qwen_tokenizer():
    """Load Qwen3-0.6B tokenizer for real tokenization tests."""
    import os

    from transformers import AutoTokenizer

    tokenizer_path = "./tmp_models/Qwen3-0.6B"
    if not os.path.exists(tokenizer_path):
        pytest.skip("Qwen3-0.6B tokenizer not available")
    return AutoTokenizer.from_pretrained(tokenizer_path)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: marks tests requiring remote SGLang server (deselect with '-m \"not e2e\"')"
    )


@pytest.fixture
def remote_sglang_available():
    """Check if remote SGLang server is accessible."""
    try:
        from tests.router.mocks.remote_sglang_client import RemoteSGLangClient

        client = RemoteSGLangClient()
        is_available = client.health_check()

        if not is_available:
            pytest.skip("Remote SGLang server not available")

        return client
    except ImportError:
        pytest.skip("Remote SGLang client not available")
    except ImportError:
        pytest.skip("Remote SGLang client not available")
