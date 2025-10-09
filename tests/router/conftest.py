"""
Router-specific pytest configuration and fixtures.

This conftest provides fixtures exclusively for router module testing,
including mock tokenizers, SGLang response mocks, and radix tree fixtures.

Configuration:
- TEST_TOKENIZER_PATH: Path to tokenizer model (default: ./tmp_models/Qwen3-0.6B)
- TEST_USE_REAL_DEPENDENCIES: "true" to prioritize real dependencies over mocks
"""

import os
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


@pytest.fixture(scope="session")
def tokenizer_model_path():
    """Get tokenizer model path, configurable via environment variable."""
    default_path = "./tmp_models/Qwen3-0.6B"
    # Support environment variable configuration
    model_path = os.getenv("TEST_TOKENIZER_PATH", default_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Tokenizer not found at {model_path}")
    return model_path


@pytest.fixture(scope="session")
def real_tokenizer(tokenizer_model_path):
    """Load real tokenizer instance for authentic tokenization tests."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_model_path)


@pytest.fixture
def flexible_tokenizer():
    """
    Simplified tokenizer fixture for stable testing.

    This fixture always uses a lightweight mock tokenizer to ensure
    CI stability and consistent test behavior across environments.

    For tests requiring real tokenization, use the 'real_tokenizer' fixture
    and mark tests with 'unit_with_deps' or 'integration' markers.
    """
    # Lightweight mock that works consistently across all environments
    from unittest.mock import Mock

    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [ord(c) for c in text]
    tokenizer.decode.side_effect = lambda token_ids: "".join(chr(tid) for tid in token_ids)
    tokenizer.model_max_length = 8192  # Mock max length

    # Add chat template support for chat completion tests
    tokenizer.apply_chat_template.side_effect = lambda messages, tokenize=False, add_generation_prompt=True: (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) +
        "\nassistant:"
    )

    return tokenizer


@pytest.fixture
def qwen_tokenizer(tokenizer_model_path):
    """Legacy fixture for backward compatibility - use flexible_tokenizer instead."""
    from transformers import AutoTokenizer
    if not os.path.exists(tokenizer_model_path):
        pytest.skip("Qwen3-0.6B tokenizer not available")
    return AutoTokenizer.from_pretrained(tokenizer_model_path)


@pytest.fixture
def real_radix_tree(flexible_tokenizer):
    """Create a StringRadixTrie instance with mock tokenizer."""
    from slime.router.middleware_hub.radix_tree import StringRadixTrie
    return StringRadixTrie(
        max_cache_size=1000,
        tokenizer=flexible_tokenizer,
        verbose=False
    )


@pytest.fixture
def simple_middleware(flexible_tokenizer, real_radix_tree):
    """
    Create a simplified RadixTreeMiddleware for testing.
    Uses mock tokenizer and radix tree for stable testing.
    """
    from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware
    from unittest.mock import Mock

    mock_router = Mock()
    mock_router.args = Mock()
    mock_router.args.hf_checkpoint = "test-checkpoint"
    mock_router.args.radix_tree_max_size = 1000
    mock_router.args.verbose = False
    mock_router.verbose = False

    # Create middleware with mocked tokenizer loading
    with pytest.MonkeyPatch().context() as m:
        m.setattr("transformers.AutoTokenizer.from_pretrained",
                  lambda *args, **kwargs: flexible_tokenizer)

        middleware = RadixTreeMiddleware(app=None, router=mock_router)
        # Replace with test radix tree and tokenizer
        middleware.radix_tree = real_radix_tree
        middleware.tokenizer = flexible_tokenizer

        return middleware


def pytest_configure(config):
    """Register custom markers for clear test categorization."""
    # Test type markers
    config.addinivalue_line(
        "markers", "unit: marks tests that test individual components in isolation (fast, stable)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring multiple components working together"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests requiring external services (slow, requires setup)"
    )

    # Dependency markers
    config.addinivalue_line(
        "markers", "requires_real_tokenizer: marks tests that need real tokenizer (may skip in CI)"
    )
    config.addinivalue_line(
        "markers", "requires_remote_sglang: marks tests that need remote SGLang server"
    )

    # Stability markers
    config.addinivalue_line(
        "markers", "stable: marks tests that are stable for CI environments"
    )
    config.addinivalue_line(
        "markers", "unstable: marks tests that may be flaky in CI environments"
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
