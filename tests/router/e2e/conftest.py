"""
E2E Test Shared Fixtures

Provides shared fixtures for all E2E tests in the router module:
- SGLang server instance (session-scoped, shared across all tests)
- Tokenizer (session-scoped, shared across all tests)
- Router instances (with/without RadixTree middleware)
- Test client instances for HTTP requests

Environment Configuration:
- SGLANG_E2E_MODEL: Model path (default: Qwen/Qwen3-4B-Thinking-2507)
- SGLANG_E2E_PORT: SGLang server port (default: 30001)
- ROUTER_E2E_PORT: Router server port (default: 30000)
- SKIP_E2E_TESTS: Set to "1" to skip E2E tests that require GPU

Test Model:
- Qwen/Qwen3-4B-Thinking-2507
- Parser configs: Tool call parser (qwen25), Reasoning parser (qwen3)
- License: Apache 2.0

Running Tests:
  pytest tests/router/e2e/ -v -s -m e2e
"""

import os
import pytest
from typing import Generator
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server


# ==================== Configuration ====================

# Model configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_PATH = os.getenv("SGLANG_E2E_MODEL", DEFAULT_MODEL_PATH)

# Port configuration
SGLANG_PORT = int(os.getenv("SGLANG_E2E_PORT", "30001"))
ROUTER_PORT = int(os.getenv("ROUTER_E2E_PORT", "30000"))

# Parser configuration for Qwen3-4B-Thinking model
TOOL_CALL_PARSER = "qwen25"
REASONING_PARSER = "qwen3"


# ==================== Session-Scoped Fixtures ====================


@pytest.fixture(scope="session")
def sglang_server() -> Generator[int, None, None]:
    """
    Launch SGLang server with parsers for the entire test session.

    This fixture is session-scoped to avoid repeated server startup/shutdown.
    All E2E tests share the same SGLang server instance.

    Environment Variables:
        SKIP_E2E_TESTS: Set to "1" to skip E2E tests
        SGLANG_E2E_EXISTING_PORT: Use existing SGLang server on this port (skips launch)

    Yields:
        int: Port number where SGLang server is running

    Raises:
        pytest.skip: If E2E tests should be skipped (SKIP_E2E_TESTS=1)
    """
    # Check if E2E tests should be skipped
    if os.getenv("SKIP_E2E_TESTS") == "1":
        pytest.skip("E2E tests disabled via SKIP_E2E_TESTS=1")

    # Check if using existing SGLang server
    existing_port = os.getenv("SGLANG_E2E_EXISTING_PORT")
    if existing_port:
        port = int(existing_port)
        print(f"\n[E2E Setup] Using existing SGLang server on port {port}")
        print(f"  Skipping server launch (SGLANG_E2E_EXISTING_PORT={existing_port})")

        # Verify server is accessible
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            print(f"  ✓ Server health check passed")
        except Exception as e:
            print(f"  ⚠ Warning: Could not verify server health: {e}")
            print(f"  Proceeding anyway...")

        yield port

        # Don't terminate pre-existing server
        print(f"\n[E2E Teardown] Skipping server termination (using existing server)")
        return

    # Build launch command
    cmd = (
        f"python -m sglang.launch_server "
        f"--model-path {MODEL_PATH} "
        f"--host 0.0.0.0 "
        f"--port {SGLANG_PORT} "
        f"--tool-call-parser {TOOL_CALL_PARSER} "
        f"--reasoning-parser {REASONING_PARSER}"
    )

    print(f"\n[E2E Setup] Launching SGLang server...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Port: {SGLANG_PORT}")
    print(f"  Tool Call Parser: {TOOL_CALL_PARSER}")
    print(f"  Reasoning Parser: {REASONING_PARSER}")
    print(f"  Command: {cmd}")
    print(f"  Note: This may take 2-5 minutes on first run...")

    # Launch server
    process, port = launch_server_cmd(cmd)
    wait_for_server(f"http://localhost:{port}")
    print(f"[E2E Setup] SGLang server ready on port {port}")

    yield port

    # Cleanup
    print("\n[E2E Teardown] Terminating SGLang server...")
    terminate_process(process)
    print("[E2E Teardown] SGLang server terminated")


@pytest.fixture(scope="session")
def tokenizer():
    """
    Load tokenizer for the test model (session-scoped).

    Shared across all E2E tests to avoid repeated tokenizer loading.

    Returns:
        Tokenizer: HuggingFace tokenizer instance
    """
    print(f"\n[E2E Setup] Loading tokenizer: {MODEL_PATH}")
    tok = get_tokenizer(MODEL_PATH)
    print(f"[E2E Setup] Tokenizer loaded (vocab_size={tok.vocab_size})")
    return tok


# ==================== Router Fixtures ====================


@pytest.fixture
def router_without_cache(sglang_server) -> "SlimeRouter":
    """
    Create Router instance WITHOUT RadixTree middleware.

    Configuration:
    - Direct proxy mode (Path 1)
    - No token-level caching
    - Directly forwards requests to SGLang /v1/chat/completions

    Args:
        sglang_server: SGLang server port from session fixture

    Returns:
        SlimeRouter: Router instance without caching
    """
    from slime.router.router import SlimeRouter
    from unittest.mock import MagicMock

    print("\n[Fixture] Creating Router (no cache)")

    # Build router args
    args = MagicMock()
    args.sglang_router_ip = "0.0.0.0"
    args.sglang_router_port = ROUTER_PORT
    args.sglang_server_concurrency = 32
    args.rollout_num_gpus = 1
    args.rollout_num_gpus_per_engine = 1
    args.slime_router_middleware_paths = []  # No middleware
    args.verbose = True
    args.model_name = "qwen3-thinking"
    # Add parser configurations (use sglang_ prefix to match arguments.py)
    args.sglang_reasoning_parser = REASONING_PARSER  # qwen3
    args.sglang_tool_call_parser = TOOL_CALL_PARSER  # qwen25

    # Create router
    router = SlimeRouter(args, verbose=True)

    # Add SGLang worker
    worker_url = f"http://localhost:{sglang_server}"
    router.worker_urls[worker_url] = 0
    print(f"[Fixture] Added worker: {worker_url}")
    print(f"[Fixture] Router created (mode: direct proxy, no cache)")

    return router


@pytest.fixture
def router_with_cache(sglang_server) -> "SlimeRouter":
    """
    Create Router instance WITH RadixTree middleware.

    Configuration:
    - Token in/token out mode (Path 2)
    - RadixTree caching enabled
    - Direct token-based communication with SGLang /generate

    Args:
        sglang_server: SGLang server port from session fixture

    Returns:
        SlimeRouter: Router instance with caching enabled
    """
    from slime.router.router import SlimeRouter
    from unittest.mock import MagicMock

    print("\n[Fixture] Creating Router (with cache)")

    # Build router args with RadixTree middleware
    args = MagicMock()
    args.sglang_router_ip = "0.0.0.0"
    args.sglang_router_port = ROUTER_PORT
    args.sglang_server_concurrency = 32
    args.rollout_num_gpus = 1
    args.rollout_num_gpus_per_engine = 1
    args.slime_router_middleware_paths = [
        "slime.router.middleware.radix_tree_middleware.RadixTreeMiddleware"
    ]
    args.hf_checkpoint = MODEL_PATH  # Required for tokenizer in middleware
    args.radix_tree_max_size = 10000  # Cache size limit
    args.verbose = True
    args.model_name = "qwen3-thinking"
    # Add parser configurations (use sglang_ prefix to match arguments.py)
    args.sglang_reasoning_parser = REASONING_PARSER  # qwen3
    args.sglang_tool_call_parser = TOOL_CALL_PARSER  # qwen25

    # Create router
    router = SlimeRouter(args, verbose=True)

    # Add SGLang worker
    worker_url = f"http://localhost:{sglang_server}"
    router.worker_urls[worker_url] = 0
    print(f"[Fixture] Added worker: {worker_url}")
    print(f"[Fixture] Router created (mode: token in/token out, cache enabled)")

    return router


# ==================== Test Client Fixtures ====================


@pytest.fixture
def client_no_cache(router_without_cache):
    """
    Create FastAPI TestClient for router without cache.

    Args:
        router_without_cache: Router fixture

    Returns:
        TestClient: FastAPI test client instance
    """
    from fastapi.testclient import TestClient
    return TestClient(router_without_cache.app, raise_server_exceptions=False)


@pytest.fixture
def client_with_cache(router_with_cache):
    """
    Create FastAPI TestClient for router with cache.

    Args:
        router_with_cache: Router fixture

    Returns:
        TestClient: FastAPI test client instance
    """
    from fastapi.testclient import TestClient
    return TestClient(router_with_cache.app, raise_server_exceptions=False)


# ==================== Helper Fixtures ====================


@pytest.fixture
def sglang_url(sglang_server) -> str:
    """
    Get SGLang server base URL.

    Args:
        sglang_server: SGLang server port

    Returns:
        str: Full URL to SGLang server (e.g., "http://localhost:30001")
    """
    return f"http://localhost:{sglang_server}"


@pytest.fixture
def test_messages_simple() -> list:
    """
    Simple test messages for basic chat completion tests.

    Returns:
        list: OpenAI-format messages
    """
    return [{"role": "user", "content": "Hello, my name is Alice"}]


@pytest.fixture
def test_messages_reasoning() -> list:
    """
    Test messages designed to trigger reasoning behavior.

    Returns:
        list: OpenAI-format messages for reasoning tests
    """
    return [
        {
            "role": "user",
            "content": "Solve this step by step: If x + 5 = 10, what is x?"
        }
    ]


@pytest.fixture
def sampling_params_deterministic() -> dict:
    """
    Sampling parameters for deterministic generation (temperature=0).

    Returns:
        dict: Sampling parameters
    """
    return {
        "temperature": 0.0,
        "max_tokens": 50,
    }


@pytest.fixture
def sampling_params_varied() -> dict:
    """
    Sampling parameters for varied generation (temperature>0).

    Returns:
        dict: Sampling parameters
    """
    return {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 50,
    }


# ==================== Configuration Access ====================


@pytest.fixture
def e2e_config() -> dict:
    """
    Provide E2E test configuration as a dictionary.

    Returns:
        dict: Configuration values for E2E tests
    """
    return {
        "model_path": MODEL_PATH,
        "sglang_port": SGLANG_PORT,
        "router_port": ROUTER_PORT,
        "tool_call_parser": TOOL_CALL_PARSER,
        "reasoning_parser": REASONING_PARSER,
    }
