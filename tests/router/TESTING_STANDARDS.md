# Slime Router Testing Standards

## Recent Reorganization (2025-10-12)

The test directory structure was reorganized to follow the 3-layer testing architecture:

- **Removed**: `comprehensive/` directory - tests have been properly categorized into `unit/` or `integration/`
- **Enhanced**: `unit/` directory - added focused unit tests for GC, version consistency, and performance
- **Enhanced**: `integration/` directory - added comprehensive integration and performance tests
- **Improved**: Documentation - updated this document to reflect the new structure

This reorganization improves maintainability and follows established testing standards while preserving all test coverage.

## File Naming Conventions

### Test File Names
- Use descriptive names prefixed with `test_`
- Unit tests: `test_<module_name>.py` (e.g., `test_radix_tree_core.py`)
- Integration tests: `test_<feature>_integration.py` or place in `integration/` directory
- E2E tests: `test_<feature>_e2e.py` or place in `e2e/` directory

### Class and Function Names
- Test classes: `Test<FeatureName>` (e.g., `TestRadixTreeCore`)
- Test functions: `test_<specific_behavior>` (e.g., `test_cache_hit_retrieval`)
- Fixture functions: `<descriptive_name>` or `mock_<component>`

## Documentation Standards

### Module Docstrings
Every test file should have a comprehensive module docstring following this format:

```python
"""
Brief description of what this test suite validates.

Tests cover:
- Specific feature 1
- Specific feature 2
- Specific feature 3

Test Strategy:
- Type of testing (unit/integration/e2e)
- Mock strategy (if applicable)
- Focus areas and verification approach
"""
```

### Test Function Docstrings
Each test function should have a clear docstring:

```python
def test_specific_behavior():
    """
    Test: Brief description of what is being tested.

    Scenario:
    - Specific condition or setup
    - Expected behavior or outcome

    Verification:
    - What assertions verify correctness
    """
```

## Test Organization

### Directory Structure
```
tests/router/
├── unit/                     # Unit tests with mocked dependencies
│   ├── test_radix_tree_core_merged.py      # Core radix tree functionality
│   ├── test_radix_tree_async_core.py       # Async radix tree core
│   ├── test_radix_tree_async_integration.py # Async integration tests
│   ├── test_gc_unit.py                     # Garbage collection unit tests
│   ├── test_version_consistency.py         # Version consistency tests
│   ├── test_performance_gc_merged.py       # Performance and GC tests
│   └── test_openai_middleware_merged.py    # OpenAI middleware tests
├── integration/              # Integration tests with real components
│   ├── test_openai_integration.py         # OpenAI API integration
│   ├── test_performance_integration.py     # Performance integration tests
│   ├── test_comprehensive_integration.py   # Comprehensive functionality tests
│   ├── test_radix_tree_middleware.py      # Radix tree middleware tests
│   ├── test_router_concurrency.py         # Router concurrency tests
│   ├── test_router_metrics_simple.py      # Router metrics tests
│   ├── test_simplified_chat_completion.py  # Chat completion tests
│   └── test_streaming_and_cache.py        # Streaming and cache tests
├── e2e/                      # End-to-end tests with full system
│   └── test_openai_e2e.py                   # OpenAI E2E tests
├── mocks/                    # Shared mock utilities
│   └── remote_sglang_client.py             # SGLang client mocks
├── conftest.py               # Shared fixtures and configuration
└── TESTING_STANDARDS.md      # This document
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
  - Core functionality (radix tree algorithms, data structures)
  - Garbage collection mechanisms
  - Version consistency and management
  - Basic OpenAI middleware operations
- **Integration Tests**: Test component interactions
  - Performance benchmarking and scalability
  - Comprehensive functionality across multiple components
  - Router concurrency and metrics
  - Streaming and caching integration
- **E2E Tests**: Test complete user workflows
  - Full OpenAI API integration
  - End-to-end request/response flows

## Mock Strategy Guidelines

### When to Mock
- External dependencies (HTTP clients, databases)
- Heavy operations (file I/O, network calls)
- Time-sensitive operations (use time mocks)

### When Not to Mock
- Core business logic under test
- Simple data structures
- Performance-critical paths

## Best Practices

### Test Structure
1. **Arrange**: Set up test data and mocks
2. **Act**: Execute the function/method being tested
3. **Assert**: Verify the expected behavior

### Async Testing
- Use `@pytest.mark.asyncio` for async test functions
- Use `AsyncMock` for async dependencies
- Test both success and error paths

### Error Testing
- Test expected exceptions with `pytest.raises`
- Test edge cases and boundary conditions
- Verify graceful degradation

## Code Quality

### Test Attributes
- Use descriptive test names
- Keep tests focused and independent
- Use fixtures for common setup

### Assertions
- Use specific assertions (`assert_equal`, `assert_in`)
- Include helpful error messages
- Test both positive and negative cases

## Examples

### Good Unit Test
```python
def test_cache_hit_returns_expected_data(mock_cache):
    """
    Test: Cache hit returns expected data for known key.

    Scenario:
    - Cache contains data for "test_key"
    - Request retrieval of "test_key"

    Verification:
    - Returns the cached data
    - Cache retrieval method called with correct key
    """
    result = mock_cache.get("test_key")
    assert result == {"data": "expected_value"}
    mock_cache.get.assert_called_once_with("test_key")
```

### Good Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_middleware_cache_flow(middleware, mock_request):
    """
    Test: Middleware correctly handles cache hit flow.

    Scenario:
    - Request matches cached content
    - Middleware should use cached data

    Verification:
    - Request modified with cached tokens
    - Next middleware not called for generation
    """
    response = await middleware.dispatch(mock_request)
    assert hasattr(mock_request, '_json')
    assert 'input_tokens' in mock_request._json
```

## Review Guidelines

When reviewing tests, check for:
- Clear test purpose and documentation
- Proper mock usage
- Comprehensive coverage
- Independent test execution
- Appropriate assertions
- Handle both success and failure cases