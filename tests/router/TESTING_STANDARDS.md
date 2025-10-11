# Slime Router Testing Standards

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
├── unit/           # Unit tests with mocked dependencies
├── integration/    # Integration tests with real components
├── e2e/           # End-to-end tests with full system
├── mocks/         # Shared mock utilities
└── conftest.py    # Shared fixtures and configuration
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows

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