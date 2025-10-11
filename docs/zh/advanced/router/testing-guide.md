# Slime Router 测试指南

## 1. 概述

本文档为 Slime Router 开发者提供完整的测试指南，包括测试标准、最佳实践和工具使用。Slime Router 采用多层测试策略，确保代码质量和系统稳定性。

### 1.1 测试哲学

- **测试先行**: TDD 方法确保代码设计合理
- **全面覆盖**: 单元测试、集成测试、E2E 测试并重
- **自动化优先**: 所有测试应能自动运行
- **文档即测试**: 测试用例本身就是最佳文档

### 1.2 测试架构

```
tests/router/
├── unit/                    # 单元测试 - 测试单个组件
├── integration/             # 集成测试 - 测试组件交互
├── e2e/                    # 端到端测试 - 测试完整流程
├── mocks/                  # 共享 Mock 工具
├── conftest.py             # 共享 fixtures
└── TESTING_STANDARDS.md    # 测试标准规范
```

---

## 2. 测试类型详解

### 2.1 单元测试 (Unit Tests)

**目标**: 测试单个函数或类的独立功能

**特点**:
- 快速执行（毫秒级）
- 隔离测试环境
- Mock 外部依赖
- 覆盖边界条件

**示例文件**:
```
tests/router/unit/
├── test_radix_tree_core.py              # Radix Tree 核心算法
├── test_openai_chat_completion.py       # OpenAI API 处理逻辑
├── test_component_registry_thread_safety.py  # 并发安全
├── test_async_read_write_lock.py        # 异步锁机制
└── test_tenacity_retry_logic.py         # 重试逻辑
```

**编写指南**:
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestRadixTreeCore:
    def test_insert_basic_case(self):
        """测试基本的插入功能"""
        # Arrange
        tree = StringRadixTrie(max_cache_size=1000)

        # Act
        result = tree.insert("test", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 1, 1], weight_version=1)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_async_concurrent_read(self):
        """测试异步并发读取"""
        # 使用 AsyncMock 模拟异步依赖
        mock_function = AsyncMock(return_value="test_result")

        result = await mock_function()
        assert result == "test_result"
```

### 2.2 集成测试 (Integration Tests)

**目标**: 测试多个组件之间的交互

**特点**:
- 测试真实组件交互
- 部分使用 Mock（如外部 API）
- 测试数据流和接口契约
- 验证组件集成正确性

**示例文件**:
```
tests/router/integration/
├── test_openai_integration.py           # OpenAI API 集成
├── test_radix_tree_middleware.py        # 中间件集成
├── test_streaming_and_cache.py          # 流式响应与缓存
└── test_simplified_chat_completion.py   # 简化对话流程
```

**编写指南**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestMiddlewareIntegration:
    async def test_cache_flow_integration(self):
        """测试缓存流程的完整集成"""
        # 使用真实组件，但 Mock 外部依赖
        with patch('slime.router.middleware_hub.radix_tree_middleware.AutoTokenizer') as mock_tokenizer:
            # 设置真实的中间件行为
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.side_effect = self.mock_tokenizer_behavior
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # 测试完整流程
            middleware = RadixTreeMiddleware(app, router=mock_router)
            response = await middleware.dispatch(mock_request, mock_call_next)

            # 验证集成结果
            assert response.status_code == 200
```

### 2.3 端到端测试 (E2E Tests)

**目标**: 测试完整的用户场景

**特点**:
- 完整的系统环境
- 真实的网络调用
- 测试用户完整工作流
- 较高的执行时间

**示例文件**:
```
tests/router/e2e/
└── test_openai_e2e.py                     # OpenAI SDK 完整流程
```

**编写指南**:
```python
@pytest.mark.e2e
class TestOpenAIE2E:
    def test_sdk_compatibility_full_workflow(self):
        """测试 OpenAI SDK 完整工作流程"""
        # 使用真实的 OpenAI SDK
        client = OpenAI(
            api_key="dummy-key",
            base_url="http://localhost:30000/v1"
        )

        # 执行完整对话流程
        response = client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ]
        )

        # 验证完整结果
        assert response.choices[0].message.content is not None
        assert response.usage.total_tokens > 0
```

---

## 3. 测试标准和规范

### 3.1 文件命名规范

- **单元测试**: `test_<module_name>.py` (如 `test_radix_tree_core.py`)
- **集成测试**: `test_<feature>_integration.py` (如 `test_openai_integration.py`)
- **E2E 测试**: `test_<feature>_e2e.py` (如 `test_openai_e2e.py`)

### 3.2 测试结构规范

每个测试文件应遵循以下结构：

```python
"""
模块功能描述

Tests cover:
- 具体功能点 1
- 具体功能点 2
- 具体功能点 3

Test Strategy:
- 测试类型说明
- Mock 策略说明
- 重点关注领域
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Fixtures
@pytest.fixture
def mock_component():
    """创建 mock 组件"""
    component = Mock()
    component.method.return_value = "expected_value"
    return component

# Test classes
class TestFeature:
    def test_specific_behavior(self, mock_component):
        """测试: 具体行为描述

        Scenario:
        - 具体条件或设置
        - 预期行为或结果

        Verification:
        - 验证点说明
        """
        # Arrange, Act, Assert
        result = mock_component.method()
        assert result == "expected_value"
```

### 3.3 断言和验证规范

**好的断言**:
```python
# 具体且有帮助的错误信息
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
assert len(tokens) > 0, "Tokens list should not be empty"

# 使用特定的断言方法
assert_in("expected_key", response_dict, "Response should contain expected key")
assert_is_instance(response, dict, "Response should be a dictionary")
```

**避免的断言**:
```python
# 模糊的断言
assert response  # 太宽泛

# 没有错误信息
assert len(tokens) == 5

# 硬编码值
assert result == "some hardcoded value"
```

### 3.4 Mock 使用规范

**何时使用 Mock**:
- 外部 HTTP 调用
- 数据库操作
- 文件系统操作
- 时间敏感操作

**Mock 最佳实践**:
```python
# 1. 明确 Mock 的行为
mock_tokenizer = Mock()
mock_tokenizer.apply_chat_template.return_value = "formatted_text"

# 2. 使用 AsyncMock 处理异步方法
mock_async_function = AsyncMock(return_value="async_result")

# 3. 验证调用次数和参数
mock_function.assert_called_once_with(expected_arg)
mock_function.assert_called_with(
    messages=expected_messages,
    tokenize=True
)
```

---

## 4. 异步测试指南

### 4.1 异步测试基础

**必须使用 `@pytest.mark.asyncio`**:
```python
@pytest.mark.asyncio
async def test_async_function():
    """测试异步函数"""
    result = await some_async_function()
    assert result is not None
```

**AsyncMock 使用**:
```python
@pytest.mark.asyncio
async def test_with_async_mock():
    """使用 AsyncMock 测试异步依赖"""
    mock_async_func = AsyncMock(return_value="async_result")

    result = await mock_async_func("arg1", "arg2")

    assert result == "async_result"
    mock_async_func.assert_called_once_with("arg1", "arg2")
```

### 4.2 并发测试

**测试并发安全性**:
```python
@pytest.mark.asyncio
async def test_concurrent_access():
    """测试并发访问的安全性"""
    import asyncio

    async def worker(worker_id):
        # 模拟并发操作
        return await some_concurrent_operation(worker_id)

    # 创建多个并发任务
    tasks = [worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # 验证结果
    assert len(results) == 10
    assert all(r is not None for r in results)
```

### 4.3 异步上下文管理

**测试异步上下文管理器**:
```python
@pytest.mark.asyncio
async def test_async_context_manager():
    """测试异步上下文管理器"""
    async with AsyncReadWriteLock() as lock:
        # 在锁保护下执行操作
        result = await protected_operation()
        assert result is not None
```

---

## 5. 运行和调试测试

### 5.1 运行测试

**运行所有测试**:
```bash
pytest tests/router/ -v
```

**运行特定类型测试**:
```bash
# 只运行单元测试
pytest tests/router/unit/ -v

# 只运行集成测试
pytest tests/router/integration/ -v --maxfail=1

# 只运行 E2E 测试
pytest tests/router/e2e/ -v -s
```

**按标记运行**:
```bash
# 运行异步测试
pytest tests/router/ -m asyncio -v

# 运行集成测试
pytest tests/router/ -m integration -v

# 运行性能测试
pytest tests/router/ -m performance -v
```

### 5.2 调试测试

**使用 pdb 调试**:
```bash
pytest tests/router/unit/test_radix_tree.py::TestRadixTree::test_insert -s --pdb
```

**查看详细输出**:
```bash
pytest tests/router/ -v -s --tb=long
```

**运行特定测试**:
```bash
pytest tests/router/unit/test_openai_chat_completion.py::TestChatCompletionHandler::test_handle_request -v
```

### 5.3 测试覆盖率

**生成覆盖率报告**:
```bash
pytest tests/router/ --cov=slime/router --cov-report=html
```

**查看覆盖率阈值**:
```bash
pytest tests/router/ --cov=slime/router --cov-fail-under=80
```

---

## 6. 性能测试

### 6.1 基准测试

**简单的性能测试**:
```python
import time
import pytest

@pytest.mark.performance
def test_cache_lookup_performance():
    """测试缓存查找性能"""
    tree = StringRadixTrie(max_cache_size=10000)

    # 预填充数据
    for i in range(1000):
        tree.insert(f"test_{i}", [i], [0.1], [1], weight_version=1)

    # 测试查找性能
    start_time = time.time()
    for i in range(1000):
        result = tree.find_longest_prefix(f"test_{i}")
    end_time = time.time()

    # 验证性能要求
    duration = end_time - start_time
    assert duration < 1.0, f"Lookup took too long: {duration:.3f}s"
```

### 6.2 并发性能测试

**测试并发性能**:
```python
@pytest.mark.asyncio
@pytest.mark.performance
async def test_concurrent_read_performance():
    """测试并发读取性能"""
    import asyncio

    tree = StringRadixTrie(max_cache_size=1000)
    tree.insert("test", [1, 2, 3], [0.1, 0.2, 0.3], [1, 1, 1], weight_version=1)

    async def reader():
        return await tree.find_longest_prefix_async("test")

    # 并发执行
    start_time = time.time()
    tasks = [reader() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # 验证性能
    duration = end_time - start_time
    assert duration < 0.5, f"Concurrent reads took too long: {duration:.3f}s"
    assert len(results) == 100
```

---

## 7. 持续集成

### 7.1 Pre-commit 钩子

**安装 pre-commit**:
```bash
pre-commit install
```

**手动运行**:
```bash
pre-commit run --all-files
```

**Pre-commit 配置**:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/router/unit/
        language: system
        pass_filenames: false
        always_run: true
```

### 7.2 CI 配置

**GitHub Actions 示例**:
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest tests/router/ -v --cov=slime/router
```

---

## 8. 常见问题和解决方案

### 8.1 异步测试问题

**问题**: `RuntimeError: asyncio.run() cannot be called from a running event loop`

**解决方案**:
```python
# 在测试中使用 pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### 8.2 Mock 问题

**问题**: Mock 对象未正确配置

**解决方案**:
```python
# 确保 Mock 的 side_effect 正确设置
mock_tokenizer = Mock()
mock_tokenizer.apply_chat_template.side_effect = lambda x, **kwargs: f"formatted_{x}"

# 验证 Mock 调用
mock_tokenizer.apply_chat_template.assert_called()
```

### 8.3 测试隔离问题

**问题**: 测试之间相互影响

**解决方案**:
```python
# 使用 fixtures 确保测试隔离
@pytest.fixture(autouse=True)
def reset_global_state():
    """每个测试前重置全局状态"""
    # 重置逻辑
    yield
    # 清理逻辑
```

### 8.4 性能测试不稳定

**问题**: 性能测试结果不稳定

**解决方案**:
```python
# 增加测试容忍度
def test_performance_with_tolerance():
    start_time = time.time()
    result = function_under_test()
    duration = time.time() - start_time

    # 允许一定的时间浮动
    assert duration < 1.0 + 0.2, f"Too slow: {duration:.3f}s"
```

---

## 9. 最佳实践总结

### 9.1 DO's

✅ **使用描述性的测试名称**
```python
def test_cache_hit_returns_expected_data():
    # 而不是
def test_test_1():
```

✅ **每个测试只验证一个行为**
```python
def test_insertion_success():
    # 验证插入成功

def test_insertion_returns_correct_result():
    # 验证返回值正确
```

✅ **使用适当的 fixtures**
```python
@pytest.fixture
def sample_radix_tree():
    return StringRadixTrie(max_cache_size=1000)
```

✅ **提供清晰的错误信息**
```python
assert result == expected, f"Expected {expected}, got {result}"
```

### 9.2 DON'Ts

❌ **不要在测试中测试多个功能**
```python
def test_everything():
    # 插入 + 查找 + 删除 + 更新 = 不好
```

❌ **不要硬编码测试数据**
```python
def test_with_specific_data():
    # 避免
    result = process_data("hardcoded_string_12345")
```

❌ **不要忽略测试清理**
```python
def test_with_temp_files():
    # 创建临时文件后要清理
    temp_file = tempfile.NamedTemporaryFile()
    try:
        # 测试逻辑
        pass
    finally:
        temp_file.close()
```

---

## 10. 相关资源

### 10.1 文档链接

- **[Pytest 官方文档](https://docs.pytest.org/)**
- **[pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)**
- **[Python unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)**

### 10.2 工具推荐

- **pytest**: 主要测试框架
- **pytest-asyncio**: 异步测试支持
- **pytest-cov**: 覆盖率测试
- **pytest-mock**: Mock 工具
- **pytest-benchmark**: 性能基准测试

### 10.3 学习资源

- **[Effective Testing with pytest](https://effective-testing.com/)**
- **[Python Testing with pytest](https://pragprog.com/titles/bopytest/)**
- **[AsyncIO Testing Best Practices](https://asyncio.readthedocs.io/)**

---

**最后更新**: 2025-10-11
**版本**: v1.0.0
**维护者**: Slime Router 开发团队