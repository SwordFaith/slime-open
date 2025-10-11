# Slime Router 开发指南

## 1. 快速开始

### 1.1 环境搭建

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/router/unit/ -v
```

### 1.2 项目结构

```
slime/router/
├── router.py                     # FastAPI 路由服务
├── openai_chat_completion.py     # OpenAI API 兼容层
├── component_registry.py         # 组件依赖注入
└── middleware_hub/               # 中间件插件系统
    ├── radix_tree_middleware.py  # Radix Tree 缓存中间件
    ├── radix_tree.py            # 前缀缓存数据结构
    └── async_read_write_lock.py # 异步读写锁

tests/router/
├── unit/                        # 单元测试（快速，无依赖）
├── integration/                 # 集成测试（Mock 外部依赖）
└── e2e/                        # 端到端测试（需要真实服务）
```

---

## 2. 添加 Middleware

### 2.1 基础模板

```python
from starlette.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.args = router.args

    async def dispatch(self, request: Request, call_next):
        # 预处理请求
        # response = await call_next(request)
        # 后处理响应
        return response
```

### 2.2 注册使用

```bash
python -m slime.ray.rollout \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.custom_middleware.CustomMiddleware
```

### 2.3 日志示例

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.verbose = getattr(router, "verbose", False)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)

        if self.verbose:
            latency = (time.time() - start_time) * 1000
            print(f"[{request.method}] {request.url.path} - {response.status_code} ({latency:.1f}ms)")

        return response
```

---

## 3. ComponentRegistry 组件管理

### 3.1 基本使用

```python
# 在 Middleware 中注册组件
class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router

        # 注册自定义组件
        cache = CustomCache(max_size=5000)
        router.component_registry.register("custom_cache", cache)

        # 获取已注册组件
        tokenizer = router.component_registry.get("tokenizer")
        radix_tree = router.component_registry.get("radix_tree")
```

### 3.2 推荐命名

- `tokenizer` - HuggingFace tokenizer
- `radix_tree` - Radix Tree 缓存
- `metrics` - 指标收集器
- `cache` - 自定义缓存
- `logger` - 日志记录器

### 3.3 错误处理

```python
def safe_get_component(router, name: str, fallback=None):
    try:
        return router.component_registry.get(name)
    except RuntimeError:
        return fallback
```

---

## 4. 异步开发最佳实践

### 4.1 异步编程模式

**✅ 推荐模式**：

```python
# 正确用法 - 异步友好
class GoodExample:
    def __init__(self):
        self._lock = AsyncReadWriteLock()
        self._data = {}

    async def get_data(self, key: str) -> Any:
        async with self._lock.reader():
            return self._data.get(key)
```

**❌ 错误模式**：

```python
# 错误用法 - 阻塞事件循环
import time

def blocking_operation():
    time.sleep(1)  # 阻塞操作
    return "result"
```

### 4.2 并发测试模式

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_concurrent_operations():
    # 并发执行多个操作
    tasks = [some_async_function(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
```

### 4.3 性能优化技巧

1. **减少锁竞争** - 使用读写锁替代互斥锁
2. **批量操作** - 合并多个小操作
3. **内存优化** - 使用弱引用缓存，避免大对象复制

---

## 5. 测试策略

### 5.1 分层测试架构

**三层测试金字塔**：
- **Unit Tests** - 纯数据结构逻辑，无外部依赖，极快 (<1s)，>90% 覆盖率
- **Integration Tests** - Mock 外部依赖，Mock FastAPI/SGLang，快 (~10s)，>80% 覆盖率
- **E2E Tests** - 完整服务测试，真实 SGLang server，慢 (~60s)，>60% 覆盖率

### 5.2 Unit Tests

**目录**: `tests/router/unit/`

```python
import pytest
from slime.router.middleware_hub.radix_tree import StringRadixTrie

def test_radix_tree_insert_and_query():
    """Test basic insert and query operations."""
    tree = StringRadixTrie(max_cache_size=1000)

    # Insert
    tree.insert(
        text="Hello\nWorld",
        token_ids=[1, 2, 3, 4, 5],
        logp=[-0.1, -0.2, -0.3, -0.4, -0.5],
        loss_mask=[0, 0, 0, 1, 1]
    )

    # Query
    result = tree.find_longest_prefix("Hello\nWorld")

    # Assertions
    assert result.matched_prefix == "Hello\nWorld"
    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.remaining_string == ""
```

**运行**：
```bash
pytest tests/router/unit/ -v
```

### 5.3 Integration Tests

**目录**: `tests/router/integration/`

```python
@pytest.mark.asyncio
async def test_middleware_cache_insertion(mocker):
    """Test middleware inserts generated tokens into cache."""
    # Mock tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

    # Mock SGLang response
    sglang_response = {
        "text": " World",
        "output_ids": [4, 5],
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "weight_version": 10,
            "output_token_logprobs": [[-0.4, 4], [-0.5, 5]]
        }
    }

    mock_call_next = AsyncMock(return_value=create_response(sglang_response))
    middleware = RadixTreeMiddleware(app, router=mock_router)
    middleware.tokenizer = mock_tokenizer
    request = create_request({"text": "Hello"})

    # Dispatch
    response = await middleware.dispatch(request, mock_call_next)

    # Verify cache insertion
    result = middleware.radix_tree.find_longest_prefix("Hello World")
    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.loss_mask == [0, 0, 0, 1, 1]
```

### 5.4 TDD 工作流

**Test-Driven Development 流程**：
1. 写测试 (验证预期失败)
2. 修复代码 (实现功能逻辑)
3. 运行测试 (验证通过)
4. 重构代码 (优化实现,保持测试通过)

### 5.5 异步性能测试

**关键性能指标**：

| 指标 | 同步 RLock | 异步 RWLock | 目标 |
|------|------------|-------------|------|
| 并发读取延迟 | ~45ms | ~0.4ms | >99% 改善 |
| 系统吞吐量 | ~1K ops/s | ~100K ops/s | >100倍提升 |
| 事件循环阻塞 | 是 | 否 | 完全消除 |

**运行性能测试**：
```bash
pytest tests/router/unit/test_performance_comparison.py -v -s
```

### 5.6 测试覆盖率要求

```bash
pytest tests/router/ -m "not e2e" --cov=slime/router --cov-report=term-missing
```

**覆盖率目标**：
- 修改的核心文件: >80% 覆盖率
- 新增功能: 100% 覆盖率
- 整体项目: >70% 覆盖率

---

## 6. 代码审查要点

### 6.1 关键检查项

**Async/Await 正确性**：
- [ ] 所有 I/O 操作使用 `await`
- [ ] 使用 `await asyncio.sleep()` 而非 `sleep()`
- [ ] Middleware 的 `dispatch()` 方法是 `async def`

**异步并发优化**：
- [ ] 避免使用 `threading.RLock` 等会阻塞事件循环的同步锁
- [ ] 高频读取操作使用 `AsyncReadWriteLock` 支持并发读取
- [ ] 包含异步性能测试验证优化效果

**并发安全**：
- [ ] 共享状态（如 `worker_urls`）使用 `asyncio.Lock` 保护
- [ ] 避免 race condition（多个 coroutine 修改同一变量）

**Loss Mask 语义**：
- [ ] Prompt tokens 标记为 `loss_mask=0`
- [ ] Response tokens 标记为 `loss_mask=1`
- [ ] 部分匹配时，剩余 text tokenize 后标记为 `0`

**Weight Version 传递**：
- [ ] SGLang response 中的 `weight_version` 正确解析
- [ ] 插入缓存时传递 `weight_version` 参数
- [ ] Traversed nodes 的 `weight_version` 更新到最新

**测试覆盖率**：
- [ ] 新增功能有对应的单元测试
- [ ] 测试覆盖率 >80%
- [ ] 包含边界条件和异常场景的测试

### 6.2 已知问题修复记录

**Critical Issues（已修复）**：

| Issue | 优先级 | 修复版本 | 描述 |
|-------|-------|---------|------|
| Async sleep blocking | P0 | Phase 3.1 | `sleep(30)` → `await asyncio.sleep(30)` |
| Concurrency safety | P0 | Phase 3.2 | 添加 `asyncio.Lock` 保护 worker 选择 |
| Weight version update | P0 | Phase 3.3 | 更新所有 traversed nodes 的 `weight_version` |
| **RLock event loop blocking** | **P0** | **Phase 4.1** | **实现 `AsyncReadWriteLock`，性能提升 99.1%** |

**测试验证**：
```bash
pytest tests/router/unit/test_radix_tree_core.py -v            # Weight version fix
pytest tests/router/unit/test_performance_comparison.py -v     # Async performance optimization
pytest tests/router/unit/test_async_read_write_lock.py -v      # AsyncReadWriteLock tests
```

---

## 7. 故障排查

### 7.1 缓存未命中

**症状**: `/metrics` 显示 `hit_rate` 接近 0

**可能原因**：
1. Prompt 格式不一致（多余空格/换行）
2. 每次请求的 text 都不同，无公共前缀
3. GC 过于激进，频繁删除缓存

**排查步骤**：
```bash
# 1. 检查缓存统计
curl http://localhost:30000/metrics | jq '.cache'

# 2. 检查 prompt 格式，确保多轮对话的前缀一致
```

**解决方案**：
```python
# 标准化 prompt 格式
def normalize_prompt(text):
    text = text.replace('\r\n', '\n')  # 统一换行符
    text = ' '.join(text.split())      # 去除多余空格
    return text
```

### 7.2 内存持续增长

**症状**: Router 进程内存占用不断增长

**可能原因**：
- `max_cache_size` 设置过大
- GC 未正常触发
- Weight version 未正确传递

**排查步骤**：
```bash
# 1. 检查当前缓存大小
curl http://localhost:30000/metrics | jq '.cache.cur_cache_size'

# 2. 监控内存占用
ps aux | grep "python.*router"
```

**解决方案**：
```python
# 1. 降低 max_cache_size
max_cache_size = 10000  # 从 100000 降低

# 2. 更激进的 GC
gc_threshold_k = 3  # 从 5 降低

# 3. 手动触发 GC
radix_tree.gc_by_weight_version(current_version, gc_threshold_k=3)
```

### 7.3 负载不均衡

**症状**: 某个 worker 负载远高于其他 workers

**可能原因**：
- 并发请求时的 race condition（已修复）
- Worker 性能差异（GPU 型号/内存不同）

**排查步骤**：
```bash
# 1. 检查 worker 负载分布
curl http://localhost:30000/metrics | jq '.router.worker_loads'

# 2. 检查 worker 健康状态
for url in $(curl -s http://localhost:30000/list_workers | jq -r '.urls[]'); do
    curl -s "$url/health" || echo "$url FAILED"
done
```

**解决方案**：
```bash
# 1. 升级到包含 asyncio.Lock 修复的代码
git pull origin main

# 2. 重启负载高的 worker
# (重新启动 SGLang server)

# 3. 移除异常 worker
curl -X POST "http://localhost:30000/remove_worker?url=http://worker3:10090"
```

### 7.4 Async Sleep 阻塞

**症状**: 单个 abort 请求导致所有请求被阻塞 30 秒

**根因**: 使用 `sleep(30)` 而非 `await asyncio.sleep(30)`

**验证**：
```python
# 检查代码中是否有同步 sleep
grep -r "sleep(" slime/router/middleware_hub/
# 应该只有 "asyncio.sleep"
```

**修复**（已完成）：
```python
# ❌ BEFORE
from time import sleep
sleep(30)

# ✅ AFTER
import asyncio
await asyncio.sleep(30)
```

---

## 8. 贡献流程

### 8.1 代码提交规范

**Commit Message 格式**：
```
<type>(<scope>): <subject>
```

**Type 类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例**：
```
feat(router): add cache invalidation middleware

Implement CacheInvalidationMiddleware to automatically
trigger GC when weight_version changes.

Closes #123
```

### 8.2 Pre-commit 检查

**安装 pre-commit hooks**：
```bash
pip install pre-commit
pre-commit install
```

**手动运行检查**：
```bash
pre-commit run --all-files
```

**检查项**：
- Black 代码格式化
- isort import 排序
- Flake8 代码规范

### 8.3 Pull Request 流程

**1. Fork 仓库并创建分支**：
```bash
git checkout -b feat/my-new-feature
```

**2. 编写代码 + 测试**：
```bash
# 编写代码
vim slime/router/middleware_hub/my_middleware.py

# 编写测试
vim tests/router/unit/test_my_middleware.py

# 运行测试
pytest tests/router/unit/test_my_middleware.py -v
```

**3. 运行 pre-commit 检查并提交**：
```bash
pre-commit run --all-files
git add .
git commit -m "feat(router): add my new middleware"
git push origin feat/my-new-feature
```

**4. 创建 Pull Request**：
- 访问 https://github.com/THUDM/slime/pulls
- 点击 "New Pull Request"
- 填写 PR 描述（包括修改内容、测试结果、性能影响）

---

## 9. 相关资源

### 内部文档
- **架构设计**: [architecture.md](architecture.md)
- **用户手册**: [user-guide.md](user-guide.md)
- **测试指南**: [testing-guide.md](testing-guide.md)
- **迁移指南**: [migration-guide.md](migration-guide.md)

### 外部资源
- **FastAPI 文档**: https://fastapi.tiangolo.com/
- **Starlette Middleware**: https://www.starlette.io/middleware/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/

### 代码位置
- **Router 实现**: `slime/router/router.py`
- **Radix Tree**: `slime/router/middleware_hub/radix_tree.py`
- **Middleware**: `slime/router/middleware_hub/radix_tree_middleware.py`
- **测试代码**: `tests/router/unit/`, `tests/router/integration/`

### 社区
- **GitHub Issues**: https://github.com/THUDM/slime/issues
- **Discussions**: https://github.com/THUDM/slime/discussions

---

## 10. 总结

### 10.1 核心原则

1. **异步优先**: 使用 FastAPI 和异步编程模式
2. **组件化**: 通过 ComponentRegistry 实现松耦合
3. **中间件模式**: 通过中间件扩展功能
4. **线程安全**: 正确处理并发访问

### 10.2 最佳实践

1. **错误处理**: 完整的异常处理和恢复机制
2. **性能优化**: 缓存、批量操作、内存管理
3. **监控告警**: 全面的指标监控和告警
4. **安全防护**: 认证授权、输入验证、防护措施

### 10.3 运维考虑

1. **部署架构**: 容器化、负载均衡、高可用
2. **故障排查**: 系统化的诊断和恢复流程
3. **日志管理**: 结构化日志和分析
4. **自动化**: 自动恢复和运维工具

通过遵循这些原则和实践，可以构建高质量、高可靠的 Slime Router 系统。