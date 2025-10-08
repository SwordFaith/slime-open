# Slime Router 开发指南

## 1. 开发环境搭建

### 1.1 Python 环境

**推荐配置**：
- Python 3.10+
- uv (快速依赖管理工具)
- CPU-only torch (本地测试无需 GPU)

**环境初始化**：
```bash
# 创建虚拟环境
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

### 1.2 依赖说明

**核心依赖** (`requirements.txt`):
```txt
fastapi           # Router HTTP 服务
httpx[http2]      # HTTP 客户端（支持 HTTP/2）
transformers      # Tokenizer
torch             # 基础依赖
ray[default]      # 分布式计算
tenacity          # 重试机制
```

**开发依赖** (`requirements-dev.txt`):
```txt
pytest            # 测试框架
pytest-asyncio    # 异步测试支持
pytest-mock       # Mock 工具
pytest-cov        # 代码覆盖率
pre-commit        # 代码质量检查
black             # 代码格式化
```

### 1.3 测试框架配置

**pyproject.toml 配置**：
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # 自动识别 async 测试
markers = [
    "e2e: End-to-end tests (require server)",
]
```

**运行测试**：
```bash
# 本地测试（无需 GPU）
pytest tests/router/unit/ -v
pytest tests/router/integration/ -v

# 跳过 E2E 测试
pytest tests/ -m "not e2e" --cov=slime/router

# E2E 测试（需服务器）
pytest tests/ -m "e2e" -v
```

---

## 2. 添加新 Middleware

### 2.1 接口规范

**基类**：
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        """
        Initialize middleware.

        Args:
            app: FastAPI application instance
            router: SlimeRouter instance (provides access to args, workers, etc.)
        """
        super().__init__(app)
        self.router = router
        self.args = router.args

    async def dispatch(self, request: Request, call_next):
        """
        Process request and response.

        Args:
            request: Incoming HTTP request
            call_next: Callable to invoke next middleware/route handler

        Returns:
            Response: Modified or original response
        """
        # 1. Pre-processing (modify request)
        modified_request = self.process_request(request)

        # 2. Call downstream
        response = await call_next(modified_request)

        # 3. Post-processing (modify response)
        modified_response = self.process_response(response)

        return modified_response
```

### 2.2 注册方式

**命令行参数**：
```bash
python -m slime.ray.rollout \
  --use-slime-router \
  --slime-router-middleware-paths \
    slime.router.middleware_hub.custom_middleware.CustomMiddleware \
    slime.router.middleware_hub.another_middleware.AnotherMiddleware
```

**加载顺序**：
- Middleware 按注册顺序执行
- 请求流: CustomMiddleware → AnotherMiddleware → Route Handler
- 响应流: Route Handler → AnotherMiddleware → CustomMiddleware

### 2.3 完整示例：日志 Middleware

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request/response metadata for debugging."""

    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.verbose = getattr(router, "verbose", False)

    async def dispatch(self, request: Request, call_next):
        # Record start time
        start_time = time.time()
        path = request.url.path

        # Log request
        if self.verbose:
            print(f"[LoggingMiddleware] Request: {request.method} {path}")

        # Call downstream
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        if self.verbose:
            print(f"[LoggingMiddleware] Response: {path} - {response.status_code} ({latency_ms:.2f}ms)")

        # Add custom header
        response.headers["X-Latency-Ms"] = str(latency_ms)

        return response
```

**使用**：
```bash
python -m slime.ray.rollout \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.logging_middleware.LoggingMiddleware
```

### 2.4 高级示例：缓存失效 Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class CacheInvalidationMiddleware(BaseHTTPMiddleware):
    """Invalidate Radix Tree cache when weight version changes."""

    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.last_known_version = 0

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Intercept /generate responses
        if path == "/generate":
            response = await call_next(request)

            # Parse response to get weight_version
            if hasattr(response, "body"):
                import json
                response_data = json.loads(response.body.decode("utf-8"))

                current_version = response_data.get("meta_info", {}).get("weight_version", 0)

                # Detect version change
                if current_version > self.last_known_version:
                    if self.verbose:
                        print(f"[CacheInvalidation] Detected version change: {self.last_known_version} → {current_version}")

                    # Trigger GC
                    if hasattr(self.router, "radix_tree"):
                        removed = self.router.radix_tree.gc_by_weight_version(
                            current_version,
                            gc_threshold_k=5
                        )
                        if self.verbose:
                            print(f"[CacheInvalidation] GC removed {removed} entries")

                    self.last_known_version = current_version

            return response
        else:
            return await call_next(request)
```

---

## 3. 测试策略

### 3.1 分层测试架构

**三层测试金字塔**：

```
        E2E Tests (少量)
       /               \
      /   Integration   \
     /      Tests        \
    /    (中等数量)       \
   /_______________________\
  /                         \
 /       Unit Tests          \
/       (大量,快速)           \
\_____________________________/
```

**分层说明**：

| 层级 | 特点 | 依赖 | 运行速度 | 覆盖率目标 |
|-----|------|------|---------|----------|
| **Unit Tests** | 纯数据结构逻辑 | 无外部依赖 | 极快 (<1s) | >90% |
| **Integration Tests** | Mock 外部依赖 | Mock FastAPI, SGLang | 快 (~10s) | >80% |
| **E2E Tests** | 完整服务测试 | 真实 SGLang server | 慢 (~60s) | >60% |

### 3.2 Unit Tests

**目录**: `tests/router/unit/`

**特点**：
- 测试 Radix Tree 核心逻辑
- 无需网络、GPU、异步环境
- 使用 Mock tokenizer

**示例**：
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

### 3.3 Integration Tests

**目录**: `tests/router/integration/`

**特点**：
- 测试 Middleware 与 Router 集成
- Mock FastAPI Request/Response
- Mock SGLang worker 响应
- 使用真实 asyncio 环境

**示例**：
```python
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from slime.router.middleware_hub.radix_tree_middleware import RadixTreeMiddleware

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
            "output_token_logprobs": [
                [-0.4, 4],
                [-0.5, 5]
            ]
        }
    }

    mock_call_next = AsyncMock(return_value=create_response(sglang_response))

    # Create middleware
    middleware = RadixTreeMiddleware(app, router=mock_router)
    middleware.tokenizer = mock_tokenizer

    # Create request
    request = create_request({"text": "Hello"})

    # Dispatch
    response = await middleware.dispatch(request, mock_call_next)

    # Verify cache insertion
    result = middleware.radix_tree.find_longest_prefix("Hello World")
    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.loss_mask == [0, 0, 0, 1, 1]
```

**运行**：
```bash
pytest tests/router/integration/ -v
```

### 3.4 E2E Tests

**目录**: `tests/router/e2e/`

**特点**：
- 真实 SGLang server
- 真实 Router 服务
- 完整 HTTP 请求流程

**标记**：
```python
import pytest

@pytest.mark.e2e
async def test_full_generate_workflow():
    """Test full generate workflow with real SGLang server."""
    # Requires SGLang server running at localhost:10090
    # ...
```

**运行**（需启动 SGLang server）：
```bash
# 启动 SGLang server (另一个终端)
python -m sglang.launch_server --model-path /path/to/model --port 10090

# 运行 E2E 测试
pytest tests/router/e2e/ -m "e2e" -v
```

### 3.5 TDD 工作流

**Test-Driven Development 变体流程**：

```
1. 写测试 (验证预期失败)
   └─> pytest tests/test_new_feature.py -v
       └─> ❌ FAIL (expected)

2. 修复代码
   └─> 实现功能逻辑

3. 运行测试 (验证通过)
   └─> pytest tests/test_new_feature.py -v
       └─> ✅ PASS (9/9 tests)

4. 重构代码
   └─> 优化实现,保持测试通过
```

**示例**（修复 Weight Version bug）：

```bash
# Step 1: 写测试
cat > tests/router/unit/test_weight_version_fix.py <<EOF
def test_traversed_nodes_update():
    """Test: Traversed nodes update weight_version."""
    tree = StringRadixTrie()
    tree.insert("Hello", [1, 2], [-0.1, -0.2], [0, 0], weight_version=1)
    tree.insert("Hello\nWorld", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 1], weight_version=5)

    # ❌ Expected FAIL: "Hello" node should have weight_version=5
    # (Before fix: weight_version=1)
EOF

# Step 2: 运行测试,验证失败
pytest tests/router/unit/test_weight_version_fix.py -v
# Output: FAILED (weight_version=1, expected 5)

# Step 3: 修复代码
# Edit slime/router/middleware_hub/radix_tree.py
# (Add: for node in traversed_nodes: node.weight_version = weight_version)

# Step 4: 再次运行测试,验证通过
pytest tests/router/unit/test_weight_version_fix.py -v
# Output: PASSED (1/1 tests)
```

### 3.6 测试覆盖率要求

**命令**：
```bash
pytest tests/router/ -m "not e2e" --cov=slime/router --cov-report=term-missing
```

**覆盖率目标**：
- **修改的核心文件**: >80% 覆盖率
- **新增功能**: 100% 覆盖率
- **整体项目**: >70% 覆盖率

**报告示例**：
```
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
slime/router/router.py                          120      8    93%   45-47, 102
slime/router/middleware_hub/radix_tree.py       350     15    96%   234-236, 456
slime/router/middleware_hub/radix_tree_middleware.py  180      5    97%   112-114
---------------------------------------------------------------------------
TOTAL                                           650     28    96%
```

---

## 4. 代码审查要点

### 4.1 关键检查项

**Async/Await 正确性**：
- [ ] 所有 I/O 操作使用 `await`
- [ ] 使用 `await asyncio.sleep()` 而非 `sleep()`
- [ ] Middleware 的 `dispatch()` 方法是 `async def`

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

### 4.2 已知问题修复记录

**Critical Issues（已修复）**：

| Issue | 优先级 | 修复版本 | 描述 |
|-------|-------|---------|------|
| Async sleep blocking | P0 | Phase 3.1 | `sleep(30)` → `await asyncio.sleep(30)` |
| Concurrency safety | P0 | Phase 3.2 | 添加 `asyncio.Lock` 保护 worker 选择 |
| Weight version update | P0 | Phase 3.3 | 更新所有 traversed nodes 的 `weight_version` |
| Tenacity retry 重构 | P1 | Phase 3.5 | 使用 `tenacity.AsyncRetrying` 替代手动循环 |
| /metrics API | P1 | Phase 3.6 | 添加监控 API |

**测试验证**：
```bash
# 运行所有修复验证测试
pytest tests/router/unit/test_radix_tree_core.py -v            # Weight version fix
pytest tests/router/integration/test_router_concurrency.py -v  # Concurrency fix
pytest tests/router/unit/test_tenacity_retry_logic.py -v       # Tenacity refactor
```

---

## 5. 故障排查

### 5.1 缓存未命中

**症状**: `/metrics` 显示 `hit_rate` 接近 0

**可能原因**：
1. Prompt 格式不一致（多余空格/换行）
2. 每次请求的 text 都不同，无公共前缀
3. GC 过于激进，频繁删除缓存

**排查步骤**：
```bash
# 1. 检查缓存统计
curl http://localhost:30000/metrics | jq '.cache'

# 2. 检查 Radix Tree 状态
# (In code: print(radix_tree.get_stats()))

# 3. 检查 prompt 格式
# 确保多轮对话的前缀一致
```

**解决方案**：
```python
# 标准化 prompt 格式
def normalize_prompt(text):
    # 统一换行符
    text = text.replace('\r\n', '\n')
    # 去除多余空格（根据需要）
    text = ' '.join(text.split())
    return text
```

### 5.2 内存持续增长

**症状**: Router 进程内存占用不断增长

**可能原因**：
- `max_cache_size` 设置过大
- GC 未正常触发
- Weight version 未正确传递

**排查步骤**：
```bash
# 1. 检查当前缓存大小
curl http://localhost:30000/metrics | jq '.cache.cur_cache_size'

# 2. 检查 weight_version 是否传递
# (需查看 SGLang response 格式)

# 3. 监控内存占用
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

### 5.3 负载不均衡

**症状**: 某个 worker 负载远高于其他 workers

**可能原因**：
- 并发请求时的 race condition（已在 Phase 3.2 修复）
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

### 5.4 Async Sleep 阻塞

**症状**: 单个 abort 请求导致所有请求被阻塞 30 秒

**根因**: 使用 `sleep(30)` 而非 `await asyncio.sleep(30)`

**验证**：
```python
# 检查代码中是否有同步 sleep
grep -r "sleep(" slime/router/middleware_hub/
# 应该只有 "asyncio.sleep"
```

**修复**（已在 Phase 3.1 完成）：
```python
# ❌ BEFORE
from time import sleep
sleep(30)

# ✅ AFTER
import asyncio
await asyncio.sleep(30)
```

---

## 6. 贡献流程

### 6.1 代码提交规范

**Commit Message 格式**：
```
<type>(<scope>): <subject>

<body>

<footer>
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

- Add weight_version change detection
- Trigger gc_by_weight_version() on version update
- Add tests in test_cache_invalidation.py

Closes #123
```

### 6.2 Pre-commit 检查

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
- Mypy 类型检查（可选）

### 6.3 Pull Request 流程

**1. Fork 仓库**：
```bash
git clone https://github.com/YOUR_USERNAME/slime.git
cd slime
git remote add upstream https://github.com/THUDM/slime.git
```

**2. 创建 feature 分支**：
```bash
git checkout -b feat/my-new-feature
```

**3. 编写代码 + 测试**：
```bash
# 编写代码
vim slime/router/middleware_hub/my_middleware.py

# 编写测试
vim tests/router/unit/test_my_middleware.py

# 运行测试
pytest tests/router/unit/test_my_middleware.py -v
```

**4. 运行 pre-commit 检查**：
```bash
pre-commit run --all-files
```

**5. 提交代码**：
```bash
git add .
git commit -m "feat(router): add my new middleware"
```

**6. 推送到 GitHub**：
```bash
git push origin feat/my-new-feature
```

**7. 创建 Pull Request**：
- 访问 https://github.com/THUDM/slime/pulls
- 点击 "New Pull Request"
- 选择 `feat/my-new-feature` 分支
- 填写 PR 描述（包括修改内容、测试结果、性能影响）

**8. 代码审查**：
- 等待 maintainer 审查
- 根据反馈修改代码
- 重新推送更新

---

## 7. 相关资源

### 内部文档
- **架构设计**: [architecture.md](architecture.md)
- **Radix Tree 详解**: [radix-tree.md](radix-tree.md)
- **用户手册**: [user-guide.md](user-guide.md)

### 外部资源
- **FastAPI 文档**: https://fastapi.tiangolo.com/
- **Starlette Middleware**: https://www.starlette.io/middleware/
- **Tenacity 文档**: https://tenacity.readthedocs.io/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/

### 代码位置
- **Router 实现**: `slime/router/router.py`
- **Radix Tree**: `slime/router/middleware_hub/radix_tree.py`
- **Middleware**: `slime/router/middleware_hub/radix_tree_middleware.py`
- **测试代码**: `tests/router/unit/`, `tests/router/integration/`

### 社区
- **GitHub Issues**: https://github.com/THUDM/slime/issues
- **Discussions**: https://github.com/THUDM/slime/discussions