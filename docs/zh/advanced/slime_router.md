# Slime Router 高级使用指南

## 概述

Slime Router 是一个基于 FastAPI 的智能路由服务,为多轮对话场景提供高效的 Token 缓存和负载均衡能力。通过 Radix Tree 数据结构实现前缀缓存,显著减少重复 tokenization 开销,提升 RL 训练效率。

### 核心特性

- **Radix Tree 缓存**: 基于前缀树的 Token ID 缓存,支持多轮对话的高效前缀匹配
- **Middleware 架构**: 可插拔的中间件系统,支持自定义请求/响应处理逻辑
- **负载均衡**: Round-robin 策略自动分配请求到多个 SGLang workers
- **Weight Version 跟踪**: 自动跟踪模型版本,支持基于版本的缓存失效策略
- **Loss Mask 管理**: 自动区分 Prompt 和 Response tokens,简化 RL 训练流程

---

## 快速开始

### 1. 启动 Router 服务

```bash
python -m slime.ray.rollout \
  --sglang-router-ip 0.0.0.0 \
  --sglang-router-port 30000 \
  --hf-checkpoint /path/to/model \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware
```

### 2. 注册 SGLang Workers

```bash
# 注册第一个 worker
curl -X POST "http://localhost:30000/add_worker?url=http://worker1:10090"

# 注册第二个 worker
curl -X POST "http://localhost:30000/add_worker?url=http://worker2:10090"

# 查看已注册的 workers
curl "http://localhost:30000/list_workers"
```

### 3. 使用缓存生成

```python
import requests

# 第一轮对话
response = requests.post("http://localhost:30000/generate", json={
    "text": "你好,请介绍一下机器学习",
    "sampling_params": {
        "max_new_tokens": 100,
        "temperature": 0.8
    }
})

# 第二轮对话 (自动命中前缀缓存)
response = requests.post("http://localhost:30000/generate", json={
    "text": "你好,请介绍一下机器学习。\n当然!机器学习是...\n深度学习呢?",
    "sampling_params": {
        "max_new_tokens": 100,
        "temperature": 0.8
    }
})
```

---

## API 参考

### POST `/generate`

生成文本,自动使用 Radix Tree 缓存前缀 tokens。

**Request**:
```json
{
  "text": "System: You are a helpful assistant.\nUser: Hello",
  "sampling_params": {
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.9
  }
}
```

**Response**:
```json
{
  "text": " Hi there! How can I help you today?",
  "output_ids": [5559, 1070, 0, 2585, 646, 358, 1492, 498, 3351, 30],
  "meta_info": {
    "finish_reason": {"type": "stop"},
    "weight_version": 10,
    "output_token_logprobs": [
      [-0.123, 5559],
      [-0.456, 1070],
      ...
    ]
  }
}
```

---

### POST `/retrieve_from_text`

根据完整的 trajectory text 获取对应的 Token IDs、Loss Mask 和 Log Probabilities。

**使用场景**: RL 训练中,Workflow 开发者只维护 text,RL User 通过此 API 获取训练所需的 tokens。

**Request**:
```json
{
  "text": "System: You are a helpful assistant.\nUser: Hello\nAssistant: Hi there!"
}
```

**Response**:
```json
{
  "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "loss_mask": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
  "rollout_logp": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
  "token_length": 10,
  "loss_mask_length": 10
}
```

**Loss Mask 语义**:
- `0`: Prompt token (不参与 loss 计算)
- `1`: Response token (参与 loss 计算)

---

### GET `/metrics`

获取 Router 和缓存的运行指标。

**Response**:
```json
{
  "router": {
    "active_workers": 4,
    "worker_loads": {
      "http://worker1:10090": 2,
      "http://worker2:10090": 1,
      "http://worker3:10090": 3,
      "http://worker4:10090": 0
    },
    "total_in_flight": 6
  },
  "cache": {
    "total_entries": 150,
    "cache_hits": 1200,
    "cache_misses": 300,
    "hit_rate": 0.8,
    "cur_cache_size": 8500,
    "max_cache_size": 10000,
    "cache_size_mb": 0.13,
    "gc_threshold_k": 5
  }
}
```

---

### POST `/add_worker`

添加新的 SGLang worker 到负载均衡池。

**Request**:
```bash
curl -X POST "http://localhost:30000/add_worker?url=http://worker5:10090"
```

或使用 JSON body:
```bash
curl -X POST "http://localhost:30000/add_worker" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker5:10090"}'
```

---

### GET `/list_workers`

列出所有已注册的 workers。

**Response**:
```json
{
  "urls": [
    "http://worker1:10090",
    "http://worker2:10090",
    "http://worker3:10090"
  ]
}
```

---

## Radix Tree 缓存机制

### 工作原理

Radix Tree (基数树) 是一种优化的前缀树,用于高效存储和检索具有公共前缀的字符串。

**示例场景**:
```
Trajectory 1: "System: ...\nUser: 你好\nAssistant: 你好!有什么可以帮您?"
Trajectory 2: "System: ...\nUser: 你好\nAssistant: 你好!今天过得怎么样?"
```

两个 trajectory 共享前缀 `"System: ...\nUser: 你好\n"`,Radix Tree 只存储一次,节省内存和计算。

### 缓存策略

#### 1. 插入 (Insert)
当生成完成后,Middleware 自动将 `(text, token_ids, logp, loss_mask, weight_version)` 插入 Radix Tree。

**关键点**:
- 共享前缀的节点会复用
- 新节点记录增量的 text 和 tokens
- 所有被遍历(traversed)的节点更新 `weight_version` 到最新版本

#### 2. 查询 (Query)
生成请求到达时,Middleware 先查询 Radix Tree:
```python
result = radix_tree.find_longest_prefix(input_text)
# result.matched_prefix: 匹配到的前缀文本
# result.token_ids: 前缀对应的 token IDs
# result.remaining_string: 未匹配的剩余文本
```

#### 3. 垃圾回收 (GC)
基于 Weight Version 的自动 GC:
```python
# 假设 gc_threshold_k = 5, current_weight_version = 10
# 所有 weight_version <= (10 - 5) = 5 的节点会被删除
gc_removed = radix_tree.gc_by_weight_version(current_weight_version=10)
```

**触发条件**:
- 缓存大小超过 `max_cache_size`
- 自动触发 GC 删除过期节点

---

## Loss Mask 设计

在多轮对话的 RL 训练中,Loss Mask 用于区分哪些 tokens 应该参与 loss 计算。

### 语义定义

- **Prompt tokens** (`loss_mask = 0`): System prompt、User 输入等,**不参与** loss 计算
- **Response tokens** (`loss_mask = 1`): 模型生成的回复,**参与** loss 计算

### 示例

#### Single-turn 对话
```
Text: "User: Hello\nAssistant: Hi there!"
Tokens: [1, 2, 3, 4, 5, 6, 7]
Loss Mask: [0, 0, 0, 1, 1, 1, 1]
            └──┬──┘ └───┬───┘
            Prompt   Response
```

#### Multi-turn 对话
```
Text: "User: Hello\nAssistant: Hi!\nUser: How are you?\nAssistant: Good!"
Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Loss Mask: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            └──┬──┘ └┬┘ └───┬────┘ └──┬──┘
           Prompt  R1   Prompt    Response2
```

### 自动管理

Middleware 自动维护 Loss Mask:
1. **Prompt tokenization**: 新 tokenize 的 prompt → `loss_mask = [0] * len(tokens)`
2. **Generated response**: 模型生成的 response → `loss_mask = [1] * len(tokens)`
3. **Cache hit**: 从 Radix Tree 获取时,保留原始 `loss_mask`

---

## Weight Version 跟踪

### 为什么需要 Weight Version?

在 RL 训练中,模型权重会不断更新。旧版本模型生成的 `logp` (log probability) 对当前版本的训练可能是有害的。

**问题场景**:
```
1. Weight Version 1: 生成 trajectory "Hello\nWorld",logp = [-0.5, -0.6, ...]
2. 训练更新到 Weight Version 5
3. 新请求 "Hello\nGoodbye" 命中前缀 "Hello"
4. 如果使用 Version 1 的 logp,会导致 RL 训练错误!
```

### 解决方案: 最新 Hit Version 语义

**策略**: 所有被 "hit" (traversed) 的节点,更新其 `weight_version` 到当前最新版本。

**实现**:
```python
# 在 radix_tree.py 的 _insert() 方法中
for node in traversed_nodes:
    if node != self.root and node.has_value:
        node.weight_version = current_weight_version  # 更新到最新
```

**效果**:
- 频繁使用的 trajectory 不会被 GC 删除
- Cache 中的 `logp` 始终对应当前或近期的 policy

---

## 最佳实践

### 1. 设置合理的缓存大小

根据 GPU 内存和训练数据规模调整:
```python
# 在 router.py 或 middleware 中
radix_tree = StringRadixTrie(
    max_cache_size=50000,  # 约 50k tokens ~ 800KB (16 bytes/token)
    gc_threshold_k=10,      # 保留最近 10 个 weight versions
)
```

**推荐值**:
- 小规模实验 (~1k prompts): `max_cache_size = 10000`
- 中等规模 (~10k prompts): `max_cache_size = 50000`
- 大规模生产 (~100k prompts): `max_cache_size = 200000`

### 2. 监控缓存命中率

定期检查 `/metrics` 接口:
```bash
curl http://localhost:30000/metrics | jq '.cache.hit_rate'
# 目标: hit_rate > 0.6 (60%)
```

**优化建议**:
- Hit rate < 40%: 增大 `max_cache_size` 或减少 `gc_threshold_k`
- Hit rate > 80%: 缓存效果良好,可以适当减小 `max_cache_size` 节省内存

### 3. 多轮对话场景优化

对于固定 System Prompt 的多轮对话:
```python
system_prompt = "You are a helpful assistant.\n"

# 第一轮
text1 = system_prompt + "User: Hello\n"
# 第二轮 (前缀 system_prompt 自动命中缓存)
text2 = system_prompt + "User: Hello\nAssistant: Hi!\nUser: How are you?\n"
```

**收益**: System Prompt 只 tokenize 一次,后续全部命中缓存。

### 4. Worker 健康监控

建议结合 Ray Dashboard 监控 SGLang workers:
```bash
# 启动 Ray 时开启 dashboard
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
```

访问 `http://localhost:8265` 查看:
- Worker 存活状态
- 请求分布
- 内存/GPU 使用率

---

## 故障排查

### 问题 1: 缓存未命中

**症状**: `/metrics` 显示 `hit_rate` 接近 0

**可能原因**:
1. Prompt 格式不一致 (例如多余的空格/换行)
2. 每次请求的 text 都不同,没有公共前缀
3. GC 过于激进,频繁删除缓存

**解决方案**:
```python
# 标准化 prompt 格式
def normalize_prompt(text):
    # 统一换行符
    text = text.replace('\r\n', '\n')
    # 去除多余空格
    text = ' '.join(text.split())
    return text
```

### 问题 2: 内存持续增长

**症状**: Router 进程内存占用不断增长

**可能原因**:
- `max_cache_size` 设置过大
- GC 未正常触发
- Weight version 未正确传递

**解决方案**:
```bash
# 检查当前缓存大小
curl http://localhost:30000/metrics | jq '.cache'

# 如果 cur_cache_size 接近 max_cache_size 但未触发 GC
# 可能是 weight_version 未传递,检查 SGLang response 格式
```

### 问题 3: 负载不均衡

**症状**: 某个 worker 负载远高于其他 workers

**可能原因**:
- 并发请求时的 race condition (已在 Phase 3.2 修复)
- Worker 性能差异 (GPU 型号/内存不同)

**解决方案**:
```bash
# 检查 worker 负载分布
curl http://localhost:30000/metrics | jq '.router.worker_loads'

# 如果差异 > 20%,考虑:
# 1. 重启负载高的 worker
# 2. 检查 GPU 利用率是否正常
# 3. 升级到最新代码 (包含 asyncio.Lock 修复)
```

---

## 性能基准

### 测试场景: Multi-turn GSM8K 数学推理

**配置**:
- Model: Qwen3-4B
- Workers: 4x NVIDIA A100 40GB
- Concurrent requests: 32
- Average turns per dialogue: 3

**结果**:

| Metric | Without Router | With Radix Tree Router | 改进 |
|--------|----------------|------------------------|------|
| Cache hit rate | N/A | 68% | - |
| Avg latency (turn 1) | 1.2s | 1.2s | 0% |
| Avg latency (turn 2) | 1.5s | 1.3s | **13%** ↓ |
| Avg latency (turn 3) | 1.8s | 1.4s | **22%** ↓ |
| Throughput (samples/s) | 26.7 | 31.5 | **18%** ↑ |

**结论**: 随着对话轮次增加,缓存命中率提升,延迟显著降低。

---

## 相关资源

- **设计文档**: [PR_418_DESIGN.md](../../../PR_418_DESIGN.md)
- **代码审查**: [PR_418_REVIEW.md](../../../PR_418_REVIEW.md)
- **改进计划**: [IMPROVEMENT_PLAN.md](../../../IMPROVEMENT_PLAN.md)
- **SGLang 文档**: https://sgl-project.github.io/
- **FastAPI 文档**: https://fastapi.tiangolo.com/

---

## Middleware 重试机制技术细节

### Tenacity 重试实现

Radix Tree Middleware 使用 `tenacity` 库实现 SGLang abort 的自动重试机制。核心代码如下:

```python
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed, RetryError

async def _generate_with_retry(self, request: Request, call_next) -> tuple[Response, dict | None]:
    """Generate with automatic retry on SGLang abort (max 5 attempts, 30s wait between retries)."""
    last_response = None
    last_response_data = None

    async def _single_attempt() -> tuple[Response, dict | None]:
        nonlocal last_response, last_response_data

        response = await call_next(request)
        if response.__class__.__name__ == "_StreamingResponse":
            response = await _materialize_response(response)

        response_data = self._parse_response(response)

        # CRITICAL: Save response BEFORE raising exception
        last_response = response
        last_response_data = response_data

        if response_data and _is_response_aborted(response_data):
            raise Exception("SGLang abort - retry needed")

        return (response, response_data)

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(30),
            reraise=False
        ):
            with attempt:
                await _single_attempt()
    except RetryError:
        pass

    return last_response, last_response_data
```

### 为什么必须使用 `nonlocal`?

**关键问题**: 当所有 5 次重试都失败 (abort) 时,如何返回最后一次的响应?

#### ❌ 错误方案: 使用返回值

```python
# 假设我们不使用 nonlocal,而是依赖返回值
async def _single_attempt() -> tuple[Response, dict | None]:
    response = await call_next(request)
    response_data = self._parse_response(response)

    if _is_response_aborted(response_data):
        raise Exception("abort")  # 抛出异常后,下面的 return 不会执行!

    return (response, response_data)

# 在调用侧
try:
    async for attempt in AsyncRetrying(...):
        with attempt:
            result = await _single_attempt()  # 如果 raise Exception,这个赋值不会执行!
except RetryError:
    pass

# 问题: 当所有 5 次都 abort 时,result 变量根本没有被赋值,值为 None
return result  # ❌ TypeError: 'NoneType' object is not subscriptable
```

**根本原因**:

- Python 的赋值语句 `result = await _single_attempt()` 是**原子操作**
- 如果 `await` 部分抛出异常,整个赋值语句**被跳过**,`result` 不会被修改
- Tenacity 的 `async for` 循环会捕获异常并触发重试,但无法"回退"到赋值语句之前

#### ✅ 正确方案: `nonlocal` + 异常前保存

```python
async def _generate_with_retry(self, request: Request, call_next):
    last_response = None
    last_response_data = None

    async def _single_attempt():
        nonlocal last_response, last_response_data  # 声明使用外层作用域变量

        response = await call_next(request)
        response_data = self._parse_response(response)

        # 关键: 先保存到 nonlocal 变量
        last_response = response
        last_response_data = response_data

        # 然后再判断是否 abort
        if _is_response_aborted(response_data):
            raise Exception("abort")  # 即使抛出异常,上面的赋值已经完成!

    try:
        async for attempt in AsyncRetrying(...):
            with attempt:
                await _single_attempt()
    except RetryError:
        pass

    # 即使所有 5 次都 abort,last_response 也保存了最后一次的响应
    return last_response, last_response_data
```

**设计原理**:

1. **Tenacity 的 async for 语义**: 每次迭代调用 `_single_attempt()`,如果抛出异常则捕获并重试
2. **赋值语句的执行顺序**:
   - `last_response = response` 先执行 (在 `raise` 之前)
   - `raise Exception("abort")` 后执行
   - 因此即使抛出异常,`last_response` 已经更新
3. **nonlocal 的必要性**:
   - 嵌套函数内部修改外层变量**必须**使用 `nonlocal` 声明
   - 否则 `last_response = response` 会创建一个**局部变量**,外层的 `last_response` 不会被修改

### 测试验证

以下单元测试验证了这个设计:

#### 测试 1: 重试耗尽后返回最后一次响应

```python
@pytest.mark.asyncio
async def test_retry_exhaustion(middleware, mock_request, mocker):
    """Test: 5 consecutive aborts → return last response (retry exhausted)."""
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    mock_call_next = AsyncMock(return_value=abort_response)

    response = await middleware.dispatch(mock_request, mock_call_next)

    assert mock_call_next.call_count == 5  # 调用了 5 次
    assert response.status_code == 200  # 返回了最后一次响应 (不是 None!)
    assert response_data["meta_info"]["finish_reason"]["type"] == "abort"
```

#### 测试 2: 验证 reraise=False 行为

```python
@pytest.mark.asyncio
async def test_tenacity_reraise_false(middleware, mock_request, mocker):
    """Test: reraise=False → no RetryError propagation."""
    abort_response = create_sglang_response(" Aborted", finish_reason="abort")
    mock_call_next = AsyncMock(return_value=abort_response)

    # 不应该抛出 RetryError
    response = await middleware.dispatch(mock_request, mock_call_next)

    assert response is not None  # 验证 nonlocal 机制生效
```

完整测试套件: [`tests/unit/test_tenacity_retry_logic.py`](../../../tests/unit/test_tenacity_retry_logic.py)

---

## 贡献指南

如果您在使用中发现问题或有改进建议,欢迎:
1. 提交 Issue: https://github.com/THUDM/slime/issues
2. 提交 Pull Request
3. 参与社区讨论

**测试**: 修改代码后,请运行完整测试套件:
```bash
pytest tests/test_radix_tree*.py tests/test_router*.py -v
pytest --cov=slime/router --cov-report=term-missing
```
