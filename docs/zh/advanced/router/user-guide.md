# Slime Router 用户指南

## 概述

Slime Router 是一个基于 FastAPI 的智能路由服务，为多轮对话场景提供高效的 Token 缓存和负载均衡能力。通过 Radix Tree 数据结构实现前缀缓存，显著减少重复 tokenization 开销，提升 RL 训练效率。

### 核心特性

- **Radix Tree 缓存**: 基于前缀树的 Token ID 缓存，支持多轮对话的高效前缀匹配
- **Middleware 架构**: 可插拔的中间件系统，支持自定义请求/响应处理逻辑
- **负载均衡**: Round-robin 策略自动分配请求到多个 SGLang workers
- **Weight Version 跟踪**: 自动跟踪模型版本，支持基于版本的缓存失效策略
- **Loss Mask 管理**: 自动区分 Prompt 和 Response tokens，简化 RL 训练流程

---

## 快速开始

### 1. 启动 Router 服务

```bash
# 基础启动命令
python -m slime.ray.rollout \
  --sglang-router-ip 0.0.0.0 \
  --sglang-router-port 30000 \
  --hf-checkpoint /path/to/model \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware

# 完整启动命令（包含所有可选参数）
python -m slime.ray.rollout \
  --sglang-router-ip 0.0.0.0 \
  --sglang-router-port 30000 \
  --hf-checkpoint /path/to/model \
  --radix-tree-max-size 10000 \
  --verbose \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware
```

### 2. 启动参数详解

#### 必需参数

- **`--hf-checkpoint`**: HuggingFace 模型检查点路径
  - 用途：指定用于 tokenizer 初始化的模型路径
  - 示例：`--hf-checkpoint /models/Qwen3-0.6B`
  - 验证：启动时会检查路径是否存在，缺失则立即报错

#### 可选参数

- **`--radix-tree-max-size`**: Radix Tree 最大缓存大小
  - 默认值：`10000`
  - 用途：控制缓存的最大 token 数量
  - 内存估算：约 16 bytes/token，10K tokens ≈ 160KB
  - 推荐设置：
    - 小规模实验：`10000` (默认)
    - 中等规模：`50000` (≈ 800KB)
    - 大规模生产：`200000` (≈ 3MB)

- **`--verbose`**: 启用详细日志输出
  - 默认值：`False`
  - 用途：调试和性能分析时启用
  - 输出：缓存命中率、请求处理时间等详细信息

#### 其他 Router 参数

- **`--sglang-router-ip`**: Router 服务监听 IP（默认：`0.0.0.0`）
- **`--sglang-router-port`**: Router 服务端口（默认：`30000`）
- **`--use-slime-router`**: 启用 Slime Router 功能
- **`--slime-router-middleware-paths`**: 中间件模块路径，多个路径用逗号分隔

### 3. 注册 SGLang Workers

```bash
# 注册第一个 worker
curl -X POST "http://localhost:30000/add_worker?url=http://worker1:10090"

# 注册第二个 worker
curl -X POST "http://localhost:30000/add_worker?url=http://worker2:10090"

# 查看已注册的 workers
curl "http://localhost:30000/list_workers"
```

### 4. 使用缓存生成

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

## ComponentRegistry 架构

### 什么是 ComponentRegistry？

ComponentRegistry 是 Slime Router 的核心组件管理系统，提供统一的组件注册和获取机制，消除硬编码依赖。

### 核心优势

- **零硬编码**: 所有组件通过配置驱动，启动时验证依赖完整性
- **快速失败**: 缺失组件会立即报错，避免运行时才发现问题
- **统一管理**: 集中管理 tokenizer、radix_tree 等共享组件
- **易于扩展**: 新组件只需注册即可使用

### 自动组件注册

当启用 `RadixTreeMiddleware` 时，以下组件会自动注册：

```python
# 自动注册的组件
router.component_registry.register("tokenizer", tokenizer)           # HuggingFace tokenizer
router.component_registry.register("radix_tree", radix_tree)         # Radix Tree 缓存
```

### 使用已注册组件

```python
# 获取 tokenizer
tokenizer = router.component_registry.get("tokenizer")

# 获取 radix tree
radix_tree = router.component_registry.get("radix_tree")
```

### 错误处理

```python
# 组件缺失时的错误信息
RuntimeError: Required component 'tokenizer' not found.
Available components: ['radix_tree']
```

### 开发者自定义组件

```python
# 在自定义 middleware 中注册组件
class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)

        # 创建自定义组件
        self.metrics = MetricsCollector()

        # 注册到全局注册表
        router.component_registry.register("metrics", self.metrics)
```

---

## API 参考

### POST `/generate`

生成文本，自动使用 Radix Tree 缓存前缀 tokens。

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

**使用场景**: RL 训练中，Workflow 开发者只维护 text，RL User 通过此 API 获取训练所需的 tokens。

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

### 为什么需要 `/retrieve_from_text`？

**职责分离**：

- **Workflow 开发者**：只关心 messages 抽象（text in/out），不需要手动维护 token IDs
- **RL User**：在 rollout 结束后，通过此 API 获取训练所需的 tokens、loss_mask、logp

**性能开销**：

- 额外 HTTP 调用开销：~1-5ms
- 生成延迟：秒级
- 结论：开销相比生成延迟可接受（<0.5%），换取清晰的抽象分层

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

## 缓存机制概览

### Radix Tree 缓存

Radix Tree (基数树) 是一种优化的前缀树，用于高效存储和检索具有公共前缀的字符串。

**示例场景**:
```
Trajectory 1: "System: ...\nUser: 你好\nAssistant: 你好!有什么可以帮您?"
Trajectory 2: "System: ...\nUser: 你好\nAssistant: 你好!今天过得怎么样?"
```

两个 trajectory 共享前缀 `"System: ...\nUser: 你好\n"`，Radix Tree 只存储一次，节省内存和计算。

### 缓存策略

1. **插入 (Insert)**: 当生成完成后，Middleware 自动将 `(text, token_ids, logp, loss_mask, weight_version)` 插入 Radix Tree
2. **查询 (Query)**: 生成请求到达时，Middleware 先查询 Radix Tree 找到最长匹配前缀
3. **垃圾回收 (GC)**: 基于 Weight Version 的自动 GC，删除过期的缓存节点

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
- Hit rate > 80%: 缓存效果良好，可以适当减小 `max_cache_size` 节省内存

### 3. 多轮对话场景优化

对于固定 System Prompt 的多轮对话:
```python
system_prompt = "You are a helpful assistant.\n"

# 第一轮
text1 = system_prompt + "User: Hello\n"
# 第二轮 (前缀 system_prompt 自动命中缓存)
text2 = system_prompt + "User: Hello\nAssistant: Hi!\nUser: How are you?\n"
```

**收益**: System Prompt 只 tokenize 一次，后续全部命中缓存。

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

## 故障排查快速指南

### 问题 1: 缓存未命中

**症状**: `/metrics` 显示 `hit_rate` 接近 0

**可能原因**:
1. Prompt 格式不一致 (例如多余的空格/换行)
2. 每次请求的 text 都不同，没有公共前缀
3. GC 过于激进，频繁删除缓存

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
# 可能是 weight_version 未传递，检查 SGLang response 格式
```

### 问题 3: 负载不均衡

**症状**: 某个 worker 负载远高于其他 workers

**可能原因**:
- 并发请求时的 race condition (已在最新代码修复)
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

**环境配置**:

- Model: Qwen3-4B
- Workers: 4x NVIDIA A100 40GB
- SGLang version: v0.5.2
- Concurrent requests: 32
- Test dataset: GSM8K (1000 samples)
- Average turns per dialogue: 3

**缓存效果分析**:

| Turn | Avg Input Tokens | Cache Hit Rate | Tokenization Time | 改进 |
|------|-----------------|----------------|-------------------|------|
| 1 | 100 | 0% | 12ms | - |
| 2 | 150 | 68% | 4ms | **67%** ↓ |
| 3 | 200 | 75% | 3ms | **75%** ↓ |

**端到端延迟**:

| Metric | Turn 1 | Turn 2 | Turn 3 |
|--------|--------|--------|--------|
| Without Router | 1.2s | 1.5s | 1.8s |
| With Router | 1.2s | 1.3s | 1.4s |
| 改进 | 0% | **13%** ↓ | **22%** ↓ |

**吞吐量**:

- Without Router: 26.7 samples/s
- With Router: 31.5 samples/s
- 改进: **18%** ↑

**结论**: 随着对话轮次增加，缓存命中率提升，延迟显著降低（Turn 3 达到 22% 改进）。

---

## 相关资源

### 技术文档
- **[系统架构](architecture.md)** - 详细的三层架构设计
- **[Radix Tree](radix-tree.md)** - 数据结构和算法实现
- **[开发指南](development.md)** - 中间件开发和测试策略

### 外部资源
- **SGLang 文档**: https://sgl-project.github.io/
- **FastAPI 文档**: https://fastapi.tiangolo.com/