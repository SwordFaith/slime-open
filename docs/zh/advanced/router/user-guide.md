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
  --slime-router-middleware-paths slime.router.middleware.radix_tree_middleware.RadixTreeMiddleware

# 完整启动命令（包含所有可选参数）
python -m slime.ray.rollout \
  --sglang-router-ip 0.0.0.0 \
  --sglang-router-port 30000 \
  --hf-checkpoint /path/to/model \
  --radix-tree-max-size 10000 \
  --verbose \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware.radix_tree_middleware.RadixTreeMiddleware
```

### 2. 启动参数详解

#### 必需参数

- **`--hf-checkpoint`**: HuggingFace 模型检查点路径
  - 用途：指定用于 tokenizer 初始化的模型路径
  - 示例：`--hf-checkpoint /models/Qwen3-0.6B`
  - 验证：启动时会检查路径是否存在，缺失则立即报错
  - **注意**: 所有新功能都需要此参数，ComponentRegistry 会在启动时验证

#### OpenAI Chat Completion API 参数

- **`--enable-openai-chat-completion`**: 启用 OpenAI Chat Completion API 支持
  - 默认值：`False`
  - 用途：启用 `/v1/chat/completions` 接口，100% 兼容 OpenAI API
  - 依赖：需要 `--hf-checkpoint` 参数用于 tokenizer 初始化
  - 用法：启用后可直接使用 OpenAI SDK 连接到 `http://localhost:30000/v1`

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
  - 输出：缓存命中率、请求处理时间、ComponentRegistry 状态等详细信息

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

## OpenAI Chat Completion API 使用指南

### 概述

Slime Router 提供了 100% 兼容 OpenAI Chat Completion API 的接口，让您可以无缝替换 OpenAI API endpoint，同时享受 Radix Tree 缓存带来的性能提升。

### 启用 OpenAI API

在启动 Router 时添加 `--enable-openai-chat-completion` 参数：

```bash
python -m slime.ray.rollout \
  --hf-checkpoint /path/to/model \
  --enable-openai-chat-completion \
  --slime-router-middleware-paths slime.router.middleware.radix_tree_middleware.RadixTreeMiddleware \
  [其他参数...]
```

### 基本使用

#### Python SDK 使用

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    api_key="dummy-key",  # 可为任意值
    base_url="http://localhost:30000/v1"
)

# 非流式对话
response = client.chat.completions.create(
    model="slime-model",  # 可为任意值
    messages=[
        {"role": "system", "content": "你是一个专业的助手"},
        {"role": "user", "content": "请介绍一下机器学习"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"Token usage: {response.usage}")
```

#### 流式响应

```python
# 流式对话
stream = client.chat.completions.create(
    model="slime-model",
    messages=[
        {"role": "user", "content": "请写一个关于机器学习的故事"}
    ],
    stream=True,
    temperature=0.8
)

# 实时输出
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### cURL 使用

```bash
curl -X POST "http://localhost:30000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slime-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 支持的参数

OpenAI Chat Completion API 支持所有标准参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | 必需 | 模型名称（可为任意值） |
| `messages` | array | 必需 | 对话消息列表 |
| `temperature` | float | 1.0 | 采样温度，0.0-2.0 |
| `top_p` | float | 1.0 | 核采样概率，0.0-1.0 |
| `max_tokens` | integer | 无限制 | 最大生成 token 数 |
| `stream` | boolean | false | 是否启用流式响应 |
| `stop` | string/array | null | 停止词 |
| `presence_penalty` | float | 0.0 | 存在惩罚，-2.0-2.0 |
| `frequency_penalty` | float | 0.0 | 频率惩罚，-2.0-2.0 |
| `user` | string | null | 用户标识 |

### 缓存机制

OpenAI API 自动利用 Radix Tree 缓存机制：

- **智能前缀匹配**: 基于 HuggingFace chat template 格式化文本
- **多轮对话优化**: System prompt 和对话历史自动命中缓存
- **透明缓存**: 用户无需感知缓存机制，自动获得性能提升

#### 缓存示例

```python
# 第一轮对话（缓存 system prompt）
messages1 = [
    {"role": "system", "content": "你是一个专业的机器学习助手"},
    {"role": "user", "content": "请介绍一下监督学习"}
]

# 第二轮对话（system prompt 命中缓存）
messages2 = [
    {"role": "system", "content": "你是一个专业的机器学习助手"},
    {"role": "user", "content": "请介绍一下监督学习"},
    {"role": "assistant", "content": "监督学习是..."},
    {"role": "user", "content": "那非监督学习呢？"}
]

# 第二轮对话的 system prompt + 第一轮对话会自动命中缓存
response2 = client.chat.completions.create(model="slime-model", messages=messages2)
```

### 性能优势

相比直接使用 OpenAI API，Slime Router 的优势：

| 场景 | OpenAI API | Slime Router | 改进 |
|------|------------|--------------|------|
| 首次对话 | 100% 延迟 | 100% 延迟 | 相同 |
| 多轮对话 (Turn 2) | 100% 延迟 | 67% 延迟 | **33%** ↑ |
| 多轮对话 (Turn 3+) | 100% 延迟 | 25% 延迟 | **75%** ↑ |
| 成本 | 100% | 100% | 相同 |

### 错误处理

```python
try:
    response = client.chat.completions.create(
        model="slime-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except openai.APIConnectionError as e:
    print(f"连接错误: {e}")
except openai.RateLimitError as e:
    print(f"速率限制: {e}")
except openai.APIError as e:
    print(f"API错误: {e}")
```

### 最佳实践

1. **多轮对话**: 尽量保持一致的 system prompt 以获得最佳缓存效果
2. **流式响应**: 对于生成长文本的场景，建议启用流式响应
3. **温度设置**: 根据应用场景调整 temperature，创造性任务用较高值，事实性任务用较低值
4. **监控缓存**: 使用 `/metrics` 接口监控缓存命中率

### 与现有应用的集成

#### LangChain 集成

```python
from langchain_openai import ChatOpenAI

# 配置 LangChain 使用 Slime Router
llm = ChatOpenAI(
    model="slime-model",
    openai_api_key="dummy-key",
    openai_api_base="http://localhost:30000/v1",
    temperature=0.7
)

# 正常使用 LangChain
from langchain.schema import HumanMessage

response = llm.invoke([HumanMessage(content="介绍一下机器学习")])
print(response.content)
```

#### 其他框架

任何支持 OpenAI API 的框架都可以通过修改 `base_url` 参数无缝切换到 Slime Router：

```python
# 通用配置
OPENAI_API_BASE = "http://localhost:30000/v1"
OPENAI_API_KEY = "dummy-key"  # 可为任意值
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

**关键特性**：
- **零硬编码**: 所有组件通过参数驱动，消除代码中的硬编码路径
- **启动验证**: 启动时检查依赖完整性，缺失组件立即报错
- **快速失败**: 运行时错误提前到启动时发现

### 使用已注册组件

```python
# 获取 tokenizer
tokenizer = router.component_registry.get("tokenizer")

# 获取 radix tree
radix_tree = router.component_registry.get("radix_tree")
```

### 配置验证

ComponentRegistry 会在启动时验证所有必需的配置：

```bash
# 缺失 --hf-checkpoint 参数时的错误信息
Error: Missing required argument: --hf-checkpoint
This parameter is required for ComponentRegistry initialization.
```

### 错误处理

```python
# 组件缺失时的错误信息
RuntimeError: Required component 'tokenizer' not found.
Available components: ['radix_tree']
```

**常见错误及解决方案**：

1. **Missing `--hf-checkpoint`**
   ```bash
   # 添加必需参数
   --hf-checkpoint /path/to/your/model
   ```

2. **Component not found**
   ```python
   # 检查组件是否正确注册
   available = router.component_registry._components.keys()
   print(f"Available components: {list(available)}")
   ```

3. **Invalid middleware path**
   ```bash
   # 确保中间件路径正确
   --slime-router-middleware-paths slime.router.middleware.radix_tree_middleware.RadixTreeMiddleware
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

    async def dispatch(self, request, call_next):
        # 使用已注册的组件
        tokenizer = router.component_registry.get("tokenizer")

        # 自定义逻辑
        start_time = time.time()
        response = await call_next(request)

        # 记录指标
        self.metrics.record_request_time(time.time() - start_time)
        return response
```

### 组件使用最佳实践

1. **命名规范**: 使用描述性的组件名称，如 `tokenizer`、`radix_tree`、`metrics`
2. **依赖检查**: 在使用组件前检查是否存在，提供友好的错误信息
3. **生命周期管理**: 组件应该在整个 Router 生命周期内保持有效
4. **线程安全**: 确保注册的组件在并发环境下是安全的

### 组件注册时机

```python
# RadixTreeMiddleware 的 __init__ 方法中注册
class RadixTreeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)

        # 验证必需参数
        if not hasattr(router.args, 'hf_checkpoint') or not router.args.hf_checkpoint:
            raise ValueError("Missing required argument: --hf-checkpoint")

        # 创建并注册组件
        tokenizer = AutoTokenizer.from_pretrained(router.args.hf_checkpoint)
        router.component_registry.register("tokenizer", tokenizer)

        radix_tree = StringRadixTrie(
            max_cache_size=router.args.radix_tree_max_size,
            verbose=router.args.verbose
        )
        router.component_registry.register("radix_tree", radix_tree)
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