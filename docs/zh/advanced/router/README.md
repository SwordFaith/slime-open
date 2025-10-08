# Slime Router

一个基于 FastAPI 的智能路由服务，为多轮对话场景提供高效的 Token 缓存和负载均衡能力。

## 核心价值

### 🚀 性能提升
- **Radix Tree 缓存**: 前缀缓存机制，减少重复 tokenization 开销
- **负载均衡**: Round-robin 策略自动分配请求到多个 SGLang workers
- **并发优化**: 异步处理，支持高并发请求

### 🎯 职责分离
- **Workflow 开发者**: 继续使用 text in/out，无需关心 token 细节
- **RL User**: 通过 API 获取精确的 token IDs、loss mask、log probabilities
- **训练一致性**: 解决 tokenization 不可逆问题，保证训练数据正确性

### 🔧 生产就绪
- **Middleware 架构**: 可插拔中间件系统，支持自定义扩展
- **故障容错**: 自动重试机制，处理 SGLang abort 场景
- **监控指标**: 实时缓存命中率和负载状态监控

## 典型使用场景

### 多轮对话 RL 训练
```
System: You are a helpful assistant.
User: 你好
Assistant: 你好！有什么可以帮您？
User: 推荐一本机器学习的书
Assistant: 我推荐《Pattern Recognition and Machine Learning》...
```

**收益**: System Prompt 只 tokenize 一次，后续轮次自动命中缓存。

### Agent Framework 集成
- **LangChain**: 继续使用 text 抽象
- **自定义 Agent**: 通过 `/retrieve_from_text` 获取训练数据
- **批量生成**: 高效处理多个对话轨迹

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
# 注册 worker
curl -X POST "http://localhost:30000/add_worker?url=http://worker1:10090"

# 查看已注册 workers
curl "http://localhost:30000/list_workers"
```

### 3. 使用缓存生成
```python
import requests

# 第一轮对话
response = requests.post("http://localhost:30000/generate", json={
    "text": "你好,请介绍一下机器学习",
    "sampling_params": {"max_new_tokens": 100, "temperature": 0.8}
})

# 第二轮对话 (自动命中前缀缓存)
response = requests.post("http://localhost:30000/generate", json={
    "text": "你好,请介绍一下机器学习。\n当然!机器学习是...\n深度学习呢?",
    "sampling_params": {"max_new_tokens": 100, "temperature": 0.8}
})
```

## 文档导航

### 👥 用户文档
- **[用户指南](user-guide.md)** - 完整的使用指南和 API 参考
- **[最佳实践](user-guide.md#最佳实践)** - 生产环境配置建议

### 🏗️ 技术设计
- **[系统架构](architecture.md)** - 三层架构设计和关键技术决策
- **[Radix Tree](radix-tree.md)** - 前缀缓存数据结构详解

### 🛠️ 开发文档
- **[开发指南](development.md)** - 环境搭建、中间件开发、测试策略
- **[故障排查](development.md#故障排查)** - 常见问题和解决方案

## 性能收益

### 异步并发优化 (2025-10-08)
- **并发读取延迟降低**: 99.1%
- **系统吞吐量提升**: 超过 100 倍
- **事件循环阻塞**: 完全消除
- **向后兼容性**: 100% 保持

### 多轮对话场景 (GSM8K 测试)
- **Turn 3 缓存命中率**: 75%
- **端到端延迟降低**: 22%
- **吞吐量提升**: 18%

### 内存占用
- **10K tokens**: ~210 KB
- **100K tokens**: ~2 MB
- **开销**: 可忽略不计

*详细的异步优化技术细节请参考 [架构文档](architecture.md#42-radix-tree-异步并发优化)*

## 核心概念

### Loss Mask 语义
- `0`: Prompt token (不参与 loss 计算)
- `1`: Response token (参与 loss 计算)

### Weight Version 跟踪
自动跟踪模型权重版本，确保 RL 训练使用当前或近期的 policy logp。

### API 概览
- `POST /generate` - 生成文本，自动使用缓存
- `POST /retrieve_from_text` - 根据 text 获取 tokens 和训练数据
- `GET /metrics` - 监控缓存和负载状态
- `POST /add_worker` - 添加 SGLang worker

---

开始探索: [用户指南](user-guide.md) → [系统架构](architecture.md) → [开发指南](development.md)