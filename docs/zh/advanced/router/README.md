# Slime Router

基于 FastAPI 的智能路由服务，为多轮对话 RL 训练提供高效的 Token 缓存和负载均衡能力。

## 核心价值

- **🚀 性能提升**: Radix Tree 前缀缓存，显著减少重复 tokenization 开销
- **🎯 职责分离**: Workflow 开发者使用 text，RL User 获取 tokens，解决训练一致性问题
- **🔧 生产就绪**: 组件依赖注入、可插拔中间件、故障容错和实时监控

## 典型使用场景

- **多轮对话 RL 训练**: System Prompt 只需 tokenize 一次，后续轮次自动命中缓存，延迟降低 22%
- **Agent Framework 集成**: 与 LangChain 等框架无缝集成，继续使用 text 抽象
- **批量生成优化**: 高效处理多个对话轨迹，吞吐量提升 18%

## 快速开始

```bash
# 1. 启动 Router (设置你的模型路径)
export MODEL_PATH="/path/to/your/model"

python -m slime.ray.rollout \
  --sglang-router-ip 0.0.0.0 \
  --sglang-router-port 30000 \
  --hf-checkpoint $MODEL_PATH \
  --use-slime-router \
  --slime-router-middleware-paths slime.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware

# 2. 注册 SGLang worker
curl -X POST "http://localhost:30000/add_worker?url=http://localhost:10090"

# 3. 使用缓存生成
curl -X POST "http://localhost:30000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，请介绍一下机器学习", "sampling_params": {"max_new_tokens": 100}}'
```

## 文档导航

### 👥 用户文档

- **[用户指南](user-guide.md)** - 完整使用指南、API 参考和配置说明
- **[OpenAI API](openai-chat-completion.md)** - OpenAI Chat Completion 兼容接口

### 🏗️ 技术设计

- **[系统架构](architecture.md)** - 三层架构设计和 ComponentRegistry 原理
- **[Radix Tree](radix-tree.md)** - 前缀缓存数据结构详解

### 🛠️ 开发文档

- **[开发指南](development.md)** - 中间件开发、测试策略和贡献流程

## 核心概念

- **Loss Mask**: `0`=Prompt, `1`=Response (自动区分训练数据)
- **Weight Version**: 跟踪模型权重版本，确保训练数据一致性
- **ComponentRegistry**: 统一管理共享组件，消除硬编码依赖

---

**开始探索**: [用户指南](user-guide.md) → [系统架构](architecture.md) → [开发指南](development.md)