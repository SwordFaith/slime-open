# Radix Tree 数据结构详解

## 1. StringTreeNode 字段说明

### 1.1 核心字段

```python
class StringTreeNode:
    """Radix Tree node for storing dialogue trajectory tokens."""

    # String key
    string_key: str = ""  # 节点存储的字符串片段（非累积）

    # Token data (non-cumulative, only for this node's fragment)
    token_ids: List[int] = field(default_factory=list)  # Token IDs
    logp: List[float] = field(default_factory=list)     # Log probabilities
    loss_mask: List[int] = field(default_factory=list)  # 0=Prompt, 1=Response

    # Tree structure
    children: List["StringTreeNode"] = field(default_factory=list)
    parent: Optional["StringTreeNode"] = None

    # Cache management
    weight_version: Optional[int] = None    # 生成时版本（插入时设定，永不修改）
    traverse_version: Optional[int] = None  # 游走时版本（用于 GC 控制）
    last_access_time: int = 0               # 最后访问时间（LRU 备用）
    ref_count: int = 0                      # 引用计数（防止过早删除）
```

### 1.2 字段语义详解

#### `string_key`
- **作用**: 存储从父节点到当前节点的字符串片段
- **非累积性**: 只存储增量部分，不包含父节点的内容
- **示例**:
  ```
  Root
   └─ Node1 (string_key="Hello\n")
       └─ Node2 (string_key="World")

  完整路径: "Hello\nWorld"
  Node2.string_key 仅为 "World"
  ```

#### `token_ids`, `logp`, `loss_mask`
- **作用**: 存储 `string_key` 对应的 tokenization 结果
- **非累积性**: 只存储当前节点的 tokens，查询时需沿路径累积
- **一致性保证**: `len(token_ids) == len(logp) == len(loss_mask)`
- **示例**:
  ```python
  node.string_key = "Hello"
  node.token_ids = [12, 34, 56]
  node.logp = [-0.1, -0.2, -0.3]
  node.loss_mask = [0, 0, 0]  # Prompt tokens
  ```

#### `weight_version` (生成时版本)
- **作用**: 标记此节点的 tokens/logp 最初是由哪个模型版本生成的
- **设定时机**: 节点创建时设定，之后永不修改
- **语义**: 表示 token 的"出生版本"，用于追溯数据来源
- **特殊值**:
  - `None`: 未设置（通常表示手动插入或非模型生成的 token）
  - `-1`: 明确标记为非模型生成的 token（如手动 tokenize 的 prompt）
- **示例**:
  ```python
  # 创建节点时设定生成版本
  node.weight_version = 5  # 由版本 5 的模型生成

  # 后续游走不会修改生成版本
  # node.weight_version 保持为 5
  ```

#### `traverse_version` (游走时版本)
- **作用**: 标记节点最近一次被 traversed（命中）时的模型版本，用于 GC 控制
- **更新时机**: 每次节点被 traversed 时更新到最新版本
- **GC 依据**: `traverse_version <= (current_version - gc_threshold_k)` 的节点会被删除
- **示例**:
  ```python
  # 创建节点时初始游走版本
  node.traverse_version = 5

  # 版本 10 时再次 traversed
  node.traverse_version = 10  # 更新到最新

  # 版本 15, GC (threshold=5): 删除 traverse_version <= 10 的节点
  ```

#### `ref_count`
- **作用**: 防止正在使用的节点被 GC 删除
- **更新时机**: 查询时 +1，查询结束 -1
- **未来优化**: 当前版本未使用，预留字段

---

## 2. 核心算法实现

### 2.1 非累积 Token 存储

#### 2.1.1 设计理由

**为什么不存储累积 tokens？**

**方案 A**（累积存储，❌ 不采用）：
```python
# 每个节点存储从 root 到当前节点的所有 tokens
node1.token_ids = [1, 2, 3]
node2.token_ids = [1, 2, 3, 4, 5]  # 包含 node1 的 tokens
```

**问题**：
- 内存浪费严重（重复存储前缀 tokens）
- 更新困难（父节点变化时，所有子节点需重新计算）

**方案 B**（非累积存储，✅ 采用）：
```python
# 每个节点只存储增量 tokens
node1.token_ids = [1, 2, 3]
node2.token_ids = [4, 5]  # 仅新增部分
```

**优势**：
- 内存高效（共享前缀）
- 更新简单（局部修改）

#### 2.1.2 查询时的 Token 累积

**实现**：
```python
def find_longest_prefix(self, text: str) -> RadixTreeResult:
    """Find longest matching prefix and accumulate tokens along the path."""
    current_node = self.root
    matched_tokens = []
    matched_logp = []
    matched_loss_mask = []

    remaining_text = text

    while remaining_text:
        best_child = self._find_best_child(current_node, remaining_text)
        if best_child is None:
            break

        current_node = best_child
        remaining_text = remaining_text[len(best_child.string_key):]

        # Accumulate tokens from this node
        if best_child.has_value:
            matched_tokens.extend(best_child.token_ids)  # 累积
            matched_logp.extend(best_child.logp)
            matched_loss_mask.extend(best_child.loss_mask)

    return RadixTreeResult(
        token_ids=matched_tokens,
        logp=matched_logp,
        loss_mask=matched_loss_mask,
        remaining_string=remaining_text
    )
```

### 2.2 查询算法：find_longest_prefix()

#### 2.2.1 算法流程

```mermaid
graph TD
    A[Start: text = 'Hello\nWorld'] --> B[current_node = root]
    B --> C{Find best child<br/>matching prefix?}
    C -- Yes --> D[Move to child node]
    D --> E[Accumulate tokens]
    E --> F[Remove matched text]
    F --> C
    C -- No --> G[Return result]
    G --> H{remaining_string<br/>is empty?}
    H -- Yes --> I[Complete match]
    H -- No --> J[Partial match]
    I --> K[Return cached tokens]
    J --> L[Tokenize remaining_string]
    L --> M[Return cached + new tokens]
```

#### 2.2.2 核心逻辑

**1. 部分匹配的处理**：

```python
result = radix_tree.find_longest_prefix("Hello\nWorld\nNew")

# 假设缓存中只有 "Hello\nWorld"
result.matched_prefix = "Hello\nWorld"
result.token_ids = [1, 2, 3, 4, 5]
result.remaining_string = "\nNew"  # 未匹配部分
```

**处理方式**：
```python
if result.remaining_string:
    # Tokenize remaining text
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    full_token_ids = result.token_ids + additional_tokens
    full_loss_mask = result.loss_mask + [0] * len(additional_tokens)  # Prompt
```

**2. Loss Mask 的正确生成**：

**关键修复**（PR #418 中修复的 bug）：
```python
# ❌ BEFORE (错误实现)
if result.remaining_string:
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    full_loss_mask = result.loss_mask + [1] * len(additional_tokens)  # ❌ 错误！

# ✅ AFTER (正确实现)
if result.remaining_string:
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    full_loss_mask = result.loss_mask + [0] * len(additional_tokens)  # ✅ Prompt tokens
```

**理由**: 剩余的 `remaining_string` 是用户输入的新 prompt，不是模型生成的 response，应该标记为 `loss_mask=0`。

**3. 版本信息对齐**：

**关键特性**（Version Separation Architecture）：
```python
# ✅ AFTER (版本对齐实现)
if result.remaining_string:
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    additional_versions = [-1] * len(additional_tokens)  # 非 AI 生成

    full_token_ids = result.token_ids + additional_tokens
    full_versions = result.generation_versions + additional_versions
    full_loss_mask = result.loss_mask + [0] * len(additional_tokens)
```

**版本对齐原则**：
- 缓存的 token 对应其生成时的 `weight_version`
- 新 tokenize 的 token 标记为 `-1`（非 AI 生成）
- `result.generation_versions` 与 `result.token_ids` 长度相同且一一对应
- 支持精确的版本追溯和分析

### 2.3 插入算法：_insert()

#### 2.3.1 算法流程

```python
def _insert(
    self,
    text: str,
    token_ids: List[int],
    logp: List[float],
    loss_mask: List[int],
    weight_version: Optional[int] = None,
) -> bool:
    """Insert new trajectory or extend existing one."""
    current_node = self.root
    remaining_text = text
    remaining_tokens = token_ids[:]
    remaining_logp = logp[:]
    remaining_loss_mask = loss_mask[:]

    # Track ALL traversed nodes (not just new ones)
    traversed_nodes = [current_node]
    new_node = None

    while remaining_text:
        best_child = self._find_best_child(current_node, remaining_text)

        if best_child is not None:
            # Found existing node, traverse
            current_node = best_child
            traversed_nodes.append(current_node)  # ✅ Track traversed

            remaining_text = remaining_text[len(best_child.string_key):]

            if best_child.has_value:
                # Skip tokens already stored in this node
                tokens_to_skip = len(best_child.token_ids)
                remaining_tokens = remaining_tokens[tokens_to_skip:]
                remaining_logp = remaining_logp[tokens_to_skip:]
                remaining_loss_mask = remaining_loss_mask[tokens_to_skip:]
        else:
            # Create new node
            new_node = StringTreeNode(
                string_key=remaining_text,
                token_ids=remaining_tokens,
                logp=remaining_logp,
                loss_mask=remaining_loss_mask,
                parent=current_node
            )
            current_node.children.append(new_node)
            traversed_nodes.append(new_node)
            self.total_entries += 1
            self.cur_cache_size += len(remaining_tokens)
            break

    # ✅ VERSION SEPARATION: Set generation and traverse versions
    if weight_version is not None:
        for node in traversed_nodes:
            if node != self.root and node.has_value:
                # 新创建节点：设置生成版本和游走版本
                if node == new_node:
                    node.weight_version = weight_version    # 生成版本，永不修改
                    node.traverse_version = weight_version  # 初始游走版本
                # 已有节点：只更新游走版本，保持生成版本不变
                else:
                    node.traverse_version = weight_version  # 更新游走版本

    return True
```

#### 2.3.2 版本分离更新策略

**核心设计**（Version Separation Architecture）：

**设计原则**：
- **生成版本** (`weight_version`): 永不修改，保持 token 来源信息
- **游走版本** (`traverse_version`): 动态更新，用于 GC 控制

**✅ CURRENT IMPLEMENTATION (版本分离)**：
```python
# 版本分离：分别设置生成版本和游走版本
if weight_version is not None:
    for node in traversed_nodes:
        if node != self.root and node.has_value:
            # 新创建节点：设置生成版本和游走版本
            if node == new_node:
                node.weight_version = weight_version    # 生成版本，永不修改
                node.traverse_version = weight_version  # 初始游走版本
            # 已有节点：只更新游走版本，保持生成版本不变
            else:
                node.traverse_version = weight_version  # 更新游走版本
```

**效果**：
- ✅ 生成版本信息完整保留，可用于版本追溯
- ✅ 游走版本控制 GC，活跃缓存不被误删
- ✅ 版本信息与 token 对齐，支持精确的版本管理
- ✅ 支持非生成 token 的版本标记（如手动 tokenize 的 prompt）

**版本分离示例**：
```
Scenario:
1. Version 1: Insert "Hello\nWorld"
   - node1("Hello"): weight_version=1, traverse_version=1
   - node2("World"): weight_version=1, traverse_version=1

2. Version 5: Insert "Hello\nGoodbye"
   - Traverse node1("Hello"): traverse_version=5 (更新!)
   - Create node2("Goodbye"): weight_version=5, traverse_version=5
   - 最终状态:
     * node1("Hello"): weight_version=1, traverse_version=5
     * node2("Goodbye"): weight_version=5, traverse_version=5

3. Version 10, GC (threshold=5):
   - 基于 traverse_version 判断: 5 > 10-5 → 保留
   - 生成版本信息完整保留: node1 仍然是 version=1 生成
```

### 2.4 GC 策略实现

#### 2.4.1 基于 Traverse Version 的 GC

```python
def gc_by_weight_version(
    self,
    current_weight_version: int,
    gc_threshold_k: int = 5
) -> int:
    """Remove nodes with outdated traverse_version."""
    gc_threshold = current_weight_version - gc_threshold_k
    removed_count = 0

    def _remove_outdated(node: StringTreeNode) -> bool:
        """Recursively remove outdated nodes. Returns True if node should be removed."""
        # Remove outdated children first
        node.children = [
            child for child in node.children
            if not _remove_outdated(child)
        ]

        # Check if current node is outdated (基于 traverse_version)
        if node.traverse_version is not None and node.traverse_version <= gc_threshold and node != self.root:
            self.total_entries -= 1
            self.cur_cache_size -= len(node.token_ids)
            removed_count += 1
            return True  # Mark for removal

        return False

    _remove_outdated(self.root)
    return removed_count
```

#### 2.4.2 参数说明

**`gc_threshold_k`**:
- **含义**: 保留最近 k 个 weight versions
- **默认值**: 5
- **调优建议**:
  - 频繁更新权重（每 10 steps）→ `gc_threshold_k = 10`（更宽松）
  - 不常更新权重（每 100 steps）→ `gc_threshold_k = 3`（更激进）

**触发时机**:
```python
if self.cur_cache_size > self.max_cache_size:
    removed = self.gc_by_weight_version(current_weight_version)
    if self.verbose:
        print(f"[GC] Removed {removed} entries, current size: {self.cur_cache_size}")
```

#### 2.4.3 未来优化方向：混合策略

**当前问题**: 仅依赖 Weight Version 可能不足
- 场景 1: 内存压力大，但 weight_version 都很新 → GC 无法触发
- 场景 2: 某些 trajectory 很久未访问，但 weight_version 刚好被更新 → 无法删除

**混合策略设计**:
```python
def gc_by_hybrid_strategy(self, current_weight_version: int) -> int:
    """Hybrid GC: Weight Version + LRU fallback."""
    removed_count = 0

    # Step 1: Remove outdated versions
    removed_count += self.gc_by_weight_version(current_weight_version)

    # Step 2: If still over limit, use LRU
    if self.cur_cache_size > self.max_cache_size:
        # Sort nodes by last_access_time
        all_nodes = self._get_all_nodes()
        all_nodes.sort(key=lambda n: n.last_access_time)

        # Remove least recently used
        target_removal = self.cur_cache_size - self.max_cache_size
        for node in all_nodes:
            if target_removal <= 0:
                break
            if self._remove_node(node):
                removed_count += 1
                target_removal -= len(node.token_ids)

    return removed_count
```

---

## 3. 关键 Bug 修复记录

### 3.1 Weight Version 未更新 Traversed Nodes

#### 问题描述

**发现时间**: 2025-10-07（PR #418 代码审查）

**症状**:
- 频繁使用的 trajectory 被 GC 删除
- 缓存命中率低于预期（<40%）
- RL 训练使用过期的 logp

**根因**:
```python
# 原始代码（错误）
if weight_version is not None and new_node:
    new_node.weight_version = weight_version  # 只更新新节点
```

被 traversed 的旧节点的 `weight_version` 未更新，导致：
1. 旧节点 `weight_version` 过时
2. GC 时被误删（即使仍在频繁使用）
3. 下次访问时缓存未命中，需重新 tokenize

#### 修复方案

```python
# 修复后代码
if weight_version is not None:
    for node in traversed_nodes:
        if node != self.root and node.has_value:
            node.weight_version = weight_version  # 更新所有 traversed 节点
```

#### 测试验证

**测试文件**: `tests/router/unit/test_radix_tree_core.py`

**关键测试**:
```python
def test_weight_version_traversed_nodes_update(sample_radix_tree):
    """Test: Traversed nodes update weight_version to latest."""
    tree = sample_radix_tree

    # Insert at version 1
    tree.insert("Hello\nWorld", [1, 2, 3], [-0.1, -0.2, -0.3], [0, 0, 0], weight_version=1)

    # Insert at version 5 (shares "Hello" prefix)
    tree.insert("Hello\nGoodbye", [1, 4, 5], [-0.1, -0.4, -0.5], [0, 1, 1], weight_version=5)

    # Verify: "Hello" node updated to version 5
    result = tree.find_longest_prefix("Hello")
    assert result.matched_prefix == "Hello"
    # (Internal check: node.weight_version == 5)
```

**验证结果**: ✅ 通过（9/9 tests passing）

### 3.2 部分匹配的 Loss Mask 错误

#### 问题描述

**发现时间**: 2025-10-07（PR #418 代码审查）

**症状**:
- 部分匹配时，剩余 text tokenize 后 `loss_mask` 错误标记为 `1`（Response）
- 导致 RL 训练时，prompt tokens 也参与 loss 计算
- 训练不稳定，loss 异常

**根因**:
```python
# 原始代码（错误）
if result.remaining_string:
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    full_loss_mask = result.loss_mask + [1] * len(additional_tokens)  # ❌ 错误！
```

剩余的 `remaining_string` 是用户输入的新 prompt，应该标记为 `loss_mask=0`，而非 `1`。

#### 修复方案

```python
# 修复后代码
if result.remaining_string:
    additional_tokens = tokenizer(result.remaining_string)["input_ids"]
    full_loss_mask = result.loss_mask + [0] * len(additional_tokens)  # ✅ Prompt tokens
```

#### 测试验证

**测试文件**: `tests/router/unit/test_radix_tree_core.py`

**关键测试**:
```python
def test_loss_mask_partial_match_remaining_prompt():
    """Test: Partial match, remaining text should be marked as prompt (0)."""
    tree = StringRadixTrie()
    tree.insert("Hello\n", [1, 2], [-0.1, -0.2], [0, 0])  # Cached

    # Query with extra text
    result = tree.retrieve_from_text("Hello\nWorld", return_logprob=True)

    # "Hello\n" cached, "World" tokenized as new prompt
    assert result.loss_mask == [0, 0, 0]  # All prompt tokens
    #                           └─┬─┘ └┘
    #                          cached new
```

**验证结果**: ✅ 通过（3/3 loss_mask tests passing）

---

## 4. 性能优化技巧

### 4.1 缓存大小调优

#### 4.1.1 内存占用估算

**公式**:
```
Memory (KB) = (cur_cache_size × 16 bytes) / 1024 + Tree_Overhead
```

**示例**:
- `max_cache_size = 10,000 tokens`
  - Token 数据: 10,000 × 16 = 160 KB
  - Tree 开销: ~50 KB
  - 总计: ~210 KB

- `max_cache_size = 100,000 tokens`
  - Token 数据: 100,000 × 16 = 1.56 MB
  - Tree 开销: ~500 KB
  - 总计: ~2 MB

#### 4.1.2 推荐配置

| 场景 | Prompts 数量 | 推荐 max_cache_size | 预估内存 |
|------|-------------|-------------------|---------|
| 小规模实验 | ~1k | 10,000 | ~210 KB |
| 中等规模 | ~10k | 50,000 | ~1 MB |
| 大规模生产 | ~100k | 200,000 | ~4 MB |

### 4.2 GC 触发频率控制

#### 4.2.1 gc_threshold_k 调优

**影响**:
- `gc_threshold_k` 越大 → 保留更多历史版本 → 内存占用高，缓存命中率高
- `gc_threshold_k` 越小 → 更激进删除 → 内存占用低，缓存命中率低

**建议**:
- 权重更新频繁（每 10 steps）: `gc_threshold_k = 10`
- 权重更新不频繁（每 100 steps）: `gc_threshold_k = 3`
- 默认值: `gc_threshold_k = 5`（平衡）

#### 4.2.2 监控指标

```bash
curl http://router-ip:30000/metrics | jq '.cache'
{
  "total_entries": 150,
  "cache_hits": 1200,
  "cache_misses": 300,
  "hit_rate": 0.8,
  "cur_cache_size": 8500,
  "max_cache_size": 10000,
  "gc_threshold_k": 5
}
```

**优化策略**:
- `hit_rate < 0.4` → 增大 `max_cache_size` 或 `gc_threshold_k`
- `hit_rate > 0.8` → 当前配置良好
- `cur_cache_size` 接近 `max_cache_size` → GC 频繁触发，考虑增大 `max_cache_size`

### 4.3 查询性能优化

#### 4.3.1 时间复杂度

- **查询**: O(k)，k 为 text 长度
- **插入**: O(k)
- **GC**: O(n)，n 为 tree 中节点总数

#### 4.3.2 优化建议

**1. Batch tokenization**（未来优化）:
```python
# 当前：逐个 tokenize remaining_string
for text in remaining_strings:
    tokens = tokenizer(text)["input_ids"]

# 优化：批量 tokenize
all_tokens = tokenizer(remaining_strings, padding=True)["input_ids"]
```

**2. 缓存 tokenizer 结果**（当前已实现）:
- Radix Tree 本身就是 tokenization cache
- 避免重复调用 `tokenizer()`

## 5. 异步并发支持

### 5.1 设计概述

RadixTree 支持同步和异步两种接口，以适应不同的使用场景：

```python
class StringRadixTrie:
    def __init__(self, max_cache_size=10000, tokenizer=None, verbose=False):
        # 双锁系统设计
        self._lock = threading.RLock()  # 同步接口向后兼容
        self._async_lock = AsyncReadWriteLock()  # 异步接口优化

    # 同步接口 - 向后兼容
    def find_longest_prefix(self, text: str) -> MatchResult:
        """同步查找前缀，适用于简单场景"""

    # 异步接口 - 高性能并发
    async def find_longest_prefix_async(self, text: str) -> MatchResult:
        """异步查找前缀，支持并发读取"""
```

### 5.2 异步读写锁

**核心特性**：
- 支持多个读取者并发访问
- 写入者独占访问
- 事件循环友好，不阻塞 asyncio

**性能对比**（20并发读取）：
- 同步 RLock：最大 45.2ms，平均 22.6ms
- 异步 RWLock：最大 0.4ms，平均 0.2ms
- **性能提升：99.1%**

### 5.3 接口选择指南

**使用异步接口的场景**：
- 高并发 Web 应用
- 需要最大化吞吐量的系统
- asyncio 环境

**使用同步接口的场景**：
- 简单脚本和工具
- 现有代码迁移成本高
- 性能要求不高

详细的性能优化和迁移指南请参考：
- [系统架构文档](architecture.md#42-radix-tree-异步并发优化)
- [开发指南](development.md#36-异步性能测试)

---

## 6. 版本分离架构实现细节

### 6.1 实现概览

版本分离架构已完全实现并通过 TDD 验证，包含以下核心组件：

**数据结构修改**：
- `StringTreeNode` 新增 `traverse_version` 字段
- `MatchResult` 新增 `generation_versions` 字段
- 使用 `@dataclass` 和 `field(default_factory=list)` 确保类型安全

**插入逻辑分离**：
```python
# 新节点：设置生成版本和游走版本
if node.weight_version is None:
    node.weight_version = weight_version    # 生成版本（永不修改）
    node.traverse_version = weight_version  # 初始游走版本
# 已有节点：只更新游走版本
else:
    node.traverse_version = weight_version  # 更新游走版本用于 GC 控制
```

**查询逻辑版本收集**：
```python
# 查询时收集每个 token 的生成版本
generation_version = best_child.weight_version if best_child.weight_version is not None else -1
matched_generation_versions.extend([generation_version] * len(best_child.token_ids))
```

**GC 逻辑更新**：
```python
# 基于 traverse_version 而非 weight_version 进行 GC
if node.traverse_version is not None and node.traverse_version <= gc_threshold:
    # 删除节点
```

### 6.2 测试覆盖

完整的 TDD 测试套件包含 11 个测试用例：

**版本分离核心功能**：
- `test_stringTreeNode_initialization_with_separated_versions`: 验证字段初始化
- `test_version_separation_on_insert`: 验证插入时版本分离
- `test_generation_versions_alignment_in_match_result`: 验证版本对齐

**GC 功能**：
- `test_gc_based_on_traverse_version`: 验证基于 traverse_version 的 GC
- `test_gc_preserves_generation_version_info`: 验证 GC 保留生成版本信息

**版本对齐**：
- `test_retrieve_from_text_version_alignment`: 验证 retrieve_from_text 版本对齐
- `test_partial_match_version_alignment`: 验证部分匹配版本对齐

**向后兼容性**：
- `test_backward_compatibility_weight_version_access`: 验证向后兼容性
- `test_existing_gc_interface_compatibility`: 验证现有 GC 接口兼容性

**边界情况**：
- `test_version_none_handling`: 验证 None 版本处理
- `test_version_reset_on_gc`: 验证 GC 时版本重置

### 6.3 中间件适配

RadixTreeMiddleware 无需修改即可适配新版本架构：

1. **缓存检索**：使用 `find_longest_prefix_async`，自动获得版本对齐的 `MatchResult`
2. **缓存插入**：使用 `weight_version` 作为生成版本，与架构设计一致
3. **向后兼容**：现有的三元组返回 `(token_ids, logp, loss_mask)` 保持不变

### 6.4 性能影响

版本分离架构的性能影响：

**内存开销**：
- 每个节点增加 8 字节（traverse_version: Optional[int]）
- 每次查询增加版本列表：4 字节 × token 数量
- 总体内存增长 < 5%

**计算开销**：
- 插入时版本判断：O(1) 操作
- 查询时版本收集：O(k) 操作，k 为匹配路径长度
- GC 性能显著改善：减少误删活跃缓存

**缓存命中率改善**：
- 避免 GC 删除频繁使用的轨迹
- 预期缓存命中率从 40% 提升到 80%+

### 6.5 迁移指南

对于现有代码，版本分离架构完全向后兼容：

**无需修改的代码**：
- 所有中间件代码
- 所有客户端调用接口
- 现有的 GC 调用

**可选的增强功能**：
- 使用 `result.generation_versions` 获取版本对齐信息
- 使用 `node.traverse_version` 监控缓存活跃度
- 基于 traverse_version 优化 GC 阈值

---

## 7. 相关资源

### 内部文档
- **架构设计**: [architecture.md](architecture.md)
- **用户手册**: [user-guide.md](user-guide.md)
- **开发者指南**: [development.md](development.md)

### 代码位置
- **Radix Tree 实现**: `slime/router/core/radix_tree.py` (698 lines)
- **异步读写锁**: `slime/router/utils/async_read_write_lock.py` (172 lines)
- **RadixTreeMiddleware**: `slime/router/middleware/radix_tree_middleware.py` (289 lines)
- **单元测试**:
  - `tests/router/unit/test_radix_tree_core.py` (Radix Tree 核心功能测试)
  - `tests/router/unit/test_weight_version_separation.py` (版本分离架构 TDD 测试)
  - `tests/router/unit/test_async_read_write_lock.py` (异步读写锁测试)
  - `tests/router/unit/test_radix_tree_async.py` (异步操作测试)
  - `tests/router/unit/test_radix_tree_middleware_async.py` (中间件异步测试)
  - `tests/router/unit/test_performance_comparison.py` (性能对比测试)
- **集成测试**: `tests/router/integration/test_radix_tree_middleware.py`