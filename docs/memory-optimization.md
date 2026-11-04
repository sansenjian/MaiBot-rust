# A_memorix 内存优化设计（v1.3.19 实码对照）

> 目标：降低 Core 数据栈内存。前提修正：当前 A_memorix **不是**旧计划假设的
> "pandas/pyarrow 管道"——v1.3.19 的元数据已落在 **SQLite**（metadata_store /
> metadata_fts，无大字典缓存，RAM 占用天然很低）；真正的内存集中在
> **vector_store 的双 faiss 索引**。优化重心相应调整。

## 现状内存模型（代码锚点：src/A_memorix/core/storage/vector_store.py @ v1.3.19）

| 组件 | 存储 | RAM 占用 |
|---|---|---|
| 向量磁盘格式 | float16 `vectors.bin`（L401） | —（已优于 float32） |
| 主索引 `_index` | `IndexIDMap2(IndexScalarQuantizer(QT_8bit))`（L487） | dim × 1B × N |
| **兜底索引 `_fallback_index`** | `IndexIDMap2(IndexFlatIP)` **float32 全量**（L493-496） | **dim × 4B × N ← 大头** |
| 启动加载 `_load_index` | 全量 float16→float32 一次性物化（L266-288） | **瞬时 4x 尖峰** |
| 元数据/FTS | SQLite（metadata_store/fts，无常驻缓存） | 低 ✅ |

兜底索引的语义（L1784）：仅当主索引未训练或为空时才参与查询——它是冷启动
自举用的（L2043 "Fallback index self-bootstrapped"），却让全部向量以 float32
常驻 RAM。

**估算（1536 维、10 万条记忆）**：兜底 600MB + 主索引 150MB + 开销 ≈ 0.75-1GB。

## 方案

### D1 启动加载分批化（小改，先做这个）
`_load_index` 按块（如 5 万条/批）读 `vectors.bin` → 转 float32 → `add_with_ids`，
把启动瞬时尖峰从 4×dim×N 压到一批的大小；同时缩短启动时间。

### D2 兜底索引限额（核心收益，4x 级）
三条路线按侵入度排：
- **D2a 训练完成后清空兜底**：主索引 trained 且 ntotal 覆盖全集后
  `fallback.reset()`，仅在需要恢复时惰性重建（现有 L980/L1244 已有重建路径可复用）
- **D2b 兜底也换 QT_8bit**：兜底查询路径低频，8bit 平替 float32，RAM 直降 4x，
  查询精度在自举场景完全够
- **D2c 只为未入主的条目保留兜底**：把兜底水位限在批大小量级（L1658-1659 已有
  remove_ids 模式，推广到全生命周期）

推荐 D2b（改动最小、行为差异最小）或 D2a（收益最彻底）。

### D3 主索引换 IVF-PQ（可选，再 2-4x）
QT_8bit 是逐维标量量化（dim 字节/向量）；IVF-PQ 降到与维度无关的编码
（如 1536 维 → ~100-150B/向量），代价是近似检索与训练需求。内存敏感部署再考虑。

### 不做的事
- 元数据侧不动（SQLite 已优）
- 不动 faiss-cpu 本身；不动检索语义的训练数据流
- L4 存储格式重设计（上一轮梯子的底层）维持搁置

## 验收

1. 行为：记忆写入/检索回归测试通过（检索 Top-K 结果与原实现 diff，容差随 D2/D3 路线定义）
2. 内存：启动峰值与稳态 RSS 前后对照（psrecord / tracemalloc 采启动段）
3. 目标数字（10 万条、1536 维）：稳态 0.75-1GB → D1+D2b 后 ~150-250MB（4-6x），D3 后 <100MB
