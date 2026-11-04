# 记忆向量 mmap 扫描器（Rust/PyO3）设计

> 目标：替换 A_memorix 训练前窗口期的 float32 兜底索引（IndexIDMap2(IndexFlatIP)），
> 实现"窗口期内存 ≈ 0 堆内增长 + 全精度检索"。同时可作为低内存部署的可选检索后端。

## 原理

兜底索引在窗口期做的事：对全量向量做精确内积扫描（float16 存储 → float32 计算）。
这不需要常驻索引——**mmap 磁盘上的 vectors.bin，逐行转 float32 + SIMD 点积**即可，
堆内内存 O(1)，页缓存按需换入。

## 形态

`crates/mem-search`（PyO3，加入 MaiBot-rust workspace）：

```python
import mem_search
# 打开现有 A_memorix 数据文件（零迁移：同一份 vectors.bin / ids bin）
store = mem_search.open("/data/vectors.bin", "/data/vectors_ids.bin", dim=1536)
# 查询（query 为已 L2 归一化的 float32 列表；返回 [(id, score)] 按内积降序）
hits = store.search(query, k=20)
```

## 实现要点

- `memmap2` 只读映射两个文件；`half::f16` → f32 逐行转换
- SIMD：AVX2 f32x8 点积（`#[target_feature(enable = "avx2")]` + 标量回退），
  必要时 `rayon` 并行分片（默认单线程，先测延迟）
- Top-K：固定大小最大堆（k ≤ 1000），堆大小无关 N
- 删除过滤：接受 deleted_ids 集合参数（与上游 `valid_mask` 语义对齐）
- 复用上游持久化格式：float16 行主序 + big-endian int64 ids，写入路径不动

## 集成（替换兜底）

vector_store 窗口期路径：
`_bootstrap_fallback_from_disk` → 改为 `self._mem_fallback = mem_search.open(...)`；
search 的兜底分支（L1784 附近 "main 未训练"）→ `mem_fallback.search(query, k)`。
训练完成 replay 后 `self._mem_fallback.close()`。

## 验收标准（对齐本项目行为对齐方法论）

| 项 | 标准 |
|---|---|
| Top-K 一致性 | 对 float32-exact（f16 源）暴力扫描：集合重叠率 ≥ 99.9%，同分时顺序差异允许 |
| 内存 | 窗口期进程 RSS 增量 < 50MB（vs 兜底 float32 的 dim×4B×N） |
| 延迟 | p95 < 50ms（10 万条 × 1536 维，单线程 AVX2）；并行后 < 15ms |
| 回归 | 记忆写入/检索全流程测试通过；训练完成后的主索引路径不受影响 |

## 不做

- 不动主索引（QT_8bit）与训练流程
- 不引入近似（这是与 D2b/IVF-PQ 的本质区别：全精度）
- 暂不做增量写入接口（窗口期只读；写入仍走上游路径）
