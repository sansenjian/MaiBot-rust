# MaiBot 部署内存调优（零代码三项 + 观察方案）

> 依据：真实部署 8 天 648MB RSS 的 memray/smaps 分析（增量分配仅 ~410KB/30s，
> 内存全部为历史滞留）。本文件只含环境/配置级改动，不改代码逻辑。

## 背景结论（为什么不是 Rust 化的靶子）

- 增量分配极少（~4KB/s）→ 无运行期分配热点
- 648MB 构成：~300MB numpy/faiss 匿名大块（滞留 + 碎片）+ 74MB glibc arena 碎裂
  （151 个 arena）+ 15-30MB sqlite 页缓存 + ~200MB 常驻对象
- 这类"滞留 + 碎片化"是分配器行为，jemalloc/arena 限制比任何语言重写都对症

## 三项零代码改动

### 1. 压扁 glibc arena（省 ~50-70MB，阻止碎片化增长）

systemd unit（或启动脚本）加环境变量：

```ini
[Service]
Environment=MALLOC_ARENA_MAX=2
```

更彻底（可选）：换 jemalloc

```ini
Environment=LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
# 需 apt install libjemalloc2；路径按发行版调整
```

### 2. sqlite 页缓存移出堆（省 15-30MB）

`src/A_memorix/core/storage/sqlite_connection.py` 的 PRAGMA 块（L80-84）加一行：

```python
connection.execute("PRAGMA mmap_size=268435456")  # 256MB，页缓存走文件映射而非堆
```

注：现有 `cache_size=-64000`（每连接 64MB）在多连接下堆内缓存可观；
mmap_size 生效后页缓存由 OS 管理，堆占用趋零。该行属于本 fork 的可选补丁，
不改语义。

### 3. 重启释放滞留 + 观察新曲线

已累积的 648MB 只有重启能释放。重启后按新曲线决策：

```bash
# 观察脚本：每小时记一次 RSS
while true; do
  echo "$(date +%F_%T) $(ps -o rss= -p $(pgrep -f 'bot.py' | head -1) | awk '{print $1/1024}') MB" >> ~/maibot_rss.log
  sleep 3600
done
```

- 曲线平（日增 < 20MB）：三项生效，Rust 侧只需收尾 mem-store
- 曲线再涨：对 300MB 匿名块做一次 `memray run --follow-fork python bot.py`
  全程录制（10 分钟活跃期），定位复发源头再定 Rust 化靶子

## 状态

- rust-memorix 分支的 mem-search 集成已截断"兜底建 float32 全量"这一复发路径
  （待 CI + 端到端验证收尾）
- reconcile_settled_delete_operations 经核实为低分配（~4KB/s），非热点，不处理
