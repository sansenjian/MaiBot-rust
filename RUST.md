# Rust 加速组件（本 fork 新增）

本仓库是 [MaiBot](https://github.com/Mai-with-u/MaiBot) 的 fork，与原版**完全兼容、可直接使用**
（clone → `pip install -r requirements.txt` → 按原版文档配置 → `python bot.py`）。

在此之上，`crates/` 目录提供两个可选的 Rust 加速件（绞杀者模式 P2/P3，均带 CI 行为对齐验证）：

## jieba-pyo3（分词加速）

- 与 CPython `jieba` 行为对齐：官方 34.9 万词词典 + 移植 HMM 新词发现，语料对齐率 100%
- 热路径约 3.2x 加速，冷启动词典构建 22x（0.44s → 20ms）
- 使用：CI 产物下载 wheel 后 `pip install`，把 `import jieba` 换成 `import jieba_pyo3 as jieba`

## llm-payload（payload 热路径）

- `compress_message_images`：对齐 `src/llm_models/utils.py` 的 PIL 图片压缩管线
- `items_to_messages`：对齐 `payload_content/context_item.py` 的语义投影
- `count_tokens`：cl100k_base，与 tiktoken 逐 token 一致
- 使用：CI 产物下载 wheel 后 `pip install`，按需调用

## 开发

```bash
cargo test --no-default-features --manifest-path crates/jieba-pyo3/Cargo.toml
cargo test --no-default-features --manifest-path crates/llm-payload/Cargo.toml
```

行为对齐验证见各 crate 的 `tests/python_diff.py`（CI 自动运行）。
