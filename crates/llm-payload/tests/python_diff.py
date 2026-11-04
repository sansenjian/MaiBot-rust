#!/usr/bin/env python3
"""llm_payload 与 Python 参考实现的行为对齐（P3 验收）。

- count_tokens：与 tiktoken (cl100k_base) 逐 token 完全一致（硬断言）
- items_to_messages：与内嵌参考实现逐字节一致（硬断言）
- compress_message_images：属性断言（大小上限含 JPEG 编码器差异容差、格式、透传路径）
"""
import base64
import io
import json
import sys

import llm_payload

# ---------- 1. count_tokens ----------
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
TEXTS = [
    "hello world",
    "麦麦是最可爱的机器人",
    "def quicksort(arr):\n    return sorted(arr)",
    "Emoji test 🎉🎉 combined with 中文 and English 123",
    "a" * 1000,
    "",
    "SELECT * FROM users WHERE id = 1; -- comment",
]
for t in TEXTS:
    expected = len(enc.encode(t, allowed_special=set()))
    actual = llm_payload.count_tokens(t)
    assert actual == expected, f"token 数不一致: {t[:30]!r} py={expected} rs={actual}"
print(f"count_tokens: {len(TEXTS)}/{len(TEXTS)} 完全一致")

# ---------- 2. items_to_messages（参考实现镜像 crate 规则） ----------
def ref_items_to_messages(items):
    out = []
    for item in items:
        t = item["type"]
        if t == "message":
            parts = item["parts"]
            if all(p["type"] == "text" for p in parts):
                content = "".join(p["text"] for p in parts)
            else:
                content = []
                for p in parts:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        fmt = p["image_format"].lower()
                        if fmt == "jpg":
                            fmt = "jpeg"
                        content.append({"type": "image_url",
                                        "image_url": {"url": f"data:image/{fmt};base64,{p['image_base64']}"}})
                    elif p["type"] == "refusal":
                        content.append({"type": "text", "text": p["refusal"]})
            out.append({"role": item["role"], "content": content})
        elif t == "reasoning":
            pass
        elif t == "function_call":
            tc = item["tool_call"]
            out.append({"role": "assistant", "content": None,
                        "tool_calls": [{"id": tc["call_id"], "type": "function",
                                        "function": {"name": tc["func_name"], "arguments": tc["args_json"]}}]})
        elif t == "function_call_output":
            msg = {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
            if item.get("tool_name"):
                msg["name"] = item["tool_name"]
            out.append(msg)
    return out

ITEMS = [
    {"type": "message", "role": "system", "parts": [{"type": "text", "text": "你是麦麦"}]},
    {"type": "message", "role": "user", "parts": [
        {"type": "text", "text": "看这张图"},
        {"type": "image", "image_format": "PNG", "image_base64": "QUJD"},
        {"type": "image", "image_format": "jpg", "image_base64": "REVG"},
    ]},
    {"type": "reasoning", "text_parts": ["先想想"]},
    {"type": "message", "role": "assistant", "parts": [{"type": "text", "text": "好的"}]},
    {"type": "function_call", "tool_call": {"call_id": "c1", "func_name": "get_time", "args_json": "{\"tz\":\"UTC\"}"}},
    {"type": "function_call_output", "call_id": "c1", "output": "2026-11-04", "tool_name": "get_time"},
]
expected = ref_items_to_messages(ITEMS)
actual = json.loads(llm_payload.items_to_messages(json.dumps(ITEMS, ensure_ascii=False)))
assert actual == expected, f"items_to_messages 不一致:\n{json.dumps(actual, ensure_ascii=False)}\nvs\n{json.dumps(expected, ensure_ascii=False)}"
print(f"items_to_messages: 结构完全一致（{len(ITEMS)} items → {len(actual)} messages）")

# ---------- 3. compress_message_images ----------
from PIL import Image

def make_b64_png(w, h, color=(200, 60, 60)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

TARGET = 200 * 1024
# 噪声图（类照片纹理）：PNG 体积大、JPEG 压缩率高——符合 413 压缩的真实场景
noise = Image.effect_noise((1200, 1200), 60).convert("RGB")
buf = io.BytesIO()
noise.save(buf, format="PNG")
big_b64 = base64.b64encode(buf.getvalue()).decode()

small_b64 = make_b64_png(20, 20)
msgs = json.dumps([
    {"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
    ]},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{small_b64}"}},
    ]},
    {"role": "user", "content": "纯文本不动"},
], ensure_ascii=False)

out = json.loads(llm_payload.compress_message_images(msgs, TARGET))
big_url = out[0]["content"][1]["image_url"]["url"]
assert big_url.startswith("data:image/png;base64,"), big_url[:50]
compressed_b64 = big_url.split(",", 1)[1]
assert base64.b64decode(compressed_b64)[:2] == b"\xff\xd8", "未重格式化为 JPEG"
# 真实契约：输出不超过目标（编码器差异容差 10%），且显著小于原 PNG
orig = len(big_b64)
assert len(compressed_b64) <= TARGET * 1.10, f"超目标: {len(compressed_b64)} vs {TARGET}"
assert len(compressed_b64) < orig * 0.3, f"压缩不足: {len(compressed_b64)} vs {orig}"
# 小图：重格式化会改变字节（PNG→JPEG q95），但仍须可解码且不超目标
small_url = out[1]["content"][0]["image_url"]["url"]
small_out_b64 = small_url.split(",", 1)[1]
assert small_out_b64 != small_b64, "静态 PNG 应被重格式化"
small_jpeg = base64.b64decode(small_out_b64)
assert small_jpeg[:2] == b"\xff\xd8"
Image.open(io.BytesIO(small_jpeg)).verify()
assert len(small_out_b64) <= TARGET
# 纯文本不动
assert out[2]["content"] == "纯文本不动"
# 非法 base64 整体不炸
bad = json.dumps([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}]}])
bad_out = json.loads(llm_payload.compress_message_images(bad, TARGET))
assert "!!!" in bad_out[0]["content"][0]["image_url"]["url"]
print("compress_message_images: 属性断言全部通过")
print("\nALL P3 ASSERTIONS PASSED")
