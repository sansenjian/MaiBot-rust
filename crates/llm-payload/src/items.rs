//! 对齐 `payload_content/context_item.py` 的语义投影：
//! ContextItem 形状 JSON → OpenAI chat messages JSON。
//!
//! 规则（与 Python 语义对齐，diff 脚本中有参考实现逐条校验）：
//! - message（system/user/assistant）：parts 全为 text → content 为拼接字符串；
//!   含 image/refusal → content 为数组（image → image_url data-uri，格式 jpg/jpeg 归一 jpeg）
//! - reasoning：不投影为消息（辅助内容）
//! - function_call → assistant 消息 + tool_calls（content 为 null）
//! - function_call_output → role=tool 消息

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

fn part_to_content(part: &Value) -> Result<Value> {
    match part.get("type").and_then(Value::as_str) {
        Some("text") => Ok(json!({"type": "text", "text": part["text"]})),
        Some("image") => {
            let fmt = part["image_format"].as_str().unwrap_or("png").to_lowercase();
            let fmt = if fmt == "jpg" { "jpeg".to_owned() } else { fmt };
            let b64 = part["image_base64"].as_str().unwrap_or("");
            Ok(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:image/{fmt};base64,{b64}") }
            }))
        }
        Some("refusal") => Ok(json!({"type": "text", "text": part["refusal"]})),
        other => bail!("未知 part 类型: {other:?}"),
    }
}

pub fn items_to_messages_json(items: &str) -> Result<String> {
    let items: Vec<Value> = serde_json::from_str(items).context("items 不是合法 JSON 数组")?;
    let mut messages: Vec<Value> = Vec::new();
    for item in &items {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let role = item["role"].as_str().unwrap_or("user");
                let parts: Vec<&Value> = item["parts"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
                let all_text = parts
                    .iter()
                    .all(|p| p.get("type").and_then(Value::as_str) == Some("text"));
                let content = if all_text {
                    Value::String(
                        parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().concat(),
                    )
                } else {
                    let arr: Result<Vec<Value>> = parts.iter().map(|p| part_to_content(p)).collect();
                    Value::Array(arr?)
                };
                messages.push(json!({"role": role, "content": content}));
            }
            Some("reasoning") => {}
            Some("function_call") => {
                let tc = &item["tool_call"];
                messages.push(json!({
                    "role": "assistant",
                    "content": Value::Null,
                    "tool_calls": [{
                        "id": tc["call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["func_name"],
                            "arguments": tc["args_json"],
                        }
                    }]
                }));
            }
            Some("function_call_output") => {
                let mut msg = json!({
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                });
                if let Some(name) = item.get("tool_name").and_then(Value::as_str) {
                    if !name.is_empty() {
                        msg["name"] = json!(name);
                    }
                }
                messages.push(msg);
            }
            other => bail!("未知 item 类型: {other:?}"),
        }
    }
    Ok(serde_json::to_string(&messages)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_roundtrip() {
        let items = r#"[
            {"type":"function_call","tool_call":{"call_id":"c1","func_name":"get_time","args_json":"{}"}},
            {"type":"function_call_output","call_id":"c1","output":"2026","tool_name":"get_time"}
        ]"#;
        let out = items_to_messages_json(items).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "get_time");
        assert_eq!(v[1]["role"], "tool");
    }
}
