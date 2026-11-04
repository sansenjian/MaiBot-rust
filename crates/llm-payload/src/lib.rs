//! P3：LLM payload 热路径（对齐 MaiBot v1.3.19）。
//!
//! - `compress_message_images`：对齐 `src/llm_models/utils.py::compress_messages`
//!   （PIL 管线：重格式化为 JPEG q95 → 超限时按 scale=target/len LANCZOS 缩放再编码）。
//!   输入输出为 OpenAI chat 消息 JSON。动图 GIF v1 原样透传（Python 版会重打包帧）。
//! - `items_to_messages`：对齐 `payload_content/context_item.py` 的语义投影
//!   （ContextItem 形状 JSON → OpenAI messages JSON；reasoning 不投影为消息）。
//! - `count_tokens`：cl100k_base 精确计数（tiktoken-rs），与 Python tiktoken 逐 token 一致。

mod compress;
mod items;
mod tokens;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// 对齐 compress_messages：压缩 OpenAI 消息里的 data-uri 图片。
/// messages: OpenAI chat 消息 JSON 字符串；返回压缩后的 JSON 字符串。
#[pyfunction]
#[pyo3(signature = (messages, target_size=None))]
fn compress_message_images(messages: &str, target_size: Option<usize>) -> PyResult<String> {
    let target = target_size.unwrap_or(compress::DEFAULT_TARGET);
    compress::compress_messages_json(messages, target)
        .map_err(|e| PyValueError::new_err(format!("{e:?}")))
}

/// ContextItem 列表（JSON）→ OpenAI messages（JSON）。
#[pyfunction]
fn items_to_messages(items: &str) -> PyResult<String> {
    items::items_to_messages_json(items).map_err(|e| PyValueError::new_err(format!("{e:?}")))
}

/// cl100k_base token 计数（与 tiktoken 完全一致）。
#[pyfunction]
fn count_tokens(text: &str) -> usize {
    tokens::count_tokens(text)
}

#[pymodule]
fn llm_payload(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_message_images, m)?)?;
    m.add_function(wrap_pyfunction!(items_to_messages, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_basic() {
        assert_eq!(tokens::count_tokens("hello world"), 2);
        assert_eq!(tokens::count_tokens(""), 0);
        assert!(tokens::count_tokens("麦麦是最可爱的机器人") > 0);
    }

    #[test]
    fn items_mapping() {
        let items = r#"[{"type":"message","role":"user","parts":[{"type":"text","text":"你好"}]}]"#;
        let out = items::items_to_messages_json(items).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"], "你好");
    }

    #[test]
    fn invalid_items_error() {
        assert!(items::items_to_messages_json("not json").is_err());
    }
}
