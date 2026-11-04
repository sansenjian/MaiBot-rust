//! 对齐 MaiBot `src/llm_models/utils.py` 的 compress_messages 图片管线。
//!
//! Python 流程：b64decode → 非动图且 JPEG/PNG/WEBP 则转 JPEG q95 →
//! 若仍超 target，scale = target / 当前 b64 长度，LANCZOS 缩放到 (w*scale, h*scale)，
//! JPEG q95 重编码 → b64encode。任何解码失败原样透传。

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::engine::Engine;
use serde_json::Value;

pub const DEFAULT_TARGET: usize = 1024 * 1024;

fn encode_jpeg_q95(img: &image::DynamicImage) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 95);
    enc.encode_image(img).ok()?;
    Some(buf)
}

fn is_animated_gif(data: &[u8]) -> bool {
    let mut opts = gif::DecodeOptions::new();
    opts.allow_unknown_blocks(true);
    let Ok(mut dec) = opts.read_info(std::io::Cursor::new(data)) else {
        return false;
    };
    let mut frames = 0usize;
    while let Ok(Some(_)) = dec.read_next_frame() {
        frames += 1;
        if frames > 1 {
            return true;
        }
    }
    false
}

/// 对齐 compress_base64_image；失败/动图一律透传原字符串。
pub fn compress_base64_image(base64_data: &str, target_size: usize) -> String {
    let trimmed = base64_data.trim();
    let Ok(image_data) = B64.decode(trimmed) else {
        return base64_data.to_owned();
    };
    let Ok(img) = image::load_from_memory(&image_data) else {
        return base64_data.to_owned();
    };
    // v1 差异：多帧 GIF 原样透传（Python 版会逐帧缩放重打包）
    if is_animated_gif(&image_data) {
        return base64_data.to_owned();
    }

    // 1) 重格式化：仅 JPEG/PNG/WEBP 转 JPEG q95（对齐 Python 的格式白名单）
    let reader = image::ImageReader::new(std::io::Cursor::new(&image_data))
        .with_guessed_format();
    let fmt = reader.ok().and_then(|r| r.format());
    let current_bytes = match fmt {
        Some(image::ImageFormat::Jpeg) | Some(image::ImageFormat::Png) | Some(image::ImageFormat::WebP) => {
            match encode_jpeg_q95(&img) {
                Some(b) => b,
                None => return base64_data.to_owned(),
            }
        }
        // 其他格式（静态 GIF 等）不重格式化，用原始字节进尺寸判断
        _ => image_data.clone(),
    };
    let mut out = B64.encode(&current_bytes);
    if out.len() <= target_size {
        return out;
    }

    // 2) 尺寸压缩：scale = target / 当前 b64 长度（对齐 Python 的缩放公式）
    let scale = target_size as f32 / out.len() as f32;
    let w = ((img.width() as f32) * scale).max(1.0).round() as u32;
    let h = ((img.height() as f32) * scale).max(1.0).round() as u32;
    let resized = img
        .resize(w, h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    if let Some(b) = encode_jpeg_q95(&image::DynamicImage::ImageRgb8(resized)) {
        out = B64.encode(&b);
    }
    out
}

/// 对齐 compress_messages：遍历 OpenAI 消息，压缩 content 数组里的 data-uri 图片。
pub fn compress_messages_json(messages: &str, target_size: usize) -> Result<String> {
    let mut msgs: Vec<Value> = serde_json::from_str(messages).context("messages 不是合法 JSON 数组")?;
    for msg in &mut msgs {
        let Some(content) = msg.get_mut("content") else { continue };
        let Some(parts) = content.as_array_mut() else { continue };
        for part in parts {
            let is_image_url = part.get("type").and_then(Value::as_str) == Some("image_url");
            if !is_image_url {
                continue;
            }
            let Some(url) = part.pointer_mut("/image_url/url").and_then(|u| u.as_str().map(str::to_owned)) else {
                continue;
            };
            let Some(rest) = url.strip_prefix("data:image/") else { continue };
            let Some((fmt, b64)) = rest.split_once(";base64,") else { continue };
            let compressed = compress_base64_image(b64, target_size);
            if let Some(obj) = part.get_mut("image_url") {
                obj["url"] = Value::String(format!("data:image/{fmt};base64,{compressed}"));
            }
        }
    }
    Ok(serde_json::to_string(&msgs)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_invalid_base64() {
        assert_eq!(compress_base64_image("!!!not-base64!!!", 100), "!!!not-base64!!!");
    }

    #[test]
    fn passthrough_text_only_messages() {
        // 键序不保证（serde_json 按字母序重排），做语义比较
        let msgs = r#"[{"role":"user","content":"hello"}]"#;
        let out = compress_messages_json(msgs, 100).unwrap();
        let a: serde_json::Value = serde_json::from_str(&out).unwrap();
        let b: serde_json::Value = serde_json::from_str(msgs).unwrap();
        assert_eq!(a, b);
    }
}
