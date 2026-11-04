//! cl100k_base token 计数（tiktoken-rs，与 Python tiktoken 逐 token 一致）。

static BPE: std::sync::OnceLock<tiktoken_rs::CoreBPE> = std::sync::OnceLock::new();

pub fn count_tokens(text: &str) -> usize {
    let bpe = BPE.get_or_init(|| {
        tiktoken_rs::get_bpe_from_model("gpt-4").expect("cl100k_base 加载失败")
    });
    bpe.encode_ordinary(text).len()
}
