//! jieba-rs 的 PyO3 封装，API 对齐 Python `jieba`（P2 绞杀件）。
//!
//! 与 Python jieba 的关系：jieba-rs 是其 Rust 移植，通用语料上分词结果高度一致
//! （CI 中用 diff 脚本量化对齐率），算法实现独立、不依赖 CPython。
//!
//! ```python
//! import jieba_pyo3
//!
//! jieba_pyo3.lcut("麦麦是最可爱的机器人")     # 模块级默认实例
//!
//! j = jieba_pyo3.Jieba()                      # 独立实例（隔离词典）
//! j.load_dict("user_dict.txt")                # 自定义词典：每行 "词 [词频]"
//! j.add_word("麦麦", 1000)
//! j.lcut("麦麦来了", cut_all=False)
//! ```

mod hmm;

use std::io::BufReader;
use std::sync::Mutex;

use anyhow::Context;

/// CPython jieba 官方词典（MIT，fxsjy/jieba，34.9 万词条）——与 Python 版完全同一份数据，
/// 保证词汇层与 `import jieba` 零差异。每实例构建一次（约百毫秒级）。
static DICT: &[u8] = include_bytes!("../data/dict.txt");

fn new_jieba() -> Jieba {
    Jieba::with_dict(&mut BufReader::new(DICT)).expect("内置词典加载失败")
}
use jieba_rs::Jieba;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// 实例级封装（词典隔离，Mutex 保证并发安全）
#[pyclass(name = "Jieba")]
struct PyJieba {
    inner: Mutex<Jieba>,
}

#[pymethods]
impl PyJieba {
    #[new]
    fn new() -> Self {
        PyJieba {
            inner: Mutex::new(new_jieba()),
        }
    }

    /// 对齐 jieba.lcut(text, cut_all=False)；cut_all=True 对应全模式。
    #[pyo3(signature = (text, cut_all=false))]
    fn lcut(&self, text: &str, cut_all: bool) -> Vec<String> {
        let jieba = self.inner.lock().unwrap();
        hmm::cut_with_hmm(&jieba, text, cut_all)
    }

    /// 对齐 jieba.cut（Python 版返回生成器；这里直接返回列表，迭代行为一致）
    #[pyo3(signature = (text, cut_all=false))]
    fn cut(&self, text: &str, cut_all: bool) -> Vec<String> {
        self.lcut(text, cut_all)
    }

    /// 对齐 jieba.add_word(word, freq=None)
    #[pyo3(signature = (word, freq=None))]
    fn add_word(&self, word: &str, freq: Option<usize>) {
        self.inner.lock().unwrap().add_word(word, freq, None);
    }

    /// 对齐 jieba.load_userdict(path)：每行 "词 [词频]"，# 开头为注释
    fn load_dict(&self, path: &str) -> PyResult<()> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("读取词典失败: {path}"))
            .map_err(|e| PyIOError::new_err(format!("{e:?}")))?;
        let mut jieba = self.inner.lock().unwrap();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let word = parts.next().unwrap_or("");
            if word.is_empty() {
                continue;
            }
            let freq = parts.next().and_then(|f| f.parse::<usize>().ok());
            jieba.add_word(word, freq, None);
        }
        Ok(())
    }
}

// ---- 模块级默认实例（对齐 `import jieba; jieba.lcut(...)` 的隐式单例）----
static DEFAULT: std::sync::OnceLock<Mutex<Jieba>> = std::sync::OnceLock::new();

fn default_jieba() -> &'static Mutex<Jieba> {
    DEFAULT.get_or_init(|| Mutex::new(new_jieba()))
}

/// 对齐 jieba.lcut(text, cut_all=False)
#[pyfunction]
#[pyo3(signature = (text, cut_all=false))]
fn lcut(text: &str, cut_all: bool) -> Vec<String> {
    let jieba = default_jieba().lock().unwrap();
    hmm::cut_with_hmm(&jieba, text, cut_all)
}

/// 对齐 jieba.cut（返回列表，迭代行为与生成器一致）
#[pyfunction]
#[pyo3(signature = (text, cut_all=false))]
fn cut(text: &str, cut_all: bool) -> Vec<String> {
    lcut(text, cut_all)
}

#[pymodule]
fn jieba_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lcut, m)?)?;
    m.add_function(wrap_pyfunction!(cut, m)?)?;
    m.add_class::<PyJieba>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_basic() {
        let jieba = new_jieba();
        let words: Vec<String> = jieba
            .cut("麦麦是最可爱的机器人", false)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(words.contains(&"机器人".to_owned()));
        assert!(words.contains(&"是".to_owned()));
    }

    #[test]
    fn add_word_changes_segmentation() {
        // 注意：jieba-rs 默认词典已含"俄亥俄州立大学"，选必然不在默认词典里的词
        let mut jieba = new_jieba();
        let before: Vec<String> = jieba
            .cut("启动绞杀者模式", false)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(!before.contains(&"绞杀者模式".to_owned()));
        jieba.add_word("绞杀者模式", None, None);
        let after: Vec<String> = jieba
            .cut("启动绞杀者模式", false)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(after.contains(&"绞杀者模式".to_owned()));
    }

    #[test]
    fn cut_all_mode() {
        let jieba = new_jieba();
        let words = jieba.cut("麦麦", true);
        assert!(!words.is_empty());
    }
}
