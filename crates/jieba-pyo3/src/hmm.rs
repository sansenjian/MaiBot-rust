//! HMM 新词发现，移植自 CPython jieba 的 finalseg（MIT，fxsjy/jieba）。
//!
//! jieba-rs 无 HMM 能力，这里用官方概率表（prob_start/trans/emit，已转为 JSON）
//! 实现 viterbi，对"连续单字未登录片段"做 BEMS 标注并合并——与 Python jieba
//! 的未登录词识别行为对齐（麦麦→一个词、杭研→一个词）。
//! 模型全局共享（只读），各 Jieba 实例零构建成本。

use serde_json::Value;
use std::collections::HashMap;

const MIN_FLOAT: f64 = -3.14e100;
const STATES: [&str; 4] = ["B", "M", "E", "S"];

static MODEL: std::sync::OnceLock<Value> = std::sync::OnceLock::new();

fn model() -> &'static Value {
    MODEL.get_or_init(|| serde_json::from_slice(include_bytes!("../data/hmm_model.json")).unwrap())
}

fn viterbi(chars: &[char]) -> Vec<String> {
    let m = model();
    let mut v: Vec<HashMap<&str, f64>> = vec![Default::default()];
    let mut path: HashMap<&str, Vec<&str>> = Default::default();
    let emit = |y: &str, c: char| -> f64 {
        m["emit"][y].get(c.to_string()).and_then(Value::as_f64).unwrap_or(MIN_FLOAT)
    };
    let tr = |y0: &str, y: &str| -> f64 {
        m["trans"][y0].get(y).and_then(Value::as_f64).unwrap_or(MIN_FLOAT)
    };
    for y in STATES {
        v[0].insert(y, m["start"][y].as_f64().unwrap() + emit(y, chars[0]));
        path.insert(y, vec![y]);
    }
    for t in 1..chars.len() {
        let mut nv: HashMap<&str, f64> = Default::default();
        let mut np: HashMap<&str, Vec<&str>> = Default::default();
        for y in STATES {
            let (mut best, mut bst) = (f64::NEG_INFINITY, "");
            for y0 in STATES {
                let p = v[t - 1][y0] + tr(y0, y) + emit(y, chars[t]);
                if p > best {
                    best = p;
                    bst = y0;
                }
            }
            nv.insert(y, best);
            let mut pp = path[bst].clone();
            pp.push(y);
            np.insert(y, pp);
        }
        v.push(nv);
        path = np;
    }
    let (mut best, mut bst) = (f64::NEG_INFINITY, "");
    for y in ["E", "S"] {
        if v[chars.len() - 1][y] > best {
            best = v[chars.len() - 1][y];
            bst = y;
        }
    }
    path[bst].iter().map(|s| s.to_string()).collect()
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c) || ('\u{3400}'..='\u{4dbf}').contains(&c)
}

/// 在 jieba-rs 分词结果上叠 HMM：连续单字 CJK token 视为未登录片段，
/// viterbi 标注后按 B/M..E 合并，对齐 Python jieba 的未登录词行为。
pub fn cut_with_hmm(jieba: &jieba_rs::Jieba, text: &str, cut_all: bool) -> Vec<String> {
    let tokens: Vec<String> = jieba.cut(text, cut_all).iter().map(|s| s.to_string()).collect();
    let mut out: Vec<String> = Vec::new();
    let mut buf: Vec<char> = Vec::new();
    let flush = |buf: &mut Vec<char>, out: &mut Vec<String>| {
        if buf.is_empty() {
            return;
        }
        let states = viterbi(buf);
        let mut cur = String::new();
        for (c, st) in buf.iter().zip(states.iter()) {
            cur.push(*c);
            if st == "E" || st == "S" {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        buf.clear();
    };
    for tok in tokens {
        let mut ch = tok.chars();
        if tok.chars().count() == 1 && is_cjk(ch.next().unwrap()) {
            buf.push(tok.chars().next().unwrap());
        } else {
            flush(&mut buf, &mut out);
            out.push(tok);
        }
    }
    flush(&mut buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmm_merges_unknown_words() {
        let jieba = jieba_rs::Jieba::new();
        // 连续未登录单字会被 HMM 合并为一个词（对齐 Python jieba 行为）
        assert_eq!(
            cut_with_hmm(&jieba, "麦麦去杭研", false),
            vec!["麦麦", "去", "杭研"]
        );
    }
}
