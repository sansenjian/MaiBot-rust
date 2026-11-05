//! mem-search：mmap float16 向量精确扫描器（AVX2），Rust 重写记忆子系统的第一块。
//! 替换 A_memorix 训练前窗口期的 float32 兜底索引：堆内 O(1)，页缓存按需换入，
//! 全精度（f16→f32 计算，优于主索引的 8bit 量化路径）。
//! 设计/验收：docs/mem-scanner-design.md（Top-K 重叠 ≥99.9% vs float32-exact）。

use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct Store {
    vectors: Mmap,
    ids: Vec<String>,
    dim: usize,
}

/// 最小堆元素（堆顶 = top-k 中最低分）。cmp 反转实现。
struct Hit {
    score: f32,
    idx: usize,
}
impl PartialEq for Hit {
    fn eq(&self, o: &Self) -> bool {
        self.score.total_cmp(&o.score).then(self.idx.cmp(&o.idx)).is_eq()
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        o.score.total_cmp(&self.score).then(o.idx.cmp(&self.idx))
    }
}

impl Store {
    pub fn open(vectors_path: &str, dim: usize, ids: Vec<String>) -> Result<Store> {
        let file = std::fs::File::open(vectors_path).context("open vectors file")?;
        let mmap = unsafe { Mmap::map(&file)? };
        if dim == 0 || mmap.len() % (dim * 2) != 0 {
            bail!("文件大小 {} 不是 dim={} 的整数倍", mmap.len(), dim);
        }
        let n = mmap.len() / (dim * 2);
        if n != ids.len() {
            bail!("向量数 {n} 与 ids 数 {} 不一致", ids.len());
        }
        Ok(Store { vectors: mmap, ids, dim })
    }

    #[inline]
    fn row_bytes(&self, i: usize) -> &[u8] {
        let start = i * self.dim * 2;
        &self.vectors[start..start + self.dim * 2]
    }

    /// 精确内积扫描，返回按分数降序的 Top-K (id, score)。
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;

        let mut heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(k);
        let mut row = vec![0f32; self.dim];
        for i in 0..n {
            f16_row_to_f32(self.row_bytes(i), &mut row);
            #[cfg(target_arch = "x86_64")]
            let score = if avx2 {
                // SAFETY: avx2 已运行时检测
                unsafe { dot_avx2(query, &row) }
            } else {
                dot_scalar(query, &row)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let score = dot_scalar(query, &row);

            if heap.len() < k {
                heap.push(Hit { score, idx: i });
            } else if let Some(top) = heap.peek() {
                if score > top.score {
                    heap.pop();
                    heap.push(Hit { score, idx: i });
                }
            }
        }
        let mut out: Vec<(String, f32)> = heap
            .into_iter()
            .map(|h| (self.ids[h.idx].clone(), h.score))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

#[inline]
fn f16_row_to_f32(src: &[u8], dst: &mut [f32]) {
    for (c, d) in src.chunks_exact(2).zip(dst.iter_mut()) {
        *d = f16::from_le_bytes([c[0], c[1]]).to_f32();
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut buf = [0f32; 8];
    _mm256_storeu(buf.as_mut_ptr(), acc);
    let mut sum: f32 = buf.iter().sum();
    while i < n {
        sum += a[i] * b[i];
    }
    sum
}

#[pyclass(name = "MemSearch")]
struct PyStore {
    store: Store,
}

#[pymethods]
impl PyStore {
    #[new]
    #[pyo3(signature = (vectors_path, dim, ids))]
    fn new(vectors_path: &str, dim: usize, ids: Vec<String>) -> PyResult<Self> {
        Store::open(vectors_path, dim, ids)
            .map(|store| PyStore { store })
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }

    fn search(&self, query: Vec<f32>, k: usize) -> Vec<(String, f32)> {
        self.store.search(&query, k)
    }

    fn __len__(&self) -> usize {
        self.store.ids.len()
    }
}

#[pymodule]
fn mem_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn write_fixture(path: &str, dim: usize, vecs: &[Vec<f32>]) {
        let mut buf = Vec::new();
        for v in vecs {
            for x in v {
                buf.extend_from_slice(&f16::from_f32(*x).to_le_bytes());
            }
        }
        std::fs::write(path, buf).unwrap();
        let _ = dim;
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn finds_expected_topk() {
        let dim = 8;
        let base = vec![1f32, 0., 0., 0., 0., 0., 0., 0.];
        let mut rows = vec![unit(&[0f32, 1., 0., 0., 0., 0., 0., 0.])];
        rows.push(unit(&base));
        for i in 0..98 {
            let mut v = vec![0f32; dim];
            v[i % dim] = 1.0;
            rows.push(unit(&v));
        }
        write_fixture("/tmp/memsearch_test.bin", dim, &rows);
        let ids: Vec<String> = (0..100).map(|i| format!("id-{i}")).collect();
        let store = Store::open("/tmp/memsearch_test.bin", dim, ids).unwrap();
        let q = unit(&base);
        let hits = store.search(&q, 3);
        assert_eq!(hits[0].0, "id-1");
        assert!((hits[0].1 - 1.0).abs() < 1e-3);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn rejects_mismatched_ids() {
        std::fs::write("/tmp/memsearch_bad.bin", vec![0u8; 32]).unwrap();
        assert!(Store::open("/tmp/memsearch_bad.bin", 8, vec!["a".to_string()]).is_err());
    }

    #[test]
    fn avx2_matches_scalar() {
        let a: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.002).cos()).collect();
        let s = dot_scalar(&a, &b);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let v = dot_avx2(&a, &b);
            assert!((s - v).abs() < 1e-2 * s.abs().max(1.0));
        }
    }
}
