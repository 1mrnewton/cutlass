//! Tile layouts that balance the coding work between threads.
//!
//! Tiles are coded in parallel, one thread each, so a frame takes as long
//! as its most expensive tile. Coding time follows the partition closely
//! (a fixed cost per 64x64 superblock plus one per block), and the
//! partition changes little from frame to frame, so the previous frame's
//! partition sets the tile boundaries of the next one. Everything here is a
//! function of the coded frames: the bitstream does not depend on timing.

use crate::dec::frame::FrameState;
use crate::dec::types::{bh4, bw4};

/// Work of a 64x64 superblock with a single block, in blocks.
const SB_BASE: u64 = 2;

/// Work of coding each 64x64 superblock of the frame `fs` describes, in
/// 1/256 blocks (row-major, `ceil(mi_cols / 16)` per row).
pub fn sb_work(fs: &FrameState) -> Vec<u32> {
    let sb_cols = fs.mi_cols.div_ceil(16);
    let sb_rows = fs.mi_rows.div_ceil(16);
    let mut work = vec![(SB_BASE * 256) as u32; sb_cols * sb_rows];
    for r in 0..fs.mi_rows {
        let row = &fs.mi[r * fs.mi_stride..][..fs.mi_cols];
        for (c, mi) in row.iter().enumerate() {
            // Each 4x4 unit holds its share of its block.
            let area = (bw4(mi.bsize) * bh4(mi.bsize)).max(1) as u32;
            work[(r >> 4) * sb_cols + (c >> 4)] += 256 / area;
        }
    }
    work
}

/// Splits `0..n` into `k` nonempty ranges of at most `max_len` that
/// minimize the largest `cost(a, b)` of a range `a..b` (`cost` grows with
/// the range). Returns the range lengths, or `None` when impossible.
fn split(
    n: usize,
    k: usize,
    max_len: usize,
    cost: &dyn Fn(usize, usize) -> u64,
) -> Option<Vec<u32>> {
    if k == 0 || n < k || n > k * max_len {
        return None;
    }
    // best[j][i]: the smallest largest cost of `0..i` in `j` ranges.
    let mut best = vec![vec![u64::MAX; n + 1]; k + 1];
    let mut from = vec![vec![0usize; n + 1]; k + 1];
    best[0][0] = 0;
    for j in 1..=k {
        for i in j..=n {
            for a in (j - 1).max(i.saturating_sub(max_len))..i {
                if best[j - 1][a] == u64::MAX {
                    continue;
                }
                let v = best[j - 1][a].max(cost(a, i));
                if v < best[j][i] {
                    best[j][i] = v;
                    from[j][i] = a;
                }
            }
        }
    }
    if best[k][n] == u64::MAX {
        return None;
    }
    let mut lens = vec![0u32; k];
    let mut i = n;
    for j in (1..=k).rev() {
        let a = from[j][i];
        lens[j - 1] = (i - a) as u32;
        i = a;
    }
    Some(lens)
}

/// Uniform lengths of `k` ranges over `n`.
fn even(n: usize, k: usize) -> Vec<u32> {
    (0..k)
        .map(|i| ((i + 1) * n / k - i * n / k) as u32)
        .collect()
}

/// The work of the most loaded tile of a layout.
fn makespan(work: &[u32], sb_cols: usize, widths: &[u32], heights: &[u32]) -> u64 {
    let mut worst = 0;
    let mut r0 = 0;
    for &h in heights {
        let mut c0 = 0;
        for &w in widths {
            let mut s = 0u64;
            for r in r0..r0 + h as usize {
                s += work[r * sb_cols + c0..][..w as usize]
                    .iter()
                    .map(|&v| u64::from(v))
                    .sum::<u64>();
            }
            worst = worst.max(s);
            c0 += w as usize;
        }
        r0 += h as usize;
    }
    worst
}

/// Column widths and row heights (in superblocks) of a layout of
/// `cols` x `rows` tiles that balances `work` (from [`sb_work`]): the
/// boundaries minimize the work of the most loaded tile. Tiles are at most
/// `max_w` superblocks wide. (A single row or column of tiles balances
/// better but costs about 1% in BD-rate at 1080p.)
pub fn balanced_layout(
    work: &[u32],
    sb_cols: usize,
    sb_rows: usize,
    cols: usize,
    rows: usize,
    max_w: usize,
) -> Option<(Vec<u32>, Vec<u32>)> {
    if work.len() != sb_cols * sb_rows {
        return None;
    }
    // Prefix sums of every superblock row.
    let pre: Vec<Vec<u64>> = (0..sb_rows)
        .map(|r| {
            let mut p = vec![0u64; sb_cols + 1];
            for c in 0..sb_cols {
                p[c + 1] = p[c] + u64::from(work[r * sb_cols + c]);
            }
            p
        })
        .collect();
    // Column ranges given the row bands, and row ranges given the columns.
    let fit_cols = |heights: &[u32], k: usize| -> Option<Vec<u32>> {
        let bands: Vec<(usize, usize)> = heights
            .iter()
            .scan(0usize, |s, &h| {
                let b = (*s, *s + h as usize);
                *s = b.1;
                Some(b)
            })
            .collect();
        split(sb_cols, k, max_w, &|a, b| {
            bands
                .iter()
                .map(|&(r0, r1)| (r0..r1).map(|r| pre[r][b] - pre[r][a]).sum::<u64>())
                .max()
                .unwrap_or(0)
        })
    };
    let fit_rows = |widths: &[u32], k: usize| -> Option<Vec<u32>> {
        let mut edges = vec![0usize];
        for &w in widths {
            edges.push(edges[edges.len() - 1] + w as usize);
        }
        split(sb_rows, k, sb_rows, &|a, b| {
            edges
                .windows(2)
                .map(|e| (a..b).map(|r| pre[r][e[1]] - pre[r][e[0]]).sum::<u64>())
                .max()
                .unwrap_or(0)
        })
    };
    // Columns and rows refined in turn, from even rows; the best round wins.
    let mut heights = even(sb_rows, rows);
    let mut best = (u64::MAX, Vec::new(), Vec::new());
    for _ in 0..3 {
        let w = fit_cols(&heights, cols)?;
        let h = fit_rows(&w, rows)?;
        heights = h.clone();
        let m = makespan(work, sb_cols, &w, &h);
        if m < best.0 {
            best = (m, w, h);
        }
    }
    Some((best.1, best.2))
}

#[cfg(test)]
mod tests {
    use super::{balanced_layout, makespan};

    #[test]
    fn balances_a_hot_spot() {
        // 30 x 17 superblocks, a hot area in the lower right.
        let (sc, sr) = (30, 17);
        let mut work = vec![600u32; sc * sr];
        for r in 10..15 {
            for c in 18..26 {
                work[r * sc + c] = 6000;
            }
        }
        let total: u64 = work.iter().map(|&v| u64::from(v)).sum();
        let Some((w, h)) = balanced_layout(&work, sc, sr, 2, 2, 64) else {
            panic!("no layout");
        };
        assert_eq!(w.iter().sum::<u32>() as usize, sc);
        assert_eq!(h.iter().sum::<u32>() as usize, sr);
        assert_eq!(w.len() * h.len(), 4);
        let m = makespan(&work, sc, &w, &h);
        let uniform = makespan(&work, sc, &[15, 15], &[9, 8]);
        assert!(
            m < uniform && m * 10 < total * 3,
            "{m} of {total}, uniform {uniform}"
        );
    }

    #[test]
    fn respects_the_width_limit() {
        let work = vec![1u32; 100 * 4];
        let Some((w, h)) = balanced_layout(&work, 100, 4, 2, 1, 64) else {
            panic!("no layout");
        };
        assert!(w.iter().all(|&x| x <= 64) && h == [4]);
        assert!(balanced_layout(&work, 100, 4, 1, 1, 64).is_none());
    }
}
