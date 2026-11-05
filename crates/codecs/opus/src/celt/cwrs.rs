//! PVQ codeword indexing (RFC 6716 section 4.3.4.2, `cwrs.c`).
//!
//! V(N,K) counts the integer vectors of N dimensions whose absolute values
//! sum to K: V(N,K) = V(N-1,K) + V(N,K-1) + V(N-1,K-1), V(N,0) = 1, and
//! V(0,K) = 0 for K > 0. The table is built once with saturating
//! arithmetic; the allocation only asks for codebooks that fit 32 bits.

use std::sync::OnceLock;

use super::coder::Coder;

/// Largest band in bins: 22 bins of band 20 at LM 3.
pub(crate) const MAX_N: usize = 176;
/// Largest pulse count the pulse cache can produce (`MAX_PULSES`), plus one.
pub(crate) const MAX_K: usize = 130;

struct Table {
    v: Vec<u32>,
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut v = vec![0u32; (MAX_N + 1) * (MAX_K + 1)];
        v[0] = 1;
        for n in 1..=MAX_N {
            v[n * (MAX_K + 1)] = 1;
            for k in 1..=MAX_K {
                let a = v[(n - 1) * (MAX_K + 1) + k];
                let b = v[n * (MAX_K + 1) + k - 1];
                let c = v[(n - 1) * (MAX_K + 1) + k - 1];
                v[n * (MAX_K + 1) + k] = a.saturating_add(b).saturating_add(c);
            }
        }
        Table { v }
    })
}

/// V(n, k), saturated at `u32::MAX`; 0 outside the table.
pub(crate) fn v(n: usize, k: usize) -> u32 {
    if n > MAX_N || k > MAX_K {
        return u32::MAX;
    }
    table().v.get(n * (MAX_K + 1) + k).copied().unwrap_or(0)
}

/// The vector with index `i` of V(n, k), written to `y`.
fn index_to_vector(mut i: u32, k: usize, y: &mut [i32]) {
    let n = y.len();
    let mut k = k;
    for (j, slot) in y.iter_mut().enumerate() {
        let rest = n - j - 1;
        let full = (u64::from(v(rest, k)) + u64::from(v(rest + 1, k))) / 2;
        let mut p = full;
        let sign = if u64::from(i) < p {
            1
        } else {
            i = (u64::from(i) - p) as u32;
            -1
        };
        let k0 = k;
        p -= u64::from(v(rest, k));
        while p > u64::from(i) && k > 0 {
            k -= 1;
            p -= u64::from(v(rest, k)).min(p);
        }
        *slot = sign * (k0 - k) as i32;
        i = (u64::from(i).saturating_sub(p)) as u32;
    }
}

/// The index of `y` (whose magnitudes sum to `k`) among V(len, k).
fn vector_to_index(y: &[i32], k: usize) -> u32 {
    let n = y.len();
    let mut k = k;
    let mut index = 0u64;
    for (j, &x) in y.iter().enumerate() {
        let rest = n - j - 1;
        let m = x.unsigned_abs() as usize;
        if x < 0 {
            index += (u64::from(v(rest, k)) + u64::from(v(rest + 1, k))) / 2;
        }
        // Values with a larger magnitude than m sit below this one.
        let below: u64 = if m == 0 {
            (0..k).map(|r| u64::from(v(rest, r))).sum()
        } else {
            (0..k - m.min(k)).map(|r| u64::from(v(rest, r))).sum()
        };
        index += below;
        k -= m.min(k);
    }
    index as u32
}

/// Codes the pulses `y` (magnitudes summing to `k`) as one uniform index;
/// returns them unchanged for the encoder and fills them for the decoder.
pub(crate) fn code_pulses(coder: &mut Coder<'_, '_>, y: &mut [i32], k: usize) {
    let total = v(y.len(), k);
    if coder.encoding() {
        let index = vector_to_index(y, k);
        coder.uint(index, total);
    } else {
        let index = coder.uint(0, total);
        index_to_vector(index, k, y);
    }
}

#[cfg(test)]
mod tests {
    use super::{index_to_vector, v, vector_to_index};

    /// V(N,K) from the closed form sum_d 2^d C(N,d) C(K-1,d-1).
    fn v_closed(n: u64, k: u64) -> u128 {
        if k == 0 {
            return 1;
        }
        let choose = |a: u64, b: u64| -> u128 {
            if b > a {
                return 0;
            }
            let mut r = 1u128;
            for i in 0..b {
                r = r * u128::from(a - i) / u128::from(i + 1);
            }
            r
        };
        (1..=n.min(k))
            .map(|d| (1u128 << d) * choose(n, d) * choose(k - 1, d - 1))
            .sum()
    }

    #[test]
    fn counts_match_the_closed_form() {
        for n in 0..=40u64 {
            for k in 0..=40u64 {
                let want = v_closed(n, k);
                if want <= u128::from(u32::MAX) {
                    assert_eq!(u128::from(v(n as usize, k as usize)), want, "V({n},{k})");
                }
            }
        }
        assert_eq!(v(2, 128), 512);
        assert_eq!(v(176, 1), 352);
    }

    #[test]
    fn every_index_round_trips_on_small_codebooks() {
        for n in 1..=6usize {
            for k in 1..=6usize {
                let total = v(n, k);
                let mut seen = std::collections::HashSet::new();
                for i in 0..total {
                    let mut y = vec![0i32; n];
                    index_to_vector(i, k, &mut y);
                    assert_eq!(
                        y.iter().map(|x| x.unsigned_abs() as usize).sum::<usize>(),
                        k
                    );
                    assert!(seen.insert(y.clone()), "duplicate vector");
                    assert_eq!(vector_to_index(&y, k), i, "n {n} k {k}");
                }
            }
        }
    }

    #[test]
    fn large_codebooks_round_trip() {
        let mut seed = 7u32;
        for &(n, k) in &[
            (176usize, 4usize),
            (16, 12),
            (5, 90),
            (2, 128),
            (24, 8),
            (3, 128),
            (4, 128),
        ] {
            assert!(u64::from(v(n, k)) < u64::from(u32::MAX), "fits 32 bits");
            for _ in 0..200 {
                let mut y = vec![0i32; n];
                for _ in 0..k {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let at = (seed >> 8) as usize % n;
                    y[at] += if y[at] < 0 || (y[at] == 0 && seed & 1 == 0) {
                        -1
                    } else {
                        1
                    };
                }
                let i = vector_to_index(&y, k);
                assert!(i < v(n, k));
                let mut back = vec![0i32; n];
                index_to_vector(i, k, &mut back);
                assert_eq!(back, y);
            }
        }
    }
}
