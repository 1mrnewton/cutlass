//! The inverse DCT on AVX2: both passes as sums of `pmaddwd` products of
//! 16-bit pairs.
//!
//! The first pass works on 16 columns at a time (8 for 8×8 blocks): two
//! input rows interleaved, times the pair of matrix entries of each
//! output row. The second pass works on 8 outputs of a row at a time:
//! a pair of intermediate values broadcast, times the interleaved matrix
//! rows of a table.

// The loops index rows, strips and taps by one position at once.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_adds_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_madd_epi16, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set_epi32, _mm_set_epi64x, _mm_set1_epi32, _mm_setzero_si128,
    _mm_sra_epi32, _mm_srai_epi32, _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_extract_epi64, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_sra_epi32, _mm256_srai_epi32, _mm256_unpackhi_epi16,
    _mm256_unpacklo_epi16,
};

use super::{DCT32, DST4};

/// Where the residual goes: stored as 32-bit values (rows of `N`), or
/// added to 8-bit samples (rows `stride` apart) and clipped.
pub(super) enum Sink<'a> {
    Res(&'a mut [i32]),
    Add(&'a mut [u8], usize),
}

/// Add eight residuals (32-bit) to the eight samples at `d[at..]`,
/// clipped to bytes. The residuals saturate at 16 bits first: a value
/// past either end clips to the same sample.
#[inline]
#[target_feature(enable = "avx2")]
fn add8(d: &mut [u8], at: usize, v: __m256i) {
    let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) else {
        return;
    };
    let r = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let px = _mm_cvtepu8_epi16(_mm_set_epi64x(0, i64::from_le_bytes(*d)));
    let sum = _mm_adds_epi16(px, r);
    *d = _mm_cvtsi128_si64(_mm_packus_epi16(sum, sum)).to_le_bytes();
}

/// Add four residuals to the four samples at `d[at..]`, clipped.
#[inline]
#[target_feature(enable = "avx2")]
fn add4(d: &mut [u8], at: usize, v: __m128i) {
    let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<4>()) else {
        return;
    };
    let r = _mm_packs_epi32(v, v);
    let px = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(i32::from_le_bytes(*d)));
    let sum = _mm_adds_epi16(px, r);
    let b = _mm_cvtsi128_si32(_mm_packus_epi16(sum, sum));
    *d = b.to_le_bytes();
}

/// Entry `[k][n]` of the `N`-point matrix.
const fn m<const N: usize>(k: usize, n: usize) -> i32 {
    DCT32[k * (32 / N)][n] as i32
}

/// The pairs `(M[2p][y], M[2p + 1][y])` of the first pass, as 32-bit
/// values: `FIRST[p][y]`.
const fn first<const N: usize>() -> [[i32; 32]; 16] {
    let mut out = [[0i32; 32]; 16];
    let mut p = 0;
    while p < N / 2 {
        let mut y = 0;
        while y < N {
            out[p][y] = (m::<N>(2 * p, y) & 0xffff) | (m::<N>(2 * p + 1, y) << 16);
            y += 1;
        }
        p += 1;
    }
    out
}

/// The interleaved matrix rows of the second pass: for pair `p` and the
/// outputs `8 s .. 8 s + 8`, the sixteen values `M[2p][x], M[2p + 1][x]`.
const fn second<const N: usize>() -> [[[i16; 16]; 4]; 16] {
    let mut out = [[[0i16; 16]; 4]; 16];
    let mut p = 0;
    while p < N / 2 {
        let mut s = 0;
        while s < N / 8 {
            let mut i = 0;
            while i < 8 {
                let x = 8 * s + i;
                out[p][s][2 * i] = m::<N>(2 * p, x) as i16;
                out[p][s][2 * i + 1] = m::<N>(2 * p + 1, x) as i16;
                i += 1;
            }
            s += 1;
        }
        p += 1;
    }
    out
}

static FIRST8: [[i32; 32]; 16] = first::<8>();
static FIRST16: [[i32; 32]; 16] = first::<16>();
static FIRST32: [[i32; 32]; 16] = first::<32>();
static SECOND8: [[[i16; 16]; 4]; 16] = second::<8>();
static SECOND16: [[[i16; 16]; 4]; 16] = second::<16>();
static SECOND32: [[[i16; 16]; 4]; 16] = second::<32>();

/// Four 16-bit values as one 64-bit lane.
#[inline]
fn q16(a: &[i16]) -> i64 {
    let mut v = 0u64;
    for (k, &x) in a.iter().take(4).enumerate() {
        v |= u64::from(x as u16) << (16 * k);
    }
    v as i64
}

/// Two 32-bit values as one 64-bit lane.
#[inline]
fn q32(a: i32, b: i32) -> i64 {
    (u64::from(a as u32) | u64::from(b as u32) << 32) as i64
}

/// Sixteen 16-bit values.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16(a: &[i16; 16]) -> __m256i {
    _mm256_set_m128i(
        _mm_set_epi64x(q16(&a[12..]), q16(&a[8..])),
        _mm_set_epi64x(q16(&a[4..]), q16(&a[..4])),
    )
}

/// Eight 32-bit values from `at`, zeros past the end of `s`.
#[inline]
#[target_feature(enable = "avx2")]
fn ld32x8(s: &[i32], at: usize) -> __m256i {
    match s.get(at..).and_then(|t| t.first_chunk::<8>()) {
        Some(a) => _mm256_set_m128i(
            _mm_set_epi64x(q32(a[6], a[7]), q32(a[4], a[5])),
            _mm_set_epi64x(q32(a[2], a[3]), q32(a[0], a[1])),
        ),
        None => _mm256_setzero_si256(),
    }
}

/// Eight 32-bit values to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st32x8(d: &mut [i32], at: usize, v: __m256i) {
    let w = |q: i64| [q as i32, (q >> 32) as i32];
    let (a, b) = (
        w(_mm256_extract_epi64::<0>(v)),
        w(_mm256_extract_epi64::<1>(v)),
    );
    let (c, e) = (
        w(_mm256_extract_epi64::<2>(v)),
        w(_mm256_extract_epi64::<3>(v)),
    );
    if let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) {
        *d = [a[0], a[1], b[0], b[1], c[0], c[1], e[0], e[1]];
    }
}

/// Sixteen values of `v` (16-bit) to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16x16(d: &mut [i16], at: usize, v: __m256i) {
    let w = |q: i64| {
        let q = q as u64;
        [
            q as i16,
            (q >> 16) as i16,
            (q >> 32) as i16,
            (q >> 48) as i16,
        ]
    };
    let parts = [
        w(_mm256_extract_epi64::<0>(v)),
        w(_mm256_extract_epi64::<1>(v)),
        w(_mm256_extract_epi64::<2>(v)),
        w(_mm256_extract_epi64::<3>(v)),
    ];
    if let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<16>()) {
        for (i, p) in parts.iter().enumerate() {
            d[4 * i..4 * i + 4].copy_from_slice(p);
        }
    }
}

/// Eight values of `v` (16-bit) to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16x8(d: &mut [i16], at: usize, v: __m128i) {
    let w = |q: i64| {
        let q = q as u64;
        [
            q as i16,
            (q >> 16) as i16,
            (q >> 32) as i16,
            (q >> 48) as i16,
        ]
    };
    let (a, b) = (w(_mm_cvtsi128_si64(v)), w(_mm_extract_epi64::<1>(v)));
    if let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) {
        d[..4].copy_from_slice(&a);
        d[4..].copy_from_slice(&b);
    }
}

/// Sixteen coefficients of row `k` from column `x`, as 16-bit values.
#[inline]
#[target_feature(enable = "avx2")]
fn row16(c: &[i32], at: usize) -> __m256i {
    let v = _mm256_packs_epi32(ld32x8(c, at), ld32x8(c, at + 8));
    _mm256_permute4x64_epi64::<0b11_01_10_00>(v)
}

#[target_feature(enable = "avx2")]
pub(super) fn inverse<const N: usize>(
    coeffs: &[i32],
    mut out: Sink<'_>,
    (cols, rows): (usize, usize),
    bit_depth: u32,
) {
    let (first, second) = match N {
        8 => (&FIRST8, &SECOND8),
        16 => (&FIRST16, &SECOND16),
        _ => (&FIRST32, &SECOND32),
    };
    // The intermediate values, rows of N; columns past `cols` (rounded up
    // to the strip) are never read.
    let mut mid = [[0i16; N]; N];
    let mid = mid.as_flattened_mut();
    let pairs = rows.div_ceil(2);
    let row = |k: usize, x: usize| -> __m256i {
        if k < rows {
            row16(coeffs, k * N + x)
        } else {
            _mm256_setzero_si256()
        }
    };
    if N >= 16 {
        for x in (0..cols).step_by(16) {
            let mut lo = [_mm256_setzero_si256(); 16];
            let mut hi = [_mm256_setzero_si256(); 16];
            for p in 0..pairs {
                let (a, b) = (row(2 * p, x), row(2 * p + 1, x));
                lo[p] = _mm256_unpacklo_epi16(a, b);
                hi[p] = _mm256_unpackhi_epi16(a, b);
            }
            for y in 0..N {
                let (mut sl, mut sh) = (_mm256_set1_epi32(64), _mm256_set1_epi32(64));
                for p in 0..pairs {
                    let f = _mm256_set1_epi32(first[p][y]);
                    sl = _mm256_add_epi32(sl, _mm256_madd_epi16(lo[p], f));
                    sh = _mm256_add_epi32(sh, _mm256_madd_epi16(hi[p], f));
                }
                let v = _mm256_packs_epi32(_mm256_srai_epi32::<7>(sl), _mm256_srai_epi32::<7>(sh));
                st16x16(mid, y * N + x, v);
            }
        }
    } else {
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for p in 0..pairs {
            let half = |k: usize| -> __m128i {
                if k < rows {
                    let v = _mm256_packs_epi32(ld32x8(coeffs, k * N), _mm256_setzero_si256());
                    let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(v);
                    _mm256_castsi256_si128(v)
                } else {
                    _mm_setzero_si128()
                }
            };
            let (a, b) = (half(2 * p), half(2 * p + 1));
            lo[p] = _mm_unpacklo_epi16(a, b);
            hi[p] = _mm_unpackhi_epi16(a, b);
        }
        for y in 0..N {
            let (mut sl, mut sh) = (_mm_set1_epi32(64), _mm_set1_epi32(64));
            for p in 0..pairs {
                let f = _mm_set1_epi32(first[p][y]);
                sl = _mm_add_epi32(sl, _mm_madd_epi16(lo[p], f));
                sh = _mm_add_epi32(sh, _mm_madd_epi16(hi[p], f));
            }
            let v = _mm_packs_epi32(_mm_srai_epi32::<7>(sl), _mm_srai_epi32::<7>(sh));
            st16x8(mid, y * N, v);
        }
    }
    // Second pass: each row of intermediate values times the matrix.
    let shift = _mm_cvtsi32_si128((20 - bit_depth) as i32);
    let add = _mm256_set1_epi32(1 << (19 - bit_depth));
    let pairs = cols.div_ceil(2);
    let mut table = [_mm256_setzero_si256(); 64];
    for p in 0..pairs {
        for s in 0..N / 8 {
            table[p * 4 + s] = ld16(&second[p][s]);
        }
    }
    for y in 0..N {
        let mut acc = [add; 4];
        let t = &mid[y * N..y * N + N];
        for p in 0..pairs {
            let pair = (i32::from(t[2 * p] as u16))
                | (i32::from(t.get(2 * p + 1).copied().unwrap_or(0)) << 16);
            let v = _mm256_set1_epi32(pair);
            for s in 0..N / 8 {
                acc[s] = _mm256_add_epi32(acc[s], _mm256_madd_epi16(table[p * 4 + s], v));
            }
        }
        for (s, a) in acc.iter().enumerate().take(N / 8) {
            let v = _mm256_sra_epi32(*a, shift);
            match &mut out {
                Sink::Res(r) => st32x8(r, y * N + 8 * s, v),
                Sink::Add(d, stride) => add8(d, y * *stride + 8 * s, v),
            }
        }
    }
}

/// The 4×4 matrix: the DCT's rows 0, 8, 16 and 24, or the DST.
const fn m4(dst: bool, k: usize, n: usize) -> i32 {
    if dst {
        DST4[k][n] as i32
    } else {
        DCT32[k * 8][n] as i32
    }
}

/// Entries `k` and `k + 1` of column `n` as one 32-bit pair.
const fn pair4(dst: bool, k: usize, n: usize) -> i32 {
    (m4(dst, k, n) & 0xffff) | (m4(dst, k + 1, n) << 16)
}

/// For each pair of rows `(0, 1)` and `(2, 3)`, the pairs of every column.
const fn table4(dst: bool) -> [[i32; 4]; 2] {
    [
        [
            pair4(dst, 0, 0),
            pair4(dst, 0, 1),
            pair4(dst, 0, 2),
            pair4(dst, 0, 3),
        ],
        [
            pair4(dst, 2, 0),
            pair4(dst, 2, 1),
            pair4(dst, 2, 2),
            pair4(dst, 2, 3),
        ],
    ]
}

static DCT4: [[i32; 4]; 2] = table4(false);
static DST4T: [[i32; 4]; 2] = table4(true);

/// The four 32-bit lanes of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: __m128i) -> [i32; 4] {
    let (a, b) = (_mm_cvtsi128_si64(v), _mm_extract_epi64::<1>(v));
    [a as i32, (a >> 32) as i32, b as i32, (b >> 32) as i32]
}

#[target_feature(enable = "avx2")]
pub(super) fn inverse4(c: &[i32; 16], mut out: Sink<'_>, dst: bool, bit_depth: u32) {
    let t = if dst { &DST4T } else { &DCT4 };
    let tv = |p: usize| _mm_set_epi32(t[p][3], t[p][2], t[p][1], t[p][0]);
    let (t01, t23) = (tv(0), tv(1));
    let clip = |v: i32| v.clamp(-32768, 32767);
    let row = |k: usize| -> [i32; 4] { [c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]] };
    // First pass, column x of the intermediates: sum over k of
    // M[k][y] · c[k][x]. Lanes are y: the pairs (c[0][x], c[1][x]) and
    // (c[2][x], c[3][x]) broadcast, times the matrix pairs of each y.
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let mut mid = [[0i32; 4]; 4];
    for x in 0..4 {
        let a = _mm_set1_epi32((r0[x] & 0xffff) | (r1[x] << 16));
        let b = _mm_set1_epi32((r2[x] & 0xffff) | (r3[x] << 16));
        let sum = _mm_add_epi32(_mm_madd_epi16(a, t01), _mm_madd_epi16(b, t23));
        let v = lanes(_mm_srai_epi32::<7>(_mm_add_epi32(sum, _mm_set1_epi32(64))));
        for y in 0..4 {
            mid[y][x] = clip(v[y]);
        }
    }
    // Second pass, row y: sum over k of M[k][x] · mid[y][k], lanes x.
    let shift = _mm_cvtsi32_si128((20 - bit_depth) as i32);
    let add = _mm_set1_epi32(1 << (19 - bit_depth));
    for (y, m) in mid.iter().enumerate() {
        let a = _mm_set1_epi32((m[0] & 0xffff) | (m[1] << 16));
        let b = _mm_set1_epi32((m[2] & 0xffff) | (m[3] << 16));
        let sum = _mm_add_epi32(_mm_madd_epi16(a, t01), _mm_madd_epi16(b, t23));
        let v = _mm_sra_epi32(_mm_add_epi32(sum, add), shift);
        match &mut out {
            Sink::Res(r) => {
                if let Some(r) = r.get_mut(4 * y..4 * y + 4) {
                    r.copy_from_slice(&lanes(v));
                }
            }
            Sink::Add(d, stride) => add4(d, y * *stride, v),
        }
    }
}
