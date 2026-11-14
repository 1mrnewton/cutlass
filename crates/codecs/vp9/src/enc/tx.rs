//! Forward transforms and quantization.
//!
//! Each forward transform is the inverse of the decoder's: the matrices
//! of the 1-D inverse transforms are measured by running the decoder's own
//! integer transforms on impulses, inverted once, and scaled by the
//! decoder's final shift. Any forward transform decodes correctly (the
//! encoder reconstructs with the decoder's inverse); this one makes the
//! reconstruction match the source as closely as the quantizer allows.

use std::sync::OnceLock;

use crate::dsp::itx1d::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};
use crate::dsp::lanes::{L, Lanes};

type V = L<i64, 1>;

/// Forward matrices by size (4, 8, 16, 32) and kind (DCT, ADST), row
/// major: `f[k * n + j]` multiplies sample `j` into coefficient `k`.
struct Mats {
    m: [[Vec<f32>; 2]; 4],
    /// The DCT's rows fold about their middle: even rows are symmetric in
    /// their input, odd ones antisymmetric. These are the halves (even
    /// rows, then odd, each `n / 2` wide) the folded pass multiplies by,
    /// when the matrix folds to within rounding.
    half: [Option<Vec<f32>>; 4],
}

fn inverse_1d(n: usize, adst: bool, i: &[V], o: &mut [V]) {
    match (n, adst) {
        (4, false) => idct4(i, o),
        (4, true) => iadst4(i, o),
        (8, false) => idct8(i, o),
        (8, true) => iadst8(i, o),
        (16, false) => idct16(i, o),
        (16, true) => iadst16(i, o),
        _ => idct32(i, o),
    }
}

/// The inverse of an `n`×`n` matrix (Gauss-Jordan, partial pivoting).
fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap(piv * n + c, col * n + c);
                inv.swap(piv * n + c, col * n + c);
            }
        }
        let d = m[col * n + col];
        if d.abs() < 1e-12 {
            continue;
        }
        for c in 0..n {
            m[col * n + c] /= d;
            inv[col * n + c] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                if f != 0.0 {
                    for c in 0..n {
                        m[r * n + c] -= f * m[col * n + c];
                        inv[r * n + c] -= f * inv[col * n + c];
                    }
                }
            }
        }
    }
    inv
}

fn mats() -> &'static Mats {
    static M: OnceLock<Mats> = OnceLock::new();
    M.get_or_init(|| {
        let build = |log2: usize, adst: bool| -> Vec<f32> {
            let n = 4usize << log2;
            // inv[j * n + k]: output sample j of the impulse at coefficient k.
            let scale = 1i64 << 16;
            let mut inv = vec![0.0f64; n * n];
            let mut input = vec![V::ZERO; n];
            let mut out = vec![V::ZERO; n];
            for k in 0..n {
                input.iter_mut().for_each(|v| *v = V::ZERO);
                input[k] = L([scale]);
                inverse_1d(n, adst && n <= 16, &input, &mut out);
                for j in 0..n {
                    inv[j * n + k] = out[j].0[0] as f64 / scale as f64;
                }
            }
            invert(&inv, n).iter().map(|&v| v as f32).collect()
        };
        let m: [[Vec<f32>; 2]; 4] = core::array::from_fn(|l| [build(l, false), build(l, true)]);
        let half = core::array::from_fn(|l| {
            let n = 4usize << l;
            let h = n / 2;
            let full = &m[l][0];
            let mut out = vec![0.0f32; n * h];
            let mut worst = 0.0f32;
            let mut peak = 0.0f32;
            for k in 0..n {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                for j in 0..h {
                    let (a, b) = (full[k * n + j], full[k * n + n - 1 - j]);
                    out[((k % 2) * h + k / 2) * h + j] = (a + sign * b) / 2.0;
                    worst = worst.max((a - sign * b).abs());
                    peak = peak.max(a.abs());
                }
            }
            (worst <= peak * 1e-3).then_some(out)
        });
        Mats { m, half }
    })
}

/// The final shift of the decoder's inverse by size.
fn shift_of(log2n: usize) -> u32 {
    match log2n {
        0 => 4,
        1 => 5,
        _ => 6,
    }
}

/// The forward transform of an `n`×`n` residual (rows `stride` apart)
/// into `out`, transposed as the decoder stores coefficients
/// (`out[c * n + r]` is row `r`, column `c`). `col_adst` and `row_adst`
/// pick ADST down the columns and along the rows.
pub fn forward(
    res: &[i32],
    stride: usize,
    log2n: usize,
    col_adst: bool,
    row_adst: bool,
    out: &mut [f32],
) {
    let m = mats();
    let fc = &m.m[log2n][usize::from(col_adst)];
    let fr = &m.m[log2n][usize::from(row_adst)];
    let gain = (1u32 << shift_of(log2n)) as f32;
    let half = m.half[log2n].as_deref();
    let hc = if col_adst { None } else { half };
    let hr = if row_adst { None } else { half };
    cutlass_simd::vp9::wide(|| match log2n {
        0 => fwd::<4>(res, stride, (fc, hc), (fr, hr), gain, out),
        1 => fwd::<8>(res, stride, (fc, hc), (fr, hr), gain, out),
        2 => fwd::<16>(res, stride, (fc, hc), (fr, hr), gain, out),
        _ => fwd::<32>(res, stride, (fc, hc), (fr, hr), gain, out),
    });
}

/// `X = Fc · res · Frᵀ`, stored transposed. Both passes are sums of
/// scaled rows, so that they run in vector lanes; a DCT pass folds its
/// rows first (`half`) and multiplies by half as many weights.
#[inline(always)]
fn fwd<const N: usize>(
    res: &[i32],
    stride: usize,
    (fc, hc): (&[f32], Option<&[f32]>),
    (fr, hr): (&[f32], Option<&[f32]>),
    gain: f32,
    out: &mut [f32],
) {
    let mut src = [[0.0f32; N]; N];
    for (j, row) in src.iter_mut().enumerate() {
        let Some(r) = res[j * stride..].first_chunk::<N>() else {
            return;
        };
        for x in 0..N {
            row[x] = r[x] as f32;
        }
    }
    // t[k] = Σj Fc[k][j] · src[j], then its transpose.
    let mut tt = [[0.0f32; N]; N];
    pass::<N>(&src, fc, hc, |k, acc| {
        for x in 0..N {
            tt[x][k] = acc[x];
        }
    });
    // out[c] = gain · Σx Fr[c][x] · tᵀ[x].
    pass::<N>(&tt, fr, hr, |c, acc| {
        if let Some(o) = out[c * N..].first_chunk_mut::<N>() {
            for r in 0..N {
                o[r] = acc[r] * gain;
            }
        }
    });
}

/// One pass: `put(k, Σj w[k][j] · v[j])` for each output `k`, with the
/// rows folded when `half` holds the folded weights.
#[inline(always)]
fn pass<const N: usize>(
    v: &[[f32; N]; N],
    w: &[f32],
    half: Option<&[f32]>,
    mut put: impl FnMut(usize, &[f32; N]),
) {
    match half {
        Some(hw) if N >= 8 => {
            let h = N / 2;
            // Sums and differences of the rows about the middle.
            let mut sd = [[[0.0f32; N]; N]; 2];
            for j in 0..h {
                for x in 0..N {
                    let (a, b) = (v[j][x], v[N - 1 - j][x]);
                    sd[0][j][x] = a + b;
                    sd[1][j][x] = a - b;
                }
            }
            for k in 0..N {
                let (par, row) = (k % 2, k / 2);
                let ws = &hw[(par * h + row) * h..(par * h + row) * h + h];
                let mut acc = [0.0f32; N];
                for (j, &wj) in ws.iter().enumerate() {
                    for x in 0..N {
                        acc[x] += wj * sd[par][j][x];
                    }
                }
                put(k, &acc);
            }
        }
        _ => {
            for k in 0..N {
                let mut acc = [0.0f32; N];
                for (j, s) in v.iter().enumerate() {
                    let wj = w[k * N + j];
                    for x in 0..N {
                        acc[x] += wj * s[x];
                    }
                }
                put(k, &acc);
            }
        }
    }
}

/// The Walsh-Hadamard of the lossless mode, libvpx `vp9_fwht4x4`, into
/// `out` transposed.
pub fn fwht(res: &[i32], stride: usize, out: &mut [i32]) {
    let mut t = [0i32; 16];
    for col in 0..4 {
        let mut a = res[col];
        let mut b = res[stride + col];
        let mut c = res[2 * stride + col];
        let mut d = res[3 * stride + col];
        a += b;
        d -= c;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= c;
        d += b;
        t[col] = a;
        t[4 + col] = c;
        t[8 + col] = d;
        t[12 + col] = b;
    }
    for row in 0..4 {
        let mut a = t[row * 4];
        let mut b = t[row * 4 + 1];
        let mut c = t[row * 4 + 2];
        let mut d = t[row * 4 + 3];
        a += b;
        d -= c;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= c;
        d += b;
        // Output (row, k) to transposed positions.
        out[row] = a * 4;
        out[4 + row] = c * 4;
        out[8 + row] = d * 4;
        out[12 + row] = b * 4;
    }
}

/// How much a forward transform of each size scales energy: the sum of
/// squared coefficients over the sum of squared residuals, measured once
/// on a pseudo-random residual. Errors in coefficients over this are
/// errors in samples (the transforms are near orthogonal).
pub fn gain(log2n: usize) -> f32 {
    static G: OnceLock<[f32; 4]> = OnceLock::new();
    G.get_or_init(|| {
        core::array::from_fn(|l| {
            let n = 4usize << l;
            let mut x = 12_345u32;
            let res: Vec<i32> = (0..n * n)
                .map(|_| {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((x >> 16) % 61) as i32 - 30
                })
                .collect();
            let mut out = vec![0.0f32; n * n];
            forward(&res, n, l, false, false, &mut out);
            let e: f64 = out.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let r: f64 = res.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            (e / r) as f32
        })
    })[log2n.min(3)]
}

#[cfg(test)]
mod tests {
    use super::forward;
    use crate::dsp::itx::{Kind, add};

    /// Forward then the decoder's inverse gives back the residual.
    #[test]
    fn forward_inverts_the_decoder() {
        for log2n in 0..4 {
            let n = 4usize << log2n;
            for t in 0..4u8 {
                if n == 32 && t != 0 {
                    continue;
                }
                let res: Vec<i32> = (0..n * n)
                    .map(|i| ((i * 37 + i / n * 11) % 61) as i32 - 30)
                    .collect();
                let mut f = vec![0.0f32; n * n];
                forward(&res, n, log2n, t & 1 == 1, t & 2 == 2, &mut f);
                let mut coef: Vec<i32> = f.iter().map(|&v| v.round() as i32).collect();
                let mut dst = vec![128u8; n * n];
                let (col, row) = (
                    if t & 1 == 1 { Kind::Adst } else { Kind::Dct },
                    if t & 2 == 2 { Kind::Adst } else { Kind::Dct },
                );
                add(
                    &mut dst,
                    n,
                    &mut coef,
                    log2n,
                    col,
                    row,
                    n * n,
                    (n, n),
                    255,
                    &mut crate::dsp::itx::Tmp::new(),
                );
                let worst = dst
                    .iter()
                    .zip(&res)
                    .map(|(&d, &r)| (i32::from(d) - 128 - r).abs())
                    .max()
                    .unwrap_or(0);
                assert!(worst <= 1, "n {n} type {t} worst {worst}");
            }
        }
    }
}
