//! Temporal noise shaping.
//!
//! Reflection coefficients are `sin` of the signed quantized index, so a
//! positive index is a positive coefficient. The index grid
//! follows ISO/IEC 14496-3: a resolution of 3 or 4 bits, optionally compressed
//! by dropping the largest magnitudes. `coef_res` false selects 3 bits and
//! true selects 4. Compression maps the shorter code onto that grid by
//! skipping the middle, high-magnitude codes.
//!
//! The reflections are stepped up to an LPC and applied as an all-pole filter
//! on the spectral coefficients of each window. AAC-LC order is at most 12
//! on a long window and 7 on a short window.

use cutlass_decode::CodecError;

use crate::bits::BitReader;

pub const MAX_ORDER_LONG: usize = 12;
pub const MAX_ORDER_SHORT: usize = 7;

#[derive(Clone, Debug)]
pub struct TnsFilter {
    pub length: usize,
    pub order: usize,
    pub direction: bool,
    pub coef: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct TnsWindow {
    pub filters: Vec<TnsFilter>,
}

pub fn read_tns(
    bits: &mut BitReader<'_>,
    windows: usize,
    short: bool,
) -> Result<Vec<TnsWindow>, CodecError> {
    let max_order = if short {
        MAX_ORDER_SHORT
    } else {
        MAX_ORDER_LONG
    };
    let filt_bits = if short { 1 } else { 2 };
    let length_bits = if short { 4 } else { 6 };
    let order_bits = if short { 3 } else { 5 };
    let mut out = Vec::with_capacity(windows);
    for _ in 0..windows {
        let n_filt = bits.read(filt_bits)? as usize;
        let mut filters = Vec::new();
        if n_filt > 0 {
            let coef_res = bits.read(1)? == 1;
            for _ in 0..n_filt {
                let length = bits.read(length_bits)? as usize;
                let order = bits.read(order_bits)? as usize;
                if order > max_order {
                    return Err(CodecError::Malformed {
                        at_bit: bits.position() as u64,
                        what: "tns order",
                    });
                }
                if order == 0 {
                    filters.push(TnsFilter {
                        length,
                        order,
                        direction: false,
                        coef: Vec::new(),
                    });
                    continue;
                }
                let direction = bits.read(1)? == 1;
                let coef_compress = bits.read(1)? == 1;
                let mut coef = Vec::with_capacity(order);
                let code_len = u32::from(coef_res) + 3 - u32::from(coef_compress);
                for _ in 0..order {
                    let code = bits.read(code_len)?;
                    coef.push(reflection(code, coef_res, coef_compress));
                }
                filters.push(TnsFilter {
                    length,
                    order,
                    direction,
                    coef,
                });
            }
        }
        out.push(TnsWindow { filters });
    }
    Ok(out)
}

fn reflection(code: u32, coef_res: bool, coef_compress: bool) -> f32 {
    let full_len = u32::from(coef_res) + 3;
    let index = if coef_compress {
        let half = 1u32 << (full_len - 2);
        if code < half {
            code
        } else {
            code + (1u32 << (full_len - 1))
        }
    } else {
        code
    };
    dequant(index, full_len)
}

/// The reflection coefficient of a `len`-bit two's-complement index, as
/// ISO/IEC 14496-3 4.6.9.3 defines it: `sin(index / iqfac)`, with
/// `iqfac = (2^(len-1) - 0.5) / (pi/2)` for an index of zero or more and
/// `(2^(len-1) + 0.5) / (pi/2)` below zero. A positive index gives a
/// positive coefficient. `reflections_to_lpc` and `apply_filter` use the
/// coefficient with the standard's signs, so the sign is decided here only.
fn dequant(index: u32, len: u32) -> f32 {
    let half = 1u32 << (len - 1);
    let (signed, denom) = if index < half {
        (index as f32, half as f32 - 0.5)
    } else {
        (-(((1u32 << len) - index) as f32), half as f32 + 0.5)
    };
    if denom == 0.0 {
        return 0.0;
    }
    (signed * std::f32::consts::PI / (2.0 * denom)).sin()
}

pub fn apply(
    coeffs: &mut [f32],
    windows: &[TnsWindow],
    bands: &[u16],
    tns_max_bands: usize,
    max_sfb: usize,
    window_bins: usize,
) {
    let num_swb = bands.len().saturating_sub(1);
    let limit = tns_max_bands.min(max_sfb).min(num_swb);
    if limit == 0 {
        return;
    }
    for (w, window) in windows.iter().enumerate() {
        let base = w * window_bins;
        let mut top = num_swb;
        for filter in &window.filters {
            let bottom = top.saturating_sub(filter.length);
            let start_sfb = bottom.min(limit);
            let end_sfb = top.min(limit);
            top = bottom;
            if filter.order == 0 || end_sfb <= start_sfb {
                continue;
            }
            let Some(&start_bin) = bands.get(start_sfb) else {
                continue;
            };
            let Some(&end_bin) = bands.get(end_sfb) else {
                continue;
            };
            let start = usize::from(start_bin);
            let end = usize::from(end_bin);
            if end <= start || end > window_bins {
                continue;
            }
            let lpc = reflections_to_lpc(&filter.coef);
            apply_filter(coeffs, base, start, end, filter.direction, &lpc);
        }
    }
}

fn reflections_to_lpc(k: &[f32]) -> Vec<f32> {
    let mut a = vec![0.0; k.len()];
    for (m, &km) in k.iter().enumerate() {
        let mut next = a.clone();
        for j in 0..m {
            next[j] = a[j] + km * a[m - 1 - j];
        }
        next[m] = km;
        a = next;
    }
    a
}

fn apply_filter(
    coeffs: &mut [f32],
    base: usize,
    start: usize,
    end: usize,
    direction: bool,
    lpc: &[f32],
) {
    let size = end - start;
    for m in 0..size {
        let pos = if direction { end - 1 - m } else { start + m };
        let mut value = coeffs[base + pos];
        let max_i = (m).min(lpc.len());
        for i in 1..=max_i {
            let other = if direction { pos + i } else { pos - i };
            value -= coeffs[base + other] * lpc[i - 1];
        }
        coeffs[base + pos] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::dequant;

    /// ISO/IEC 14496-3 4.6.9.3: `sin(index / iqfac)` keeps the index's sign.
    #[test]
    fn four_bit_coefficients_match_the_sine_grid() {
        let pos = dequant(1, 4);
        let expect = (std::f32::consts::PI / 15.0).sin();
        assert!((pos - expect).abs() < 1.0e-6, "{pos} {expect}");
        // Index 8 is -8 in four bits.
        let neg = dequant(8, 4);
        let expect = -(8.0 * std::f32::consts::PI / 17.0).sin();
        assert!((neg - expect).abs() < 1.0e-6, "{neg} {expect}");
        assert!(dequant(0, 4).abs() < 1.0e-7);
    }

    /// Every index of both resolutions against the standard's formula.
    #[test]
    fn every_index_matches_the_standard() {
        for len in [3u32, 4] {
            let half = 1i32 << (len - 1);
            for index in 0..(1u32 << len) {
                let signed = if (index as i32) < half {
                    index as i32
                } else {
                    index as i32 - (1 << len)
                };
                let offset = if signed >= 0 { -0.5 } else { 0.5 };
                let iqfac = (f64::from(half) + offset) / std::f64::consts::FRAC_PI_2;
                let expect = (f64::from(signed) / iqfac).sin();
                let found = f64::from(dequant(index, len));
                assert!((found - expect).abs() < 1.0e-6, "{len} {index}");
                assert_eq!(found > 0.0, signed > 0, "{len} {index}");
            }
        }
    }

    #[test]
    fn compression_skips_large_magnitudes() {
        let compressed = super::reflection(4, true, true);
        let full = dequant(12, 4);
        assert!((compressed - full).abs() < 1.0e-6);
        // A compressed code is sign-extended: 3 of three bits is +3 and
        // 7 is -1.
        assert!((super::reflection(3, true, true) - dequant(3, 4)).abs() < 1.0e-6);
        assert!((super::reflection(7, true, true) - dequant(15, 4)).abs() < 1.0e-6);
        assert!(super::reflection(7, true, true) < 0.0);
    }

    fn one_filter(coef: Vec<f32>, direction: bool) -> [super::TnsWindow; 1] {
        [super::TnsWindow {
            filters: vec![super::TnsFilter {
                length: 1,
                order: coef.len(),
                direction,
                coef,
            }],
        }]
    }

    /// One reflection `k` gives `y[n] = x[n] - k * y[n-1]`, so an impulse
    /// becomes `(-k)^n`.
    #[test]
    fn a_single_reflection_filters_an_impulse() {
        let bands = [0u16, 4];
        let mut coeffs = vec![1.0, 0.0, 0.0, 0.0];
        super::apply(&mut coeffs, &one_filter(vec![0.5], false), &bands, 1, 1, 4);
        assert_eq!(coeffs, [1.0, -0.5, 0.25, -0.125]);
    }

    /// A downward filter starts at the top line and runs toward zero.
    #[test]
    fn a_downward_filter_runs_from_the_top_line() {
        let bands = [0u16, 4];
        let mut coeffs = vec![0.0, 0.0, 0.0, 1.0];
        super::apply(&mut coeffs, &one_filter(vec![0.5], true), &bands, 1, 1, 4);
        assert_eq!(coeffs, [-0.125, 0.25, -0.5, 1.0]);
    }

    /// Reflections 0.5 and -0.25 step up to `a = [0.375, -0.25]`
    /// (`a1 = k1 + k2 * k1`, `a2 = k2`), so
    /// `y[n] = x[n] - 0.375 * y[n-1] + 0.25 * y[n-2]`.
    #[test]
    fn two_reflections_step_up_to_the_standard_filter() {
        let lpc = super::reflections_to_lpc(&[0.5, -0.25]);
        assert_eq!(lpc, [0.375, -0.25]);
        let bands = [0u16, 8];
        let mut coeffs = vec![0.0; 8];
        coeffs[0] = 1.0;
        coeffs[3] = 2.0;
        let input = coeffs.clone();
        super::apply(
            &mut coeffs,
            &one_filter(vec![0.5, -0.25], false),
            &bands,
            1,
            1,
            8,
        );
        let mut expect = [0.0f32; 8];
        for n in 0..8 {
            let mut y = input[n];
            if n >= 1 {
                y -= 0.375 * expect[n - 1];
            }
            if n >= 2 {
                y += 0.25 * expect[n - 2];
            }
            expect[n] = y;
        }
        assert_eq!(coeffs, expect);
        assert_eq!(&coeffs[..3], &[1.0, -0.375, 0.390_625]);
    }

    /// The decoder's filter undoes the encoder's analysis filter
    /// `x[n] = y[n] + sum(a[i] * y[n-i])` built from the same reflections.
    #[test]
    fn the_filter_inverts_the_analysis_filter() {
        let k = [0.6f32, -0.3, 0.2];
        let lpc = super::reflections_to_lpc(&k);
        let original: Vec<f32> = (0..16u8)
            .map(|n| f32::from(n % 5) - 2.0 + f32::from(n) * 0.125)
            .collect();
        let mut coded = original.clone();
        for n in 0..original.len() {
            for (i, a) in lpc.iter().enumerate() {
                if n > i {
                    coded[n] += a * original[n - 1 - i];
                }
            }
        }
        let bands = [0u16, 16];
        super::apply(&mut coded, &one_filter(k.to_vec(), false), &bands, 1, 1, 16);
        for (found, expect) in coded.iter().zip(&original) {
            assert!((found - expect).abs() < 1.0e-4, "{found} {expect}");
        }
    }

    /// Lines outside the filter's bands, and beyond `tns_max_bands` or
    /// `max_sfb`, are left alone.
    #[test]
    fn the_filter_stays_inside_its_bands() {
        let bands = [0u16, 2, 4, 6, 8];
        let window = one_filter(vec![0.5], false);
        let mut coeffs = vec![1.0; 8];
        // Length 1 from the top covers band 3 only; the limit of 3 bands
        // leaves nothing.
        super::apply(&mut coeffs, &window, &bands, 3, 4, 8);
        assert_eq!(coeffs, [1.0; 8]);
        super::apply(&mut coeffs, &window, &bands, 4, 3, 8);
        assert_eq!(coeffs, [1.0; 8]);
        super::apply(&mut coeffs, &window, &bands, 4, 4, 8);
        assert_eq!(coeffs, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn an_order_past_the_window_limit_is_malformed() {
        // One short window: one filter, 3-bit grid, length 1, order 7 is
        // the limit; a long window refuses order 13.
        let mut writer = crate::bits::BitWriter::with_cap(8);
        use crate::bits::BitSink;
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(1, 6).unwrap();
        writer.write_bits(13, 5).unwrap();
        writer.write_bits(0, 16).unwrap();
        let bytes = writer.finish();
        let mut bits = crate::bits::BitReader::new(&bytes);
        assert!(matches!(
            super::read_tns(&mut bits, 1, false),
            Err(cutlass_decode::CodecError::Malformed {
                what: "tns order",
                ..
            })
        ));
        let mut bits = crate::bits::BitReader::new(&bytes[..1]);
        assert!(super::read_tns(&mut bits, 1, false).is_err());
    }
}
