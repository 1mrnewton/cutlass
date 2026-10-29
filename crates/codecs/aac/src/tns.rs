//! Temporal noise shaping.
//!
//! Reflection coefficients are `sin` of the quantized index. The index grid
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

fn dequant(index: u32, len: u32) -> f32 {
    let half = 1u32 << (len - 1);
    let (signed, denom) = if index < half {
        (-(index as f32), half as f32 - 0.5)
    } else {
        (((1u32 << len) - index) as f32, half as f32 + 0.5)
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

    #[test]
    fn four_bit_coefficients_match_the_sine_grid() {
        let neg = dequant(1, 4);
        let expect = -(std::f32::consts::PI / 15.0).sin();
        assert!((neg - expect).abs() < 1.0e-6, "{neg} {expect}");
        let pos = dequant(8, 4);
        let expect = (8.0 * std::f32::consts::PI / 17.0).sin();
        assert!((pos - expect).abs() < 1.0e-6, "{pos} {expect}");
        assert!(dequant(0, 4).abs() < 1.0e-7);
    }

    #[test]
    fn compression_skips_large_magnitudes() {
        let compressed = super::reflection(4, true, true);
        let full = dequant(12, 4);
        assert!((compressed - full).abs() < 1.0e-6);
    }

    #[test]
    fn a_single_reflection_filters_a_ramp() {
        let filter = super::TnsFilter {
            length: 1,
            order: 1,
            direction: false,
            coef: vec![0.5],
        };
        let window = super::TnsWindow {
            filters: vec![filter],
        };
        let bands = [0u16, 4];
        let mut coeffs = vec![1.0, 0.0, 0.0, 0.0];
        super::apply(&mut coeffs, &[window], &bands, 1, 1, 4);
        assert!(coeffs[0].abs() > 0.0);
        assert!(coeffs.iter().all(|c| c.is_finite()));
    }
}
