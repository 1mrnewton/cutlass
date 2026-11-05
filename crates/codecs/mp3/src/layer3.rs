//! Layer III main data: scalefactors, Huffman decode, requantization,
//! stereo processing, and short-block reordering.
//!
//! ISO/IEC 11172-3 clauses 2.4.1.7, 2.4.2.7 and 2.4.3.4; ISO/IEC 13818-3
//! clauses 2.4.3.2 and 2.4.3.4 for LSF scalefactors and intensity stereo.

use std::f64::consts::PI;
use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::header::{FrameHeader, Mode};
use crate::huffman::tables;
use crate::sideinfo::{GranuleInfo, SHORT};
use crate::tables::{
    LSF_PARTITIONS, PRETAB, SHORT_WIDTHS, SLEN, TABLE_SELECT, long_starts, short_starts,
};

/// Largest quantized magnitude: 15 plus 13 linbits.
pub(crate) const MAX_QUANT: usize = 15 + 8191;

/// Scalefactors of one granule of one channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Scalefactors {
    pub long: [u8; 22],
    pub short: [[u8; 3]; 13],
    /// LSF intensity: the position of this band is the illegal value.
    pub illegal_long: [bool; 22],
    pub illegal_short: [[bool; 3]; 13],
}

/// The parsed spectrum of one granule of one channel.
#[derive(Clone)]
pub(crate) struct Parsed {
    pub info: GranuleInfo,
    pub scalefactors: Scalefactors,
    pub quant: [i32; 576],
    /// One past the last line that can be non-zero.
    pub nonzero: usize,
}

impl Default for Parsed {
    fn default() -> Self {
        Self {
            info: GranuleInfo::default(),
            scalefactors: Scalefactors::default(),
            quant: [0; 576],
            nonzero: 0,
        }
    }
}

/// Where long and short scalefactor bands meet in one granule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Long bands `0..long_end` are coded.
    pub long_end: usize,
    /// Short bands `short_start..13` are coded; 13 means none.
    pub short_start: usize,
    /// First line of the short region (before reordering).
    pub short_line: usize,
}

pub(crate) fn layout(header: &FrameHeader, info: &GranuleInfo) -> Layout {
    if info.window_switching && info.block_type == SHORT {
        if info.mixed {
            let long_end = if header.version.lsf() { 6 } else { 8 };
            let starts = long_starts(header.rate_table());
            Layout {
                long_end,
                short_start: 3,
                short_line: usize::from(starts[long_end]),
            }
        } else {
            Layout {
                long_end: 0,
                short_start: 0,
                short_line: 0,
            }
        }
    } else {
        Layout {
            long_end: 22,
            short_start: 13,
            short_line: 576,
        }
    }
}

/// Read the scalefactors of one granule.
///
/// `previous` is granule 0 of the same channel, used by MPEG-1 `scfsi`.
/// `intensity_right` selects the LSF intensity partition of the right channel.
pub(crate) fn read_scalefactors(
    reader: &mut BitReader<'_>,
    header: &FrameHeader,
    info: &mut GranuleInfo,
    scfsi: Option<&[bool; 4]>,
    previous: &Scalefactors,
    intensity_right: bool,
) -> Result<Scalefactors, CodecError> {
    let mut sf = Scalefactors::default();
    let shape = layout(header, info);
    if !header.version.lsf() {
        let compress = usize::from(info.scalefac_compress & 15);
        let slen1 = u32::from(SLEN[0][compress]);
        let slen2 = u32::from(SLEN[1][compress]);
        if shape.short_start < 13 {
            for band in 0..shape.long_end {
                sf.long[band] = reader.read(slen1, "scalefactor")? as u8;
            }
            for band in shape.short_start..12 {
                let slen = if band < 6 { slen1 } else { slen2 };
                for w in 0..3 {
                    sf.short[band][w] = reader.read(slen, "scalefactor")? as u8;
                }
            }
        } else {
            const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
            for (group, (from, to)) in GROUPS.iter().copied().enumerate() {
                let reuse = scfsi.is_some_and(|bits| bits[group]);
                let slen = if group < 2 { slen1 } else { slen2 };
                for band in from..to {
                    sf.long[band] = if reuse {
                        previous.long[band]
                    } else {
                        reader.read(slen, "scalefactor")? as u8
                    };
                }
            }
        }
        if intensity_right {
            for band in 0..22 {
                sf.illegal_long[band] = sf.long[band] >= 7;
            }
            for band in 0..13 {
                for w in 0..3 {
                    sf.illegal_short[band][w] = sf.short[band][w] >= 7;
                }
            }
        }
    } else {
        let (slen, case) = lsf_slen(info, intensity_right);
        let block = if shape.short_start == 13 {
            0
        } else if shape.long_end == 0 {
            1
        } else {
            2
        };
        let counts = LSF_PARTITIONS[case][block];
        let mut values = [0u8; 39];
        let mut illegal = [false; 39];
        let mut n = 0usize;
        for (part, &count) in counts.iter().enumerate() {
            let bits = u32::from(slen[part]);
            for _ in 0..count {
                if n >= values.len() {
                    return Err(CodecError::Malformed {
                        at_bit: reader.pos() as u64,
                        what: "scalefactor partitions",
                    });
                }
                let value = reader.read(bits, "scalefactor")? as u8;
                values[n] = value;
                illegal[n] = intensity_right && bits > 0 && u32::from(value) == (1 << bits) - 1;
                n += 1;
            }
        }
        let mut index = 0usize;
        for band in 0..shape.long_end.min(21) {
            sf.long[band] = values[index];
            sf.illegal_long[band] = illegal[index];
            index += 1;
        }
        if shape.short_start < 13 {
            for band in shape.short_start..12 {
                for w in 0..3 {
                    sf.short[band][w] = values[index];
                    sf.illegal_short[band][w] = illegal[index];
                    index += 1;
                }
            }
        }
    }
    // The last band carries no scalefactor; its intensity position is the
    // position of the band below it (ISO/IEC 11172-3 2.4.3.4.9.3).
    sf.illegal_long[21] = sf.illegal_long[20];
    sf.illegal_short[12] = sf.illegal_short[11];
    Ok(sf)
}

/// LSF `slen` per partition and the partition table row.
fn lsf_slen(info: &mut GranuleInfo, intensity_right: bool) -> ([u8; 4], usize) {
    let compress = u32::from(info.scalefac_compress);
    if intensity_right {
        let isc = compress >> 1;
        if isc < 180 {
            (
                [(isc / 36) as u8, ((isc % 36) / 6) as u8, (isc % 6) as u8, 0],
                3,
            )
        } else if isc < 244 {
            let isc = isc - 180;
            (
                [(isc / 16) as u8, ((isc % 16) / 4) as u8, (isc % 4) as u8, 0],
                4,
            )
        } else {
            let isc = isc - 244;
            ([(isc / 3) as u8, (isc % 3) as u8, 0, 0], 5)
        }
    } else if compress < 400 {
        (
            [
                (compress / 80) as u8,
                ((compress / 16) % 5) as u8,
                ((compress % 16) / 4) as u8,
                (compress % 4) as u8,
            ],
            0,
        )
    } else if compress < 500 {
        let c = compress - 400;
        ([(c / 20) as u8, ((c / 4) % 5) as u8, (c % 4) as u8, 0], 1)
    } else {
        let c = compress - 500;
        info.preflag = true;
        ([(c / 3) as u8, (c % 3) as u8, 0, 0], 2)
    }
}

/// Huffman-decode the big-value and count1 regions into `quant`.
///
/// `reader` ends at the granule's `part2_3_length`. Returns one past the
/// last line that may be non-zero.
pub(crate) fn read_spectrum(
    reader: &mut BitReader<'_>,
    header: &FrameHeader,
    info: &GranuleInfo,
    quant: &mut [i32; 576],
) -> Result<usize, CodecError> {
    let all = tables();
    let table = header.rate_table();
    let starts = long_starts(table);
    let big_end = usize::from(info.big_values) * 2;
    let (mut r0, mut r1) = if info.window_switching {
        let r0 = if info.block_type == SHORT {
            if table == 8 { 72 } else { 36 }
        } else if !header.version.lsf() {
            36
        } else if table == 8 {
            108
        } else {
            54
        };
        (r0, 576)
    } else {
        let a = usize::from(info.region0_count) + 1;
        let b = (usize::from(info.region0_count) + usize::from(info.region1_count) + 2).min(22);
        (usize::from(starts[a]), usize::from(starts[b]))
    };
    r0 = r0.min(big_end);
    r1 = r1.min(big_end);
    let bounds = [(0, r0), (r0, r1), (r1, big_end)];
    for (region, (from, to)) in bounds.iter().copied().enumerate() {
        if from >= to {
            continue;
        }
        let (index, linbits) = TABLE_SELECT[usize::from(info.table_select[region] & 31)];
        if index == 255 {
            return Err(CodecError::Malformed {
                at_bit: reader.pos() as u64,
                what: "table_select",
            });
        }
        if index == 0 {
            continue;
        }
        let Some(code) = all.big(index) else {
            return Err(CodecError::Malformed {
                at_bit: reader.pos() as u64,
                what: "table_select",
            });
        };
        let linbits = u32::from(linbits);
        let mut line = from;
        while line < to {
            let at = reader.pos() as u64;
            let Some(sym) = code.decode(reader) else {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "big_values past part2_3_length",
                });
            };
            let mut x = i32::from(sym >> 4);
            let mut y = i32::from(sym & 15);
            if linbits > 0 && x == 15 {
                x += reader.read(linbits, "linbits")? as i32;
            }
            if x != 0 && reader.bit("sign")? {
                x = -x;
            }
            if linbits > 0 && y == 15 {
                y += reader.read(linbits, "linbits")? as i32;
            }
            if y != 0 && reader.bit("sign")? {
                y = -y;
            }
            quant[line] = x;
            quant[line + 1] = y;
            line += 2;
        }
    }
    let quad = if info.count1_table_b {
        None
    } else {
        Some(&all.quad_a)
    };
    let mut line = big_end;
    while line + 4 <= 576 && reader.left() > 0 {
        let mark = reader.clone();
        let sym = match quad {
            Some(code) => match code.decode(reader) {
                Some(sym) => sym,
                None => break,
            },
            None => match reader.read(4, "count1") {
                Ok(bits) => 15 - bits as u8,
                Err(_) => break,
            },
        };
        let mut values = [0i32; 4];
        let mut overrun = false;
        for (slot, value) in values.iter_mut().enumerate() {
            if sym & (8 >> slot) != 0 {
                match reader.bit("sign") {
                    Ok(true) => *value = -1,
                    Ok(false) => *value = 1,
                    Err(_) => overrun = true,
                }
            }
        }
        if overrun {
            // A quadruple that runs past part2_3_length belongs to the next
            // granule's bits; it is not part of this spectrum.
            *reader = mark;
            break;
        }
        quant[line..line + 4].copy_from_slice(&values);
        line += 4;
    }
    Ok(line)
}

fn pow43() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=MAX_QUANT)
            .map(|value| (value as f64).powf(4.0 / 3.0) as f32)
            .collect()
    })
}

/// Requantize one granule into `xr` (before reordering).
pub(crate) fn requantize(header: &FrameHeader, parsed: &Parsed, xr: &mut [f32; 576]) {
    let info = &parsed.info;
    let sf = &parsed.scalefactors;
    let table = header.rate_table();
    let shape = layout(header, info);
    let gain = f64::from(info.global_gain) - 210.0;
    let multiplier = if info.scalefac_scale { 1.0 } else { 0.5 };
    let pow = pow43();
    *xr = [0.0; 576];
    let end = parsed.nonzero.min(576);
    let lstarts = long_starts(table);
    for band in 0..shape.long_end {
        let from = usize::from(lstarts[band]);
        if from >= end {
            break;
        }
        let to = usize::from(lstarts[band + 1]).min(end);
        let pre = if info.preflag { PRETAB[band] } else { 0 };
        let exponent =
            gain / 4.0 - multiplier * f64::from(u32::from(sf.long[band]) + u32::from(pre));
        let scale = exponent.exp2() as f32;
        dequantize(&parsed.quant[from..to], &mut xr[from..to], pow, scale);
    }
    if shape.short_start < 13 {
        let sstarts = short_starts(table);
        let widths = SHORT_WIDTHS[table];
        for band in shape.short_start..13 {
            let width = usize::from(widths[band]);
            let base = 3 * usize::from(sstarts[band]);
            if base >= end {
                break;
            }
            for w in 0..3 {
                let exponent = (gain - 8.0 * f64::from(info.subblock_gain[w])) / 4.0
                    - multiplier * f64::from(sf.short[band][w]);
                let scale = exponent.exp2() as f32;
                let from = base + w * width;
                let to = (from + width).min(end);
                dequantize(&parsed.quant[from..to], &mut xr[from..to], pow, scale);
            }
        }
    }
}

fn dequantize(quant: &[i32], xr: &mut [f32], pow: &[f32], scale: f32) {
    for (out, &q) in xr.iter_mut().zip(quant) {
        let value = pow[(q.unsigned_abs() as usize).min(MAX_QUANT)] * scale;
        *out = if q < 0 { -value } else { value };
    }
}

/// Joint-stereo processing of one granule, in place, before reordering.
pub(crate) fn stereo(
    header: &FrameHeader,
    left: &mut [f32; 576],
    right: &mut [f32; 576],
    right_parsed: &Parsed,
) {
    if header.mode != Mode::JointStereo {
        return;
    }
    let ms = header.mode_extension & 2 != 0;
    let intensity = header.mode_extension & 1 != 0;
    if !intensity {
        if ms {
            mid_side(left, right, 0, 576);
        }
        return;
    }
    let table = header.rate_table();
    let info = &right_parsed.info;
    let shape = layout(header, info);
    let lstarts = long_starts(table);
    let sstarts = short_starts(table);
    let widths = SHORT_WIDTHS[table];
    let quant = &right_parsed.quant;
    let sf = &right_parsed.scalefactors;
    let lsf_scale = if header.version.lsf() {
        Some(info.scalefac_compress & 1)
    } else {
        None
    };
    let process = |left: &mut [f32; 576],
                   right: &mut [f32; 576],
                   from: usize,
                   to: usize,
                   position: u8,
                   illegal: bool,
                   is_band: bool| {
        if is_band && !illegal {
            let (kl, kr) = intensity_gains(position, lsf_scale);
            for line in from..to {
                let value = left[line];
                left[line] = value * kl;
                right[line] = value * kr;
            }
        } else if ms {
            mid_side(left, right, from, to);
        }
    };
    let mut any_short_nonzero = false;
    if shape.short_start < 13 {
        for w in 0..3 {
            let mut bound = shape.short_start;
            for band in (shape.short_start..13).rev() {
                let from = 3 * usize::from(sstarts[band]) + w * usize::from(widths[band]);
                let to = from + usize::from(widths[band]);
                if quant[from..to].iter().any(|q| *q != 0) {
                    bound = band + 1;
                    any_short_nonzero = true;
                    break;
                }
            }
            for band in shape.short_start..13 {
                let from = 3 * usize::from(sstarts[band]) + w * usize::from(widths[band]);
                let to = from + usize::from(widths[band]);
                let source = band.min(11);
                process(
                    left,
                    right,
                    from,
                    to,
                    sf.short[source][w],
                    sf.illegal_short[band][w],
                    band >= bound,
                );
            }
        }
    }
    if shape.long_end > 0 {
        let mut bound = shape.long_end;
        if !any_short_nonzero {
            bound = 0;
            for band in (0..shape.long_end).rev() {
                let from = usize::from(lstarts[band]);
                let to = usize::from(lstarts[band + 1]);
                if quant[from..to].iter().any(|q| *q != 0) {
                    bound = band + 1;
                    break;
                }
            }
        }
        for band in 0..shape.long_end {
            let from = usize::from(lstarts[band]);
            let to = usize::from(lstarts[band + 1]);
            let source = band.min(20);
            process(
                left,
                right,
                from,
                to,
                sf.long[source],
                sf.illegal_long[band],
                band >= bound,
            );
        }
    }
}

fn mid_side(left: &mut [f32; 576], right: &mut [f32; 576], from: usize, to: usize) {
    let half = std::f32::consts::FRAC_1_SQRT_2;
    for line in from..to {
        let m = left[line];
        let s = right[line];
        left[line] = (m + s) * half;
        right[line] = (m - s) * half;
    }
}

/// Left and right gains of an intensity position.
fn intensity_gains(position: u8, lsf_scale: Option<u16>) -> (f32, f32) {
    match lsf_scale {
        None => {
            let angle = f64::from(position) * PI / 12.0;
            let (sin, cos) = angle.sin_cos();
            let sum = sin + cos;
            ((sin / sum) as f32, (cos / sum) as f32)
        }
        Some(scale) => {
            let step = if scale == 1 { 0.5 } else { 0.25 };
            let p = f64::from(position);
            if position == 0 {
                (1.0, 1.0)
            } else if position % 2 == 1 {
                ((-step * (p + 1.0) / 2.0).exp2() as f32, 1.0)
            } else {
                (1.0, (-step * p / 2.0).exp2() as f32)
            }
        }
    }
}

/// Reorder short-block lines so subband `sb` holds `18 * sb + 3 * k + w`.
pub(crate) fn reorder(header: &FrameHeader, info: &GranuleInfo, xr: &mut [f32; 576]) {
    let shape = layout(header, info);
    if shape.short_start >= 13 {
        return;
    }
    let table = header.rate_table();
    let sstarts = short_starts(table);
    let widths = SHORT_WIDTHS[table];
    let mut scratch = [0f32; 576];
    for band in shape.short_start..13 {
        let width = usize::from(widths[band]);
        let base = 3 * usize::from(sstarts[band]);
        for j in 0..width {
            for w in 0..3 {
                scratch[base + 3 * j + w] = xr[base + w * width + j];
            }
        }
    }
    let from = shape.short_line;
    xr[from..].copy_from_slice(&scratch[from..]);
}

#[cfg(test)]
mod tests {
    use super::intensity_gains;

    #[test]
    fn mpeg1_intensity_positions_split_the_energy_by_tangent() {
        let (l, r) = intensity_gains(0, None);
        assert!((l - 0.0).abs() < 1e-7 && (r - 1.0).abs() < 1e-7);
        let (l, r) = intensity_gains(6, None);
        assert!((l - 1.0).abs() < 1e-6 && r.abs() < 1e-6);
        let (l, r) = intensity_gains(3, None);
        assert!((l - 0.5).abs() < 1e-6 && (r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lsf_intensity_positions_step_by_the_scale() {
        assert_eq!(intensity_gains(0, Some(0)), (1.0, 1.0));
        let (l, r) = intensity_gains(1, Some(0));
        assert!((l - 2f32.powf(-0.25)).abs() < 1e-6 && r == 1.0);
        let (l, r) = intensity_gains(2, Some(1));
        assert!(l == 1.0 && (r - 0.5f32.sqrt()).abs() < 1e-6);
    }
}
