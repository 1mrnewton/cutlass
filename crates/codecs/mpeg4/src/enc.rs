//! A Simple-profile encoder, and the Advanced Simple encoder export uses.
//!
//! An I-VOP, then P-VOPs: one or four motion vectors per macroblock, no
//! AC prediction. [`encode_advanced`] puts a B-VOP between references and
//! can use quarter-sample motion and the MPEG quantiser. [`Pictures`] is
//! the export path: the same tools, quarter-sample on, H.263 quant, and a
//! quantiser that follows a bitrate. GMC, sprites, interlace, data
//! partitioning, resync and RVLC are not written. The batch encoders
//! refuse a size that is not a whole number of macroblocks. [`Pictures`]
//! pads one and keeps the visible size in the layer. Motion search stays
//! inside ±8 pixels so every vector fits `fcode` 1.

use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::bits::BitWriter;
use crate::block::{Quant, dequant_inter, dequant_intra};
use crate::frame::{Frame, Kind, MbRecord};
use crate::idct;
use crate::mc::{self, Pel, Rows};
use crate::tables::{
    CBPY, CoefTable, DC_CHROMA, DC_LUMA, DEFAULT_INTER_MATRIX, DEFAULT_INTRA_MATRIX, MB_TYPE_B,
    MCBPC_INTER, MCBPC_INTRA, MVD, TCOEF_ESCAPE, ZIGZAG, dc_scaler_chroma, dc_scaler_luma, tables,
};
use crate::vop::clip_block;

const TIME_RESOLUTION: u32 = 25;

/// Seconds already written into the VOP clock. `last` is the base from
/// before the latest non-B picture, which a B-VOP measures from.
#[derive(Default)]
struct TimeBase {
    seconds: u32,
    last: u32,
}

struct When<'a> {
    ticks: u32,
    resolution: u32,
    base: &'a mut TimeBase,
}

/// What `write_vol` puts in the object layer. The batch encoders keep a
/// 25 Hz clock so their streams stay bit-identical.
struct Layer {
    advanced: bool,
    quarter: bool,
    mpeg: bool,
    low_delay: bool,
    resolution: u32,
}

fn simple_layer() -> Layer {
    Layer {
        advanced: false,
        quarter: false,
        mpeg: false,
        low_delay: true,
        resolution: TIME_RESOLUTION,
    }
}

/// One picture's quantiser: the scale, and whether levels use the MPEG
/// matrices instead of the H.263 reconstruction.
#[derive(Clone, Copy)]
struct Q {
    scale: i32,
    mpeg: bool,
}

/// One intra picture as an elementary stream: the video object layer,
/// then one I-VOP. `quant` is `vop_quant`, 1 to 31.
pub fn encode_intra(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    quant: u8,
) -> Result<Vec<u8>, CodecError> {
    encode_pictures(&[(y, u, v)], width, height, quant)
}

/// An elementary stream of one I-VOP and a following P-VOP for every
/// later picture. `quant` is `vop_quant`, 1 to 31, and stays constant.
fn encode_pictures(
    frames: &[(&[u8], &[u8], &[u8])],
    width: u32,
    height: u32,
    quant: u8,
) -> Result<Vec<u8>, CodecError> {
    if frames.is_empty()
        || width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || u64::from(width) * u64::from(height) > 8192 * 8192
    {
        return Err(CodecError::InvalidApi { what: "frame size" });
    }
    if !(1..=31).contains(&quant) {
        return Err(CodecError::InvalidApi { what: "quantiser" });
    }
    let (w, h) = (width as usize, height as usize);
    let chroma = (w / 2) * (h / 2);
    for &(y, u, v) in frames {
        if y.len() < w * h || u.len() < chroma || v.len() < chroma {
            return Err(CodecError::InvalidApi {
                what: "plane length",
            });
        }
    }
    let mbw = w / 16;
    let mbh = h / 16;
    let q = Q {
        scale: i32::from(quant),
        mpeg: false,
    };
    let mut recon = Frame::new(mbw, mbh);
    let mut mv = vec![(0i32, 0i32); mbw * 2 * mbh * 2];
    let mut clock = TimeBase::default();
    let mut out = BitWriter::new();
    // A video object before its layer. A raw demuxer uses that start code
    // to recognise the stream; the pictures do not depend on it.
    start(&mut out, 0x00);
    start(&mut out, 0x20);
    write_vol(&mut out, width, height, simple_layer());
    for (index, &(y, u, v)) in frames.iter().enumerate() {
        let planes = Planes { y, u, v, stride: w };
        start(&mut out, 0xB6);
        let mut when = When {
            ticks: index as u32,
            resolution: TIME_RESOLUTION,
            base: &mut clock,
        };
        if index == 0 {
            write_i_vop(&mut out, &planes, q, &mut recon, &mut mv, &mut when);
        } else {
            write_p_vop(&mut out, &planes, q, &mut when, &mut recon, &mut mv, false);
        }
    }
    Ok(out.finish())
}

/// Display-order pictures. With three or more, every other picture is a
/// B-VOP coded after the reference it precedes, and the layer is Advanced
/// Simple with `low_delay` clear so those pictures come back in display
/// order. `quarter` selects quarter-sample motion, which also makes the
/// layer Advanced Simple. `mpeg` selects the MPEG quantiser and the
/// default matrices. `quant` is `vop_quant`, 1 to 31.
pub fn encode_advanced(
    frames: &[(&[u8], &[u8], &[u8])],
    width: u32,
    height: u32,
    quant: u8,
    quarter: bool,
    mpeg: bool,
) -> Result<Vec<u8>, CodecError> {
    if frames.is_empty()
        || width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || u64::from(width) * u64::from(height) > 8192 * 8192
    {
        return Err(CodecError::InvalidApi { what: "frame size" });
    }
    if !(1..=31).contains(&quant) {
        return Err(CodecError::InvalidApi { what: "quantiser" });
    }
    let (w, h) = (width as usize, height as usize);
    let chroma = (w / 2) * (h / 2);
    for &(y, u, v) in frames {
        if y.len() < w * h || u.len() < chroma || v.len() < chroma {
            return Err(CodecError::InvalidApi {
                what: "plane length",
            });
        }
    }
    let refs = reference_indices(frames.len());
    let has_b = refs.windows(2).any(|pair| pair[1] > pair[0] + 1);
    let advanced = has_b || quarter;
    let mbw = w / 16;
    let mbh = h / 16;
    let q = Q {
        scale: i32::from(quant),
        mpeg,
    };
    let mut reference = Frame::new(mbw, mbh);
    let mut mv = vec![(0i32, 0i32); mbw * 2 * mbh * 2];
    let mut earlier: Option<Frame> = None;
    let mut clock = TimeBase::default();
    let mut out = BitWriter::new();
    start(&mut out, 0x00);
    start(&mut out, 0x20);
    write_vol(
        &mut out,
        width,
        height,
        Layer {
            advanced,
            quarter,
            mpeg,
            low_delay: !has_b,
            resolution: TIME_RESOLUTION,
        },
    );
    for (n, &display) in refs.iter().enumerate() {
        let (y, u, v) = frames[display];
        let planes = Planes { y, u, v, stride: w };
        start(&mut out, 0xB6);
        if n == 0 {
            let mut when = When {
                ticks: 0,
                resolution: TIME_RESOLUTION,
                base: &mut clock,
            };
            write_i_vop(&mut out, &planes, q, &mut reference, &mut mv, &mut when);
        } else {
            {
                let mut when = When {
                    ticks: display as u32,
                    resolution: TIME_RESOLUTION,
                    base: &mut clock,
                };
                write_p_vop(
                    &mut out,
                    &planes,
                    q,
                    &mut when,
                    &mut reference,
                    &mut mv,
                    quarter,
                );
            }
            if let Some(past) = earlier.as_ref() {
                let prev = refs[n - 1];
                for (step, &(y, u, v)) in frames[prev + 1..display].iter().enumerate() {
                    let between = Planes { y, u, v, stride: w };
                    start(&mut out, 0xB6);
                    let mut when = When {
                        ticks: (prev + 1 + step) as u32,
                        resolution: TIME_RESOLUTION,
                        base: &mut clock,
                    };
                    write_b_vop(
                        &mut out,
                        &between,
                        q,
                        &mut when,
                        &Refs {
                            past,
                            future: &reference,
                            pp: (display - prev) as i32,
                            pb: (step + 1) as i32,
                            quarter,
                        },
                    );
                }
            }
        }
        earlier = Some(reference.clone());
    }
    Ok(out.finish())
}

/// Reference pictures in display order: the first, then every second, and
/// always the last.
fn reference_indices(count: usize) -> Vec<usize> {
    let mut refs = vec![0];
    let mut next = 2;
    while next < count {
        refs.push(next);
        next += 2;
    }
    if count > 1 && refs[refs.len() - 1] != count - 1 {
        refs.push(count - 1);
    }
    refs
}

fn start(out: &mut BitWriter, code: u8) {
    out.bytes(&[0, 0, 1, code]);
}

/// Simple object, or Advanced Simple when `advanced` so a B-VOP or
/// quarter-sample motion is legal. 4:2:0, square pixels. `mpeg` selects
/// the default MPEG matrices; otherwise the quantiser is H.263. Version 5
/// is where `quarter_sample` sits. `low_delay` stays set when the stream
/// has no B-VOP.
fn write_vol(out: &mut BitWriter, width: u32, height: u32, layer: Layer) {
    out.put(1, 0); // random_accessible_vol
    out.put(8, if layer.advanced { 17 } else { 1 });
    if layer.advanced {
        out.put(1, 1); // is_object_layer_identifier
        out.put(4, 5); // verid
        out.put(3, 1); // priority
    } else {
        out.put(1, 0);
    }
    out.put(4, 1); // square
    out.put(1, 1); // vol_control_parameters
    out.put(2, 1); // 4:2:0
    out.put(1, u32::from(layer.low_delay));
    out.put(1, 0); // no vbv
    out.put(2, 0); // rectangular
    out.put(1, 1);
    out.put(16, layer.resolution);
    out.put(1, 1);
    out.put(1, 0); // fixed_vop_rate
    out.put(1, 1);
    out.put(13, width);
    out.put(1, 1);
    out.put(13, height);
    out.put(1, 1);
    out.put(1, 0); // progressive
    out.put(1, 1); // obmc_disable
    if layer.advanced {
        out.put(2, 0); // no sprite
    } else {
        out.put(1, 0); // no sprite
    }
    out.put(1, 0); // not_8_bit
    out.put(1, u32::from(layer.mpeg));
    if layer.mpeg {
        out.put(1, 0); // default intra matrix
        out.put(1, 0); // default non-intra matrix
    }
    if layer.advanced {
        out.put(1, u32::from(layer.quarter));
    }
    out.put(1, 1); // complexity_estimation_disable
    out.put(1, 1); // resync_marker_disable
    out.put(1, 0); // data_partitioned
    if layer.advanced {
        out.put(1, 0); // newpred
        out.put(1, 0); // reduced resolution
    }
    out.put(1, 0); // scalability
    out.stuff();
}

fn write_i_vop(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    q: Q,
    recon: &mut Frame,
    mv: &mut [(i32, i32)],
    when: &mut When<'_>,
) {
    out.put(2, 0); // I
    write_time(out, when, false);
    out.put(1, 1); // coded
    out.put(3, 0); // intra_dc_vlc_thr: DC stays in the DC VLC
    out.put(5, q.scale as u32);
    let mbw = recon.mbw;
    let mbh = recon.mbh;
    recon.mbs.resize(mbw * mbh, MbRecord::default());
    mv.fill((0, 0));
    let mut dc = [
        vec![1024i32; mbw * 2 * mbh * 2],
        vec![1024i32; mbw * mbh],
        vec![1024i32; mbw * mbh],
    ];
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let (levels, dcs) = write_intra_mb(out, planes, mbx, mby, q, &mut dc);
            for k in 0..6 {
                paint_intra(recon, mbx, mby, k, &levels[k], dcs[k], q);
            }
            set_record(recon, mby * mbw + mbx, Kind::Intra, [(0, 0); 4]);
        }
    }
    out.stuff();
}

/// `modulo_time_base`, the marker, `vop_time_increment`, and the marker.
///
/// The modulo is how many seconds passed since the previous non-B base, or
/// since the base before that for a B-VOP. An absolute second count would
/// be added again on the next picture.
fn write_time(out: &mut BitWriter, when: &mut When<'_>, bidirectional: bool) {
    let resolution = when.resolution.max(1);
    let seconds = when.ticks / resolution;
    let increment = when.ticks % resolution;
    let origin = if bidirectional {
        when.base.last
    } else {
        when.base.seconds
    };
    let modulo = seconds.saturating_sub(origin);
    let time_bits = (32 - (resolution - 1).leading_zeros()).max(1);
    for _ in 0..modulo {
        out.bit(true);
    }
    out.bit(false);
    out.bit(true);
    out.put(time_bits, increment);
    out.bit(true);
    if !bidirectional {
        when.base.last = when.base.seconds;
        when.base.seconds = seconds;
    }
}

struct Planes<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    stride: usize,
}

fn write_intra_mb(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    mbx: usize,
    mby: usize,
    q: Q,
    dc: &mut [Vec<i32>; 3],
) -> ([[i32; 64]; 6], [i32; 6]) {
    let mut coded = [false; 6];
    let mut levels = [[0i32; 64]; 6];
    let mut reconstructed = [0i32; 6];
    let stride = planes.stride;
    for k in 0..6 {
        let (plane, x, y, step) = if k < 4 {
            (
                planes.y,
                mbx * 16 + (k & 1) * 8,
                mby * 16 + (k >> 1) * 8,
                stride,
            )
        } else if k == 4 {
            (planes.u, mbx * 8, mby * 8, stride / 2)
        } else {
            (planes.v, mbx * 8, mby * 8, stride / 2)
        };
        let coef = forward(&sample(plane, step, x, y));
        let across = if k < 4 { stride / 8 } else { stride / 16 };
        let (bx, by) = if k < 4 {
            ((2 * mbx + (k & 1)) as isize, (2 * mby + (k >> 1)) as isize)
        } else {
            (mbx as isize, mby as isize)
        };
        let scale = if k < 4 {
            dc_scaler_luma(q.scale as u32)
        } else {
            dc_scaler_chroma(q.scale as u32)
        };
        let predicted = predict(&dc[plane_of(k)], across, bx, by);
        let qf = div_round(coef[0], scale).clamp(0, 2047 / scale);
        let diff = qf - (predicted + (scale >> 1)) / scale;
        let stored = qf * scale;
        dc[plane_of(k)][by as usize * across + bx as usize] = stored;
        reconstructed[k] = stored;
        levels[k][0] = diff;
        for i in 1..64 {
            levels[k][i] = quantise(coef[i], q, i, true);
            coded[k] |= levels[k][i] != 0;
        }
    }
    let cbpc = u32::from(coded[4]) << 1 | u32::from(coded[5]);
    let (len, code) = MCBPC_INTRA[cbpc as usize];
    out.put(u32::from(len), u32::from(code));
    out.bit(false); // ac_pred
    let cbpy = u32::from(coded[0]) << 3
        | u32::from(coded[1]) << 2
        | u32::from(coded[2]) << 1
        | u32::from(coded[3]);
    let (len, code) = CBPY[cbpy as usize];
    out.put(u32::from(len), u32::from(code));
    for k in 0..6 {
        write_dc(out, levels[k][0], k >= 4);
        if coded[k] {
            write_levels(out, &levels[k], 1, false);
        }
    }
    (levels, reconstructed)
}

/// Replaces one intra block with the same inverse quantiser and transform
/// the decoder uses, so a later P-VOP predicts from that picture.
fn paint_intra(
    frame: &mut Frame,
    mbx: usize,
    mby: usize,
    k: usize,
    levels: &[i32; 64],
    dc: i32,
    q: Q,
) {
    let mut ac = [0i32; 64];
    ac[1..].copy_from_slice(&levels[1..]);
    let mut coef = dequant_intra(&ac, dc, q.scale, method(q));
    put_block(frame, mbx, mby, k, &mut coef, false);
}

fn plane_of(k: usize) -> usize {
    if k < 4 { 0 } else { k - 3 }
}

/// Section 7.4.3.1. A missing neighbour predicts 1024, mid-grey.
fn predict(dc: &[i32], across: usize, bx: isize, by: isize) -> i32 {
    let get = |dx: isize, dy: isize| -> i32 {
        let x = bx + dx;
        let y = by + dy;
        if x < 0 || y < 0 || across == 0 || x as usize >= across {
            return 1024;
        }
        dc.get(y as usize * across + x as usize)
            .copied()
            .unwrap_or(1024)
    };
    let (left, corner, above) = (get(-1, 0), get(-1, -1), get(0, -1));
    if (left - corner).abs() < (corner - above).abs() {
        above
    } else {
        left
    }
}

fn write_dc(out: &mut BitWriter, diff: i32, chroma: bool) {
    let size = if diff == 0 {
        0
    } else {
        32 - diff.unsigned_abs().leading_zeros()
    };
    let size = size.min(12);
    let (len, code) = if chroma {
        DC_CHROMA[size as usize]
    } else {
        DC_LUMA[size as usize]
    };
    out.put(u32::from(len), u32::from(code));
    if size == 0 {
        return;
    }
    // A positive differential is itself. A negative one is
    // `diff + 2^size - 1`, whose top bit is clear.
    let bits = if diff > 0 {
        diff as u32
    } else {
        (diff + (1 << size) - 1) as u32
    };
    out.put(size, bits);
    if size > 8 {
        out.bit(true);
    }
}

fn write_levels(out: &mut BitWriter, levels: &[i32; 64], start: usize, inter: bool) {
    let table = if inter {
        &tables().inter
    } else {
        &tables().intra
    };
    let mut coded = [(0usize, 0i32); 64];
    let mut count = 0usize;
    for pos in start..64 {
        let level = levels[usize::from(ZIGZAG[pos])];
        if level != 0 {
            coded[count] = (pos, level);
            count += 1;
        }
    }
    let mut cursor = start;
    for (n, &(pos, level)) in coded[..count].iter().enumerate() {
        write_coef(out, table, n + 1 == count, pos - cursor, level);
        cursor = pos + 1;
    }
}

fn write_coef(out: &mut BitWriter, table: &CoefTable, last: bool, run: usize, level: i32) {
    let mag = level.unsigned_abs();
    if let Some((len, code)) = table.code(last, run, mag) {
        out.put(u32::from(len), u32::from(code));
        out.bit(level < 0);
        return;
    }
    let (elen, ecode) = TCOEF_ESCAPE;
    out.put(u32::from(elen), u32::from(ecode));
    out.bit(true);
    out.bit(true);
    out.bit(last);
    out.put(6, run as u32);
    out.bit(true);
    out.put_signed(12, level);
    out.bit(true);
}

/// The inverse quantiser `q` selects.
fn method(q: Q) -> Quant<'static> {
    if q.mpeg {
        Quant::Mpeg {
            intra: &DEFAULT_INTRA_MATRIX,
            inter: &DEFAULT_INTER_MATRIX,
        }
    } else {
        Quant::H263
    }
}

/// Level whose reconstruction is nearest `coef`. `index` is the raster
/// position, and it selects the MPEG matrix weight.
fn quantise(coef: i32, q: Q, index: usize, intra: bool) -> i32 {
    if !q.mpeg {
        return quant_ac(coef, q.scale);
    }
    let weight = i32::from(if intra {
        DEFAULT_INTRA_MATRIX[index]
    } else {
        DEFAULT_INTER_MATRIX[index]
    });
    if coef == 0 || weight == 0 || q.scale <= 0 {
        return 0;
    }
    let mag = coef.unsigned_abs() as i32;
    let step = if intra {
        2 * q.scale * weight
    } else {
        q.scale * weight
    };
    let recon = |level: i32| -> i32 {
        if level == 0 {
            0
        } else if intra {
            (level * step) >> 4
        } else {
            ((2 * level + 1) * step) >> 4
        }
    };
    let guess = if intra {
        (mag << 4) / step
    } else {
        ((mag << 4) / step).saturating_sub(1) / 2
    };
    let mut best = 0i32;
    let mut best_err = mag;
    let low = guess.saturating_sub(2);
    let high = (guess + 2).min(2047);
    for level in low..=high {
        let err = (recon(level) - mag).abs();
        if err < best_err {
            best = level;
            best_err = err;
        }
    }
    coef.signum() * best
}

/// H.263 intra AC level whose reconstruction is nearest `coef`.
fn quant_ac(coef: i32, quant: i32) -> i32 {
    if coef == 0 {
        return 0;
    }
    let mul = 2 * quant;
    let add = (quant - 1) | 1;
    let sign = coef.signum();
    let mag = coef.unsigned_abs() as i32;
    if mag * 2 <= mul + add {
        return 0;
    }
    let level = ((mag - add + mul / 2) / mul).clamp(1, 2047);
    sign * level
}

fn div_round(value: i32, divisor: i32) -> i32 {
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

fn sample(plane: &[u8], stride: usize, x: usize, y: usize) -> [i32; 64] {
    let mut out = [0; 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] = i32::from(plane[(y + row) * stride + x + col]);
        }
    }
    out
}

/// The forward transform paired with [`crate::idct`]: coefficients are
/// the analysis of the same cosine basis the inverse uses.
fn forward(pixels: &[i32; 64]) -> [i32; 64] {
    let b = basis();
    let mut out = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f64;
            for y in 0..8 {
                for x in 0..8 {
                    sum += f64::from(pixels[y * 8 + x]) * b[u][x] * b[v][y];
                }
            }
            out[v * 8 + u] = nearest(sum);
        }
    }
    out
}

fn basis() -> &'static [[f64; 8]; 8] {
    static BASIS: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    BASIS.get_or_init(|| {
        let mut rows = [[0.0f64; 8]; 8];
        for (k, row) in rows.iter_mut().enumerate() {
            let scale = if k == 0 {
                0.5 / std::f64::consts::SQRT_2
            } else {
                0.5
            };
            for (n, value) in row.iter_mut().enumerate() {
                let angle = (2.0 * n as f64 + 1.0) * k as f64 * std::f64::consts::PI / 16.0;
                *value = scale * angle.cos();
            }
        }
        rows
    })
}

fn nearest(value: f64) -> i32 {
    let rounded = value.round();
    if rounded >= f64::from(i32::MAX) {
        i32::MAX
    } else if rounded <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        rounded as i32
    }
}

/// Half-sample search range, in half-sample units: ±8 pixels, which is
/// inside the `fcode` 1 vector range the decoder wraps with a shift.
const SEARCH: i32 = 16;

fn write_p_vop(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    q: Q,
    when: &mut When<'_>,
    recon: &mut Frame,
    mv: &mut [(i32, i32)],
    quarter: bool,
) {
    out.put(2, 1); // P
    write_time(out, when, false);
    out.bit(true); // coded
    out.bit(false); // rounding
    out.put(3, 0); // intra_dc_vlc_thr
    out.put(5, q.scale as u32);
    out.put(3, 1); // fcode_forward
    let past = recon.clone();
    let mut next = Frame::new(past.mbw, past.mbh);
    next.clip_width = past.clip_width;
    next.clip_height = past.clip_height;
    next.mbs.resize(past.mbw * past.mbh, MbRecord::default());
    let mbw = past.mbw;
    let mbh = past.mbh;
    {
        let mut picture = Predicted {
            y: planes.y,
            u: planes.u,
            v: planes.v,
            width: planes.stride,
            height: past.mbh * 16,
            q,
            past: &past,
            next: &mut next,
            mv,
            quarter,
        };
        for mby in 0..mbh {
            for mbx in 0..mbw {
                picture.write_mb(out, mbx, mby);
            }
        }
    }
    *recon = next;
    out.stuff();
}

struct Predicted<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: usize,
    height: usize,
    q: Q,
    past: &'a Frame,
    next: &'a mut Frame,
    mv: &'a mut [(i32, i32)],
    quarter: bool,
}

impl Predicted<'_> {
    fn write_mb(&mut self, out: &mut BitWriter, mbx: usize, mby: usize) {
        let mbw = self.width / 16;
        let pred = mv_pred(self.mv, mbw, mbx, mby, 0);
        let vector = self.search_at(mbx * 16, mby * 16, 16, pred);
        let cost16 = self.block_cost(mbx * 16, mby * 16, 16, vector, pred);
        let mut four = [(0i32, 0i32); 4];
        let mut cost4 = 0u32;
        for (k, slot) in four.iter_mut().enumerate() {
            let block_pred = mv_pred(self.mv, mbw, mbx, mby, k);
            let (x, y) = (mbx * 16 + (k & 1) * 8, mby * 16 + (k >> 1) * 8);
            *slot = self.search_at(x, y, 8, block_pred);
            cost4 = cost4.saturating_add(self.block_cost(x, y, 8, *slot, block_pred));
            set_block_mv(self.mv, mbw, mbx, mby, k, *slot);
        }
        // Four vectors cost three extra differences. Use them when the
        // prediction is actually better, not when the two modes tie.
        let use_four = cost4 + 64 < cost16;
        if !use_four {
            set_mv(self.mv, mbw, mbx, mby, vector);
        }
        let (py, pu, pv) = if use_four {
            compensate_four(self.past, mbx, mby, four, self.quarter)
        } else {
            compensate(self.past, mbx, mby, vector, self.quarter)
        };
        let mut levels = [[0i32; 64]; 6];
        let mut coded = [false; 6];
        for k in 0..6 {
            let pixels = self.residual(mbx, mby, k, &py, &pu, &pv);
            let coef = forward(&pixels);
            for i in 0..64 {
                levels[k][i] = quantise(coef[i], self.q, i, false);
                coded[k] |= levels[k][i] != 0;
            }
        }
        let any = coded.iter().any(|&bit| bit);
        let still = if use_four {
            four == [(0, 0); 4]
        } else {
            vector == (0, 0)
        };
        if still && !any {
            out.bit(true);
            set_mv(self.mv, mbw, mbx, mby, (0, 0));
            blit(self.next, mbx, mby, &py, &pu, &pv);
            set_record(self.next, mby * mbw + mbx, Kind::Skipped, [(0, 0); 4]);
            return;
        }
        out.bit(false);
        let cbpc = u32::from(coded[4]) << 1 | u32::from(coded[5]);
        let (len, code) = mcbpc_inter(if use_four { 16 + cbpc } else { cbpc });
        out.put(len, code);
        let luma = u32::from(coded[0]) << 3
            | u32::from(coded[1]) << 2
            | u32::from(coded[2]) << 1
            | u32::from(coded[3]);
        let (len, code) = CBPY[(luma ^ 0xF) as usize];
        out.put(u32::from(len), u32::from(code));
        if use_four {
            for (k, vector) in four.iter().enumerate() {
                let block_pred = mv_pred(self.mv, mbw, mbx, mby, k);
                write_mv(out, vector.0, block_pred.0);
                write_mv(out, vector.1, block_pred.1);
            }
        } else {
            write_mv(out, vector.0, pred.0);
            write_mv(out, vector.1, pred.1);
        }
        for k in 0..6 {
            if coded[k] {
                write_levels(out, &levels[k], 0, true);
            }
        }
        if !use_four {
            set_mv(self.mv, mbw, mbx, mby, vector);
        }
        blit(self.next, mbx, mby, &py, &pu, &pv);
        for k in 0..6 {
            if coded[k] {
                add_inter(self.next, mbx, mby, k, &levels[k], self.q);
            }
        }
        if use_four {
            set_record(self.next, mby * mbw + mbx, Kind::Four, four);
        } else {
            set_record(self.next, mby * mbw + mbx, Kind::Frame, [vector; 4]);
        }
    }

    fn residual(
        &self,
        mbx: usize,
        mby: usize,
        k: usize,
        y: &[u8; 256],
        u: &[u8; 64],
        v: &[u8; 64],
    ) -> [i32; 64] {
        if k < 4 {
            let (px, py) = ((k & 1) * 8, (k >> 1) * 8);
            diff_block(
                self.y,
                self.width,
                (mbx * 16 + px, mby * 16 + py),
                y,
                16,
                (px, py),
            )
        } else {
            let plane = if k == 4 { self.u } else { self.v };
            let pred = if k == 4 { u } else { v };
            diff_block(plane, self.width / 2, (mbx * 8, mby * 8), pred, 8, (0, 0))
        }
    }

    fn search_at(&self, x: usize, y: usize, size: usize, pred: (i32, i32)) -> (i32, i32) {
        let mut best = clamp_mv(pred.0, pred.1);
        let mut best_cost = self.block_cost(x, y, size, best, pred);
        let step = motion_step(self.quarter);
        for dy in (-SEARCH..=SEARCH).step_by(step) {
            for dx in (-SEARCH..=SEARCH).step_by(step) {
                let mv = clamp_mv(pred.0 + dx, pred.1 + dy);
                let cost = self.block_cost(x, y, size, mv, pred);
                if cost < best_cost {
                    best = mv;
                    best_cost = cost;
                }
            }
        }
        let center = best;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let mv = clamp_mv(center.0 + dx, center.1 + dy);
                let cost = self.block_cost(x, y, size, mv, pred);
                if cost < best_cost {
                    best = mv;
                    best_cost = cost;
                }
            }
        }
        let zero = self.block_cost(x, y, size, (0, 0), pred);
        if zero < best_cost { (0, 0) } else { best }
    }

    fn block_cost(&self, x: usize, y: usize, size: usize, mv: (i32, i32), pred: (i32, i32)) -> u32 {
        let mut predicted = [0u8; 256];
        let [reference, _, _] = self.past.refs((self.width as u32, self.height as u32));
        mc::luma_block(
            &reference,
            x as i32,
            y as i32,
            size,
            size,
            mv.0,
            mv.1,
            motion_pel(self.quarter),
            Rows::Frame,
            false,
            &mut predicted,
        );
        let mut sad = 0u32;
        for row in 0..size {
            for col in 0..size {
                let sample = self
                    .y
                    .get((y + row) * self.width + x + col)
                    .copied()
                    .unwrap_or(0);
                sad += u32::from(sample.abs_diff(predicted[row * size + col]));
            }
        }
        sad.saturating_mul(2) + (mv.0 - pred.0).unsigned_abs() + (mv.1 - pred.1).unsigned_abs()
    }
}

fn clamp_mv(x: i32, y: i32) -> (i32, i32) {
    (x.clamp(-SEARCH, SEARCH), y.clamp(-SEARCH, SEARCH))
}

fn motion_pel(quarter: bool) -> Pel {
    if quarter { Pel::Quarter } else { Pel::Half }
}

/// One pixel, in the vector units `quarter` selects.
fn motion_step(quarter: bool) -> usize {
    if quarter { 4 } else { 2 }
}

fn chroma_vector(m: i32, quarter: bool) -> i32 {
    mc::chroma_hpel(if quarter { m / 2 } else { m })
}

fn compensate(
    past: &Frame,
    mbx: usize,
    mby: usize,
    mv: (i32, i32),
    quarter: bool,
) -> ([u8; 256], [u8; 64], [u8; 64]) {
    let mut y = [0u8; 256];
    let mut u = [0u8; 64];
    let mut v = [0u8; 64];
    let [ry, ru, rv] = past.refs(((past.mbw * 16) as u32, (past.mbh * 16) as u32));
    mc::luma_block(
        &ry,
        (mbx * 16) as i32,
        (mby * 16) as i32,
        16,
        16,
        mv.0,
        mv.1,
        motion_pel(quarter),
        Rows::Frame,
        false,
        &mut y,
    );
    let (cx, cy) = (chroma_vector(mv.0, quarter), chroma_vector(mv.1, quarter));
    mc::chroma_block(
        &ru,
        (mbx * 8) as i32,
        (mby * 8) as i32,
        8,
        8,
        cx,
        cy,
        Rows::Frame,
        false,
        &mut u,
    );
    mc::chroma_block(
        &rv,
        (mbx * 8) as i32,
        (mby * 8) as i32,
        8,
        8,
        cx,
        cy,
        Rows::Frame,
        false,
        &mut v,
    );
    (y, u, v)
}

/// Four 8x8 predictions. Each luma vector is clipped the way the decoder
/// clips it, and chroma uses the rounded sum of the four.
fn compensate_four(
    past: &Frame,
    mbx: usize,
    mby: usize,
    mvs: [(i32, i32); 4],
    quarter: bool,
) -> ([u8; 256], [u8; 64], [u8; 64]) {
    let mut y = [0u8; 256];
    let mut u = [0u8; 64];
    let mut v = [0u8; 64];
    let width = past.clip_width;
    let height = past.clip_height;
    let [ry, ru, rv] = past.refs(((past.mbw * 16) as u32, (past.mbh * 16) as u32));
    let (x, yy) = ((mbx * 16) as i32, (mby * 16) as i32);
    let (mut sx, mut sy) = (0i32, 0i32);
    let shift = if quarter { 2 } else { 1 };
    for (k, &(mx, my)) in mvs.iter().enumerate() {
        let (bx, by) = ((k & 1) * 8, (k >> 1) * 8);
        let (bx0, by0) = (x + bx as i32, yy + by as i32);
        let mut block = [0u8; 64];
        mc::luma_block(
            &ry,
            bx0,
            by0,
            8,
            8,
            clip_block(bx0, mx, shift, -16, width),
            clip_block(by0, my, shift, -16, height),
            motion_pel(quarter),
            Rows::Frame,
            false,
            &mut block,
        );
        for row in 0..8 {
            let dst = (by + row) * 16 + bx;
            y[dst..dst + 8].copy_from_slice(&block[row * 8..row * 8 + 8]);
        }
        if quarter {
            sx += mx / 2;
            sy += my / 2;
        } else {
            sx += mx;
            sy += my;
        }
    }
    let (cx0, cy0) = ((mbx * 8) as i32, (mby * 8) as i32);
    let cx = clip_block(cx0, mc::chroma_4mv(sx), 1, -8, width >> 1);
    let cy = clip_block(cy0, mc::chroma_4mv(sy), 1, -8, height >> 1);
    mc::chroma_block(&ru, cx0, cy0, 8, 8, cx, cy, Rows::Frame, false, &mut u);
    mc::chroma_block(&rv, cx0, cy0, 8, 8, cx, cy, Rows::Frame, false, &mut v);
    (y, u, v)
}

fn diff_block(
    src: &[u8],
    src_stride: usize,
    at: (usize, usize),
    pred: &[u8],
    pred_stride: usize,
    from: (usize, usize),
) -> [i32; 64] {
    let (sx, sy) = at;
    let (px, py) = from;
    let mut out = [0i32; 64];
    for row in 0..8 {
        for col in 0..8 {
            let sample = src
                .get((sy + row) * src_stride + sx + col)
                .copied()
                .unwrap_or(0);
            let predicted = pred
                .get((py + row) * pred_stride + px + col)
                .copied()
                .unwrap_or(0);
            out[row * 8 + col] = i32::from(sample) - i32::from(predicted);
        }
    }
    out
}

fn blit(frame: &mut Frame, mbx: usize, mby: usize, y: &[u8; 256], u: &[u8; 64], v: &[u8; 64]) {
    let ls = frame.luma_stride();
    let base = mby * 16 * ls + mbx * 16;
    for row in 0..16 {
        let start = base + row * ls;
        if let Some(dst) = frame.y.get_mut(start..start + 16) {
            dst.copy_from_slice(&y[row * 16..row * 16 + 16]);
        }
    }
    let cs = frame.chroma_stride();
    let base = mby * 8 * cs + mbx * 8;
    for row in 0..8 {
        let start = base + row * cs;
        if let Some(dst) = frame.u.get_mut(start..start + 8) {
            dst.copy_from_slice(&u[row * 8..row * 8 + 8]);
        }
        if let Some(dst) = frame.v.get_mut(start..start + 8) {
            dst.copy_from_slice(&v[row * 8..row * 8 + 8]);
        }
    }
}

fn add_inter(frame: &mut Frame, mbx: usize, mby: usize, k: usize, levels: &[i32; 64], q: Q) {
    let mut coef = if q.mpeg {
        dequant_inter(levels, q.scale, method(q))
    } else {
        let mut values = [0i32; 64];
        let mul = 2 * q.scale;
        let add = (q.scale - 1) | 1;
        for (slot, &level) in values.iter_mut().zip(levels) {
            *slot = if level > 0 {
                level * mul + add
            } else if level < 0 {
                level * mul - add
            } else {
                0
            };
        }
        dequant_inter(&values, q.scale, Quant::H263)
    };
    put_block(frame, mbx, mby, k, &mut coef, true);
}

fn put_block(frame: &mut Frame, mbx: usize, mby: usize, k: usize, coef: &mut [i16; 64], add: bool) {
    if k < 4 {
        let stride = frame.luma_stride();
        let offset = (mby * 16 + (k >> 1) * 8) * stride + mbx * 16 + (k & 1) * 8;
        if add {
            idct::add(coef, &mut frame.y, offset, stride);
        } else {
            idct::put(coef, &mut frame.y, offset, stride);
        }
    } else {
        let stride = frame.chroma_stride();
        let offset = mby * 8 * stride + mbx * 8;
        let plane = if k == 4 { &mut frame.u } else { &mut frame.v };
        if add {
            idct::add(coef, plane, offset, stride);
        } else {
            idct::put(coef, plane, offset, stride);
        }
    }
}

/// The median predictor of 8x8 block `k` (Section 7.6.5). A 16x16 vector
/// uses block 0.
fn mv_pred(mv: &[(i32, i32)], mbw: usize, mbx: usize, mby: usize, k: usize) -> (i32, i32) {
    let w = 2 * mbw;
    let bx = (2 * mbx + (k & 1)) as isize;
    let by = (2 * mby + (k >> 1)) as isize;
    let third = if k == 3 {
        -1
    } else if k == 0 {
        2
    } else {
        1
    };
    let mut valid = [None; 3];
    for (slot, (dx, dy)) in valid
        .iter_mut()
        .zip([(-1isize, 0isize), (0, -1), (third, -1)])
    {
        let x = bx + dx;
        let y = by + dy;
        if x < 0 || y < 0 || x as usize >= w {
            continue;
        }
        *slot = mv.get(y as usize * w + x as usize).copied();
    }
    match valid {
        [Some(a), Some(b), Some(c)] => (median(a.0, b.0, c.0), median(a.1, b.1, c.1)),
        [Some(a), Some(b), None] | [Some(a), None, Some(b)] | [None, Some(a), Some(b)] => {
            (median(a.0, b.0, 0), median(a.1, b.1, 0))
        }
        [Some(a), None, None] | [None, Some(a), None] | [None, None, Some(a)] => a,
        [None, None, None] => (0, 0),
    }
}

fn median(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

fn set_mv(mv: &mut [(i32, i32)], mbw: usize, mbx: usize, mby: usize, vector: (i32, i32)) {
    for k in 0..4 {
        set_block_mv(mv, mbw, mbx, mby, k, vector);
    }
}

fn set_block_mv(
    mv: &mut [(i32, i32)],
    mbw: usize,
    mbx: usize,
    mby: usize,
    k: usize,
    vector: (i32, i32),
) {
    let w = 2 * mbw;
    let i = (2 * mby + (k >> 1)) * w + 2 * mbx + (k & 1);
    if let Some(slot) = mv.get_mut(i) {
        *slot = vector;
    }
}

fn mcbpc_inter(cbpc: u32) -> (u32, u32) {
    for &(symbol, len, code) in &MCBPC_INTER {
        if u32::from(symbol) == cbpc {
            return (u32::from(len), u32::from(code));
        }
    }
    (1, 0b1)
}

fn write_mv(out: &mut BitWriter, value: i32, pred: i32) {
    let diff = value - pred;
    if diff == 0 {
        let (len, code) = MVD[0];
        out.put(u32::from(len), u32::from(code));
        return;
    }
    let mag = (diff.unsigned_abs() as usize).min(MVD.len() - 1);
    let (len, code) = MVD[mag];
    out.put(u32::from(len), u32::from(code));
    out.bit(diff < 0);
}

fn set_record(frame: &mut Frame, mb: usize, kind: Kind, mv: [(i32, i32); 4]) {
    if let Some(slot) = frame.mbs.get_mut(mb) {
        *slot = MbRecord {
            kind,
            mv,
            ..MbRecord::default()
        };
    }
}

/// The two references a B-VOP sits between, and the clock distances the
/// decoder will derive from their times. `pb` is this picture's distance
/// from the past reference; `pp` is the distance between the references.
struct Refs<'a> {
    past: &'a Frame,
    future: &'a Frame,
    pp: i32,
    pb: i32,
    quarter: bool,
}

fn write_b_vop(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    q: Q,
    when: &mut When<'_>,
    refs: &Refs<'_>,
) {
    out.put(2, 2); // B
    write_time(out, when, true);
    out.bit(true); // coded
    out.put(3, 0); // intra_dc_vlc_thr
    out.put(5, q.scale as u32);
    out.put(3, 1); // fcode_forward
    out.put(3, 1); // fcode_backward
    let mbw = refs.past.mbw;
    let mbh = refs.past.mbh;
    for mby in 0..mbh {
        let mut last = [(0i32, 0i32); 2];
        for mbx in 0..mbw {
            write_b_mb(out, planes, q, refs, (mbx, mby), &mut last);
        }
    }
    out.stuff();
}

fn write_b_mb(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    q: Q,
    refs: &Refs<'_>,
    at: (usize, usize),
    last: &mut [(i32, i32); 2],
) {
    let (mbx, mby) = at;
    let mb = mby * refs.past.mbw + mbx;
    if refs
        .future
        .mbs
        .get(mb)
        .is_some_and(|record| record.kind == Kind::Skipped)
    {
        return;
    }
    let direct = direct_prediction(refs, mbx, mby);
    let forward = search_frame(
        planes.y,
        planes.stride,
        refs.past,
        (mbx * 16, mby * 16, 16),
        last[0],
        refs.quarter,
    );
    let backward = search_frame(
        planes.y,
        planes.stride,
        refs.future,
        (mbx * 16, mby * 16, 16),
        last[1],
        refs.quarter,
    );
    let fwd = compensate(refs.past, mbx, mby, forward, refs.quarter);
    let bwd = compensate(refs.future, mbx, mby, backward, refs.quarter);
    let both = blend(fwd, bwd);
    let mut best_sad = plane_sad(planes.y, planes.stride, (mbx * 16, mby * 16), &direct.0, 16);
    let mut mode = BMotion::Direct;
    let mut prediction = direct;
    let fwd_sad = plane_sad(planes.y, planes.stride, (mbx * 16, mby * 16), &fwd.0, 16);
    if fwd_sad < best_sad {
        best_sad = fwd_sad;
        mode = BMotion::Forward(forward);
        prediction = fwd;
    }
    let bwd_sad = plane_sad(planes.y, planes.stride, (mbx * 16, mby * 16), &bwd.0, 16);
    if bwd_sad < best_sad {
        best_sad = bwd_sad;
        mode = BMotion::Backward(backward);
        prediction = bwd;
    }
    let both_sad = plane_sad(planes.y, planes.stride, (mbx * 16, mby * 16), &both.0, 16);
    if both_sad < best_sad {
        mode = BMotion::Both(forward, backward);
        prediction = both;
    }
    let (levels, coded) = inter_residual(planes, mbx, mby, q, &prediction);
    let cbp = coded_pattern(&coded);
    if matches!(mode, BMotion::Direct) && cbp == 0 {
        out.bit(true);
        return;
    }
    out.bit(false);
    out.bit(cbp == 0);
    let type_index = match mode {
        BMotion::Direct => 0,
        BMotion::Both(_, _) => 1,
        BMotion::Backward(_) => 2,
        BMotion::Forward(_) => 3,
    };
    let (len, code) = MB_TYPE_B[type_index];
    out.put(u32::from(len), u32::from(code));
    if cbp != 0 {
        out.put(6, cbp);
        if type_index != 0 {
            out.bit(false);
        }
    }
    match mode {
        BMotion::Direct => {
            write_mv(out, 0, 0);
            write_mv(out, 0, 0);
        }
        BMotion::Forward(vector) => {
            write_mv(out, vector.0, last[0].0);
            write_mv(out, vector.1, last[0].1);
            last[0] = vector;
        }
        BMotion::Backward(vector) => {
            write_mv(out, vector.0, last[1].0);
            write_mv(out, vector.1, last[1].1);
            last[1] = vector;
        }
        BMotion::Both(forward, backward) => {
            write_mv(out, forward.0, last[0].0);
            write_mv(out, forward.1, last[0].1);
            write_mv(out, backward.0, last[1].0);
            write_mv(out, backward.1, last[1].1);
            last[0] = forward;
            last[1] = backward;
        }
    }
    for (k, level) in levels.iter().enumerate() {
        if coded[k] {
            write_levels(out, level, 0, true);
        }
    }
}

enum BMotion {
    Direct,
    Forward((i32, i32)),
    Backward((i32, i32)),
    Both((i32, i32), (i32, i32)),
}

/// Direct mode with a zero delta. Each colocated vector is scaled by
/// `pb / pp` forward and `(pb - pp) / pp` backward, the same truncating
/// division the decoder uses.
fn direct_prediction(refs: &Refs<'_>, mbx: usize, mby: usize) -> ([u8; 256], [u8; 64], [u8; 64]) {
    let record = refs
        .future
        .mbs
        .get(mby * refs.future.mbw + mbx)
        .copied()
        .unwrap_or_default();
    let scale = |vector: (i32, i32)| {
        let pp = refs.pp.max(1);
        let forward = (vector.0 * refs.pb / pp, vector.1 * refs.pb / pp);
        let backward = (
            vector.0 * (refs.pb - pp) / pp,
            vector.1 * (refs.pb - pp) / pp,
        );
        (forward, backward)
    };
    let (forward, backward) = scale(record.mv[0]);
    // Quarter-sample direct mode filters each 8x8 block, even when the
    // colocated macroblock had one vector.
    if record.kind == Kind::Four || refs.quarter {
        let mut forward_blocks = [forward; 4];
        let mut backward_blocks = [backward; 4];
        if record.kind == Kind::Four {
            for (k, vector) in record.mv.iter().enumerate() {
                (forward_blocks[k], backward_blocks[k]) = scale(*vector);
            }
        }
        blend(
            compensate_four(refs.past, mbx, mby, forward_blocks, refs.quarter),
            compensate_four(refs.future, mbx, mby, backward_blocks, refs.quarter),
        )
    } else {
        blend(
            compensate(refs.past, mbx, mby, forward, refs.quarter),
            compensate(refs.future, mbx, mby, backward, refs.quarter),
        )
    }
}

fn blend(
    forward: ([u8; 256], [u8; 64], [u8; 64]),
    backward: ([u8; 256], [u8; 64], [u8; 64]),
) -> ([u8; 256], [u8; 64], [u8; 64]) {
    let (mut y, mut u, mut v) = forward;
    mc::average(&mut y, &backward.0);
    mc::average(&mut u, &backward.1);
    mc::average(&mut v, &backward.2);
    (y, u, v)
}

fn plane_sad(
    src: &[u8],
    stride: usize,
    at: (usize, usize),
    pred: &[u8],
    pred_stride: usize,
) -> u32 {
    let (x, y) = at;
    let mut sad = 0u32;
    for row in 0..16 {
        for col in 0..16 {
            let sample = src.get((y + row) * stride + x + col).copied().unwrap_or(0);
            let predicted = pred.get(row * pred_stride + col).copied().unwrap_or(0);
            sad += u32::from(sample.abs_diff(predicted));
        }
    }
    sad
}

fn coded_pattern(coded: &[bool; 6]) -> u32 {
    let mut pattern = 0u32;
    for (k, &on) in coded.iter().enumerate() {
        if on {
            pattern |= 32 >> k;
        }
    }
    pattern
}

fn inter_residual(
    planes: &Planes<'_>,
    mbx: usize,
    mby: usize,
    q: Q,
    pred: &([u8; 256], [u8; 64], [u8; 64]),
) -> ([[i32; 64]; 6], [bool; 6]) {
    let (y, u, v) = pred;
    let mut levels = [[0i32; 64]; 6];
    let mut coded = [false; 6];
    for k in 0..6 {
        let pixels = if k < 4 {
            let (px, py) = ((k & 1) * 8, (k >> 1) * 8);
            diff_block(
                planes.y,
                planes.stride,
                (mbx * 16 + px, mby * 16 + py),
                y,
                16,
                (px, py),
            )
        } else {
            let plane = if k == 4 { planes.u } else { planes.v };
            let block = if k == 4 { u } else { v };
            diff_block(
                plane,
                planes.stride / 2,
                (mbx * 8, mby * 8),
                block,
                8,
                (0, 0),
            )
        };
        let coef = forward(&pixels);
        for (i, (level, value)) in levels[k].iter_mut().zip(coef).enumerate() {
            *level = quantise(value, q, i, false);
            coded[k] |= *level != 0;
        }
    }
    (levels, coded)
}

fn search_frame(
    y: &[u8],
    width: usize,
    reference: &Frame,
    at: (usize, usize, usize),
    pred: (i32, i32),
    quarter: bool,
) -> (i32, i32) {
    let (x, y0, size) = at;
    let cost = |mv: (i32, i32)| -> u32 {
        let mut predicted = [0u8; 256];
        let [plane, _, _] =
            reference.refs(((reference.mbw * 16) as u32, (reference.mbh * 16) as u32));
        mc::luma_block(
            &plane,
            x as i32,
            y0 as i32,
            size,
            size,
            mv.0,
            mv.1,
            motion_pel(quarter),
            Rows::Frame,
            false,
            &mut predicted,
        );
        let mut sad = 0u32;
        for row in 0..size {
            for col in 0..size {
                let sample = y.get((y0 + row) * width + x + col).copied().unwrap_or(0);
                sad += u32::from(sample.abs_diff(predicted[row * size + col]));
            }
        }
        sad.saturating_mul(2) + (mv.0 - pred.0).unsigned_abs() + (mv.1 - pred.1).unsigned_abs()
    };
    let mut best = clamp_mv(pred.0, pred.1);
    let mut best_cost = cost(best);
    let step = motion_step(quarter);
    for dy in (-SEARCH..=SEARCH).step_by(step) {
        for dx in (-SEARCH..=SEARCH).step_by(step) {
            let mv = clamp_mv(pred.0 + dx, pred.1 + dy);
            let score = cost(mv);
            if score < best_cost {
                best = mv;
                best_cost = score;
            }
        }
    }
    let center = best;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let mv = clamp_mv(center.0 + dx, center.1 + dy);
            let score = cost(mv);
            if score < best_cost {
                best = mv;
                best_cost = score;
            }
        }
    }
    if cost((0, 0)) < best_cost {
        (0, 0)
    } else {
        best
    }
}

/// One accepted picture, padded out to whole macroblocks.
struct Held {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    pts: i64,
    duration: i64,
}

/// One VOP, without the object-layer headers.
pub(crate) struct Access {
    pub(crate) bytes: Vec<u8>,
    pub(crate) pts: i64,
    pub(crate) duration: i64,
    pub(crate) keyframe: bool,
}

/// A picture the export encoder has not padded yet.
pub(crate) struct PictureIn<'a> {
    pub(crate) y: &'a [u8],
    pub(crate) u: &'a [u8],
    pub(crate) v: &'a [u8],
    pub(crate) pts: i64,
    pub(crate) duration: i64,
    pub(crate) timescale: u32,
    pub(crate) keyframe: bool,
}

/// Advanced Simple pictures for export.
///
/// The first picture is an I-VOP. Each later picture is held so the next
/// one can be a P-VOP and the held one a B-VOP between them. `low_delay`
/// is clear, so a decoder shows them in presentation order. The quantiser
/// starts from the bitrate and moves when the bytes spent leave the budget.
/// Zero bitrate keeps the starting quantiser.
pub(crate) struct Pictures {
    width: u32,
    height: u32,
    stride: usize,
    chroma_stride: usize,
    quant: i32,
    initial: i32,
    bitrate: u32,
    constant: bool,
    spent: u64,
    budget: u64,
    window: i64,
    resolution: u32,
    headers: Vec<u8>,
    reference: Frame,
    reference_pts: i64,
    coded: bool,
    mv: Vec<(i32, i32)>,
    clock: TimeBase,
    held: Option<Held>,
    drained: bool,
}

impl Pictures {
    /// `bitrate` is bits a second. Zero keeps one quantiser for every picture.
    pub(crate) fn open(
        width: u32,
        height: u32,
        bitrate: u32,
        constant: bool,
    ) -> Result<Self, CodecError> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(CodecError::InvalidApi { what: "frame size" });
        }
        if width > crate::header::MAX_EDGE
            || height > crate::header::MAX_EDGE
            || u64::from(width) * u64::from(height) > 8192 * 8192
        {
            return Err(CodecError::Limit { what: "frame size" });
        }
        let mbw = usize::try_from(width.div_ceil(16)).unwrap_or(0);
        let mbh = usize::try_from(height.div_ceil(16)).unwrap_or(0);
        if mbw == 0 || mbh == 0 {
            return Err(CodecError::InvalidApi { what: "frame size" });
        }
        let mut reference = Frame::new(mbw, mbh);
        reference.clip_width =
            i32::try_from(width).map_err(|_| CodecError::Limit { what: "frame size" })?;
        reference.clip_height =
            i32::try_from(height).map_err(|_| CodecError::Limit { what: "frame size" })?;
        let quant = initial_quant(bitrate, width, height);
        Ok(Self {
            width,
            height,
            stride: mbw * 16,
            chroma_stride: mbw * 8,
            quant,
            initial: quant,
            bitrate,
            constant,
            spent: 0,
            budget: 0,
            window: 0,
            resolution: 0,
            headers: Vec::new(),
            reference,
            reference_pts: 0,
            coded: false,
            mv: vec![(0, 0); mbw * 2 * mbh * 2],
            clock: TimeBase::default(),
            held: None,
            drained: false,
        })
    }

    pub(crate) fn headers(&self) -> &[u8] {
        &self.headers
    }

    #[cfg(test)]
    pub(crate) fn quant(&self) -> i32 {
        self.quant
    }

    pub(crate) fn push(&mut self, input: PictureIn<'_>) -> Result<Vec<Access>, CodecError> {
        self.prepare(input.timescale)?;
        self.drained = false;
        let duration = if input.duration <= 0 {
            1
        } else {
            input.duration
        };
        let picture = self.store(input.y, input.u, input.v, input.pts, duration)?;
        if input.keyframe || !self.coded {
            return self.intra(picture);
        }
        let Some(between) = self.held.take() else {
            self.held = Some(picture);
            return Ok(Vec::new());
        };
        self.pair(between, picture)
    }

    pub(crate) fn drain(&mut self) -> Result<Vec<Access>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        if self.coded
            && let Some(held) = &self.held
        {
            self.time_ok(held.pts)?;
        }
        let Some(held) = self.held.take() else {
            return Ok(Vec::new());
        };
        if !self.coded {
            let bytes = self.encode_i(&held)?;
            self.coded = true;
            self.finish_reference(&held, bytes.len());
            return Ok(vec![access(bytes, &held, true)]);
        }
        let bytes = self.encode_p(&held)?;
        self.finish_reference(&held, bytes.len());
        Ok(vec![access(bytes, &held, false)])
    }

    pub(crate) fn reset(&mut self) {
        self.quant = self.initial;
        self.spent = 0;
        self.budget = 0;
        self.window = 0;
        self.reference_pts = 0;
        self.coded = false;
        self.clock = TimeBase::default();
        self.held = None;
        self.drained = false;
        self.mv.fill((0, 0));
    }

    fn prepare(&mut self, timescale: u32) -> Result<(), CodecError> {
        if timescale == 0 || timescale > 65_535 {
            return Err(CodecError::InvalidApi {
                what: "time resolution",
            });
        }
        if self.resolution == 0 {
            self.resolution = timescale;
            self.headers = layer_bytes(self.width, self.height, timescale);
            return Ok(());
        }
        if timescale != self.resolution {
            return Err(CodecError::InvalidApi {
                what: "time resolution",
            });
        }
        Ok(())
    }

    fn intra(&mut self, picture: Held) -> Result<Vec<Access>, CodecError> {
        if let Some(held) = &self.held {
            self.time_ok(held.pts)?;
            let later = ticks_of(picture.pts)?;
            let earlier = ticks_of(held.pts)?;
            if later / self.resolution.max(1) < earlier / self.resolution.max(1) {
                return Err(CodecError::InvalidApi {
                    what: "picture time",
                });
            }
        }
        let mut packets = Vec::new();
        if let Some(held) = self.held.take() {
            let bytes = self.encode_p(&held)?;
            self.finish_reference(&held, bytes.len());
            packets.push(access(bytes, &held, false));
        }
        self.time_ok(picture.pts)?;
        let bytes = self.encode_i(&picture)?;
        self.coded = true;
        self.finish_reference(&picture, bytes.len());
        packets.push(access(bytes, &picture, true));
        Ok(packets)
    }

    fn pair(&mut self, between: Held, picture: Held) -> Result<Vec<Access>, CodecError> {
        let (pp, pb) = match distances(self.reference_pts, picture.pts, between.pts) {
            Ok(times) => times,
            Err(error) => {
                self.held = Some(between);
                return Err(error);
            }
        };
        if let Err(error) = self.time_ok(picture.pts) {
            self.held = Some(between);
            return Err(error);
        }
        let past = self.reference.clone();
        let predicted = self.encode_p(&picture)?;
        self.note(predicted.len(), picture.duration);
        let bidirectional = self.encode_b(&between, &past, pp, pb)?;
        self.note(bidirectional.len(), between.duration);
        self.reference_pts = picture.pts;
        Ok(vec![
            access(predicted, &picture, false),
            access(bidirectional, &between, false),
        ])
    }

    fn finish_reference(&mut self, picture: &Held, bytes: usize) {
        self.note(bytes, picture.duration);
        self.reference_pts = picture.pts;
    }

    fn encode_i(&mut self, picture: &Held) -> Result<Vec<u8>, CodecError> {
        let mut out = BitWriter::new();
        start(&mut out, 0xB6);
        let planes = planes_of(picture, self.stride);
        let quant = self.q();
        let ticks = ticks_of(picture.pts)?;
        let resolution = self.resolution;
        write_i_vop(
            &mut out,
            &planes,
            quant,
            &mut self.reference,
            &mut self.mv,
            &mut When {
                ticks,
                resolution,
                base: &mut self.clock,
            },
        );
        Ok(out.finish())
    }

    fn encode_p(&mut self, picture: &Held) -> Result<Vec<u8>, CodecError> {
        let mut out = BitWriter::new();
        start(&mut out, 0xB6);
        let planes = planes_of(picture, self.stride);
        let quant = self.q();
        let ticks = ticks_of(picture.pts)?;
        let resolution = self.resolution;
        write_p_vop(
            &mut out,
            &planes,
            quant,
            &mut When {
                ticks,
                resolution,
                base: &mut self.clock,
            },
            &mut self.reference,
            &mut self.mv,
            true,
        );
        Ok(out.finish())
    }

    fn encode_b(
        &mut self,
        picture: &Held,
        past: &Frame,
        pp: i32,
        pb: i32,
    ) -> Result<Vec<u8>, CodecError> {
        let mut out = BitWriter::new();
        start(&mut out, 0xB6);
        let planes = planes_of(picture, self.stride);
        let quant = self.q();
        let ticks = ticks_of(picture.pts)?;
        let resolution = self.resolution;
        write_b_vop(
            &mut out,
            &planes,
            quant,
            &mut When {
                ticks,
                resolution,
                base: &mut self.clock,
            },
            &Refs {
                past,
                future: &self.reference,
                pp,
                pb,
                quarter: true,
            },
        );
        Ok(out.finish())
    }

    fn q(&self) -> Q {
        Q {
            scale: self.quant.clamp(1, 31),
            mpeg: false,
        }
    }

    fn time_ok(&self, pts: i64) -> Result<(), CodecError> {
        let ticks = ticks_of(pts)?;
        let seconds = ticks / self.resolution.max(1);
        if seconds < self.clock.seconds {
            return Err(CodecError::InvalidApi {
                what: "picture time",
            });
        }
        Ok(())
    }

    fn note(&mut self, bytes: usize, duration: i64) {
        if self.bitrate == 0 || self.resolution == 0 || duration <= 0 {
            return;
        }
        let bits = u64::try_from(bytes).unwrap_or(u64::MAX).saturating_mul(8);
        let target = u64::from(self.bitrate).saturating_mul(u64::try_from(duration).unwrap_or(0))
            / u64::from(self.resolution);
        self.budget = self.budget.saturating_add(target);
        self.spent = self.spent.saturating_add(bits);
        self.quant = next_quant(self.quant, self.spent, self.budget);
        if self.constant {
            self.window = self.window.saturating_add(duration);
            if self.window >= i64::from(self.resolution) {
                self.window = 0;
                self.spent = 0;
                self.budget = 0;
            }
        }
    }

    fn store(
        &self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        pts: i64,
        duration: i64,
    ) -> Result<Held, CodecError> {
        let width = usize::try_from(self.width).unwrap_or(0);
        let height = usize::try_from(self.height).unwrap_or(0);
        let rows = self.reference.mbh * 16;
        let chroma_rows = self.reference.mbh * 8;
        Ok(Held {
            y: pad_plane(y, width, height, self.stride, rows)?,
            u: pad_plane(u, width / 2, height / 2, self.chroma_stride, chroma_rows)?,
            v: pad_plane(v, width / 2, height / 2, self.chroma_stride, chroma_rows)?,
            pts,
            duration,
        })
    }
}

fn access(bytes: Vec<u8>, picture: &Held, keyframe: bool) -> Access {
    Access {
        bytes,
        pts: picture.pts,
        duration: picture.duration,
        keyframe,
    }
}

fn planes_of(picture: &Held, stride: usize) -> Planes<'_> {
    Planes {
        y: &picture.y,
        u: &picture.u,
        v: &picture.v,
        stride,
    }
}

fn layer_bytes(width: u32, height: u32, resolution: u32) -> Vec<u8> {
    let mut out = BitWriter::new();
    start(&mut out, 0x00);
    start(&mut out, 0x20);
    write_vol(
        &mut out,
        width,
        height,
        Layer {
            advanced: true,
            quarter: true,
            mpeg: false,
            low_delay: false,
            resolution,
        },
    );
    out.finish()
}

fn ticks_of(pts: i64) -> Result<u32, CodecError> {
    u32::try_from(pts).map_err(|_| CodecError::InvalidApi {
        what: "picture time",
    })
}

fn distances(past: i64, future: i64, between: i64) -> Result<(i32, i32), CodecError> {
    let bad = CodecError::InvalidApi {
        what: "picture time",
    };
    let pp = future.checked_sub(past).ok_or(bad)?;
    let pb = between.checked_sub(past).ok_or(bad)?;
    if pp <= 0 || pb <= 0 || pb >= pp {
        return Err(bad);
    }
    let pp = i32::try_from(pp).map_err(|_| bad)?;
    let pb = i32::try_from(pb).map_err(|_| bad)?;
    Ok((pp, pb))
}

fn pad_plane(
    src: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    rows: usize,
) -> Result<Vec<u8>, CodecError> {
    if width == 0 || height == 0 || stride < width || rows < height {
        return Err(CodecError::InvalidApi { what: "frame size" });
    }
    let need = width
        .checked_mul(height)
        .ok_or(CodecError::Limit { what: "frame size" })?;
    if src.len() < need {
        return Err(CodecError::InvalidApi {
            what: "plane length",
        });
    }
    let bytes = stride
        .checked_mul(rows)
        .ok_or(CodecError::Limit { what: "frame size" })?;
    let mut out = vec![0u8; bytes];
    for y in 0..height {
        let from = y * width;
        let dst = y * stride;
        out[dst..dst + width].copy_from_slice(&src[from..from + width]);
        let edge = src[from + width - 1];
        out[dst + width..dst + stride].fill(edge);
    }
    let last = (height - 1) * stride;
    for y in height..rows {
        out.copy_within(last..last + stride, y * stride);
    }
    Ok(out)
}

fn initial_quant(bitrate: u32, width: u32, height: u32) -> i32 {
    if bitrate == 0 || width == 0 || height == 0 {
        return 8;
    }
    let pixels = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(30);
    if pixels == 0 {
        return 8;
    }
    let bpp_milli = u64::from(bitrate).saturating_mul(1000) / pixels;
    if bpp_milli == 0 {
        return 28;
    }
    (16 + 4 * log2_ratio(200, bpp_milli)).clamp(4, 28)
}

fn log2_ratio(numer: u64, denom: u64) -> i32 {
    if numer == 0 || denom == 0 {
        return 0;
    }
    let mut n = numer;
    let mut d = denom;
    let mut shift = 0i32;
    while n < d && shift > -8 {
        n = n.saturating_mul(2);
        shift -= 1;
    }
    while n >= d.saturating_mul(2) && shift < 8 {
        d = d.saturating_mul(2);
        shift += 1;
    }
    shift
}

fn next_quant(quant: i32, spent: u64, budget: u64) -> i32 {
    if budget == 0 {
        return quant.clamp(1, 31);
    }
    let ratio = spent.saturating_mul(100) / budget;
    let stepped = if ratio > 150 {
        quant + 4
    } else if ratio > 120 {
        quant + 2
    } else if ratio > 105 {
        quant + 1
    } else if ratio < 50 {
        quant - 4
    } else if ratio < 80 {
        quant - 2
    } else if ratio < 95 {
        quant - 1
    } else {
        quant
    };
    stepped.clamp(1, 31)
}

#[cfg(test)]
mod tests {
    use super::{compensate, encode_advanced, encode_intra, encode_pictures};
    use crate::dec::Decoder;
    use crate::frame::Frame;
    use cutlass_decode::CodecError;

    fn decode_all(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut decoder = Decoder::with_max_pixels(8192 * 8192);
        let mut pictures = decoder.decode(bytes).expect("decode").pictures;
        pictures.extend(decoder.drain());
        pictures.into_iter().map(|picture| picture.bytes).collect()
    }

    fn decode(bytes: &[u8]) -> Vec<u8> {
        let pictures = decode_all(bytes);
        assert_eq!(pictures.len(), 1);
        pictures.into_iter().next().expect("picture")
    }

    fn mae(got: &[u8], want: &[u8]) -> f64 {
        let n = got.len().min(want.len()).max(1);
        let sum: u64 = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();
        sum as f64 / n as f64
    }

    #[test]
    fn a_flat_intra_picture_comes_back() {
        let y = vec![140u8; 48 * 32];
        let c = vec![120u8; 24 * 16];
        let bytes = encode_intra(&y, &c, &c, 48, 32, 4).expect("encode");
        let got = decode(&bytes);
        assert!(
            mae(&got[..48 * 32], &y) < 1.5,
            "{}",
            mae(&got[..48 * 32], &y)
        );
        assert!(
            mae(&got[48 * 32..], &{
                let mut chroma = c.clone();
                chroma.extend_from_slice(&c);
                chroma
            }) < 2.0
        );
    }

    #[test]
    fn a_ramp_intra_picture_stays_close() {
        let mut y = vec![0u8; 64 * 48];
        for row in 0..48 {
            for col in 0..64 {
                y[row * 64 + col] = (col * 3 + row) as u8;
            }
        }
        let mut u = vec![0u8; 32 * 24];
        let mut v = vec![0u8; 32 * 24];
        for (i, (cu, cv)) in u.iter_mut().zip(&mut v).enumerate() {
            *cu = 80 + (i % 40) as u8;
            *cv = 160;
        }
        let bytes = encode_intra(&y, &u, &v, 64, 48, 4).expect("encode");
        let got = decode(&bytes);
        assert!(
            mae(&got[..64 * 48], &y) < 6.0,
            "{}",
            mae(&got[..64 * 48], &y)
        );
    }

    #[test]
    fn ffmpeg_decodes_the_same_intra_picture() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!("ffmpeg is not installed; the intra stream was not checked independently");
            return;
        };
        let mut y = vec![0u8; 48 * 32];
        for (i, sample) in y.iter_mut().enumerate() {
            *sample = (40 + (i * 5) % 180) as u8;
        }
        let u = vec![90u8; 24 * 16];
        let v = vec![150u8; 24 * 16];
        let bytes = encode_intra(&y, &u, &v, 48, 32, 6).expect("encode");
        ffmpeg_matches(&ffmpeg, &bytes, &decode(&bytes), "intra");
    }

    #[test]
    fn a_repeated_picture_is_skipped() {
        let y = vec![140u8; 48 * 32];
        let c = vec![120u8; 24 * 16];
        let bytes = encode_pictures(&[(y.as_slice(), c.as_slice(), c.as_slice()); 2], 48, 32, 4)
            .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 2);
        assert_eq!(pictures[0], pictures[1]);
        assert_eq!(vop_kind(&bytes, 1), 1);
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the repeated picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "repeat");
    }

    #[test]
    fn a_shifted_picture_is_predicted() {
        let y0 = pattern(64, 48, 3);
        let u0 = pattern(32, 24, 9);
        let v0 = pattern(32, 24, 15);
        let y1 = shift_right(&y0, 64, 48, 4);
        let u1 = shift_right(&u0, 32, 24, 2);
        let v1 = shift_right(&v0, 32, 24, 2);
        let y2 = shift_right(&y0, 64, 48, 8);
        let u2 = shift_right(&u0, 32, 24, 4);
        let v2 = shift_right(&v0, 32, 24, 4);
        let bytes = encode_pictures(
            &[
                (y0.as_slice(), u0.as_slice(), v0.as_slice()),
                (y1.as_slice(), u1.as_slice(), v1.as_slice()),
                (y2.as_slice(), u2.as_slice(), v2.as_slice()),
            ],
            64,
            48,
            5,
        )
        .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 3);
        assert_eq!(vop_kind(&bytes, 0), 0);
        assert_eq!(vop_kind(&bytes, 1), 1);
        assert_eq!(vop_kind(&bytes, 2), 1);
        let (intra, predicted) = vop_lengths(&bytes);
        assert!(
            predicted < intra / 2,
            "P-VOP {predicted} bytes, I-VOP {intra} bytes"
        );
        let luma = 64 * 48;
        assert!(
            mae(&pictures[1][..luma], &y1) < 8.0,
            "{}",
            mae(&pictures[1][..luma], &y1)
        );
        assert!(
            mae(&pictures[2][..luma], &y2) < 8.0,
            "{}",
            mae(&pictures[2][..luma], &y2)
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the predicted picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "shift");
    }

    #[test]
    fn a_split_picture_uses_four_vectors() {
        let y0 = pattern(64, 48, 5);
        let u0 = pattern(32, 24, 11);
        let v0 = pattern(32, 24, 17);
        let y1 = shift_left_half(&y0, 64, 48, 16, 4);
        let u1 = shift_left_half(&u0, 32, 24, 8, 2);
        let v1 = shift_left_half(&v0, 32, 24, 8, 2);
        let bytes = encode_pictures(
            &[
                (y0.as_slice(), u0.as_slice(), v0.as_slice()),
                (y1.as_slice(), u1.as_slice(), v1.as_slice()),
            ],
            64,
            48,
            5,
        )
        .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 2);
        assert_eq!(vop_kind(&bytes, 1), 1);
        let (intra, predicted) = vop_lengths(&bytes);
        // One vector cannot describe both halves, and that P-VOP is larger
        // than the I-VOP. Four vectors are what makes this one smaller.
        assert!(
            predicted < intra,
            "P-VOP {predicted} bytes, I-VOP {intra} bytes"
        );
        assert!(
            mae(&pictures[1][..64 * 48], &y1) < 8.0,
            "{}",
            mae(&pictures[1][..64 * 48], &y1)
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the four-vector picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "four");
    }

    #[test]
    fn a_bidirectional_picture_is_shown_between_its_references() {
        let y0 = pattern(64, 48, 3);
        let u0 = pattern(32, 24, 9);
        let v0 = pattern(32, 24, 15);
        let y1 = shift_right(&y0, 64, 48, 2);
        let u1 = shift_right(&u0, 32, 24, 1);
        let v1 = shift_right(&v0, 32, 24, 1);
        let y2 = shift_right(&y0, 64, 48, 4);
        let u2 = shift_right(&u0, 32, 24, 2);
        let v2 = shift_right(&v0, 32, 24, 2);
        let bytes = encode_advanced(
            &[
                (y0.as_slice(), u0.as_slice(), v0.as_slice()),
                (y1.as_slice(), u1.as_slice(), v1.as_slice()),
                (y2.as_slice(), u2.as_slice(), v2.as_slice()),
            ],
            64,
            48,
            5,
            false,
            false,
        )
        .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 3);
        assert_eq!(vop_kind(&bytes, 0), 0);
        assert_eq!(vop_kind(&bytes, 1), 1);
        assert_eq!(vop_kind(&bytes, 2), 2);
        let luma = 64 * 48;
        assert!(
            mae(&pictures[0][..luma], &y0) < 8.0,
            "{}",
            mae(&pictures[0][..luma], &y0)
        );
        assert!(
            mae(&pictures[1][..luma], &y1) < 8.0,
            "{}",
            mae(&pictures[1][..luma], &y1)
        );
        assert!(
            mae(&pictures[2][..luma], &y2) < 8.0,
            "{}",
            mae(&pictures[2][..luma], &y2)
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the bidirectional picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "bidirectional");
    }

    #[test]
    fn a_repeated_bidirectional_picture_matches_its_references() {
        let y = vec![140u8; 48 * 32];
        let c = vec![120u8; 24 * 16];
        let frame = (y.as_slice(), c.as_slice(), c.as_slice());
        let bytes =
            encode_advanced(&[frame, frame, frame], 48, 32, 4, false, false).expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 3);
        assert_eq!(vop_kind(&bytes, 2), 2);
        assert_eq!(pictures[0], pictures[1]);
        assert_eq!(pictures[1], pictures[2]);
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the skipped bidirectional picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "b-skip");
    }

    #[test]
    fn a_quarter_sample_picture_lands_between_its_references() {
        let y0 = pattern(64, 48, 3);
        let u0 = pattern(32, 24, 9);
        let v0 = pattern(32, 24, 15);
        let (y1, u1, v1) = quarter_moved(&y0, &u0, &v0, 64, 48, (-1, 0));
        let (y2, u2, v2) = quarter_moved(&y0, &u0, &v0, 64, 48, (-2, 0));
        let bytes = encode_advanced(
            &[
                (y0.as_slice(), u0.as_slice(), v0.as_slice()),
                (y1.as_slice(), u1.as_slice(), v1.as_slice()),
                (y2.as_slice(), u2.as_slice(), v2.as_slice()),
            ],
            64,
            48,
            4,
            true,
            false,
        )
        .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 3);
        assert_eq!(vop_kind(&bytes, 0), 0);
        assert_eq!(vop_kind(&bytes, 1), 1);
        assert_eq!(vop_kind(&bytes, 2), 2);
        let luma = 64 * 48;
        assert!(
            mae(&pictures[1][..luma], &y1) < 8.0,
            "{}",
            mae(&pictures[1][..luma], &y1)
        );
        assert!(
            mae(&pictures[2][..luma], &y2) < 8.0,
            "{}",
            mae(&pictures[2][..luma], &y2)
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the quarter-sample picture was not checked independently"
            );
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "quarter");
    }

    #[test]
    fn an_mpeg_quantiser_reconstructs_with_the_default_matrices() {
        let y0 = pattern(64, 48, 7);
        let u0 = pattern(32, 24, 13);
        let v0 = pattern(32, 24, 19);
        let y1 = shift_right(&y0, 64, 48, 4);
        let u1 = shift_right(&u0, 32, 24, 2);
        let v1 = shift_right(&v0, 32, 24, 2);
        let bytes = encode_advanced(
            &[
                (y0.as_slice(), u0.as_slice(), v0.as_slice()),
                (y1.as_slice(), u1.as_slice(), v1.as_slice()),
            ],
            64,
            48,
            5,
            false,
            true,
        )
        .expect("encode");
        let pictures = decode_all(&bytes);
        assert_eq!(pictures.len(), 2);
        assert_eq!(vop_kind(&bytes, 0), 0);
        assert_eq!(vop_kind(&bytes, 1), 1);
        let luma = 64 * 48;
        assert!(
            mae(&pictures[0][..luma], &y0) < 8.0,
            "{}",
            mae(&pictures[0][..luma], &y0)
        );
        assert!(
            mae(&pictures[1][..luma], &y1) < 8.0,
            "{}",
            mae(&pictures[1][..luma], &y1)
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!("ffmpeg is not installed; the MPEG quantiser was not checked independently");
            return;
        };
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &bytes, &ours, "mpeg-quant");
    }

    /// The picture a quarter-sample vector `mv` predicts from these planes.
    fn quarter_moved(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        width: usize,
        height: usize,
        mv: (i32, i32),
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut reference = Frame::new(width / 16, height / 16);
        let chroma = (width / 2) * (height / 2);
        reference.y[..width * height].copy_from_slice(&y[..width * height]);
        reference.u[..chroma].copy_from_slice(&u[..chroma]);
        reference.v[..chroma].copy_from_slice(&v[..chroma]);
        let mut yo = vec![0u8; width * height];
        let mut uo = vec![0u8; chroma];
        let mut vo = vec![0u8; chroma];
        for mby in 0..height / 16 {
            for mbx in 0..width / 16 {
                let (py, pu, pv) = compensate(&reference, mbx, mby, mv, true);
                for row in 0..16 {
                    let start = (mby * 16 + row) * width + mbx * 16;
                    yo[start..start + 16].copy_from_slice(&py[row * 16..row * 16 + 16]);
                }
                for row in 0..8 {
                    let start = (mby * 8 + row) * (width / 2) + mbx * 8;
                    uo[start..start + 8].copy_from_slice(&pu[row * 8..row * 8 + 8]);
                    vo[start..start + 8].copy_from_slice(&pv[row * 8..row * 8 + 8]);
                }
            }
        }
        (yo, uo, vo)
    }

    fn pattern(width: usize, height: usize, salt: usize) -> Vec<u8> {
        let mut plane = vec![0u8; width * height];
        for row in 0..height {
            for col in 0..width {
                plane[row * width + col] =
                    ((col * 13 + row * 7 + (col ^ row) + salt) % 220 + 16) as u8;
            }
        }
        plane
    }

    /// Moves the left half of every `block` columns right by `by` pixels.
    /// The right half stays, so one vector per macroblock cannot describe it.
    fn shift_left_half(
        src: &[u8],
        width: usize,
        height: usize,
        block: usize,
        by: usize,
    ) -> Vec<u8> {
        let mut out = src.to_vec();
        let half = block / 2;
        for row in 0..height {
            for origin in (0..width).step_by(block) {
                for col in 0..half {
                    let x = origin + col;
                    let from = x.saturating_sub(by);
                    out[row * width + x] = src[row * width + from];
                }
            }
        }
        out
    }

    fn shift_right(src: &[u8], width: usize, height: usize, by: usize) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for row in 0..height {
            for col in 0..width {
                let from = col.saturating_sub(by);
                out[row * width + col] = src[row * width + from];
            }
        }
        out
    }

    /// `vop_coding_type` of VOP `index`: 0 intra, 1 predicted, 2 bidirectional.
    fn vop_kind(bytes: &[u8], index: usize) -> u8 {
        let mut seen = 0usize;
        for i in 0..bytes.len().saturating_sub(4) {
            if bytes[i..i + 4] == [0, 0, 1, 0xB6] {
                if seen == index {
                    return bytes[i + 4] >> 6;
                }
                seen += 1;
            }
        }
        0xFF
    }

    fn vop_lengths(bytes: &[u8]) -> (usize, usize) {
        let mut starts = Vec::new();
        for i in 0..bytes.len().saturating_sub(3) {
            if bytes[i..i + 4] == [0, 0, 1, 0xB6] {
                starts.push(i);
            }
        }
        let intra =
            starts.get(1).copied().unwrap_or(bytes.len()) - starts.first().copied().unwrap_or(0);
        let predicted = starts.get(2).copied().unwrap_or(bytes.len())
            - starts.get(1).copied().unwrap_or(bytes.len());
        (intra, predicted)
    }

    fn ffmpeg_matches(ffmpeg: &std::path::Path, bytes: &[u8], ours: &[u8], name: &str) {
        let dir = std::env::temp_dir().join(format!("cutlass-mpeg4-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let stream = dir.join("picture.m4v");
        let yuv = dir.join("picture.yuv");
        std::fs::write(&stream, bytes).expect("stream");
        let status = std::process::Command::new(ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-cpuflags",
                "0",
                "-flags",
                "+bitexact",
                "-idct",
                "simple",
                "-f",
                "m4v",
                "-i",
            ])
            .arg(&stream)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&yuv)
            .status()
            .expect("spawn");
        assert!(status.success(), "ffmpeg decode");
        let theirs = std::fs::read(&yuv).expect("yuv");
        assert_eq!(ours, theirs);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .map(std::path::PathBuf::from)
        .find(|path| {
            std::process::Command::new(path)
                .arg("-version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    }

    #[test]
    fn a_partial_macroblock_size_is_refused() {
        let y = vec![0u8; 16 * 8];
        let c = vec![0u8; 8 * 4];
        let refused = encode_intra(&y, &c, &c, 16, 8, 4);
        assert!(matches!(refused, Err(CodecError::InvalidApi { .. })));
        assert!(encode_intra(&y, &c, &c, 16, 16, 0).is_err());
    }

    #[test]
    fn quant_moves_toward_the_budget() {
        assert_eq!(super::next_quant(10, 200, 100), 14);
        assert_eq!(super::next_quant(10, 40, 100), 6);
        assert_eq!(super::next_quant(1, 10, 100), 1);
        assert_eq!(super::next_quant(31, 300, 100), 31);
        assert_eq!(super::next_quant(8, 100, 100), 8);
    }

    #[test]
    fn a_padded_picture_keeps_its_visible_size() {
        let (headers, packets, pictures) = coded_movie(64, 40, 0, 4);
        assert_eq!(pictures.len(), 4);
        assert_eq!(pictures[0].len(), 64 * 40 * 3 / 2);
        assert!(
            !headers.windows(4).any(|w| w == [0, 0, 1, 0xB6]),
            "a VOP does not belong in the layer headers"
        );
        assert!(
            packets
                .iter()
                .all(|packet| packet.starts_with(&[0, 0, 1, 0xB6]))
        );
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!("ffmpeg is not installed; the padded picture was not checked independently");
            return;
        };
        let mut stream = headers;
        for packet in &packets {
            stream.extend_from_slice(packet);
        }
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &stream, &ours, "padded");
    }

    #[test]
    fn a_second_of_motion_stays_near_the_requested_bitrate() {
        for target in [200_000u32, 400_000, 800_000] {
            let mut encoder = super::Pictures::open(64, 48, target, false).expect("open");
            let mut bytes = 0usize;
            for n in 0..30usize {
                let (y, u, v) = moving(64, 48, n);
                let packets = encoder
                    .push(super::PictureIn {
                        y: &y,
                        u: &u,
                        v: &v,
                        pts: i64::try_from(n).unwrap_or(0),
                        duration: 1,
                        timescale: 30,
                        keyframe: false,
                    })
                    .expect("push");
                for packet in packets {
                    bytes += packet.bytes.len();
                }
            }
            for packet in encoder.drain().expect("drain") {
                bytes += packet.bytes.len();
            }
            bytes += encoder.headers().len();
            let bits = bytes.saturating_mul(8);
            let wanted = usize::try_from(target).unwrap_or(0);
            let low = wanted * 3 / 4;
            let high = wanted * 5 / 4;
            assert!(
                bits >= low && bits <= high,
                "target {target}: payload {bits} bits, quant {}, wanted {low}..={high}",
                encoder.quant()
            );
        }
    }

    /// Pictures through the next second, so the time base increments, and
    /// FFmpeg reconstructs the same samples.
    #[test]
    fn pictures_past_one_second_match_ffmpeg() {
        let (headers, packets, pictures) = coded_movie(64, 48, 120_000, 32);
        assert!(pictures.len() > 30, "{}", pictures.len());
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!(
                "ffmpeg is not installed; the picture past one second was not checked independently"
            );
            return;
        };
        let mut stream = headers;
        for packet in &packets {
            stream.extend_from_slice(packet);
        }
        let mut ours = Vec::new();
        for picture in &pictures {
            ours.extend_from_slice(picture);
        }
        ffmpeg_matches(&ffmpeg, &stream, &ours, "one-second");
    }

    /// `count` moving pictures. The returned pictures are display order.
    fn coded_movie(
        width: usize,
        height: usize,
        bitrate: u32,
        count: usize,
    ) -> (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut encoder =
            super::Pictures::open(width as u32, height as u32, bitrate, false).expect("open");
        let mut packets = Vec::new();
        for n in 0..count {
            let (y, u, v) = moving(width, height, n);
            packets.extend(
                encoder
                    .push(super::PictureIn {
                        y: &y,
                        u: &u,
                        v: &v,
                        pts: i64::try_from(n).unwrap_or(0),
                        duration: 1,
                        timescale: 30,
                        keyframe: false,
                    })
                    .expect("push"),
            );
        }
        packets.extend(encoder.drain().expect("drain"));
        let headers = encoder.headers().to_vec();
        let mut stream = headers.clone();
        let bytes: Vec<Vec<u8>> = packets.into_iter().map(|packet| packet.bytes).collect();
        for packet in &bytes {
            stream.extend_from_slice(packet);
        }
        let pictures = decode_all(&stream);
        (headers, bytes, pictures)
    }

    fn moving(width: usize, height: usize, n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let shift = n * 2;
        let mut y = vec![0u8; width * height];
        let mut u = vec![0u8; (width / 2) * (height / 2)];
        let mut v = vec![0u8; (width / 2) * (height / 2)];
        for row in 0..height {
            for col in 0..width {
                y[row * width + col] = (((row + col + shift) * 3 + (col ^ row)) % 180 + 20) as u8;
            }
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                let at = row * (width / 2) + col;
                u[at] = ((col * 5 + row + shift) % 80 + 80) as u8;
                v[at] = ((row * 5 + col + shift / 2) % 80 + 80) as u8;
            }
        }
        (y, u, v)
    }
}
