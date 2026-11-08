//! A Simple-profile encoder.
//!
//! An I-VOP, then P-VOPs: H.263 quantisation, one or four motion vectors
//! per macroblock, no AC prediction. [`encode_advanced`] puts a B-VOP
//! between references and can use quarter-sample motion. Rate control is
//! not emitted here yet. The coded size is a
//! whole number of macroblocks; a caller with another size is refused
//! rather than cropped. Motion search stays inside ±8 pixels so every
//! vector fits `fcode` 1.

use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::bits::BitWriter;
use crate::block::{Quant, dequant_inter, dequant_intra};
use crate::frame::{Frame, Kind, MbRecord};
use crate::idct;
use crate::mc::{self, Pel, Rows};
use crate::tables::{
    CBPY, CoefTable, DC_CHROMA, DC_LUMA, MB_TYPE_B, MCBPC_INTER, MCBPC_INTRA, MVD, TCOEF_ESCAPE,
    ZIGZAG, dc_scaler_chroma, dc_scaler_luma, tables,
};
use crate::vop::clip_block;

const TIME_RESOLUTION: u32 = 25;

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
    let mut recon = Frame::new(mbw, mbh);
    let mut mv = vec![(0i32, 0i32); mbw * 2 * mbh * 2];
    let mut out = BitWriter::new();
    // A video object before its layer. A raw demuxer uses that start code
    // to recognise the stream; the pictures do not depend on it.
    start(&mut out, 0x00);
    start(&mut out, 0x20);
    write_vol(&mut out, width, height, false, false, true);
    for (index, &(y, u, v)) in frames.iter().enumerate() {
        let planes = Planes { y, u, v, stride: w };
        start(&mut out, 0xB6);
        if index == 0 {
            write_i_vop(&mut out, &planes, i32::from(quant), &mut recon, &mut mv);
        } else {
            write_p_vop(
                &mut out,
                &planes,
                i32::from(quant),
                index as u32,
                &mut recon,
                &mut mv,
                false,
            );
        }
    }
    Ok(out.finish())
}

/// Display-order pictures. With three or more, every other picture is a
/// B-VOP coded after the reference it precedes, and the layer is Advanced
/// Simple with `low_delay` clear so those pictures come back in display
/// order. `quarter` selects quarter-sample motion, which also makes the
/// layer Advanced Simple. `quant` is `vop_quant`, 1 to 31.
pub fn encode_advanced(
    frames: &[(&[u8], &[u8], &[u8])],
    width: u32,
    height: u32,
    quant: u8,
    quarter: bool,
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
    let quant = i32::from(quant);
    let mut reference = Frame::new(mbw, mbh);
    let mut mv = vec![(0i32, 0i32); mbw * 2 * mbh * 2];
    let mut earlier: Option<Frame> = None;
    let mut out = BitWriter::new();
    start(&mut out, 0x00);
    start(&mut out, 0x20);
    write_vol(&mut out, width, height, advanced, quarter, !has_b);
    for (n, &display) in refs.iter().enumerate() {
        let (y, u, v) = frames[display];
        let planes = Planes { y, u, v, stride: w };
        start(&mut out, 0xB6);
        if n == 0 {
            write_i_vop(&mut out, &planes, quant, &mut reference, &mut mv);
        } else {
            write_p_vop(
                &mut out,
                &planes,
                quant,
                display as u32,
                &mut reference,
                &mut mv,
                quarter,
            );
            if let Some(past) = earlier.as_ref() {
                let prev = refs[n - 1];
                for (step, &(y, u, v)) in frames[prev + 1..display].iter().enumerate() {
                    let between = Planes { y, u, v, stride: w };
                    start(&mut out, 0xB6);
                    write_b_vop(
                        &mut out,
                        &between,
                        quant,
                        (prev + 1 + step) as u32,
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
/// quarter-sample motion is legal. 4:2:0, square pixels, H.263 quantiser.
/// Version 5 is where `quarter_sample` sits. `low_delay` stays set when
/// the stream has no B-VOP.
fn write_vol(
    out: &mut BitWriter,
    width: u32,
    height: u32,
    advanced: bool,
    quarter: bool,
    low_delay: bool,
) {
    out.put(1, 0); // random_accessible_vol
    out.put(8, if advanced { 17 } else { 1 });
    if advanced {
        out.put(1, 1); // is_object_layer_identifier
        out.put(4, 5); // verid
        out.put(3, 1); // priority
    } else {
        out.put(1, 0);
    }
    out.put(4, 1); // square
    out.put(1, 1); // vol_control_parameters
    out.put(2, 1); // 4:2:0
    out.put(1, u32::from(low_delay));
    out.put(1, 0); // no vbv
    out.put(2, 0); // rectangular
    out.put(1, 1);
    out.put(16, TIME_RESOLUTION);
    out.put(1, 1);
    out.put(1, 0); // fixed_vop_rate
    out.put(1, 1);
    out.put(13, width);
    out.put(1, 1);
    out.put(13, height);
    out.put(1, 1);
    out.put(1, 0); // progressive
    out.put(1, 1); // obmc_disable
    if advanced {
        out.put(2, 0); // no sprite
    } else {
        out.put(1, 0); // no sprite
    }
    out.put(1, 0); // not_8_bit
    out.put(1, 0); // h.263 quantiser
    if advanced {
        out.put(1, u32::from(quarter));
    }
    out.put(1, 1); // complexity_estimation_disable
    out.put(1, 1); // resync_marker_disable
    out.put(1, 0); // data_partitioned
    if advanced {
        out.put(1, 0); // newpred
        out.put(1, 0); // reduced resolution
    }
    out.put(1, 0); // scalability
    out.stuff();
}

fn write_i_vop(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    quant: i32,
    recon: &mut Frame,
    mv: &mut [(i32, i32)],
) {
    out.put(2, 0); // I
    write_time(out, 0);
    out.put(1, 1); // coded
    out.put(3, 0); // intra_dc_vlc_thr: DC stays in the DC VLC
    out.put(5, quant as u32);
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
            let (levels, dcs) = write_intra_mb(out, planes, mbx, mby, quant, &mut dc);
            for k in 0..6 {
                paint_intra(recon, mbx, mby, k, &levels[k], dcs[k], quant);
            }
            set_record(recon, mby * mbw + mbx, Kind::Intra, [(0, 0); 4]);
        }
    }
    out.stuff();
}

/// `modulo_time_base`, the marker, `vop_time_increment`, and the marker.
fn write_time(out: &mut BitWriter, frame: u32) {
    let time_bits = (32 - (TIME_RESOLUTION - 1).leading_zeros()).max(1);
    let modulo = frame / TIME_RESOLUTION;
    let increment = frame % TIME_RESOLUTION;
    for _ in 0..modulo {
        out.bit(true);
    }
    out.bit(false);
    out.bit(true);
    out.put(time_bits, increment);
    out.bit(true);
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
    quant: i32,
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
            dc_scaler_luma(quant as u32)
        } else {
            dc_scaler_chroma(quant as u32)
        };
        let predicted = predict(&dc[plane_of(k)], across, bx, by);
        let qf = div_round(coef[0], scale).clamp(0, 2047 / scale);
        let diff = qf - (predicted + (scale >> 1)) / scale;
        let stored = qf * scale;
        dc[plane_of(k)][by as usize * across + bx as usize] = stored;
        reconstructed[k] = stored;
        levels[k][0] = diff;
        for i in 1..64 {
            levels[k][i] = quant_ac(coef[i], quant);
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
    quant: i32,
) {
    let mut ac = [0i32; 64];
    ac[1..].copy_from_slice(&levels[1..]);
    let mut coef = dequant_intra(&ac, dc, quant, Quant::H263);
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
    quant: i32,
    frame: u32,
    recon: &mut Frame,
    mv: &mut [(i32, i32)],
    quarter: bool,
) {
    out.put(2, 1); // P
    write_time(out, frame);
    out.bit(true); // coded
    out.bit(false); // rounding
    out.put(3, 0); // intra_dc_vlc_thr
    out.put(5, quant as u32);
    out.put(3, 1); // fcode_forward
    let past = recon.clone();
    let mut next = Frame::new(past.mbw, past.mbh);
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
            quant,
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
    quant: i32,
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
                levels[k][i] = quant_ac(coef[i], self.quant);
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
                add_inter(self.next, mbx, mby, k, &levels[k], self.quant);
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
    let width = (past.mbw * 16) as i32;
    let height = (past.mbh * 16) as i32;
    let [ry, ru, rv] = past.refs((width as u32, height as u32));
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

fn add_inter(frame: &mut Frame, mbx: usize, mby: usize, k: usize, levels: &[i32; 64], quant: i32) {
    let mut values = [0i32; 64];
    let mul = 2 * quant;
    let add = (quant - 1) | 1;
    for (slot, &level) in values.iter_mut().zip(levels) {
        *slot = if level > 0 {
            level * mul + add
        } else if level < 0 {
            level * mul - add
        } else {
            0
        };
    }
    let mut coef = dequant_inter(&values, quant, Quant::H263);
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

fn write_b_vop(out: &mut BitWriter, planes: &Planes<'_>, quant: i32, frame: u32, refs: &Refs<'_>) {
    out.put(2, 2); // B
    write_time(out, frame);
    out.bit(true); // coded
    out.put(3, 0); // intra_dc_vlc_thr
    out.put(5, quant as u32);
    out.put(3, 1); // fcode_forward
    out.put(3, 1); // fcode_backward
    let mbw = refs.past.mbw;
    let mbh = refs.past.mbh;
    for mby in 0..mbh {
        let mut last = [(0i32, 0i32); 2];
        for mbx in 0..mbw {
            write_b_mb(out, planes, quant, refs, (mbx, mby), &mut last);
        }
    }
    out.stuff();
}

fn write_b_mb(
    out: &mut BitWriter,
    planes: &Planes<'_>,
    quant: i32,
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
    let (levels, coded) = inter_residual(planes, mbx, mby, quant, &prediction);
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
    quant: i32,
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
        for (level, value) in levels[k].iter_mut().zip(coef) {
            *level = quant_ac(value, quant);
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
        let bytes = encode_advanced(&[frame, frame, frame], 48, 32, 4, false).expect("encode");
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
}
