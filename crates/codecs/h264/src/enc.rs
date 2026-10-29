//! High-profile CABAC encoder.
//!
//! Each picture is one slice. A new picture is an intra slice. A later picture
//! that equals the deblocked reconstruction is a P slice of skipped macroblocks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};
use cutlass_encode::{VideoEncoder, VideoInput};

use crate::dec::H264Decoder;
use crate::nal;
use crate::tables::{CABAC_MN, RANGE_LPS, TRANS_LPS, TRANS_MPS};

pub struct H264Encoder {
    width: u32,
    height: u32,
    mb_w: u32,
    mb_h: u32,
    length_size: u8,
    extra: Vec<u8>,
    cancel: Arc<AtomicBool>,
    drained: bool,
    idr_pic_id: u32,
    next_fn: u32,
    next_poc: u32,
    mirror: H264Decoder,
    reference: Option<Planes>,
    sps: Vec<u8>,
    pps: Vec<u8>,
}

struct Planes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

pub fn register_encoders(factory: &mut cutlass_encode::EncodeFactory) {
    factory.register_video(CodecId::H264, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(H264Encoder::open(params)?))
}

impl H264Encoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.width == 0 || params.height == 0 || params.width > 4096 || params.height > 4096 {
            return Err(CodecError::Limit { what: "coded size" });
        }
        if !params.width.is_multiple_of(2) || !params.height.is_multiple_of(2) {
            return Err(CodecError::Unsupported {
                what: "chroma format",
            });
        }
        let pixels = u64::from(params.width) * u64::from(params.height);
        if pixels > 8_388_608 {
            return Err(CodecError::Limit { what: "coded size" });
        }
        let length_size = if params.nal_length_size == 0 {
            0
        } else if params.nal_length_size == 4 {
            4
        } else {
            return Err(CodecError::Unsupported {
                what: "nal length size",
            });
        };
        let mb_w = params.width.div_ceil(16);
        let mb_h = params.height.div_ceil(16);
        let sps = sequence(mb_w, mb_h, params.width, params.height);
        let pps = picture();
        let extra = avcc(&sps, &pps);
        Ok(Self {
            width: params.width,
            height: params.height,
            mb_w,
            mb_h,
            length_size,
            extra,
            cancel: Arc::new(AtomicBool::new(false)),
            drained: false,
            idr_pic_id: 0,
            next_fn: 1,
            next_poc: 2,
            mirror: H264Decoder::open(params)?,
            reference: None,
            sps,
            pps,
        })
    }

    fn access_unit(&mut self, y: &[u8], u: &[u8], v: &[u8], key: bool) -> Vec<u8> {
        if !key && self.same_as_reference(y, u, v) {
            let slice = p_skip_slice(self.mb_w, self.mb_h, self.next_fn, self.next_poc);
            self.next_fn = (self.next_fn + 1) & 15;
            self.next_poc = (self.next_poc + 2) & 15;
            let mut unit = Vec::new();
            unit.extend(self.wrap(7, &self.sps));
            unit.extend(self.wrap(8, &self.pps));
            unit.extend(self.wrap(1, &slice));
            return unit;
        }
        let slice = intra_slice(
            self.mb_w,
            self.mb_h,
            self.idr_pic_id,
            &Source {
                y,
                u,
                v,
                width: self.width,
                height: self.height,
            },
        );
        self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
        self.next_fn = 1;
        self.next_poc = 2;
        let mut unit = Vec::new();
        unit.extend(self.wrap(7, &self.sps));
        unit.extend(self.wrap(8, &self.pps));
        unit.extend(self.wrap(5, &slice));
        unit
    }

    fn wrap(&self, kind: u8, rbsp: &[u8]) -> Vec<u8> {
        if self.length_size == 0 {
            nal::annex_b_nal(3, kind, rbsp)
        } else {
            nal::length_nal(self.length_size, 3, kind, rbsp)
        }
    }
}

impl VideoEncoder for H264Encoder {
    fn codec(&self) -> CodecId {
        CodecId::H264
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if input.picture.width() != self.width || input.picture.height() != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        if input.timescale == 0 {
            return Err(CodecError::InvalidApi { what: "timescale" });
        }
        self.drained = false;
        let duration = if input.duration == 0 {
            1
        } else {
            input.duration
        };
        let unit = self.access_unit(
            input.picture.y(),
            input.picture.u(),
            input.picture.v(),
            input.force_keyframe,
        );
        self.remember(&unit)?;
        Ok(vec![Packet::video(
            unit,
            input.pts,
            duration,
            input.timescale,
            true,
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.drained = false;
        self.idr_pic_id = 0;
        self.next_fn = 1;
        self.next_poc = 2;
        self.reference = None;
        self.mirror.reset();
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    fn extradata(&self) -> &[u8] {
        &self.extra
    }
}

fn sequence(mb_w: u32, mb_h: u32, width: u32, height: u32) -> Vec<u8> {
    let mut bits = Writer::new();
    bits.u(8, 100);
    bits.u(8, 0);
    bits.u(8, 31);
    bits.ue(0);
    bits.ue(1);
    bits.ue(0);
    bits.ue(0);
    bits.bit(false);
    bits.bit(false);
    bits.ue(0);
    bits.ue(0);
    bits.ue(0);
    bits.ue(1);
    bits.bit(false);
    bits.ue(mb_w.saturating_sub(1));
    bits.ue(mb_h.saturating_sub(1));
    bits.bit(true);
    bits.bit(true);
    let coded_w = mb_w.saturating_mul(16);
    let coded_h = mb_h.saturating_mul(16);
    let crop_r = coded_w.saturating_sub(width) / 2;
    let crop_b = coded_h.saturating_sub(height) / 2;
    if crop_r == 0 && crop_b == 0 {
        bits.bit(false);
    } else {
        bits.bit(true);
        bits.ue(0);
        bits.ue(crop_r);
        bits.ue(0);
        bits.ue(crop_b);
    }
    bits.bit(true);
    write_vui(&mut bits);
    bits.trailing();
    bits.bytes
}

fn write_vui(bits: &mut Writer) {
    for _ in 0..5 {
        bits.bit(false);
    }
    bits.bit(false);
    bits.bit(false);
    bits.bit(false);
    bits.bit(true);
    bits.bit(true);
    bits.ue(0);
    bits.ue(0);
    bits.ue(16);
    bits.ue(16);
    bits.ue(0);
    bits.ue(1);
}

fn picture() -> Vec<u8> {
    let mut bits = Writer::new();
    bits.ue(0);
    bits.ue(0);
    bits.bit(true);
    bits.bit(false);
    bits.ue(0);
    bits.ue(0);
    bits.ue(0);
    bits.bit(false);
    bits.u(2, 0);
    bits.se(0);
    bits.se(0);
    bits.se(0);
    bits.bit(true);
    bits.bit(false);
    bits.bit(false);
    bits.bit(true);
    bits.bit(false);
    bits.se(0);
    bits.trailing();
    bits.bytes
}

const QP: i32 = 26;

struct Source<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
}

struct Residual {
    /// Sixteen luma blocks in macroblock scan order, each in zigzag order.
    luma: [[i32; 16]; 16],
    cb: [i32; 4],
    cr: [i32; 4],
    /// Four chroma AC blocks in raster order, 15 non-DC zigzag coefficients each.
    cb_ac: [[i32; 15]; 4],
    cr_ac: [[i32; 15]; 4],
}

struct Nnz<'a> {
    luma: &'a mut [u8],
    cb: &'a mut [u8],
    cr: &'a mut [u8],
}

struct Flags<'a> {
    cb: &'a [bool],
    cr: &'a [bool],
    luma_cbp: &'a [u8],
    chroma_cbp: &'a [u8],
}

impl H264Encoder {
    fn same_as_reference(&self, y: &[u8], u: &[u8], v: &[u8]) -> bool {
        self.reference
            .as_ref()
            .is_some_and(|plane| plane.y == y && plane.u == u && plane.v == v)
    }

    fn remember(&mut self, unit: &[u8]) -> Result<(), CodecError> {
        let packet = Packet::video(unit.to_vec(), 0, 1, 1, true);
        let frames = self.mirror.push(&packet)?;
        let Some(frame) = frames.first() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "reference picture",
            });
        };
        self.reference = Some(Planes {
            y: frame.picture.y().to_vec(),
            u: frame.picture.u().to_vec(),
            v: frame.picture.v().to_vec(),
        });
        Ok(())
    }
}

/// P slice, cabac_init_idc 0. Every macroblock is skipped, so the decoder copies
/// the previous reference. FFmpeg's skip context is 11 when both neighbours are
/// skipped or absent, which is this whole picture.
fn p_skip_slice(mb_w: u32, mb_h: u32, frame_num: u32, poc: u32) -> Vec<u8> {
    let mut bits = Writer::new();
    bits.ue(0);
    bits.ue(0);
    bits.ue(0);
    bits.u(4, frame_num & 15);
    bits.u(4, poc & 15);
    bits.bit(false);
    bits.bit(false);
    bits.bit(false);
    bits.ue(0);
    bits.se(0);
    bits.ue(0);
    bits.se(0);
    bits.se(0);
    bits.ones_to_byte();
    {
        let mut cab = CabacEnc::start(&mut bits, QP, 1);
        let total = mb_w.saturating_mul(mb_h);
        for addr in 0..total {
            cab.decision(11, 1);
            cab.terminate(addr + 1 == total);
        }
    }
    bits.pad_byte();
    bits.bytes
}

fn intra_slice(mb_w: u32, mb_h: u32, idr_pic_id: u32, src: &Source<'_>) -> Vec<u8> {
    let mut bits = Writer::new();
    bits.ue(0);
    bits.ue(2);
    bits.ue(0);
    bits.u(4, 0);
    bits.ue(idr_pic_id);
    bits.u(4, 0);
    bits.bit(false);
    bits.bit(false);
    bits.se(0);
    bits.ue(0);
    bits.se(0);
    bits.se(0);
    bits.ones_to_byte();
    {
        let mut cab = CabacEnc::start(&mut bits, QP, 0);
        write_picture(&mut cab, mb_w, mb_h, src);
    }
    bits.pad_byte();
    bits.bytes
}

fn write_picture(cab: &mut CabacEnc<'_>, mb_w: u32, mb_h: u32, src: &Source<'_>) {
    let coded_w = mb_w.saturating_mul(16);
    let coded_h = mb_h.saturating_mul(16);
    let chroma_w = mb_w.saturating_mul(8);
    let count = mb_w as usize * mb_h as usize;
    let mut luma = vec![0u8; coded_w as usize * coded_h as usize];
    let mut cb = vec![0u8; chroma_w as usize * mb_h as usize * 8];
    let mut cr = vec![0u8; chroma_w as usize * mb_h as usize * 8];
    let mut cb_dc = vec![false; count];
    let mut cr_dc = vec![false; count];
    let mut luma_cbp = vec![0u8; count];
    let mut chroma_cbp = vec![0u8; count];
    let mut luma_nnz = vec![0u8; count * 16];
    let mut cb_nnz = vec![0u8; count * 4];
    let mut cr_nnz = vec![0u8; count * 4];
    let total = mb_w.saturating_mul(mb_h);
    for addr in 0..total {
        let mx = addr % mb_w;
        let my = addr / mb_w;
        let luma_blocks = intra4(&mut luma, coded_w, src.y, src.width, src.height, mx, my);
        let (cb_levels, cb_ac) =
            chroma_mb(&cb, chroma_w, src.u, src.width / 2, src.height / 2, mx, my);
        let (cr_levels, cr_ac) =
            chroma_mb(&cr, chroma_w, src.v, src.width / 2, src.height / 2, mx, my);
        let residual = Residual {
            luma: luma_blocks,
            cb: cb_levels,
            cr: cr_levels,
            cb_ac,
            cr_ac,
        };
        write_intra4(
            cab,
            mx,
            my,
            &residual,
            &Flags {
                cb: &cb_dc,
                cr: &cr_dc,
                luma_cbp: &luma_cbp,
                chroma_cbp: &chroma_cbp,
            },
            mb_w,
            &mut Nnz {
                luma: &mut luma_nnz,
                cb: &mut cb_nnz,
                cr: &mut cr_nnz,
            },
        );
        reconstruct_chroma(&mut cb, chroma_w, mx, my, &residual.cb, &residual.cb_ac);
        reconstruct_chroma(&mut cr, chroma_w, mx, my, &residual.cr, &residual.cr_ac);
        let index = addr as usize;
        luma_cbp[index] = luma_pattern(&residual.luma);
        chroma_cbp[index] = chroma_pattern(&residual);
        cb_dc[index] = residual.cb.iter().any(|level| *level != 0);
        cr_dc[index] = residual.cr.iter().any(|level| *level != 0);
        cab.terminate(addr + 1 == total);
    }
}

fn write_intra4(
    cab: &mut CabacEnc<'_>,
    mx: u32,
    my: u32,
    residual: &Residual,
    flags: &Flags<'_>,
    mb_w: u32,
    nnz: &mut Nnz<'_>,
) {
    let chroma = chroma_pattern(residual);
    let luma = luma_pattern(&residual.luma);
    cab.decision(3, 0);
    cab.decision(399, 0);
    for _ in 0..16 {
        cab.decision(68, 1);
    }
    cab.decision(64, 0);
    write_luma_cbp(cab, luma, flags.luma_cbp, mb_w, mx, my);
    write_chroma_cbp(cab, chroma, flags.chroma_cbp, mb_w, mx, my);
    if luma != 0 || chroma != 0 {
        cab.decision(60, 0);
    }
    write_luma4(cab, &residual.luma, luma, nnz.luma, mb_w, mx, my);
    if chroma != 0 {
        write_dc4(cab, &residual.cb, dc_inc(mx, my, flags.cb, mb_w));
        write_dc4(cab, &residual.cr, dc_inc(mx, my, flags.cr, mb_w));
    }
    if chroma == 2 {
        write_chroma_ac(cab, &residual.cb_ac, nnz.cb, mb_w, mx, my);
        write_chroma_ac(cab, &residual.cr_ac, nnz.cr, mb_w, mx, my);
    }
}

fn luma_pattern(blocks: &[[i32; 16]; 16]) -> u8 {
    let mut bits = 0u8;
    for (i, block) in blocks.iter().enumerate() {
        if block.iter().any(|level| *level != 0) {
            let group = (blk_y(i) / 2) * 2 + blk_x(i) / 2;
            bits |= 1 << group;
        }
    }
    bits
}

fn chroma_pattern(residual: &Residual) -> u8 {
    let ac = plane_ac(&residual.cb_ac) || plane_ac(&residual.cr_ac);
    let dc =
        residual.cb.iter().any(|level| *level != 0) || residual.cr.iter().any(|level| *level != 0);
    if ac { 2 } else { u8::from(dc) }
}

fn plane_ac(ac: &[[i32; 15]; 4]) -> bool {
    ac.iter().any(|block| block.iter().any(|level| *level != 0))
}

fn write_luma_cbp(cab: &mut CabacEnc<'_>, bits: u8, prev: &[u8], mb_w: u32, mx: u32, my: u32) {
    let mut current = 0u8;
    for bit in 0..4 {
        let (left_mask, above_mask) = match bit {
            0 => (0x02u8, 0x04u8),
            1 => (0x01, 0x08),
            2 => (0x08, 0x01),
            _ => (0x04, 0x02),
        };
        let left = if bit == 1 || bit == 3 {
            current & left_mask != 0
        } else {
            cbp_bit(prev, mb_w, mx as i32 - 1, my as i32, left_mask)
        };
        let above = if bit == 2 || bit == 3 {
            current & above_mask != 0
        } else {
            cbp_bit(prev, mb_w, mx as i32, my as i32 - 1, above_mask)
        };
        let ctx = usize::from(!left) + 2 * usize::from(!above);
        let on = bits & (1 << bit) != 0;
        cab.decision(73 + ctx, u8::from(on));
        if on {
            current |= 1 << bit;
        }
    }
}

fn cbp_bit(prev: &[u8], mb_w: u32, mx: i32, my: i32, mask: u8) -> bool {
    if mx < 0 || my < 0 {
        return true;
    }
    prev.get((my as u32 * mb_w + mx as u32) as usize)
        .is_some_and(|bits| bits & mask != 0)
}

fn write_chroma_cbp(cab: &mut CabacEnc<'_>, value: u8, prev: &[u8], mb_w: u32, mx: u32, my: u32) {
    let left = chroma_cbp_at(prev, mb_w, mx as i32 - 1, my as i32);
    let above = chroma_cbp_at(prev, mb_w, mx as i32, my as i32 - 1);
    let ctx = usize::from(left > 0) + 2 * usize::from(above > 0);
    if value == 0 {
        cab.decision(77 + ctx, 0);
        return;
    }
    cab.decision(77 + ctx, 1);
    let ctx = 4 + usize::from(left == 2) + 2 * usize::from(above == 2);
    cab.decision(77 + ctx, u8::from(value == 2));
}

fn chroma_cbp_at(prev: &[u8], mb_w: u32, mx: i32, my: i32) -> u8 {
    if mx < 0 || my < 0 {
        return 0;
    }
    prev.get((my as u32 * mb_w + mx as u32) as usize)
        .copied()
        .unwrap_or(0)
}

fn write_luma4(
    cab: &mut CabacEnc<'_>,
    blocks: &[[i32; 16]; 16],
    cbp: u8,
    nnz: &mut [u8],
    mb_w: u32,
    mx: u32,
    my: u32,
) {
    let stride = mb_w as usize * 4;
    for (i, block) in blocks.iter().enumerate() {
        let bx = blk_x(i);
        let by = blk_y(i);
        let group = (by / 2) * 2 + bx / 2;
        if cbp & (1 << group) == 0 {
            continue;
        }
        let inc = coded_nnz(nnz, mb_w, mx, my, bx as i32 - 1, by as i32, 4)
            + 2 * coded_nnz(nnz, mb_w, mx, my, bx as i32, by as i32 - 1, 4);
        let coded = block.iter().any(|level| *level != 0);
        cab.decision(93 + inc, u8::from(coded));
        if coded {
            write_residual(cab, 2, block, 16);
        }
        let count = block.iter().filter(|level| **level != 0).count();
        let x = mx as usize * 4 + bx;
        let y = my as usize * 4 + by;
        nnz[y * stride + x] = u8::try_from(count).unwrap_or(0);
    }
}

fn write_chroma_ac(
    cab: &mut CabacEnc<'_>,
    ac: &[[i32; 15]; 4],
    nnz: &mut [u8],
    mb_w: u32,
    mx: u32,
    my: u32,
) {
    let stride = mb_w as usize * 2;
    for (i, block) in ac.iter().enumerate() {
        let bx = i & 1;
        let by = i >> 1;
        let inc = coded_nnz(nnz, mb_w, mx, my, bx as i32 - 1, by as i32, 2)
            + 2 * coded_nnz(nnz, mb_w, mx, my, bx as i32, by as i32 - 1, 2);
        let coded = block.iter().any(|level| *level != 0);
        cab.decision(101 + inc, u8::from(coded));
        if coded {
            write_residual(cab, 4, block, 15);
        }
        let count = block.iter().filter(|level| **level != 0).count();
        let x = mx as usize * 2 + bx;
        let y = my as usize * 2 + by;
        nnz[y * stride + x] = u8::try_from(count).unwrap_or(0);
    }
}

fn coded_nnz(nnz: &[u8], mb_w: u32, mx: u32, my: u32, bx: i32, by: i32, grid: usize) -> usize {
    let (mx, bx) = if bx < 0 {
        (mx as i32 - 1, grid - 1)
    } else {
        (mx as i32, bx as usize)
    };
    let (my, by) = if by < 0 {
        (my as i32 - 1, grid - 1)
    } else {
        (my as i32, by as usize)
    };
    if mx < 0 || my < 0 {
        return 1;
    }
    let stride = mb_w as usize * grid;
    let x = mx as usize * grid + bx;
    let y = my as usize * grid + by;
    usize::from(nnz.get(y * stride + x).is_some_and(|count| *count > 0))
}

fn blk_x(i: usize) -> usize {
    (i & 1) | ((i >> 1) & 2)
}

fn blk_y(i: usize) -> usize {
    ((i >> 1) & 1) | ((i >> 2) & 2)
}

fn write_dc4(cab: &mut CabacEnc<'_>, levels: &[i32; 4], inc: usize) {
    let coded = levels.iter().any(|level| *level != 0);
    cab.decision(97 + inc, u8::from(coded));
    if coded {
        write_residual(cab, 3, levels, 4);
    }
}

/// Intra_4x4 DC prediction. Each block is reconstructed before the next one reads it.
fn intra4(
    recon: &mut [u8],
    stride: u32,
    src: &[u8],
    width: u32,
    height: u32,
    mx: u32,
    my: u32,
) -> [[i32; 16]; 16] {
    let mut blocks = [[0i32; 16]; 16];
    let x0 = mx * 16;
    let y0 = my * 16;
    for (i, levels) in blocks.iter_mut().enumerate() {
        let x = x0 + blk_x(i) as u32 * 4;
        let y = y0 + blk_y(i) as u32 * 4;
        let pred = predict4(recon, stride, x, y);
        *levels = quant4(&pred, src, width, height, x, y);
        paint4(recon, stride, x, y, &pred, levels);
    }
    blocks
}

fn predict4(recon: &[u8], stride: u32, x: u32, y: u32) -> [u8; 16] {
    let above_ok = y > 0;
    let left_ok = x > 0;
    let mut above = [128u8; 8];
    let mut left = [128u8; 4];
    let step = stride as usize;
    if above_ok {
        let row = (y - 1) as usize * step + x as usize;
        above[..4].copy_from_slice(&recon[row..row + 4]);
    }
    if left_ok {
        for (k, sample) in left.iter_mut().enumerate() {
            *sample = recon[(y as usize + k) * step + x as usize - 1];
        }
    }
    crate::recon::pred4(2, &above, &left, 128, above_ok, left_ok)
}

fn quant4(pred: &[u8; 16], src: &[u8], width: u32, height: u32, x: u32, y: u32) -> [i32; 16] {
    let mut pix = [0i32; 16];
    for row in 0..4u32 {
        for col in 0..4u32 {
            let sample = i32::from(at(src, width, height, x + col, y + row));
            let predicted = i32::from(pred[(row * 4 + col) as usize]);
            pix[(row * 4 + col) as usize] = sample - predicted;
        }
    }
    forward4(&mut pix);
    let mut levels = [0i32; 16];
    for (n, &pos) in crate::set::ZIGZAG4.iter().enumerate() {
        levels[n] = quant_coeff(pix[pos], QP, 15, qclass(pos % 4, pos / 4));
    }
    levels
}

fn paint4(recon: &mut [u8], stride: u32, x: u32, y: u32, pred: &[u8; 16], levels: &[i32; 16]) {
    let mut block = [0i32; 16];
    for (n, &pos) in crate::set::ZIGZAG4.iter().enumerate() {
        let (ry, rx) = (pos / 4, pos % 4);
        block[ry * 4 + rx] = crate::recon::scale4(levels[n], QP, rx, ry, 16);
    }
    crate::recon::idct4(&mut block);
    let step = stride as usize;
    for row in 0..4usize {
        for col in 0..4usize {
            let sample = (i32::from(pred[row * 4 + col]) + block[row * 4 + col]).clamp(0, 255);
            let at = (y as usize + row) * step + x as usize + col;
            recon[at] = u8::try_from(sample).unwrap_or(0);
        }
    }
}

fn qclass(x: usize, y: usize) -> usize {
    if x.is_multiple_of(2) && y.is_multiple_of(2) {
        0
    } else if !x.is_multiple_of(2) && !y.is_multiple_of(2) {
        1
    } else {
        2
    }
}

fn chroma_mb(
    recon: &[u8],
    stride: u32,
    src: &[u8],
    width: u32,
    height: u32,
    mx: u32,
    my: u32,
) -> ([i32; 4], [[i32; 15]; 4]) {
    let pred = predict_chroma(recon, stride, mx, my);
    let qp = crate::recon::chroma_qp(QP, 0);
    let mut dc = [0i32; 4];
    let mut ac = [[0i32; 15]; 4];
    let x0 = mx * 8;
    let y0 = my * 8;
    for (i, block) in ac.iter_mut().enumerate() {
        let bx = i & 1;
        let by = i >> 1;
        let pix = chroma_forward(&pred, src, width, height, (x0, y0), (bx, by));
        dc[i] = pix[0];
        for (n, &pos) in crate::set::ZIGZAG4.iter().enumerate().skip(1) {
            block[n - 1] = quant_coeff(pix[pos], qp, 15, qclass(pos % 4, pos / 4));
        }
    }
    let mixed = hadamard2(dc);
    let mut levels = [0i32; 4];
    for (index, coeff) in mixed.iter().enumerate() {
        levels[index] = quant_coeff(*coeff, qp, 16, 0);
    }
    (levels, ac)
}

fn chroma_forward(
    pred: &[u8; 64],
    src: &[u8],
    width: u32,
    height: u32,
    origin: (u32, u32),
    block: (usize, usize),
) -> [i32; 16] {
    let (x0, y0) = origin;
    let (bx, by) = block;
    let mut pix = [0i32; 16];
    for row in 0..4usize {
        for col in 0..4usize {
            let x = x0 + bx as u32 * 4 + col as u32;
            let y = y0 + by as u32 * 4 + row as u32;
            let sample = i32::from(at(src, width, height, x, y));
            let predicted = i32::from(pred[(by * 4 + row) * 8 + bx * 4 + col]);
            pix[row * 4 + col] = sample - predicted;
        }
    }
    forward4(&mut pix);
    pix
}

fn predict_chroma(recon: &[u8], stride: u32, mx: u32, my: u32) -> [u8; 64] {
    let mut above = [128u8; 8];
    let mut left = [128u8; 8];
    let x0 = mx * 8;
    let y0 = my * 8;
    if my > 0 {
        let row = (y0 - 1) * stride + x0;
        above.copy_from_slice(&recon[row as usize..row as usize + 8]);
    }
    if mx > 0 {
        for row in 0..8 {
            left[row] = recon[((y0 + row as u32) * stride + x0 - 1) as usize];
        }
    }
    crate::recon::pred_chroma(0, &above, &left, my > 0, mx > 0)
}

fn hadamard2(dc: [i32; 4]) -> [i32; 4] {
    [
        dc[0] + dc[1] + dc[2] + dc[3],
        dc[0] - dc[1] + dc[2] - dc[3],
        dc[0] + dc[1] - dc[2] - dc[3],
        dc[0] - dc[1] - dc[2] + dc[3],
    ]
}

fn at(src: &[u8], width: u32, height: u32, x: u32, y: u32) -> u8 {
    if x >= width || y >= height {
        128
    } else {
        src.get((y * width + x) as usize).copied().unwrap_or(128)
    }
}

fn forward4(block: &mut [i32; 16]) {
    for row in 0..4 {
        let i = row * 4;
        let s0 = block[i] + block[i + 3];
        let s3 = block[i] - block[i + 3];
        let s1 = block[i + 1] + block[i + 2];
        let s2 = block[i + 1] - block[i + 2];
        block[i] = s0 + s1;
        block[i + 1] = (s3 << 1) + s2;
        block[i + 2] = s0 - s1;
        block[i + 3] = s3 - (s2 << 1);
    }
    for col in 0..4 {
        let s0 = block[col] + block[12 + col];
        let s3 = block[col] - block[12 + col];
        let s1 = block[4 + col] + block[8 + col];
        let s2 = block[4 + col] - block[8 + col];
        block[col] = s0 + s1;
        block[4 + col] = (s3 << 1) + s2;
        block[8 + col] = s0 - s1;
        block[12 + col] = s3 - (s2 << 1);
    }
}

fn quant_coeff(coeff: i32, qp: i32, qbits_base: i32, class: usize) -> i32 {
    // Columns are the even/even, odd/odd, and mixed positions.
    const MF: [[i32; 3]; 6] = [
        [13_107, 8_192, 10_082],
        [11_916, 7_282, 9_362],
        [10_082, 6_554, 8_192],
        [9_362, 5_699, 7_282],
        [8_192, 5_243, 6_554],
        [7_282, 4_520, 5_699],
    ];
    let qbits = u32::try_from(qbits_base + qp / 6).unwrap_or(16);
    let row = usize::try_from(qp.rem_euclid(6)).unwrap_or(0);
    let mf = MF[row][class.min(2)];
    let bias = i64::from(1i32 << qbits.saturating_sub(1)) / 3;
    let mag = i64::from(coeff.unsigned_abs()) * i64::from(mf) + bias;
    let level = i32::try_from(mag >> qbits).unwrap_or(0);
    if coeff < 0 { -level } else { level }
}

fn reconstruct_chroma(
    recon: &mut [u8],
    stride: u32,
    mx: u32,
    my: u32,
    levels: &[i32; 4],
    ac: &[[i32; 15]; 4],
) {
    let pred = predict_chroma(recon, stride, mx, my);
    let qp = crate::recon::chroma_qp(QP, 0);
    let scaled = crate::recon::chroma_dc(levels[0], levels[1], levels[2], levels[3], qp, 16);
    let x0 = (mx * 8) as usize;
    let y0 = (my * 8) as usize;
    let stride = stride as usize;
    for (i, samples) in ac.iter().enumerate() {
        let bx = i & 1;
        let by = i >> 1;
        let mut coeffs = [0i32; 16];
        for (n, level) in samples.iter().enumerate() {
            coeffs[n + 1] = *level;
        }
        let mut block = [0i32; 16];
        for (n, &pos) in crate::set::ZIGZAG4.iter().enumerate() {
            let (ry, rx) = (pos / 4, pos % 4);
            let level = if n == 0 { scaled[i] } else { coeffs[n] };
            block[ry * 4 + rx] = if n == 0 {
                level
            } else {
                crate::recon::scale4(level, qp, rx, ry, 16)
            };
        }
        crate::recon::idct4(&mut block);
        for row in 0..4 {
            for col in 0..4 {
                let pred_s = i32::from(pred[(by * 4 + row) * 8 + bx * 4 + col]);
                let sample = (pred_s + block[row * 4 + col]).clamp(0, 255);
                let x = x0 + bx * 4 + col;
                let y = y0 + by * 4 + row;
                recon[y * stride + x] = u8::try_from(sample).unwrap_or(0);
            }
        }
    }
}

fn write_residual(cab: &mut CabacEnc<'_>, cat: usize, coeffs: &[i32], max: usize) {
    let (sig_base, last_base, level_base) = match cat {
        0 => (105, 166, 227),
        1 => (120, 181, 237),
        2 => (134, 195, 247),
        3 => (149, 210, 257),
        _ => (152, 213, 266),
    };
    let last = coeffs
        .iter()
        .take(max)
        .rposition(|coeff| *coeff != 0)
        .unwrap_or(0);
    let mut index = Vec::new();
    let end = max.saturating_sub(1);
    for (pos, coeff) in coeffs.iter().take(end).enumerate() {
        let on = *coeff != 0;
        let sig = if cat == 3 {
            sig_base + pos.min(2)
        } else {
            sig_base + pos
        };
        cab.decision(sig, u8::from(on));
        if on {
            index.push(pos);
            let last_ctx = if cat == 3 {
                last_base + pos.min(2)
            } else {
                last_base + pos
            };
            cab.decision(last_ctx, u8::from(pos == last));
            if pos == last {
                break;
            }
        }
    }
    if last == end {
        index.push(end);
    }
    write_levels(cab, level_base, &index, coeffs);
}

fn write_levels(cab: &mut CabacEnc<'_>, level_base: usize, index: &[usize], coeffs: &[i32]) {
    let level1 = [1usize, 2, 3, 4, 0, 0, 0, 0];
    let gt1 = [5usize, 5, 5, 5, 6, 7, 8, 9];
    let trans1 = [1usize, 2, 3, 3, 4, 5, 6, 7];
    let trans_gt = [4usize, 4, 4, 4, 5, 6, 7, 7];
    let mut node = 0usize;
    for &scan in index.iter().rev() {
        let value = coeffs.get(scan).copied().unwrap_or(0);
        let abs = value.unsigned_abs();
        let neg = u8::from(value < 0);
        if abs <= 1 {
            cab.decision(level_base + level1[node], 0);
            node = trans1[node];
        } else {
            cab.decision(level_base + level1[node], 1);
            let ctx = level_base + gt1[node];
            node = trans_gt[node];
            let mut coded = 2u32;
            while coded < abs && coded < 15 {
                cab.decision(ctx, 1);
                coded += 1;
            }
            if abs < 15 {
                cab.decision(ctx, 0);
            } else {
                write_suffix(cab, abs);
            }
        }
        cab.bypass(neg);
    }
}

fn write_suffix(cab: &mut CabacEnc<'_>, abs: u32) {
    let mag = abs.saturating_sub(14).max(1);
    let zeros = 31 - mag.leading_zeros();
    for _ in 0..zeros {
        cab.bypass(1);
    }
    cab.bypass(0);
    for shift in (0..zeros).rev() {
        cab.bypass(u8::from(((mag >> shift) & 1) == 1));
    }
}

fn dc_inc(mx: u32, my: u32, dc_coded: &[bool], mb_w: u32) -> usize {
    let left = if mx == 0 {
        1
    } else {
        usize::from(dc_coded[(my * mb_w + mx - 1) as usize])
    };
    let above = if my == 0 {
        1
    } else {
        usize::from(dc_coded[((my - 1) * mb_w + mx) as usize])
    };
    left + 2 * above
}

fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut out = vec![1, 100, 0, 31, 0xff, 0xe1];
    push_avcc_nal(&mut out, &raw_nal(3, 7, sps));
    out.push(1);
    push_avcc_nal(&mut out, &raw_nal(3, 8, pps));
    out
}

fn raw_nal(ref_idc: u8, kind: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut nal = nal::length_nal(1, ref_idc, kind, rbsp);
    nal.remove(0);
    nal
}

fn push_avcc_nal(out: &mut Vec<u8>, nal: &[u8]) {
    let len = u16::try_from(nal.len()).unwrap_or(u16::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(nal);
}

struct Writer {
    bytes: Vec<u8>,
    acc: u32,
    bits: u32,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            acc: 0,
            bits: 0,
        }
    }

    fn bit(&mut self, one: bool) {
        self.acc = (self.acc << 1) | u32::from(one);
        self.bits += 1;
        if self.bits == 8 {
            self.bytes.push(u8::try_from(self.acc).unwrap_or(0));
            self.acc = 0;
            self.bits = 0;
        }
    }

    fn u(&mut self, n: u32, value: u32) {
        for shift in (0..n).rev() {
            self.bit(((value >> shift) & 1) == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value.saturating_add(1);
        let width = 31 - code.leading_zeros();
        for _ in 0..width {
            self.bit(false);
        }
        self.u(width + 1, code);
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            u32::try_from(value).unwrap_or(0).saturating_mul(2) - 1
        } else {
            value.unsigned_abs().saturating_mul(2)
        };
        self.ue(code);
    }

    fn ones_to_byte(&mut self) {
        while !self.bits.is_multiple_of(8) {
            self.bit(true);
        }
    }

    fn pad_byte(&mut self) {
        while self.bits != 0 {
            self.bit(false);
        }
    }

    fn trailing(&mut self) {
        self.bit(true);
        self.pad_byte();
    }
}

struct CabacEnc<'a> {
    out: &'a mut Writer,
    low: u32,
    range: u32,
    outstanding: u32,
    first: bool,
    state: [u8; 1024],
    mps: [u8; 1024],
}

impl<'a> CabacEnc<'a> {
    fn start(out: &'a mut Writer, qp: i32, init: usize) -> Self {
        let mut eng = Self {
            out,
            low: 0,
            range: 510,
            outstanding: 0,
            first: true,
            state: [0; 1024],
            mps: [0; 1024],
        };
        let slice_qp = qp.clamp(0, 51);
        let init = init.min(3);
        for (ctx, &(m, n)) in CABAC_MN[init].iter().enumerate() {
            if ctx == 276 {
                eng.state[276] = 63;
                eng.mps[276] = 0;
                continue;
            }
            let pre = (((i32::from(m) * slice_qp) >> 4) + i32::from(n)).clamp(1, 126);
            if pre <= 63 {
                eng.state[ctx] = u8::try_from(63 - pre).unwrap_or(0);
                eng.mps[ctx] = 0;
            } else {
                eng.state[ctx] = u8::try_from(pre - 64).unwrap_or(0);
                eng.mps[ctx] = 1;
            }
        }
        eng
    }

    fn decision(&mut self, ctx: usize, bin: u8) {
        let ctx = ctx.min(1023);
        let state = usize::from(self.state[ctx]);
        let mps = self.mps[ctx];
        let zone = usize::try_from((self.range >> 6) & 3).unwrap_or(0);
        let lps = u32::from(RANGE_LPS[state][zone]);
        self.range -= lps;
        if bin == mps {
            self.state[ctx] = TRANS_MPS[state];
        } else {
            self.low += self.range;
            self.range = lps;
            if state == 0 {
                self.mps[ctx] = 1 - mps;
            }
            self.state[ctx] = TRANS_LPS[state];
        }
        self.renorm();
    }

    fn bypass(&mut self, bin: u8) {
        self.low <<= 1;
        if bin != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.low -= 1024;
            self.put_bit(true);
        } else if self.low < 512 {
            self.put_bit(false);
        } else {
            self.outstanding += 1;
            self.low -= 512;
        }
    }

    fn terminate(&mut self, end: bool) {
        self.range = self.range.saturating_sub(2);
        if end {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    fn renorm(&mut self) {
        for _ in 0..32 {
            if self.range >= 256 {
                break;
            }
            if self.low < 256 {
                self.put_bit(false);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(true);
            } else {
                self.outstanding += 1;
                self.low -= 256;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    fn put_bit(&mut self, bit: bool) {
        if self.first {
            self.first = false;
        } else {
            self.out.bit(bit);
        }
        while self.outstanding > 0 {
            self.out.bit(!bit);
            self.outstanding -= 1;
        }
    }

    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        self.put_bit(((self.low >> 9) & 1) == 1);
        let tail = ((self.low >> 7) & 3) | 1;
        self.out.bit(((tail >> 1) & 1) == 1);
        self.out.bit((tail & 1) == 1);
    }
}

#[cfg(test)]
mod tests {
    use super::H264Encoder;
    use cutlass_decode::{CodecId, CodecParams, VideoDecoder, Yuv420p};
    use cutlass_encode::{VideoEncoder, VideoInput};

    use crate::dec::H264Decoder;

    #[test]
    fn cabac_decision_round_trips() {
        use crate::bits::Bits;
        use crate::cabac::Cabac;
        let mut bits = super::Writer::new();
        {
            let mut cab = super::CabacEnc::start(&mut bits, 26, 0);
            cab.decision(3, 1);
            cab.decision(6, 0);
            cab.decision(7, 0);
            cab.decision(9, 1);
            cab.decision(10, 0);
            cab.bypass(1);
            cab.bypass(0);
            cab.bypass(1);
            cab.terminate(true);
        }
        bits.pad_byte();
        let mut reader = Bits::new(&bits.bytes);
        let mut dec = Cabac::start(&mut reader, 26, 0).unwrap();
        assert_eq!(dec.decision(3).unwrap(), 1);
        assert_eq!(dec.decision(6).unwrap(), 0);
        assert_eq!(dec.decision(7).unwrap(), 0);
        assert_eq!(dec.decision(9).unwrap(), 1);
        assert_eq!(dec.decision(10).unwrap(), 0);
        assert_eq!(dec.bypass().unwrap(), 1);
        assert_eq!(dec.bypass().unwrap(), 0);
        assert_eq!(dec.bypass().unwrap(), 1);
        assert!(dec.terminate().unwrap());
    }

    #[test]
    fn slice_header_reaches_intra4() {
        use crate::bits::Bits;
        use crate::cabac::Cabac;
        let (y, u, v) = gray_planes();
        let rbsp = super::intra_slice(
            4,
            4,
            0,
            &super::Source {
                y: &y,
                u: &u,
                v: &v,
                width: 64,
                height: 64,
            },
        );
        let mut reader = Bits::new(&rbsp);
        let _first = reader.ue().unwrap();
        let _slice_type = reader.ue().unwrap();
        let _pps = reader.ue().unwrap();
        let _fn = reader.u(4).unwrap();
        let _idr = reader.ue().unwrap();
        let _poc = reader.u(4).unwrap();
        let _no = reader.bit().unwrap();
        let _long = reader.bit().unwrap();
        let _qp = reader.se().unwrap();
        let deblock = reader.ue().unwrap();
        if deblock != 1 {
            let _ = reader.se().unwrap();
            let _ = reader.se().unwrap();
        }
        reader.cabac_align().unwrap();
        let mut cab = Cabac::start(&mut reader, 26, 0).unwrap();
        for addr in 0..16 {
            let mx = addr % 4;
            let my = addr / 4;
            assert_eq!(cab.decision(3).unwrap(), 0, "mb {addr} is intra 4x4");
            assert_eq!(cab.decision(399).unwrap(), 0, "mb {addr} transform");
            for _ in 0..16 {
                assert_eq!(cab.decision(68).unwrap(), 1, "mb {addr} mode");
            }
            assert_eq!(cab.decision(64).unwrap(), 0, "mb {addr} chroma mode");
            for bit in 0..4 {
                let left = if bit == 1 || bit == 3 { false } else { mx == 0 };
                let above = if bit == 2 || bit == 3 { false } else { my == 0 };
                let ctx = usize::from(!left) + 2 * usize::from(!above);
                assert_eq!(cab.decision(73 + ctx).unwrap(), 0, "mb {addr} cbp {bit}");
            }
            assert_eq!(cab.decision(77).unwrap(), 0, "mb {addr} chroma cbp");
            let end = cab.terminate().unwrap();
            assert_eq!(end, addr == 15, "mb {addr} end");
        }
    }

    #[test]
    fn gray_intra_round_trips() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let picture = Yuv420p::solid(64, 64, 128, 128, 128).unwrap();
        let input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let packets = encoder.push(&input).unwrap();
        assert_eq!(packets.len(), 1);
        let nals = crate::nal::access_unit(&packets[0].data, 0).unwrap();
        assert!(nals.len() >= 3, "nals {}", nals.len());
        let slice = nals.iter().find(|nal| nal.kind == 5).unwrap();
        assert_eq!(slice.rbsp, {
            let (y, u, v) = gray_planes();
            super::intra_slice(
                4,
                4,
                0,
                &super::Source {
                    y: &y,
                    u: &u,
                    v: &v,
                    width: 64,
                    height: 64,
                },
            )
        });
        let sps_nal = nals.iter().find(|nal| nal.kind == 7).unwrap();
        let sps = crate::set::parse_sps(&sps_nal.rbsp).unwrap();
        assert_eq!(sps.log2_max_frame_num, 4, "frame num");
        assert_eq!(sps.log2_max_poc_lsb, 4, "poc");
        assert_eq!(sps.mb_width, 4);
        assert_eq!((sps.width, sps.height), (64, 64));
        let mut stored = [const { None }; crate::set::MAX_SPS];
        stored[0] = Some(sps);
        let pps_nal = nals.iter().find(|nal| nal.kind == 8).unwrap();
        let pps = crate::set::parse_pps(&pps_nal.rbsp, &stored).unwrap();
        assert!(pps.cabac);
        assert!(pps.deblocking_control);
        assert_eq!(pps.init_qp, 26);
        assert!(pps.transform_8x8);
        assert!(!pps.weighted_pred);
        assert_eq!(pps.weighted_bipred, 0);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let frames = decoder.push(&packets[0]).unwrap();
        let rest = decoder.drain().unwrap();
        let frames = if frames.is_empty() { rest } else { frames };
        assert_eq!(frames.len(), 1);
        assert!(frames[0].picture.y().iter().all(|sample| *sample == 128));
        assert!(frames[0].picture.u().iter().all(|sample| *sample == 128));
        assert!(frames[0].picture.v().iter().all(|sample| *sample == 128));
        if let Some(ffmpeg) = ffmpeg() {
            let dir = std::env::temp_dir().join("cutlass-h264-enc-gray");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let coded = dir.join("gray.h264");
            let yuv = dir.join("gray.yuv");
            std::fs::write(&coded, &packets[0].data).unwrap();
            let decoded = std::process::Command::new(ffmpeg)
                .args(["-y", "-i"])
                .arg(&coded)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(decoded.success(), "ffmpeg rejected the intra picture");
            let raw = std::fs::read(&yuv).unwrap();
            assert!(raw.iter().all(|sample| *sample == 128));
        }
    }

    #[test]
    fn length_prefixed_intra_round_trips() {
        let params = CodecParams::video(CodecId::H264, 32, 16);
        let mut encoder = H264Encoder::open(&params).unwrap();
        let picture = Yuv420p::solid(32, 16, 128, 128, 128).unwrap();
        let input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let packets = encoder.push(&input).unwrap();
        let mut open = params.clone();
        open.extra = encoder.extradata().to_vec();
        let mut decoder = H264Decoder::open(&open).unwrap();
        let frames = decoder.push(&packets[0]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].picture.width(), 32);
        assert!(frames[0].picture.y().iter().all(|sample| *sample == 128));
    }

    #[test]
    fn solid_luma_keeps_its_level() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let picture = Yuv420p::solid(64, 64, 200, 40, 220).unwrap();
        let input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let packets = encoder.push(&input).unwrap();
        let mut decoder = H264Decoder::open(&params).unwrap();
        let frames = decoder.push(&packets[0]).unwrap();
        assert_eq!(frames.len(), 1);
        let y = frames[0].picture.y();
        let worst = y
            .iter()
            .map(|sample| sample.abs_diff(200))
            .max()
            .unwrap_or(255);
        assert!(
            worst <= 8,
            "solid 200 reconstructed with error {worst}, first {}",
            y[0]
        );
        let u_err = frames[0]
            .picture
            .u()
            .iter()
            .map(|sample| sample.abs_diff(40))
            .max()
            .unwrap_or(255);
        let v_err = frames[0]
            .picture
            .v()
            .iter()
            .map(|sample| sample.abs_diff(220))
            .max()
            .unwrap_or(255);
        assert!(u_err <= 8, "chroma u error {u_err}");
        assert!(v_err <= 8, "chroma v error {v_err}");
        if let Some(ffmpeg) = ffmpeg() {
            let dir = std::env::temp_dir().join("cutlass-h264-enc-solid");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let coded = dir.join("solid.h264");
            let yuv = dir.join("solid.yuv");
            std::fs::write(&coded, &packets[0].data).unwrap();
            let decoded = std::process::Command::new(ffmpeg)
                .args(["-y", "-i"])
                .arg(&coded)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(decoded.success(), "ffmpeg rejected the solid picture");
            let raw = std::fs::read(&yuv).unwrap();
            assert_eq!(raw[..y.len()], y[..]);
        }
    }

    #[test]
    fn ramp_ac_matches_ffmpeg() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let mut y = vec![0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                y[row * 64 + col] = u8::try_from(col * 3).unwrap_or(0);
            }
        }
        let picture =
            Yuv420p::try_new(64, 64, y.clone(), vec![128; 32 * 32], vec![128; 32 * 32]).unwrap();
        let input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let packets = encoder.push(&input).unwrap();
        let mut decoder = H264Decoder::open(&params).unwrap();
        let frames = decoder.push(&packets[0]).unwrap();
        assert_eq!(frames.len(), 1);
        let got = frames[0].picture.y();
        let left: u32 = (0..64).map(|row| u32::from(got[row * 64])).sum::<u32>() / 64;
        let right: u32 = (0..64)
            .map(|row| u32::from(got[row * 64 + 63]))
            .sum::<u32>()
            / 64;
        assert!(
            left + 100 < right,
            "ramp did not rise, left {left} right {right}"
        );
        if let Some(ffmpeg) = ffmpeg() {
            let dir = std::env::temp_dir().join("cutlass-h264-enc-ramp");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let coded = dir.join("ramp.h264");
            let yuv = dir.join("ramp.yuv");
            std::fs::write(&coded, &packets[0].data).unwrap();
            let decoded = std::process::Command::new(ffmpeg)
                .args(["-y", "-i"])
                .arg(&coded)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(decoded.success(), "ffmpeg rejected the ramp");
            let raw = std::fs::read(&yuv).unwrap();
            assert_eq!(&raw[..got.len()], got);
        }
    }

    #[test]
    fn chroma_ramp_matches_ffmpeg() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let mut u = vec![0u8; 32 * 32];
        for row in 0..32 {
            for col in 0..32 {
                u[row * 32 + col] = u8::try_from(col * 6).unwrap_or(0);
            }
        }
        let picture =
            Yuv420p::try_new(64, 64, vec![128; 64 * 64], u.clone(), vec![128; 32 * 32]).unwrap();
        let input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let packets = encoder.push(&input).unwrap();
        let mut decoder = H264Decoder::open(&params).unwrap();
        let frames = decoder.push(&packets[0]).unwrap();
        assert_eq!(frames.len(), 1);
        let got = frames[0].picture.u();
        let left: u32 = (0..32).map(|row| u32::from(got[row * 32])).sum::<u32>() / 32;
        let right: u32 = (0..32)
            .map(|row| u32::from(got[row * 32 + 31]))
            .sum::<u32>()
            / 32;
        assert!(
            left + 80 < right,
            "chroma ramp did not rise, left {left} right {right}"
        );
        if let Some(ffmpeg) = ffmpeg() {
            let dir = std::env::temp_dir().join("cutlass-h264-enc-chroma");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let coded = dir.join("chroma.h264");
            let yuv = dir.join("chroma.yuv");
            std::fs::write(&coded, &packets[0].data).unwrap();
            let decoded = std::process::Command::new(ffmpeg)
                .args(["-y", "-i"])
                .arg(&coded)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(decoded.success(), "ffmpeg rejected the chroma ramp");
            let raw = std::fs::read(&yuv).unwrap();
            let y_len = frames[0].picture.y().len();
            let u_len = got.len();
            assert_eq!(&raw[..y_len], frames[0].picture.y());
            assert_eq!(&raw[y_len..y_len + u_len], got);
            assert_eq!(&raw[y_len + u_len..], frames[0].picture.v());
        }
    }

    #[test]
    fn repeated_constant_picture_is_a_p_skip() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let picture = Yuv420p::solid(64, 64, 128, 128, 128).unwrap();
        let mut input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let first = encoder.push(&input).unwrap();
        input.force_keyframe = false;
        input.pts = 1;
        let second = encoder.push(&input).unwrap();
        let second_nals = crate::nal::access_unit(&second[0].data, 0).unwrap();
        assert!(
            second_nals.iter().any(|nal| nal.kind == 1),
            "a repeated constant picture stays intra"
        );
        let mut decoder = H264Decoder::open(&params).unwrap();
        let a = decoder.push(&first[0]).unwrap();
        let b = decoder.push(&second[0]).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert!(b[0].picture.y().iter().all(|sample| *sample == 128));
        assert!(b[0].picture.u().iter().all(|sample| *sample == 128));
        assert!(b[0].picture.v().iter().all(|sample| *sample == 128));
        let mut ramp = vec![0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                ramp[row * 64 + col] = u8::try_from(col * 3).unwrap_or(0);
            }
        }
        let ramp_pic =
            Yuv420p::try_new(64, 64, ramp, vec![128; 32 * 32], vec![128; 32 * 32]).unwrap();
        input.picture = ramp_pic.clone();
        input.force_keyframe = false;
        input.pts = 2;
        let _ = encoder.push(&input).unwrap();
        input.pts = 3;
        let again = encoder.push(&input).unwrap();
        let again_nals = crate::nal::access_unit(&again[0].data, 0).unwrap();
        assert!(
            again_nals.iter().any(|nal| nal.kind == 5),
            "a repeated ramp is not a skip; deblock changes the reference"
        );
        if let Some(ffmpeg) = ffmpeg() {
            let dir = std::env::temp_dir().join("cutlass-h264-enc-pskip");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let coded = dir.join("skip.h264");
            let yuv = dir.join("skip.yuv");
            let mut bytes = first[0].data.clone();
            bytes.extend_from_slice(&second[0].data);
            std::fs::write(&coded, &bytes).unwrap();
            let decoded = std::process::Command::new(ffmpeg)
                .args(["-y", "-i"])
                .arg(&coded)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&yuv)
                .status()
                .unwrap();
            assert!(decoded.success(), "ffmpeg rejected the P skip");
            let raw = std::fs::read(&yuv).unwrap();
            assert_eq!(raw.len(), 64 * 64 * 3 / 2 * 2);
            assert!(raw.iter().all(|sample| *sample == 128));
        }
    }

    #[test]
    fn reconstructed_picture_is_a_p_skip() {
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let mut y = vec![0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                y[row * 64 + col] = u8::try_from(col * 3).unwrap_or(0);
            }
        }
        let picture = Yuv420p::try_new(64, 64, y, vec![128; 32 * 32], vec![128; 32 * 32]).unwrap();
        let mut input = VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: true,
        };
        let first = encoder.push(&input).unwrap();
        let mut decoder = H264Decoder::open(&params).unwrap();
        let decoded = decoder.push(&first[0]).unwrap();
        let again = Yuv420p::try_new(
            64,
            64,
            decoded[0].picture.y().to_vec(),
            decoded[0].picture.u().to_vec(),
            decoded[0].picture.v().to_vec(),
        )
        .unwrap();
        input.picture = again;
        input.force_keyframe = false;
        input.pts = 1;
        let second = encoder.push(&input).unwrap();
        let nals = crate::nal::access_unit(&second[0].data, 0).unwrap();
        assert!(
            nals.iter().any(|nal| nal.kind == 1),
            "the reconstructed picture is coded again"
        );
        let frames = decoder.push(&second[0]).unwrap();
        assert_eq!(frames[0].picture.y(), decoded[0].picture.y());
        assert_eq!(frames[0].picture.u(), decoded[0].picture.u());
        assert_eq!(frames[0].picture.v(), decoded[0].picture.v());
    }

    fn gray_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (vec![128; 64 * 64], vec![128; 32 * 32], vec![128; 32 * 32])
    }

    fn ffmpeg() -> Option<String> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .find(|path| {
            std::process::Command::new(path)
                .arg("-version")
                .output()
                .is_ok()
        })
        .map(str::to_string)
    }
}
