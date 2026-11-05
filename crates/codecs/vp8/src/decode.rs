//! Frame decoding (RFC 6386 Sections 9 to 19) and the decoder contract.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorRange, MatrixCoefficients, Packet, Plane, VideoFrame,
};
use cutlass_encode::PixelFormat;

use crate::bool_coder::BoolDecoder;
use crate::filter::{self, MbFilter, Planes};
use crate::header::{self, FrameHeader, Persist};
use crate::inter::{self, Filter, RefPlane};
use crate::modes::{
    self, ALTREF, Bounds, GOLDEN, INTRA, MbInfo, ModeContext, Mv, SPLITMV, SubInfo, ZEROMV,
};
use crate::predict::{self, B_PRED, CHROMA_BUF, CHROMA_STRIDE, LUMA_BUF, LUMA_STRIDE};
use crate::tokens::{self, Context, Dequant, MbCoeffs};
use crate::transform::{idct_add, idct_dc_add, iwht};

/// A reconstructed picture at its macroblock-aligned size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub mbw: usize,
    pub mbh: usize,
}

impl Frame {
    #[must_use]
    pub fn new(mbw: usize, mbh: usize) -> Self {
        Self {
            y: vec![0; mbw * mbh * 256],
            u: vec![0; mbw * mbh * 64],
            v: vec![0; mbw * mbh * 64],
            mbw,
            mbh,
        }
    }

    #[must_use]
    pub fn luma(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.y,
            stride: self.mbw * 16,
            width: self.mbw * 16,
            height: self.mbh * 16,
        }
    }

    #[must_use]
    pub fn chroma(&self, v: bool) -> RefPlane<'_> {
        RefPlane {
            data: if v { &self.v } else { &self.u },
            stride: self.mbw * 8,
            width: self.mbw * 8,
            height: self.mbh * 8,
        }
    }
}

/// The three reference pictures.
#[derive(Clone, Debug)]
pub struct Refs {
    pub last: Arc<Frame>,
    pub golden: Arc<Frame>,
    pub altref: Arc<Frame>,
}

impl Refs {
    fn get(&self, r: u8) -> &Frame {
        match r {
            GOLDEN => &self.golden,
            ALTREF => &self.altref,
            _ => &self.last,
        }
    }
}

/// A decoded picture and how it should be shown.
#[derive(Clone, Debug)]
pub struct Picture {
    pub frame: Arc<Frame>,
    pub width: u32,
    pub height: u32,
    pub hscale: u8,
    pub vscale: u8,
}

/// Decoder state between frames.
#[derive(Clone, Debug, Default)]
pub struct Decoder {
    persist: Persist,
    refs: Option<Refs>,
    width: u32,
    height: u32,
    hscale: u8,
    vscale: u8,
    segment_map: Vec<u8>,
    pool: Vec<Frame>,
}

/// Work for one frame, kept apart so a frame's results commit at once.
struct Work {
    mbs: Vec<MbInfo>,
    subs: Vec<SubInfo>,
    filters: Vec<MbFilter>,
    segments: Vec<u8>,
}

impl Decoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Display size of the stream so far.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Decodes one frame. Returns the picture when the frame is shown.
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Picture>, CodecError> {
        let parsed = header::parse(data, &self.persist)?;
        let h = parsed.header;
        let (mbw, mbh) = if h.tag.key {
            (
                (h.tag.width as usize).div_ceil(16),
                (h.tag.height as usize).div_ceil(16),
            )
        } else {
            let Some(refs) = &self.refs else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "vp8 inter frame before a key frame",
                });
            };
            (refs.last.mbw, refs.last.mbh)
        };
        let count = mbw * mbh;
        let mut segments = if self.segment_map.len() == count {
            self.segment_map.clone()
        } else {
            vec![0; count]
        };
        let mut frame = self
            .pool
            .iter()
            .position(|f| f.mbw == mbw && f.mbh == mbh)
            .map_or_else(|| Frame::new(mbw, mbh), |i| self.pool.swap_remove(i));
        let mut work = Work {
            mbs: vec![MbInfo::default(); count],
            subs: vec![SubInfo::default(); count],
            filters: vec![MbFilter::default(); count],
            segments: Vec::new(),
        };
        let refs = if h.tag.key { None } else { self.refs.as_ref() };
        decode_frame(
            &h,
            parsed.modes,
            &parsed.tokens,
            refs,
            &mut frame,
            &mut work,
            &mut segments,
        );
        if h.filter_level > 0 {
            let mut planes = Planes {
                y: &mut frame.y,
                u: &mut frame.u,
                v: &mut frame.v,
                mbw,
                mbh,
            };
            filter::filter_frame(&mut planes, &work.filters, h.filter_simple);
        }
        work.segments = segments;

        // Commit.
        self.persist = parsed.next;
        self.segment_map = work.segments;
        if h.tag.key {
            self.width = h.tag.width;
            self.height = h.tag.height;
            self.hscale = h.tag.hscale;
            self.vscale = h.tag.vscale;
        }
        let current = Arc::new(frame);
        let old = self.refs.take();
        let mut next = match &old {
            Some(r) if !h.tag.key => r.clone(),
            _ => Refs {
                last: Arc::clone(&current),
                golden: Arc::clone(&current),
                altref: Arc::clone(&current),
            },
        };
        if !h.tag.key {
            match h.copy_to_altref {
                1 => next.altref = Arc::clone(&next.last),
                2 => next.altref = Arc::clone(&next.golden),
                _ => {}
            }
            match h.copy_to_golden {
                1 => next.golden = Arc::clone(&next.last),
                2 => next.golden = Arc::clone(&next.altref),
                _ => {}
            }
            if h.refresh_golden {
                next.golden = Arc::clone(&current);
            }
            if h.refresh_altref {
                next.altref = Arc::clone(&current);
            }
            if h.refresh_last {
                next.last = Arc::clone(&current);
            }
        }
        self.refs = Some(next);
        if let Some(old) = old {
            for frame in [old.last, old.golden, old.altref] {
                if let Ok(frame) = Arc::try_unwrap(frame) {
                    if self.pool.len() < 4 {
                        self.pool.push(frame);
                    }
                }
            }
        }
        if !h.tag.show {
            return Ok(None);
        }
        Ok(Some(Picture {
            frame: current,
            width: self.width,
            height: self.height,
            hscale: self.hscale,
            vscale: self.vscale,
        }))
    }
}

/// Section 9.3 and 9.6: quantizer factors for each segment.
fn dequant_factors(h: &FrameHeader) -> [Dequant; 4] {
    let mut out = [Dequant::default(); 4];
    let base = i32::from(h.quant.y_ac);
    for (s, dq) in out.iter_mut().enumerate() {
        let q = if h.segmentation.enabled {
            let value = i32::from(h.segmentation.quant[s]);
            if h.segmentation.absolute {
                value
            } else {
                base + value
            }
        } else {
            base
        };
        *dq = Dequant::new(q, &h.quant);
    }
    out
}

/// Section 15.1 and 9.6: the filter parameters of one macroblock.
#[must_use]
pub fn mb_filter(h: &FrameHeader, mb: &MbInfo, inner: bool) -> MbFilter {
    let mut level = i32::from(h.filter_level);
    if h.segmentation.enabled {
        let value = i32::from(h.segmentation.filter[usize::from(mb.segment & 3)]);
        level = if h.segmentation.absolute {
            value
        } else {
            level + value
        };
        level = level.clamp(0, 63);
    }
    if h.deltas.enabled {
        level += i32::from(h.deltas.ref_frame[usize::from(mb.ref_frame & 3)]);
        let mode = if mb.ref_frame == INTRA {
            (mb.y_mode == B_PRED).then_some(0)
        } else if mb.y_mode == ZEROMV {
            Some(1)
        } else if mb.y_mode == SPLITMV {
            Some(3)
        } else {
            Some(2)
        };
        if let Some(i) = mode {
            level += i32::from(h.deltas.mode[i]);
        }
        level = level.clamp(0, 63);
    }
    let level = level.clamp(0, 63) as u8;
    MbFilter {
        level,
        interior: filter::interior_limit(level, h.sharpness),
        hev: filter::hev_threshold(level, h.tag.key),
        inner: inner || mb.y_mode == B_PRED || mb.y_mode == SPLITMV,
    }
}

fn read_segment(bits: &mut BoolDecoder<'_>, probs: &[u8; 3]) -> u8 {
    if bits.read(probs[0]) {
        2 + u8::from(bits.read(probs[2]))
    } else {
        u8::from(bits.read(probs[1]))
    }
}

fn decode_frame(
    h: &FrameHeader,
    mut bits: BoolDecoder<'_>,
    partitions: &[&[u8]],
    refs: Option<&Refs>,
    frame: &mut Frame,
    work: &mut Work,
    segments: &mut [u8],
) {
    let mbw = frame.mbw;
    let mbh = frame.mbh;
    let dq = dequant_factors(h);
    let mut token_bits: Vec<BoolDecoder<'_>> =
        partitions.iter().map(|p| BoolDecoder::new(p)).collect();
    let mut above_ctx: Vec<Context> = vec![[0; 9]; mbw];
    let mut coeffs = MbCoeffs::default();
    let sign_bias = [false, false, h.sign_bias_golden, h.sign_bias_altref];
    let filter = if h.tag.version == 0 {
        Filter::SixTap
    } else {
        Filter::Bilinear
    };
    let full_pixel = h.tag.version == 3;
    for mby in 0..mbh {
        let mut left_ctx: Context = [0; 9];
        let part = &mut token_bits[mby % partitions.len()];
        for mbx in 0..mbw {
            let i = mby * mbw + mbx;
            let mut mb = MbInfo::default();
            if h.segmentation.update_map {
                segments[i] = read_segment(&mut bits, &h.segmentation.tree_probs);
            } else if h.tag.key {
                segments[i] = 0;
            }
            mb.segment = segments[i];
            mb.skip = h.skip_enabled && bits.read(h.prob_skip);
            let mut sub = SubInfo::default();
            {
                let above = (mby > 0).then(|| &work.mbs[i - mbw]);
                let left = (mbx > 0).then(|| &work.mbs[i - 1]);
                let above_left = (mby > 0 && mbx > 0).then(|| &work.mbs[i - mbw - 1]);
                let ctx = ModeContext {
                    key: h.tag.key,
                    above,
                    left,
                    above_left,
                    above_sub: (mby > 0).then(|| &work.subs[i - mbw]),
                    left_sub: (mbx > 0).then(|| &work.subs[i - 1]),
                    bounds: Bounds::new(mbx, mby, mbw, mbh),
                    sign_bias,
                    y_probs: &h.probs.y_mode,
                    uv_probs: &h.probs.uv_mode,
                    mv_probs: &h.probs.mv,
                    prob_intra: h.prob_intra,
                    prob_last: h.prob_last,
                    prob_golden: h.prob_golden,
                };
                modes::read_modes(&mut bits, &ctx, &mut mb, &mut sub);
            }
            let has_y2 = mb.y_mode != B_PRED && mb.y_mode != SPLITMV;
            coeffs.clear();
            if mb.skip {
                tokens::skip_mb(&mut left_ctx, &mut above_ctx[mbx], has_y2);
            } else {
                let segment = if h.segmentation.enabled {
                    usize::from(mb.segment & 3)
                } else {
                    0
                };
                tokens::read_mb(
                    part,
                    &h.probs.coeff,
                    &mut left_ctx,
                    &mut above_ctx[mbx],
                    has_y2,
                    &dq[segment],
                    &mut coeffs,
                );
            }
            if has_y2 && coeffs.shape[24] != 0 {
                let dc = iwht(&coeffs.blocks[24]);
                for (b, &value) in dc.iter().enumerate() {
                    coeffs.blocks[b][0] = value;
                    if value != 0 && coeffs.shape[b] == 0 {
                        coeffs.shape[b] = 1;
                    }
                }
            }
            if mb.ref_frame == INTRA {
                reconstruct_intra(frame, mbx, mby, &mb, &sub, &coeffs);
            } else if let Some(refs) = refs {
                let reference = refs.get(mb.ref_frame);
                reconstruct_inter(
                    frame, reference, mbx, mby, &mb, &sub, &coeffs, filter, full_pixel,
                );
            }
            work.filters[i] = mb_filter(h, &mb, coeffs.any);
            work.mbs[i] = mb;
            work.subs[i] = sub;
        }
    }
}

/// Adds one block's residual at `at` in a buffer.
#[inline]
fn add_block(coeffs: &MbCoeffs, b: usize, buf: &mut [u8], at: usize, stride: usize) {
    match coeffs.shape[b] {
        0 => {}
        1 => idct_dc_add(coeffs.blocks[b][0], buf, at, stride),
        _ => idct_add(&coeffs.blocks[b], buf, at, stride),
    }
}

fn reconstruct_intra(
    frame: &mut Frame,
    mbx: usize,
    mby: usize,
    mb: &MbInfo,
    sub: &SubInfo,
    coeffs: &MbCoeffs,
) {
    let ys = frame.mbw * 16;
    let mut buf = [0u8; LUMA_BUF];
    predict::load_edges(&mut buf, &frame.y, ys, mbx, mby, 16, 4);
    if mb.y_mode == B_PRED {
        predict::copy_above_right(&mut buf);
        for b in 0..16 {
            let (r, c) = (b >> 2, b & 3);
            predict::predict_sub(&mut buf, r, c, sub.b_modes[b]);
            add_block(
                coeffs,
                b,
                &mut buf,
                (4 * r + 1) * LUMA_STRIDE + 4 * c + 1,
                LUMA_STRIDE,
            );
        }
    } else {
        predict::predict_block(&mut buf, 16, LUMA_STRIDE, mb.y_mode, mby > 0, mbx > 0);
        for b in 0..16 {
            let (r, c) = (b >> 2, b & 3);
            add_block(
                coeffs,
                b,
                &mut buf,
                (4 * r + 1) * LUMA_STRIDE + 4 * c + 1,
                LUMA_STRIDE,
            );
        }
    }
    predict::store(&buf, &mut frame.y, ys, mbx, mby, 16, LUMA_STRIDE);
    let cs = frame.mbw * 8;
    for (plane, first) in [(&mut frame.u, 16), (&mut frame.v, 20)] {
        let mut buf = [0u8; CHROMA_BUF];
        predict::load_edges(&mut buf, plane, cs, mbx, mby, 8, 0);
        predict::predict_block(&mut buf, 8, CHROMA_STRIDE, mb.uv_mode, mby > 0, mbx > 0);
        for k in 0..4 {
            let (r, c) = (k >> 1, k & 1);
            add_block(
                coeffs,
                first + k,
                &mut buf,
                (4 * r + 1) * CHROMA_STRIDE + 4 * c + 1,
                CHROMA_STRIDE,
            );
        }
        predict::store(&buf, plane, cs, mbx, mby, 8, CHROMA_STRIDE);
    }
}

/// Chroma vectors for each 4×4 chroma block, in eighth chroma pixels.
#[must_use]
pub fn chroma_mvs(mb: &MbInfo, sub: &SubInfo, full_pixel: bool) -> [Mv; 4] {
    let fix = |mv: Mv| {
        if full_pixel {
            Mv::new(mv.row & !7, mv.col & !7)
        } else {
            mv
        }
    };
    if mb.y_mode != SPLITMV {
        return [fix(mb.mv); 4];
    }
    let mut out = [Mv::ZERO; 4];
    for (k, slot) in out.iter_mut().enumerate() {
        let b = 8 * (k >> 1) + 2 * (k & 1);
        let (mut row, mut col) = (0, 0);
        for o in [0, 1, 4, 5] {
            row += sub.mvs[b + o].row;
            col += sub.mvs[b + o].col;
        }
        let avg = |s: i32| if s >= 0 { (s + 2) / 4 } else { (s - 2) / 4 };
        *slot = fix(Mv::new(avg(row), avg(col)));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_inter(
    frame: &mut Frame,
    reference: &Frame,
    mbx: usize,
    mby: usize,
    mb: &MbInfo,
    sub: &SubInfo,
    coeffs: &MbCoeffs,
    filter: Filter,
    full_pixel: bool,
) {
    let ys = frame.mbw * 16;
    let x0 = (mbx * 16) as i32;
    let y0 = (mby * 16) as i32;
    let base = mby * 16 * ys + mbx * 16;
    if mb.y_mode == SPLITMV {
        for b in 0..16 {
            let (r, c) = (b >> 2, b & 3);
            let mv = sub.mvs[b];
            let at = base + 4 * r * ys + 4 * c;
            inter::predict(
                reference.luma(),
                x0 + 4 * c as i32 + (mv.col >> 2),
                y0 + 4 * r as i32 + (mv.row >> 2),
                ((mv.col & 3) * 2) as usize,
                ((mv.row & 3) * 2) as usize,
                4,
                4,
                filter,
                &mut frame.y[at..],
                ys,
            );
        }
    } else {
        let mv = mb.mv;
        inter::predict(
            reference.luma(),
            x0 + (mv.col >> 2),
            y0 + (mv.row >> 2),
            ((mv.col & 3) * 2) as usize,
            ((mv.row & 3) * 2) as usize,
            16,
            16,
            filter,
            &mut frame.y[base..],
            ys,
        );
    }
    for b in 0..16 {
        let (r, c) = (b >> 2, b & 3);
        add_block(coeffs, b, &mut frame.y, base + 4 * r * ys + 4 * c, ys);
    }
    let cs = frame.mbw * 8;
    let cmv = chroma_mvs(mb, sub, full_pixel);
    let cx0 = (mbx * 8) as i32;
    let cy0 = (mby * 8) as i32;
    let cbase = mby * 8 * cs + mbx * 8;
    for (v, first) in [(false, 16), (true, 20)] {
        let src = reference.chroma(v);
        let plane = if v { &mut frame.v } else { &mut frame.u };
        if mb.y_mode == SPLITMV {
            for (k, mv) in cmv.iter().enumerate() {
                let (r, c) = (k >> 1, k & 1);
                let at = cbase + 4 * r * cs + 4 * c;
                inter::predict(
                    src,
                    cx0 + 4 * c as i32 + (mv.col >> 3),
                    cy0 + 4 * r as i32 + (mv.row >> 3),
                    (mv.col & 7) as usize,
                    (mv.row & 7) as usize,
                    4,
                    4,
                    filter,
                    &mut plane[at..],
                    cs,
                );
            }
        } else {
            let mv = cmv[0];
            inter::predict(
                src,
                cx0 + (mv.col >> 3),
                cy0 + (mv.row >> 3),
                (mv.col & 7) as usize,
                (mv.row & 7) as usize,
                8,
                8,
                filter,
                &mut plane[cbase..],
                cs,
            );
        }
        for k in 0..4 {
            let (r, c) = (k >> 1, k & 1);
            add_block(coeffs, first + k, plane, cbase + 4 * r * cs + 4 * c, cs);
        }
    }
}

/// VP8 decoder on the video codec contract.
///
/// `push` consumes one frame. A hidden frame (an alt-ref) returns no
/// picture. `drain` is empty. `reset` drops the reference pictures. A
/// failed `push` leaves the decoder as it was.
pub struct Vp8Decoder {
    inner: Decoder,
    cancel: Arc<AtomicBool>,
}

impl Vp8Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp8 {
            return Err(CodecError::Unsupported {
                what: "vp8 decoder",
            });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp8 pixel format",
            });
        }
        if params.width > header::MAX_EDGE || params.height > header::MAX_EDGE {
            return Err(CodecError::Limit {
                what: "vp8 frame edge",
            });
        }
        Ok(Self {
            inner: Decoder::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The horizontal and vertical upscaling the stream asks for
    /// (Section 9.2), each 0 to 3.
    #[must_use]
    pub fn scaling(&self) -> (u8, u8) {
        (self.inner.hscale, self.inner.vscale)
    }
}

impl VideoDecoder for Vp8Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if packet.data.is_empty() {
            // An empty packet is a dropped frame: nothing to show.
            return Ok(Vec::new());
        }
        match self.inner.decode(&packet.data)? {
            Some(picture) => Ok(vec![VideoOutput {
                frame: picture_frame(&picture)?,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
            }]),
            None => Ok(Vec::new()),
        }
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// Crops a picture to its display size.
pub fn picture_frame(picture: &Picture) -> Result<VideoFrame, CodecError> {
    let (w, h) = (picture.width as usize, picture.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let f = &picture.frame;
    let crop = |src: &[u8], stride: usize, width: usize, height: usize| {
        let mut out = Vec::with_capacity(width * height);
        for row in src.chunks_exact(stride).take(height) {
            out.extend_from_slice(&row[..width]);
        }
        out
    };
    let y = crop(&f.y, f.mbw * 16, w, h);
    let u = crop(&f.u, f.mbw * 8, cw, ch);
    let v = crop(&f.v, f.mbw * 8, cw, ch);
    let mut frame = VideoFrame::yuv420p(
        picture.width,
        picture.height,
        Plane::tight(picture.width, picture.height, y)?,
        Plane::tight(cw as u32, ch as u32, u)?,
        Plane::tight(cw as u32, ch as u32, v)?,
    )?;
    frame.range = ColorRange::Limited;
    frame.matrix = MatrixCoefficients::Bt601;
    Ok(frame)
}
