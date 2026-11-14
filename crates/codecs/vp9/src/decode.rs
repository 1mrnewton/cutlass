//! Stateful VP9 decoder.
//!
//! A failed frame leaves references and probabilities as they were. Reference
//! slots update only after reconstruction succeeds. `show_existing_frame`
//! displays a stored picture and does not refresh those slots.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, ColorRange, MatrixCoefficients, Packet,
    Plane, Transfer, VideoFrame,
};
use cutlass_encode::PixelFormat;

use cutlass_simd::vp9::frame::SharedFrame;

use crate::dec::frame::{Dims, MvRef, Pic};
use crate::dec::{ParseBufs, decode_frame};
use crate::dsp::pixel::Pixel;
use crate::geom::{self, SWITCHABLE, TX_MODE_SELECT};
use crate::header::{Header, Parsed, Persist, parse};
use crate::prob::{self, Counts, FrameContext};
use crate::recon::Picture;

/// A reference picture of either sample width.
#[derive(Clone)]
pub enum Held {
    Eight(Arc<Pic<u8>>),
    Deep(Arc<Pic<u16>>),
}

/// Picture buffers whose readers are gone, for later pictures.
type Spare<P> = Vec<(Dims, Arc<SharedFrame<P>>)>;

trait Store: Pixel {
    fn view(held: &Held) -> Option<&Pic<Self>>;
    fn keep(frame: Arc<Pic<Self>>) -> Held;
    fn spare(d: &mut Decoder) -> &mut Spare<Self>;
}

impl Store for u8 {
    fn view(held: &Held) -> Option<&Pic<Self>> {
        match held {
            Held::Eight(f) => Some(f),
            Held::Deep(_) => None,
        }
    }
    fn keep(frame: Arc<Pic<Self>>) -> Held {
        Held::Eight(frame)
    }
    fn spare(d: &mut Decoder) -> &mut Spare<Self> {
        &mut d.spare8
    }
}

impl Store for u16 {
    fn view(held: &Held) -> Option<&Pic<Self>> {
        match held {
            Held::Deep(f) => Some(f),
            Held::Eight(_) => None,
        }
    }
    fn keep(frame: Arc<Pic<Self>>) -> Held {
        Held::Deep(frame)
    }
    fn spare(d: &mut Decoder) -> &mut Spare<Self> {
        &mut d.spare16
    }
}

pub struct Decoder {
    persist: Persist,
    refs: [Option<Held>; 8],
    prev: Option<Arc<Vec<MvRef>>>,
    last_seg: Vec<u8>,
    spare8: Spare<u8>,
    spare16: Spare<u16>,
    bufs: ParseBufs,
    /// Frames whose segmentation map was coded relative to the previous map.
    #[cfg(test)]
    temporal_frames: u32,
}

impl Clone for Decoder {
    fn clone(&self) -> Self {
        Self {
            persist: self.persist.clone(),
            refs: self.refs.clone(),
            prev: self.prev.clone(),
            last_seg: self.last_seg.clone(),
            spare8: Spare::default(),
            spare16: Spare::default(),
            bufs: ParseBufs::default(),
            #[cfg(test)]
            temporal_frames: self.temporal_frames,
        }
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            persist: Persist::new(),
            refs: [const { None }; 8],
            prev: None,
            last_seg: Vec::new(),
            spare8: Spare::default(),
            spare16: Spare::default(),
            bufs: ParseBufs::default(),
            #[cfg(test)]
            temporal_frames: 0,
        }
    }

    /// How many decoded frames coded the segmentation map temporally.
    #[cfg(test)]
    pub fn temporal_frames(&self) -> u32 {
        self.temporal_frames
    }

    /// Decodes one frame and returns the picture to show, if any, as
    /// tightly packed planes.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<Picture>, CodecError> {
        Ok(self.decode_held(packet)?.map(|h| picture_of(&h)))
    }

    /// Decodes one frame and returns the picture to show, if any.
    pub fn decode_held(&mut self, packet: &[u8]) -> Result<Option<Held>, CodecError> {
        if packet.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let parsed = parse(&self.persist, packet)?;
        #[cfg(test)]
        if parsed.header.seg.temporal {
            self.temporal_frames = self.temporal_frames.saturating_add(1);
        }
        if parsed.header.show_existing {
            let slot = usize::from(parsed.header.frame_to_show);
            let held = self
                .refs
                .get(slot)
                .and_then(|s| s.clone())
                .ok_or(CodecError::Malformed {
                    at_bit: 0,
                    what: "shown reference",
                })?;
            self.persist.last_width = parsed.header.width;
            self.persist.last_height = parsed.header.height;
            return Ok(Some(held));
        }
        if parsed.header.bit_depth > 8 {
            self.frame::<u16>(&parsed)
        } else {
            self.frame::<u8>(&parsed)
        }
    }

    pub fn reset(&mut self) {
        self.persist = Persist::new();
        self.refs = [const { None }; 8];
        self.prev = None;
        self.last_seg.clear();
        #[cfg(test)]
        {
            self.temporal_frames = 0;
        }
    }

    fn frame<P: Store>(&mut self, parsed: &Parsed<'_>) -> Result<Option<Held>, CodecError> {
        let header = &parsed.header;
        if !header.is_intra() && header.use_prev_mvs && self.prev.is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "previous motion",
            });
        }
        let last_seg = seg_source(&self.last_seg, header);
        let prev = self.prev.clone();
        // Pictures that nothing holds any more come back as buffers.
        let refs_now = self.refs.clone();
        let bufs = std::mem::take(&mut self.bufs);
        let (pic, built) = {
            let view = |i: usize| -> Option<&Pic<P>> {
                let slot = usize::from(header.ref_index[i]);
                refs_now.get(slot)?.as_ref().and_then(P::view)
            };
            let refs = [view(0), view(1), view(2)];
            let spare = P::spare(self);
            decode_frame(parsed, refs, prev.as_deref().map(|v| v.as_slice()), &last_seg, bufs, spare)?
        };
        drop(refs_now);
        let after_key = self.persist.key_frame;
        let fc = adapted(parsed, &built.counts, after_key);
        self.bufs = built.spare;
        self.bufs.tiles = built.tiles;
        self.bufs.masks = built.masks;
        self.bufs.grid = built.grid;
        let persist = &mut self.persist;
        persist.width = header.width;
        persist.height = header.height;
        persist.last_width = header.width;
        persist.last_height = header.height;
        persist.need_resync = header.need_resync;
        persist.key_frame = header.key_frame;
        persist.intra_only = header.intra_only;
        persist.last_show_frame = header.show_frame;
        persist.sign_bias = header.sign_bias;
        persist.ss_x = header.ss_x;
        persist.ss_y = header.ss_y;
        persist.bit_depth = header.bit_depth;
        persist.color_space = header.color_space;
        persist.color_range = header.color_range;
        persist.seg = header.seg.clone();
        persist.ref_deltas = header.ref_deltas;
        persist.mode_deltas = header.mode_deltas;
        persist.contexts = parsed.contexts.clone();
        if header.refresh_context {
            persist.contexts[usize::from(header.context_idx)] = fc;
        }
        // The previous motion goes back to the spare buffers if no one
        // else holds it.
        if let Some(old) = self.prev.take()
            && let Ok(v) = Arc::try_unwrap(old)
        {
            self.bufs.mvs = v;
        }
        self.prev = Some(Arc::new(built.mvs));
        let cells = geom::mi_count(header.width).saturating_mul(geom::mi_count(header.height));
        if header.seg.enabled {
            self.last_seg = built.segmap;
        } else if header.is_intra() || header.error_resilient {
            self.last_seg = vec![0u8; cells];
        }
        let shared = Arc::new(pic);
        for slot in 0..8 {
            if header.refresh_flags & (1u8 << slot) == 0 {
                continue;
            }
            let old = self.refs[slot].replace(P::keep(Arc::clone(&shared)));
            self.persist.ref_ok[slot] = true;
            self.persist.ref_w[slot] = header.width;
            self.persist.ref_h[slot] = header.height;
            if let Some(old) = old {
                self.recycle(old);
            }
        }
        if header.show_frame {
            Ok(Some(P::keep(shared)))
        } else {
            Ok(None)
        }
    }

    /// Takes back a picture's buffers when this was its last holder.
    pub fn recycle(&mut self, held: Held) {
        match held {
            Held::Eight(f) => {
                if let Ok(f) = Arc::try_unwrap(f)
                    && self.spare8.len() < 4
                {
                    self.spare8.push((f.dims, f.buf));
                }
            }
            Held::Deep(f) => {
                if let Ok(f) = Arc::try_unwrap(f)
                    && self.spare16.len() < 4
                {
                    self.spare16.push((f.dims, f.buf));
                }
            }
        }
    }
}

fn seg_source(last: &[u8], header: &Header) -> Vec<u8> {
    let cells = geom::mi_count(header.width).saturating_mul(geom::mi_count(header.height));
    if header.is_intra() || header.error_resilient || last.len() != cells {
        vec![0u8; cells]
    } else {
        last.to_vec()
    }
}

fn adapted(parsed: &Parsed<'_>, counts: &Counts, after_key: bool) -> FrameContext {
    let header = &parsed.header;
    let mut fc = parsed.fc.clone();
    if header.error_resilient || header.frame_parallel {
        return fc;
    }
    let pre = &parsed.contexts[usize::from(header.context_idx)];
    let key = header.is_intra();
    for tx in 0..4 {
        prob::adapt_coef(&mut fc, pre, counts, tx, key, after_key);
    }
    if !key {
        prob::adapt_modes(
            &mut fc,
            pre,
            counts,
            header.tx_mode == TX_MODE_SELECT,
            header.interp == SWITCHABLE,
        );
        prob::adapt_mv(&mut fc, pre, counts, header.allow_hp);
    }
    fc
}

/// The visible samples of plane `p`, rows packed, little-endian above 8
/// bits.
fn packed<P: Pixel>(f: &Pic<P>, p: usize) -> Vec<u8> {
    let (w, h) = (f.dims.w[p], f.dims.h[p]);
    f.buf.wait(usize::MAX);
    let (data, stride) = f.buf.plane(p);
    let mut out = Vec::with_capacity(w * h * if P::HIGH { 2 } else { 1 });
    for r in 0..h {
        let row = &data[r * stride..r * stride + w];
        if P::HIGH {
            for s in row {
                out.extend_from_slice(&(s.get() as u16).to_le_bytes());
            }
        } else {
            out.extend(row.iter().map(|s| s.get() as u8));
        }
    }
    out
}

fn picture_of(held: &Held) -> Picture {
    fn of<P: Pixel>(f: &Pic<P>) -> Picture {
        Picture {
            y: packed(f, 0),
            u: packed(f, 1),
            v: packed(f, 2),
            width: f.dims.width as u32,
            height: f.dims.height as u32,
            ss_x: f.dims.ss_x as u32,
            ss_y: f.dims.ss_y as u32,
            color_space: f.color_space,
            color_range: f.color_range,
            bit_depth: f.dims.bit_depth,
            render_width: f.render_width,
            render_height: f.render_height,
        }
    }
    match held {
        Held::Eight(f) => of(f),
        Held::Deep(f) => of(f),
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// VP9 decoder on the video codec contract.
///
/// `push` consumes one access unit. A hidden frame returns no picture.
/// `drain` is empty. `reset` drops reference pictures. A failed `push`
/// leaves those pictures as they were.
pub struct Vp9Decoder {
    inner: Decoder,
    cancel: Arc<AtomicBool>,
}

impl Vp9Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported {
                what: "vp9 decoder",
            });
        }
        if !matches!(
            params.pixel,
            PixelFormat::Yuv420p
                | PixelFormat::Yuv420p10
                | PixelFormat::Yuv420p12
                | PixelFormat::Yuv422p
                | PixelFormat::Yuv422p10
                | PixelFormat::Yuv422p12
                | PixelFormat::Yuv440p
                | PixelFormat::Yuv440p10
                | PixelFormat::Yuv440p12
                | PixelFormat::Yuv444p
                | PixelFormat::Yuv444p10
                | PixelFormat::Yuv444p12
                | PixelFormat::Gbrp
                | PixelFormat::Gbrp10
                | PixelFormat::Gbrp12
        ) {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        }
        check_open_size(params.width, params.height)?;
        Ok(Self {
            inner: Decoder::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl VideoDecoder for Vp9Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let parts = split_superframe(&packet.data)?;
        let mut checkpoint = (parts.len() > 1).then(|| self.inner.clone());
        let mut outputs = Vec::new();
        for part in parts {
            match self.inner.decode_held(part) {
                Ok(Some(held)) => outputs.push(VideoOutput {
                    frame: held_frame(&held)?,
                    pts: packet.pts,
                    duration: packet.duration,
                    timescale: packet.timescale,
                }),
                Ok(None) => {}
                Err(err) => {
                    if let Some(saved) = checkpoint.take() {
                        self.inner = saved;
                    }
                    return Err(err);
                }
            }
        }
        Ok(outputs)
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

#[cfg(test)]
impl Vp9Decoder {
    pub(crate) fn temporal_frames(&self) -> u32 {
        self.inner.temporal_frames()
    }
}

fn held_frame(held: &Held) -> Result<VideoFrame, CodecError> {
    match held {
        Held::Eight(f) => video_frame(f),
        Held::Deep(f) => video_frame(f),
    }
}

fn video_frame<P: Pixel>(f: &Pic<P>) -> Result<VideoFrame, CodecError> {
    let format = match (f.dims.bit_depth, f.dims.ss_x, f.dims.ss_y, f.color_space) {
        (10, 1, 1, _) => PixelFormat::Yuv420p10,
        (12, 1, 1, _) => PixelFormat::Yuv420p12,
        (10, 1, 0, _) => PixelFormat::Yuv422p10,
        (12, 1, 0, _) => PixelFormat::Yuv422p12,
        (10, 0, 1, _) => PixelFormat::Yuv440p10,
        (12, 0, 1, _) => PixelFormat::Yuv440p12,
        (10, 0, 0, 7) => PixelFormat::Gbrp10,
        (12, 0, 0, 7) => PixelFormat::Gbrp12,
        (10, 0, 0, _) => PixelFormat::Yuv444p10,
        (12, 0, 0, _) => PixelFormat::Yuv444p12,
        (8, 1, 1, _) => PixelFormat::Yuv420p,
        (8, 1, 0, _) => PixelFormat::Yuv422p,
        (8, 0, 1, _) => PixelFormat::Yuv440p,
        (8, 0, 0, 7) => PixelFormat::Gbrp,
        (8, 0, 0, _) => PixelFormat::Yuv444p,
        _ => {
            return Err(CodecError::Unsupported { what: "vp9 chroma" });
        }
    };
    let bytes = if P::HIGH { 2 } else { 1 };
    let plane = |p: usize| {
        let samples = packed(f, p);
        let (w, h) = (f.dims.w[p] as u32, f.dims.h[p] as u32);
        if bytes == 1 {
            Plane::tight(w, h, samples)
        } else {
            Plane::bytes(w, h, bytes, samples)
        }
    };
    let mut frame = VideoFrame::planar(
        format,
        f.dims.width as u32,
        f.dims.height as u32,
        plane(0)?,
        plane(1)?,
        plane(2)?,
    )?;
    let (range, primaries, transfer, matrix) = color_of(f.color_space, f.color_range);
    frame.range = range;
    frame.primaries = primaries;
    frame.transfer = transfer;
    frame.matrix = matrix;
    frame.render_width = f.render_width;
    frame.render_height = f.render_height;
    Ok(frame)
}

fn color_of(space: u8, range: u8) -> (ColorRange, ColorPrimaries, Transfer, MatrixCoefficients) {
    let range = if range == 0 {
        ColorRange::Studio
    } else {
        ColorRange::Full
    };
    let (primaries, transfer, matrix) = match space {
        1 | 3 => (
            ColorPrimaries::Bt601,
            Transfer::Bt601,
            MatrixCoefficients::Bt601,
        ),
        2 => (
            ColorPrimaries::Bt709,
            Transfer::Bt709,
            MatrixCoefficients::Bt709,
        ),
        5 => (
            ColorPrimaries::Bt2020,
            Transfer::Bt2020,
            MatrixCoefficients::Bt2020,
        ),
        7 => (
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            MatrixCoefficients::Rgb,
        ),
        _ => (
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            MatrixCoefficients::Unspecified,
        ),
    };
    (range, primaries, transfer, matrix)
}

fn split_superframe(data: &[u8]) -> Result<Vec<&[u8]>, CodecError> {
    let Some(marker) = data.last().copied() else {
        return Ok(vec![data]);
    };
    if marker & 0xe0 != 0xc0 {
        return Ok(vec![data]);
    }
    let frames = usize::from(marker & 0x7) + 1;
    let mag = usize::from((marker >> 3) & 0x3) + 1;
    let index = 2 + mag * frames;
    if data.len() < index || data[data.len() - index] != marker {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "superframe",
        });
    }
    let base = data.len() - index + 1;
    let payload = data.len() - index;
    let mut offset = 0usize;
    let mut parts = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut size = 0usize;
        for byte in 0..mag {
            let shift = u32::try_from(8 * byte).unwrap_or(0);
            size |= usize::from(data[base + frame * mag + byte]) << shift;
        }
        let end = offset
            .checked_add(size)
            .ok_or(CodecError::Limit { what: "superframe" })?;
        if end > payload {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "superframe",
            });
        }
        parts.push(&data[offset..end]);
        offset = end;
    }
    if offset != payload {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "superframe",
        });
    }
    Ok(parts)
}

fn check_open_size(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if !geom::level_picture(width, height) {
        return Err(CodecError::Limit {
            what: "vp9 frame edge",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use cutlass_encode::video::VideoEncoder;

    #[test]
    fn normal_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_normal32_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_normal32_1.bin").as_slice(),
            include_bytes!("../fixtures/vp9_normal32_2.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_normal32.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index} {err}"))
                .unwrap_or_else(|| panic!("frame {index} hidden"));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn altref_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_altref64_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_1.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_2.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_3.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_4.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_5.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_6.bin").as_slice(),
            include_bytes!("../fixtures/vp9_altref64_7.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_altref64.yuv");
        decode_all(&frames, yuv);
    }

    fn decode_all(frames: &[&[u8]], yuv: &[u8]) {
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index} {err}"))
                .unwrap_or_else(|| panic!("frame {index} hidden"));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn error_resilient_inter_matches_ffvp9() {
        let frames = [
            include_bytes!("../fixtures/vp9_inter32_0.bin").as_slice(),
            include_bytes!("../fixtures/vp9_inter32_1.bin").as_slice(),
        ];
        let yuv = include_bytes!("../fixtures/vp9_inter32.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for frame in frames {
            let picture = decoder.decode(frame).expect("frame").expect("shown");
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn failed_inter_does_not_drop_references() {
        let key = include_bytes!("../fixtures/vp9_inter32_0.bin");
        let mut decoder = Decoder::new();
        let shown = decoder.decode(key).expect("key").expect("shown key");
        let broken = [0x87, 0x00];
        let err = decoder.decode(&broken);
        assert!(err.is_err());
        let inter = include_bytes!("../fixtures/vp9_inter32_1.bin");
        let picture = decoder
            .decode(inter)
            .expect("refs survived")
            .expect("shown");
        let mut got = shown.y.clone();
        got.extend_from_slice(&shown.u);
        got.extend_from_slice(&shown.v);
        got.extend_from_slice(&picture.y);
        got.extend_from_slice(&picture.u);
        got.extend_from_slice(&picture.v);
        let yuv = include_bytes!("../fixtures/vp9_inter32.yuv");
        assert_eq!(got.as_slice(), yuv.as_slice());
    }

    #[test]
    fn decoder_fuzz_does_not_panic() {
        let secs = std::env::var("VP9_FUZZ_SECS")
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .unwrap_or(2);
        let seed = std::env::var("VP9_FUZZ_SEED")
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .unwrap_or(0x5650_3900_d4af);
        let corpus = fuzz_corpus();
        let mut rng = FuzzRng { state: seed };
        let started = std::time::Instant::now();
        let mut inputs = 0u64;
        let mut decoded = 0u64;
        let limit = std::time::Duration::from_secs(secs);
        while started.elapsed() < limit {
            let bytes = mutate(&mut rng, &corpus);
            let mut decoder = Decoder::new();
            if let Ok(Some(picture)) = decoder.decode(&bytes) {
                assert!(sane(&picture), "seed {seed} input {inputs}");
                decoded += 1;
            }
            if inputs.is_multiple_of(32) {
                let mut decoder = Decoder::new();
                let key = &corpus[0];
                decoder.decode(key).expect("corpus key").expect("shown");
                let _ = decoder.decode(&bytes);
                let again = decoder.decode(key).expect("key after noise");
                assert!(again.is_some(), "seed {seed} dropped the key");
            }
            inputs += 1;
        }
        eprintln!(
            "vp9 fuzz seed {seed:#x} seconds {secs} inputs {inputs} pictures {decoded} findings 0"
        );
    }

    fn fuzz_corpus() -> Vec<Vec<u8>> {
        let mut encoder = crate::encode::Vp9Encoder::with_quantizer(32, 16, 32).expect("encoder");
        let mut frames = Vec::new();
        for index in 0..4 {
            let frame = fuzz_picture(index);
            let packets = encoder.push(&frame).expect("corpus");
            frames.push(packets[0].data.clone());
        }
        frames.push(vec![0xa0]);
        frames.push(vec![0x00]);
        frames.push(vec![0xc0]);
        frames
    }

    fn fuzz_picture(index: u32) -> cutlass_encode::video::VideoInput {
        let width = 32u32;
        let height = 16u32;
        let mut y = Vec::new();
        let mut u = Vec::new();
        let mut v = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let value = row
                    .wrapping_mul(3)
                    .wrapping_add(col)
                    .wrapping_add(index * 5);
                y.push(u8::try_from(value % 220).unwrap_or(0));
            }
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                u.push(u8::try_from((row + col + index) % 80 + 40).unwrap_or(40));
                v.push(u8::try_from((col * 2 + index) % 70 + 60).unwrap_or(60));
            }
        }
        let frame = cutlass_encode::VideoFrame::yuv420p(
            width,
            height,
            cutlass_encode::Plane::tight(width, height, y).expect("y"),
            cutlass_encode::Plane::tight(width / 2, height / 2, u).expect("u"),
            cutlass_encode::Plane::tight(width / 2, height / 2, v).expect("v"),
        )
        .expect("frame");
        cutlass_encode::video::VideoInput {
            frame,
            pts: i64::from(index),
            duration: 1,
            timescale: 30,
        }
    }

    fn mutate(rng: &mut FuzzRng, corpus: &[Vec<u8>]) -> Vec<u8> {
        let pick = rng.below(corpus.len() + 1);
        if pick == corpus.len() {
            let len = rng.below(2048);
            return (0..len).map(|_| rng.byte()).collect();
        }
        let mut bytes = corpus[pick].clone();
        if bytes.is_empty() {
            return bytes;
        }
        match rng.below(4) {
            0 => {
                let at = rng.below(bytes.len());
                bytes[at] ^= rng.byte() | 1;
            }
            1 => {
                let at = rng.below(bytes.len());
                bytes.truncate(at);
            }
            2 => {
                let at = rng.below(bytes.len() + 1);
                bytes.insert(at, rng.byte());
            }
            _ => {
                let extra: Vec<u8> = (0..rng.below(16)).map(|_| rng.byte()).collect();
                bytes.extend(extra);
            }
        }
        bytes
    }

    fn sane(picture: &crate::recon::Picture) -> bool {
        let width = usize::try_from(picture.width).unwrap_or(usize::MAX);
        let height = usize::try_from(picture.height).unwrap_or(usize::MAX);
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        picture.width > 0
            && picture.height > 0
            && crate::geom::level_picture(picture.width, picture.height)
            && picture.y.len() == width.saturating_mul(height)
            && picture.u.len() == chroma_w.saturating_mul(chroma_h)
            && picture.v.len() == picture.u.len()
    }

    struct FuzzRng {
        state: u64,
    }

    impl FuzzRng {
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.state
        }

        fn below(&mut self, n: usize) -> usize {
            if n == 0 {
                return 0;
            }
            usize::try_from(self.next() % u64::try_from(n).unwrap_or(1)).unwrap_or(0)
        }

        fn byte(&mut self) -> u8 {
            u8::try_from(self.next() & 0xff).unwrap_or(0)
        }
    }

    #[test]
    fn scaled_references_match_libvpx() {
        let frames: [&[u8]; 3] = [
            include_bytes!("../fixtures/vp9_scale_0.bin"),
            include_bytes!("../fixtures/vp9_scale_1.bin"),
            include_bytes!("../fixtures/vp9_scale_2.bin"),
        ];
        let yuv = include_bytes!("../fixtures/vp9_scale.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        let mut sizes = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index}: {err}"))
                .expect("shown");
            sizes.push((picture.width, picture.height));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(sizes, [(64, 64), (32, 32), (52, 52)]);
        assert_eq!(got, yuv);
    }

    #[test]
    fn odd_size_loop_filter_matches_libvpx() {
        let frames: [&[u8]; 2] = [
            include_bytes!("../fixtures/vp9_odd_0.bin"),
            include_bytes!("../fixtures/vp9_odd_1.bin"),
        ];
        let yuv = include_bytes!("../fixtures/vp9_odd.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|err| panic!("frame {index}: {err}"))
                .expect("shown");
            assert_eq!((picture.width, picture.height), (52, 52));
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got, yuv);
    }

    #[test]
    fn segmentation_matches_libvpx() {
        let frames: [&[u8]; 8] = [
            include_bytes!("../fixtures/vp9_aq_0.bin"),
            include_bytes!("../fixtures/vp9_aq_1.bin"),
            include_bytes!("../fixtures/vp9_aq_2.bin"),
            include_bytes!("../fixtures/vp9_aq_3.bin"),
            include_bytes!("../fixtures/vp9_aq_4.bin"),
            include_bytes!("../fixtures/vp9_aq_5.bin"),
            include_bytes!("../fixtures/vp9_aq_6.bin"),
            include_bytes!("../fixtures/vp9_aq_7.bin"),
        ];
        let yuv = include_bytes!("../fixtures/vp9_aq.yuv");
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for frame in frames {
            let picture = decoder.decode(frame).expect("frame").expect("shown");
            got.extend_from_slice(&picture.y);
            got.extend_from_slice(&picture.u);
            got.extend_from_slice(&picture.v);
        }
        assert_eq!(got.len(), yuv.len());
        let mismatch = got.iter().zip(yuv.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "first mismatch {mismatch:?}");
    }

    #[test]
    fn reset_drops_references() {
        let key = include_bytes!("../fixtures/vp9_inter32_0.bin");
        let inter = include_bytes!("../fixtures/vp9_inter32_1.bin");
        let mut decoder = Decoder::new();
        decoder.decode(key).expect("key");
        decoder.reset();
        let err = decoder.decode(inter);
        assert!(err.is_err());
    }
}
