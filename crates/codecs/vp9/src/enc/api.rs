//! The encoder on the codec traits: [`Vp9Encoder`] takes 4:2:0 frames on
//! the video codec contract, and [`EngineEncoder`] is the encoder export
//! opens through the engine's factory, with its `vpcC` configuration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{CodecError, CodecId, CodecParams, EncodeFactory, Packet, PixelFormat, Plane, VideoFrame};

use super::{FrameEncoder, FrameParams};

/// The quantizer index a factory encoder with no bitrate keeps.
pub const DEFAULT_Q: u8 = 60;

/// How the encoder codes.
#[derive(Clone, Copy, Debug)]
pub struct Vp9Options {
    /// Bits a second to aim for; 0 codes every frame at `qindex`.
    pub bitrate: u32,
    /// The quantizer index (0: lossless) without a bitrate, and the first
    /// one with it.
    pub qindex: u8,
    /// The most frames from one key frame to the next.
    pub keyint: u32,
    /// Search effort, 0 (most) to 9 (least).
    pub speed: u8,
    /// The fewest and most quantizer indices the rate control may use.
    pub min_q: u8,
    pub max_q: u8,
    /// Threads (tile columns) to code with; 0: one per core.
    pub threads: usize,
}

impl Vp9Options {
    fn threads(&self) -> usize {
        match self.threads {
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
            n => n,
        }
    }
}

impl Default for Vp9Options {
    fn default() -> Self {
        Self {
            bitrate: 0,
            qindex: DEFAULT_Q,
            keyint: 240,
            speed: 5,
            min_q: 4,
            max_q: 236,
            threads: 0,
        }
    }
}

/// VP9 profile 0 (8-bit 4:2:0) on the video codec contract.
pub struct Vp9Encoder {
    width: u32,
    height: u32,
    opts: Vp9Options,
    enc: FrameEncoder,
    since_key: u32,
    force_key: bool,
    rc: Rate,
    cancel: Arc<AtomicBool>,
}

impl Vp9Encoder {
    /// Every frame at quantizer index `qindex` (0: lossless).
    pub fn with_quantizer(width: u32, height: u32, qindex: u8) -> Result<Self, CodecError> {
        Self::with_options(width, height, Vp9Options { qindex, ..Vp9Options::default() })
    }

    pub fn with_options(width: u32, height: u32, opts: Vp9Options) -> Result<Self, CodecError> {
        check_size(width, height)?;
        Ok(Self {
            width,
            height,
            opts,
            enc: FrameEncoder::new(width, height, opts.threads()),
            since_key: 0,
            force_key: true,
            rc: Rate::new(opts),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Bitrate 0 keeps the default quantizer; a positive bitrate aims at
    /// that rate.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported { what: "vp9 encoder" });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported { what: "vp9 pixel format" });
        }
        let opts = Vp9Options {
            bitrate: params.bitrate,
            ..Vp9Options::default()
        };
        Self::with_options(params.width, params.height, opts)
    }

    /// The quantizer index of the next frame.
    pub fn qindex(&self) -> u8 {
        self.rc.last_q
    }

    /// Makes the next frame a key frame.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    /// Codes one 4:2:0 picture: `planes` are Y, U, V with their strides.
    pub fn encode_planes(&mut self, planes: [(&[u8], usize); 3], seconds: f64) -> Result<(Vec<u8>, bool), CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_planes(self.width, self.height, &planes)?;
        let key = self.force_key || self.since_key >= self.opts.keyint;
        // A golden frame every `GF_INTERVAL` frames, coded finer: the
        // frames after it predict from it.
        let golden = !key && self.since_key % GF_INTERVAL == 0;
        let kind = if key { 0 } else if golden { 1 } else { 2 };
        self.enc.load(planes);
        let qindex = self.rc.qindex(kind);
        let coded = self.enc.encode(FrameParams {
            key,
            golden,
            qindex,
            speed: self.opts.speed,
        })?;
        let kind = if coded.key { 0 } else { kind };
        self.rc.account(coded.data.len(), seconds, kind, qindex);
        self.force_key = false;
        self.since_key = if coded.key { 1 } else { self.since_key + 1 };
        Ok((coded.data, coded.key))
    }

    /// The decoder configuration (`vpcC`, version 1) for the given colour
    /// codes: profile 0, level unset, 8-bit 4:2:0, limited range.
    pub fn vpcc_with_colour(&self, [primaries, transfer, matrix]: [u8; 3]) -> Vec<u8> {
        vec![1, 0, 0, 0, 0, 0, (8 << 4) | (1 << 1), primaries, transfer, matrix, 0, 0]
    }
}

impl VideoEncoder for Vp9Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let f = &input.frame;
        if f.format != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported { what: "vp9 pixel format" });
        }
        if f.width != self.width || f.height != self.height {
            return Err(CodecError::InvalidApi { what: "picture size" });
        }
        if self.opts.bitrate > 0 && (input.duration <= 0 || input.timescale == 0) {
            return Err(CodecError::InvalidApi { what: "frame clock" });
        }
        let seconds = if input.timescale == 0 {
            0.0
        } else {
            input.duration as f64 / f64::from(input.timescale)
        };
        let planes = [(&f.y.samples[..], f.y.stride), (&f.u.samples[..], f.u.stride), (&f.v.samples[..], f.v.stride)];
        let (data, key) = self.encode_planes(planes, seconds)?;
        Ok(vec![Packet::video(data, input.pts, input.duration, input.timescale, key)])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.enc = FrameEncoder::new(self.width, self.height, self.opts.threads());
        self.force_key = true;
        self.since_key = 0;
        self.rc = Rate::new(self.opts);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// Registers the encoder export opens.
pub fn register_engine_encoders(factory: &mut EncodeFactory) {
    factory.register_video(CodecId::Vp9, open_engine_encoder);
}

fn open_engine_encoder(params: &CodecParams) -> Result<Box<dyn cutlass_encode::VideoEncoder>, CodecError> {
    Ok(Box::new(EngineEncoder::open(params)?))
}

/// The encoder export opens: 4:2:0 pictures converted from its RGBA
/// frames, BT.709.
pub struct EngineEncoder {
    inner: Vp9Encoder,
    extradata: Vec<u8>,
}

impl EngineEncoder {
    /// Opens with the export's parameters, which may name RGBA: export
    /// converts each frame to 4:2:0 before it reaches this encoder.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if !matches!(params.pixel, PixelFormat::Yuv420p | PixelFormat::Rgba8) {
            return Err(CodecError::Unsupported { what: "vp9 pixel format" });
        }
        let mut planar = params.clone();
        planar.pixel = PixelFormat::Yuv420p;
        let inner = Vp9Encoder::open(&planar)?;
        let extradata = inner.vpcc_with_colour([1, 1, 1]);
        Ok(Self { inner, extradata })
    }
}

impl cutlass_encode::VideoEncoder for EngineEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, input: &cutlass_encode::VideoInput) -> Result<Vec<Packet>, CodecError> {
        let picture = &input.picture;
        let (width, height) = (picture.width(), picture.height());
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let plane = |w: u32, h: u32, bytes: &[u8]| {
            Plane::tight(w, h, bytes.to_vec()).map_err(|_| CodecError::InvalidApi { what: "picture size" })
        };
        let frame = VideoFrame::yuv420p(
            width,
            height,
            plane(width, height, picture.y())?,
            plane(cw, ch, picture.u())?,
            plane(cw, ch, picture.v())?,
        )
        .map_err(|_| CodecError::InvalidApi { what: "picture size" })?;
        if input.force_keyframe {
            self.inner.force_keyframe();
        }
        self.inner.push(&VideoInput {
            frame,
            pts: input.pts,
            duration: input.duration,
            timescale: input.timescale,
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        VideoEncoder::drain(&mut self.inner)
    }

    fn reset(&mut self) {
        VideoEncoder::reset(&mut self.inner);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        VideoEncoder::cancel_flag(&self.inner)
    }

    fn extradata(&self) -> &[u8] {
        &self.extradata
    }
}

fn check_size(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi { what: "empty picture" });
    }
    if width > 65536 || height > 65536 {
        return Err(CodecError::Limit { what: "vp9 frame edge" });
    }
    Ok(())
}

fn check_planes(width: u32, height: u32, planes: &[(&[u8], usize); 3]) -> Result<(), CodecError> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    for (p, (data, stride)) in planes.iter().enumerate() {
        let (pw, ph) = if p == 0 { (w, h) } else { (cw, ch) };
        if *stride < pw || data.len() < (ph - 1) * stride + pw {
            return Err(CodecError::InvalidApi { what: "picture planes" });
        }
    }
    Ok(())
}

/// Frames from one golden frame to the next.
const GF_INTERVAL: u32 = 12;

/// The quantizer step of each frame type (key, golden, inter) over the
/// inter frames'.
const STEP_SCALE: [f64; 3] = [0.45, 0.7, 1.0];

/// The share of a frame's budget each frame type takes.
const SHARE: [f64; 3] = [6.0, 2.5, (GF_INTERVAL as f64 - 2.5) / (GF_INTERVAL as f64 - 1.0)];

/// One-pass rate control: each frame's quantizer from a model of bits
/// against quantizer step (bits ∝ step^-k) per frame type, corrected by how
/// far the coded bits run from the target. Without a bitrate, the
/// quantizer is fixed for inter frames and finer by `STEP_SCALE` for key
/// and golden frames.
struct Rate {
    opts: Vp9Options,
    /// Bits over (positive) or under the target so far.
    debt: f64,
    /// Bits a frame at step 1 would take, per frame type.
    model: [Option<f64>; 3],
    last_q: u8,
    per_frame: f64,
}

/// How bits fall with the quantizer step: bits ∝ step^-EXP.
const EXP: f64 = 1.25;

fn step_of(q: u8) -> f64 {
    f64::from(crate::tables::AC_QLOOKUP[usize::from(q)])
}

impl Rate {
    fn new(opts: Vp9Options) -> Self {
        Self {
            opts,
            debt: 0.0,
            model: [None; 3],
            last_q: opts.qindex,
            per_frame: 0.0,
        }
    }

    fn qindex(&mut self, kind: usize) -> u8 {
        if self.opts.bitrate == 0 {
            let q = self.opts.qindex;
            if q == 0 {
                return 0;
            }
            return q_of_step(step_of(q) * STEP_SCALE[kind]).clamp(self.opts.min_q, q.max(self.opts.min_q));
        }
        if self.per_frame == 0.0 {
            return q_of_step(step_of(self.last_q) * STEP_SCALE[kind]);
        }
        let m = match (self.model[kind], self.model[0]) {
            (Some(m), _) => m,
            (None, Some(k)) => k * [1.0, 0.5, 0.2][kind],
            (None, None) => return self.last_q,
        };
        // The debt is paid back over about a second.
        let target = (self.per_frame * SHARE[kind] - self.debt / 30.0).max(self.per_frame * 0.2);
        let step = (m / target).powf(1.0 / EXP);
        let q = q_of_step(step).clamp(self.opts.min_q, self.opts.max_q);
        // Inter frames move the quantizer at most a few steps a frame.
        let q = if kind == 2 {
            q.clamp(self.last_q.saturating_sub(24), self.last_q.saturating_add(24))
        } else {
            q
        };
        if kind == 2 {
            self.last_q = q;
        }
        q
    }

    fn account(&mut self, bytes: usize, seconds: f64, kind: usize, qindex: u8) {
        if self.opts.bitrate == 0 {
            return;
        }
        let bits = bytes as f64 * 8.0;
        self.per_frame = f64::from(self.opts.bitrate) * seconds;
        self.debt += bits - self.per_frame;
        let m = bits * step_of(qindex).powf(EXP);
        self.model[kind] = Some(match self.model[kind] {
            Some(old) => old * 0.6 + m * 0.4,
            None => m,
        });
        if kind == 0 && self.model[2].is_none() {
            self.last_q = q_of_step(step_of(qindex) / STEP_SCALE[0]);
        }
    }
}

/// The quantizer index whose AC step is nearest `step`.
fn q_of_step(step: f64) -> u8 {
    let t = &crate::tables::AC_QLOOKUP;
    let mut best = 0u8;
    let mut err = f64::MAX;
    for (i, &s) in t.iter().enumerate() {
        let e = (f64::from(s) - step).abs();
        if e < err {
            err = e;
            best = i as u8;
        }
    }
    best.max(1)
}

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
