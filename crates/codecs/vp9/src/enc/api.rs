//! The encoder on the codec traits: [`Vp9Encoder`] takes 4:2:0 frames on
//! the video codec contract, and [`EngineEncoder`] is the encoder export
//! opens through the engine's factory, with its `vpcC` configuration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{
    CodecError, CodecId, CodecParams, EncodeFactory, Packet, PixelFormat, Plane, VideoFrame,
};

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
    /// Pictures of lookahead: with 15 and more, each golden interval
    /// starts with a hidden alt-ref frame, the temporally filtered picture
    /// at its end, that the pictures between predict from. Pictures then
    /// come out that many pushes later (and from `drain`).
    pub lag: u32,
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
            min_q: 1,
            max_q: 236,
            threads: 0,
            lag: 0,
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
    /// Pictures waiting for the lookahead, oldest first.
    queue: std::collections::VecDeque<Queued>,
    /// Where the overlay of the alt-ref coded last goes (`since_key`).
    overlay_at: Option<u32>,
}

/// A picture in the lookahead.
struct Queued {
    planes: [Vec<u8>; 3],
    seconds: f64,
    pts: i64,
    duration: i64,
    timescale: u32,
}

impl Vp9Encoder {
    /// Every frame at quantizer index `qindex` (0: lossless).
    pub fn with_quantizer(width: u32, height: u32, qindex: u8) -> Result<Self, CodecError> {
        Self::with_options(
            width,
            height,
            Vp9Options {
                qindex,
                ..Vp9Options::default()
            },
        )
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
            rc: Rate::new(opts, u64::from(width) * u64::from(height)),
            cancel: Arc::new(AtomicBool::new(false)),
            queue: std::collections::VecDeque::new(),
            overlay_at: None,
        })
    }

    /// Bitrate 0 keeps the default quantizer; a positive bitrate aims at
    /// that rate.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported {
                what: "vp9 encoder",
            });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
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
    pub fn encode_planes(
        &mut self,
        planes: [(&[u8], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_planes(self.width, self.height, &planes)?;
        self.code(planes, seconds)
    }

    /// Takes one picture into the lookahead; returns the pictures coded
    /// (data, key frame, and the picture's `pts`, `duration`, `timescale`).
    #[allow(clippy::type_complexity)]
    pub fn push_planes(
        &mut self,
        planes: [(&[u8], usize); 3],
        seconds: f64,
        clock: (i64, i64, u32),
    ) -> Result<Vec<(Vec<u8>, bool, (i64, i64, u32))>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_planes(self.width, self.height, &planes)?;
        if self.opts.lag == 0 {
            let (d, k) = self.code(planes, seconds)?;
            return Ok(vec![(d, k, clock)]);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let dims = [
            (w, h),
            (w.div_ceil(2), h.div_ceil(2)),
            (w.div_ceil(2), h.div_ceil(2)),
        ];
        let tight: [Vec<u8>; 3] = std::array::from_fn(|p| {
            let (data, stride) = planes[p];
            let (pw, ph) = dims[p];
            (0..ph)
                .flat_map(|r| data[r * stride..r * stride + pw].iter().copied())
                .collect()
        });
        self.queue.push_back(Queued {
            planes: tight,
            seconds,
            pts: clock.0,
            duration: clock.1,
            timescale: clock.2,
        });
        let mut out = Vec::new();
        while self.queue.len() > self.opts.lag as usize {
            out.push(self.code_front()?);
        }
        Ok(out)
    }

    /// Codes what the lookahead still holds.
    #[allow(clippy::type_complexity)]
    pub fn flush(&mut self) -> Result<Vec<(Vec<u8>, bool, (i64, i64, u32))>, CodecError> {
        let mut out = Vec::new();
        while !self.queue.is_empty() {
            out.push(self.code_front()?);
        }
        Ok(out)
    }

    /// Codes the oldest picture of the lookahead, after an alt-ref when one
    /// starts here.
    #[allow(clippy::type_complexity)]
    fn code_front(&mut self) -> Result<(Vec<u8>, bool, (i64, i64, u32)), CodecError> {
        let Some(f) = self.queue.pop_front() else {
            return Err(CodecError::InvalidApi {
                what: "empty lookahead",
            });
        };
        let (w, h) = (self.width as usize, self.height as usize);
        let cw = w.div_ceil(2);
        // This picture has index `since_key` (0 for a key frame); the next
        // golden position is the alt-ref's target.
        let key_next = self.force_key || self.since_key >= self.opts.keyint;
        let idx = if key_next { 0 } else { self.since_key };
        let target = (idx / GF_INTERVAL + 1) * GF_INTERVAL;
        let k = (target - idx) as usize; // the target is queue[k - 1]
        let mut arf_source = None;
        if idx % GF_INTERVAL == 1
            && self.overlay_at.is_none()
            && k >= 2
            && k <= self.queue.len()
            && target < self.opts.keyint
            && super::arf::reachable(&f.planes[0], &self.queue[k - 1].planes[0], w, h)
        {
            let c = k - 1;
            let lo = c.saturating_sub(3);
            let hi = (c + 3).min(self.queue.len() - 1);
            let pics: Vec<super::arf::Picture<'_>> = (lo..=hi)
                .map(|i| super::arf::Picture {
                    planes: [
                        &self.queue[i].planes[0],
                        &self.queue[i].planes[1],
                        &self.queue[i].planes[2],
                    ],
                })
                .collect();
            let q = self.rc.peek_q();
            let strength = if q > 100 {
                5
            } else if q > 50 {
                4
            } else {
                3
            };
            arf_source = Some(super::arf::filter(&pics, c - lo, w, h, strength));
        }
        let mut data = Vec::new();
        if let Some(src) = arf_source {
            self.enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
            let qindex = self.rc.qindex(1, f.seconds);
            let coded = self.enc.encode(FrameParams {
                key: false,
                golden: false,
                hidden: true,
                qindex,
                speed: self.opts.speed,
            })?;
            self.rc.account(coded.data.len(), 0.0, 1, qindex);
            self.overlay_at = Some(target);
            data = coded.data;
        }
        let (frame, key) = self.code(
            [(&f.planes[0], w), (&f.planes[1], cw), (&f.planes[2], cw)],
            f.seconds,
        )?;
        let data = if data.is_empty() {
            frame
        } else {
            superframe(&[&data, &frame])
        };
        Ok((data, key, (f.pts, f.duration, f.timescale)))
    }

    /// Codes one shown picture.
    fn code(
        &mut self,
        planes: [(&[u8], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        let key = self.force_key || self.since_key >= self.opts.keyint;
        // A golden frame every `GF_INTERVAL` frames, coded finer: the
        // frames after it predict from it. After an alt-ref, the golden
        // frame is its overlay, coded as an inter frame.
        let golden = !key && self.since_key.is_multiple_of(GF_INTERVAL);
        let overlay = golden && self.overlay_at == Some(self.since_key);
        let kind = if key {
            0
        } else if golden && !overlay {
            1
        } else {
            2
        };
        if key || overlay {
            self.overlay_at = None;
        }
        self.enc.load(planes);
        let qindex = self.rc.qindex(kind, seconds);
        let coded = self.enc.encode(FrameParams {
            key,
            golden,
            hidden: false,
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
        vec![
            1,
            0,
            0,
            0,
            0,
            0,
            (8 << 4) | (1 << 1),
            primaries,
            transfer,
            matrix,
            0,
            0,
        ]
    }
}

impl VideoEncoder for Vp9Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let f = &input.frame;
        if f.format != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        }
        if f.width != self.width || f.height != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        if self.opts.bitrate > 0 && (input.duration <= 0 || input.timescale == 0) {
            return Err(CodecError::InvalidApi {
                what: "frame clock",
            });
        }
        let seconds = if input.timescale == 0 {
            0.0
        } else {
            input.duration as f64 / f64::from(input.timescale)
        };
        let planes = [
            (&f.y.samples[..], f.y.stride),
            (&f.u.samples[..], f.u.stride),
            (&f.v.samples[..], f.v.stride),
        ];
        let coded = self.push_planes(
            planes,
            seconds,
            (input.pts, input.duration, input.timescale),
        )?;
        Ok(coded
            .into_iter()
            .map(|(d, k, (pts, dur, ts))| Packet::video(d, pts, dur, ts, k))
            .collect())
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        let coded = self.flush()?;
        Ok(coded
            .into_iter()
            .map(|(d, k, (pts, dur, ts))| Packet::video(d, pts, dur, ts, k))
            .collect())
    }

    fn reset(&mut self) {
        self.enc = FrameEncoder::new(self.width, self.height, self.opts.threads());
        self.force_key = true;
        self.since_key = 0;
        self.rc = Rate::new(self.opts, u64::from(self.width) * u64::from(self.height));
        self.queue.clear();
        self.overlay_at = None;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// Registers the encoder export opens.
pub fn register_engine_encoders(factory: &mut EncodeFactory) {
    factory.register_video(CodecId::Vp9, open_engine_encoder);
}

fn open_engine_encoder(
    params: &CodecParams,
) -> Result<Box<dyn cutlass_encode::VideoEncoder>, CodecError> {
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
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
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
            Plane::tight(w, h, bytes.to_vec()).map_err(|_| CodecError::InvalidApi {
                what: "picture size",
            })
        };
        let frame = VideoFrame::yuv420p(
            width,
            height,
            plane(width, height, picture.y())?,
            plane(cw, ch, picture.u())?,
            plane(cw, ch, picture.v())?,
        )
        .map_err(|_| CodecError::InvalidApi {
            what: "picture size",
        })?;
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
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > 65536 || height > 65536 {
        return Err(CodecError::Limit {
            what: "vp9 frame edge",
        });
    }
    Ok(())
}

fn check_planes(width: u32, height: u32, planes: &[(&[u8], usize); 3]) -> Result<(), CodecError> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    for (p, (data, stride)) in planes.iter().enumerate() {
        let (pw, ph) = if p == 0 { (w, h) } else { (cw, ch) };
        if *stride < pw || data.len() < (ph - 1) * stride + pw {
            return Err(CodecError::InvalidApi {
                what: "picture planes",
            });
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
const SHARE: [f64; 3] = [
    6.0,
    2.5,
    (GF_INTERVAL as f64 - 2.5) / (GF_INTERVAL as f64 - 1.0),
];

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
    pixels: f64,
    /// Frames coded so far.
    frames: u32,
}

/// How bits fall with the quantizer step: bits ∝ step^-EXP.
const EXP: f64 = 1.25;

/// Bits a sample of an inter frame at quantizer step 1 (bits ∝ step^-EXP):
/// a start for the model, from a detailed 720p picture moving.
const FIRST_K: f64 = 48.0;

fn step_of(q: u8) -> f64 {
    f64::from(crate::tables::AC_QLOOKUP[usize::from(q)])
}

impl Rate {
    fn new(opts: Vp9Options, pixels: u64) -> Self {
        Self {
            opts,
            debt: 0.0,
            model: [None; 3],
            last_q: opts.qindex,
            per_frame: 0.0,
            pixels: pixels as f64,
            frames: 0,
        }
    }

    fn qindex(&mut self, kind: usize, seconds: f64) -> u8 {
        if self.opts.bitrate == 0 {
            let q = self.opts.qindex;
            if q == 0 {
                return 0;
            }
            return q_of_step(step_of(q) * STEP_SCALE[kind])
                .clamp(self.opts.min_q, q.max(self.opts.min_q));
        }
        if self.per_frame == 0.0 {
            // The first frame: the inter quantizer that a typical picture
            // would take at this many bits a sample, finer for its type.
            let target = f64::from(self.opts.bitrate) * seconds.max(1e-3) * SHARE[2];
            let step = (FIRST_K * self.pixels / target.max(1.0)).powf(1.0 / EXP);
            self.last_q = q_of_step(step).clamp(self.opts.min_q, self.opts.max_q);
            return q_of_step(step * STEP_SCALE[kind]).clamp(self.opts.min_q, self.opts.max_q);
        }
        let m = match (self.model[kind], self.model[0]) {
            (Some(m), _) => m,
            (None, Some(k)) => k * [1.0, 0.5, 0.2][kind],
            (None, None) => return self.last_q,
        };
        // The debt is paid back over about a second, sooner at the start
        // while the model is still a guess.
        let horizon = f64::from(self.frames.saturating_add(4).min(30));
        let target = (self.per_frame * SHARE[kind] - self.debt / horizon).max(self.per_frame * 0.2);
        let step = (m / target).powf(1.0 / EXP);
        let q = q_of_step(step).clamp(self.opts.min_q, self.opts.max_q);
        // Inter frames move the quantizer at most a few steps a frame.
        let q = if kind == 2 {
            q.clamp(
                self.last_q.saturating_sub(24),
                self.last_q.saturating_add(24),
            )
        } else {
            q
        };
        if kind == 2 {
            self.last_q = q;
        }
        q
    }

    /// The inter frames' quantizer now.
    fn peek_q(&self) -> u8 {
        self.last_q
    }

    fn account(&mut self, bytes: usize, seconds: f64, kind: usize, qindex: u8) {
        if self.opts.bitrate == 0 {
            return;
        }
        let bits = bytes as f64 * 8.0;
        if seconds > 0.0 {
            // A shown picture: its budget; a hidden one only spends.
            self.frames = self.frames.saturating_add(1);
            self.per_frame = f64::from(self.opts.bitrate) * seconds;
            self.debt -= self.per_frame;
        }
        self.debt += bits;
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

/// Frames joined into one packet with a superframe index.
fn superframe(frames: &[&[u8]]) -> Vec<u8> {
    let largest = frames.iter().map(|f| f.len()).max().unwrap_or(0);
    let mag = match largest {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xff_ffff => 3,
        _ => 4,
    };
    let marker = 0xc0 | ((mag - 1) << 3) as u8 | (frames.len() - 1) as u8;
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(f);
    }
    out.push(marker);
    for f in frames {
        out.extend_from_slice(&(f.len() as u32).to_le_bytes()[..mag]);
    }
    out.push(marker);
    out
}
