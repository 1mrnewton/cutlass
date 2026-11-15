//! The encoder on the codec traits: [`Vp9Encoder`] takes 8-, 10- and
//! 12-bit frames on the video codec contract, and [`EngineEncoder`] is the encoder export
//! opens through the engine's factory, with its `vpcC` configuration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{
    CodecError, CodecId, CodecParams, EncodeFactory, Packet, PixelFormat, Plane, VideoFrame,
};

use super::{EncPixel, FrameEncoder, FrameParams};

/// The quantizer index a factory encoder with no bitrate keeps.
pub const DEFAULT_Q: u8 = 60;

/// The chroma planes' size against luma's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Chroma {
    /// Half across and down (profile 0).
    #[default]
    Yuv420,
    /// Half across (profile 1).
    Yuv422,
    /// Half down (profile 1).
    Yuv440,
    /// Full size (profile 1).
    Yuv444,
}

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
    /// Search effort, 0 (most) to 10 (least).
    pub speed: u8,
    /// The fewest and most quantizer indices the rate control may use.
    pub min_q: u8,
    pub max_q: u8,
    /// Threads (tile columns) to code with; 0: one per core.
    pub threads: usize,
    /// Pictures of lookahead: with 15 and more, each golden interval
    /// starts with a hidden alt-ref frame, the temporally filtered picture
    /// at its end, that the pictures between predict from. Pictures then
    /// come out that many pushes later (and from `drain`). `None`:
    /// [`DEFAULT_LAG`] below speed 9, none at the realtime speeds 9 and 10
    /// (as libvpx's good and realtime deadlines).
    pub lag: Option<u32>,
    /// With a bitrate: 0 aims at it on average (libvpx's `end-usage=vbr`);
    /// more holds it constant (`end-usage=cbr`) through a buffer this many
    /// milliseconds long, as streaming and calls want: frame types take
    /// smaller shares and a frame over its budget is paid back within
    /// half the buffer.
    pub cbr_buffer_ms: u32,
    /// The chroma planes' size: 4:2:0 (profile 0, the default) or, in
    /// profile 1, 4:2:2, 4:4:0 or 4:4:4 (FFmpeg's `-pix_fmt yuv422p`,
    /// `yuv440p`, `yuv444p`). The chroma planes pushed are that size.
    pub chroma: Chroma,
    /// Tile rows, log2 of the count (0 to 2; libvpx's `tile-rows`). They
    /// restart the arithmetic coder at each tile row and change nothing
    /// else.
    pub tile_rows: u8,
    /// Adaptive quantization (libvpx's `aq-mode=1`): flat regions finer,
    /// busy ones coarser, by segment. Better looking, a little worse in
    /// PSNR.
    pub aq: bool,
    /// Bits a sample: 8 (profiles 0 and 1, the default), or 10 or 12
    /// (profiles 2 and 3), whose pictures go in as `u16` samples in the
    /// low bits ([`Vp9Encoder::push_planes16`]; FFmpeg's `yuv420p10le`
    /// and the like).
    pub bit_depth: u8,
}

/// The lookahead below speed 9 unless set: with it, alt-refs take 10 to
/// 11% off the bits at the same PSNR (Y or averaged over the planes) on
/// a grainy pan and 6 to 10% on a clean clip, for about a fifth more
/// time.
pub const DEFAULT_LAG: u32 = 16;

/// The fastest speed an alt-ref frame is coded at.
const ARF_SPEED: u8 = 6;

impl Vp9Options {
    /// The lookahead in pictures.
    pub fn lag(&self) -> u32 {
        self.lag
            .unwrap_or(if self.speed >= 9 { 0 } else { DEFAULT_LAG })
    }

    /// Chroma subsampling across and down.
    fn ss(&self) -> (usize, usize) {
        match self.chroma {
            Chroma::Yuv420 => (1, 1),
            Chroma::Yuv422 => (1, 0),
            Chroma::Yuv440 => (0, 1),
            Chroma::Yuv444 => (0, 0),
        }
    }

    /// The chroma planes' size for a `w`×`h` picture.
    fn chroma_dims(&self, w: usize, h: usize) -> (usize, usize) {
        let (sx, sy) = self.ss();
        (w.div_ceil(1 << sx), h.div_ceil(1 << sy))
    }

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
            lag: None,
            cbr_buffer_ms: 0,
            chroma: Chroma::Yuv420,
            tile_rows: 0,
            aq: false,
            bit_depth: 8,
        }
    }
}

/// VP9 on the video codec contract: profile 0 (8-bit 4:2:0), 1 (8-bit
/// 4:2:2, 4:4:0, 4:4:4), 2 and 3 (the same at 10 and 12 bits).
pub struct Vp9Encoder {
    lane: Lane,
}

/// The encoder of the stream's sample size.
enum Lane {
    Eight(Core<u8>),
    Deep(Core<u16>),
}

/// The encoder of pictures of `P` samples.
struct Core<P: EncPixel> {
    width: u32,
    height: u32,
    opts: Vp9Options,
    enc: FrameEncoder<P>,
    since_key: u32,
    force_key: bool,
    rc: Rate,
    cancel: Arc<AtomicBool>,
    /// Pictures waiting for the lookahead, oldest first.
    queue: std::collections::VecDeque<Queued<P>>,
    /// Where the overlay of the alt-ref coded last goes (`since_key`).
    overlay_at: Option<u32>,
    /// The luma of the picture coded last, tight, for finding a cut.
    last_luma: Vec<P>,
    /// Whether the lookahead found that the next picture starts a scene.
    cut_next: bool,
}

/// A picture in the lookahead.
struct Queued<P> {
    planes: [Vec<P>; 3],
    seconds: f64,
    pts: i64,
    duration: i64,
    timescale: u32,
}

/// The pictures a push codes: data, key frame, and the picture's `pts`,
/// `duration`, `timescale`.
type Coded = Vec<(Vec<u8>, bool, (i64, i64, u32))>;

/// A call on the encoder of the stream's sample size.
macro_rules! on_lane {
    ($self:expr, $c:ident => $e:expr) => {
        match &mut $self.lane {
            Lane::Eight($c) => $e,
            Lane::Deep($c) => $e,
        }
    };
}

impl Vp9Encoder {
    /// Every frame at quantizer index `qindex` (0: lossless), each coded
    /// as it is pushed (no lookahead).
    pub fn with_quantizer(width: u32, height: u32, qindex: u8) -> Result<Self, CodecError> {
        Self::with_options(
            width,
            height,
            Vp9Options {
                qindex,
                lag: Some(0),
                ..Vp9Options::default()
            },
        )
    }

    pub fn with_options(width: u32, height: u32, opts: Vp9Options) -> Result<Self, CodecError> {
        check_size(width, height)?;
        let lane = match opts.bit_depth {
            8 => Lane::Eight(Core::new(width, height, opts)),
            10 | 12 => Lane::Deep(Core::new(width, height, opts)),
            _ => {
                return Err(CodecError::Unsupported {
                    what: "vp9 bit depth",
                });
            }
        };
        Ok(Self { lane })
    }

    /// Bitrate 0 keeps the default quantizer; a positive bitrate aims at
    /// that rate. The pixel format picks the chroma size and bit depth.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported {
                what: "vp9 encoder",
            });
        }
        let Some((chroma, bit_depth)) = format_of(params.pixel) else {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        };
        let opts = Vp9Options {
            bitrate: params.bitrate,
            chroma,
            bit_depth,
            ..Vp9Options::default()
        };
        Self::with_options(params.width, params.height, opts)
    }

    /// libvpx's quantizer index for a quality level of 0 to 63 (its
    /// `quantizer_to_qindex`; FFmpeg's `-crf`, `-qmin` and `-qmax` take
    /// levels), for [`Vp9Options::qindex`], `min_q` and `max_q`.
    pub fn qindex_of_level(level: u8) -> u8 {
        match level.min(63) {
            63 => 255,
            62 => 249,
            l => l * 4,
        }
    }

    /// The quantizer index of the next frame.
    pub fn qindex(&self) -> u8 {
        match &self.lane {
            Lane::Eight(c) => c.rc.last_q,
            Lane::Deep(c) => c.rc.last_q,
        }
    }

    /// The options the encoder codes with.
    fn opts(&self) -> &Vp9Options {
        match &self.lane {
            Lane::Eight(c) => &c.opts,
            Lane::Deep(c) => &c.opts,
        }
    }

    /// The second pass of a two-pass encode (FFmpeg's `-pass 2`), from
    /// the first: a pass over the same pictures at the fixed quantizer
    /// `first_q` (no bitrate), whose packets took `first_bits` bits each.
    /// With a bitrate, every frame is then coded at the one quantizer that
    /// should spend it (bits fall as the step to the power 1.25), which
    /// puts the bits where the pictures need them; as the coding goes, the
    /// step follows how what has been spent compares with the share of the
    /// budget the first pass gives the pictures so far.
    pub fn set_second_pass(&mut self, first_bits: &[u64], first_q: u8) {
        on_lane!(self, c => c.set_second_pass(first_bits, first_q));
    }

    /// Makes the next frame a key frame.
    pub fn force_keyframe(&mut self) {
        on_lane!(self, c => c.force_key = true);
    }

    /// Codes one 8-bit picture: `planes` are Y, U, V with their strides
    /// (chroma of [`Vp9Options::chroma`]'s size).
    pub fn encode_planes(
        &mut self,
        planes: [(&[u8], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        match &mut self.lane {
            Lane::Eight(c) => c.encode_planes(planes, seconds),
            Lane::Deep(_) => Err(deep_only()),
        }
    }

    /// [`Vp9Encoder::encode_planes`] of a 10- or 12-bit picture: samples
    /// in the low bits of each `u16`, strides in samples.
    pub fn encode_planes16(
        &mut self,
        planes: [(&[u16], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        match &mut self.lane {
            Lane::Deep(c) => {
                c.check_range(&planes)?;
                c.encode_planes(planes, seconds)
            }
            Lane::Eight(_) => Err(eight_only()),
        }
    }

    /// Takes one 8-bit picture into the lookahead; returns the pictures
    /// coded (data, key frame, and the picture's `pts`, `duration`,
    /// `timescale`).
    pub fn push_planes(
        &mut self,
        planes: [(&[u8], usize); 3],
        seconds: f64,
        clock: (i64, i64, u32),
    ) -> Result<Coded, CodecError> {
        match &mut self.lane {
            Lane::Eight(c) => c.push_planes(planes, seconds, clock),
            Lane::Deep(_) => Err(deep_only()),
        }
    }

    /// [`Vp9Encoder::push_planes`] of a 10- or 12-bit picture: samples in
    /// the low bits of each `u16`, strides in samples.
    pub fn push_planes16(
        &mut self,
        planes: [(&[u16], usize); 3],
        seconds: f64,
        clock: (i64, i64, u32),
    ) -> Result<Coded, CodecError> {
        match &mut self.lane {
            Lane::Deep(c) => {
                c.check_range(&planes)?;
                c.push_planes(planes, seconds, clock)
            }
            Lane::Eight(_) => Err(eight_only()),
        }
    }

    /// Codes what the lookahead still holds.
    pub fn flush(&mut self) -> Result<Coded, CodecError> {
        on_lane!(self, c => c.flush())
    }

    /// The decoder configuration (`vpcC`, version 1) for the given colour
    /// codes: the profile, level unset, the bit depth and chroma, limited
    /// range.
    pub fn vpcc_with_colour(&self, [primaries, transfer, matrix]: [u8; 3]) -> Vec<u8> {
        let o = self.opts();
        let profile = u8::from(o.chroma != Chroma::Yuv420) | if o.bit_depth > 8 { 2 } else { 0 };
        // ISO BMFF's chroma codes: 4:2:0 (co-sited with luma's left
        // column), 4:2:2, 4:4:4; 4:4:0 has none of its own.
        let chroma = match o.chroma {
            Chroma::Yuv420 | Chroma::Yuv440 => 1,
            Chroma::Yuv422 => 2,
            Chroma::Yuv444 => 3,
        };
        vec![
            1,
            0,
            0,
            0,
            profile,
            0,
            (o.bit_depth << 4) | (chroma << 1),
            primaries,
            transfer,
            matrix,
            0,
            0,
        ]
    }
}

/// The chroma and bit depth of a pixel format the encoder takes.
fn format_of(f: PixelFormat) -> Option<(Chroma, u8)> {
    Some(match f {
        PixelFormat::Yuv420p => (Chroma::Yuv420, 8),
        PixelFormat::Yuv422p => (Chroma::Yuv422, 8),
        PixelFormat::Yuv440p => (Chroma::Yuv440, 8),
        PixelFormat::Yuv444p => (Chroma::Yuv444, 8),
        PixelFormat::Yuv420p10 => (Chroma::Yuv420, 10),
        PixelFormat::Yuv422p10 => (Chroma::Yuv422, 10),
        PixelFormat::Yuv440p10 => (Chroma::Yuv440, 10),
        PixelFormat::Yuv444p10 => (Chroma::Yuv444, 10),
        PixelFormat::Yuv420p12 => (Chroma::Yuv420, 12),
        PixelFormat::Yuv422p12 => (Chroma::Yuv422, 12),
        PixelFormat::Yuv440p12 => (Chroma::Yuv440, 12),
        PixelFormat::Yuv444p12 => (Chroma::Yuv444, 12),
        _ => return None,
    })
}

fn deep_only() -> CodecError {
    CodecError::InvalidApi {
        what: "8-bit planes to a 10- or 12-bit encoder",
    }
}

fn eight_only() -> CodecError {
    CodecError::InvalidApi {
        what: "16-bit planes to an 8-bit encoder",
    }
}

impl Core<u16> {
    /// Whether the picture's planes hold it, every sample within the bit
    /// depth.
    fn check_range(&self, planes: &[(&[u16], usize); 3]) -> Result<(), CodecError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let c = self.opts.chroma_dims(w, h);
        check_planes(self.width, self.height, c, planes)?;
        let max = (1u16 << self.opts.bit_depth) - 1;
        for (p, (data, stride)) in planes.iter().enumerate() {
            let (pw, ph) = if p == 0 { (w, h) } else { c };
            if (0..ph).any(|r| data[r * stride..r * stride + pw].iter().any(|&v| v > max)) {
                return Err(CodecError::InvalidApi {
                    what: "sample past the bit depth",
                });
            }
        }
        Ok(())
    }
}

impl<P: EncPixel> Core<P> {
    fn new(width: u32, height: u32, opts: Vp9Options) -> Self {
        Self {
            width,
            height,
            opts,
            enc: {
                let mut enc = FrameEncoder::with_depth(
                    width,
                    height,
                    opts.threads(),
                    opts.ss(),
                    opts.bit_depth,
                );
                enc.set_tile_rows(opts.tile_rows);
                enc
            },
            since_key: 0,
            force_key: true,
            rc: Rate::new(opts, u64::from(width) * u64::from(height)),
            cancel: Arc::new(AtomicBool::new(false)),
            queue: std::collections::VecDeque::new(),
            overlay_at: None,
            last_luma: Vec::new(),
            cut_next: false,
        }
    }

    fn set_second_pass(&mut self, first_bits: &[u64], first_q: u8) {
        let mut cum = Vec::with_capacity(first_bits.len());
        let mut sum = 0.0;
        for &b in first_bits {
            sum += b as f64;
            cum.push(sum);
        }
        self.rc.pass2 = Some(Pass2 {
            cum,
            q: first_q.max(1),
            steps: Vec::new(),
        });
    }

    /// Codes one picture: `planes` are Y, U, V with their strides.
    fn encode_planes(
        &mut self,
        planes: [(&[P], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_planes(
            self.width,
            self.height,
            self.opts
                .chroma_dims(self.width as usize, self.height as usize),
            &planes,
        )?;
        self.code(planes, seconds)
    }

    /// Takes one picture into the lookahead; returns the pictures coded.
    fn push_planes(
        &mut self,
        planes: [(&[P], usize); 3],
        seconds: f64,
        clock: (i64, i64, u32),
    ) -> Result<Coded, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_planes(
            self.width,
            self.height,
            self.opts
                .chroma_dims(self.width as usize, self.height as usize),
            &planes,
        )?;
        if self.opts.lag() == 0 {
            let (d, k) = self.code(planes, seconds)?;
            return Ok(vec![(d, k, clock)]);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let c = self.opts.chroma_dims(w, h);
        let dims = [(w, h), c, c];
        let tight: [Vec<P>; 3] = std::array::from_fn(|p| {
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
        while self.queue.len() > self.opts.lag() as usize {
            out.push(self.code_front()?);
        }
        Ok(out)
    }

    /// Codes what the lookahead still holds.
    fn flush(&mut self) -> Result<Coded, CodecError> {
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
        let cw = self.opts.chroma_dims(w, h).0;
        // This picture has index `since_key` (0 for a key frame); the next
        // golden position is the alt-ref's target.
        // A cut (which no alt-ref spans: `reachable` fails across it).
        if self.is_cut((&f.planes[0], w)) {
            self.cut_next = true;
        }
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
            && super::arf::reachable(
                &f.planes[0],
                &self.queue[k - 1].planes[0],
                w,
                h,
                self.opts.bit_depth,
            )
        {
            let c = k - 1;
            let lo = c.saturating_sub(3);
            let hi = (c + 3).min(self.queue.len() - 1);
            let pics: Vec<super::arf::Picture<'_, P>> = (lo..=hi)
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
            arf_source = Some(super::arf::filter(
                &pics,
                c - lo,
                w,
                h,
                strength,
                self.opts.threads(),
                self.opts.ss(),
                self.opts.bit_depth,
            ));
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
                // Its match is a golden interval away: the one-pass path's
                // short search would code it as intra.
                speed: self.opts.speed.min(ARF_SPEED),
                altref_future: false,
                aq: self.opts.aq,
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
        planes: [(&[P], usize); 3],
        seconds: f64,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        let key = self.force_key || self.since_key >= self.opts.keyint;
        // A cut starts a golden frame: LAST and GOLDEN take the new scene,
        // each block still chooses intra or inter, and the probabilities
        // the stream has learnt are kept (a key frame would reset them, and
        // costs more on a cut to another picture of the same subject).
        let cut = !key && (self.cut_next || self.is_cut(planes[0]));
        self.cut_next = false;
        // A golden frame every `GF_INTERVAL` frames, coded finer: the
        // frames after it predict from it. After an alt-ref, the golden
        // frame is its overlay, coded as an inter frame.
        let golden = !key && (cut || self.since_key.is_multiple_of(GF_INTERVAL));
        let overlay = golden && !cut && self.overlay_at == Some(self.since_key);
        let kind = if key {
            0
        } else if golden && !overlay {
            1
        } else {
            2
        };
        if key || overlay || cut {
            // (An alt-ref of the scene before a cut has nothing to show.)
            self.overlay_at = None;
        }
        self.enc.load(planes);
        let qindex = self.rc.qindex(kind, seconds);
        // Between an alt-ref and its overlay, ALTREF is ahead.
        let altref_future = !key && !overlay && self.overlay_at.is_some();
        let coded = self.enc.encode(FrameParams {
            key,
            golden,
            hidden: false,
            qindex,
            speed: self.opts.speed,
            altref_future,
            aq: self.opts.aq,
        })?;
        let kind = if coded.key { 0 } else { kind };
        self.rc.account(coded.data.len(), seconds, kind, qindex);
        self.force_key = false;
        self.since_key = if coded.key { 1 } else { self.since_key + 1 };
        let (w, h) = (self.width as usize, self.height as usize);
        let (y, stride) = planes[0];
        self.last_luma.clear();
        for r in 0..h {
            self.last_luma
                .extend_from_slice(&y[r * stride..r * stride + w]);
        }
        Ok((coded.data, coded.key))
    }

    /// Whether luma `y` (rows `stride` apart) cuts from the picture coded
    /// last: few of its blocks are anywhere near in that one, so that a
    /// key frame costs little more than an inter frame and starts the
    /// scene's references afresh (libvpx's scene cut detection).
    fn is_cut(&self, (y, stride): (&[P], usize)) -> bool {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.last_luma.len() != w * h || self.since_key == 0 {
            return false;
        }
        let tight: std::borrow::Cow<'_, [P]> = if stride == w {
            std::borrow::Cow::Borrowed(&y[..w * h])
        } else {
            std::borrow::Cow::Owned(
                (0..h)
                    .flat_map(|r| y[r * stride..r * stride + w].iter().copied())
                    .collect(),
            )
        };
        super::arf::cut(&self.last_luma, &tight, w, h)
    }

    fn reset(&mut self) {
        self.enc = FrameEncoder::with_depth(
            self.width,
            self.height,
            self.opts.threads(),
            self.opts.ss(),
            self.opts.bit_depth,
        );
        self.enc.set_tile_rows(self.opts.tile_rows);
        self.force_key = true;
        self.since_key = 0;
        self.rc = Rate::new(self.opts, u64::from(self.width) * u64::from(self.height));
        self.queue.clear();
        self.overlay_at = None;
    }
}

impl VideoEncoder for Vp9Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let f = &input.frame;
        let o = *self.opts();
        if format_of(f.format) != Some((o.chroma, o.bit_depth)) {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        }
        let (width, height) = match &self.lane {
            Lane::Eight(c) => (c.width, c.height),
            Lane::Deep(c) => (c.width, c.height),
        };
        if f.width != width || f.height != height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        if o.bitrate > 0 && (input.duration <= 0 || input.timescale == 0) {
            return Err(CodecError::InvalidApi {
                what: "frame clock",
            });
        }
        let seconds = if input.timescale == 0 {
            0.0
        } else {
            input.duration as f64 / f64::from(input.timescale)
        };
        let clock = (input.pts, input.duration, input.timescale);
        let coded = if o.bit_depth > 8 {
            // Little-endian samples, two bytes each, rows `stride` bytes
            // apart.
            let samples = |p: &Plane| -> Vec<u16> {
                p.samples[..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| u16::from_le_bytes(b))
                    .collect()
            };
            let (y, u, v) = (samples(&f.y), samples(&f.u), samples(&f.v));
            self.push_planes16(
                [
                    (&y[..], f.y.stride / 2),
                    (&u[..], f.u.stride / 2),
                    (&v[..], f.v.stride / 2),
                ],
                seconds,
                clock,
            )?
        } else {
            let planes = [
                (&f.y.samples[..], f.y.stride),
                (&f.u.samples[..], f.u.stride),
                (&f.v.samples[..], f.v.stride),
            ];
            self.push_planes(planes, seconds, clock)?
        };
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
        on_lane!(self, c => c.reset());
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        match &self.lane {
            Lane::Eight(c) => Arc::clone(&c.cancel),
            Lane::Deep(c) => Arc::clone(&c.cancel),
        }
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

fn check_planes<P>(
    width: u32,
    height: u32,
    (cw, ch): (usize, usize),
    planes: &[(&[P], usize); 3],
) -> Result<(), CodecError> {
    let (w, h) = (width as usize, height as usize);
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
const STEP_SCALE: [f64; 3] = [0.35, 0.55, 1.0];

/// At a constant bitrate, the first key frame's step over the one the
/// bitrate guesses for inter frames.
const CBR_FIRST_SCALE: f64 = 1.5;

/// At a constant bitrate, how many quantizer indices a golden or key frame
/// is at most finer than the inter frames. On a 720p pan at 2 Mb/s, half-
/// second windows stay within -15% and +15% of the rate at speed 6 (they
/// were -33% and +38% on average bitrate), -24% and +13% at speed 9.
const CBR_BOOST: u8 = 16;

/// [`SHARE`] at a constant bitrate: smaller boosts, which the buffer can
/// hold.
const SHARE_CBR: [f64; 3] = [
    4.0,
    1.5,
    (GF_INTERVAL as f64 - 1.5) / (GF_INTERVAL as f64 - 1.0),
];

/// The share of a frame's budget each frame type takes.
const SHARE: [f64; 3] = [
    8.0,
    3.5,
    (GF_INTERVAL as f64 - 3.5) / (GF_INTERVAL as f64 - 1.0),
];

/// One-pass rate control: each frame's quantizer from a model of bits
/// against quantizer step (bits ∝ step^-k) per frame type, corrected by how
/// far the coded bits run from the target. Without a bitrate, the
/// quantizer is fixed for inter frames and finer by `STEP_SCALE` for key
/// and golden frames.
/// A first pass's bits, summed picture by picture, at its quantizer.
struct Pass2 {
    cum: Vec<f64>,
    q: u8,
    /// The inter step each picture was coded at.
    steps: Vec<f64>,
}

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
    /// The first pass of two.
    pass2: Option<Pass2>,
    /// Bits spent so far.
    spent: f64,
}

/// How bits fall with the quantizer step: bits ∝ step^-EXP.
const EXP: f64 = 1.25;

/// Bits a sample of an inter frame at quantizer step 1 (bits ∝ step^-EXP):
/// a start for the model, from a detailed 720p picture moving.
const FIRST_K: f64 = 48.0;

/// The AC step of quantizer index `q` at `bd` bits, on the 8-bit scale
/// (the 10- and 12-bit steps over 4 and 16): the model's constants hold
/// at every bit depth.
fn step_of(q: u8, bd: u8) -> f64 {
    let (t, scale) = ac_table(bd);
    f64::from(t[usize::from(q)]) / scale
}

/// The AC steps of a bit depth, and their scale against 8 bits'.
fn ac_table(bd: u8) -> (&'static [i16; 256], f64) {
    match bd {
        10 => (&crate::tables::AC_QLOOKUP_10, 4.0),
        12 => (&crate::tables::AC_QLOOKUP_12, 16.0),
        _ => (&crate::tables::AC_QLOOKUP, 1.0),
    }
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
            pass2: None,
            spent: 0.0,
        }
    }

    fn qindex(&mut self, kind: usize, seconds: f64) -> u8 {
        if self.opts.bitrate == 0 {
            let q = self.opts.qindex;
            if q == 0 {
                return 0;
            }
            return q_of_step(
                step_of(q, self.opts.bit_depth) * STEP_SCALE[kind],
                self.opts.bit_depth,
            )
            .clamp(self.opts.min_q, q.max(self.opts.min_q));
        }
        if let Some(p) = &mut self.pass2 {
            // Two passes. Frame i's bits are about k·b_i·(s/s1)^-EXP, with
            // b_i its first-pass bits at step s1 and s the inter step it is
            // coded at; k, the two passes' scale (a faster first pass
            // spends more), is measured on the frames coded so far. The
            // step chosen spends what is left of the budget on what is
            // left of the first pass; frame types take their steps from it
            // as a fixed quantizer does.
            let frames = p.cum.len().max(1);
            let s1 = step_of(p.q, self.opts.bit_depth);
            let first = p.cum.last().copied().unwrap_or(1.0).max(1.0);
            let total = f64::from(self.opts.bitrate) * seconds.max(1e-3) * frames as f64;
            let done = (self.frames as usize).min(frames);
            let bits_at = |i: usize| p.cum[i] - if i == 0 { 0.0 } else { p.cum[i - 1] };
            let predicted: f64 = (0..done)
                .map(|i| bits_at(i) * (p.steps.get(i).copied().unwrap_or(s1) / s1).powf(-EXP))
                .sum();
            let k = if done >= 2 && predicted > 0.0 {
                (self.spent / predicted).clamp(0.1, 10.0)
            } else {
                1.0
            };
            let left_first = (first - if done == 0 { 0.0 } else { p.cum[done - 1] }).max(1.0);
            let left =
                (total - self.spent).max(total * 0.02 * (frames - done) as f64 / frames as f64);
            let base = s1 * (k * left_first / left.max(1.0)).powf(1.0 / EXP);
            let q = q_of_step(base * STEP_SCALE[kind], self.opts.bit_depth)
                .clamp(self.opts.min_q, self.opts.max_q);
            if p.steps.len() <= done {
                p.steps.resize(done + 1, s1);
            }
            p.steps[done] = step_of(q, self.opts.bit_depth) / STEP_SCALE[kind];
            if kind == 2 {
                self.last_q = q;
            }
            return q;
        }
        if self.per_frame == 0.0 {
            // The first frame: the inter quantizer that a typical picture
            // would take at this many bits a sample, finer for its type.
            let target = f64::from(self.opts.bitrate) * seconds.max(1e-3) * SHARE[2];
            let step = (FIRST_K * self.pixels / target.max(1.0)).powf(1.0 / EXP);
            self.last_q =
                q_of_step(step, self.opts.bit_depth).clamp(self.opts.min_q, self.opts.max_q);
            // At a constant bitrate the first key frame takes a coarser
            // step, not a finer one: the buffer cannot hold the ten frames'
            // worth a fine one costs.
            let scale = if self.opts.cbr_buffer_ms > 0 {
                CBR_FIRST_SCALE
            } else {
                STEP_SCALE[kind]
            };
            return q_of_step(step * scale, self.opts.bit_depth)
                .clamp(self.opts.min_q, self.opts.max_q);
        }
        let m = match (self.model[kind], self.model[0]) {
            (Some(m), _) => m,
            (None, Some(k)) => k * [1.0, 0.5, 0.2][kind],
            (None, None) => return self.last_q,
        };
        let cbr = self.opts.cbr_buffer_ms > 0;
        // The debt is paid back over about a second, sooner at the start
        // while the model is still a guess; at a constant bitrate within
        // half the buffer.
        let mut horizon = f64::from(self.frames.saturating_add(4).min(30));
        if cbr && seconds > 0.0 {
            let buffer = f64::from(self.opts.cbr_buffer_ms) / 1000.0 / seconds;
            horizon = horizon.min((buffer / 2.0).max(2.0));
        }
        let share = if cbr { SHARE_CBR[kind] } else { SHARE[kind] };
        let target = (self.per_frame * share - self.debt / horizon).max(self.per_frame * 0.2);
        let step = (m / target).powf(1.0 / EXP);
        let q = q_of_step(step, self.opts.bit_depth).clamp(self.opts.min_q, self.opts.max_q);
        // Inter frames move the quantizer at most a few steps a frame (more
        // at a constant bitrate, which must follow the content).
        let reach = if cbr { 48 } else { 24 };
        let q = if kind == 2 {
            q.clamp(
                self.last_q.saturating_sub(reach),
                self.last_q.saturating_add(reach),
            )
        } else if cbr {
            // A golden or key frame at a constant bitrate is at most a few
            // steps finer than the inter frames: a model of one golden
            // frame, a dozen frames old, can be far off.
            q.max(self.last_q.saturating_sub(CBR_BOOST))
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
        self.spent += bits;
        if seconds > 0.0 {
            // A shown picture: its budget; a hidden one only spends.
            self.frames = self.frames.saturating_add(1);
            self.per_frame = f64::from(self.opts.bitrate) * seconds;
            self.debt -= self.per_frame;
        }
        self.debt += bits;
        let m = bits * step_of(qindex, self.opts.bit_depth).powf(EXP);
        self.model[kind] = Some(match self.model[kind] {
            Some(old) => old * 0.6 + m * 0.4,
            None => m,
        });
        if kind == 0 && self.model[2].is_none() {
            self.last_q = q_of_step(
                step_of(qindex, self.opts.bit_depth) / STEP_SCALE[0],
                self.opts.bit_depth,
            );
        }
    }
}

/// The quantizer index whose AC step (on the 8-bit scale, as
/// [`step_of`]) is nearest `step`.
fn q_of_step(step: f64, bd: u8) -> u8 {
    let (t, scale) = ac_table(bd);
    let mut best = 0u8;
    let mut err = f64::MAX;
    for (i, &s) in t.iter().enumerate() {
        let e = (f64::from(s) / scale - step).abs();
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
