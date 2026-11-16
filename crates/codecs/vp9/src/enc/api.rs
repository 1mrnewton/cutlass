//! The encoder on the codec traits: [`Vp9Encoder`] takes 8-, 10- and
//! 12-bit frames on the video codec contract, and [`EngineEncoder`] is the encoder export
//! opens through the engine's factory, with its `vpcC` configuration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{
    CodecError, CodecId, CodecParams, ColorRange, EncodeFactory, MatrixCoefficients, Packet,
    PixelFormat, Plane, VideoFrame,
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

/// The colour space a stream signals (VP9's `color_space`; libvpx's
/// `color-space`, FFmpeg's `-colorspace`). The encoder codes the samples
/// as given; this only tells a player how to show them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorSpace {
    Unknown,
    /// ITU-R BT.601 (the default, as libvpx's).
    #[default]
    Bt601,
    Bt709,
    Smpte170,
    Smpte240,
    Bt2020,
    /// RGB: the planes are G, B, R (FFmpeg's `gbrp`), 4:4:4 only
    /// (profiles 1 and 3), always full range.
    Srgb,
}

impl ColorSpace {
    /// The code in the key frame header.
    fn code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Bt601 => 1,
            Self::Bt709 => 2,
            Self::Smpte170 => 3,
            Self::Smpte240 => 4,
            Self::Bt2020 => 5,
            Self::Srgb => 7,
        }
    }
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
    /// The colour space the key frames signal (libvpx's `color-space`).
    pub color_space: ColorSpace,
    /// Full (`true`) or studio range (libvpx's `color-range`).
    pub full_range: bool,
    /// The size a player should show the pictures at, when not their own
    /// (VP9's render size; FFmpeg sets it from a sample aspect ratio).
    pub render_size: Option<(u32, u32)>,
    /// The loop filter's sharpness, 0 to 7 (libvpx's `sharpness`): higher
    /// filters edges less, keeping more detail at some PSNR.
    pub sharpness: u8,
    /// Error resilient frames (libvpx's `error-resilient`): each frame
    /// starts from the default probabilities and uses neither the
    /// previous frame's motion nor its segmentation, so that a lost frame
    /// spoils only the frames that predict from its picture. Costs bits.
    pub error_resilient: bool,
    /// Frame parallel decodability (libvpx's `frame-parallel`): the
    /// probabilities carry only the headers' explicit updates, without
    /// adapting to each frame's symbols, so that a decoder can parse a
    /// frame before the one before it is through. Costs a few bits.
    pub frame_parallel: bool,
    /// libvpx's `static-thresh`: an inter block whose prediction error is
    /// below this threshold (16 × it over a 64×64 block, scaled to the
    /// block, at least the DC quantizer step squared over 64) codes no
    /// residual. 0: off.
    pub static_thresh: u32,
    /// Alt-ref frames with the lookahead (libvpx's `auto-alt-ref`; on by
    /// default).
    pub alt_ref: bool,
    /// The alt-ref temporal filter's strength, 0 (none) to 6 (libvpx's
    /// `arnr-strength`); `None`: 3 to 5 by quantizer (3 in two passes).
    pub arnr_strength: Option<u8>,
    /// The pictures the alt-ref filter averages, 1 to 15, centred on the
    /// target (libvpx's `arnr-maxframes`); `None`: 7.
    pub arnr_max_frames: Option<u8>,
    /// With a bitrate: bits under the target so far by up to this share
    /// of it, in percent, are not spent later (libvpx's `undershoot-pct`).
    /// 0: every bit is.
    pub undershoot_pct: u32,
    /// With a bitrate: bits over the target so far by up to this share of
    /// it, in percent, are not paid back (libvpx's `overshoot-pct`). 0:
    /// every bit is.
    pub overshoot_pct: u32,
    /// A scene cut at least this many frames after the last key frame
    /// starts a key frame (libvpx's automatic key frames past
    /// `kf-min-dist`; FFmpeg's `-keyint_min`), so that a player can seek
    /// to it. 0: cuts start golden frames, which cost less.
    pub keyint_min: u32,
    /// Constrained quality (libvpx's `end-usage=cq`, FFmpeg's `-crf` with
    /// a bitrate): with a bitrate, no inter frame is finer than this
    /// quantizer index (key and golden frames by their own steps from
    /// it), so that easy content takes fewer bits than the bitrate.
    pub cq_level: Option<u8>,
    /// With a bitrate, the most bits a key frame may take, in percent of
    /// a frame's share of the bitrate (libvpx's `max-intra-rate`). 0: no
    /// limit.
    pub max_intra_pct: u32,
    /// Tile columns, log2 of the count (libvpx's `tile-columns`; as many
    /// as the width allows), each coded on its own thread; `None`: one
    /// per thread.
    pub tile_columns: Option<u8>,
    /// The level the `vpcC` configuration names (10 × the level, e.g. 31
    /// for 3.1; libvpx's `level`, which also holds the stream to it). 0:
    /// unset.
    pub level: u8,
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

    /// What every frame signals.
    fn stream(&self) -> super::Stream {
        super::Stream {
            sharpness: self.sharpness,
            color_space: self.color_space.code(),
            full_range: self.full_range || self.color_space == ColorSpace::Srgb,
            render_size: self.render_size,
            error_resilient: self.error_resilient,
            frame_parallel: self.frame_parallel,
            static_thresh: self.static_thresh,
        }
    }

    /// The frame encoder these options code with.
    fn frame_encoder<P: EncPixel>(&self, width: u32, height: u32) -> FrameEncoder<P> {
        let mut enc =
            FrameEncoder::with_depth(width, height, self.threads(), self.ss(), self.bit_depth);
        enc.set_tile_rows(self.tile_rows);
        if let Some(log2) = self.tile_columns {
            enc.set_tile_columns(log2);
        }
        enc.set_stream(self.stream());
        enc
    }

    /// Whether the options make a stream.
    fn check(&self) -> Result<(), CodecError> {
        let bad = |what| Err(CodecError::Unsupported { what });
        if self.sharpness > 7 {
            return bad("vp9 sharpness past 7");
        }
        if self.color_space == ColorSpace::Srgb && self.chroma != Chroma::Yuv444 {
            return bad("vp9 sRGB without 4:4:4");
        }
        if let Some((w, h)) = self.render_size
            && (w == 0 || h == 0 || w > 65536 || h > 65536)
        {
            return bad("vp9 render size");
        }
        if self.arnr_strength.is_some_and(|s| s > 6) {
            return bad("vp9 arnr strength past 6");
        }
        if self.arnr_max_frames.is_some_and(|n| n == 0 || n > 15) {
            return bad("vp9 arnr frames outside 1 to 15");
        }
        if self.tile_columns.is_some_and(|t| t > 6) {
            return bad("vp9 tile columns past 6");
        }
        Ok(())
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
            color_space: ColorSpace::Bt601,
            full_range: false,
            render_size: None,
            sharpness: 0,
            error_resilient: false,
            frame_parallel: false,
            static_thresh: 0,
            alt_ref: true,
            arnr_strength: None,
            arnr_max_frames: None,
            undershoot_pct: 0,
            overshoot_pct: 0,
            keyint_min: 0,
            cq_level: None,
            max_intra_pct: 0,
            tile_columns: None,
            level: 0,
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
    /// The places (`since_key`) of the last golden (or key) frame and of
    /// the next.
    gold: (u32, u32),
    /// The luma of the picture coded last, tight, for finding a cut.
    last_luma: Vec<P>,
    /// Two passes: the pictures in the reference pool's slots (see
    /// [`POOL`]), and the golden frame's.
    pool: Vec<Held<P>>,
    gold_src: Option<Held<P>>,
    /// Where the content of the picture coded last stands
    /// ([`super::arf::shift`] summed over the pictures).
    pos: (isize, isize),
    /// Pictures shown so far.
    shown: u64,
}

/// A picture a reference slot holds, as its source.
struct Held<P> {
    slot: u8,
    /// Its number among the pictures shown.
    shown: u64,
    /// Where its content stood (see [`Core::pos`]).
    pos: (isize, isize),
    planes: [Vec<P>; 3],
}

/// A picture in the lookahead.
struct Queued<P> {
    planes: [Vec<P>; 3],
    seconds: f64,
    pts: i64,
    duration: i64,
    timescale: u32,
    /// Whether it cuts from the picture before it (see [`Core::is_cut`]).
    cut: bool,
    /// Two passes: how far its content moved from the picture before it
    /// ([`super::arf::shift`]).
    step: (isize, isize),
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
        opts.check()?;
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
        // The colour the parameters name (BT.601 when they name none).
        let color_space = match params.matrix {
            MatrixCoefficients::Bt709 => ColorSpace::Bt709,
            MatrixCoefficients::Bt2020 => ColorSpace::Bt2020,
            MatrixCoefficients::Rgb if chroma == Chroma::Yuv444 => ColorSpace::Srgb,
            _ => ColorSpace::Bt601,
        };
        let opts = Vp9Options {
            bitrate: params.bitrate,
            chroma,
            bit_depth,
            color_space,
            full_range: params.range == ColorRange::Full,
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
    /// `first_q` (no bitrate; [`Self::first_pass_qindex`] is a good one),
    /// whose packets took `first_bits` bits each. With a bitrate, every
    /// frame is then coded at the one quantizer that should spend it (bits
    /// fall as the step to the power 1.25), which puts the bits where the
    /// pictures need them; as the coding goes, the step follows how what
    /// has been spent compares with what the first pass's bits predicted,
    /// for inter frames, key and cut frames, and golden frames and alt-refs
    /// apart. A cut takes a key frame's quantizer, and blocks that code
    /// much also try a smaller transform.
    pub fn set_second_pass(&mut self, first_bits: &[u64], first_q: u8) {
        on_lane!(self, c => c.set_second_pass(first_bits, first_q));
    }

    /// The quantizer index a first pass should code at for this encoder's
    /// bitrate, pictures `seconds` apart: about the one the bitrate will
    /// take, so that the second pass's model of bits against quantizer
    /// need not reach far (bits fall faster with the quantizer on grain
    /// than on clean pictures). Without a bitrate, the encoder's own.
    pub fn first_pass_qindex(&self, seconds: f64) -> u8 {
        match &self.lane {
            Lane::Eight(c) => c.rc.first_pass_qindex(seconds),
            Lane::Deep(c) => c.rc.first_pass_qindex(seconds),
        }
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
    /// codes: the profile, [`Vp9Options::level`], the bit depth, chroma
    /// and range.
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
            o.level,
            (o.bit_depth << 4) | (chroma << 1) | u8::from(o.stream().full_range),
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
            // The largest sample of each row, a loop the compiler
            // vectorizes (a search for the first sample past `max` did
            // not).
            let top = (0..ph).fold(0, |m, r| {
                data[r * stride..r * stride + pw]
                    .iter()
                    .fold(m, |m, &v| m.max(v))
            });
            if top > max {
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
            enc: opts.frame_encoder(width, height),
            since_key: 0,
            force_key: true,
            rc: Rate::new(opts, u64::from(width) * u64::from(height)),
            cancel: Arc::new(AtomicBool::new(false)),
            queue: std::collections::VecDeque::new(),
            overlay_at: None,
            gold: (0, GF_INTERVAL),
            last_luma: Vec::new(),
            pool: Vec::new(),
            gold_src: None,
            shown: 0,
            pos: (0, 0),
        }
    }

    fn set_second_pass(&mut self, first_bits: &[u64], first_q: u8) {
        self.rc.pass2 = Some(Pass2::new(first_bits, first_q, self.opts.keyint));
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
        self.code(planes, seconds, None)
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
            let (d, k) = self.code(planes, seconds, None)?;
            return Ok(vec![(d, k, clock)]);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let c = self.opts.chroma_dims(w, h);
        let dims = [(w, h), c, c];
        let tight: [Vec<P>; 3] = std::array::from_fn(|p| {
            let (data, stride) = planes[p];
            let (pw, ph) = dims[p];
            let mut tight = Vec::with_capacity(pw * ph);
            for r in 0..ph {
                tight.extend_from_slice(&data[r * stride..r * stride + pw]);
            }
            tight
        });
        // Whether it cuts from the picture before it, found once: from
        // the one queued last, or with the queue empty the one coded last.
        let cut = match self.queue.back() {
            Some(before) => super::arf::cut(&before.planes[0], &tight[0], w, h),
            None => self.is_cut((&tight[0], w)),
        };
        let step = if self.pooled() {
            match self.queue.back() {
                Some(before) => super::arf::shift(&before.planes[0], &tight[0], w, h, before.step),
                None if self.last_luma.len() == w * h => {
                    super::arf::shift(&self.last_luma, &tight[0], w, h, (0, 0))
                }
                None => (0, 0),
            }
        } else {
            (0, 0)
        };
        self.queue.push_back(Queued {
            step,
            cut,
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
        if self.shown == 0 && self.rc.wants_probe() {
            self.probe(&f)?;
        }
        // This picture has index `since_key` (0 for a key frame); the next
        // golden position is the alt-ref's target.
        // A cut (which no alt-ref spans: `reachable` fails across it); the
        // first picture has nothing to cut from.
        let cut = f.cut && self.since_key != 0;
        let key_next = self.force_key || self.since_key >= self.opts.keyint;
        let idx = if key_next { 0 } else { self.since_key };
        let target = self.gold.1;
        let k = target.saturating_sub(idx) as usize; // the target is queue[k - 1]
        let mut arf_source = None;
        if !key_next
            && self.opts.alt_ref
            && idx == self.gold.0 + 1
            // (One pass: only after a golden place, not after a cut.)
            && (self.rc.pass2.is_some() || idx % GF_INTERVAL == 1)
            && !cut
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
            // One pass at a fixed quantizer: see Q_TUNED_SPEED.
            let q_tuned = self.opts.bitrate == 0 && self.opts.speed <= Q_TUNED_SPEED;
            let frames = if q_tuned { Q_ARF_FRAMES } else { 7 };
            let radius = usize::from(self.opts.arnr_max_frames.unwrap_or(frames) - 1) / 2;
            let lo = c.saturating_sub(radius);
            let hi = (c + radius).min(self.queue.len() - 1);
            let pics: Vec<super::arf::Picture<'_, P>> = (lo..=hi)
                .map(|i| super::arf::Picture {
                    planes: [
                        &self.queue[i].planes[0],
                        &self.queue[i].planes[1],
                        &self.queue[i].planes[2],
                    ],
                })
                .collect();
            // (Two passes: 3 at any quantizer, the pictures predicting
            // from the pool's unfiltered ones too: on the grainy pan, 1%
            // fewer bits at the same averaged PSNR, 0.6% on the clip from
            // a fractal to `testsrc2`.)
            let q = self.rc.peek_q();
            let strength = if let Some(s) = self.opts.arnr_strength {
                u32::from(s)
            } else if self.pooled() {
                3
            } else if q_tuned {
                Q_ARF_STRENGTH
            } else if q > 100 {
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
                self.rc.pass2.is_some(),
            ));
        }
        let mut data = Vec::new();
        if let Some(src) = arf_source {
            // (Where its target stands: the steps of the pictures to it.)
            let at = self
                .queue
                .iter()
                .take(k)
                .fold((self.pos.0 + f.step.0, self.pos.1 + f.step.1), |p, q| {
                    (p.0 + q.step.0, p.1 + q.step.1)
                });
            let golden_slot = self.pick_golden([&src[0], &src[1], &src[2]], at);
            self.enc.load([(&src[0], w), (&src[1], cw), (&src[2], cw)]);
            self.rc.alt_ref(k);
            let qindex = self.rc.qindex(1, f.seconds, false);
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
                second_pass: self.rc.pass2.is_some(),
                golden_slot: golden_slot.0,
                also_refresh: 0,
                noise: golden_slot.1,
            })?;
            self.rc.account(coded.data.len(), 0.0, 1, qindex);
            self.overlay_at = Some(target);
            data = coded.data;
        }
        let (frame, key) = self.code(
            [(&f.planes[0], w), (&f.planes[1], cw), (&f.planes[2], cw)],
            f.seconds,
            Some((cut, f.step)),
        )?;
        let data = if data.is_empty() {
            frame
        } else {
            superframe(&[&data, &frame])
        };
        Ok((data, key, (f.pts, f.duration, f.timescale)))
    }

    /// Codes one shown picture; `cut` is whether it cuts from the picture
    /// before it, when the lookahead has found that already.
    fn code(
        &mut self,
        planes: [(&[P], usize); 3],
        seconds: f64,
        ahead: Option<(bool, (isize, isize))>,
    ) -> Result<(Vec<u8>, bool), CodecError> {
        let cut = ahead.map(|a| a.0);
        let forced = self.force_key || self.since_key >= self.opts.keyint;
        // A cut starts a golden frame: LAST and GOLDEN take the new scene,
        // each block still chooses intra or inter, and the probabilities
        // the stream has learnt are kept (a key frame would reset them, and
        // costs more on a cut to another picture of the same subject).
        let cut = !forced && cut.unwrap_or_else(|| self.is_cut(planes[0]));
        // What looks a cut may be the picture moved far ([`super::arf::jump`]):
        // then it is an inter frame whose blocks also search there.
        let jump = if cut { self.jump_of(planes[0]) } else { None };
        let cut = cut && jump.is_none();
        if let Some(g) = jump {
            self.enc.set_global(g);
        }
        // (With `keyint_min`, a cut far enough from the last key frame is
        // one.)
        let min = self.opts.keyint_min;
        let key = forced || (cut && min > 0 && self.since_key >= min);
        let cut = cut && !key;
        // A golden frame every `GF_INTERVAL` frames, coded finer: the
        // frames after it predict from it. After an alt-ref, the golden
        // frame is its overlay, coded as an inter frame.
        let golden = !key && (cut || self.since_key == self.gold.1);
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
        // With the reference pool: the picture's tight planes.
        let tight: Option<[Vec<P>; 3]> = self.pooled().then(|| {
            let (w, h) = (self.width as usize, self.height as usize);
            let c = self.opts.chroma_dims(w, h);
            std::array::from_fn(|p| {
                let (data, stride) = planes[p];
                let (pw, ph) = if p == 0 { (w, h) } else { c };
                let mut t = Vec::with_capacity(pw * ph);
                for r in 0..ph {
                    t.extend_from_slice(&data[r * stride..r * stride + pw]);
                }
                t
            })
        });
        if key {
            self.pool.clear();
            self.gold_src = None;
            self.pos = (0, 0);
        } else if let Some(t) = &tight {
            let step = match ahead {
                Some(a) => a.1,
                None if self.last_luma.len() == t[0].len() => {
                    let (w, h) = (self.width as usize, self.height as usize);
                    super::arf::shift(&self.last_luma, &t[0], w, h, (0, 0))
                }
                None => (0, 0),
            };
            self.pos = (self.pos.0 + step.0, self.pos.1 + step.1);
        }
        let golden_slot = match &tight {
            Some(t) if !key => self.pick_golden([&t[0], &t[1], &t[2]], self.pos),
            _ => (1, 0.0),
        };
        let pool_slot = POOL_FIRST + (self.shown % u64::from(POOL)) as u8;
        let also_refresh = if tight.is_some() && !key {
            1 << pool_slot
        } else {
            0
        };
        self.enc.load(planes);
        let qindex = self.rc.qindex(kind, seconds, cut);
        // Between an alt-ref and its overlay, ALTREF is ahead.
        // (Error resilient frames take no compound prediction.)
        let altref_future =
            !key && !overlay && self.overlay_at.is_some() && !self.opts.error_resilient;
        let coded = self.enc.encode(FrameParams {
            key,
            golden,
            hidden: false,
            qindex,
            speed: self.opts.speed,
            altref_future,
            aq: self.opts.aq,
            second_pass: self.rc.pass2.is_some(),
            golden_slot: golden_slot.0,
            also_refresh,
            noise: golden_slot.1,
        })?;
        let kind = if coded.key { 0 } else { kind };
        if let Some(t) = tight {
            if coded.key || golden {
                self.gold_src = Some(Held {
                    slot: 1,
                    shown: self.shown,
                    pos: self.pos,
                    planes: t.clone(),
                });
            }
            if !coded.key {
                self.pool.retain(|e| e.slot != pool_slot);
                self.pool.push(Held {
                    slot: pool_slot,
                    shown: self.shown,
                    pos: self.pos,
                    planes: t,
                });
            }
        }
        self.shown += 1;
        if coded.key || golden {
            let at = if coded.key { 0 } else { self.since_key };
            let next = self.next_golden(at, planes[0]);
            self.gold = (at, next);
            self.rc.golden_at((next - at) as usize);
        }
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

    /// One pass at a variable bitrate, before the first picture: the
    /// pictures of the lookahead (`first` and the queue) coded fast at a
    /// fixed quantizer, whose bits start the rate control's model in
    /// place of a guess (a first pass over the lookahead, see
    /// [`Rate::probe`]).
    fn probe(&mut self, first: &Queued<P>) -> Result<(), CodecError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let cw = self.opts.chroma_dims(w, h).0;
        let q = self.rc.first_pass_qindex(first.seconds);
        let mut enc = FrameEncoder::<P>::with_depth(
            self.width,
            self.height,
            self.opts.threads(),
            self.opts.ss(),
            self.opts.bit_depth,
        );
        let mut bits = Vec::new();
        for (i, f) in std::iter::once(first)
            .chain(self.queue.iter())
            .take(PROBE_FRAMES)
            .enumerate()
        {
            enc.load([(&f.planes[0], w), (&f.planes[1], cw), (&f.planes[2], cw)]);
            let coded = enc.encode(FrameParams {
                key: i == 0,
                golden: false,
                hidden: false,
                qindex: q,
                speed: 10,
                altref_future: false,
                aq: false,
                second_pass: false,
                golden_slot: 1,
                also_refresh: 0,
                noise: 0.0,
            })?;
            bits.push(coded.data.len() as f64 * 8.0);
        }
        self.rc.probe(&bits, q);
        Ok(())
    }

    /// Whether the reference pool is kept ([`POOL`]): two passes, and one
    /// pass up to [`POOL_SPEED`].
    fn pooled(&self) -> bool {
        self.rc.pass2.is_some() || self.opts.speed <= POOL_SPEED
    }

    /// The slot GOLDEN reads for the picture `to` (tight planes): the
    /// golden frame's, or, with the reference pool, a picture of it that
    /// predicts `to` far better in chroma ([`super::arf::phase_error`]):
    /// content moved by an even number of luma samples, which chroma
    /// predicts at its whole samples.
    fn pick_golden(&self, to: [&[P]; 3], at: (isize, isize)) -> (u8, f32) {
        if !self.pooled() {
            return (1, 0.0);
        }
        let Some(g) = &self.gold_src else {
            return (1, 0.0);
        };
        let (w, h) = (self.width as usize, self.height as usize);
        let ss = self.opts.ss();
        // The golden picture and those of the pool (but the one shown last,
        // LAST's), measured on threads.
        let last = self.shown.checked_sub(1);
        let cands: Vec<&Held<P>> = std::iter::once(g)
            .chain(self.pool.iter().filter(|e| Some(e.shown) != last))
            .collect();
        let measure = |c: &&Held<P>| {
            // The vector from the picture: how far the content moved since.
            let seed = (at.0 - c.pos.0, at.1 - c.pos.1);
            let f = &c.planes;
            super::arf::phase_error([&f[0], &f[1], &f[2]], to, w, h, ss, seed)
        };
        let errs: Vec<(f32, u64)> = if self.opts.threads() <= 1 {
            cands.iter().map(measure).collect()
        } else {
            std::thread::scope(|sc| {
                let hs: Vec<_> = cands.iter().map(|c| sc.spawn(move || measure(c))).collect();
                hs.into_iter()
                    .map(|h| h.join().unwrap_or((f32::MAX, u64::MAX)))
                    .collect()
            })
        };
        let base = errs[0];
        let mut best = (1u8, base);
        let mut least = base.0;
        for (c, e) in cands.iter().zip(&errs).skip(1) {
            least = least.min(e.0);
            if e.1 * 10 < base.1 * 7 && e.0 <= base.0 * 1.25 && e.1 < best.1.1 {
                best = (c.slot, *e);
            }
        }
        (best.0, least)
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

    /// [`super::arf::jump`] of luma `y` (rows `stride` apart) from the
    /// picture coded last.
    fn jump_of(&self, (y, stride): (&[P], usize)) -> Option<(isize, isize)> {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.last_luma.len() != w * h {
            return None;
        }
        let tight: Vec<P> = (0..h)
            .flat_map(|r| y[r * stride..r * stride + w].iter().copied())
            .collect();
        super::arf::jump(&self.last_luma, &tight, w, h, self.opts.threads())
    }

    fn reset(&mut self) {
        self.enc = self.opts.frame_encoder(self.width, self.height);
        self.force_key = true;
        self.since_key = 0;
        self.rc = Rate::new(self.opts, u64::from(self.width) * u64::from(self.height));
        self.queue.clear();
        self.overlay_at = None;
        self.gold = (0, GF_INTERVAL);
        self.pool.clear();
        self.gold_src = None;
        self.shown = 0;
        self.pos = (0, 0);
    }

    /// The place of the golden frame after the one (or the key frame) at
    /// `at` being coded now, whose luma is `y` (rows `stride` apart). One
    /// pass: the next multiple of `GF_INTERVAL`. Two passes, with the
    /// next picture in the lookahead: [`GF_REACH`] samples of motion
    /// away (at most `GF_INTERVAL`, at least [`GF_MIN_INTERVAL`]), so that
    /// the alt-ref and golden frames the pictures between predict from
    /// stay near (libvpx's two passes end a golden group sooner as the
    /// motion grows, too).
    fn next_golden(&self, at: u32, (y, stride): (&[P], usize)) -> u32 {
        let fixed = (at / GF_INTERVAL + 1) * GF_INTERVAL;
        let Some(next) = self.queue.front() else {
            return fixed;
        };
        if self.rc.pass2.is_none() {
            return fixed;
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let tight: std::borrow::Cow<'_, [P]> = if stride == w {
            std::borrow::Cow::Borrowed(&y[..w * h])
        } else {
            std::borrow::Cow::Owned(
                (0..h)
                    .flat_map(|r| y[r * stride..r * stride + w].iter().copied())
                    .collect(),
            )
        };
        let m = super::arf::motion(&tight, &next.planes[0], w, h);
        if m == 0 {
            return at + GF_INTERVAL;
        }
        at + (GF_REACH / m).clamp(GF_MIN_INTERVAL, GF_INTERVAL)
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
        // Export converts with BT.709, which the stream signals too.
        planar.matrix = MatrixCoefficients::Bt709;
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

/// Two passes, and one pass up to [`POOL_SPEED`], keep the last [`POOL`]
/// shown pictures in the reference slots from [`POOL_FIRST`], and GOLDEN
/// reads whichever of them, or the golden frame, predicts the picture's
/// chroma best at whole samples.
const POOL: u8 = 5;
const POOL_FIRST: u8 = 3;

/// The fastest speed whose one pass keeps the reference pool. At speed 5
/// against FFmpeg's `libvpx-vp9` on the README's grainy pan, it took 16
/// points off the bits at the same PSNR-Y and 32 at the same PSNR-U; 16
/// and 19 on the clip from a still picture, 1 and 5 on the three scenes
/// (where the pan is a third of the pictures). At speed 7, 17 and 36 on
/// the pan and 20 and 29 on the still picture's clip; at speed 8 it gained
/// nothing (one candidate a block, its search seldom finds the pool's
/// picture). It costs no time at a fixed quantizer; at a bitrate the
/// pictures it predicts better leave the bitrate finer quantizers, which
/// take longer (see `tile::CHEAP_DIST`).
const POOL_SPEED: u8 = 7;

/// Frames from one golden frame to the next (at most, in two passes).
const GF_INTERVAL: u32 = 12;

/// The pictures of the lookahead [`Core::probe`] codes.
const PROBE_FRAMES: usize = 17;

/// In two passes, how far (in samples) the content may move from one
/// golden frame to the next ([`super::arf::motion`] a picture times the
/// interval): on the grainy pan, 7 to 8 samples a picture, 8 pictures.
/// On the pan alone (1 to 3 Mb/s), 4% fewer bits at the same PSNR-Y and
/// at the same averaged PSNR; within a point either way on the clips with
/// cuts. Fixed intervals swing the pan by where its vectors fall against
/// chroma's half samples (from +0.4% to +13% averaged against libvpx's
/// between 4 and 10), and 8 for every scene lost 9 points of PSNR-Y
/// BD-rate on the clip with a still picture.
const GF_REACH: u32 = 64;

/// The shortest golden interval of [`GF_REACH`].
const GF_MIN_INTERVAL: u32 = 6;

/// The quantizer step of each frame type (key, golden, inter) over the
/// inter frames'.
const STEP_SCALE: [f64; 3] = [0.35, 0.55, 1.0];

/// Up to this speed, one pass at a fixed quantizer (FFmpeg's `-crf N -b:v
/// 0`) codes its golden frames and alt-refs at [`Q_GOLDEN_SCALE`] of the
/// inter step, and filters its alt-refs at [`Q_ARF_STRENGTH`] over
/// [`Q_ARF_FRAMES`] pictures. (Not at 9 and 10: speed 10 at a fixed
/// quantizer is the first pass of two, whose bits the second reads.)
const Q_TUNED_SPEED: u8 = 8;

/// Against FFmpeg's `-crf 24` to `48`, at `q=60` to `210` (speed 5, one
/// pass): finer golden frames and alt-refs, filtered less over fewer
/// pictures, take the clip of a still scene and two jumps from +1.3% /
/// +1.3% to −4.4% / −4.1% (BD-rate at the same PSNR-Y / averaged PSNR),
/// the three scenes from −4.5% / −0.3% to −5.3% / −1.4% and the grainy pan
/// from −8.4% / −4.9% to −13.9% / −10.8%; at `q=50` to `225`, from
/// +3.8% / +4.1% to 0.0% / −0.1% on the still scene. (The golden scale
/// alone took the still scene to +0.1% / −0.3%, the weaker filter alone to
/// −0.2% / +0.4%.) At speeds 6 and 8, fewer bits on all three too.
const Q_GOLDEN_SCALE: f64 = 0.5;
const Q_ARF_STRENGTH: u32 = 1;
const Q_ARF_FRAMES: u8 = 5;

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

/// A first pass's bits, picture by picture, at its quantizer, and what
/// the second pass has spent against them. The pictures fall in three
/// classes, each with its own scale k between the passes: inter frames
/// (0); key frames and cuts (1: a picture at a key frame's place, or one
/// that took well over the pictures around it); and the pictures at the
/// second pass's golden places (2), whose bits its alt-refs take. A
/// faster first pass spends more on inter frames than on key frames, and
/// an alt-ref costs nothing like the inter frame the first pass coded
/// there.
struct Pass2 {
    /// Each picture's first-pass bits and class.
    first: Vec<(f64, usize)>,
    /// Each picture's first-pass bits as an inter frame's, and a golden
    /// frame's there over an inter frame's.
    inter: Vec<f64>,
    gratio: Vec<f64>,
    q: u8,
    /// First-pass bits of the pictures still to code, by class.
    left: [f64; 3],
    /// Bits spent, and the model's bits at k = 1, by class.
    spent: [f64; 3],
    pred: [f64; 3],
    /// The first-pass bits and class an alt-ref about to be coded takes.
    arf: (f64, usize),
    /// The inter step chosen last.
    base: f64,
    /// First-pass bits a picture, on average.
    mean: f64,
}

/// The share of a golden picture's first-pass bits its overlay keeps
/// when an alt-ref stands for it.
const OVERLAY_SHARE: f64 = 0.1;

/// A picture outside a golden place whose first-pass bits are over this
/// many times the median of the six around it is a key frame's class (a
/// cut).
const CUT_BITS: f64 = 1.8;

/// The second pass's start, in k (the second pass at speed 6 spends about
/// half a speed-10 first pass's bits), and how many average pictures'
/// worth of the overall k each class's own starts with.
const K_START: f64 = 0.5;
const K_PRIOR_FRAMES: f64 = 4.0;

/// Over the last pictures the second pass spends as if the stream went
/// on at its average for this many: a miss near the end is not made up
/// on the last few frames alone (whose quantizer would swing).
const END_FRAMES: f64 = 10.0;

impl Pass2 {
    fn new(bits: &[u64], q: u8, keyint: u32) -> Self {
        let keyint = keyint.max(1) as usize;
        let b: Vec<f64> = bits.iter().map(|&b| b as f64).collect();
        // The first pass codes a golden frame every `GF_INTERVAL` at its
        // finer step. For when the second pass places its golden frames
        // elsewhere (`golden_at`): each picture's bits as an inter frame's
        // (at a first-pass golden place, the median of the six around),
        // and what a golden frame there would take over that, as the
        // nearest first-pass golden frame took over its neighbours.
        let is_gold = |i: usize| {
            let at = i % keyint;
            at != 0 && at.is_multiple_of(GF_INTERVAL as usize)
        };
        let inter: Vec<f64> = (0..b.len())
            .map(|i| {
                if !is_gold(i) {
                    return b[i];
                }
                let mut near: Vec<f64> = (i.saturating_sub(3)..(i + 4).min(b.len()))
                    .filter(|&j| j != i && !is_gold(j))
                    .map(|j| b[j])
                    .collect();
                near.sort_by(f64::total_cmp);
                near.get(near.len() / 2).copied().unwrap_or(b[i]).min(b[i])
            })
            .collect();
        let golds: Vec<(usize, f64)> = (0..b.len())
            .filter(|&i| is_gold(i))
            .map(|i| (i, (b[i] / inter[i].max(1.0)).max(1.0)))
            .collect();
        let gratio: Vec<f64> = (0..b.len())
            .map(|i| {
                golds
                    .iter()
                    .min_by_key(|g| g.0.abs_diff(i))
                    .map_or(1.0 / STEP_SCALE[1].powf(EXP), |g| g.1)
            })
            .collect();
        let first: Vec<(f64, usize)> = (0..b.len())
            .map(|i| {
                let at = i % keyint;
                if at == 0 {
                    return (b[i], 1);
                }
                if at.is_multiple_of(GF_INTERVAL as usize) {
                    return (b[i], 2);
                }
                let mut near: Vec<f64> = (i.saturating_sub(3)..(i + 4).min(b.len()))
                    .filter(|&j| j != i)
                    .map(|j| b[j])
                    .collect();
                near.sort_by(f64::total_cmp);
                let median = near.get(near.len() / 2).copied().unwrap_or(0.0);
                (b[i], usize::from(b[i] > CUT_BITS * median))
            })
            .collect();
        let mut left = [0.0; 3];
        for &(b, c) in &first {
            left[c] += b;
        }
        Self {
            first,
            inter,
            gratio,
            q: q.max(1),
            left,
            spent: [0.0; 3],
            pred: [0.0; 3],
            arf: (0.0, 2),
            base: 0.0,
            mean: b.iter().sum::<f64>() / b.len().max(1) as f64,
        }
    }

    /// The next golden place is picture `at`, `n` after the one coded
    /// now: it and the places every `n` after it (as far as is known) take
    /// the golden class (an alt-ref will take their bits), and no other
    /// picture still to code does.
    fn golden_at(&mut self, at: usize, n: usize) {
        for i in at.saturating_sub(n) + 1..self.first.len() {
            let t = &mut self.first[i];
            if t.1 == 1 {
                continue;
            }
            let c = if i >= at && (i - at).is_multiple_of(n.max(1)) {
                2
            } else {
                0
            };
            let b = self.inter[i] * if c == 2 { self.gratio[i] } else { 1.0 };
            self.left[t.1] -= t.0;
            self.left[c] += b;
            *t = (b, c);
        }
    }

    /// An alt-ref is coded now for picture `target`: it takes most of that
    /// picture's first-pass bits, its overlay the rest.
    fn alt_ref(&mut self, target: usize) {
        if let Some(t) = self.first.get_mut(target) {
            // Its bits stay left in their class until it is coded.
            let keep = t.0 * OVERLAY_SHARE;
            self.left[t.1] -= keep;
            self.left[0] += keep;
            self.arf = (t.0 - keep, t.1);
            *t = (keep, 0);
        }
    }
}

/// One-pass rate control: each frame's quantizer from a model of bits
/// against quantizer step (bits ∝ step^-k) per frame type, corrected by how
/// far the coded bits run from the target. Without a bitrate, the
/// quantizer is fixed for inter frames and finer by `STEP_SCALE` for key
/// and golden frames. With a first pass, see [`Pass2`].
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
    /// What the last key frame took past a frame's share, and the frames
    /// since (see [`KEY_SPREAD`]).
    key_extra: (f64, f64),
    /// One pass at a variable bitrate: the key frame and inter frame
    /// models [`Core::probe`] measured.
    probed: Option<(f64, f64)>,
    /// The next golden-type frame is an alt-ref.
    arf_next: bool,
}

/// At this speed (no lookahead by default, so no alt-refs) one pass at a
/// variable bitrate budgets the golden frames to come at the key frame's
/// step they take, not at an alt-ref's: budgeted at the alt-ref's, a
/// golden frame every 12 took 1.8 times the share it was given, and the
/// debt it left stood, so that streams came in 6 to 9% over the rate (at
/// 3 and 4 Mb/s on the three scenes, where FFmpeg's realtime 8 comes in
/// under it, 3.24 and 4.27 Mb/s). Budgeted at the key frame's, within 4% of
/// the rate on the three clips of the README, with 1 point fewer bits at
/// the same PSNR.
const GOLDEN_KEY_SPEED: u8 = 9;

/// How bits fall with the quantizer step: bits ∝ step^-EXP.
const EXP: f64 = 1.25;

/// Bits a sample of an inter frame at quantizer step 1 (bits ∝ step^-EXP):
/// a start for the model, from a detailed 720p picture moving.
const FIRST_K: f64 = 48.0;

/// One pass at a variable bitrate: the weight of each inter frame in the
/// inter frames' model of bits against the step, once the frames so far
/// outnumber its inverse (a running mean until then). The model moved by
/// 0.4 of each frame before: on a clip of three scenes, 11% fewer bits at
/// the same PSNR-Y (9% averaged over the planes), 5% on a still scene
/// with two cuts, 0.5 to 1% on a grainy pan.
const INTER_MODEL_WEIGHT: f64 = 0.05;

/// One pass at a variable bitrate: the frames over which a key frame's
/// bits past a frame's share fall due, and over which the first key
/// frame's quantizer is planned.
const KEY_SPREAD: f64 = 30.0;

/// The most frames' worth of bits the first key frame is planned to take
/// (what the lookahead's still pictures would leave it is much more).
const KEY_MOST: f64 = 20.0;

/// The frames the probe's inter model counts as in the model's running
/// mean.
const PROBE_WEIGHT: f64 = 5.0;

/// The range of [`Vp9Encoder::first_pass_qindex`].
const FIRST_PASS_Q: (u8, u8) = (40, 200);

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
            key_extra: (0.0, 0.0),
            probed: None,
            arf_next: false,
        }
    }

    /// The quantizer of a frame of type `kind` (key, golden or alt-ref,
    /// inter); `cut` for a golden frame that starts a scene.
    fn qindex(&mut self, kind: usize, seconds: f64, cut: bool) -> u8 {
        let q = self.qindex_in(kind, seconds, cut);
        if kind == 0 && self.opts.bitrate > 0 && self.opts.max_intra_pct > 0 {
            q.max(self.intra_cap(seconds)).min(self.opts.max_q.max(q))
        } else {
            q
        }
    }

    /// The finest quantizer a key frame may take under
    /// [`Vp9Options::max_intra_pct`]: the step at which the key frame
    /// model (two passes: the first pass's bits of this frame, at the
    /// scale measured on key frames) spends at most that share of a
    /// frame's bits; the first key frame of one pass takes key frames to
    /// cost four inter frames at the same step.
    fn intra_cap(&self, seconds: f64) -> u8 {
        let cap =
            (f64::from(self.opts.bitrate) * seconds.max(1e-3) * f64::from(self.opts.max_intra_pct)
                / 100.0)
                .max(1.0);
        let bd = self.opts.bit_depth;
        let step = if let Some(p) = &self.pass2 {
            let b = p
                .first
                .get(self.frames as usize)
                .map_or(p.mean, |f| f.0)
                .max(1.0);
            let k = if p.pred[0] > 0.0 && p.spent[0] > 0.0 {
                p.spent[0] / p.pred[0]
            } else {
                K_START
            };
            step_of(p.q, bd) * (k * b / cap).powf(1.0 / EXP)
        } else {
            let m = self.model[0].unwrap_or(4.0 * FIRST_K * self.pixels);
            (m / cap).powf(1.0 / EXP)
        };
        q_of_step(step, bd)
    }

    /// The finest quantizer of a frame of type `kind`: `min_q`, or with
    /// [`Vp9Options::cq_level`] that level's step scaled as the frame
    /// type's.
    fn floor(&self, kind: usize) -> u8 {
        match self.opts.cq_level {
            None => self.opts.min_q,
            Some(cq) => {
                let bd = self.opts.bit_depth;
                let q = q_of_step(step_of(cq, bd) * STEP_SCALE[kind], bd);
                q.max(self.opts.min_q).min(self.opts.max_q)
            }
        }
    }

    fn qindex_in(&mut self, kind: usize, seconds: f64, cut: bool) -> u8 {
        let floor = [self.floor(0), self.floor(1), self.floor(2)];
        let o = self.opts;
        let banded = o.undershoot_pct > 0 || o.overshoot_pct > 0;
        let band = |diff: f64, budget: f64| band(diff, budget, o.undershoot_pct, o.overshoot_pct);
        if self.opts.bitrate == 0 {
            let q = self.opts.qindex;
            if q == 0 {
                return 0;
            }
            let scale = if kind == 1 && self.opts.speed <= Q_TUNED_SPEED {
                Q_GOLDEN_SCALE
            } else {
                STEP_SCALE[kind]
            };
            return q_of_step(step_of(q, self.opts.bit_depth) * scale, self.opts.bit_depth)
                .clamp(self.opts.min_q, q.max(self.opts.min_q));
        }
        if let Some(p) = &mut self.pass2 {
            // Two passes. Frame i's bits are about k·b_i·(s/s1)^-EXP, with
            // b_i its first-pass bits at step s1 and s the inter step it is
            // coded at; k, the two passes' scale, is measured on the frames
            // coded so far, for each class of picture (see [`Pass2`]). The
            // step chosen spends what is left of the budget on what is
            // left of the first pass; frame types take their steps from it
            // as a fixed quantizer does, and a cut a key frame's.
            let frames = p.first.len().max(1);
            let s1 = step_of(p.q, self.opts.bit_depth);
            let total = f64::from(self.opts.bitrate) * seconds.max(1e-3) * frames as f64;
            let done = (self.frames as usize).min(frames);
            let (spent, pred) = (p.spent.iter().sum::<f64>(), p.pred.iter().sum::<f64>());
            let prior = K_PRIOR_FRAMES * p.mean;
            let all = if done >= 2 && pred > 0.0 {
                ((spent + K_START * prior) / (pred + prior)).clamp(0.1, 10.0)
            } else {
                K_START
            };
            let k: [f64; 3] = std::array::from_fn(|c| {
                if done >= 2 && p.pred[c] > 0.0 && p.spent[c] > 0.0 {
                    ((p.spent[c] + all * prior) / (p.pred[c] + prior)).clamp(0.05, 20.0)
                } else {
                    all
                }
            });
            let more = (END_FRAMES - (frames - done) as f64).max(0.0);
            let need = k[0] * p.left[0] + k[1] * p.left[1] + k[2] * p.left[2] + all * p.mean * more;
            let spent_now = if banded {
                let budget = total * done as f64 / frames as f64;
                budget + band(self.spent - budget, budget)
            } else {
                self.spent
            };
            let left = (total - spent_now)
                .max(total * 0.02 * (frames - done) as f64 / frames as f64)
                + total / frames as f64 * more;
            let base = s1 * (need.max(1.0) / left.max(1.0)).powf(1.0 / EXP);
            p.base = base;
            let class = if cut { 0 } else { kind };
            let q = q_of_step(base * STEP_SCALE[class], self.opts.bit_depth)
                .clamp(floor[class], self.opts.max_q);
            if kind == 2 {
                self.last_q = q;
            }
            return q;
        }
        if self.per_frame == 0.0
            && let Some((key, inter)) = self.probed
        {
            // The first frame, after the probe: the inter step at which
            // the key frame and the inter frames after it spend
            // [`KEY_SPREAD`] frames' bits, the key frame at most
            // [`KEY_MOST`] frames' worth.
            let target = (f64::from(self.opts.bitrate) * seconds.max(1e-3)).max(1.0);
            let n = KEY_SPREAD;
            let step = ((key + (n - 1.0) * inter) / (n * target)).powf(1.0 / EXP);
            let bd = self.opts.bit_depth;
            self.last_q = q_of_step(step, bd).clamp(floor[2], self.opts.max_q);
            let own = (step * STEP_SCALE[kind]).max((key / (KEY_MOST * target)).powf(1.0 / EXP));
            return q_of_step(own, bd).clamp(floor[kind], self.opts.max_q);
        }
        if self.per_frame == 0.0 {
            // The first frame: the inter quantizer that a typical picture
            // would take at this many bits a sample, finer for its type.
            let target = f64::from(self.opts.bitrate) * seconds.max(1e-3) * SHARE[2];
            let step = (FIRST_K * self.pixels / target.max(1.0)).powf(1.0 / EXP);
            self.last_q = q_of_step(step, self.opts.bit_depth).clamp(floor[2], self.opts.max_q);
            // At a constant bitrate the first key frame takes a coarser
            // step, not a finer one: the buffer cannot hold the ten frames'
            // worth a fine one costs.
            let scale = if self.opts.cbr_buffer_ms > 0 {
                CBR_FIRST_SCALE
            } else {
                STEP_SCALE[kind]
            };
            return q_of_step(step * scale, self.opts.bit_depth)
                .clamp(floor[kind], self.opts.max_q);
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
        if !cbr {
            // At a variable bitrate the frame types' steps are the inter
            // frames' scaled (as with a fixed quantizer and in two passes),
            // and the inter step is the one at which the frames to come,
            // golden ones among them, spend the bitrate: frame types
            // budgeted apart left alt-refs coarser than the frames that
            // predict from them (on a grainy pan 2 dB of PSNR at 1 Mb/s).
            let bd = self.opts.bit_depth;
            if kind != 2 {
                let arf = std::mem::take(&mut self.arf_next);
                // A golden frame that is no alt-ref (no lookahead, or none
                // reached its end) takes a key frame's step: it is what the
                // frames up to the next one predict from (9% fewer bits on
                // a grainy pan without lookahead, 19% at speed 10).
                let class = if cut || (kind == 1 && !arf) { 0 } else { kind };
                return q_of_step(step_of(self.last_q, bd) * STEP_SCALE[class], bd)
                    .clamp(floor[class], self.opts.max_q);
            }
            let golden = self.model[1]
                .or(self.model[0].map(|k| k * 0.5))
                .unwrap_or(m * 3.0);
            let w = 1.0 / f64::from(GF_INTERVAL);
            // The golden frames to come at the step they take: at
            // [`GOLDEN_KEY_SPEED`], with no lookahead and so no alt-refs,
            // a key frame's.
            let gold_scale = if self.opts.speed == GOLDEN_KEY_SPEED && self.opts.lag() == 0 {
                STEP_SCALE[0]
            } else {
                STEP_SCALE[1]
            };
            let unit = match (self.model[2], self.probed) {
                (None, Some((_, inter))) => inter,
                _ => (1.0 - w) * m + w * golden * gold_scale.powf(-EXP),
            };
            let mut debt = if banded {
                band(self.debt, self.per_frame * f64::from(self.frames))
            } else {
                self.debt
            };
            // What the last key frame took past a frame's share falls due
            // over the [`KEY_SPREAD`] frames after it, not at once.
            let (extra, since) = self.key_extra;
            debt -= extra * (1.0 - since / KEY_SPREAD).max(0.0);
            let target = (self.per_frame - debt / horizon).max(self.per_frame * 0.2);
            let step = (unit / target).powf(1.0 / EXP);
            let q = q_of_step(step, bd).clamp(floor[2], self.opts.max_q).clamp(
                self.last_q.saturating_sub(24),
                self.last_q.saturating_add(24),
            );
            self.last_q = q;
            return q;
        }
        let share = if cbr { SHARE_CBR[kind] } else { SHARE[kind] };
        let debt = if banded {
            band(self.debt, self.per_frame * f64::from(self.frames))
        } else {
            self.debt
        };
        let target = (self.per_frame * share - debt / horizon).max(self.per_frame * 0.2);
        let step = (m / target).powf(1.0 / EXP);
        let q = q_of_step(step, self.opts.bit_depth).clamp(floor[kind], self.opts.max_q);
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

    /// Whether the first picture waits for [`Core::probe`].
    fn wants_probe(&self) -> bool {
        self.opts.bitrate > 0
            && self.opts.cbr_buffer_ms == 0
            && self.pass2.is_none()
            && self.per_frame == 0.0
    }

    /// The bits of the lookahead's pictures coded fast at quantizer `q`
    /// (a key frame, then inter frames): the models of the key frame and
    /// of an inter frame, at the scale between such a pass and the encode
    /// (as two passes start, [`K_START`]).
    fn probe(&mut self, bits: &[f64], q: u8) {
        let Some((&key, inter)) = bits.split_first() else {
            return;
        };
        if inter.is_empty() {
            return;
        }
        let mean = inter.iter().sum::<f64>() / inter.len() as f64;
        let s = K_START * step_of(q, self.opts.bit_depth).powf(EXP);
        self.probed = Some((s * key, s * mean));
    }

    /// See [`Vp9Encoder::first_pass_qindex`]: the inter quantizer the
    /// one-pass model guesses for the bitrate.
    fn first_pass_qindex(&self, seconds: f64) -> u8 {
        if self.opts.bitrate == 0 {
            return self.opts.qindex;
        }
        let target = f64::from(self.opts.bitrate) * seconds.max(1e-3);
        let step = (FIRST_K * self.pixels / target.max(1.0)).powf(1.0 / EXP);
        q_of_step(step, self.opts.bit_depth).clamp(FIRST_PASS_Q.0, FIRST_PASS_Q.1)
    }

    /// The next golden place is `ahead` pictures after the one being
    /// coded.
    fn golden_at(&mut self, ahead: usize) {
        let at = self.frames as usize + ahead;
        if let Some(p) = &mut self.pass2 {
            p.golden_at(at, ahead);
        }
    }

    /// An alt-ref is coded now for the picture `ahead` after the next
    /// shown one.
    fn alt_ref(&mut self, ahead: usize) {
        self.arf_next = true;
        let at = self.frames as usize + ahead;
        if let Some(p) = &mut self.pass2 {
            p.alt_ref(at);
        }
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
        if let Some(p) = &mut self.pass2 {
            let at = (p.base.max(1e-3) / step_of(p.q, self.opts.bit_depth)).powf(-EXP);
            let (b, c) = if seconds == 0.0 {
                // An alt-ref: the bits it took from its target.
                std::mem::replace(&mut p.arf, (0.0, 2))
            } else {
                p.first
                    .get(self.frames as usize)
                    .copied()
                    .unwrap_or((0.0, 0))
            };
            p.spent[c] += bits;
            p.pred[c] += b * at;
            p.left[c] -= b;
        }
        if seconds > 0.0 {
            // A shown picture: its budget; a hidden one only spends.
            self.frames = self.frames.saturating_add(1);
            self.per_frame = f64::from(self.opts.bitrate) * seconds;
            self.debt -= self.per_frame;
        }
        self.debt += bits;
        if kind == 0 && seconds > 0.0 {
            self.key_extra = ((bits - self.per_frame).max(0.0), 0.0);
        } else if seconds > 0.0 {
            self.key_extra.1 += 1.0;
        }
        let m = bits * step_of(qindex, self.opts.bit_depth).powf(EXP);
        // (The probe's inter model counts as [`PROBE_WEIGHT`] frames.)
        let prior = (kind == 2).then_some(self.probed.map(|p| p.1)).flatten();
        let first = self.probed.map_or(0.0, |_| PROBE_WEIGHT);
        self.model[kind] = Some(match self.model[kind].or(prior) {
            Some(old) => {
                // One pass at a variable bitrate: the inter frames' model
                // moves slowly once it has settled, so that the quantizer
                // holds steady across content that costs less or more for a
                // while (a scene of its own) instead of spending each
                // picture's share on it; the debt still pays back the
                // difference.
                let a = if kind == 2 && self.opts.cbr_buffer_ms == 0 {
                    INTER_MODEL_WEIGHT.max(1.0 / (f64::from(self.frames.max(1)) + first))
                } else {
                    0.4
                };
                old * (1.0 - a) + m * a
            }
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

/// `diff` bits over (positive) or under the target `budget` so far, less
/// what [`Vp9Options::overshoot_pct`] and `undershoot_pct` (`over` and
/// `under`) let stand.
fn band(diff: f64, budget: f64, under: u32, over: u32) -> f64 {
    if diff < 0.0 {
        (diff + budget * f64::from(under) / 100.0).min(0.0)
    } else {
        (diff - budget * f64::from(over) / 100.0).max(0.0)
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
