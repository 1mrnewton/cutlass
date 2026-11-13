//! Frame-level encoding: headers, tiles, packets and references.
//!
//! Every frame is coded by the decoder's own tile code running with a writing
//! [`Coder`], so its reconstruction is the decoder's. The in-loop filters of
//! the final header are applied to it and the result is installed as the
//! reference in an internal [`Av1Core`] (which decodes the other headers),
//! exactly as any decoder will make it.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::dec::cdf::CdfContext;
use crate::dec::coder::Coder;
use crate::dec::decoder::{AnyPicture, Av1Core, PicturePixel};
use crate::dec::frame::{FrameState, Picture};
use crate::dec::obu::{
    Cdef, FrameHeader, HeaderContext, INTER_FRAME, KEY_FRAME, LoopFilter, OBU_FRAME,
    OBU_FRAME_HEADER, OBU_SEQUENCE_HEADER, OBU_TEMPORAL_DELIMITER, OperatingPoint,
    PRIMARY_REF_NONE, Quant, SequenceHeader, TX_MODE_SELECT, parse_frame_header,
};
use crate::dsp::Pixel;
use crate::spec::tables::{AC_QLOOKUP, DC_QLOOKUP};

use super::bitwriter::write_obu;
use super::msac_enc::MsacEnc;
use super::obu_write::{
    FrameHeaderWriteOptions, explicit_tile_info, tile_limits, uniform_tile_info,
    write_frame_header, write_sequence_header, write_tile_group,
};
use super::quant::QuantParams;
use super::rdo::ModeParams;

type R<T> = Result<T, CodecError>;

impl Coder for MsacEnc {
    const WRITE: bool = true;
    #[inline(always)]
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], v: bool) -> bool {
        MsacEnc::bool_adapt(self, cdf, v);
        v
    }
    #[inline(always)]
    fn bool_prob(&mut self, f: u32, v: bool) -> bool {
        MsacEnc::bool_prob(self, f, v);
        v
    }
    #[inline(always)]
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], v: usize) -> usize {
        let v = v.min(N - 1);
        MsacEnc::symbol(self, cdf, v);
        v
    }
    #[inline(always)]
    fn literal(&mut self, n: u32, v: u32) -> u32 {
        let v = if n >= 32 { v } else { v & ((1u32 << n) - 1) };
        MsacEnc::literal(self, n, v);
        v
    }
    #[inline(always)]
    fn golomb(&mut self, v: u32) -> u32 {
        MsacEnc::golomb(self, v);
        v
    }
    #[inline(always)]
    fn ns(&mut self, n: u32, v: u32) -> u32 {
        let v = if n == 0 { 0 } else { v.min(n - 1) };
        MsacEnc::ns(self, n, v);
        v
    }
    #[inline(always)]
    fn mark(&mut self, tag: u32) {
        MsacEnc::mark(self, tag);
    }
    fn checkpoint(
        &self,
        base: Option<&super::msac_enc::CoderMark>,
    ) -> Option<super::msac_enc::CoderMark> {
        let base = match base {
            Some(super::msac_enc::CoderMark::Enc(b)) => Some(b),
            Some(_) => return None,
            None => None,
        };
        Some(super::msac_enc::CoderMark::Enc(MsacEnc::checkpoint(
            self, base,
        )))
    }
    fn restore(&mut self, s: &super::msac_enc::CoderMark) {
        if let super::msac_enc::CoderMark::Enc(s) = s {
            MsacEnc::restore(self, s);
        }
    }
    fn bits_written(&self) -> f64 {
        self.tell_bits() as f64
    }
}

/// How the encoder chooses quantizers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateControl {
    /// A fixed base_q_idx (0..=255) for every frame.
    Qindex(u8),
    /// Target bits per second at the given frame rate.
    Bitrate { bps: u64, fps: f64 },
}

#[derive(Clone, Debug)]
pub struct EncConfig {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub sub_x: bool,
    pub sub_y: bool,
    pub mono: bool,
    /// 0 is slowest and best, 10 fastest.
    pub speed: u8,
    pub rc: RateControl,
    /// With a bitrate target: hold it over each second, so that a second
    /// that spends less leaves nothing for a later one. Off, the target is
    /// the average over the whole stream.
    pub constant_rate: bool,
    /// Pictures per second, for the level the sequence header names. A
    /// bitrate target carries its own rate, which is used instead.
    pub fps: f64,
    /// Distance between key frames.
    pub keyint: u32,
    /// Start a key frame where the picture changes all at once (a cut),
    /// no closer to the key frame before it than a mini-GOP or a tenth of
    /// `keyint`.
    pub scene_cut: bool,
    /// Mini-GOP length: frames coded as a hierarchical pyramid behind a
    /// hidden alt-ref (1 = each frame predicts from the previous one only).
    pub gop: u32,
    /// Threads for tile-parallel coding and the internal decoder.
    pub threads: usize,
    /// Denoise alt-refs with their neighbours (coded with an overlay frame).
    pub temporal_filter: bool,
    /// Doubles or quadruples the mini-GOP where the content barely changes.
    pub adaptive_gop: bool,
    /// Wiener loop restoration per 64x64 unit (experimental: about 1% on
    /// a key frame, but it costs the natural test clips about as much).
    pub loop_restoration: bool,
    /// Estimate a global motion model per reference (experimental: it does
    /// not pay off on the natural test clip yet).
    pub global_motion: bool,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub color_primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
}

impl EncConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            sub_x: true,
            sub_y: true,
            mono: false,
            speed: 6,
            rc: RateControl::Qindex(120),
            constant_rate: false,
            fps: 30.0,
            keyint: 120,
            scene_cut: true,
            gop: 8,
            adaptive_gop: true,
            loop_restoration: false,
            temporal_filter: false,
            threads: 1,
            global_motion: false,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            color_primaries: 2,
            transfer: 2,
            matrix: 2,
            full_range: false,
        }
    }
}

/// One input picture, as plane slices with strides.
pub struct InputFrame<'a, P> {
    pub planes: [&'a [P]; 3],
    pub strides: [usize; 3],
}

/// The AV1 encoder.
pub struct Av1Enc {
    cfg: EncConfig,
    seq: SequenceHeader,
    core: Av1Core,
    frame_num: u64,
    rc_q: f64,
    /// Bits produced so far and frames coded, for rate control.
    bits_total: u64,
    /// Bits and pictures of the second being coded, at a constant rate.
    second_bits: u64,
    second_frames: u64,
    /// Last reconstructed frame (shown output of the internal decoder).
    last_recon: Option<Arc<AnyPicture>>,
    /// Source pictures waiting to be coded, in display order from `next_disp`.
    lookahead: std::collections::VecDeque<Arc<AnyPicture>>,
    next_disp: u64,
    /// Display index of the next key frame.
    next_key: u64,
    /// Display indices a caller asked to be key frames, in order, all past
    /// `next_disp` or equal to it.
    forced_keys: std::collections::VecDeque<u64>,
    /// The last picture given to `encode`, for the cut detector.
    last_input: Option<Arc<AnyPicture>>,
    /// How much each of the last pictures differed from the one before.
    changes: std::collections::VecDeque<f64>,
    /// Display index of the last key frame started, by distance or by a
    /// cut or a request.
    last_key_at: u64,
    /// Display index of the frame each reference slot holds.
    slot_disp: [Option<u64>; 8],
    /// Temporal importance of the frames of the current mini-GOP (display
    /// index, importance), from the TPL model.
    tpl_imp: Vec<(u64, f64)>,
    /// The current mini-GOP was found static (and the content is not screen
    /// content): its alt-ref is temporally filtered.
    static_gop: bool,
    /// The frame being coded is screen content.
    screen: bool,
    /// The model's block statistics and the display index of each of its
    /// frames, for per-superblock quantizers.
    tpl: Option<TplGroup>,
    /// The tile count was chosen for the threads: the tile boundaries
    /// follow `sb_work`, the coding work of the last frame's superblocks.
    auto_tiles: bool,
    /// The alt-ref of a still mini-GOP may be filtered: the hidden frame
    /// that leaves stays inside the decode rate of the stream's level.
    still_filter: bool,
    sb_work: Vec<u32>,
    /// Loop filter level and CDEF strengths (y pri, y sec, uv pri, uv sec)
    /// the last frame nothing predicts from chose, until the next such
    /// frame takes them.
    leaf_filters: Option<(u8, [u8; 4])>,
}

/// log2 of the delta q step of per-superblock quantizers.
const DELTA_Q_RES: u8 = 2;

/// TPL block statistics and the display index of each frame they cover.
type TplGroup = (super::tpl::TplStats, Vec<u64>);

/// Threads per automatic tile. Superblock rows run in a wavefront inside
/// each tile, so fewer tiles than threads still keep the threads busy, but
/// a frame whose work sits in a few superblock rows parallelizes better
/// across tiles. One tile per thread is the fastest at 1080p; a single
/// tile (any value above the thread count) costs about 2% less rate there
/// but takes about 10% longer on content whose motion is concentrated.
const TILE_THREADS: usize = 1;

/// A global motion type and its parameters.
type GmModel = (u8, [i32; 6]);

/// What a decoder makes of a coded frame, for [`Av1Core::install_frame`].
struct FrameRecon {
    hdr: FrameHeader,
    fs: FrameState,
    pic: AnyPicture,
    cdf: Box<CdfContext>,
}

/// One temporal unit: the packet and the reconstruction of its shown frame.
pub struct EncodedUnit {
    pub data: Vec<u8>,
    pub recon: Option<Arc<AnyPicture>>,
    pub key: bool,
    /// Display index of the shown frame.
    pub display: u64,
}

/// One step of a mini-GOP in coding order.
#[derive(Clone, Copy, Debug)]
enum GopStep {
    /// Code the frame at a display index; `level` 0 is the top of the pyramid.
    Code {
        disp: u64,
        show: bool,
        level: u32,
    },
    ShowExisting {
        disp: u64,
    },
}

/// Coding order of frames `lo + 1 ..= hi` where `lo` is already coded and `hi`
/// has been coded (hidden): every frame comes out in display order.
fn gop_inner(lo: u64, hi: u64, level: u32, out: &mut Vec<GopStep>) {
    if hi - lo == 1 {
        out.push(GopStep::ShowExisting { disp: hi });
        return;
    }
    let mid = (lo + hi) / 2;
    if mid - lo == 1 {
        out.push(GopStep::Code {
            disp: mid,
            show: true,
            level: u32::MAX,
        });
    } else {
        out.push(GopStep::Code {
            disp: mid,
            show: false,
            level,
        });
        gop_inner(lo, mid, level + 1, out);
    }
    gop_inner(mid, hi, level + 1, out);
}

/// Per-frame coding parameters.
struct FrameSpec {
    disp: u64,
    key: bool,
    show: bool,
    q: u8,
    ref_frame_idx: [u8; 7],
    refresh: u8,
    reference_select: bool,
    /// Pyramid level (0 top, `u32::MAX` for frames nothing predicts from).
    level: u32,
}

fn bits_for(v: usize) -> u32 {
    let mut n = 1;
    while (1usize << n) < v {
        n += 1;
    }
    n
}

impl Av1Enc {
    pub fn new(cfg: EncConfig) -> R<Self> {
        if cfg.width == 0 || cfg.height == 0 || cfg.width > 16384 || cfg.height > 16384 {
            return Err(CodecError::Unsupported { what: "frame size" });
        }
        if !matches!(cfg.bit_depth, 8 | 10 | 12) {
            return Err(CodecError::Unsupported { what: "bit depth" });
        }
        let profile = if cfg.bit_depth == 12 || (cfg.sub_x && !cfg.sub_y && !cfg.mono) {
            2
        } else if !cfg.sub_x && !cfg.sub_y && !cfg.mono {
            1
        } else {
            0
        };
        let (sub_x, sub_y) = if cfg.mono {
            (true, true)
        } else {
            (cfg.sub_x, cfg.sub_y)
        };
        let mut seq = SequenceHeader {
            profile,
            operating_points: vec![OperatingPoint::default()],
            frame_width_bits: bits_for(cfg.width),
            frame_height_bits: bits_for(cfg.height),
            max_frame_width: cfg.width as u32,
            max_frame_height: cfg.height as u32,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_order_hint: true,
            order_hint_bits: 7,
            seq_force_screen_content_tools: 2,
            seq_force_integer_mv: 2,
            enable_cdef: true,
            enable_ref_frame_mvs: true,
            enable_warped_motion: true,
            enable_dual_filter: ev("AV1_DUAL", 0.0) > 0.0,
            enable_restoration: cfg.loop_restoration && cfg.speed < 8,
            bit_depth: cfg.bit_depth,
            mono_chrome: cfg.mono,
            color_primaries: cfg.color_primaries,
            transfer_characteristics: cfg.transfer,
            matrix_coefficients: cfg.matrix,
            color_range: cfg.full_range,
            subsampling_x: sub_x,
            subsampling_y: sub_y,
            ..SequenceHeader::default()
        };
        let mut cfg = cfg;
        let mut auto_tiles = false;
        // Tiles per thread (their superblock rows are coded side by side
        // too, see `rowmt`).
        let tiles = cfg.threads / TILE_THREADS;
        if tiles >= 2 && cfg.tile_cols_log2 == 0 && cfg.tile_rows_log2 == 0 && cfg.width >= 1280 {
            // Columns at least 512 samples wide, then rows at least 512
            // samples high.
            let mut l = 0;
            while (2usize << l) <= tiles && (cfg.width >> (l + 1)) >= 512 && l < 6 {
                l += 1;
            }
            cfg.tile_cols_log2 = l;
            let mut r = 0;
            while (2usize << (l + r)) <= tiles && (cfg.height >> (r + 1)) >= 512 && r < 6 {
                r += 1;
            }
            cfg.tile_rows_log2 = r;
            auto_tiles = l + r > 0 && !seq.use_128x128_superblock;
        }
        // The level the stream fits, now that its tiles are known. A
        // filtered alt-ref is hidden for good (an overlay frame is shown in
        // its place), so it is decoded besides the pictures shown: one in
        // every mini-GOP when the filter is asked for.
        let (fps, bitrate) = match cfg.rc {
            RateControl::Qindex(_) => (cfg.fps, None),
            RateControl::Bitrate { bps, fps } => (fps, Some(bps)),
        };
        let fps = if fps > 0.0 { fps } else { 30.0 };
        let gop = f64::from(cfg.gop.max(1));
        let level_with = |decoded_per_shown: f64| {
            super::level::pick(&super::level::Stream {
                width: cfg.width,
                height: cfg.height,
                shown_per_second: fps,
                decoded_per_second: fps * decoded_per_shown,
                bitrate,
                profile,
                tiles: 1usize << (cfg.tile_cols_log2 + cfg.tile_rows_log2),
                tile_cols: 1usize << cfg.tile_cols_log2,
            })
        };
        let asked = cfg.gop > 1 && cfg.temporal_filter;
        let (level, tier) = level_with(if asked { (gop + 1.0) / gop } else { 1.0 });
        // Unasked, the alt-ref of a still mini-GOP is filtered too, and such
        // a mini-GOP is longer than `gop`. Where those hidden frames would
        // pass the decode rate of the level the picture and its rate fit
        // (4K at 30 and 60 pictures a second), they are left unfiltered and
        // the level stays.
        let still_filter = asked || level_with((gop + 2.0) / (gop + 1.0)) == (level, tier);
        if let Some(op) = seq.operating_points.first_mut() {
            op.seq_level_idx = level;
            op.seq_tier = tier;
        }
        let threads = cfg.threads.max(1);
        let q0 = match cfg.rc {
            RateControl::Qindex(q) => f64::from(q),
            RateControl::Bitrate { bps, fps } => {
                // Starting point from the bits per sample (fitted on natural video).
                let bpp = bps as f64 / fps.max(1.0) / (cfg.width * cfg.height) as f64;
                (157.0 - 58.0 * (bpp.max(1e-4) / 0.14).log2()).clamp(20.0, 240.0)
            }
        };
        Ok(Self {
            cfg,
            seq,
            core: {
                let mut c = Av1Core::new();
                c.threads = threads;
                c
            },
            frame_num: 0,
            rc_q: q0,
            bits_total: 0,
            second_bits: 0,
            second_frames: 0,
            last_recon: None,
            lookahead: std::collections::VecDeque::new(),
            next_disp: 0,
            next_key: 0,
            forced_keys: std::collections::VecDeque::new(),
            last_input: None,
            changes: std::collections::VecDeque::new(),
            last_key_at: 0,
            slot_disp: [None; 8],
            tpl_imp: Vec::new(),
            static_gop: false,
            screen: false,
            tpl: None,
            auto_tiles,
            still_filter,
            sb_work: Vec::new(),
            leaf_filters: None,
        })
    }

    pub fn sequence_header(&self) -> &SequenceHeader {
        &self.seq
    }

    /// The `av1C` record of this stream (AV1 Codec ISO Media File Format
    /// Binding 2.3): the marker and version, the profile, level, tier, bit
    /// depth and chroma layout of the sequence header, no presentation
    /// delay, and the sequence header OBU each key frame also carries. An
    /// ISOBMFF sample entry and a Matroska `CodecPrivate` hold these bytes.
    pub fn config_record(&self) -> Vec<u8> {
        let s = &self.seq;
        let (level, tier) = s
            .operating_points
            .first()
            .map_or((0, 0), |op| (op.seq_level_idx, op.seq_tier));
        // chroma_sample_position is in the sequence header for 4:2:0 only.
        let position = if s.subsampling_x && s.subsampling_y && !s.mono_chrome {
            s.chroma_sample_position & 3
        } else {
            0
        };
        let mut out = vec![
            0x81,
            (s.profile << 5) | (level & 31),
            ((tier & 1) << 7)
                | (u8::from(s.bit_depth > 8) << 6)
                | (u8::from(s.bit_depth == 12) << 5)
                | (u8::from(s.mono_chrome) << 4)
                | (u8::from(s.subsampling_x) << 3)
                | (u8::from(s.subsampling_y) << 2)
                | position,
            0,
        ];
        let sh = write_sequence_header(s);
        write_obu(&mut out, OBU_SEQUENCE_HEADER, 0, 0, false, &sh);
        out
    }

    /// Makes the next picture given to [`Self::encode`] a key frame. The
    /// distance to the following key frame is counted from it.
    pub fn force_keyframe(&mut self) {
        let at = self.next_disp + self.lookahead.len() as u64;
        if self.forced_keys.back() != Some(&at) {
            self.forced_keys.push_back(at);
        }
        self.next_key = self.next_key.min(at);
    }

    /// Display index of the key frame that follows the key frame at `d`.
    fn key_after(&self, d: u64) -> u64 {
        let periodic = d + u64::from(self.cfg.keyint.max(1));
        self.forced_keys
            .iter()
            .find(|&&f| f > d)
            .map_or(periodic, |&f| f.min(periodic))
    }

    /// The reconstruction of the last coded frame.
    pub fn last_recon(&self) -> Option<&Arc<AnyPicture>> {
        self.last_recon.as_ref()
    }

    /// Queues one picture and returns the temporal units that are ready
    /// (frames are reordered within a mini-GOP, so this may be none).
    pub fn encode<P: PicturePixel>(&mut self, input: &InputFrame<'_, P>) -> R<Vec<EncodedUnit>> {
        if P::HBD != (self.cfg.bit_depth > 8) {
            return Err(CodecError::InvalidApi {
                what: "sample size",
            });
        }
        let src = Arc::new(P::wrap(source_picture(&self.cfg, &self.seq, input)));
        if self.cfg.scene_cut {
            self.watch_for_a_cut::<P>(&src);
        }
        self.last_input = Some(Arc::clone(&src));
        self.lookahead.push_back(src);
        self.run::<P>(false)
    }

    /// Makes `picture`, the next one queued, a key frame when it differs
    /// from the picture before it all at once: by more than a twentieth
    /// of the sample range on average, and by three times what the last
    /// pictures changed. Steady motion, noise and a fade change every
    /// picture alike and start none.
    fn watch_for_a_cut<P: PicturePixel>(&mut self, picture: &Arc<AnyPicture>) {
        let at = self.next_disp + self.lookahead.len() as u64;
        let change = self
            .last_input
            .as_ref()
            .and_then(|last| P::view(last).zip(P::view(picture)))
            .map(|(a, b)| mean_change(a, b));
        let Some(change) = change else {
            return;
        };
        let usual = if self.changes.is_empty() {
            None
        } else {
            Some(self.changes.iter().sum::<f64>() / self.changes.len() as f64)
        };
        if self.changes.len() == 4 {
            self.changes.pop_front();
        }
        self.changes.push_back(change);
        let Some(usual) = usual else {
            return;
        };
        let range = f64::from(1u32 << self.cfg.bit_depth);
        let near = u64::from(self.cfg.gop.max(self.cfg.keyint / 10).max(1));
        let pending = self.forced_keys.back().copied().unwrap_or(self.last_key_at);
        if change > range / 20.0
            && change > 3.0 * usual + range / 256.0
            && at >= pending.max(self.last_key_at) + near
            && self.next_key > at
        {
            self.forced_keys.push_back(at);
            self.next_key = at;
        }
    }

    /// Codes every queued picture.
    pub fn flush<P: PicturePixel>(&mut self) -> R<Vec<EncodedUnit>> {
        self.run::<P>(true)
    }

    fn use_tf(&self) -> bool {
        self.cfg.gop > 1
            && (self.cfg.temporal_filter
                || (self.static_gop && self.still_filter && self.cfg.speed < 8))
    }

    fn is_key_disp(&self, d: u64) -> bool {
        d == self.next_key
    }

    fn run<P: PicturePixel>(&mut self, flush: bool) -> R<Vec<EncodedUnit>> {
        let mut units = Vec::new();
        while !self.lookahead.is_empty() {
            let d = self.next_disp;
            if self.is_key_disp(d) {
                // The key frame's quantizer follows how much of the next
                // mini-GOP is predicted from it.
                let after = self.key_after(d);
                // (A single-frame GOP codes every picture without delay.)
                let want = if self.cfg.gop > 1 {
                    u64::from(self.cfg.gop).min(after - d - 1) + 1
                } else {
                    1
                };
                if (self.lookahead.len() as u64) < want && !flush {
                    break;
                }
                let n = (want - 1).min(self.lookahead.len() as u64 - 1);
                (self.tpl_imp, self.tpl) = if n > 0 && ev("AV1_TPLK", 1.0) > 0.0 {
                    let steps = self.gop_steps(d, n);
                    self.run_tpl::<P>(d, &steps, true)
                } else {
                    (Vec::new(), None)
                };
                let mut tu = Vec::new();
                let shown = self.code_step::<P>(
                    &mut tu,
                    GopStep::Code {
                        disp: d,
                        show: true,
                        level: 0,
                    },
                )?;
                let bits = tu.len() as u64 * 8;
                units.push(self.finish_unit(tu, shown, true, d));
                self.update_rate_control(bits, 1, true);
                self.lookahead.pop_front();
                self.next_disp += 1;
                self.next_key = after;
                self.last_key_at = d;
                while self.forced_keys.front().is_some_and(|&f| f <= d) {
                    self.forced_keys.pop_front();
                }
                continue;
            }
            let gop = u64::from(self.cfg.gop.max(1));
            let to_key = self.next_key - d;
            // Static content takes mini-GOPs two or four times as long.
            let longest = if self.cfg.adaptive_gop && gop >= 4 {
                gop << ev("AV1_GOPL", 2.0) as u32
            } else {
                gop
            };
            if (self.lookahead.len() as u64) < longest.min(to_key) && !flush {
                break;
            }
            let mut n = gop.min(to_key).min(self.lookahead.len() as u64);
            let mut long = longest;
            self.static_gop = false;
            while long > gop {
                let ln = long.min(to_key).min(self.lookahead.len() as u64);
                if ln > gop && self.is_static::<P>(d - 1, ln) {
                    n = ln;
                    // (Screen content redraws what it changes: filtering
                    // across frames would blend that.)
                    self.static_gop = !self.screen;
                    break;
                }
                long /= 2;
            }
            // Frames d .. d + n - 1 follow the anchor d - 1.
            let lo = d - 1;
            let steps = self.gop_steps(lo, n);
            (self.tpl_imp, self.tpl) = if ev("AV1_TPL", 1.0) > 0.0 {
                self.run_tpl::<P>(lo, &steps, false)
            } else {
                (Vec::new(), None)
            };
            let mut tu = Vec::new();
            let first_unit = units.len();
            for step in steps {
                if let Some(shown) = self.code_step::<P>(&mut tu, step)? {
                    let disp = match step {
                        GopStep::Code { disp, .. } | GopStep::ShowExisting { disp } => disp,
                    };
                    units.push(self.finish_unit(std::mem::take(&mut tu), Some(shown), false, disp));
                }
            }
            let gop_bits: u64 = units[first_unit..]
                .iter()
                .map(|u| u.data.len() as u64 * 8)
                .sum();
            self.update_rate_control(gop_bits, n, false);
            for _ in 0..n {
                self.lookahead.pop_front();
            }
            self.next_disp += n;
        }
        Ok(units)
    }

    /// True when frame `lo + len` is predicted well from the anchor `lo`.
    fn is_static<P: PicturePixel>(&self, lo: u64, len: u64) -> bool {
        let anchor = self
            .slot_disp
            .iter()
            .position(|s| *s == Some(lo))
            .and_then(|slot| self.core.slot_picture(slot));
        let (Some(anchor), Some(target)) = (anchor, self.lookahead.get(len as usize - 1)) else {
            return false;
        };
        let (Some(a), Some(t)) = (P::view(&anchor), P::view(target)) else {
            return false;
        };
        let frames = [
            super::tpl::TplFrame {
                pic: a,
                disp: 0,
                refs: Vec::new(),
                analyze: false,
            },
            super::tpl::TplFrame {
                pic: t,
                disp: len as i64,
                refs: vec![0],
                analyze: true,
            },
        ];
        let stats = super::tpl::analyze(&frames, &[1], self.cfg.bit_depth, self.cfg.threads);
        let r = stats.inter_ratio(1);
        r < ev("AV1_GOPT", 0.5)
    }

    /// Coding order of the `n` frames that follow the anchor `lo`.
    fn gop_steps(&self, lo: u64, n: u64) -> Vec<GopStep> {
        let hi = lo + n;
        let mut steps = Vec::new();
        {
            if n == 1 {
                steps.push(GopStep::Code {
                    disp: hi,
                    show: true,
                    level: u32::MAX,
                });
            } else {
                steps.push(GopStep::Code {
                    disp: hi,
                    show: false,
                    level: 0,
                });
                gop_inner(lo, hi, 1, &mut steps);
                if self.use_tf() {
                    // The filtered alt-ref is not shown; an overlay frame
                    // codes the real picture from it.
                    for st in steps.iter_mut() {
                        if let GopStep::ShowExisting { disp } = *st
                            && disp == hi
                        {
                            *st = GopStep::Code {
                                disp: hi,
                                show: true,
                                level: u32::MAX,
                            };
                        }
                    }
                }
            }
        }
        steps
    }

    fn finish_unit(
        &mut self,
        data: Vec<u8>,
        shown: Option<Arc<AnyPicture>>,
        key: bool,
        display: u64,
    ) -> EncodedUnit {
        self.last_recon = shown.clone();
        self.frame_num += 1;
        self.bits_total += data.len() as u64 * 8;
        EncodedUnit {
            data,
            recon: shown,
            key,
            display,
        }
    }

    /// Codes one step into `tu` and decodes it; returns the shown picture, if any.
    fn code_step<P: PicturePixel>(
        &mut self,
        tu: &mut Vec<u8>,
        step: GopStep,
    ) -> R<Option<Arc<AnyPicture>>> {
        let start = tu.len();
        if tu.is_empty() {
            write_obu(tu, OBU_TEMPORAL_DELIMITER, 0, 0, false, &[]);
        }
        match step {
            GopStep::ShowExisting { disp } => {
                let Some(slot) = self.slot_disp.iter().position(|s| *s == Some(disp)) else {
                    return Err(CodecError::InvalidApi {
                        what: "frame to show",
                    });
                };
                // show_existing_frame = 1, frame_to_show_map_idx (3 bits), then
                // trailing bits (the sequence has no frame ids or decoder model).
                let bytes = [0x80 | ((slot as u8) << 4) | 0x08];
                write_obu(tu, OBU_FRAME_HEADER, 0, 0, false, &bytes);
            }
            GopStep::Code { disp, show, level } => {
                let key = self.is_key_disp(disp);
                if key {
                    let sh = write_sequence_header(&self.seq);
                    write_obu(tu, OBU_SEQUENCE_HEADER, 0, 0, false, &sh);
                }
                let idx = (disp - self.next_disp) as usize;
                let Some(src) = self.lookahead.get(idx).cloned() else {
                    return Err(CodecError::InvalidApi { what: "lookahead" });
                };
                let Some(src) = P::view(&src) else {
                    return Err(CodecError::InvalidApi {
                        what: "sample size",
                    });
                };
                self.screen = self.cfg.speed < 10 && is_screen_content(src, self.cfg.bit_depth);
                let spec = self.plan_frame(disp, key, show, level);
                let filtered;
                let mut src = src;
                if !show && level == 0 && self.use_tf() {
                    // Denoise the alt-ref with its neighbours in the lookahead.
                    let mut others = Vec::new();
                    let tn = ev("AV1_TFN", 2.0) as usize;
                    for k in idx.saturating_sub(tn)..=idx + tn {
                        if k == idx {
                            continue;
                        }
                        if let Some(o) = self.lookahead.get(k).and_then(|a| P::view(a)) {
                            others.push((o, k.abs_diff(idx) as u32));
                        }
                    }
                    let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
                    let qstep = f64::from(AC_QLOOKUP[bdi][spec.q as usize]) / 8.0;
                    filtered = super::tf::temporal_filter(src, &others, qstep, self.cfg.bit_depth);
                    src = &filtered;
                }
                let (frame, recon) = self.encode_frame(src, &spec)?;
                for (i, s) in self.slot_disp.iter_mut().enumerate() {
                    if spec.refresh & (1 << i) != 0 {
                        *s = Some(disp);
                    }
                }
                if let Some(r) = recon {
                    // The decoder takes the headers before the frame, then
                    // the encoder's own reconstruction of the frame.
                    self.core.decode(&tu[start..])?;
                    write_obu(tu, OBU_FRAME, 0, 0, false, &frame);
                    let shown = self.core.install_frame(&r.hdr, &r.fs, r.pic, r.cdf)?;
                    return Ok(shown.into_iter().last().map(|f| f.picture));
                }
                write_obu(tu, OBU_FRAME, 0, 0, false, &frame);
            }
        }
        // Reconstruct exactly as a decoder does; this also checks the packet.
        let shown = self.core.decode(&tu[start..])?;
        Ok(shown.into_iter().last().map(|f| f.picture))
    }

    /// Runs the temporal dependency model over the mini-GOP `steps` that
    /// follows the anchor `lo` (the key frame still to be coded when `key`);
    /// returns the importance of each frame analyzed, in coding order.
    fn run_tpl<P: PicturePixel>(
        &self,
        lo: u64,
        steps: &[GopStep],
        key: bool,
    ) -> (Vec<(u64, f64)>, Option<TplGroup>) {
        let anchor = if key {
            self.lookahead.front().cloned()
        } else {
            self.slot_disp
                .iter()
                .position(|s| *s == Some(lo))
                .and_then(|slot| self.core.slot_picture(slot))
        };
        let Some(anchor) = anchor else {
            return (Vec::new(), None);
        };
        let Some(apic) = P::view(&anchor) else {
            return (Vec::new(), None);
        };
        let mut frames = vec![super::tpl::TplFrame {
            pic: apic,
            disp: lo as i64,
            refs: Vec::new(),
            analyze: key,
        }];
        let mut order = Vec::new();
        if key {
            order.push(0);
        }
        for st in steps {
            let GopStep::Code { disp, .. } = *st else {
                continue;
            };
            if frames.iter().any(|f| f.disp == disp as i64) {
                continue;
            }
            let idx = (disp - self.next_disp) as usize;
            let Some(pic) = self.lookahead.get(idx).and_then(|a| P::view(a)) else {
                return (Vec::new(), None);
            };
            // Nearest past and nearest future frame coded so far.
            let d = disp as i64;
            let past = frames
                .iter()
                .enumerate()
                .filter(|(_, f)| f.disp < d)
                .max_by_key(|(_, f)| f.disp)
                .map(|(i, _)| i);
            let near = frames
                .iter()
                .enumerate()
                .filter(|(_, f)| f.disp > d)
                .min_by_key(|(_, f)| f.disp)
                .map(|(i, _)| i);
            let mut refs: Vec<usize> = Vec::new();
            for r in [past, near].into_iter().flatten() {
                if !refs.contains(&r) {
                    refs.push(r);
                }
            }
            order.push(frames.len());
            frames.push(super::tpl::TplFrame {
                pic,
                disp: d,
                refs,
                analyze: true,
            });
        }
        let stats = super::tpl::analyze(&frames, &order, self.cfg.bit_depth, self.cfg.threads);
        let imp = order
            .iter()
            .map(|&f| (frames[f].disp as u64, stats.frame_importance(f)))
            .collect();
        let disps = frames.iter().map(|f| f.disp as u64).collect();
        (imp, Some((stats, disps)))
    }

    /// Per-superblock q index (and squared step ratio) for the frame at
    /// `disp` coded at `q`: regions more of the group is predicted from get
    /// a finer quantizer. Empty when the model has nothing for the frame.
    fn sb_quantizers(&self, disp: u64, q: u8, sb_rows: usize, sb_cols: usize) -> Vec<(i32, f64)> {
        let Some((stats, disps)) = &self.tpl else {
            return Vec::new();
        };
        let Some(f) = disps.iter().position(|&d| d == disp) else {
            return Vec::new();
        };
        if stats.intra[f].iter().all(|&v| v == 0.0) {
            return Vec::new();
        }
        let beta = ev("AV1_DQB", 0.5);
        let res = 1i32 << DELTA_Q_RES;
        let maxd = ev("AV1_DQMAX", 24.0) as i32;
        let fi = stats.frame_importance(f);
        let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
        let step = |q: i32| f64::from(AC_QLOOKUP[bdi][q.clamp(0, 255) as usize]);
        let q0 = i32::from(q);
        let mut out = Vec::with_capacity(sb_rows * sb_cols);
        for r in 0..sb_rows {
            for c in 0..sb_cols {
                let si = stats.region_importance(f, c * 64, r * 64, 64);
                let target = step(q0) * (si / fi).powf(-beta);
                // The q index whose step is nearest, in delta_q_res units.
                let mut best = (f64::MAX, q0);
                let mut d = -maxd;
                while d <= maxd {
                    let qq = q0 + d;
                    if (1..=255).contains(&qq) {
                        let e = (step(qq).ln() - target.ln()).abs();
                        if e < best.0 {
                            best = (e, qq);
                        }
                    }
                    d += res;
                }
                let ratio = step(best.1) / step(q0);
                out.push((best.1, ratio * ratio));
            }
        }
        out
    }

    /// Chooses q, references and the slot to refresh for a frame.
    fn plan_frame(&self, disp: u64, key: bool, show: bool, level: u32) -> FrameSpec {
        let base = self.rc_q;
        // Better anchors pay off in every frame predicted from them: q drops
        // by a fraction that grows toward the top of the pyramid.
        let frac = if key {
            ev("AV1_QK", 0.40)
        } else if level == u32::MAX {
            0.0
        } else {
            match level {
                0 => ev("AV1_Q0", 0.33),
                1 => ev("AV1_Q1", 0.22),
                2 => ev("AV1_Q2", 0.12),
                _ => ev("AV1_Q3", 0.05),
            }
        };
        // The alt-ref's offset follows its temporal importance; the layers
        // under it keep their share of that offset.
        let frac = match self.tpl_imp.first() {
            Some(&(d0, imp)) if level != u32::MAX && (!key || d0 == disp) => {
                let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
                let qi = base.round().clamp(1.0, 255.0) as usize;
                let step = f64::from(AC_QLOOKUP[bdi][qi]);
                let alpha = if key {
                    ev("AV1_TPLKA", 0.8)
                } else {
                    ev("AV1_TPLA", 0.9)
                };
                // The model (SATD at half resolution) overstates how much
                // of screen content's detail later frames take from the
                // alt-ref (it smooths away what is redrawn every frame).
                let floor = if self.screen {
                    ev("AV1_TPLMINS", 0.5)
                } else {
                    ev("AV1_TPLMIN", 0.3)
                };
                let target = step * imp.powf(-alpha).max(floor);
                let mut q = qi;
                while q > 0 && f64::from(AC_QLOOKUP[bdi][q - 1]) >= target {
                    q -= 1;
                }
                let arf = 1.0 - q as f64 / qi as f64;
                if key {
                    arf
                } else {
                    arf * frac / ev("AV1_Q0", 0.33)
                }
            }
            _ => frac,
        };
        let q = if self.cfg.gop > 1 || key {
            base * (1.0 - frac)
        } else {
            base
        };
        let q = q.round().clamp(1.0, 255.0) as u8;
        if key {
            return FrameSpec {
                disp,
                key,
                show,
                q,
                ref_frame_idx: [0; 7],
                refresh: 0xff,
                reference_select: false,
                level,
            };
        }
        // Past frames nearest first, future frames nearest first.
        let mut past: Vec<(u64, usize)> = Vec::new();
        let mut fut: Vec<(u64, usize)> = Vec::new();
        for (i, s) in self.slot_disp.iter().enumerate() {
            if let Some(sd) = *s {
                if past.iter().chain(&fut).any(|p| p.0 == sd) {
                    continue;
                }
                if sd <= disp {
                    past.push((sd, i));
                } else if sd > disp {
                    fut.push((sd, i));
                }
            }
        }
        past.sort_by_key(|b| std::cmp::Reverse(b.0));
        fut.sort_by_key(|a| a.0);
        let any = past.first().or(fut.first()).map_or(0, |p| p.1) as u8;
        let mut idx = [any; 7];
        // LAST, LAST2, LAST3, GOLDEN (the oldest kept past frame).
        for (k, p) in past.iter().take(3).enumerate() {
            idx[k] = p.1 as u8;
        }
        if let Some(p) = past.last() {
            idx[3] = p.1 as u8;
        }
        if past.len() < 3 {
            for k in past.len()..3 {
                idx[k] = idx[0];
            }
        }
        // BWDREF, ALTREF2, ALTREF (the farthest future frame).
        if !fut.is_empty() {
            idx[4] = fut[0].1 as u8;
            idx[5] = fut.get(1).unwrap_or(&fut[0]).1 as u8;
            idx[6] = fut[fut.len() - 1].1 as u8;
        } else {
            idx[4] = idx[0];
            idx[5] = idx[1];
            idx[6] = idx[3];
        }
        // Slot to refresh: an empty one, else the oldest past frame not among
        // the three nearest and not a frame still to be shown.
        let shown_upto = self.next_disp.max(1) - 1;
        let mut victim = self.slot_disp.iter().position(|s| s.is_none());
        if victim.is_none() {
            let keep: Vec<u64> = past.iter().take(3).map(|p| p.0).collect();
            victim = self
                .slot_disp
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|d| (i, d)))
                .filter(|&(_, d)| {
                    d <= shown_upto.max(disp.saturating_sub(1)) && d < disp && !keep.contains(&d)
                })
                .min_by_key(|&(_, d)| d)
                .map(|(i, _)| i);
        }
        if victim.is_none() {
            // Fall back to the oldest slot of all.
            victim = self
                .slot_disp
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|d| (i, d)))
                .min_by_key(|&(_, d)| d)
                .map(|(i, _)| i);
        }
        let refresh = 1u8 << victim.unwrap_or(0);
        FrameSpec {
            disp,
            key,
            show,
            q,
            ref_frame_idx: idx,
            refresh,
            reference_select: !fut.is_empty() && !past.is_empty(),
            level,
        }
    }

    /// Adjusts the base q after a key frame or a mini-GOP of `frames` pictures.
    fn update_rate_control(&mut self, bits: u64, frames: u64, key: bool) {
        let RateControl::Bitrate { bps, fps } = self.cfg.rc else {
            return;
        };
        let per_frame = bps as f64 / fps.max(1.0);
        // Key frames are expected to cost several ordinary frames.
        let expect = per_frame * if key { 6.0 } else { frames.max(1) as f64 };
        let ratio = (bits as f64 / expect.max(1.0)).clamp(0.1, 10.0);
        // About 12 q-index steps per doubling of the rate.
        let step = (12.0 * ratio.log2()).clamp(-20.0, 20.0) * if key { 0.5 } else { 1.0 };
        // Pull the average toward the target as well: the long-run one,
        // or at a constant rate the one of the second being coded, which
        // starts again with each second so that an easy second saves no
        // bits for a hard one.
        let avg = if self.cfg.constant_rate {
            self.second_bits += bits;
            self.second_frames += frames.max(1);
            let avg = self.second_bits as f64 / self.second_frames as f64;
            if self.second_frames as f64 >= fps.max(1.0) {
                self.second_bits = 0;
                self.second_frames = 0;
            }
            avg
        } else {
            self.bits_total as f64 / self.frame_num.max(1) as f64
        };
        let drift = (4.0 * (avg / per_frame.max(1.0)).log2()).clamp(-4.0, 4.0);
        self.rc_q = (self.rc_q + step + drift).clamp(10.0, 250.0);
    }

    fn frame_header(&self, spec: &FrameSpec) -> FrameHeader {
        let (key, q) = (spec.key, spec.q);
        let bd = self.cfg.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let ac = i32::from(AC_QLOOKUP[bdi][q as usize]) >> (bd - 8);
        // Loop filter strength from the quantizer (libaom-style fit).
        let lvl = if key {
            (ac * 20723 + 1_015_158) >> 18
        } else {
            (ac * 20723 + 1_015_158 - 300_000) >> 18
        }
        .clamp(0, 63) as u8;
        let lvl = (f64::from(lvl) * ev("AV1_LFS", 1.0)).clamp(0.0, 63.0) as u8;
        let pri = ((f64::from(q) / 24.0) * ev("AV1_CDS", 1.0)).clamp(0.0, 15.0) as u8;
        let mut h = FrameHeader {
            frame_type: if key { KEY_FRAME } else { INTER_FRAME },
            frame_is_intra: key,
            show_frame: spec.show,
            showable_frame: !key && !spec.show,
            error_resilient_mode: key,
            order_hint: (spec.disp & 127) as u32,
            primary_ref_frame: if key { PRIMARY_REF_NONE } else { 0 },
            refresh_frame_flags: spec.refresh,
            reference_select: spec.reference_select,
            skip_mode_present: spec.reference_select && ev("AV1_SM", 1.0) > 0.0,
            use_ref_frame_mvs: !key && ev("AV1_RFMV", 1.0) > 0.0,
            allow_warped_motion: !key && self.cfg.speed < 7 && ev("AV1_WARP", 1.0) > 0.0,
            is_motion_mode_switchable: !key && self.cfg.speed < 7 && ev("AV1_WARP", 1.0) > 0.0,
            frame_width: self.cfg.width as u32,
            frame_height: self.cfg.height as u32,
            upscaled_width: self.cfg.width as u32,
            render_width: self.cfg.width as u32,
            render_height: self.cfg.height as u32,
            superres_denom: 8,
            ref_frame_idx: spec.ref_frame_idx,
            allow_high_precision_mv: !key
                && self.cfg.speed < 8
                && f64::from(q) < ev("AV1_HPQ", 256.0),
            interpolation_filter: if key || self.cfg.speed >= 8 || ev("AV1_SWF", 1.0) == 0.0 {
                0
            } else {
                4
            },
            tile_info: {
                let mi_cols = 2 * ((self.cfg.width as u32 + 7) >> 3);
                let mi_rows = 2 * ((self.cfg.height as u32 + 7) >> 3);
                let mut ti = uniform_tile_info(
                    &self.seq,
                    mi_cols,
                    mi_rows,
                    self.cfg.tile_cols_log2,
                    self.cfg.tile_rows_log2,
                );
                let l = tile_limits(&self.seq, mi_cols, mi_rows);
                if self.auto_tiles
                    && l.min_log2_tiles == 0
                    && let Some((w, h)) = super::tiles::balanced_layout(
                        &self.sb_work,
                        l.sb_cols as usize,
                        l.sb_rows as usize,
                        ti.cols,
                        ti.rows,
                        l.max_tile_width_sb as usize,
                    )
                {
                    ti = explicit_tile_info(&self.seq, mi_cols, mi_rows, &w, &h);
                }
                if ti.cols * ti.rows > 1 {
                    ti.tile_size_bytes = 4;
                }
                ti
            },
            quant: Quant {
                base_q_idx: q,
                delta_q_u_dc: ev("AV1_DUV", 0.0) as i8,
                delta_q_u_ac: ev("AV1_DUV", 0.0) as i8,
                delta_q_v_dc: ev("AV1_DUV", 0.0) as i8,
                delta_q_v_ac: ev("AV1_DUV", 0.0) as i8,
                ..Quant::default()
            },
            loop_filter: LoopFilter {
                level: [lvl, lvl, lvl, lvl],
                ref_deltas: crate::dec::obu::DEFAULT_LF_REF_DELTAS,
                ..LoopFilter::default()
            },
            cdef: Cdef {
                damping: 3 + u8::from(q > 120),
                bits: 0,
                y_pri: [pri; 8],
                y_sec: [u8::from(q > 80) * 2; 8],
                uv_pri: [pri / 2; 8],
                uv_sec: [u8::from(q > 80); 8],
            },
            tx_mode: TX_MODE_SELECT,
            ..FrameHeader::default()
        };
        if key {
            h.ref_order_hint = [0; 8];
        }
        h
    }

    fn encode_frame<P: PicturePixel>(
        &mut self,
        src: &Picture<P>,
        spec: &FrameSpec,
    ) -> R<(Vec<u8>, Option<FrameRecon>)> {
        let seq = self.seq.clone();
        let key = spec.key;
        let q = spec.q;
        let mut intent = self.frame_header(spec);
        if !key && self.cfg.global_motion && self.cfg.speed < 7 {
            // Global motion per distinct reference picture.
            let mut done: [Option<Option<GmModel>>; 8] = Default::default();
            for rf in 1..8 {
                let slot = intent.ref_frame_idx[rf - 1] as usize & 7;
                if done[slot].is_none() {
                    let pic = self.core.slot_picture(slot);
                    let est = pic
                        .as_deref()
                        .and_then(|p| P::view(p))
                        .filter(|r| {
                            r.planes[0].w == src.planes[0].w && r.planes[0].h == src.planes[0].h
                        })
                        .and_then(|r| super::gm::estimate(src, r, self.cfg.bit_depth));
                    done[slot] = Some(est);
                }
                if let Some(Some((ty, p))) = done[slot] {
                    intent.gm_type[rf] = ty;
                    intent.gm_params[rf] = p;
                }
            }
        }
        intent.allow_screen_content_tools = self.screen;
        let sb_cols = self.cfg.width.div_ceil(64);
        let sb_rows = self.cfg.height.div_ceil(64);
        let sb_q = if ev("AV1_DQ", 0.0) > 0.0 && !seq.use_128x128_superblock {
            self.sb_quantizers(spec.disp, q, sb_rows, sb_cols)
        } else {
            Vec::new()
        };
        let sb_q = if sb_q.iter().any(|&(v, _)| v != i32::from(q)) {
            intent.delta_q_present = true;
            intent.delta_q_res = DELTA_Q_RES;
            sb_q
        } else {
            Vec::new()
        };
        let refs_hdr = self.core.ref_headers().clone();
        let ctx = HeaderContext {
            seq: &seq,
            refs: &refs_hdr,
            temporal_id: 0,
            spatial_id: 0,
        };
        let opts = FrameHeaderWriteOptions::default();
        let hdr_bytes = write_frame_header(&intent, &ctx, &opts);
        // The parser derives every dependent field exactly as a decoder will.
        let hdr = parse_frame_header(&hdr_bytes, &ctx)?;
        // Initial CDFs.
        let cdf: Box<CdfContext> = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::new(hdr.quant.base_q_idx)
        } else {
            let slot = hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize;
            let Some(c) = self.core.slot_cdf(slot) else {
                return Err(CodecError::InvalidApi {
                    what: "missing reference",
                });
            };
            let mut c = Box::new((*c).clone());
            c.clear_counts();
            c
        };
        let mut fs = FrameState::new(&seq, &hdr);
        if hdr.use_ref_frame_mvs {
            // Projected motion of the references, as the decoder builds it.
            let refs_data: [Option<Arc<crate::dec::frame::RefData>>; 8] =
                std::array::from_fn(|i| self.core.slot_data(i));
            fs.motion_field =
                crate::dec::mvpred::motion_field_estimation(&seq, &hdr, &refs_data, 1);
        }
        let mut cur = Picture::<P>::new(
            hdr.frame_width as usize,
            hdr.frame_height as usize,
            seq.num_planes(),
            seq.subsampling_x,
            seq.subsampling_y,
            seq.bit_depth,
        );
        let ref_pics: Vec<Option<Arc<AnyPicture>>> =
            (0..8).map(|s| self.core.slot_picture(s)).collect();
        let bd = seq.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let ac = f64::from(AC_QLOOKUP[bdi][q as usize]);
        let dcq = f64::from(DC_QLOOKUP[bdi][q as usize]);
        let qstep = ac / 8.0;
        // Frames no other frame predicts from trade quality for rate more
        // readily.
        let lscale = if spec.level == u32::MAX && !key {
            ev("AV1_LLEAF", 1.0)
        } else {
            1.0
        };
        let speed = self.cfg.speed;
        // Wide frames move by more samples: the search reaches about a
        // fifteenth of the width (48 up to 720 columns).
        let me_range = ev(
            "AV1_MER",
            if speed >= 8 {
                16.0
            } else {
                (hdr.frame_width as f64 / 15.0).clamp(48.0, 128.0)
            },
        ) as i32;
        let mode = ModeParams {
            lambda_satd: ev("AV1_LS", 0.8) * qstep * lscale.sqrt(),
            min_block: ev("AV1_MINB", if speed >= 8 { 16.0 } else { 8.0 }) as usize,
            max_block: ev("AV1_MAXB", 64.0) as usize,
            skip_sse_per_px: ev("AV1_SK", 4.0) * qstep * qstep * lscale,
            inter_frame: !key,
            me_range,
            me_ring: ev(
                "AV1_RING",
                if me_range > 48 { me_range as f64 } else { 16.0 },
            ) as i32,
            subpel: ev(
                "AV1_SUBPEL",
                if speed >= 9 {
                    1.0
                } else if speed >= 8 {
                    2.0
                } else {
                    3.0
                },
            ) as u8,
            intra_modes: if speed >= 8 { 10 } else { 13 },
            tx_depth: if speed >= 9 {
                0
            } else if speed >= 5 {
                1
            } else {
                2
            },
            cfl: speed < 9,
            angle_deltas: speed < 7,
            filter_intra: speed < 7,
            tx64: ev("AV1_TX64", 1.0) > 0.0,
            rd_partition: ev("AV1_RDP", 1.0) > 0.0,
            // LAST, LAST2, LAST3, GOLDEN, BWDREF, ALTREF2, ALTREF.
            ref_mask: match (speed, ev("AV1_REFS", 0.0) as u32) {
                (_, 1) | (0..=3, 0) => [false, true, true, true, true, true, true, true],
                (_, 2) => [false, true, false, false, true, true, false, true],
                _ => [false, true, false, false, false, true, false, true],
            },
            rd_inter_intra: ev("AV1_RDII", 1.0) > 0.0,
            early_none: ev("AV1_EN", 16.0),
            split_bits_scale: ev("AV1_SP", 1.0),
            filter_search: speed < 8,
            me_filter: ev("AV1_MEF", 0.0) as u8,
            local_warp: hdr.allow_warped_motion,
            obmc: hdr.is_motion_mode_switchable && ev("AV1_OBMC", 1.0) > 0.0,
            skip_mode: true,
            skip_mode_sse: ev("AV1_SMS", 6.0),
            fast_tx_dist: ev("AV1_FTD", 1.0) > 0.0,
            rect_partitions: ev("AV1_RECT", 0.0) > 0.0,
            subpel_pruned: ev("AV1_SPP", if speed >= 5 { 1.0 } else { 0.0 }) > 0.0,
            subpel_sad_steps: ev("AV1_SPSAD", 0.0) as u8,
            intra_refine_ratio: ev("AV1_IRR", if speed >= 5 { 1.0 } else { 0.0 }),
            intra_sweep_ratio: ev("AV1_IRS", if speed >= 5 { 2.0 } else { 0.0 }),
            inter_clear_ratio: ev("AV1_ICR", if speed >= 5 { 1.2 } else { 0.0 }),
            est_skip_ratio: ev("AV1_EST", if speed >= 5 { 2.0 } else { 0.0 }),
            exact_rates: ev("AV1_XR", 1.0) > 0.0,
            inter_rd_cands: ev("AV1_IRK", if speed >= 8 { 1.0 } else { 2.0 }) as usize,
            inter_rd_ratio: ev(
                "AV1_IRKR",
                if hdr.allow_screen_content_tools {
                    1.5
                } else {
                    1.1
                },
            ),
            inter_rd_modes: ev("AV1_IRM", 1.0) as usize,
            intra_split_pred: ev("AV1_ITXP", 1.0) > 0.0,
            inter_flag_bits: ev("AV1_IIB", 1.0) > 0.0,
            exact_intra_rates: ev("AV1_XRI", 1.0) > 0.0,
            screen_est: ev("AV1_SCE", 1.0) > 0.0,
            compound_modes: ev("AV1_CMM", 1.0) > 0.0,
            palette_cache: ev("AV1_PCACHE", 1.0) > 0.0,
            est_intra_ratio: ev("AV1_EIP", if speed >= 5 { 2.0 } else { 0.0 }),
            part_rd: ev("AV1_PRD", if speed <= 6 { 1.0 } else { 0.0 }) > 0.0,
            merge_check: match (ev("AV1_PRDF", 0.0) as u32, spec.level == u32::MAX) {
                (1, true) | (2, false) => 0,
                _ => ev("AV1_PRDI", 0.0) as usize,
            },
            merge_none_first: ev("AV1_PRDN", 0.0) > 0.0,
            merge_accept: ev("AV1_PRDA", 0.0),
            merge_tx_search: ev("AV1_PRDL", 3.0) as u8,
            // Flat 16x16 areas are not split, busy 32x32 ones and 64x64
            // ones are not coded whole (by the estimate per sample).
            intra_split_skip: [
                ev("AV1_KS16", 0.3),
                ev("AV1_KS32", 0.03),
                ev("AV1_KS64", 0.0),
            ],
            intra_none_skip: [
                ev("AV1_KN16", f64::MAX),
                ev("AV1_KN32", 1.0),
                ev("AV1_KN64", 0.0),
            ],
            intra_rd_cands: ev(
                "AV1_IRD",
                if speed >= 7 {
                    1.0
                } else if speed >= 6 {
                    // Key frames are few: their modes get the RD check.
                    if key { 2.0 } else { 1.0 }
                } else if speed >= 4 {
                    3.0
                } else {
                    5.0
                },
            ) as u8,
        };
        let qp = QuantParams {
            // Per bit; inter frames trade residual for rate more readily.
            lambda: ev("AV1_QL", 0.07) * qstep * qstep * lscale,
            round_intra: ev("AV1_RI", 56.0) as i32,
            round_inter: ev("AV1_RN", 44.0) as i32,
            tx_type_search: if speed >= 9 {
                0
            } else {
                (if key {
                    ev(
                        "AV1_TXSK",
                        ev("AV1_TXS", if speed >= 7 { 1.0 } else { 3.0 }),
                    )
                } else {
                    ev("AV1_TXS", if speed >= 7 { 1.0 } else { 3.0 })
                }) as u8
            },
            tx_type_keep: (if key {
                ev("AV1_TXKK", ev("AV1_TXK", 3.0))
            } else {
                ev("AV1_TXK", 3.0)
            }) as u8,
            rdoq: if speed >= 9 {
                0
            } else if speed >= 7 {
                1
            } else {
                2
            },
        };
        let _ = dcq;
        let ti = hdr.tile_info.clone();
        let ctx_tile = ti.context_update_tile_id as usize;
        // Per-64x64 CDEF strengths are chosen after coding: the tiles log
        // their symbols and are coded again with the cdef_idx values.
        let cdef_fb = self.cfg.speed < 8
            && seq.enable_cdef
            && !seq.use_128x128_superblock
            && !hdr.coded_lossless
            && !hdr.allow_intrabc
            && ev("AV1_CDEFB", 0.0) > 0.0;
        // Loop restoration likewise (its syntax starts each superblock).
        let lr_on = seq.enable_restoration
            && self.cfg.speed < 8
            && !hdr.coded_lossless
            && !hdr.allow_intrabc
            && !hdr.use_superres;
        // The tiles, superblock rows in a wavefront (see `rowmt`).
        let job = super::rowmt::TileJob {
            seq: &seq,
            hdr: &hdr,
            refs: std::array::from_fn(|i| {
                if i == 0 || hdr.frame_is_intra {
                    return None;
                }
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                let pic = ref_pics[slot].as_deref()?;
                Some(super::rowmt::RefInfo {
                    pic: P::view(pic)?,
                    upscaled_width: refs_hdr[slot].upscaled_width as usize,
                    frame_height: refs_hdr[slot].frame_height as usize,
                    frame_width: refs_hdr[slot].frame_width as usize,
                })
            }),
            src,
            init_cdf: &cdf,
            qp,
            mode,
            sb_q: &sb_q,
            sb_cols,
            record: cdef_fb || lr_on,
            threads: self.cfg.threads.max(1),
        };
        let mut saved_cdf: Option<Box<CdfContext>> = None;
        let mut tile_bytes: Vec<MsacEnc> = Vec::with_capacity(ti.cols * ti.rows);
        for (m, c) in super::rowmt::code_tiles(&job, &mut fs, &mut cur)? {
            tile_bytes.push(m);
            saved_cdf = saved_cdf.or(c);
        }
        let use_wiener_cdf = cdf.use_wiener;
        let init_cdf = cdf;
        if self.auto_tiles {
            self.sb_work = super::tiles::sb_work(&fs);
        }
        // Choose the in-loop filter strengths on the reconstruction. Tile
        // syntax does not depend on them (one CDEF strength, no deltas), so
        // only the header is rewritten.
        let mut hdr_bytes = hdr_bytes;
        let mut cdef_choice: Option<super::cdef_search::CdefChoice> = None;
        let mut lr_plan: Option<super::lr_search::LrPlan> = None;
        // The CDEF output of the chosen strengths, if made.
        let mut cdef_out: Option<Picture<P>> = None;
        let mut best_lf: Option<(u64, Picture<P>)> = None;
        if self.cfg.speed < 8 && !hdr.coded_lossless && !hdr.allow_intrabc {
            let full = self.cfg.speed < 4;
            let threads = self.core.threads.max(1);
            let mut best_h = intent.clone();
            let base_lvl = intent.loop_filter.level[0];
            let lf_f: &[f64] = if full {
                &[0.0, 0.5, 0.75, 1.0, 1.25, 1.6]
            } else {
                &[0.6, 1.0, 1.4]
            };
            let mut levels: Vec<u8> = lf_f
                .iter()
                .map(|f| (f64::from(base_lvl) * f).round().clamp(0.0, 63.0) as u8)
                .collect();
            levels.dedup();
            // Above speed 4 every other frame nothing predicts from takes
            // the filter strengths the one before chose instead of
            // searching (its loop filter level is then the only one tried).
            let leaf = spec.level == u32::MAX && !key && self.cfg.speed >= 5;
            let reuse = if leaf { self.leaf_filters.take() } else { None };
            if let Some((l, _)) = reuse {
                levels = vec![l];
            }
            // Below speed 4 each level filters the whole frame. Else every
            // other 64-row area is filtered and measured (of at least 4, as
            // the CDEF search does), each area of each level a job of its
            // own on a copy of the rows it touches, and the chosen level
            // then filters the whole frame.
            let sb_rows = fs.mi_rows.div_ceil(16);
            let sampled = !full && sb_rows >= 4 && !seq.use_128x128_superblock && levels.len() > 1;
            let level_hdr = |l: u8| {
                let mut h2 = hdr.clone();
                h2.loop_filter.level = [l, l, l, l];
                h2
            };
            let n = levels.len();
            let tried: Vec<(u64, Option<Picture<P>>)> = if sampled {
                let areas: Vec<usize> = (0..fs.mi_rows).step_by(32).collect();
                let hdrs: Vec<FrameHeader> = levels.iter().map(|&l| level_hdr(l)).collect();
                let n_jobs = n * areas.len();
                let next = std::sync::atomic::AtomicUsize::new(0);
                let job_sse = |pic: &mut Picture<P>, j: usize| -> u64 {
                    let (h2, r0) = (&hdrs[j / areas.len()], areas[j % areas.len()]);
                    // Two 4x4 rows more, so the edge below the area is
                    // filtered too; the filters reach 8 rows past the rows
                    // they filter.
                    let r1 = (r0 + 18).min(fs.mi_rows);
                    for (p, (d, s)) in pic.planes.iter_mut().zip(&cur.planes).enumerate() {
                        let sy = usize::from(p > 0 && seq.subsampling_y);
                        let y0 = ((r0 * 4) >> sy).saturating_sub(8);
                        let y1 = (((r1 * 4) >> sy) + 8).min(d.alloc_h);
                        d.data[y0 * d.stride..y1 * d.stride]
                            .copy_from_slice(&s.data[y0 * s.stride..y1 * s.stride]);
                    }
                    crate::dec::postfilter::loop_filter_rows(
                        &seq,
                        h2,
                        &fs,
                        pic,
                        r0,
                        r1,
                        r1 == fs.mi_rows,
                    );
                    let mut e = 0;
                    for (p, (pa, pb)) in pic.planes.iter().zip(&src.planes).enumerate() {
                        let sy = usize::from(p > 0 && seq.subsampling_y);
                        e += plane_rows_sse(pa, pb, (r0 * 4) >> sy, ((r0 + 16) * 4) >> sy);
                    }
                    e
                };
                let run = || -> Vec<(usize, u64)> {
                    let mut pic = Picture::<P>::new(
                        hdr.frame_width as usize,
                        hdr.frame_height as usize,
                        seq.num_planes(),
                        seq.subsampling_x,
                        seq.subsampling_y,
                        seq.bit_depth,
                    );
                    let mut out = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= n_jobs {
                            return out;
                        }
                        out.push((j, job_sse(&mut pic, j)));
                    }
                };
                let t = threads.min(n_jobs);
                let parts: Vec<(usize, u64)> = if t > 1 {
                    std::thread::scope(|sc| {
                        let hs: Vec<_> = (1..t).map(|_| sc.spawn(run)).collect();
                        let mut all = run();
                        for h in hs {
                            all.extend(h.join().map_err(|_| CodecError::InvalidApi {
                                what: "loop filter thread",
                            })?);
                        }
                        Ok::<_, CodecError>(all)
                    })?
                } else {
                    run()
                };
                let mut sums = vec![0u64; n];
                for (j, e) in parts {
                    sums[j / areas.len()] += e;
                }
                sums.into_iter().map(|e| (e, None)).collect()
            } else {
                let try_level = |l: u8, t: usize| -> (u64, Option<Picture<P>>) {
                    let mut pic = cur.clone();
                    crate::dec::postfilter::loop_filter(&seq, &level_hdr(l), &fs, &mut pic, t);
                    (picture_sse(&pic, src), Some(pic))
                };
                // The levels are tried side by side, sharing the threads.
                if threads > 1 && n > 1 {
                    std::thread::scope(|sc| {
                        let hs: Vec<_> = levels
                            .iter()
                            .enumerate()
                            .map(|(i, &l)| {
                                let t = (threads / n + usize::from(i < threads % n)).max(1);
                                let try_level = &try_level;
                                sc.spawn(move || try_level(l, t))
                            })
                            .collect();
                        hs.into_iter()
                            .map(|h| {
                                h.join().map_err(|_| CodecError::InvalidApi {
                                    what: "loop filter thread",
                                })
                            })
                            .collect::<R<Vec<_>>>()
                    })?
                } else {
                    levels.iter().map(|&l| try_level(l, threads)).collect()
                }
            };
            let mut best: Option<(u64, u8, Option<Picture<P>>)> = None;
            for (l, (e, pic)) in levels.into_iter().zip(tried) {
                if best.as_ref().is_none_or(|b| e < b.0) {
                    best = Some((e, l, pic));
                }
            }
            if let Some((e, l, pic)) = best {
                best_h.loop_filter.level = [l, l, l, l];
                let pic = match pic {
                    Some(p) => p,
                    None => {
                        let mut p = cur.clone();
                        let h2 = level_hdr(l);
                        crate::dec::postfilter::loop_filter(&seq, &h2, &fs, &mut p, threads);
                        p
                    }
                };
                best_lf = Some((e, pic));
            }
            if let Some((_, deblocked)) = best_lf.as_ref()
                && cdef_fb
            {
                let p0 = intent.cdef.y_pri[0];
                let mut cands: Vec<(u8, u8)> = Vec::new();
                let np = ev("AV1_CDEFNP", 3.0) as usize;
                for pri in [p0 / 2, p0, p0 * 3 / 2, p0 * 2].into_iter().take(np) {
                    for sec in [0u8, 2] {
                        let c = (pri.min(15), sec);
                        if c != (0, 0) && !cands.contains(&c) {
                            cands.push(c);
                        }
                    }
                }
                let ch = super::cdef_search::search(
                    &seq,
                    &hdr,
                    &fs,
                    deblocked,
                    src,
                    &cands,
                    qp.lambda * ev("AV1_CDEFL", 1.0),
                    ev("AV1_CDEFB", 3.0) as u8,
                    threads,
                );
                best_h.cdef.bits = ch.bits;
                for (k, &st) in ch.strengths.iter().enumerate() {
                    super::cdef_search::set_strength(&mut best_h, k, st);
                }
                cdef_choice = Some(ch);
            } else if let Some((_, [yp, ys, up, us])) = reuse
                && seq.enable_cdef
            {
                best_h.cdef.y_pri = [yp; 8];
                best_h.cdef.y_sec = [ys; 8];
                best_h.cdef.uv_pri = [up; 8];
                best_h.cdef.uv_sec = [us; 8];
            } else if let Some((_, deblocked)) = best_lf.as_ref()
                && seq.enable_cdef
            {
                let p0 = intent.cdef.y_pri[0];
                let pris: &[u8] = if full {
                    &[p0 / 2, p0, (p0 * 3 / 2).min(15), (p0 * 2).min(15)]
                } else {
                    &[p0 / 2, p0, (p0 * 3 / 2).min(15)]
                };
                let secs: &[u8] = if full { &[0, 1, 2, 4] } else { &[0, 2] };
                let mut combos = Vec::new();
                for &pri in pris {
                    for &sec in secs {
                        if pri != 0 || sec != 0 {
                            combos.push((pri, sec));
                        }
                    }
                }
                // Below speed 4 every 64-row area is measured, else every
                // other one (of at least 4); chroma strengths are chosen on
                // their own (the chroma filter does not touch luma). The
                // choice does not depend on the thread count.
                let sb_rows = fs.mi_rows.div_ceil(16);
                let step = if full || sb_rows < 4 { 1 } else { 2 };
                let rows: Vec<usize> = (0..sb_rows).step_by(step).collect();
                let mut uv_combos = Vec::new();
                for up in [0u8, 1, 2, 4, 6] {
                    for us in [0u8, 2] {
                        uv_combos.push((up, us));
                    }
                }
                let (dl, dc) = super::cdef_search::strength_deltas(
                    &seq, &hdr, &fs, deblocked, src, &combos, &uv_combos, &rows, threads,
                );
                let mut best_c = (0i64, None::<(u8, u8)>);
                for (&c, &e) in combos.iter().zip(&dl) {
                    if e < best_c.0 {
                        best_c = (e, Some(c));
                    }
                }
                let (pri, sec) = best_c.1.unwrap_or((0, 0));
                best_h.cdef.y_pri = [pri; 8];
                best_h.cdef.y_sec = [sec; 8];
                best_h.cdef.uv_pri = [pri / 2; 8];
                best_h.cdef.uv_sec = [sec.min(2) / 2 * 2; 8];
                if !dc.is_empty() {
                    let mut best_uv = (i64::MAX, (0, 0));
                    for (&c, &e) in uv_combos.iter().zip(&dc) {
                        if e < best_uv.0 {
                            best_uv = (e, c);
                        }
                    }
                    best_h.cdef.uv_pri = [best_uv.1.0; 8];
                    best_h.cdef.uv_sec = [best_uv.1.1; 8];
                }
            }
            if lr_on && let Some((_, deblocked)) = best_lf.as_ref() {
                // Loop restoration on the final CDEF output.
                let mut h2 = hdr.clone();
                h2.cdef = best_h.cdef.clone();
                h2.loop_filter = best_h.loop_filter.clone();
                let mut cdef_pic = deblocked.clone();
                if seq.enable_cdef {
                    crate::dec::postfilter::cdef(&seq, &h2, &fs, deblocked, &mut cdef_pic, threads);
                }
                lr_plan = super::lr_search::search(
                    &seq,
                    &h2,
                    &mut fs,
                    &cdef_pic,
                    deblocked,
                    src,
                    qp.lambda * ev("AV1_LRL", 1.0),
                    ev("AV1_LRU", 64.0) as u32,
                    threads,
                );
                if let Some(p) = &lr_plan {
                    p.apply(&mut best_h);
                }
                cdef_out = Some(cdef_pic);
            }
            if leaf && reuse.is_none() {
                let c = &best_h.cdef;
                self.leaf_filters = Some((
                    best_h.loop_filter.level[0],
                    [c.y_pri[0], c.y_sec[0], c.uv_pri[0], c.uv_sec[0]],
                ));
            }
            hdr_bytes = write_frame_header(&best_h, &ctx, &opts);
        }
        let mut ctx_wiener = None;
        let tile_bytes: Vec<Vec<u8>> = tile_bytes
            .into_iter()
            .enumerate()
            .map(|(tile, m)| {
                let mut lw = lr_plan
                    .as_ref()
                    .map(|p| super::lr_search::LrWriter::new(p, use_wiener_cdf, &seq));
                let bytes = m.replay(&mut |tag: u32, e: &mut MsacEnc| {
                    if tag & crate::dec::coder::SB_MARK != 0 {
                        if let Some(w) = lw.as_mut() {
                            let r = ((tag >> 15) & 0x7fff) as usize;
                            let c = (tag & 0x7fff) as usize;
                            w.write_sb(e, r, c);
                        }
                    } else if let Some(ch) = &cdef_choice {
                        let v = ch.idx.get(tag as usize).copied().unwrap_or(0).max(0);
                        e.literal(u32::from(ch.bits), v as u32);
                    }
                });
                if tile == ctx_tile {
                    ctx_wiener = lw.map(|w| w.use_wiener_cdf());
                }
                bytes
            })
            .collect();
        // Frame OBU: header, then one tile group with every tile.
        let final_hdr = parse_frame_header(&hdr_bytes, &ctx)?;
        let mut payload = hdr_bytes;
        payload.extend_from_slice(&write_tile_group(
            &ti,
            0,
            ti.cols * ti.rows - 1,
            &tile_bytes,
        ));
        // The reference the frame leaves, reconstructed as a decoder does
        // from what the tiles left.
        let Some(mut cdf) = saved_cdf else {
            return Ok((payload, None));
        };
        if final_hdr.use_superres {
            return Ok((payload, None));
        }
        if let Some(ch) = &cdef_choice {
            fs.cdef_idx.clone_from(&ch.idx);
        }
        if let Some(p) = &lr_plan {
            fs.lr = p.units.clone();
            fs.lr_cols = p.cols;
            fs.lr_rows = p.rows;
        }
        let threads = self.core.threads.max(1);
        let deblocked = match best_lf {
            Some((_, d)) => d,
            None => {
                crate::dec::postfilter::loop_filter(&seq, &final_hdr, &fs, &mut cur, threads);
                cur
            }
        };
        let c = &final_hdr.cdef;
        let cdef_on = seq.enable_cdef
            && !final_hdr.coded_lossless
            && !final_hdr.allow_intrabc
            && (c.bits > 0
                || c.y_pri[0] != 0
                || c.y_sec[0] != 0
                || c.uv_pri[0] != 0
                || c.uv_sec[0] != 0);
        let cdef_pic = match (cdef_on, cdef_out) {
            (false, _) => None,
            (true, Some(p)) => Some(p),
            (true, None) => {
                let mut d = deblocked.clone();
                crate::dec::postfilter::cdef(&seq, &final_hdr, &fs, &deblocked, &mut d, threads);
                Some(d)
            }
        };
        let filtered = cdef_pic.as_ref().unwrap_or(&deblocked);
        let pic = if final_hdr.lr.uses_lr {
            let mut out = filtered.clone();
            crate::dec::postfilter::loop_restoration(
                &seq,
                &final_hdr,
                &fs,
                filtered,
                &deblocked,
                &mut out,
                &mut crate::dsp::lr::LrScratch::new(),
                threads,
            );
            out
        } else {
            match cdef_pic {
                Some(p) => p,
                None => deblocked,
            }
        };
        if let Some(w) = ctx_wiener {
            cdf.use_wiener = w;
        }
        let recon = FrameRecon {
            cdf: if final_hdr.disable_frame_end_update_cdf {
                init_cdf
            } else {
                cdf
            },
            hdr: final_hdr,
            fs,
            pic: P::wrap(pic),
        };
        Ok((payload, Some(recon)))
    }
}

/// A tuning constant; with the `tune` feature an environment variable of
/// the same name overrides it (for parameter sweeps).
#[cfg(feature = "tune")]
pub(crate) fn ev(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(not(feature = "tune"))]
#[inline(always)]
pub(crate) fn ev(_name: &str, default: f64) -> f64 {
    default
}

/// Squared error over the visible area of every plane.
fn picture_sse<P: Pixel>(a: &Picture<P>, b: &Picture<P>) -> u64 {
    a.planes
        .iter()
        .zip(&b.planes)
        .map(|(pa, pb)| plane_rows_sse(pa, pb, 0, pa.h))
        .sum()
}

/// SSE of rows `y0..y1` of two planes.
fn plane_rows_sse<P: Pixel>(
    pa: &crate::dec::frame::Plane<P>,
    pb: &crate::dec::frame::Plane<P>,
    y0: usize,
    y1: usize,
) -> u64 {
    let y1 = y1.min(pa.h);
    if y0 >= y1 {
        return 0;
    }
    let mut e = 0u64;
    let mut x = 0;
    while x < pa.w {
        let w = (pa.w - x).min(128);
        e += super::me::sse(
            &pa.data[y0 * pa.stride + x..],
            pa.stride,
            &pb.data[y0 * pb.stride + x..],
            pb.stride,
            w,
            y1 - y0,
        );
        x += w;
    }
    e
}

/// Screen content (text, graphics) has many blocks of few, contrasting
/// colors; palette coding pays off there.
fn is_screen_content<P: Pixel>(src: &Picture<P>, bd: u32) -> bool {
    let p = &src.planes[0];
    let thr = 32 << (bd - 8);
    let (mut few, mut total) = (0u32, 0u32);
    for by in (0..p.h.saturating_sub(15)).step_by(16) {
        for bx in (0..p.w.saturating_sub(15)).step_by(16) {
            // The distinct values, up to one more than 8.
            let mut seen = [0i32; 9];
            let mut distinct = 0;
            'block: for i in 0..16 {
                for &px in &p.data[(by + i) * p.stride + bx..][..16] {
                    let v = px.to_i32();
                    if !seen[..distinct].contains(&v) {
                        seen[distinct] = v;
                        distinct += 1;
                        if distinct > 8 {
                            break 'block;
                        }
                    }
                }
            }
            let seen = &seen[..distinct];
            let lo = seen.iter().copied().min().unwrap_or(0);
            let hi = seen.iter().copied().max().unwrap_or(0);
            total += 1;
            if (2..=8).contains(&distinct) && hi - lo >= thr {
                few += 1;
            }
        }
    }
    total > 0 && few * 100 >= total * 8
}

/// Copies the input into a picture padded by edge replication to whole superblocks.
fn source_picture<P: Pixel>(
    cfg: &EncConfig,
    seq: &SequenceHeader,
    input: &InputFrame<'_, P>,
) -> Picture<P> {
    let mut pic = Picture::<P>::new(
        cfg.width,
        cfg.height,
        seq.num_planes(),
        seq.subsampling_x,
        seq.subsampling_y,
        seq.bit_depth,
    );
    for (p, pl) in pic.planes.iter_mut().enumerate() {
        let src = input.planes[p];
        let ss = input.strides[p];
        let (w, h) = (pl.w, pl.h);
        let stride = pl.stride;
        for y in 0..pl.alloc_h {
            let sy = y.min(h - 1);
            let row = &mut pl.data[y * stride..(y + 1) * stride];
            row[..w].copy_from_slice(&src[sy * ss..sy * ss + w]);
            let last = row[w - 1];
            for v in &mut row[w..] {
                *v = last;
            }
        }
    }
    pic
}

/// The mean absolute difference of the luma of two pictures of one size,
/// over every fourth sample of every fourth row.
fn mean_change<P: Pixel>(a: &Picture<P>, b: &Picture<P>) -> f64 {
    let (pa, pb) = (&a.planes[0], &b.planes[0]);
    let (w, h) = (pa.w.min(pb.w), pa.h.min(pb.h));
    let (mut sum, mut count) = (0u64, 0u64);
    for y in (0..h).step_by(4) {
        let ra = &pa.data[y * pa.stride..][..w];
        let rb = &pb.data[y * pb.stride..][..w];
        for (x, y) in ra.iter().zip(rb).step_by(4) {
            sum += u64::from(x.to_i32().abs_diff(y.to_i32()));
            count += 1;
        }
    }
    sum as f64 / count.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::{AnyPicture, Av1Core, Av1Enc, EncConfig, InputFrame, RateControl};

    fn encode(threads: usize, tiles_log2: u32) -> Vec<u8> {
        let (w, h) = (320usize, 192usize);
        let mut cfg = EncConfig::new(w, h);
        cfg.rc = RateControl::Qindex(90);
        cfg.tile_cols_log2 = tiles_log2;
        cfg.tile_rows_log2 = tiles_log2;
        cfg.threads = threads;
        cfg.gop = 4;
        let mut enc = Av1Enc::new(cfg).expect("encoder");
        let mut out = Vec::new();
        for f in 0..5usize {
            let y: Vec<u8> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    ((x * 3 + y * 2 + f * 5) ^ (x * y / 7 + f)) as u8
                })
                .collect();
            let c = vec![100u8; (w / 2) * (h / 2)];
            for u in enc
                .encode(&InputFrame {
                    planes: [&y, &c, &c],
                    strides: [w, w / 2, w / 2],
                })
                .expect("encode")
            {
                out.extend(u.data);
            }
        }
        for u in enc.flush::<u8>().expect("flush") {
            out.extend(u.data);
        }
        out
    }

    /// Display indices of the key frames of `frames` pictures, with a key
    /// frame asked for at each index of `forced`.
    fn key_frames(keyint: u32, gop: u32, frames: u64, forced: &[u64]) -> Vec<u64> {
        let (w, h) = (64usize, 64usize);
        let mut cfg = EncConfig::new(w, h);
        cfg.rc = RateControl::Qindex(120);
        cfg.keyint = keyint;
        cfg.gop = gop;
        cfg.speed = 10;
        let mut enc = Av1Enc::new(cfg).expect("encoder");
        let mut units = Vec::new();
        for f in 0..frames {
            if forced.contains(&f) {
                enc.force_keyframe();
            }
            let y: Vec<u8> = (0..w * h).map(|i| (i as u64 * 3 + f * 7) as u8).collect();
            let c = vec![128u8; (w / 2) * (h / 2)];
            units.extend(
                enc.encode(&InputFrame {
                    planes: [&y, &c, &c],
                    strides: [w, w / 2, w / 2],
                })
                .expect("encode"),
            );
        }
        units.extend(enc.flush::<u8>().expect("flush"));
        assert_eq!(units.len() as u64, frames);
        let mut dec = Av1Core::new();
        dec.threads = 1;
        for (i, u) in units.iter().enumerate() {
            assert_eq!(u.display, i as u64);
            assert_eq!(dec.decode(&u.data).expect("decode").len(), 1);
        }
        units.iter().filter(|u| u.key).map(|u| u.display).collect()
    }

    /// The level and tier a decoder reads from the sequence header.
    fn level_of(cfg: EncConfig) -> (u8, u8) {
        let enc = Av1Enc::new(cfg).expect("encoder");
        let bytes = crate::enc::obu_write::write_sequence_header(enc.sequence_header());
        let seq = crate::dec::obu::parse_sequence_header(&bytes).expect("sequence header");
        let op = &seq.operating_points[0];
        // The av1C record repeats them.
        let record = enc.config_record();
        assert_eq!(record[1] & 31, op.seq_level_idx);
        assert_eq!(record[2] >> 7, op.seq_tier);
        (op.seq_level_idx, op.seq_tier)
    }

    #[test]
    fn the_sequence_header_names_the_level_the_stream_fits() {
        let mut small = EncConfig::new(320, 192);
        small.gop = 1;
        assert_eq!(level_of(small), (0, 0));
        // 1080p: level 4.0 at 30 pictures a second, 4.1 at 60.
        let mut hd = EncConfig::new(1920, 1080);
        assert_eq!(level_of(hd.clone()), (8, 0));
        hd.fps = 60.0;
        assert_eq!(level_of(hd.clone()), (9, 0));
        // A bitrate target carries the rate, and picks the tier.
        hd.rc = RateControl::Bitrate {
            bps: 16_000_000,
            fps: 30.0,
        };
        assert_eq!(level_of(hd.clone()), (8, 1));
        hd.rc = RateControl::Bitrate {
            bps: 4_000_000,
            fps: 30.0,
        };
        assert_eq!(level_of(hd.clone()), (8, 0));
        // 2160p is level 5.0 at 30 and 5.1 at 60, as players expect; asking
        // for a filtered alt-ref in every mini-GOP adds a decoded frame to
        // each eight shown, which 5.0 has no room for.
        let mut uhd = EncConfig::new(3840, 2160);
        assert_eq!(level_of(uhd.clone()), (12, 0));
        uhd.fps = 60.0;
        assert_eq!(level_of(uhd.clone()), (13, 0));
        uhd.fps = 30.0;
        uhd.temporal_filter = true;
        assert_eq!(level_of(uhd), (13, 0));
        // Tile columns past a level's count raise it.
        hd.tile_cols_log2 = 4;
        assert_eq!(level_of(hd), (16, 0));
        // 4:4:4 is profile 1, which doubles the bitrate of a tier.
        let mut wide = EncConfig::new(640, 360);
        wide.sub_x = false;
        wide.sub_y = false;
        wide.rc = RateControl::Bitrate {
            bps: 5_000_000,
            fps: 30.0,
        };
        assert_eq!(level_of(wide), (1, 0));
    }

    /// Display indices of the key frames of `frames` 64x64 pictures whose
    /// luma `luma(picture, sample)` gives.
    fn keys_of(cfg: EncConfig, frames: u64, luma: impl Fn(u64, usize) -> u8) -> Vec<u64> {
        let (w, h) = (cfg.width, cfg.height);
        let mut enc = Av1Enc::new(cfg).expect("encoder");
        let mut units = Vec::new();
        for f in 0..frames {
            let y: Vec<u8> = (0..w * h).map(|i| luma(f, i)).collect();
            let c = vec![128u8; (w / 2) * (h / 2)];
            units.extend(
                enc.encode(&InputFrame {
                    planes: [&y, &c, &c],
                    strides: [w, w / 2, w / 2],
                })
                .expect("encode"),
            );
        }
        units.extend(enc.flush::<u8>().expect("flush"));
        assert_eq!(units.len() as u64, frames);
        let mut dec = Av1Core::new();
        dec.threads = 1;
        for u in &units {
            assert_eq!(dec.decode(&u.data).expect("decode").len(), 1);
        }
        units.iter().filter(|u| u.key).map(|u| u.display).collect()
    }

    fn cut_config() -> EncConfig {
        let mut cfg = EncConfig::new(64, 64);
        cfg.rc = RateControl::Qindex(120);
        cfg.gop = 4;
        cfg.speed = 10;
        cfg
    }

    /// Two scenes: a dark ramp that drifts, then a bright grid.
    fn two_scenes(cut: u64) -> impl Fn(u64, usize) -> u8 {
        move |f, i| {
            let (x, y) = (i % 64, i / 64);
            if f < cut {
                (20 + (x + y + f as usize) % 40) as u8
            } else {
                (160 + ((x / 8 + y / 8 + f as usize) % 2) * 60) as u8
            }
        }
    }

    #[test]
    fn a_cut_starts_a_key_frame() {
        assert_eq!(keys_of(cut_config(), 30, two_scenes(17)), [0, 17]);
        // The key frame distance is counted from the cut.
        let mut cfg = cut_config();
        cfg.keyint = 20;
        assert_eq!(keys_of(cfg, 45, two_scenes(17)), [0, 17, 37]);
        // Off, only the distance starts one.
        let mut cfg = cut_config();
        cfg.scene_cut = false;
        assert_eq!(keys_of(cfg, 30, two_scenes(17)), [0]);
    }

    #[test]
    fn steady_change_starts_no_key_frame() {
        // A ramp that moves with every picture, noise that is new in every
        // picture, and a fade to white change each picture as much as the
        // one before.
        let drift = |f: u64, i: usize| ((i % 64 + i / 64) * 3 + f as usize * 9) as u8;
        assert_eq!(keys_of(cut_config(), 30, drift), [0]);
        let noise = |f: u64, i: usize| {
            let v = (i as u64 + 1).wrapping_mul(f.wrapping_mul(2_654_435_761) + 40_503);
            (v >> 13) as u8
        };
        assert_eq!(keys_of(cut_config(), 30, noise), [0]);
        let fade = |f: u64, i: usize| (40 + (i % 64) / 4 + f as usize * 7).min(255) as u8;
        assert_eq!(keys_of(cut_config(), 30, fade), [0]);
    }

    #[test]
    fn a_cut_close_behind_a_key_frame_starts_none() {
        // Twelve pictures (a tenth of the key frame distance) must lie
        // between: a cut at 5 is too close to the first key frame, and of
        // two cuts five apart the second starts none.
        assert_eq!(keys_of(cut_config(), 30, two_scenes(5)), [0]);
        let twice = |f: u64, i: usize| {
            if (17..22).contains(&f) {
                two_scenes(0)(f, i)
            } else {
                two_scenes(u64::MAX)(f, i)
            }
        };
        assert_eq!(keys_of(cut_config(), 30, twice), [0, 17]);
    }

    #[test]
    fn key_frames_keep_their_distance() {
        assert_eq!(key_frames(4, 4, 10, &[]), [0, 4, 8]);
        assert_eq!(key_frames(1, 4, 3, &[]), [0, 1, 2]);
    }

    #[test]
    fn a_forced_key_frame_restarts_the_distance() {
        // Inside a mini-GOP, on the periodic key frame, twice in a row,
        // and on the first picture.
        assert_eq!(key_frames(4, 4, 12, &[2]), [0, 2, 6, 10]);
        assert_eq!(key_frames(4, 2, 9, &[4]), [0, 4, 8]);
        assert_eq!(key_frames(8, 4, 9, &[3, 4]), [0, 3, 4]);
        assert_eq!(key_frames(8, 4, 5, &[0]), [0]);
        assert_eq!(key_frames(120, 8, 20, &[13]), [0, 13]);
    }

    #[test]
    fn tile_threads_do_not_change_the_bitstream() {
        assert_eq!(encode(1, 1), encode(3, 1));
    }

    #[test]
    fn row_threads_do_not_change_the_bitstream() {
        assert_eq!(encode(1, 0), encode(2, 0));
        assert_eq!(encode(1, 0), encode(4, 0));
    }

    /// Visible samples of every plane.
    fn samples(p: &AnyPicture) -> Vec<Vec<u16>> {
        fn rows<P: crate::dsp::Pixel>(p: &crate::dec::frame::Picture<P>) -> Vec<Vec<u16>> {
            p.planes
                .iter()
                .map(|pl| {
                    (0..pl.h)
                        .flat_map(|y| pl.data[y * pl.stride..][..pl.w].iter())
                        .map(|v| v.to_i32() as u16)
                        .collect()
                })
                .collect()
        }
        match p {
            AnyPicture::U8(p) => rows(p),
            AnyPicture::U16(p) => rows(p),
        }
    }

    /// The encoder's reconstruction (built from its own tiles, not decoded)
    /// matches what a decoder makes of the bitstream.
    #[test]
    fn reconstruction_matches_the_decoder() {
        let (w, h) = (136usize, 72usize);
        for (speed, lr, tiles, threads, bd) in [
            (6u8, false, 0u32, 1usize, 8u32),
            (6, true, 1, 2, 8),
            (3, true, 0, 2, 10),
            (6, true, 0, 3, 8),
            (8, false, 1, 2, 8),
            (10, false, 0, 1, 8),
        ] {
            let mut cfg = EncConfig::new(w, h);
            cfg.rc = RateControl::Qindex(110);
            cfg.speed = speed;
            cfg.loop_restoration = lr;
            cfg.tile_cols_log2 = tiles;
            cfg.threads = threads;
            cfg.bit_depth = bd;
            cfg.gop = 4;
            let mut enc = Av1Enc::new(cfg).expect("encoder");
            let mut units = Vec::new();
            for f in 0..6usize {
                let px = |i: usize, s: usize| -> u16 {
                    let (x, y) = (i % s, i / s);
                    let v = (x * 7 + y * 3 + f * 4) % 200 + ((x / 9 + y / 5 + f) % 3) * 20;
                    (v as u16) << (bd - 8)
                };
                let y: Vec<u16> = (0..w * h).map(|i| px(i, w)).collect();
                let c: Vec<u16> = (0..(w / 2) * (h / 2)).map(|i| px(i * 3, w / 2)).collect();
                let input = InputFrame {
                    planes: [&y[..], &c[..], &c[..]],
                    strides: [w, w / 2, w / 2],
                };
                if bd == 8 {
                    let y8: Vec<u8> = y.iter().map(|&v| v as u8).collect();
                    let c8: Vec<u8> = c.iter().map(|&v| v as u8).collect();
                    let input8 = InputFrame {
                        planes: [&y8[..], &c8[..], &c8[..]],
                        strides: [w, w / 2, w / 2],
                    };
                    units.extend(enc.encode(&input8).expect("encode"));
                } else {
                    units.extend(enc.encode(&input).expect("encode"));
                }
            }
            if bd == 8 {
                units.extend(enc.flush::<u8>().expect("flush"));
            } else {
                units.extend(enc.flush::<u16>().expect("flush"));
            }
            let mut dec = Av1Core::new();
            dec.threads = 1;
            for u in &units {
                let shown = dec.decode(&u.data).expect("decode");
                let got = shown.last().map(|f| samples(&f.picture));
                let want = u.recon.as_deref().map(samples);
                assert!(got == want, "speed {speed} lr {lr} bd {bd}");
            }
        }
    }
}
