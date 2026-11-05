//! Frame-level encoding: headers, tiles, packets and references.
//!
//! Every frame is coded by the decoder's own tile code running with a writing
//! [`Coder`]. The finished temporal unit is then decoded by an internal
//! [`Av1Core`], which produces the reference frames (with the in-loop filters)
//! exactly as any decoder will and checks every packet.

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
use crate::dec::tile::{RefView, TileDec};
use crate::dsp::Pixel;
use crate::spec::tables::{AC_QLOOKUP, DC_QLOOKUP};

use super::bitwriter::write_obu;
use super::msac_enc::MsacEnc;
use super::obu_write::{
    FrameHeaderWriteOptions, uniform_tile_info, write_frame_header, write_sequence_header,
    write_tile_group,
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
    /// Distance between key frames.
    pub keyint: u32,
    /// Mini-GOP length: frames coded as a hierarchical pyramid behind a
    /// hidden alt-ref (1 = each frame predicts from the previous one only).
    pub gop: u32,
    /// Threads for tile-parallel coding and the internal decoder.
    pub threads: usize,
    /// Denoise alt-refs with their neighbours (coded with an overlay frame).
    pub temporal_filter: bool,
    /// Doubles the mini-GOP where the content barely changes.
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
            keyint: 120,
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
    /// Last reconstructed frame (shown output of the internal decoder).
    last_recon: Option<Arc<AnyPicture>>,
    /// Source pictures waiting to be coded, in display order from `next_disp`.
    lookahead: std::collections::VecDeque<Arc<AnyPicture>>,
    next_disp: u64,
    /// Display index of the frame each reference slot holds.
    slot_disp: [Option<u64>; 8],
    /// Temporal importance of the frames of the current mini-GOP (display
    /// index, importance), from the TPL model.
    tpl_imp: Vec<(u64, f64)>,
    /// The model's block statistics and the display index of each of its
    /// frames, for per-superblock quantizers.
    tpl: Option<TplGroup>,
}

/// log2 of the delta q step of per-superblock quantizers.
const DELTA_Q_RES: u8 = 2;

/// TPL block statistics and the display index of each frame they cover.
type TplGroup = (super::tpl::TplStats, Vec<u64>);

/// A global motion type and its parameters.
type GmModel = (u8, [i32; 6]);

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
        let seq = SequenceHeader {
            profile,
            operating_points: vec![OperatingPoint {
                idc: 0,
                seq_level_idx: 31,
                ..OperatingPoint::default()
            }],
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
        if cfg.threads > 1
            && cfg.tile_cols_log2 == 0
            && cfg.tile_rows_log2 == 0
            && cfg.width >= 1280
        {
            // One tile per thread: columns at least 512 samples wide, then
            // rows at least 512 samples high.
            let mut l = 0;
            while (2usize << l) <= cfg.threads && (cfg.width >> (l + 1)) >= 512 && l < 6 {
                l += 1;
            }
            cfg.tile_cols_log2 = l;
            let mut r = 0;
            while (2usize << (l + r)) <= cfg.threads && (cfg.height >> (r + 1)) >= 512 && r < 6 {
                r += 1;
            }
            cfg.tile_rows_log2 = r;
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
            last_recon: None,
            lookahead: std::collections::VecDeque::new(),
            next_disp: 0,
            slot_disp: [None; 8],
            tpl_imp: Vec::new(),
            tpl: None,
        })
    }

    pub fn sequence_header(&self) -> &SequenceHeader {
        &self.seq
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
        let src = source_picture(&self.cfg, &self.seq, input);
        self.lookahead.push_back(Arc::new(P::wrap(src)));
        self.run::<P>(false)
    }

    /// Codes every queued picture.
    pub fn flush<P: PicturePixel>(&mut self) -> R<Vec<EncodedUnit>> {
        self.run::<P>(true)
    }

    fn use_tf(&self) -> bool {
        self.cfg.gop > 1 && self.cfg.temporal_filter
    }

    fn is_key_disp(&self, d: u64) -> bool {
        d.is_multiple_of(u64::from(self.cfg.keyint.max(1)))
    }

    fn run<P: PicturePixel>(&mut self, flush: bool) -> R<Vec<EncodedUnit>> {
        let mut units = Vec::new();
        while !self.lookahead.is_empty() {
            let d = self.next_disp;
            if self.is_key_disp(d) {
                // The key frame's quantizer follows how much of the next
                // mini-GOP is predicted from it.
                let keyint = u64::from(self.cfg.keyint.max(1));
                // (A single-frame GOP codes every picture without delay.)
                let want = if self.cfg.gop > 1 {
                    u64::from(self.cfg.gop).min(keyint - 1) + 1
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
                continue;
            }
            let gop = u64::from(self.cfg.gop.max(1));
            let keyint = u64::from(self.cfg.keyint.max(1));
            let to_key = keyint - d % keyint;
            // Static content takes mini-GOPs two or four times as long.
            let longest = if self.cfg.adaptive_gop && gop >= 4 {
                gop << ev("AV1_GOPL", 1.0) as u32
            } else {
                gop
            };
            if (self.lookahead.len() as u64) < longest.min(to_key) && !flush {
                break;
            }
            let mut n = gop.min(to_key).min(self.lookahead.len() as u64);
            let mut long = longest;
            while long > gop {
                let ln = long.min(to_key).min(self.lookahead.len() as u64);
                if ln > gop && self.is_static::<P>(d - 1, ln) {
                    n = ln;
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
        let stats = super::tpl::analyze(&frames, &[1], self.cfg.bit_depth);
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
                let spec = self.plan_frame(disp, key, show, level);
                let idx = (disp - self.next_disp) as usize;
                let Some(src) = self.lookahead.get(idx).cloned() else {
                    return Err(CodecError::InvalidApi { what: "lookahead" });
                };
                let Some(src) = P::view(&src) else {
                    return Err(CodecError::InvalidApi {
                        what: "sample size",
                    });
                };
                let filtered;
                let mut src = src;
                if !show && level == 0 && self.use_tf() {
                    // Denoise the alt-ref with its neighbours in the lookahead.
                    let mut others = Vec::new();
                    for k in idx.saturating_sub(3)..=idx + 3 {
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
                let frame = self.encode_frame(src, &spec)?;
                write_obu(tu, OBU_FRAME, 0, 0, false, &frame);
                for (i, s) in self.slot_disp.iter_mut().enumerate() {
                    if spec.refresh & (1 << i) != 0 {
                        *s = Some(disp);
                    }
                }
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
        let stats = super::tpl::analyze(&frames, &order, self.cfg.bit_depth);
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
                let target = step * imp.powf(-alpha).max(ev("AV1_TPLMIN", 0.3));
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
        // Pull the long-run average toward the target as well.
        let avg = self.bits_total as f64 / self.frame_num.max(1) as f64;
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

    fn encode_frame<P: PicturePixel>(&mut self, src: &Picture<P>, spec: &FrameSpec) -> R<Vec<u8>> {
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
        intent.allow_screen_content_tools =
            self.cfg.speed < 10 && is_screen_content(src, self.cfg.bit_depth);
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
        let mode = ModeParams {
            lambda_satd: ev("AV1_LS", 0.8) * qstep * lscale.sqrt(),
            min_block: ev("AV1_MINB", if speed >= 8 { 16.0 } else { 8.0 }) as usize,
            max_block: ev("AV1_MAXB", 64.0) as usize,
            skip_sse_per_px: ev("AV1_SK", 4.0) * qstep * qstep * lscale,
            inter_frame: !key,
            me_range: ev("AV1_MER", if speed >= 8 { 16.0 } else { 48.0 }) as i32,
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
            intra_refine_ratio: ev("AV1_IRR", if speed >= 5 { 1.0 } else { 0.0 }),
            intra_sweep_ratio: ev("AV1_IRS", if speed >= 5 { 2.0 } else { 0.0 }),
            inter_clear_ratio: ev("AV1_ICR", if speed >= 5 { 1.2 } else { 0.0 }),
            est_skip_ratio: ev("AV1_EST", if speed >= 5 { 2.0 } else { 0.0 }),
            exact_rates: ev("AV1_XR", 1.0) > 0.0,
            inter_rd_cands: ev("AV1_IRK", if speed >= 8 { 1.0 } else { 2.0 }) as usize,
            inter_rd_ratio: ev("AV1_IRKR", 1.1),
            inter_rd_modes: ev("AV1_IRM", 1.0) as usize,
            intra_split_pred: ev("AV1_ITXP", 1.0) > 0.0,
            inter_flag_bits: ev("AV1_IIB", 1.0) > 0.0,
            exact_intra_rates: ev("AV1_XRI", 1.0) > 0.0,
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
                ev("AV1_TXS", if speed >= 7 { 1.0 } else { 3.0 }) as u8
            },
            tx_type_keep: ev("AV1_TXK", 3.0) as u8,
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
        let n_above = fs.mi_stride + 64;
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
        let mut tile_bytes: Vec<MsacEnc> = Vec::with_capacity(ti.cols * ti.rows);
        let run_tile = |tile: usize, fs: &mut FrameState, cur: &mut Picture<P>| -> R<MsacEnc> {
            let tile_row = tile / ti.cols;
            let tile_col = tile % ti.cols;
            let mut tile_cdf = Box::new((*cdf).clone());
            let ref_views: [Option<RefView<'_, P>>; 8] = std::array::from_fn(|i| {
                if i == 0 || hdr.frame_is_intra {
                    return None;
                }
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                let pic = ref_pics[slot].as_deref()?;
                let p = P::view(pic)?;
                Some(RefView {
                    pic: p,
                    mirror: None,
                    upscaled_width: refs_hdr[slot].upscaled_width as usize,
                    frame_height: refs_hdr[slot].frame_height as usize,
                    frame_width: refs_hdr[slot].frame_width as usize,
                })
            });
            let mut td = TileDec {
                seq: &seq,
                hdr: &hdr,
                fs,
                cur,
                refs: ref_views,
                mirrors: Vec::new(),
                msac: {
                    let mut m = MsacEnc::new(!hdr.disable_cdf_update);
                    if cdef_fb || lr_on {
                        m.record();
                    }
                    m
                },
                pb: Default::default(),
                plan: crate::dec::tile::EncPlan {
                    sb_q: sb_q.clone(),
                    sb_cols,
                    base_lambda: (qp.lambda, mode.lambda_satd, mode.skip_sse_per_px),
                    ..Default::default()
                },
                plan_q: 0,
                cfl_choice: (0, 0),
                intra_choice: (0, None),
                est_hint: (0, 0),
                cur_filters: [0, 0],
                plan_delta_lf: [0; 4],
                src: Some(src),
                enc: qp,
                mode,
                me_scratch: Default::default(),
                cdf: &mut tile_cdf,
                mi_row_start: ti.mi_row_starts[tile_row] as usize,
                mi_row_end: ti.mi_row_starts[tile_row + 1] as usize,
                mi_col_start: ti.mi_col_starts[tile_col] as usize,
                mi_col_end: ti.mi_col_starts[tile_col + 1] as usize,
                sb_size: if seq.use_128x128_superblock { 15 } else { 12 },
                sb_size4: if seq.use_128x128_superblock { 32 } else { 16 },
                bd: seq.bit_depth,
                sub_x: seq.subsampling_x,
                sub_y: seq.subsampling_y,
                num_planes: seq.num_planes(),
                above_level: std::array::from_fn(|_| vec![0u8; n_above]),
                above_dc: std::array::from_fn(|_| vec![0u8; n_above]),
                above_seg_pred: vec![0u8; n_above],
                left_level: [[0; 32]; 3],
                left_dc: [[0; 32]; 3],
                left_seg_pred: [0; 32],
                above_palette: std::array::from_fn(|_| vec![[0u16; 8]; n_above]),
                left_palette: [[[0; 8]; 32]; 2],
                delta_lf: [0; 4],
                current_q_index: i32::from(hdr.quant.base_q_idx),
                read_deltas: false,
                ref_lr_wiener: [[[0; 3]; 2]; 3],
                ref_sgr_xqd: [[0; 2]; 3],
                block_decoded: [[[false; 35]; 35]; 3],
                b: Default::default(),
                color_map_y: vec![0u8; 128 * 128],
                color_map_uv: vec![0u8; 128 * 128],
                scratch: Default::default(),
                rec: None,
            };
            td.decode_tile()?;
            let TileDec { msac, .. } = td;
            Ok(msac)
        };
        let n_tiles = ti.cols * ti.rows;
        let threads = self.cfg.threads.max(1).min(n_tiles);
        if threads <= 1 {
            for tile in 0..n_tiles {
                tile_bytes.push(run_tile(tile, &mut fs, &mut cur)?);
            }
        } else {
            // Tiles are independent: each thread codes its tiles into its own
            // frame state and picture, whose tile areas are merged afterwards.
            let proto = &fs;
            let mk_pic = || {
                Picture::<P>::new(
                    hdr.frame_width as usize,
                    hdr.frame_height as usize,
                    seq.num_planes(),
                    seq.subsampling_x,
                    seq.subsampling_y,
                    seq.bit_depth,
                )
            };
            type TileOut<P> = (Vec<(usize, MsacEnc)>, FrameState, Picture<P>);
            let outs: Vec<R<TileOut<P>>> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let run_tile = &run_tile;
                        let mk_pic = &mk_pic;
                        let (seq, hdr) = (&seq, &hdr);
                        sc.spawn(move || -> R<TileOut<P>> {
                            let mut f = FrameState::new(seq, hdr);
                            f.motion_field = proto.motion_field.clone();
                            let mut c = mk_pic();
                            let mut out = Vec::new();
                            for tile in (t..n_tiles).step_by(threads) {
                                out.push((tile, run_tile(tile, &mut f, &mut c)?));
                            }
                            Ok((out, f, c))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or(Err(CodecError::InvalidApi {
                            what: "tile thread",
                        }))
                    })
                    .collect()
            });
            let mut bytes: Vec<Option<MsacEnc>> = (0..n_tiles).map(|_| None).collect();
            for o in outs {
                let (tiles, f, c) = o?;
                for (tile, b) in tiles {
                    let r0 = ti.mi_row_starts[tile / ti.cols] as usize;
                    let r1 = ti.mi_row_starts[tile / ti.cols + 1] as usize;
                    let c0 = ti.mi_col_starts[tile % ti.cols] as usize;
                    let c1 = ti.mi_col_starts[tile % ti.cols + 1] as usize;
                    merge_tile(&mut fs, &f, &mut cur, &c, r0, r1, c0, c1);
                    bytes[tile] = Some(b);
                }
            }
            tile_bytes = bytes.into_iter().flatten().collect();
        }
        let use_wiener_cdf = cdf.use_wiener;
        drop(cdf);
        // Choose the in-loop filter strengths on the reconstruction. Tile
        // syntax does not depend on them (one CDEF strength, no deltas), so
        // only the header is rewritten.
        let mut hdr_bytes = hdr_bytes;
        let mut cdef_choice: Option<super::cdef_search::CdefChoice> = None;
        let mut lr_plan: Option<super::lr_search::LrPlan> = None;
        if self.cfg.speed < 8 && !hdr.coded_lossless && !hdr.allow_intrabc {
            let full = self.cfg.speed < 4;
            let threads = self.core.threads.max(1);
            let mut best_h = intent.clone();
            let base_lvl = intent.loop_filter.level[0];
            let mut best_lf: Option<(u64, Picture<P>)> = None;
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
            for l in levels {
                let mut h2 = hdr.clone();
                h2.loop_filter.level = [l, l, l, l];
                let mut pic = cur.clone();
                crate::dec::postfilter::loop_filter(&seq, &h2, &fs, &mut pic, threads);
                let e = picture_sse(&pic, src);
                if best_lf.as_ref().is_none_or(|b| e < b.0) {
                    best_h.loop_filter.level = [l, l, l, l];
                    best_lf = Some((e, pic));
                }
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
            } else if let Some((lf_sse, deblocked)) = best_lf.as_ref()
                && seq.enable_cdef
            {
                let uv_search = seq.num_planes() > 1 && ev("AV1_CDUV", 1.0) > 0.0;
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
                // other one (of at least 4). The strengths are tried in
                // parallel; the choice does not depend on the thread count.
                let sb_rows = fs.mi_rows.div_ceil(16);
                let step = if full || sb_rows < 4 { 1 } else { 2 };
                let rows: Vec<usize> = (0..sb_rows).step_by(step).collect();
                let sy = usize::from(seq.subsampling_y);
                let mut best_c = if step == 1 {
                    (*lf_sse, None::<(u8, u8)>)
                } else {
                    (sb_rows_sse(deblocked, src, &rows, sy), None)
                };
                let eval = |&(pri, sec): &(u8, u8), out: &mut Picture<P>| -> u64 {
                    let mut h2 = hdr.clone();
                    h2.cdef.y_pri = [pri; 8];
                    h2.cdef.y_sec = [sec; 8];
                    // With the separate chroma search, luma alone is
                    // filtered here (chroma then adds a constant).
                    let (up, us) = if uv_search {
                        (0, 0)
                    } else {
                        (pri / 2, sec.min(2) / 2 * 2)
                    };
                    h2.cdef.uv_pri = [up; 8];
                    h2.cdef.uv_sec = [us; 8];
                    let mut pads = crate::dec::postfilter::cdef_pads();
                    copy_sb_rows(out, deblocked, &rows, sy);
                    for &i in &rows {
                        crate::dec::postfilter::cdef_sb_rows(
                            &seq,
                            &h2,
                            &fs,
                            deblocked,
                            out,
                            i * 16,
                            (i + 1) * 16,
                            &mut pads,
                        );
                    }
                    sb_rows_sse(out, src, &rows, sy)
                };
                let nt = threads.min(combos.len()).max(1);
                let errs: Vec<u64> = if nt <= 1 {
                    let mut out = deblocked.clone();
                    combos.iter().map(|c| eval(c, &mut out)).collect()
                } else {
                    let parts: Vec<Vec<(usize, u64)>> = std::thread::scope(|sc| {
                        let hs: Vec<_> = (0..nt)
                            .map(|t| {
                                let (eval, combos) = (&eval, &combos);
                                sc.spawn(move || {
                                    let mut out = deblocked.clone();
                                    (t..combos.len())
                                        .step_by(nt)
                                        .map(|k| (k, eval(&combos[k], &mut out)))
                                        .collect()
                                })
                            })
                            .collect();
                        hs.into_iter()
                            .map(|h| h.join().unwrap_or_default())
                            .collect()
                    });
                    let mut e = vec![u64::MAX; combos.len()];
                    for (k, v) in parts.into_iter().flatten() {
                        e[k] = v;
                    }
                    e
                };
                for (&c, &e) in combos.iter().zip(&errs) {
                    if e < best_c.0 {
                        best_c = (e, Some(c));
                    }
                }
                match best_c.1 {
                    Some((pri, sec)) => {
                        best_h.cdef.y_pri = [pri; 8];
                        best_h.cdef.y_sec = [sec; 8];
                        best_h.cdef.uv_pri = [pri / 2; 8];
                        best_h.cdef.uv_sec = [sec.min(2) / 2 * 2; 8];
                    }
                    None => {
                        best_h.cdef.y_pri = [0; 8];
                        best_h.cdef.y_sec = [0; 8];
                        best_h.cdef.uv_pri = [0; 8];
                        best_h.cdef.uv_sec = [0; 8];
                    }
                }
                if uv_search {
                    // Chroma strengths on their own (the chroma filter does
                    // not touch luma).
                    let cur_uv = (best_h.cdef.uv_pri[0], best_h.cdef.uv_sec[0]);
                    let mut h2 = hdr.clone();
                    h2.cdef = best_h.cdef.clone();
                    // Luma off: only chroma is measured.
                    h2.cdef.y_pri = [0; 8];
                    h2.cdef.y_sec = [0; 8];
                    let mut best_uv = (u64::MAX, cur_uv);
                    let mut out = deblocked.clone();
                    for up in [0u8, 1, 2, 4, 6] {
                        for us in [0u8, 2] {
                            h2.cdef.uv_pri = [up; 8];
                            h2.cdef.uv_sec = [us; 8];
                            // Chroma still holds the deblocked samples for
                            // (0, 0), which comes first; every later
                            // candidate rewrites the same blocks.
                            crate::dec::postfilter::cdef(
                                &seq, &h2, &fs, deblocked, &mut out, threads,
                            );
                            let e = chroma_sse(&out, src);
                            if e < best_uv.0 {
                                best_uv = (e, (up, us));
                            }
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
            }
            hdr_bytes = write_frame_header(&best_h, &ctx, &opts);
        }
        let tile_bytes: Vec<Vec<u8>> = tile_bytes
            .into_iter()
            .map(|m| {
                let mut lw = lr_plan
                    .as_ref()
                    .map(|p| super::lr_search::LrWriter::new(p, use_wiener_cdf, &seq));
                m.replay(&mut |tag: u32, e: &mut MsacEnc| {
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
                })
            })
            .collect();
        // Frame OBU: header, then one tile group with every tile.
        let mut payload = hdr_bytes;
        payload.extend_from_slice(&write_tile_group(
            &ti,
            0,
            ti.cols * ti.rows - 1,
            &tile_bytes,
        ));
        Ok(payload)
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

/// Copies the tile area (mode info rows r0..r1, columns c0..c1, in 4x4
/// units) of a tile's frame state and picture into the frame's.
fn merge_tile<P: Pixel>(
    fs: &mut FrameState,
    tf: &FrameState,
    cur: &mut Picture<P>,
    tc: &Picture<P>,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) {
    let st = fs.mi_stride;
    for r in r0..r1 {
        let (a, b) = (r * st + c0, r * st + c1);
        fs.mi[a..b].copy_from_slice(&tf.mi[a..b]);
        fs.uv_modes[a..b].copy_from_slice(&tf.uv_modes[a..b]);
        fs.tx_types[a..b].copy_from_slice(&tf.tx_types[a..b]);
        fs.inter_tx_sizes[a..b].copy_from_slice(&tf.inter_tx_sizes[a..b]);
        fs.segment_ids[a..b].copy_from_slice(&tf.segment_ids[a..b]);
    }
    for (p, pl) in cur.planes.iter_mut().enumerate() {
        let sx = usize::from(pl.w < tc.planes[0].w && p > 0);
        let sy = usize::from(pl.h < tc.planes[0].h && p > 0);
        let (pr0, pr1) = (r0 >> sy, r1.div_ceil(1 << sy));
        let (pc0, pc1) = (c0 >> sx, c1.div_ceil(1 << sx));
        for r in pr0..pr1 {
            let (a, b) = (r * st + pc0, r * st + pc1);
            fs.lf_tx[p][a..b].copy_from_slice(&tf.lf_tx[p][a..b]);
        }
        let tp = &tc.planes[p];
        let (y0, y1) = ((pr0 * 4).min(pl.alloc_h), (pr1 * 4).min(pl.alloc_h));
        let (x0, x1) = ((pc0 * 4).min(pl.stride), (pc1 * 4).min(pl.stride));
        for y in y0..y1 {
            pl.data[y * pl.stride + x0..y * pl.stride + x1]
                .copy_from_slice(&tp.data[y * tp.stride + x0..y * tp.stride + x1]);
        }
    }
    let cs = fs.cdef_stride;
    for r in r0 / 16..r1.div_ceil(16) {
        let (a, b) = (r * cs + c0 / 16, r * cs + c1.div_ceil(16));
        if b <= fs.cdef_idx.len() {
            fs.cdef_idx[a..b].copy_from_slice(&tf.cdef_idx[a..b]);
        }
    }
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

/// SSE of the 64-row luma areas `rows` (and the chroma rows under them,
/// `sy` the vertical chroma subsampling).
fn sb_rows_sse<P: Pixel>(a: &Picture<P>, b: &Picture<P>, rows: &[usize], sy: usize) -> u64 {
    let mut e = 0;
    for (p, (pa, pb)) in a.planes.iter().zip(&b.planes).enumerate() {
        let s = if p > 0 { sy } else { 0 };
        for &i in rows {
            e += plane_rows_sse(pa, pb, (i * 64) >> s, ((i + 1) * 64) >> s);
        }
    }
    e
}

/// Copies the 64-row luma areas `rows` (and the chroma rows under them).
fn copy_sb_rows<P: Pixel>(dst: &mut Picture<P>, src: &Picture<P>, rows: &[usize], sy: usize) {
    for (p, (pd, ps)) in dst.planes.iter_mut().zip(&src.planes).enumerate() {
        let s = if p > 0 { sy } else { 0 };
        for &i in rows {
            let y0 = ((i * 64) >> s).min(ps.alloc_h);
            let y1 = (((i + 1) * 64) >> s).min(ps.alloc_h);
            let (a, b) = (y0 * ps.stride, y1 * ps.stride);
            pd.data[a..b].copy_from_slice(&ps.data[a..b]);
        }
    }
}

/// Squared error of the chroma planes.
fn chroma_sse<P: Pixel>(a: &Picture<P>, b: &Picture<P>) -> u64 {
    let mut e = 0u64;
    for (pa, pb) in a.planes.iter().zip(&b.planes).skip(1) {
        for y in 0..pa.h {
            let ra = &pa.data[y * pa.stride..][..pa.w];
            let rb = &pb.data[y * pb.stride..][..pa.w];
            for (x, z) in ra.iter().zip(rb) {
                let d = i64::from(x.to_i32() - z.to_i32());
                e += (d * d) as u64;
            }
        }
    }
    e
}

/// Screen content (text, graphics) has many blocks of few, contrasting
/// colors; palette coding pays off there.
fn is_screen_content<P: Pixel>(src: &Picture<P>, bd: u32) -> bool {
    let p = &src.planes[0];
    let thr = 32 << (bd - 8);
    let (mut few, mut total) = (0u32, 0u32);
    let mut vals = [0i32; 256];
    for by in (0..p.h.saturating_sub(15)).step_by(16) {
        for bx in (0..p.w.saturating_sub(15)).step_by(16) {
            let mut n = 0;
            for i in 0..16 {
                for j in 0..16 {
                    vals[n] = p.data[(by + i) * p.stride + bx + j].to_i32();
                    n += 1;
                }
            }
            let v = &mut vals[..n];
            v.sort_unstable();
            let (lo, hi) = (v[0], v[n - 1]);
            let mut distinct = 1;
            for k in 1..n {
                if v[k] != v[k - 1] {
                    distinct += 1;
                }
            }
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

#[cfg(test)]
mod tests {
    use super::{Av1Enc, EncConfig, InputFrame, RateControl};

    fn encode(threads: usize) -> Vec<u8> {
        let (w, h) = (160usize, 96usize);
        let mut cfg = EncConfig::new(w, h);
        cfg.rc = RateControl::Qindex(90);
        cfg.tile_cols_log2 = 1;
        cfg.tile_rows_log2 = 1;
        cfg.threads = threads;
        cfg.gop = 4;
        let mut enc = Av1Enc::new(cfg).expect("encoder");
        let mut out = Vec::new();
        for f in 0..5usize {
            let y: Vec<u8> = (0..w * h)
                .map(|i| ((i % w) * 3 + (i / w) * 2 + f * 5) as u8)
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

    #[test]
    fn tile_threads_do_not_change_the_bitstream() {
        assert_eq!(encode(1), encode(3));
    }
}
