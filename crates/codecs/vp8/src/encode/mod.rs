//! VP8 encoder, version 0.
//!
//! Key frames code every macroblock intra (16×16 or B_PRED). Inter frames
//! choose per macroblock between intra and NEAREST, NEAR, ZERO, or a
//! searched NEWMV on the last or golden picture, by rate and distortion.
//! The golden picture is refreshed every `golden_interval` frames with a
//! finer quantizer. Coefficient, mode, and flag probabilities follow each
//! frame's counts. The loop-filter level is searched against the source.
//! A bitrate of zero keeps the quantizer fixed; a positive one steers it.

mod cost;
mod frame;
mod residual;
mod search;
mod write;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{CodecError, CodecId, CodecParams, Packet, PixelFormat};

use crate::bool_coder::BoolEncoder;
use crate::decode::{Frame, mb_filter};
use crate::filter::{self, Planes};
use crate::header::{
    self, FilterDeltas, FrameHeader, FrameTag, HeaderWriter, MAX_EDGE, Persist, Probs,
    QuantIndices, Segmentation,
};
use crate::modes::{GOLDEN, INTRA, LAST};
use crate::predict::B_PRED;
use crate::tables::{AC_Q, UV_MODE_TREE, Y_MODE_TREE};
use crate::tokens::Dequant;

use self::cost::{ModeCosts, MvCosts};
use self::frame::{Coded, Setup, Source, code_frame};
use self::residual::{Costs, choose_coeff_probs};
use self::write::{
    choose_mv_probs, choose_tree_probs, count_tokens, flag_prob, write_modes, write_tokens,
};

/// Encoder settings beyond the codec parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vp8Options {
    /// Target bits per second; 0 keeps `quantizer` fixed.
    pub bitrate: u32,
    /// Quantizer index 0..127 when `bitrate` is 0.
    pub quantizer: u8,
    /// Frames from one key frame to the next.
    pub keyframe_interval: u32,
    /// Frames between golden refreshes; 0 never refreshes.
    pub golden_interval: u32,
    /// Token partitions: 1, 2, 4, or 8.
    pub partitions: usize,
    /// 0 is the most thorough; 2 is the fastest.
    pub speed: u8,
    /// Four segments by luma activity: flat areas get a finer quantizer
    /// and busy areas a coarser one. Off by default; it trades PSNR for
    /// detail in smooth areas.
    pub segments: bool,
}

impl Default for Vp8Options {
    fn default() -> Self {
        Self {
            bitrate: 0,
            quantizer: 24,
            keyframe_interval: 150,
            golden_interval: 16,
            partitions: 1,
            speed: 1,
            segments: false,
        }
    }
}

/// Default reference and mode loop-filter deltas (intra, last, golden,
/// altref; B_PRED, ZEROMV, other MV modes, SPLITMV).
const DELTAS: FilterDeltas = FilterDeltas {
    enabled: true,
    ref_frame: [2, 0, -2, -2],
    mode: [4, -2, 2, 4],
};

/// Version-0 VP8 encoder on the video codec contract.
pub struct Vp8Encoder {
    width: u32,
    height: u32,
    opts: Vp8Options,
    persist: Persist,
    last: Option<Arc<Frame>>,
    golden: Option<Arc<Frame>>,
    frame_index: u64,
    since_key: u32,
    since_golden: u32,
    /// Flag probabilities of the last frame, as cost estimates.
    prob_skip: u8,
    prob_intra: u8,
    prob_last: u8,
    prob_golden: u8,
    rate: RateControl,
    /// Segment of each macroblock, kept between map updates.
    segment_map: Vec<u8>,
    cancel: Arc<AtomicBool>,
}

/// Bytes over (positive) or under the target so far, and a size model per
/// frame kind: bytes times quantizer step.
#[derive(Clone, Copy, Debug, Default)]
struct RateControl {
    debt: i64,
    model: [Option<f64>; 3],
    last_q: Option<u8>,
}

const KIND_KEY: usize = 0;
const KIND_GOLDEN: usize = 1;
const KIND_INTER: usize = 2;

impl Vp8Encoder {
    /// Opens with the codec parameters: a zero bitrate is a fixed quantizer.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp8 {
            return Err(CodecError::Unsupported {
                what: "vp8 encoder",
            });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp8 pixel format",
            });
        }
        let opts = Vp8Options {
            bitrate: params.bitrate,
            ..Vp8Options::default()
        };
        Self::with_options(params.width, params.height, opts)
    }

    pub fn with_options(width: u32, height: u32, opts: Vp8Options) -> Result<Self, CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "empty picture",
            });
        }
        if width > MAX_EDGE || height > MAX_EDGE {
            return Err(CodecError::Limit {
                what: "vp8 frame edge",
            });
        }
        if !matches!(opts.partitions, 1 | 2 | 4 | 8) {
            return Err(CodecError::InvalidApi {
                what: "vp8 token partitions",
            });
        }
        if opts.quantizer > 127 {
            return Err(CodecError::InvalidApi {
                what: "vp8 quantizer",
            });
        }
        Ok(Self {
            width,
            height,
            opts,
            persist: Persist::default(),
            last: None,
            golden: None,
            frame_index: 0,
            since_key: 0,
            since_golden: 0,
            prob_skip: 128,
            prob_intra: 64,
            prob_last: 200,
            prob_golden: 128,
            rate: RateControl::default(),
            segment_map: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The reconstruction a decoder holds after the last frame, at the
    /// macroblock-aligned size.
    #[must_use]
    pub fn reconstruction(&self) -> Option<&Frame> {
        self.last.as_deref()
    }

    fn choose_q(&self, kind: usize, target: i64) -> u8 {
        if self.opts.bitrate == 0 {
            let q = i32::from(self.opts.quantizer);
            let q = match kind {
                KIND_KEY => q - 4,
                KIND_GOLDEN => q - 6,
                _ => q,
            };
            return q.clamp(0, 127) as u8;
        }
        let model = self.rate.model[kind]
            .or(self.rate.model[KIND_INTER].map(|m| m * if kind == KIND_INTER { 1.0 } else { 3.0 }))
            .or(self.rate.model[KIND_KEY].map(|m| m * 0.25));
        let Some(model) = model else {
            // No history: pick from bits per pixel.
            let area = f64::from(self.width) * f64::from(self.height);
            let bpp = (target.max(1) as f64 * 8.0) / area;
            let q = (90.0 - 30.0 * (bpp / 0.05).log2()).clamp(4.0, 120.0);
            return q as u8;
        };
        let target = target.max(64) as f64;
        // bytes ≈ model / step.
        let mut best = 127u8;
        for q in 0..=127u8 {
            let step = f64::from(AC_Q[usize::from(q)]);
            if model / step <= target {
                best = q;
                break;
            }
        }
        // Do not swing too far from the last frame of an inter run.
        if kind == KIND_INTER
            && let Some(last) = self.rate.last_q
        {
            let lo = i32::from(last) - 12;
            let hi = i32::from(last) + 12;
            return i32::from(best).clamp(lo.max(0), hi.min(127)) as u8;
        }
        best
    }

    fn encode(&mut self, input: &VideoInput) -> Result<Packet, CodecError> {
        let frame = &input.frame;
        let src = Source::padded(
            &frame.y.samples,
            frame.y.stride,
            &frame.u.samples,
            frame.u.stride,
            &frame.v.samples,
            frame.v.stride,
            self.width as usize,
            self.height as usize,
        );
        let key = self.last.is_none() || self.since_key >= self.opts.keyframe_interval.max(1);
        let golden_refresh =
            !key && self.opts.golden_interval > 0 && self.since_golden >= self.opts.golden_interval;
        let kind = if key {
            KIND_KEY
        } else if golden_refresh {
            KIND_GOLDEN
        } else {
            KIND_INTER
        };
        let per_frame = target_bytes(self.opts.bitrate, input.duration, input.timescale);
        let target = match kind {
            KIND_KEY => per_frame * 5,
            KIND_GOLDEN => per_frame * 2,
            _ => per_frame,
        } - self.rate.debt / 10;
        let q = self.choose_q(kind, target.max(per_frame / 4));
        let base = if key {
            Probs::default()
        } else {
            self.persist.probs.clone()
        };
        let dq = Dequant::new(i32::from(q), &QuantIndices::default());
        let lambda16 = lambda_for(q);
        // Segments: a new map on key and golden frames, the quantizer
        // offsets on every frame so each stays inside 0..127.
        let update_map = self.opts.segments
            && (key || golden_refresh || self.segment_map.len() != src.mbw * src.mbh);
        if update_map {
            self.segment_map = activity_segments(&src);
        }
        let seg_quant: [i8; 4] = std::array::from_fn(|i| {
            (i32::from(SEGMENT_OFFSETS[i])).clamp(-i32::from(q), 127 - i32::from(q)) as i8
        });
        let seg_q: [u8; 4] =
            std::array::from_fn(|i| (i32::from(q) + i32::from(seg_quant[i])) as u8);
        let coeff_costs = Costs::new(&base.coeff);
        let mode_costs = ModeCosts::new(key, &base.y_mode, &base.uv_mode);
        let mv_costs = MvCosts::new(&base.mv);
        let golden_differs = match (&self.golden, &self.last) {
            (Some(g), Some(l)) => !Arc::ptr_eq(g, l),
            _ => false,
        };
        let setup = Setup {
            key,
            dq,
            lambda16,
            coeff_costs: &coeff_costs,
            mode_costs: &mode_costs,
            mv_costs: &mv_costs,
            prob_intra: self.prob_intra,
            prob_last: self.prob_last,
            prob_golden: self.prob_golden,
            last: if key { None } else { self.last.as_deref() },
            golden: if key || !golden_differs {
                None
            } else {
                self.golden.as_deref()
            },
            speed: self.opts.speed,
            segments: self.opts.segments.then_some(self.segment_map.as_slice()),
            seg_dq: seg_q.map(|sq| Dequant::new(i32::from(sq), &QuantIndices::default())),
            seg_lambda16: seg_q.map(lambda_for),
        };
        let mut coded = code_frame(&src, &setup);
        let (mbw, mbh) = (src.mbw, src.mbh);
        let mut h = self.header(key, golden_refresh, q, &base, &coded, mbw, mbh);
        if self.opts.segments {
            let mut counts = [0u32; 4];
            for &seg in &self.segment_map {
                counts[usize::from(seg & 3)] += 1;
            }
            h.segmentation = Segmentation {
                enabled: true,
                update_map,
                update_data: true,
                absolute: false,
                quant: seg_quant,
                filter: [0; 4],
                tree_probs: if update_map {
                    [
                        flag_prob(counts[0] + counts[1], counts[2] + counts[3]),
                        flag_prob(counts[0], counts[1]),
                        flag_prob(counts[2], counts[3]),
                    ]
                } else {
                    [255; 3]
                },
            };
        }
        h.filter_level = search_filter(&src, &coded, &h, q);
        if h.filter_level > 0 {
            let filters: Vec<_> = coded
                .mbs
                .iter()
                .zip(&coded.any)
                .map(|(mb, &any)| mb_filter(&h, mb, any))
                .collect();
            let recon = &mut coded.recon;
            let mut planes = Planes {
                y: &mut recon.y,
                u: &mut recon.u,
                v: &mut recon.v,
                mbw,
                mbh,
            };
            filter::filter_frame(&mut planes, &filters, false);
        }
        if !key {
            // A dry run counts the vector branches the frame codes.
            let mut counts = [[[0u32; 2]; 19]; 2];
            write_modes(&mut BoolEncoder::new(), &h, &coded, mbw, mbh, &mut counts);
            h.probs.mv = choose_mv_probs(&base.mv, &counts);
        }
        let mut first = BoolEncoder::new();
        HeaderWriter {
            header: &h,
            previous: &self.persist.probs,
            previous_deltas: &self.persist.deltas,
        }
        .write(&mut first);
        write_modes(&mut first, &h, &coded, mbw, mbh, &mut [[[0; 2]; 19]; 2]);
        let tokens = write_tokens(&coded, mbw, mbh, &h.probs.coeff, h.partitions);
        let data = header::assemble(&h.tag, &first.finish(), &tokens)?;

        // Keep the decoder's view of the state.
        let parsed = header::parse(&data, &self.persist)?;
        self.persist = parsed.next;
        self.prob_skip = h.prob_skip;
        if !key {
            self.prob_intra = h.prob_intra;
            self.prob_last = h.prob_last;
            self.prob_golden = h.prob_golden;
        }
        let current = Arc::new(coded.recon);
        if key || golden_refresh {
            self.golden = Some(Arc::clone(&current));
            self.since_golden = 0;
        }
        self.last = Some(current);
        if key {
            self.since_key = 0;
        }
        self.since_key = self.since_key.saturating_add(1);
        self.since_golden = self.since_golden.saturating_add(1);
        self.frame_index = self.frame_index.saturating_add(1);
        if self.opts.bitrate > 0 {
            let bytes = data.len() as i64;
            self.rate.debt += bytes - per_frame;
            let step = f64::from(AC_Q[usize::from(q)]);
            let sample = bytes as f64 * step;
            let slot = &mut self.rate.model[kind];
            *slot = Some(slot.map_or(sample, |m| 0.5 * m + 0.5 * sample));
            if kind == KIND_INTER {
                self.rate.last_q = Some(q);
            }
        }
        Ok(Packet::video(
            data,
            input.pts,
            input.duration,
            input.timescale,
            key,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn header(
        &self,
        key: bool,
        golden_refresh: bool,
        q: u8,
        base: &Probs,
        coded: &Coded,
        mbw: usize,
        mbh: usize,
    ) -> FrameHeader {
        let counts = count_tokens(coded, mbw, mbh);
        let mut probs = base.clone();
        probs.coeff = choose_coeff_probs(&base.coeff, &counts);
        let mut skips = [0u32; 2];
        let mut inter = [0u32; 2];
        let mut refs = [0u32; 2];
        let mut gold = [0u32; 2];
        let mut y_modes = [0u32; 5];
        let mut uv_modes = [0u32; 4];
        for mb in &coded.mbs {
            skips[usize::from(mb.skip)] += 1;
            inter[usize::from(mb.ref_frame != INTRA)] += 1;
            if mb.ref_frame == INTRA {
                y_modes[usize::from(mb.y_mode.min(B_PRED))] += 1;
                uv_modes[usize::from(mb.uv_mode & 3)] += 1;
            } else {
                refs[usize::from(mb.ref_frame != LAST)] += 1;
                if mb.ref_frame != LAST {
                    gold[usize::from(mb.ref_frame != GOLDEN)] += 1;
                }
            }
        }
        if !key {
            if let Some(p) = choose_tree_probs(&Y_MODE_TREE, &base.y_mode, &y_modes, 32) {
                probs.y_mode = p;
            }
            if let Some(p) = choose_tree_probs(&UV_MODE_TREE, &base.uv_mode, &uv_modes, 24) {
                probs.uv_mode = p;
            }
        }
        FrameHeader {
            tag: FrameTag {
                key,
                version: 0,
                show: true,
                first_size: 0,
                width: if key { self.width } else { 0 },
                height: if key { self.height } else { 0 },
                hscale: 0,
                vscale: 0,
            },
            color_space: false,
            clamping: false,
            segmentation: Segmentation::default(),
            filter_simple: false,
            filter_level: 0,
            sharpness: 0,
            deltas: DELTAS,
            partitions: self.opts.partitions,
            quant: QuantIndices {
                y_ac: q,
                ..QuantIndices::default()
            },
            refresh_golden: key || golden_refresh,
            refresh_altref: key,
            copy_to_golden: 0,
            copy_to_altref: 0,
            sign_bias_golden: false,
            sign_bias_altref: false,
            refresh_entropy: true,
            refresh_last: true,
            probs,
            skip_enabled: true,
            prob_skip: flag_prob(skips[0], skips[1]),
            prob_intra: if key {
                0
            } else {
                flag_prob(inter[0], inter[1])
            },
            prob_last: if key { 0 } else { flag_prob(refs[0], refs[1]) },
            prob_golden: if key { 0 } else { flag_prob(gold[0], gold[1]) },
        }
    }
}

/// Quantizer offsets of the four activity segments, flattest first.
const SEGMENT_OFFSETS: [i8; 4] = [-8, -3, 2, 6];

/// The rate-distortion multiplier for a quantizer index.
fn lambda_for(q: u8) -> u64 {
    let ac = i64::from(AC_Q[usize::from(q.min(127))]);
    ((ac * ac * 9) / 16).max(1) as u64
}

/// Splits the macroblocks into four equal groups by luma variance.
fn activity_segments(src: &Source) -> Vec<u8> {
    let ys = src.mbw * 16;
    let variance: Vec<u64> = (0..src.mbw * src.mbh)
        .map(|i| {
            let (mbx, mby) = (i % src.mbw, i / src.mbw);
            let (mut sum, mut sq) = (0u64, 0u64);
            for r in 0..16 {
                let at = (mby * 16 + r) * ys + mbx * 16;
                for &p in &src.y[at..at + 16] {
                    sum += u64::from(p);
                    sq += u64::from(p) * u64::from(p);
                }
            }
            sq * 256 - sum * sum
        })
        .collect();
    let mut sorted = variance.clone();
    sorted.sort_unstable();
    let n = sorted.len();
    let cut = [sorted[n / 4], sorted[n / 2], sorted[(3 * n) / 4]];
    variance
        .iter()
        .map(|&v| cut.iter().filter(|&&c| v >= c).count() as u8)
        .collect()
}

/// Picks the loop-filter level whose output is closest to the source.
fn search_filter(src: &Source, coded: &Coded, h: &FrameHeader, q: u8) -> u8 {
    let mbw = coded.recon.mbw;
    let mbh = coded.recon.mbh;
    let error = |level: u8| -> u64 {
        if level == 0 {
            return plane_sse(&coded.recon.y, &src.y)
                + plane_sse(&coded.recon.u, &src.u)
                + plane_sse(&coded.recon.v, &src.v);
        }
        let mut hh = h.clone();
        hh.filter_level = level;
        let filters: Vec<_> = coded
            .mbs
            .iter()
            .zip(&coded.any)
            .map(|(mb, &any)| mb_filter(&hh, mb, any))
            .collect();
        let mut y = coded.recon.y.clone();
        let mut u = coded.recon.u.clone();
        let mut v = coded.recon.v.clone();
        let mut planes = Planes {
            y: &mut y,
            u: &mut u,
            v: &mut v,
            mbw,
            mbh,
        };
        filter::filter_frame(&mut planes, &filters, false);
        plane_sse(&y, &src.y) + plane_sse(&u, &src.u) + plane_sse(&v, &src.v)
    };
    let guess = ((u32::from(q) * 3) / 8 + 4).min(63) as u8;
    let mut best = guess;
    let mut best_err = error(guess);
    let mut tried = vec![guess];
    for step in [8i32, 4, 2, 1] {
        let center = best;
        for dir in [-1, 1] {
            let level = (i32::from(center) + dir * step).clamp(0, 63) as u8;
            if tried.contains(&level) {
                continue;
            }
            tried.push(level);
            let err = error(level);
            if err < best_err {
                best = level;
                best_err = err;
            }
        }
    }
    best
}

fn plane_sse(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

fn target_bytes(bitrate: u32, duration: i64, timescale: u32) -> i64 {
    let ticks = u128::try_from(duration).unwrap_or(0);
    let bits = u128::from(bitrate).saturating_mul(ticks) / u128::from(timescale.max(1));
    i64::try_from(bits / 8).unwrap_or(i64::MAX)
}

impl VideoEncoder for Vp8Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let frame = &input.frame;
        if frame.format != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp8 pixel format",
            });
        }
        if frame.width != self.width || frame.height != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        if self.opts.bitrate > 0 && (input.duration <= 0 || input.timescale == 0) {
            return Err(CodecError::InvalidApi {
                what: "frame clock",
            });
        }
        Ok(vec![self.encode(input)?])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        let opts = self.opts;
        let cancel = Arc::clone(&self.cancel);
        if let Ok(fresh) = Self::with_options(self.width, self.height, opts) {
            *self = fresh;
        }
        self.cancel = cancel;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
