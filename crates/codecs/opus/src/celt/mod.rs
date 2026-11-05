//! The CELT layer (RFC 6716 section 4.3, with the RFC 8251 fixes).
//!
//! The decoder runs at 48 kHz internally and writes the API rate by
//! integer decimation in the de-emphasis filter. Internally the signal is
//! in the reference's 32768 scale; [`CeltDecoder::decode`] writes full
//! scale 1.0.
//!
//! The bitstream parse is integer and follows the reference symbol for
//! symbol, so the final range of every frame matches. Reconstruction is
//! single-precision float, like the reference's float build.

mod bands;
mod coder;
mod cwrs;
mod encoder;
mod energy;
mod laplace;
mod mdct;
mod pitch;
mod postfilter;
mod rate;
pub(crate) mod softclip;
mod tables;
mod vq;

#[cfg(test)]
mod tests;

use bands::{AllBands, AntiCollapse};
use coder::Coder;
use mdct::MdctScratch;
use postfilter::Comb;
use rate::{AllocRequest, BITRES};
use tables::{
    EBANDS, NB_EBANDS, OVERLAP, PREEMPH, SHORT_MDCT_SIZE, SPREAD_ICDF, SPREAD_NORMAL, TAPSET_ICDF,
    TF_SELECT_TABLE, TRIM_ICDF, WINDOW,
};

use crate::range::RangeDecoder;
use cutlass_decode::CodecError;

pub(crate) use encoder::{CeltConfig, CeltEncoder, RateMode};

/// Samples of decoded history kept per channel (`DECODE_BUFFER_SIZE`).
const DECODE_BUFFER_SIZE: usize = 2048;
/// Longest pitch period the concealment looks back (`MAX_PERIOD`).
const MAX_PERIOD: usize = 1024;
/// LPC order of the concealment (`LPC_ORDER`).
const LPC_ORDER: usize = 24;
/// Largest frame in bytes.
const MAX_FRAME_BYTES: u32 = 1275;
/// The reference's internal signal scale.
const SIG_SCALE: f32 = 32768.0;

/// CELT decoder state for one Opus stream.
#[derive(Clone, Debug)]
pub(crate) struct CeltDecoder {
    channels: usize,
    downsample: usize,
    /// RFC 8251 section 10: a mono output does not apply the intensity
    /// phase inversion.
    disable_inv: bool,
    /// Per channel: `DECODE_BUFFER_SIZE` samples of output history, then the
    /// `OVERLAP` samples of the next frame's MDCT overlap.
    decode_mem: [Vec<f32>; 2],
    lpc: [[f32; LPC_ORDER]; 2],
    old_band_e: [f32; 2 * NB_EBANDS],
    old_log_e: [f32; 2 * NB_EBANDS],
    old_log_e2: [f32; 2 * NB_EBANDS],
    background_log_e: [f32; 2 * NB_EBANDS],
    preemph_mem: [f32; 2],
    postfilter: Comb,
    postfilter_old: Comb,
    rng: u32,
    loss_count: u32,
    last_pitch_index: usize,
}

/// One CELT frame for [`CeltDecoder::decode`].
pub(crate) struct CeltFrame<'d, 'a> {
    /// The frame's range decoder. In hybrid mode it is the one SILK already
    /// read from; its storage is the frame length CELT may use. `None` asks
    /// for packet-loss concealment of `frame_size` samples.
    pub rd: Option<&'d mut RangeDecoder<'a>>,
    /// Samples per channel at 48 kHz: 120, 240, 480, or 960.
    pub frame_size: usize,
    /// Coded channels in this frame, 1 or 2.
    pub stream_channels: usize,
    /// First coded band: 0, or 17 in hybrid mode.
    pub start_band: usize,
    /// Band after the last coded one: 13, 17, 19, or 21.
    pub end_band: usize,
}

/// TF resolution per band (`tf_decode`).
fn tf_decode(
    start: usize,
    end: usize,
    transient: bool,
    lm: usize,
    rd: &mut RangeDecoder<'_>,
) -> [i32; NB_EBANDS] {
    let mut tf_res = [0i32; NB_EBANDS];
    let mut budget = rd.storage() * 8;
    let mut tell = rd.tell() as u32;
    let mut logp: u32 = if transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut curr = 0i32;
    let mut tf_changed = 0i32;
    for slot in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= i32::from(rd.bit_logp(logp));
            tell = rd.tell() as u32;
            tf_changed |= curr;
        }
        *slot = curr;
        logp = if transient { 4 } else { 5 };
    }
    let t = 4 * usize::from(transient);
    let row = &TF_SELECT_TABLE[lm];
    let mut tf_select = 0usize;
    if tf_select_rsv && row[t + tf_changed as usize] != row[t + 2 + tf_changed as usize] {
        tf_select = usize::from(rd.bit_logp(1));
    }
    for slot in tf_res.iter_mut().take(end).skip(start) {
        *slot = i32::from(row[t + 2 * tf_select + *slot as usize]);
    }
    tf_res
}

/// The inverse MDCT of one channel's spectrum into `out` (N + overlap
/// samples, the first `OVERLAP` of which are added to `overlap_mem` and the
/// last `OVERLAP` of which become the new `overlap_mem`).
fn inverse_mdct(
    freq: &[f32],
    blocks: usize,
    out: &mut [f32],
    overlap_mem: &mut [f32],
    scratch: &mut Synth,
) {
    let n = freq.len();
    let b = blocks.max(1);
    let n2 = n / b;
    let x = &mut scratch.x;
    x.clear();
    x.resize(n + OVERLAP, 0.0);
    let coef = &mut scratch.coef;
    coef.resize(n2, 0.0);
    let y = &mut scratch.y;
    y.resize(2 * n2, 0.0);
    let lead = (n2 - OVERLAP) / 2;
    for blk in 0..b {
        for (k, c) in coef.iter_mut().enumerate() {
            *c = freq[blk + k * b];
        }
        mdct::mdct().inverse(coef, y, &mut scratch.mdct);
        let base = n2 * blk;
        // Rising overlap, flat middle, falling overlap of the low-overlap
        // window, starting `lead` samples into the IMDCT output.
        for i in 0..OVERLAP {
            x[base + i] += WINDOW[i] * y[lead + i];
        }
        for i in OVERLAP..n2 {
            x[base + i] += y[lead + i];
        }
        for i in 0..OVERLAP {
            x[base + n2 + i] += WINDOW[OVERLAP - 1 - i] * y[lead + n2 + i];
        }
    }
    for j in 0..OVERLAP {
        out[j] = x[j] + overlap_mem[j];
    }
    out[OVERLAP..n].copy_from_slice(&x[OVERLAP..n]);
    overlap_mem.copy_from_slice(&x[n..n + OVERLAP]);
}

/// Reusable synthesis buffers.
#[derive(Default)]
struct Synth {
    x: Vec<f32>,
    coef: Vec<f32>,
    y: Vec<f32>,
    mdct: MdctScratch,
}

impl CeltDecoder {
    /// `channels` is the output count, 1 or 2. `api_rate` is 8, 12, 16,
    /// 24, or 48 kHz.
    pub(crate) fn new(channels: usize, api_rate: u32) -> Result<Self, CodecError> {
        let downsample = match api_rate {
            48_000 => 1,
            24_000 => 2,
            16_000 => 3,
            12_000 => 4,
            8_000 => 6,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "Opus API sample rate",
                });
            }
        };
        if !(1..=2).contains(&channels) {
            return Err(CodecError::Unsupported {
                what: "CELT channel count",
            });
        }
        let mut dec = Self {
            channels,
            downsample,
            disable_inv: channels == 1,
            decode_mem: [
                vec![0.0; DECODE_BUFFER_SIZE + OVERLAP],
                vec![0.0; DECODE_BUFFER_SIZE + OVERLAP],
            ],
            lpc: [[0.0; LPC_ORDER]; 2],
            old_band_e: [0.0; 2 * NB_EBANDS],
            old_log_e: [0.0; 2 * NB_EBANDS],
            old_log_e2: [0.0; 2 * NB_EBANDS],
            background_log_e: [0.0; 2 * NB_EBANDS],
            preemph_mem: [0.0; 2],
            postfilter: Comb::default(),
            postfilter_old: Comb::default(),
            rng: 0,
            loss_count: 0,
            last_pitch_index: 0,
        };
        dec.reset();
        Ok(dec)
    }

    /// Drops overlap, energy history, the post-filter, and PLC state.
    pub(crate) fn reset(&mut self) {
        for mem in &mut self.decode_mem {
            mem.fill(0.0);
        }
        self.lpc = [[0.0; LPC_ORDER]; 2];
        self.old_band_e = [0.0; 2 * NB_EBANDS];
        self.old_log_e = [-28.0; 2 * NB_EBANDS];
        self.old_log_e2 = [-28.0; 2 * NB_EBANDS];
        self.background_log_e = [0.0; 2 * NB_EBANDS];
        self.preemph_mem = [0.0; 2];
        self.postfilter = Comb::default();
        self.postfilter_old = Comb::default();
        self.rng = 0;
        self.loss_count = 0;
        self.last_pitch_index = 0;
    }

    /// Decodes one frame into `out`: `frame_size / downsample` interleaved
    /// samples per output channel, full scale 1.0, overwriting `out`.
    pub(crate) fn decode(
        &mut self,
        frame: CeltFrame<'_, '_>,
        out: &mut [f32],
    ) -> Result<(), CodecError> {
        let lm = match frame.frame_size {
            120 => 0,
            240 => 1,
            480 => 2,
            960 => 3,
            _ => {
                return Err(CodecError::InvalidApi {
                    what: "CELT frame size",
                });
            }
        };
        if !(1..=2).contains(&frame.stream_channels) {
            return Err(CodecError::InvalidApi {
                what: "CELT stream channels",
            });
        }
        if frame.start_band >= frame.end_band || frame.end_band > NB_EBANDS {
            return Err(CodecError::InvalidApi {
                what: "CELT band range",
            });
        }
        let n = frame.frame_size;
        let out_len = n / self.downsample * self.channels;
        let Some(out) = out.get_mut(..out_len) else {
            return Err(CodecError::InvalidApi {
                what: "CELT output buffer",
            });
        };
        let mut synth = Synth::default();
        let params = frame_params(&frame, lm);
        match frame.rd {
            Some(rd) if rd.storage() > 1 => {
                if rd.storage() > MAX_FRAME_BYTES {
                    return Err(CodecError::InvalidApi {
                        what: "CELT frame length",
                    });
                }
                self.decode_frame(&params, rd, out, &mut synth);
            }
            _ => self.conceal(&params, out, &mut synth),
        }
        Ok(())
    }

    fn decode_frame(
        &mut self,
        p: &Params,
        rd: &mut RangeDecoder<'_>,
        out: &mut [f32],
        synth: &mut Synth,
    ) {
        let (start, end, lm, c) = (p.start, p.end, p.lm, p.stream_channels);
        let m = 1usize << lm;
        let n = m * SHORT_MDCT_SIZE;
        let len = rd.storage() as i32;

        if c == 1 {
            for i in 0..NB_EBANDS {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[NB_EBANDS + i]);
            }
        }
        let total_bits = len * 8;
        let tell = rd.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            rd.bit_logp(15)
        } else {
            false
        };
        let mut freq = vec![0f32; 2 * n];
        let mut new_post = Comb::default();
        let mut transient = false;
        if silence {
            // The reference pretends every bit was read: no further
            // symbols, zero amplitudes, and the energies drop to -28.
            self.old_band_e = [-28.0; 2 * NB_EBANDS];
        } else {
            let mut tell = tell;
            if start == 0 && tell + 16 <= total_bits {
                if rd.bit_logp(1) {
                    let octave = rd.dec_uint(6);
                    let period = (16u32 << octave) + rd.dec_bits(4 + octave) - 1;
                    let qg = rd.dec_bits(3);
                    let tapset = if rd.tell() + 2 <= total_bits {
                        rd.dec_icdf(&TAPSET_ICDF, 2)
                    } else {
                        0
                    };
                    new_post = Comb {
                        period: period as usize,
                        gain: 0.093_75 * (qg + 1) as f32,
                        tapset,
                    };
                }
                tell = rd.tell();
            }
            if lm > 0 && tell + 3 <= total_bits {
                transient = rd.bit_logp(3);
                tell = rd.tell();
            }
            let intra = tell + 3 <= total_bits && rd.bit_logp(3);
            energy::unquant_coarse(&mut self.old_band_e, start, end, intra, rd, c, lm);
            let tf_res = tf_decode(start, end, transient, lm, rd);
            let tell = rd.tell();
            let spread = if tell + 4 <= total_bits {
                rd.dec_icdf(&SPREAD_ICDF, 5)
            } else {
                SPREAD_NORMAL
            };
            let cap = rate::init_caps(lm, c);
            let mut offsets = [0i32; NB_EBANDS];
            let mut dynalloc_logp = 6;
            let mut total_bits8 = total_bits << BITRES;
            let mut tell = rd.tell_frac() as i32;
            for i in start..end {
                let width = (c * tables::band_width(i, lm)) as i32;
                let quanta = (width << BITRES).min((6 << BITRES).max(width));
                let mut loop_logp = dynalloc_logp;
                let mut boost = 0;
                while tell + (loop_logp << BITRES) < total_bits8 && boost < cap[i] {
                    let flag = rd.bit_logp(loop_logp as u32);
                    tell = rd.tell_frac() as i32;
                    if !flag {
                        break;
                    }
                    boost += quanta;
                    total_bits8 -= quanta;
                    loop_logp = 1;
                }
                offsets[i] = boost;
                if boost > 0 {
                    dynalloc_logp = 2.max(dynalloc_logp - 1);
                }
            }
            let alloc_trim = if tell + (6 << BITRES) <= total_bits8 {
                rd.dec_icdf(&TRIM_ICDF, 7) as i32
            } else {
                5
            };
            let mut bits = ((len * 8) << BITRES) - rd.tell_frac() as i32 - 1;
            let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
                1 << BITRES
            } else {
                0
            };
            bits -= anti_collapse_rsv;
            let req = AllocRequest {
                start,
                end,
                offsets: &offsets,
                cap: &cap,
                alloc_trim,
                intensity: 0,
                dual_stereo: false,
                total: bits,
                channels: c,
                lm,
                prev: 0,
            };
            let alloc = rate::compute_allocation(&req, &mut Coder::Dec(&mut *rd));
            energy::unquant_fine(&mut self.old_band_e, start, end, &alloc.fine_quant, rd, c);

            let mut x = vec![0f32; c * n];
            let (x0, x1) = x.split_at_mut(n);
            let all = AllBands {
                start,
                end,
                band_e: &[],
                pulses: &alloc.pulses,
                short_blocks: transient,
                spread,
                dual_stereo: alloc.dual_stereo,
                intensity: alloc.intensity,
                tf_res: &tf_res,
                total_bits: ((len * 8) << BITRES) - anti_collapse_rsv,
                balance: alloc.balance,
                lm,
                coded_bands: alloc.coded_bands,
                disable_inv: self.disable_inv,
            };
            let y = if c == 2 { Some(&mut *x1) } else { None };
            let collapse =
                bands::quant_all_bands(&all, x0, y, &mut Coder::Dec(&mut *rd), &mut self.rng);
            let anti_collapse_on = anti_collapse_rsv > 0 && rd.dec_bits(1) != 0;
            let left = len * 8 - rd.tell();
            energy::unquant_final(&mut self.old_band_e, start..end, &alloc, left, rd, c);
            if anti_collapse_on {
                bands::anti_collapse(
                    &mut x,
                    &AntiCollapse {
                        collapse: &collapse,
                        lm,
                        channels: c,
                        size: n,
                        bands: start..end,
                        log_e: &self.old_band_e,
                        prev1_log_e: &self.old_log_e,
                        prev2_log_e: &self.old_log_e2,
                        pulses: &alloc.pulses,
                        seed: self.rng,
                    },
                );
            }
            let mut band_e = [0f32; 2 * NB_EBANDS];
            energy::log2amp(&mut band_e, &self.old_band_e, start, end, c);
            for ch in 0..c {
                bands::denormalise(
                    &x[ch * n..(ch + 1) * n],
                    &mut freq[ch * n..(ch + 1) * n],
                    &band_e[ch * NB_EBANDS..],
                    end,
                    m,
                );
            }
        }
        self.synthesize(&mut freq, p, transient, new_post, synth);
        self.update_energy_history(p, transient);
        self.rng = rd.rng();
        self.deemphasis(n, out);
        self.loss_count = 0;
    }

    /// Bands outside the coded range, the stereo mapping, the inverse MDCT,
    /// and the post-filter, into `decode_mem`.
    fn synthesize(
        &mut self,
        freq: &mut [f32],
        p: &Params,
        transient: bool,
        new_post: Comb,
        synth: &mut Synth,
    ) {
        let (start, end, lm, c) = (p.start, p.end, p.lm, p.stream_channels);
        let cc = self.channels;
        let m = 1usize << lm;
        let n = m * SHORT_MDCT_SIZE;
        for mem in self.decode_mem.iter_mut().take(cc) {
            mem.copy_within(n..DECODE_BUFFER_SIZE, 0);
        }
        let mut bound = m * EBANDS[end];
        if self.downsample != 1 {
            bound = bound.min(n / self.downsample);
        }
        for ch in 0..c {
            let f = &mut freq[ch * n..(ch + 1) * n];
            f[..m * EBANDS[start]].fill(0.0);
            f[bound.min(n)..].fill(0.0);
        }
        if cc == 2 && c == 1 {
            freq.copy_within(0..n, n);
        }
        if cc == 1 && c == 2 {
            for i in 0..n {
                freq[i] = 0.5 * (freq[i] + freq[n + i]);
            }
        }
        let blocks = if transient { m } else { 1 };
        let at = DECODE_BUFFER_SIZE - n;
        for ch in 0..cc {
            let mem = &mut self.decode_mem[ch];
            let (hist, overlap) = mem.split_at_mut(DECODE_BUFFER_SIZE);
            inverse_mdct(
                &freq[ch * n..(ch + 1) * n],
                blocks,
                &mut hist[at..],
                overlap,
                synth,
            );
        }
        self.postfilter.period = self.postfilter.period.max(postfilter::MIN_PERIOD);
        self.postfilter_old.period = self.postfilter_old.period.max(postfilter::MIN_PERIOD);
        for ch in 0..cc {
            let mem = &mut self.decode_mem[ch];
            postfilter::comb_filter_inplace(
                mem,
                at,
                SHORT_MDCT_SIZE,
                self.postfilter_old,
                self.postfilter,
                OVERLAP,
            );
            if lm != 0 {
                postfilter::comb_filter_inplace(
                    mem,
                    at + SHORT_MDCT_SIZE,
                    n - SHORT_MDCT_SIZE,
                    self.postfilter,
                    new_post,
                    OVERLAP,
                );
            }
        }
        self.postfilter_old = self.postfilter;
        self.postfilter = new_post;
        if lm != 0 {
            self.postfilter_old = self.postfilter;
        }
    }

    fn update_energy_history(&mut self, p: &Params, transient: bool) {
        let (start, end, c) = (p.start, p.end, p.stream_channels);
        let m = 1u32 << p.lm;
        if c == 1 {
            for i in 0..NB_EBANDS {
                self.old_band_e[NB_EBANDS + i] = self.old_band_e[i];
            }
        }
        if transient {
            for i in 0..2 * NB_EBANDS {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        } else {
            self.old_log_e2 = self.old_log_e;
            self.old_log_e = self.old_band_e;
            for i in 0..2 * NB_EBANDS {
                self.background_log_e[i] =
                    (self.background_log_e[i] + m as f32 * 0.001).min(self.old_band_e[i]);
            }
        }
        for ch in 0..2 {
            for i in (0..start).chain(end..NB_EBANDS) {
                let at = ch * NB_EBANDS + i;
                self.old_band_e[at] = 0.0;
                self.old_log_e[at] = -28.0;
                self.old_log_e2[at] = -28.0;
            }
        }
    }

    /// De-emphasis and decimation of the newest `n` samples into `out`.
    fn deemphasis(&mut self, n: usize, out: &mut [f32]) {
        let cc = self.channels;
        let ds = self.downsample;
        for ch in 0..cc {
            let src = &self.decode_mem[ch][DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE];
            let mut m = self.preemph_mem[ch];
            for (j, &x) in src.iter().enumerate() {
                let tmp = x + m;
                m = PREEMPH * tmp;
                if j % ds == 0
                    && let Some(slot) = out.get_mut((j / ds) * cc + ch)
                {
                    *slot = tmp * (1.0 / SIG_SCALE);
                }
            }
            // Keep denormals and runaway values out of the filter state.
            self.preemph_mem[ch] = if m.is_finite() && m.abs() > 1e-30 {
                m
            } else {
                0.0
            };
        }
    }

    /// Packet-loss concealment (`celt_decode_lost`): pitch-based LPC
    /// extrapolation for the first losses of a CELT-only stream, then noise
    /// at the decaying or background energy.
    fn conceal(&mut self, p: &Params, out: &mut [f32], synth: &mut Synth) {
        let (start, end, lm) = (p.start, p.end, p.lm);
        let cc = self.channels;
        let m = 1usize << lm;
        let n = m * SHORT_MDCT_SIZE;
        if self.loss_count >= 5 || start != 0 {
            let mut band_e = [0f32; 2 * NB_EBANDS];
            if self.loss_count >= 5 {
                energy::log2amp(&mut band_e, &self.background_log_e, start, end, cc);
            } else {
                let decay = if self.loss_count == 0 { 1.5 } else { 0.5 };
                for ch in 0..cc {
                    for i in start..end {
                        self.old_band_e[ch * NB_EBANDS + i] -= decay;
                    }
                }
                energy::log2amp(&mut band_e, &self.old_band_e, start, end, cc);
            }
            let mut seed = self.rng;
            let mut x = vec![0f32; cc * n];
            for ch in 0..cc {
                for (i, &edge) in EBANDS.iter().enumerate().take(NB_EBANDS).skip(start) {
                    let lo = ch * n + (edge << lm);
                    let w = tables::band_width(i, lm);
                    for v in &mut x[lo..lo + w] {
                        seed = bands::lcg_rand(seed);
                        *v = ((seed as i32) >> 20) as f32;
                    }
                    vq::renormalise(&mut x[lo..lo + w], 1.0);
                }
                let from = (EBANDS[end] << lm).min(n);
                x[ch * n + from..(ch + 1) * n].fill(0.0);
            }
            self.rng = seed;
            let mut freq = vec![0f32; cc * n];
            for ch in 0..cc {
                bands::denormalise(
                    &x[ch * n..(ch + 1) * n],
                    &mut freq[ch * n..(ch + 1) * n],
                    &band_e[ch * NB_EBANDS..],
                    NB_EBANDS,
                    m,
                );
                let f = &mut freq[ch * n..(ch + 1) * n];
                f[..EBANDS[start] << lm].fill(0.0);
                let mut bound = EBANDS[end] << lm;
                if self.downsample != 1 {
                    bound = bound.min(n / self.downsample);
                }
                f[bound.min(n)..].fill(0.0);
            }
            let at = DECODE_BUFFER_SIZE - n;
            for ch in 0..cc {
                let mem = &mut self.decode_mem[ch];
                // The reference writes the concealed frame without shifting
                // the history.
                let (hist, overlap) = mem.split_at_mut(DECODE_BUFFER_SIZE);
                inverse_mdct(
                    &freq[ch * n..(ch + 1) * n],
                    1,
                    &mut hist[at..],
                    overlap,
                    synth,
                );
            }
        } else {
            self.conceal_pitch(n);
        }
        self.deemphasis(n, out);
        self.loss_count += 1;
    }

    fn conceal_pitch(&mut self, n: usize) {
        let cc = self.channels;
        let overlap = OVERLAP;
        let mut fade = 1.0f32;
        let pitch_index = if self.loss_count == 0 {
            let mut lp = vec![0f32; DECODE_BUFFER_SIZE >> 1];
            let chans: Vec<&[f32]> = self
                .decode_mem
                .iter()
                .take(cc)
                .map(|m| &m[..DECODE_BUFFER_SIZE])
                .collect();
            pitch::downsample(&chans, &mut lp, DECODE_BUFFER_SIZE);
            let poffset = 720usize;
            let t = pitch::search(
                &lp[poffset >> 1..],
                &lp,
                DECODE_BUFFER_SIZE - poffset,
                poffset - 100,
            );
            let index = poffset.saturating_sub(t).max(1);
            self.last_pitch_index = index;
            index
        } else {
            fade = 0.8;
            self.last_pitch_index.max(1)
        };
        let len = n + overlap;
        let out_at = DECODE_BUFFER_SIZE - MAX_PERIOD;
        for ch in 0..cc {
            let mem = &mut self.decode_mem[ch];
            let mut e = vec![0f32; MAX_PERIOD + 2 * overlap];
            let mut exc: Vec<f32> = mem[out_at..out_at + MAX_PERIOD].to_vec();
            if self.loss_count == 0 {
                let mut ac = [0f32; LPC_ORDER + 1];
                pitch::autocorr(&exc, &mut ac, Some(&WINDOW), LPC_ORDER);
                ac[0] *= 1.0001;
                for (i, a) in ac.iter_mut().enumerate().skip(1) {
                    let f = 0.008 * i as f32;
                    *a -= *a * f * f;
                }
                pitch::lpc(&mut self.lpc[ch], &ac);
            }
            let lpc = self.lpc[ch];
            let mut fmem = [0f32; LPC_ORDER];
            for (i, v) in fmem.iter_mut().enumerate() {
                *v = mem[out_at + MAX_PERIOD - 1 - i];
            }
            pitch::fir(&mut exc, &lpc, &mut fmem);
            let period = pitch_index.min(MAX_PERIOD / 2);
            let mut e1 = 1f32;
            let mut e2 = 1f32;
            for i in 0..period {
                let a = exc[MAX_PERIOD - period + i];
                let b = exc[(MAX_PERIOD).saturating_sub(2 * period) + i];
                e1 += a * a;
                e2 += b * b;
            }
            e1 = e1.min(e2);
            let mut decay = (0.5 * e1 / e2).sqrt();
            let mut offset = MAX_PERIOD as i64 - pitch_index as i64;
            let mut s1 = 0f32;
            for (i, slot) in e.iter_mut().enumerate().take(len + overlap) {
                if offset + i as i64 >= MAX_PERIOD as i64 {
                    offset -= pitch_index as i64;
                    decay *= decay;
                }
                let at = (offset + i as i64).clamp(0, MAX_PERIOD as i64 - 1) as usize;
                *slot = decay * exc[at];
                let tmp = mem[out_at + at];
                s1 += tmp * tmp;
            }
            for (i, v) in fmem.iter_mut().enumerate() {
                *v = mem[out_at + MAX_PERIOD - 1 - i];
            }
            for v in e.iter_mut().take(len + overlap) {
                *v *= fade;
            }
            pitch::iir(&mut e[..len + overlap], &lpc, &mut fmem);
            let s2: f32 = e[..len + overlap].iter().map(|v| v * v).sum();
            // Written to catch NaN too, as the reference does.
            if s1.partial_cmp(&(0.2 * s2)) != Some(std::cmp::Ordering::Greater) {
                e[..len + overlap].fill(0.0);
            } else if s1 < s2 {
                let ratio = ((0.5 * s1 + 1.0) / (s2 + 1.0)).sqrt();
                for v in e.iter_mut().take(len + overlap) {
                    *v *= ratio;
                }
            }
            // Post-filter the previous frame's MDCT overlap, without fading.
            let pf = self.postfilter;
            postfilter::comb_filter_inplace(mem, DECODE_BUFFER_SIZE, overlap, pf, pf, 0);
            mem.copy_within(out_at + n..DECODE_BUFFER_SIZE + overlap, out_at);
            for i in 0..overlap / 2 {
                let tmp = WINDOW[i] * e[n + overlap - 1 - i] + WINDOW[overlap - i - 1] * e[n + i];
                mem[DECODE_BUFFER_SIZE + i] = WINDOW[overlap - i - 1] * tmp;
                mem[DECODE_BUFFER_SIZE + overlap - i - 1] = WINDOW[i] * tmp;
            }
            mem[DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE].copy_from_slice(&e[..n]);
            // Pre-filter the overlap for the next frame, whose post-filter
            // will run over it again.
            let neg = Comb {
                gain: -pf.gain,
                ..pf
            };
            let mut pre = vec![0f32; overlap];
            postfilter::comb_filter_fir(&mut pre, mem, DECODE_BUFFER_SIZE, overlap, neg, neg, 0);
            mem[DECODE_BUFFER_SIZE..DECODE_BUFFER_SIZE + overlap].copy_from_slice(&pre);
        }
    }
}

/// Validated per-frame parameters.
struct Params {
    start: usize,
    end: usize,
    lm: usize,
    stream_channels: usize,
}

fn frame_params(frame: &CeltFrame<'_, '_>, lm: usize) -> Params {
    Params {
        start: frame.start_band,
        end: frame.end_band,
        lm,
        stream_channels: frame.stream_channels,
    }
}
