//! The CELT encoder: RFC 6716 section 4.3 run forwards, with the analysis
//! the reference encoder (`celt_encoder.c`) uses to choose what the
//! bitstream leaves open.
//!
//! Per frame: pre-emphasis, the pitch pre-filter, transient detection, the
//! MDCT, band energies, coarse energy with an intra/inter trial, TF
//! analysis, the spreading decision, dynamic allocation, the allocation
//! trim, stereo decisions, rate control, the allocation, fine energy, PVQ
//! band shapes, anti-collapse, and the final energy bits.
//!
//! Only the bitstream-visible parts must match the decoder; everything
//! else here is a choice, written after the reference's float build.

use super::bands::{self, AllBands};
use super::coder::Coder;
use super::energy::{self, CoarseParams};
use super::mdct::{self, MdctScratch};
use super::pitch;
use super::postfilter::{self, Comb};
use super::rate::{self, AllocRequest, BITRES};
use super::tables::{
    E_MEANS, EBANDS, LOGN, NB_EBANDS, OVERLAP, PREEMPH, SHORT_MDCT_SIZE, SPREAD_AGGRESSIVE,
    SPREAD_ICDF, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF,
    WINDOW,
};
use crate::range::{RangeEncoder, ilog};
use cutlass_encode::CodecError;

/// History the pitch pre-filter keeps (`COMBFILTER_MAXPERIOD`).
const PF_HISTORY: usize = postfilter::MAX_PERIOD;
const PF_MIN_PERIOD: usize = postfilter::MIN_PERIOD;
/// Largest CELT frame in bytes.
const MAX_FRAME_BYTES: u32 = 1275;
const EPSILON: f32 = 1e-15;

/// How the frame size follows the bitrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RateMode {
    /// Every frame is exactly the bitrate's size.
    Cbr,
    /// Frames follow the signal; a slow drift correction holds the long-run
    /// mean on the bitrate.
    Vbr,
    /// Like VBR, but a one-frame reservoir bounds how far any stretch can
    /// run over.
    Constrained,
}

/// What an encoder is opened with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CeltConfig {
    pub channels: usize,
    /// 0..=3: 2.5, 5, 10, or 20 ms frames.
    pub lm: usize,
    /// Band after the last coded one: 13, 17, 19, or 21.
    pub end_band: usize,
    /// Bits per second, 6000..=510000.
    pub bitrate: u32,
    pub mode: RateMode,
}

/// CELT encoder state for one mono or stereo stream at 48 kHz.
#[derive(Clone, Debug)]
pub(crate) struct CeltEncoder {
    cfg: CeltConfig,
    preemph_mem: [f32; 2],
    /// The last `OVERLAP` pre-filtered samples per channel.
    in_mem: [[f32; OVERLAP]; 2],
    /// `PF_HISTORY` pre-emphasized samples per channel, before the
    /// pre-filter.
    prefilter_mem: [Vec<f32>; 2],
    /// Largest raw sample in the last `OVERLAP` samples of the previous
    /// frame, for the silence decision.
    tail_max: f32,
    prefilter: Comb,
    old_band_e: [f32; 2 * NB_EBANDS],
    spread_decision: usize,
    tonal_average: i32,
    hf_average: i32,
    tapset_decision: usize,
    intensity: usize,
    last_coded_bands: usize,
    consec_transient: u32,
    stereo_saving: f32,
    spec_avg: f32,
    rng: u32,
    vbr_reservoir: i32,
    vbr_drift: i32,
    vbr_offset: i32,
    vbr_count: i32,
}

/// The analysis of one frame that the coding steps share.
struct Analysis {
    pf_on: bool,
    transient: bool,
    tf_estimate: f32,
    tf_chan: usize,
    /// Linear band amplitudes and their log2 relative to the means.
    band_e: [f32; 2 * NB_EBANDS],
    band_log_e: [f32; 2 * NB_EBANDS],
    /// Unit-norm spectrum, `channels` blocks of `n`.
    x: Vec<f32>,
}

impl CeltEncoder {
    pub(crate) fn new(cfg: CeltConfig) -> Result<Self, CodecError> {
        if !(1..=2).contains(&cfg.channels) {
            return Err(CodecError::Unsupported {
                what: "CELT channel count",
            });
        }
        if cfg.lm > 3 || !matches!(cfg.end_band, 13 | 17 | 19 | 21) {
            return Err(CodecError::InvalidApi {
                what: "CELT frame size or bandwidth",
            });
        }
        if !(6_000..=510_000).contains(&cfg.bitrate) {
            return Err(CodecError::InvalidApi {
                what: "Opus bitrate",
            });
        }
        let mut enc = Self {
            cfg,
            preemph_mem: [0.0; 2],
            in_mem: [[0.0; OVERLAP]; 2],
            prefilter_mem: [vec![0.0; PF_HISTORY], vec![0.0; PF_HISTORY]],
            tail_max: 0.0,
            prefilter: Comb::default(),
            old_band_e: [0.0; 2 * NB_EBANDS],
            spread_decision: SPREAD_NORMAL,
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            intensity: 0,
            last_coded_bands: 0,
            consec_transient: 0,
            stereo_saving: 0.0,
            spec_avg: 0.0,
            rng: 0,
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
        };
        enc.reset();
        Ok(enc)
    }

    /// Drops all signal history and rate-control state.
    pub(crate) fn reset(&mut self) {
        let cfg = self.cfg;
        self.preemph_mem = [0.0; 2];
        self.in_mem = [[0.0; OVERLAP]; 2];
        for mem in &mut self.prefilter_mem {
            mem.fill(0.0);
        }
        self.tail_max = 0.0;
        self.prefilter = Comb::default();
        self.old_band_e = [0.0; 2 * NB_EBANDS];
        self.spread_decision = SPREAD_NORMAL;
        self.tonal_average = 256;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.intensity = 0;
        self.last_coded_bands = 0;
        self.consec_transient = 0;
        self.stereo_saving = 0.0;
        self.spec_avg = 0.0;
        self.rng = 0;
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.cfg = cfg;
    }

    /// Samples per channel in one frame.
    pub(crate) fn frame_size(&self) -> usize {
        SHORT_MDCT_SIZE << self.cfg.lm
    }

    /// The final range of the last frame, which a decoder must reproduce.
    #[cfg(test)]
    pub(crate) fn rng(&self) -> u32 {
        self.rng
    }

    /// Encodes one frame of interleaved full-scale samples
    /// (`frame_size() * channels`) into a CELT frame.
    pub(crate) fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        let c = self.cfg.channels;
        let lm = self.cfg.lm;
        let n = self.frame_size();
        let Some(pcm) = pcm.get(..n * c) else {
            return Err(CodecError::InvalidApi {
                what: "CELT frame length",
            });
        };
        let bitrate = self.cfg.bitrate as i32;
        let n_i = n as i32;
        // Budgets in bytes and 1/8 bits.
        let (mut nb, vbr_rate, effective_bytes) = match self.cfg.mode {
            RateMode::Cbr => {
                let packet = ((bitrate * n_i + 4 * 48_000) / (8 * 48_000)).clamp(3, 1276);
                (packet as u32 - 1, 0, packet - 1)
            }
            RateMode::Vbr | RateMode::Constrained => {
                let den = 48_000 >> BITRES;
                let vbr_rate = (bitrate * n_i + (den >> 1)) / den;
                let max = MAX_FRAME_BYTES >> (3 - lm);
                (max, vbr_rate, vbr_rate >> (3 + BITRES))
            }
        };
        if self.cfg.mode == RateMode::Constrained {
            let allowed = ((2 * vbr_rate - self.vbr_reservoir) >> (BITRES + 3)).max(2);
            nb = nb.min(allowed as u32);
        }
        let mut enc = RangeEncoder::new(nb as usize);

        // Pre-emphasis into the MDCT input (after last frame's overlap) and
        // the pre-filter's history.
        let (mut input, pre) = self.preemphasize(pcm);
        let frame_max = pcm.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tail_from = (n - OVERLAP) * c;
        let silence = frame_max == 0.0 && self.tail_max == 0.0;
        self.tail_max = pcm[tail_from..].iter().fold(0f32, |m, v| m.max(v.abs()));
        let total_bits = nb as i32 * 8;
        let silence = silence && enc.tell() == 1;
        enc.bit_logp(silence, 15);

        if silence {
            return Ok(self.finish_silence(enc, &mut input, &pre, vbr_rate));
        }

        // The pitch pre-filter.
        let enabled = nb as i32 > 12 * c as i32 && enc.tell() + 16 <= total_bits;
        let pf = self.run_prefilter(&pre, &mut input, enabled, effective_bytes);
        if enc.tell() + 16 <= total_bits {
            enc.bit_logp(pf.on, 1);
            if pf.on {
                let p = pf.comb.period as u32 + 1;
                let octave = ilog(p) - 5;
                enc.enc_uint(octave, 6);
                enc.enc_bits(p - (16 << octave), 4 + octave);
                enc.enc_bits(pf.qg, 3);
                enc.enc_icdf(pf.comb.tapset, &TAPSET_ICDF, 2);
            }
        }

        let mut an = self.analyze(&input, enc.tell() + 3 <= total_bits, pf.on);
        if lm > 0 && enc.tell() + 3 <= total_bits {
            enc.bit_logp(an.transient, 3);
        }
        self.code_frame(&mut enc, &mut an, vbr_rate, effective_bytes)?;
        enc.done();
        if enc.error() {
            return Err(CodecError::Limit {
                what: "CELT frame did not fit",
            });
        }
        self.rng = enc.rng();
        Ok(enc.into_bytes())
    }

    /// Pre-emphasis (`celt_preemphasis`): the MDCT input per channel is the
    /// saved overlap then the new samples; the pre-filter input is its
    /// history then the same new samples.
    fn preemphasize(&mut self, pcm: &[f32]) -> ([Vec<f32>; 2], [Vec<f32>; 2]) {
        let c = self.cfg.channels;
        let n = self.frame_size();
        let mut input = [Vec::new(), Vec::new()];
        let mut pre = [Vec::new(), Vec::new()];
        for ch in 0..c {
            let mut m = self.preemph_mem[ch];
            let mut inp = Vec::with_capacity(n + OVERLAP);
            inp.extend_from_slice(&self.in_mem[ch]);
            let mut p = Vec::with_capacity(PF_HISTORY + n);
            p.extend_from_slice(&self.prefilter_mem[ch]);
            for i in 0..n {
                let x = pcm[i * c + ch] * 32768.0;
                let v = (x - m).clamp(-65536.0, 65536.0);
                m = PREEMPH * x;
                p.push(v);
            }
            self.preemph_mem[ch] = m;
            inp.resize(n + OVERLAP, 0.0);
            input[ch] = inp;
            pre[ch] = p;
        }
        (input, pre)
    }

    /// The pitch pre-filter (`run_prefilter`): finds a period and gain,
    /// filters the new samples with the negated comb, and keeps the state.
    fn run_prefilter(
        &mut self,
        pre: &[Vec<f32>; 2],
        input: &mut [Vec<f32>; 2],
        enabled: bool,
        effective_bytes: i32,
    ) -> Prefilter {
        let c = self.cfg.channels;
        let n = self.frame_size();
        let prev = self.prefilter;
        let tapset = self.tapset_decision;
        let (mut period, mut gain) = (PF_MIN_PERIOD, 0f32);
        if enabled {
            let len = PF_HISTORY + n;
            let mut lp = vec![0f32; len >> 1];
            let chans: Vec<&[f32]> = pre.iter().take(c).map(Vec::as_slice).collect();
            pitch::downsample(&chans, &mut lp, len);
            let t = pitch::search(
                &lp[PF_HISTORY >> 1..],
                &lp,
                n,
                PF_HISTORY - 3 * PF_MIN_PERIOD,
            );
            let t0 = (PF_HISTORY - t.min(PF_HISTORY - PF_MIN_PERIOD)) as i32;
            let (p, g) = pitch::remove_doubling(
                &lp,
                PF_HISTORY as i32,
                PF_MIN_PERIOD as i32,
                n as i32,
                t0,
                prev.period as i32,
                prev.gain,
            );
            period = (p.max(PF_MIN_PERIOD as i32) as usize).min(PF_HISTORY - 2);
            gain = 0.7 * g;
            if !gain.is_finite() {
                gain = 0.0;
            }
        }
        let mut threshold = 0.2f32;
        if (period as i32 - prev.period as i32).abs() * 10 > period as i32 {
            threshold += 0.2;
        }
        if effective_bytes < 25 {
            threshold += 0.1;
        }
        if effective_bytes < 35 {
            threshold += 0.1;
        }
        if prev.gain > 0.4 {
            threshold -= 0.1;
        }
        if prev.gain > 0.55 {
            threshold -= 0.1;
        }
        let threshold = threshold.max(0.2);
        let (on, qg) = if gain < threshold || !enabled {
            gain = 0.0;
            (false, 0u32)
        } else {
            if (gain - prev.gain).abs() < 0.1 {
                gain = prev.gain;
            }
            let qg = (((0.5 + gain * 32.0 / 3.0).floor() as i32) - 1).clamp(0, 7);
            gain = 0.093_75 * (qg + 1) as f32;
            (true, qg as u32)
        };
        let new = Comb {
            period,
            gain,
            tapset,
        };
        let neg = |comb: Comb| Comb {
            period: comb.period.max(PF_MIN_PERIOD),
            gain: -comb.gain,
            tapset: comb.tapset,
        };
        for ch in 0..c {
            let mut out = vec![0f32; n];
            postfilter::comb_filter_fir(
                &mut out,
                &pre[ch],
                PF_HISTORY,
                n,
                neg(prev),
                neg(new),
                OVERLAP,
            );
            input[ch][OVERLAP..OVERLAP + n].copy_from_slice(&out);
            self.in_mem[ch].copy_from_slice(&input[ch][n..n + OVERLAP]);
            self.prefilter_mem[ch].copy_from_slice(&pre[ch][n..n + PF_HISTORY]);
        }
        self.prefilter = new;
        Prefilter { on, comb: new, qg }
    }

    /// Transient detection, the MDCT, band energies, and normalization.
    fn analyze(&mut self, input: &[Vec<f32>; 2], transient_room: bool, pf_on: bool) -> Analysis {
        let c = self.cfg.channels;
        let lm = self.cfg.lm;
        let n = self.frame_size();
        let m = 1usize << lm;
        let end = self.cfg.end_band;
        let (detected, tf_estimate, tf_chan) = transient_analysis(&input[..c], n + OVERLAP);
        let transient = lm > 0 && transient_room && detected;
        let mut freq = vec![0f32; c * n];
        let mut scratch = MdctScratch::default();
        for ch in 0..c {
            forward_mdct(
                &input[ch],
                transient.then_some(m),
                &mut freq[ch * n..(ch + 1) * n],
                &mut scratch,
            );
        }
        let mut band_e = [0f32; 2 * NB_EBANDS];
        let mut band_log_e = [-28f32; 2 * NB_EBANDS];
        let mut x = vec![0f32; c * n];
        for ch in 0..c {
            let f = &freq[ch * n..(ch + 1) * n];
            bands::band_energies(f, &mut band_e[ch * NB_EBANDS..], end, m);
            bands::normalise(
                f,
                &mut x[ch * n..(ch + 1) * n],
                &band_e[ch * NB_EBANDS..],
                end,
                m,
            );
        }
        energy::amp2log(&mut band_log_e, &band_e, end, end, c);
        Analysis {
            pf_on,
            transient,
            tf_estimate,
            tf_chan,
            band_e,
            band_log_e,
            x,
        }
    }

    /// Everything after the transient flag.
    fn code_frame(
        &mut self,
        enc: &mut RangeEncoder,
        an: &mut Analysis,
        vbr_rate: i32,
        effective_bytes: i32,
    ) -> Result<(), CodecError> {
        let c = self.cfg.channels;
        let lm = self.cfg.lm;
        let n = self.frame_size();
        let m = 1usize << lm;
        let start = 0usize;
        let end = self.cfg.end_band;
        let total_bits = enc.storage() as i32 * 8;
        let nb_available = enc.storage() as i32;

        // TF resolution.
        let (mut tf_res, tf_select) = if effective_bytes >= 15 * c as i32 {
            let lambda = (1280 / effective_bytes.max(1) + 2).max(5);
            tf_analysis(an, end, lambda, n, lm)
        } else {
            ([i32::from(an.transient); NB_EBANDS], 0)
        };

        // Coarse energy, trying intra and inter coding.
        let mut error = [0f32; 2 * NB_EBANDS];
        self.coarse_energy(enc, an, &mut error, nb_available);

        tf_encode(start, end, an.transient, &mut tf_res, lm, tf_select, enc);

        // Spreading.
        let spread = if enc.tell() + 4 <= total_bits {
            if an.transient || nb_available < 10 * c as i32 {
                self.spread_decision = SPREAD_NORMAL;
            } else {
                self.spread_decision = self.spreading_decision(an, end, an.pf_on && !an.transient);
            }
            enc.enc_icdf(self.spread_decision, &SPREAD_ICDF, 5);
            self.spread_decision
        } else {
            SPREAD_NORMAL
        };

        // Dynamic allocation.
        let cap = rate::init_caps(lm, c);
        let dyn_alloc = dynalloc_analysis(an, end, c, lm, effective_bytes, self.cfg.mode);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6;
        let mut total_bits8 = total_bits << BITRES;
        let mut total_boost = 0i32;
        let mut tell = enc.tell_frac() as i32;
        for i in start..end {
            let width = (c * super::tables::band_width(i, lm)) as i32;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut loop_logp = dynalloc_logp;
            let mut boost = 0;
            let mut j = 0;
            while tell + (loop_logp << BITRES) < total_bits8 - total_boost && boost < cap[i] {
                let flag = j < dyn_alloc.offsets[i];
                enc.bit_logp(flag, loop_logp as u32);
                tell = enc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                loop_logp = 1;
                j += 1;
            }
            if j > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }
        total_bits8 -= total_boost;

        // Stereo decisions.
        let equiv_rate = self.cfg.bitrate as i32 - (40 * c as i32 + 20) * ((400 >> lm) - 50);
        let mut dual_stereo = false;
        if c == 2 {
            const THRESHOLDS: [i32; 21] = [
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
            ];
            const HYSTERESIS: [i32; 21] = [
                1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
            ];
            if lm != 0 {
                dual_stereo = stereo_analysis(&an.x, lm, n);
            }
            self.intensity =
                hysteresis_decision(equiv_rate / 1000, &THRESHOLDS, &HYSTERESIS, self.intensity)
                    .clamp(start, end);
        }

        // Allocation trim.
        let alloc_trim = if enc.tell_frac() as i32 + (6 << BITRES) <= total_bits8 {
            let trim = self.alloc_trim_analysis(an, end, equiv_rate);
            enc.enc_icdf(trim, &TRIM_ICDF, 7);
            trim as i32
        } else {
            5
        };

        // Rate control: pick this frame's size and move the raw bits there.
        if vbr_rate > 0 {
            self.vbr_shrink(enc, an, vbr_rate, total_boost, dyn_alloc.max_depth);
        }
        let len = enc.storage() as i32;

        let mut bits = ((len * 8) << BITRES) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if an.transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
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
            intensity: self.intensity,
            dual_stereo,
            total: bits,
            channels: c,
            lm,
            prev: self.last_coded_bands,
        };
        let alloc = rate::compute_allocation(&req, &mut Coder::Enc(&mut *enc));
        self.last_coded_bands = if self.last_coded_bands > 0 {
            (self.last_coded_bands + 1)
                .min((self.last_coded_bands.saturating_sub(1)).max(alloc.coded_bands))
        } else {
            alloc.coded_bands
        };

        energy::quant_fine(
            &mut self.old_band_e,
            &mut error,
            start,
            end,
            &alloc.fine_quant,
            enc,
            c,
        );

        let all = AllBands {
            start,
            end,
            band_e: &an.band_e,
            pulses: &alloc.pulses,
            short_blocks: an.transient,
            spread,
            dual_stereo: alloc.dual_stereo,
            intensity: alloc.intensity,
            tf_res: &tf_res,
            total_bits: ((len * 8) << BITRES) - anti_collapse_rsv,
            balance: alloc.balance,
            lm,
            coded_bands: alloc.coded_bands,
            disable_inv: false,
        };
        let (x0, x1) = an.x.split_at_mut(n);
        let y = if c == 2 { Some(x1) } else { None };
        bands::quant_all_bands(&all, x0, y, &mut Coder::Enc(&mut *enc), &mut self.rng);

        if anti_collapse_rsv > 0 {
            enc.enc_bits(u32::from(self.consec_transient < 2), 1);
        }
        let left = len * 8 - enc.tell();
        energy::quant_final(
            &mut self.old_band_e,
            &mut error,
            start..end,
            &alloc,
            left,
            enc,
            c,
        );

        self.consec_transient = if an.transient {
            self.consec_transient.saturating_add(1)
        } else {
            0
        };
        for ch in 0..2 {
            for i in (0..start).chain(end..NB_EBANDS) {
                self.old_band_e[ch * NB_EBANDS + i] = 0.0;
            }
        }
        let _ = m;
        Ok(())
    }

    /// A silent frame: the silence flag alone. The pre-filter fades out,
    /// the energies drop to the floor, and VBR spends two bytes.
    fn finish_silence(
        &mut self,
        mut enc: RangeEncoder,
        input: &mut [Vec<f32>; 2],
        pre: &[Vec<f32>; 2],
        vbr_rate: i32,
    ) -> Vec<u8> {
        self.run_prefilter(pre, input, false, 0);
        if vbr_rate > 0 {
            enc.shrink(2);
            let target = (2 * 8) << BITRES;
            if self.cfg.mode == RateMode::Constrained {
                self.vbr_reservoir += target - vbr_rate;
                self.vbr_reservoir = self.vbr_reservoir.max(-vbr_rate * 4);
            }
            self.vbr_count = (self.vbr_count + 1).min(970);
        }
        self.old_band_e = [-28.0; 2 * NB_EBANDS];
        self.consec_transient = 0;
        enc.done();
        self.rng = enc.rng();
        enc.into_bytes()
    }

    /// Codes the coarse energies both intra and inter when there is room
    /// and keeps whichever clamps less, or is smaller (`quant_coarse_energy`).
    fn coarse_energy(
        &mut self,
        enc: &mut RangeEncoder,
        an: &Analysis,
        error: &mut [f32; 2 * NB_EBANDS],
        nb_available: i32,
    ) {
        let c = self.cfg.channels;
        let end = self.cfg.end_band;
        let budget = enc.storage() as i32 * 8;
        let max_decay = if end > 10 {
            (0.125 * nb_available as f32).min(16.0)
        } else {
            16.0
        };
        let params = CoarseParams {
            e_bands: &an.band_log_e,
            start: 0,
            end,
            budget,
            max_decay,
            channels: c,
            lm: self.cfg.lm,
        };
        let two_pass = enc.tell() + 3 <= budget;
        if !two_pass {
            energy::quant_coarse_impl(&params, &mut self.old_band_e, error, false, enc);
            return;
        }
        let start_state = enc.clone();
        let mut old_intra = self.old_band_e;
        let mut err_intra = [0f32; 2 * NB_EBANDS];
        let badness_intra =
            energy::quant_coarse_impl(&params, &mut old_intra, &mut err_intra, true, enc);
        let tell_intra = enc.tell_frac();
        let intra_state = std::mem::replace(enc, start_state);
        let badness_inter =
            energy::quant_coarse_impl(&params, &mut self.old_band_e, error, false, enc);
        if badness_intra < badness_inter
            || (badness_intra == badness_inter && enc.tell_frac() > tell_intra)
        {
            *enc = intra_state;
            self.old_band_e = old_intra;
            *error = err_intra;
        }
    }

    /// How much the band shapes spread (`spreading_decision`): spiky bands
    /// rotate less. Also updates the pre-filter tapset from the high bands.
    fn spreading_decision(&mut self, an: &Analysis, end: usize, update_hf: bool) -> usize {
        let c = self.cfg.channels;
        let m = 1usize << self.cfg.lm;
        let n0 = self.frame_size();
        if m * (EBANDS[end] - EBANDS[end - 1]) <= 8 {
            return SPREAD_NONE;
        }
        let mut sum = 0i32;
        let mut nb_bands = 0i32;
        let mut hf_sum = 0i32;
        for ch in 0..c {
            for i in 0..end {
                let n = m * (EBANDS[i + 1] - EBANDS[i]);
                if n <= 8 {
                    continue;
                }
                let x = &an.x[ch * n0 + m * EBANDS[i]..ch * n0 + m * EBANDS[i] + n];
                let mut tcount = [0i32; 3];
                for &v in x {
                    let x2n = v * v * n as f32;
                    tcount[0] += i32::from(x2n < 0.25);
                    tcount[1] += i32::from(x2n < 0.0625);
                    tcount[2] += i32::from(x2n < 0.015_625);
                }
                if i > NB_EBANDS - 4 {
                    hf_sum += 32 * (tcount[1] + tcount[0]) / n as i32;
                }
                let n_i = n as i32;
                let tmp = i32::from(2 * tcount[2] >= n_i)
                    + i32::from(2 * tcount[1] >= n_i)
                    + i32::from(2 * tcount[0] >= n_i);
                sum += tmp * 256;
                nb_bands += 1;
            }
        }
        if update_hf {
            let den = c as i32 * (4 - NB_EBANDS as i32 + end as i32);
            if hf_sum != 0 && den > 0 {
                hf_sum /= den;
            }
            self.hf_average = (self.hf_average + hf_sum) >> 1;
            let mut hf = self.hf_average;
            match self.tapset_decision {
                2 => hf += 4,
                0 => hf -= 4,
                _ => {}
            }
            self.tapset_decision = if hf > 22 {
                2
            } else if hf > 18 {
                1
            } else {
                0
            };
        }
        if nb_bands == 0 {
            return SPREAD_NORMAL;
        }
        let sum = (sum / nb_bands + self.tonal_average) >> 1;
        self.tonal_average = sum;
        let sum = (3 * sum + (((3 - self.spread_decision as i32) << 7) + 64) + 2) >> 2;
        if sum < 80 {
            SPREAD_AGGRESSIVE
        } else if sum < 256 {
            SPREAD_NORMAL
        } else if sum < 384 {
            SPREAD_LIGHT
        } else {
            SPREAD_NONE
        }
    }

    /// The allocation trim (`alloc_trim_analysis`): tilt toward the low
    /// bands for bright, correlated, or transient frames.
    fn alloc_trim_analysis(&mut self, an: &Analysis, end: usize, equiv_rate: i32) -> usize {
        let c = self.cfg.channels;
        let lm = self.cfg.lm;
        let n0 = self.frame_size();
        let mut trim = if equiv_rate < 64_000 {
            4.0
        } else if equiv_rate < 80_000 {
            4.0 + ((equiv_rate - 64_000) >> 10) as f32 / 16.0
        } else {
            5.0f32
        };
        if c == 2 {
            let inner = |i: usize| -> f32 {
                let lo = EBANDS[i] << lm;
                let hi = EBANDS[i + 1] << lm;
                an.x[lo..hi]
                    .iter()
                    .zip(&an.x[n0 + lo..n0 + hi])
                    .map(|(a, b)| a * b)
                    .sum()
            };
            let mut sum = 0f32;
            for i in 0..8 {
                sum += inner(i);
            }
            let sum = (sum / 8.0).abs().min(1.0);
            let mut min_xc = sum;
            for i in 8..self.intensity.min(end) {
                min_xc = min_xc.min(inner(i).abs());
            }
            let min_xc = min_xc.abs().min(1.0);
            let log_xc = energy::log2(1.001 - sum * sum);
            let log_xc2 = (0.5 * log_xc).max(energy::log2(1.001 - min_xc * min_xc));
            trim += (0.75 * log_xc).max(-4.0);
            self.stereo_saving = (self.stereo_saving + 0.25).min(-0.5 * log_xc2);
        }
        let mut diff = 0f32;
        for ch in 0..c {
            for i in 0..end - 1 {
                diff += an.band_log_e[i + ch * NB_EBANDS] * (2 + 2 * i as i32 - end as i32) as f32;
            }
        }
        diff /= (c * (end - 1)) as f32;
        trim -= ((diff + 1.0) / 6.0).clamp(-2.0, 2.0);
        trim -= 2.0 * an.tf_estimate;
        let index = (0.5 + trim).floor();
        if index.is_finite() {
            (index as i32).clamp(0, 10) as usize
        } else {
            5
        }
    }

    /// Chooses this frame's size from the analysis and the long-run rate,
    /// then shrinks the coder to it (the VBR part of `celt_encode_with_ec`
    /// and `compute_vbr`).
    fn vbr_shrink(
        &mut self,
        enc: &mut RangeEncoder,
        an: &Analysis,
        vbr_rate: i32,
        total_boost: i32,
        max_depth: f32,
    ) {
        let c = self.cfg.channels as i32;
        let lm = self.cfg.lm;
        let end = self.cfg.end_band;
        let lm_diff = 3 - lm as i32;
        let bitrate = self.cfg.bitrate as i32;
        let constrained = self.cfg.mode == RateMode::Constrained;
        let base_target = vbr_rate - ((40 * c + 20) << BITRES) + (self.vbr_offset >> lm_diff);

        // Temporal VBR: louder than the running average spends more.
        let offset = if an.transient { 0.5 * lm as f32 } else { 0.0 };
        let mut follow = -10f32;
        let mut frame_avg = 0f32;
        for i in 0..end {
            follow = (follow - 1.0).max(an.band_log_e[i] - offset);
            if c == 2 {
                follow = follow.max(an.band_log_e[i + NB_EBANDS] - offset);
            }
            frame_avg += follow;
        }
        frame_avg /= end as f32;
        let temporal_vbr = (frame_avg - self.spec_avg).clamp(-1.5, 3.0);
        self.spec_avg += 0.02 * temporal_vbr;

        let coded_bands = if self.last_coded_bands > 0 {
            self.last_coded_bands
        } else {
            NB_EBANDS
        };
        let mut coded_bins = (EBANDS[coded_bands] << lm) as i32;
        if c == 2 {
            coded_bins += (EBANDS[self.intensity.min(coded_bands)] << lm) as i32;
        }
        let mut target = base_target as f32;
        if c == 2 {
            let stereo_bands = self.intensity.min(coded_bands);
            let dof = ((EBANDS[stereo_bands] << lm) as i32 - stereo_bands as i32) as f32;
            let max_frac = 0.8 * dof / coded_bins.max(1) as f32;
            let saving = self.stereo_saving.min(1.0);
            target -= (max_frac * target).min((saving - 0.1) * (dof * 8.0));
        }
        target += (total_boost - (19 << lm)) as f32;
        target += (an.tf_estimate - 0.044) * target;
        let bins = (EBANDS[NB_EBANDS - 2] << lm) as f32;
        let floor_depth = (c as f32 * bins * 8.0 * max_depth).max(target / 4.0);
        target = target.min(floor_depth);
        if constrained {
            target = base_target as f32 + 0.67 * (target - base_target as f32);
        }
        if an.tf_estimate < 0.2 {
            let amount = 0.000_003_1 * (96_000 - bitrate).clamp(0, 32_000) as f32;
            target += temporal_vbr * amount * target;
        }
        let target = if target.is_finite() {
            (target as i32).min(2 * base_target)
        } else {
            base_target
        };

        let tell = enc.tell_frac() as i32;
        let target = target + tell;
        let min_allowed = ((tell + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
        let max_bytes = enc.storage() as i32;
        let mut nb_available = ((target + (1 << (BITRES + 2))) >> (BITRES + 3))
            .max(min_allowed)
            .min(max_bytes);
        let delta = target - vbr_rate;
        let target = nb_available << (BITRES + 3);
        let alpha = if self.vbr_count < 970 {
            self.vbr_count += 1;
            1.0 / (self.vbr_count + 20) as f32
        } else {
            0.001
        };
        if constrained {
            self.vbr_reservoir += target - vbr_rate;
        }
        // Both VBR modes correct their drift so the long-run mean holds.
        self.vbr_drift +=
            (alpha * ((delta << lm_diff) - self.vbr_offset - self.vbr_drift) as f32) as i32;
        self.vbr_offset = -self.vbr_drift;
        if constrained && self.vbr_reservoir < 0 {
            nb_available += -self.vbr_reservoir / (8 << BITRES);
            self.vbr_reservoir = 0;
        }
        let nb = nb_available.clamp(min_allowed.min(max_bytes), max_bytes);
        enc.shrink(nb as u32);
    }
}

/// The pre-filter decision for one frame.
struct Prefilter {
    on: bool,
    comb: Comb,
    qg: u32,
}

/// One channel's MDCT (`compute_mdcts`): one long block, or `blocks` short
/// ones interleaved bin by bin.
fn forward_mdct(input: &[f32], blocks: Option<usize>, out: &mut [f32], s: &mut MdctScratch) {
    let n = out.len();
    let b = blocks.unwrap_or(1).max(1);
    let n2 = n / b;
    let lead = (n2 - OVERLAP) / 2;
    let mut x = vec![0f32; 2 * n2];
    let mut coef = vec![0f32; n2];
    for blk in 0..b {
        let seg = &input[blk * n2..blk * n2 + n2 + OVERLAP];
        x.fill(0.0);
        for i in 0..OVERLAP {
            x[lead + i] = seg[i] * WINDOW[i];
            x[lead + n2 + i] = seg[n2 + i] * WINDOW[OVERLAP - 1 - i];
        }
        x[lead + OVERLAP..lead + n2].copy_from_slice(&seg[OVERLAP..n2]);
        mdct::mdct().forward(&x, &mut coef, s);
        for (k, &v) in coef.iter().enumerate() {
            out[blk + k * b] = v;
        }
    }
}

/// Pre-echo detection (`transient_analysis`): a high-passed energy
/// envelope with forward and backward masking, compared with its
/// harmonic mean. Returns the decision, the TF estimate, and the channel
/// that decided.
fn transient_analysis(input: &[Vec<f32>], len: usize) -> (bool, f32, usize) {
    const INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2,
    ];
    let len2 = len / 2;
    let mut tmp = vec![0f32; len];
    let mut mask_metric = 0i32;
    let mut tf_chan = 0usize;
    for (ch, x) in input.iter().enumerate() {
        let (mut mem0, mut mem1) = (0f32, 0f32);
        for i in 0..len {
            let v = x[i];
            let y = mem0 + v;
            mem0 = mem1 + y - 2.0 * v;
            mem1 = v - 0.5 * y;
            tmp[i] = y;
        }
        tmp[..12].fill(0.0);
        let mut mean = 0f32;
        mem0 = 0.0;
        for i in 0..len2 {
            let x2 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
            mean += x2;
            tmp[i] = mem0 + 0.0625 * (x2 - mem0);
            mem0 = tmp[i];
        }
        mem0 = 0.0;
        let mut max_e = 0f32;
        for i in (0..len2).rev() {
            tmp[i] = mem0 + 0.125 * (tmp[i] - mem0);
            mem0 = tmp[i];
            max_e = max_e.max(mem0);
        }
        let mean = (mean * max_e * 0.5 * len2 as f32).sqrt();
        let norm = len2 as f32 / (EPSILON + mean);
        let mut unmask = 0i32;
        let mut i = 12;
        while i + 5 < len2 {
            let id = (64.0 * norm * (tmp[i] + EPSILON)).floor();
            let id = if id.is_finite() {
                id.clamp(0.0, 127.0) as usize
            } else {
                0
            };
            unmask += i32::from(INV_TABLE[id]);
            i += 4;
        }
        let unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17)).max(1);
        if unmask > mask_metric {
            tf_chan = ch;
            mask_metric = unmask;
        }
    }
    let transient = mask_metric > 200;
    let tf_max = ((27 * mask_metric) as f32).sqrt() - 42.0;
    let tf_max = tf_max.max(0.0);
    let tf_estimate = (0.0069 * tf_max.min(163.0) - 0.139).max(0.0).sqrt();
    (transient, tf_estimate, tf_chan)
}

/// L1 norm with a bias toward frequency resolution (`l1_metric`).
fn l1_metric(x: &[f32], lm: i32, bias: f32) -> f32 {
    let l1: f32 = x.iter().map(|v| v.abs()).sum();
    l1 + lm as f32 * bias * l1
}

/// Per-band TF choice by the L1 norm under each Haar depth, smoothed with
/// a Viterbi pass (`tf_analysis`). Returns the raw per-band flags and
/// `tf_select`.
fn tf_analysis(
    an: &Analysis,
    len: usize,
    lambda: i32,
    n0: usize,
    lm: usize,
) -> ([i32; NB_EBANDS], usize) {
    let transient = an.transient;
    let lm_i = lm as i32;
    let bias = 0.04 * (-0.25f32).max(0.5 - an.tf_estimate);
    let mut metric = [0i32; NB_EBANDS];
    for i in 0..len {
        let n = (EBANDS[i + 1] - EBANDS[i]) << lm;
        let narrow = EBANDS[i + 1] - EBANDS[i] == 1;
        let base = an.tf_chan * n0 + (EBANDS[i] << lm);
        let mut tmp = an.x[base..base + n].to_vec();
        let mut best_l1 = l1_metric(&tmp, if transient { lm_i } else { 0 }, bias);
        let mut best_level = 0i32;
        if transient && !narrow {
            let mut tmp1 = tmp.clone();
            bands::haar1(&mut tmp1, n >> lm, 1 << lm);
            let l1 = l1_metric(&tmp1, lm_i + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }
        let levels = lm + usize::from(!(transient || narrow));
        for k in 0..levels {
            let b = if transient {
                lm_i - k as i32 - 1
            } else {
                k as i32 + 1
            };
            bands::haar1(&mut tmp, n >> k, 1 << k);
            let l1 = l1_metric(&tmp, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }
        metric[i] = if transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm_i) {
            metric[i] -= 1;
        }
    }
    let t = 4 * usize::from(transient);
    let row = &TF_SELECT_TABLE[lm];
    let cost = |sel: usize, bit: usize, i: usize| -> i32 {
        (metric[i] - 2 * i32::from(row[t + 2 * sel + bit])).abs()
    };
    let mut selcost = [0i32; 2];
    for (sel, slot) in selcost.iter_mut().enumerate() {
        let mut cost0 = cost(sel, 0, 0);
        let mut cost1 = cost(sel, 1, 0) + if transient { 0 } else { lambda };
        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0 + cost(sel, 0, i);
            cost1 = curr1 + cost(sel, 1, i);
        }
        *slot = cost0.min(cost1);
    }
    let tf_select = usize::from(selcost[1] < selcost[0] && transient);
    let mut path0 = [0i32; NB_EBANDS];
    let mut path1 = [0i32; NB_EBANDS];
    let mut cost0 = cost(tf_select, 0, 0);
    let mut cost1 = cost(tf_select, 1, 0) + if transient { 0 } else { lambda };
    for i in 1..len {
        let (from0, from1) = (cost0, cost1 + lambda);
        let curr0 = if from0 < from1 {
            path0[i] = 0;
            from0
        } else {
            path0[i] = 1;
            from1
        };
        let (from0, from1) = (cost0 + lambda, cost1);
        let curr1 = if from0 < from1 {
            path1[i] = 0;
            from0
        } else {
            path1[i] = 1;
            from1
        };
        cost0 = curr0 + cost(tf_select, 0, i);
        cost1 = curr1 + cost(tf_select, 1, i);
    }
    let mut tf_res = [0i32; NB_EBANDS];
    tf_res[len - 1] = i32::from(cost0 >= cost1);
    for i in (0..len - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }
    (tf_res, tf_select)
}

/// Codes the TF flags (`tf_encode`) and turns them into resolution
/// changes, as the decoder will.
fn tf_encode(
    start: usize,
    end: usize,
    transient: bool,
    tf_res: &mut [i32; NB_EBANDS],
    lm: usize,
    tf_select: usize,
    enc: &mut RangeEncoder,
) {
    let mut budget = enc.storage() * 8;
    let mut tell = enc.tell() as u32;
    let mut logp: u32 = if transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= u32::from(tf_select_rsv);
    let mut curr = 0i32;
    let mut tf_changed = 0i32;
    for slot in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            enc.bit_logp((*slot ^ curr) != 0, logp);
            tell = enc.tell() as u32;
            curr = *slot;
            tf_changed |= curr;
        } else {
            *slot = curr;
        }
        logp = if transient { 4 } else { 5 };
    }
    let t = 4 * usize::from(transient);
    let row = &TF_SELECT_TABLE[lm];
    let tf_select =
        if tf_select_rsv && row[t + tf_changed as usize] != row[t + 2 + tf_changed as usize] {
            enc.bit_logp(tf_select != 0, 1);
            tf_select
        } else {
            0
        };
    for slot in tf_res.iter_mut().take(end).skip(start) {
        *slot = i32::from(row[t + 2 * tf_select + (*slot & 1) as usize]);
    }
}

/// The dynamic allocation boosts and the deepest band below the noise
/// floor (`dynalloc_analysis`).
struct DynAlloc {
    offsets: [i32; NB_EBANDS],
    max_depth: f32,
}

fn median_of_5(x: &[f32]) -> f32 {
    let mut v = [x[0], x[1], x[2], x[3], x[4]];
    v.sort_by(f32::total_cmp);
    v[2]
}

fn median_of_3(x: &[f32]) -> f32 {
    let mut v = [x[0], x[1], x[2]];
    v.sort_by(f32::total_cmp);
    v[1]
}

fn dynalloc_analysis(
    an: &Analysis,
    end: usize,
    c: usize,
    lm: usize,
    effective_bytes: i32,
    mode: RateMode,
) -> DynAlloc {
    let mut offsets = [0i32; NB_EBANDS];
    let mut noise_floor = [0f32; NB_EBANDS];
    for (i, slot) in noise_floor.iter_mut().enumerate().take(end) {
        *slot = 0.0625 * LOGN[i] as f32 + 0.5 + (9 - 24) as f32 - E_MEANS[i]
            + 0.0062 * ((i + 5) * (i + 5)) as f32;
    }
    let mut max_depth = -31.9f32;
    for ch in 0..c {
        for (i, floor) in noise_floor.iter().enumerate().take(end) {
            max_depth = max_depth.max(an.band_log_e[ch * NB_EBANDS + i] - floor);
        }
    }
    let log_e = &an.band_log_e;
    if effective_bytes > 50 && lm >= 1 && end >= 5 {
        let mut follower = [0f32; 2 * NB_EBANDS];
        for ch in 0..c {
            let e = &log_e[ch * NB_EBANDS..ch * NB_EBANDS + NB_EBANDS];
            let f = &mut follower[ch * NB_EBANDS..ch * NB_EBANDS + NB_EBANDS];
            let mut last = 0;
            f[0] = e[0];
            for i in 1..end {
                if e[i] > e[i - 1] + 0.5 {
                    last = i;
                }
                f[i] = (f[i - 1] + 1.5).min(e[i]);
            }
            for i in (0..last).rev() {
                f[i] = f[i].min((f[i + 1] + 2.0).min(e[i]));
            }
            for i in 2..end - 2 {
                f[i] = f[i].max(median_of_5(&e[i - 2..]) - 1.0);
            }
            let tmp = median_of_3(e) - 1.0;
            f[0] = f[0].max(tmp);
            f[1] = f[1].max(tmp);
            let tmp = median_of_3(&e[end - 3..]) - 1.0;
            f[end - 2] = f[end - 2].max(tmp);
            f[end - 1] = f[end - 1].max(tmp);
            for i in 0..end {
                f[i] = f[i].max(noise_floor[i]);
            }
        }
        let mut boost_db = [0f32; NB_EBANDS];
        for i in 0..end {
            boost_db[i] = if c == 2 {
                let r = follower[NB_EBANDS + i].max(follower[i] - 4.0);
                let l = follower[i].max(r - 4.0);
                0.5 * ((log_e[i] - l).max(0.0) + (log_e[NB_EBANDS + i] - r).max(0.0))
            } else {
                (log_e[i] - follower[i]).max(0.0)
            };
        }
        let halve = mode != RateMode::Vbr && !an.transient;
        let limited = mode == RateMode::Cbr || (mode == RateMode::Constrained && !an.transient);
        let mut tot_boost = 0i32;
        for i in 0..end {
            let mut f = boost_db[i];
            if halve {
                f *= 0.5;
            }
            if i < 8 {
                f *= 2.0;
            }
            if i >= 12 {
                f *= 0.5;
            }
            let f = f.min(4.0);
            let width = (c * ((EBANDS[i + 1] - EBANDS[i]) << lm)) as i32;
            let (boost, boost_bits) = if width < 6 {
                let b = f as i32;
                (b, (b * width) << BITRES)
            } else if width > 48 {
                let b = (f * 8.0) as i32;
                (b, ((b * width) << BITRES) / 8)
            } else {
                let b = (f * width as f32 / 6.0) as i32;
                (b, (b * 6) << BITRES)
            };
            // CBR and steady constrained frames spend at most a quarter
            // of the frame on boosts.
            if limited && (tot_boost + boost_bits) >> BITRES >> 3 > effective_bytes / 4 {
                break;
            }
            offsets[i] = boost;
            tot_boost += boost_bits;
        }
    }
    DynAlloc { offsets, max_depth }
}

/// Whether mid/side beats left/right on the low bands by the L1 norm
/// (`stereo_analysis`); true asks for dual stereo.
fn stereo_analysis(x: &[f32], lm: usize, n0: usize) -> bool {
    let mut sum_lr = EPSILON;
    let mut sum_ms = EPSILON;
    for j in 0..EBANDS[13] << lm {
        let (l, r) = (x[j], x[n0 + j]);
        sum_lr += l.abs() + r.abs();
        sum_ms += (l + r).abs() + (l - r).abs();
    }
    let sum_ms = std::f32::consts::FRAC_1_SQRT_2 * sum_ms;
    let thetas = if lm <= 1 { 13 - 8 } else { 13 };
    ((EBANDS[13] << (lm + 1)) + thetas) as f32 * sum_ms > (EBANDS[13] << (lm + 1)) as f32 * sum_lr
}

/// A threshold choice that sticks to the previous one near the edges
/// (`hysteresis_decision`).
fn hysteresis_decision(val: i32, thresholds: &[i32], hysteresis: &[i32], prev: usize) -> usize {
    let n = thresholds.len();
    let mut i = thresholds.iter().position(|&t| val < t).unwrap_or(n);
    if i > prev && prev < n && val < thresholds[prev] + hysteresis[prev] {
        i = prev;
    }
    if i < prev && prev > 0 && prev <= n && val > thresholds[prev - 1] - hysteresis[prev - 1] {
        i = prev;
    }
    i
}

#[cfg(test)]
mod tests;
