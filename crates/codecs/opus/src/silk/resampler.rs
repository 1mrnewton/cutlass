//! The SILK decoder's resampler from the internal rate (8, 12, or 16 kHz)
//! to the API rate (8, 12, 16, 24, or 48 kHz): copy, 2x allpass
//! upsampling, 2x upsampling plus FIR interpolation, or an AR2 filter
//! plus FIR decimation, behind a short delay line that aligns the modes.
//! RFC 6716 section 4.2.9 describes it as non-normative; this is the
//! reference `resampler.c`, `resampler_private_up2_HQ.c`,
//! `resampler_private_IIR_FIR.c`, `resampler_private_down_FIR.c`, and
//! `resampler_private_AR2.c`, with the RFC 8251 section 5 buffer fix,
//! so the output matches the reference decoder sample for sample.

use super::fixed::{lshift, rshift_round, sat16, smlabb, smlawb, smulbb, smulwb, smulww};

/// `silk_resampler_up2_hq_0` (resampler_rom.c).
const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
/// `silk_resampler_up2_hq_1` (resampler_rom.c).
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

/// `silk_Resampler_3_4_COEFS` (resampler_rom.c): two AR2 coefficients,
/// then three phases of 9 FIR taps.
const RESAMPLER_3_4_COEFS: [i16; 2 + 3 * 9] = [
    -20694, -13867, //
    -49, 64, 17, -157, 353, -496, 163, 11047, 22205, //
    -39, 6, 91, -170, 186, 23, -896, 6336, 19928, //
    -19, -36, 102, -89, -24, 328, -951, 2568, 15909,
];
/// `silk_Resampler_2_3_COEFS` (resampler_rom.c).
const RESAMPLER_2_3_COEFS: [i16; 2 + 2 * 9] = [
    -14457, -14019, //
    64, 128, -122, 36, 310, -768, 584, 9267, 17733, //
    12, 128, 18, -142, 288, -117, -865, 4123, 14459,
];
/// `silk_Resampler_1_2_COEFS` (resampler_rom.c): half of a symmetric
/// 24-tap FIR.
const RESAMPLER_1_2_COEFS: [i16; 2 + 12] = [
    616, -14323, //
    -10, 39, 58, -46, -84, 120, 184, -315, -541, 1284, 5380, 9024,
];
/// `silk_resampler_frac_FIR_12` (resampler_rom.c): 12 phases of half of
/// an 8-tap interpolation filter.
const FRAC_FIR_12: [[i32; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

/// `delay_matrix_dec` (resampler.c): input delay that equalizes the total
/// delay, by input rate (8, 12, 16 kHz) and output rate (8, 12, 16, 24,
/// 48 kHz).
const DELAY_MATRIX_DEC: [[usize; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];

/// `RESAMPLER_MAX_BATCH_SIZE_MS`.
const MAX_BATCH_SIZE_MS: usize = 10;
/// `RESAMPLER_MAX_BATCH_SIZE_IN` for the decoder's 16 kHz maximum input.
const MAX_BATCH_SIZE_IN: usize = MAX_BATCH_SIZE_MS * 16;
/// `RESAMPLER_ORDER_FIR_12`.
const ORDER_FIR_12: usize = 8;
/// `RESAMPLER_DOWN_ORDER_FIR0`, used for 3:4 and 2:3.
const DOWN_ORDER_FIR0: usize = 18;
/// `RESAMPLER_DOWN_ORDER_FIR1`, used for 1:2.
const DOWN_ORDER_FIR1: usize = 24;

/// Which resampler runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    /// Same rate.
    #[default]
    Copy,
    /// Exactly twice the rate: allpass 2x upsampler.
    Up2,
    /// Any other upsampling: 2x upsampler then 12-phase FIR.
    IirFir,
    /// Downsampling: AR2 filter then a polyphase FIR.
    DownFir,
}

/// `silk_resampler_state_struct` for the decoder.
#[derive(Clone, Debug, Default)]
pub(crate) struct Resampler {
    kind: Kind,
    s_iir: [i32; 6],
    /// FIR history of the up-and-interpolate path, 16-bit as RFC 8251
    /// section 5 corrects.
    s_fir_i16: [i16; ORDER_FIR_12],
    /// FIR history of the down path.
    s_fir_i32: [i32; DOWN_ORDER_FIR1],
    delay_buf: [i16; 16],
    batch_size: usize,
    inv_ratio_q16: i32,
    fir_order: usize,
    fir_fracs: i32,
    fs_in_khz: usize,
    fs_out_khz: usize,
    input_delay: usize,
    coefs: &'static [i16],
}

/// Index of a rate in the delay matrix: 8, 12, 16, 24, 48 kHz to 0..5.
fn rate_id(hz: i32) -> Option<usize> {
    match hz {
        8000 => Some(0),
        12000 => Some(1),
        16000 => Some(2),
        24000 => Some(3),
        48000 => Some(4),
        _ => None,
    }
}

impl Resampler {
    /// `silk_resampler_init` for the decoder. An unsupported pair gives a
    /// copy resampler; [`super::SilkDecoder`] checks rates before this.
    pub(crate) fn new(fs_in_hz: i32, fs_out_hz: i32) -> Self {
        let mut s = Self::default();
        let (Some(in_id), Some(out_id)) = (rate_id(fs_in_hz), rate_id(fs_out_hz)) else {
            return s;
        };
        if in_id > 2 {
            return s;
        }
        s.input_delay = DELAY_MATRIX_DEC[in_id][out_id];
        s.fs_in_khz = (fs_in_hz / 1000) as usize;
        s.fs_out_khz = (fs_out_hz / 1000) as usize;
        s.batch_size = s.fs_in_khz * MAX_BATCH_SIZE_MS;
        let mut up2x = 0;
        if fs_out_hz > fs_in_hz {
            if fs_out_hz == 2 * fs_in_hz {
                s.kind = Kind::Up2;
            } else {
                s.kind = Kind::IirFir;
                up2x = 1;
            }
        } else if fs_out_hz < fs_in_hz {
            s.kind = Kind::DownFir;
            if fs_out_hz * 4 == fs_in_hz * 3 {
                s.fir_fracs = 3;
                s.fir_order = DOWN_ORDER_FIR0;
                s.coefs = &RESAMPLER_3_4_COEFS;
            } else if fs_out_hz * 3 == fs_in_hz * 2 {
                s.fir_fracs = 2;
                s.fir_order = DOWN_ORDER_FIR0;
                s.coefs = &RESAMPLER_2_3_COEFS;
            } else {
                // 1:2 is the only remaining decoder ratio (16 to 8 kHz).
                s.fir_fracs = 1;
                s.fir_order = DOWN_ORDER_FIR1;
                s.coefs = &RESAMPLER_1_2_COEFS;
            }
        }
        // Input samples per output sample, rounded up.
        s.inv_ratio_q16 = lshift((lshift(fs_in_hz, 14 + up2x)) / fs_out_hz, 2);
        while smulww(s.inv_ratio_q16, fs_out_hz) < lshift(fs_in_hz, up2x) {
            s.inv_ratio_q16 += 1;
        }
        s
    }

    /// Output samples for `in_len` input samples.
    pub(crate) fn output_len(&self, in_len: usize) -> usize {
        (in_len * self.fs_out_khz)
            .checked_div(self.fs_in_khz)
            .unwrap_or(in_len)
    }

    /// `silk_resampler`: resamples `input` (at least 1 ms) into `out`.
    pub(crate) fn process(&mut self, out: &mut [i16], input: &[i16]) {
        let fs_in = self.fs_in_khz;
        if fs_in == 0 || input.len() < fs_in {
            // Not initialized for a supported pair: pass through.
            let n = out.len().min(input.len());
            out[..n].copy_from_slice(&input[..n]);
            return;
        }
        let delay = self.input_delay.min(fs_in);
        let n_samples = fs_in - delay;
        // The first millisecond goes through the delay line.
        self.delay_buf[delay..fs_in].copy_from_slice(&input[..n_samples]);
        let first: [i16; 16] = self.delay_buf;
        let (out_a, out_b) = out.split_at_mut(self.fs_out_khz.min(out.len()));
        let rest = &input[n_samples..input.len() - delay];
        match self.kind {
            Kind::Up2 => {
                up2_hq(&mut self.s_iir, out_a, &first[..fs_in]);
                up2_hq(&mut self.s_iir, out_b, rest);
            }
            Kind::IirFir => {
                self.iir_fir(out_a, &first[..fs_in]);
                self.iir_fir(out_b, rest);
            }
            Kind::DownFir => {
                self.down_fir(out_a, &first[..fs_in]);
                self.down_fir(out_b, rest);
            }
            Kind::Copy => {
                let n = out_a.len().min(fs_in);
                out_a[..n].copy_from_slice(&first[..n]);
                let n = out_b.len().min(rest.len());
                out_b[..n].copy_from_slice(&rest[..n]);
            }
        }
        self.delay_buf[..delay].copy_from_slice(&input[input.len() - delay..]);
    }

    /// `silk_resampler_private_IIR_FIR`: 2x allpass upsampling, then
    /// fractional interpolation with the 12-phase FIR.
    fn iir_fir(&mut self, out: &mut [i16], input: &[i16]) {
        let mut buf = [0i16; 2 * MAX_BATCH_SIZE_IN + ORDER_FIR_12];
        buf[..ORDER_FIR_12].copy_from_slice(&self.s_fir_i16);
        let step = self.inv_ratio_q16;
        let batch = self.batch_size.clamp(1, MAX_BATCH_SIZE_IN);
        let mut written = 0;
        let mut n_in = 0;
        for chunk in input.chunks(batch) {
            if n_in > 0 {
                // Carry the end of the previous batch.
                buf.copy_within(2 * n_in..2 * n_in + ORDER_FIR_12, 0);
            }
            n_in = chunk.len();
            up2_hq(
                &mut self.s_iir,
                &mut buf[ORDER_FIR_12..ORDER_FIR_12 + 2 * n_in],
                chunk,
            );
            let max_index_q16 = lshift(n_in as i32, 16 + 1);
            let mut index_q16: i32 = 0;
            while index_q16 < max_index_q16 {
                let phase = smulwb(index_q16 & 0xffff, 12) as usize;
                let at = (index_q16 >> 16) as usize;
                let x = &buf[at..at + ORDER_FIR_12];
                let lo = &FRAC_FIR_12[phase.min(11)];
                let hi = &FRAC_FIR_12[11 - phase.min(11)];
                let mut res_q15 = smulbb(i32::from(x[0]), lo[0]);
                res_q15 = smlabb(res_q15, i32::from(x[1]), lo[1]);
                res_q15 = smlabb(res_q15, i32::from(x[2]), lo[2]);
                res_q15 = smlabb(res_q15, i32::from(x[3]), lo[3]);
                res_q15 = smlabb(res_q15, i32::from(x[4]), hi[3]);
                res_q15 = smlabb(res_q15, i32::from(x[5]), hi[2]);
                res_q15 = smlabb(res_q15, i32::from(x[6]), hi[1]);
                res_q15 = smlabb(res_q15, i32::from(x[7]), hi[0]);
                if let Some(o) = out.get_mut(written) {
                    *o = sat16(rshift_round(res_q15, 15));
                }
                written += 1;
                index_q16 = index_q16.wrapping_add(step);
                if step <= 0 {
                    break;
                }
            }
        }
        if n_in > 0 {
            self.s_fir_i16
                .copy_from_slice(&buf[2 * n_in..2 * n_in + ORDER_FIR_12]);
        }
    }

    /// `silk_resampler_private_down_FIR`: AR2 prefilter, then polyphase
    /// FIR decimation.
    fn down_fir(&mut self, out: &mut [i16], mut input: &[i16]) {
        let order = self.fir_order.min(DOWN_ORDER_FIR1);
        let mut buf = [0i32; MAX_BATCH_SIZE_IN + DOWN_ORDER_FIR1];
        buf[..order].copy_from_slice(&self.s_fir_i32[..order]);
        let (ar, fir) = self.coefs.split_at(2.min(self.coefs.len()));
        let step = self.inv_ratio_q16;
        let batch = self.batch_size.clamp(1, MAX_BATCH_SIZE_IN);
        let mut written = 0;
        let mut n_in;
        loop {
            n_in = input.len().min(batch);
            ar2(
                &mut self.s_iir,
                &mut buf[order..order + n_in],
                &input[..n_in],
                ar,
            );
            let max_index_q16 = lshift(n_in as i32, 16);
            let mut index_q16: i32 = 0;
            while index_q16 < max_index_q16 {
                let at = (index_q16 >> 16) as usize;
                let x = &buf[at..at + order];
                let res_q6 = if order == DOWN_ORDER_FIR0 {
                    let phase = smulwb(index_q16 & 0xffff, self.fir_fracs);
                    let taps = DOWN_ORDER_FIR0 / 2;
                    let p0 = (phase as usize) * taps;
                    let p1 = ((self.fir_fracs - 1 - phase).max(0) as usize) * taps;
                    let (Some(c0), Some(c1)) = (fir.get(p0..p0 + taps), fir.get(p1..p1 + taps))
                    else {
                        break;
                    };
                    let mut acc = 0i32;
                    for j in 0..taps {
                        acc = smlawb(acc, x[j], i32::from(c0[j]));
                    }
                    for j in 0..taps {
                        acc = smlawb(acc, x[order - 1 - j], i32::from(c1[j]));
                    }
                    acc
                } else {
                    let mut acc = 0i32;
                    for j in 0..order / 2 {
                        let c = i32::from(fir.get(j).copied().unwrap_or(0));
                        acc = smlawb(acc, x[j].wrapping_add(x[order - 1 - j]), c);
                    }
                    acc
                };
                if let Some(o) = out.get_mut(written) {
                    *o = sat16(rshift_round(res_q6, 6));
                }
                written += 1;
                index_q16 = index_q16.wrapping_add(step);
                if step <= 0 {
                    break;
                }
            }
            input = &input[n_in..];
            if input.len() > 1 {
                buf.copy_within(n_in..n_in + order, 0);
            } else {
                break;
            }
        }
        self.s_fir_i32[..order].copy_from_slice(&buf[n_in..n_in + order]);
    }
}

/// `silk_resampler_private_up2_HQ`: 2x upsampling with two three-stage
/// allpass chains, one per output phase. State and arithmetic in Q10.
fn up2_hq(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
    for (k, &x) in input.iter().enumerate() {
        let in32 = i32::from(x) << 10;
        let phase = |state: &mut [i32], c: &[i32; 3]| {
            let y = in32.wrapping_sub(state[0]);
            let xx = smulwb(y, c[0]);
            let out1 = state[0].wrapping_add(xx);
            state[0] = in32.wrapping_add(xx);
            let y = out1.wrapping_sub(state[1]);
            let xx = smulwb(y, c[1]);
            let out2 = state[1].wrapping_add(xx);
            state[1] = out1.wrapping_add(xx);
            let y = out2.wrapping_sub(state[2]);
            let xx = smlawb(y, y, c[2]);
            let out3 = state[2].wrapping_add(xx);
            state[2] = out2.wrapping_add(xx);
            sat16(rshift_round(out3, 10))
        };
        let (even, odd) = s.split_at_mut(3);
        let a = phase(even, &UP2_HQ_0);
        let b = phase(odd, &UP2_HQ_1);
        if let Some(o) = out.get_mut(2 * k) {
            *o = a;
        }
        if let Some(o) = out.get_mut(2 * k + 1) {
            *o = b;
        }
    }
}

/// `silk_resampler_private_AR2`: second-order AR filter, output in Q8.
fn ar2(s: &mut [i32; 6], out_q8: &mut [i32], input: &[i16], a_q14: &[i16]) {
    let a0 = i32::from(a_q14.first().copied().unwrap_or(0));
    let a1 = i32::from(a_q14.get(1).copied().unwrap_or(0));
    for (o, &x) in out_q8.iter_mut().zip(input) {
        let out32 = s[0].wrapping_add(i32::from(x) << 8);
        *o = out32;
        let out32 = lshift(out32, 2);
        s[0] = smlawb(s[1], out32, a0);
        s[1] = smulwb(out32, a1);
    }
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    const RATES_IN: [i32; 3] = [8000, 12000, 16000];
    const RATES_OUT: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

    fn run(r: &mut Resampler, input: &[i16], fs_in: i32, fs_out: i32) -> Vec<i16> {
        // 10 ms per call, as the decoder calls it.
        let block = (fs_in / 100) as usize;
        let out_block = (fs_out / 100) as usize;
        let mut out = Vec::new();
        for chunk in input.chunks(block) {
            let mut o = vec![0i16; out_block];
            r.process(&mut o, chunk);
            out.extend_from_slice(&o);
        }
        out
    }

    #[test]
    fn dc_passes_with_unit_gain_at_every_ratio() {
        for fs_in in RATES_IN {
            for fs_out in RATES_OUT {
                let mut r = Resampler::new(fs_in, fs_out);
                let input = vec![10_000i16; (fs_in / 10) as usize];
                let out = run(&mut r, &input, fs_in, fs_out);
                assert_eq!(out.len(), (fs_out / 10) as usize);
                let tail = &out[out.len() / 2..];
                let mean = tail.iter().map(|&v| f64::from(v)).sum::<f64>() / tail.len() as f64;
                assert!((mean - 10_000.0).abs() < 150.0, "{fs_in}->{fs_out}: {mean}");
            }
        }
    }

    #[test]
    fn impulse_response_is_finite_centered_and_sums_to_the_ratio() {
        for fs_in in RATES_IN {
            for fs_out in RATES_OUT {
                let mut r = Resampler::new(fs_in, fs_out);
                let mut input = vec![0i16; (fs_in / 10) as usize];
                input[(fs_in / 100) as usize] = 16_000;
                let out = run(&mut r, &input, fs_in, fs_out);
                let peak = out
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, v)| v.unsigned_abs())
                    .map(|(i, _)| i)
                    .unwrap();
                // The peak lands near the impulse time plus a few ms of delay.
                let impulse_ms = 10.0;
                let peak_ms = peak as f64 * 1000.0 / f64::from(fs_out);
                assert!(
                    peak_ms >= impulse_ms && peak_ms < impulse_ms + 2.0,
                    "{fs_in}->{fs_out}: {peak_ms}"
                );
                // Energy settles: the last 20 ms are quiet.
                let tail = &out[out.len() - (fs_out / 50) as usize..];
                assert!(tail.iter().all(|v| v.abs() < 8), "{fs_in}->{fs_out}");
                // The sum of the response is the DC gain times out/in.
                let sum: f64 = out.iter().map(|&v| f64::from(v)).sum();
                let expect = 16_000.0 * f64::from(fs_out) / f64::from(fs_in);
                assert!(
                    (sum - expect).abs() < expect * 0.03,
                    "{fs_in}->{fs_out}: {sum} vs {expect}"
                );
            }
        }
    }

    #[test]
    fn copy_mode_delays_by_nothing_at_matching_rates() {
        let mut r = Resampler::new(12000, 12000);
        let input: Vec<i16> = (0..120).map(|i| i as i16).collect();
        let mut out = vec![0i16; 120];
        r.process(&mut out, &input);
        // delay_matrix_dec[12][12] is 9 samples.
        assert_eq!(&out[9..], &input[..111]);
        assert_eq!(&out[..9], &[0; 9]);
    }

    #[test]
    fn short_or_oversized_buffers_do_not_panic() {
        let mut r = Resampler::new(16000, 48000);
        let mut small = [0i16; 10];
        r.process(&mut small, &[1000; 160]);
        let mut big = [0i16; 2000];
        r.process(&mut big, &[1000; 3]);
        let mut down = Resampler::new(16000, 8000);
        down.process(&mut small, &[i16::MAX; 320]);
    }
}
