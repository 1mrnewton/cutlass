//! MDCT filterbank for 1024- and 960-sample frames.
//!
//! The analysis transform is the unnormalized sum
//! `X[k] = Σ x[n] cos(π/n · (n_index + n/2 + 1/2) · (k + 1/2))`
//! over a block of `2n` windowed samples. The synthesis transform uses the
//! same cosine with gain `2/n`. A power-complementary window on both sides
//! then overlap-adds to the original samples. Gain `1/n` (the factor written
//! as `2/N` when `N` is the full window) reconstructs at half amplitude; that
//! was measured on an impulse, so this pair uses `2/n`.
//!
//! Both sums run through a complex FFT of `2n` points: radix 2 for the
//! 1024- and 128-line blocks, and one mixed-radix step over the factor 15
//! for the 960- and 120-line blocks. The direct sums stay as the test oracle.
//!
//! Window shapes are the sine window and the Kaiser-Bessel derived window
//! (alpha 4 on a long slope, alpha 6 on a short slope), ISO/IEC 14496-3
//! filterbank. The left slope of a frame uses the previous frame's shape.

use cutlass_decode::CodecError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sequence {
    OnlyLong = 0,
    LongStart = 1,
    EightShort = 2,
    LongStop = 3,
}

impl Sequence {
    pub fn from_bits(bits: u32) -> Result<Self, CodecError> {
        match bits {
            0 => Ok(Self::OnlyLong),
            1 => Ok(Self::LongStart),
            2 => Ok(Self::EightShort),
            3 => Ok(Self::LongStop),
            _ => Err(CodecError::Malformed {
                at_bit: 0,
                what: "window sequence",
            }),
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Self::OnlyLong => 0,
            Self::LongStart => 1,
            Self::EightShort => 2,
            Self::LongStop => 3,
        }
    }
}

pub fn transition_ok(prev: Sequence, next: Sequence) -> bool {
    match prev {
        Sequence::OnlyLong | Sequence::LongStop => {
            matches!(next, Sequence::OnlyLong | Sequence::LongStart)
        }
        Sequence::LongStart | Sequence::EightShort => {
            matches!(next, Sequence::EightShort | Sequence::LongStop)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Windows {
    frame: usize,
    long_sine: Vec<f32>,
    long_kbd: Vec<f32>,
    short_sine: Vec<f32>,
    short_kbd: Vec<f32>,
}

impl Windows {
    pub fn new(frame: usize) -> Result<Self, CodecError> {
        if frame == 0 || !frame.is_multiple_of(8) {
            return Err(CodecError::InvalidApi {
                what: "frame length",
            });
        }
        let short = frame / 8;
        Ok(Self {
            frame,
            long_sine: sine_half(frame),
            long_kbd: kbd_half(frame, 4.0),
            short_sine: sine_half(short),
            short_kbd: kbd_half(short, 6.0),
        })
    }

    fn long(&self, kbd: bool) -> &[f32] {
        if kbd { &self.long_kbd } else { &self.long_sine }
    }

    fn short(&self, kbd: bool) -> &[f32] {
        if kbd {
            &self.short_kbd
        } else {
            &self.short_sine
        }
    }
}

fn sine_half(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * std::f64::consts::PI / (2.0 * n as f64);
            x.sin() as f32
        })
        .collect()
}

fn kbd_half(n: usize, alpha: f64) -> Vec<f32> {
    let alpha2 = (alpha * std::f64::consts::PI / n as f64).powi(2);
    let mut total = 0.0;
    let mut acc = Vec::with_capacity(n);
    for i in 0..n {
        let tmp = (i * (n - i)) as f64 * alpha2;
        let mut bessel = 1.0;
        for j in (1..50).rev() {
            bessel = bessel * tmp / f64::from(j * j) + 1.0;
        }
        total += bessel;
        acc.push(total);
    }
    total += 1.0;
    acc.into_iter().map(|v| (v / total).sqrt() as f32).collect()
}

#[derive(Clone, Debug)]
pub struct Overlap {
    delay: Vec<f32>,
    prev: Sequence,
    prev_kbd: bool,
    started: bool,
}

impl Overlap {
    pub fn new(frame: usize) -> Self {
        Self {
            delay: vec![0.0; frame],
            prev: Sequence::OnlyLong,
            prev_kbd: false,
            started: false,
        }
    }

    pub fn is_silent(&self) -> bool {
        self.delay.iter().all(|s| s.abs() <= 1.0e-8)
    }

    pub(crate) fn prev_shape_kbd(&self) -> bool {
        self.prev_kbd
    }

    /// Sequence that releases the overlap without breaking the window rules.
    pub fn flush_sequence(&self) -> Sequence {
        match self.prev {
            Sequence::LongStart | Sequence::EightShort => Sequence::LongStop,
            Sequence::OnlyLong | Sequence::LongStop => Sequence::OnlyLong,
        }
    }

    /// Forget the last window sequence. The frame a drain makes up is not in
    /// the stream, so it does not decide which sequence may follow.
    pub(crate) fn restart(&mut self) {
        self.started = false;
    }
}

/// Gain between this unnormalized `2/N` pair and the float AAC pair that
/// scales analysis by 32768 and synthesis by `1/(N·32768)`.
///
/// A sample of 1.0 in float PCM is full scale for those codecs. Multiplying
/// analysis coefficients by this factor and dividing synthesis coefficients
/// by it leaves this pair reconstructing the same samples, and puts an
/// access unit on the amplitude an independent decoder expects.
pub(crate) const FULL_SCALE: f32 = 65536.0;

pub fn analyze(
    windows: &Windows,
    history: &[f32],
    current: &[f32],
    seq: Sequence,
    prev_kbd: bool,
    kbd: bool,
) -> Result<Vec<f32>, CodecError> {
    let frame = windows.frame;
    if history.len() != frame || current.len() != frame {
        return Err(CodecError::InvalidApi {
            what: "analysis block",
        });
    }
    let mut block = Vec::with_capacity(frame * 2);
    block.extend_from_slice(history);
    block.extend_from_slice(current);
    let mut spec = vec![0.0; frame];
    transform_block(windows, &block, seq, prev_kbd, kbd, &mut spec, false)?;
    Ok(spec)
}

pub fn synthesize(
    windows: &Windows,
    overlap: &mut Overlap,
    spec: &[f32],
    seq: Sequence,
    kbd: bool,
) -> Result<Vec<f32>, CodecError> {
    let frame = windows.frame;
    if spec.len() != frame || overlap.delay.len() != frame {
        return Err(CodecError::InvalidApi {
            what: "synthesis block",
        });
    }
    if overlap.started && !transition_ok(overlap.prev, seq) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "window sequence",
        });
    }
    let mut time = vec![0.0; frame * 2];
    transform_block(windows, spec, seq, overlap.prev_kbd, kbd, &mut time, true)?;
    let mut pcm = vec![0.0; frame];
    let mut delay = vec![0.0; frame];
    for i in 0..frame {
        pcm[i] = overlap.delay[i] + time[i];
        delay[i] = time[frame + i];
    }
    if pcm.iter().any(|s| !s.is_finite()) {
        return Err(CodecError::NonFinite);
    }
    overlap.delay = delay;
    overlap.prev = seq;
    overlap.prev_kbd = kbd;
    overlap.started = true;
    Ok(pcm)
}

fn transform_block(
    windows: &Windows,
    input: &[f32],
    seq: Sequence,
    prev_kbd: bool,
    kbd: bool,
    output: &mut [f32],
    synthesis: bool,
) -> Result<(), CodecError> {
    if seq == Sequence::EightShort {
        short_block(windows, input, prev_kbd, kbd, output, synthesis)
    } else {
        long_block(windows, input, seq, prev_kbd, kbd, output, synthesis)
    }
}

fn long_block(
    windows: &Windows,
    input: &[f32],
    seq: Sequence,
    prev_kbd: bool,
    kbd: bool,
    output: &mut [f32],
    synthesis: bool,
) -> Result<(), CodecError> {
    let frame = windows.frame;
    let window = long_window(windows, seq, prev_kbd, kbd);
    if synthesis {
        let mut time = vec![0.0; frame * 2];
        imdct(input, &mut time)?;
        for i in 0..frame * 2 {
            output[i] = time[i] * window[i];
        }
    } else {
        let mut windowed = vec![0.0; frame * 2];
        for i in 0..frame * 2 {
            windowed[i] = input[i] * window[i];
        }
        mdct(&windowed, output)?;
    }
    Ok(())
}

fn short_block(
    windows: &Windows,
    input: &[f32],
    prev_kbd: bool,
    kbd: bool,
    output: &mut [f32],
    synthesis: bool,
) -> Result<(), CodecError> {
    let frame = windows.frame;
    let sb = frame / 8;
    let point0 = frame / 2 - sb / 2;
    if synthesis {
        let mut pcm = vec![0.0; frame + sb];
        for w in 0..8 {
            let mut time = vec![0.0; sb * 2];
            imdct(&input[w * sb..(w + 1) * sb], &mut time)?;
            let win = short_window(windows, w == 0, prev_kbd, kbd);
            let start = w * sb;
            for i in 0..sb * 2 {
                pcm[start + i] += time[i] * win[i];
            }
        }
        for (i, sample) in pcm.iter().enumerate() {
            output[point0 + i] += *sample;
        }
    } else {
        for w in 0..8 {
            let src = point0 + w * sb;
            let chunk = &input[src..src + sb * 2];
            let win = short_window(windows, w == 0, prev_kbd, kbd);
            let mut windowed = vec![0.0; sb * 2];
            for i in 0..sb * 2 {
                windowed[i] = chunk[i] * win[i];
            }
            mdct(&windowed, &mut output[w * sb..(w + 1) * sb])?;
        }
    }
    Ok(())
}

fn long_window(windows: &Windows, seq: Sequence, prev_kbd: bool, kbd: bool) -> Vec<f32> {
    let frame = windows.frame;
    let sb = frame / 8;
    let point0 = frame / 2 - sb / 2;
    let prev = windows.long(prev_kbd);
    let curr = windows.long(kbd);
    let short = windows.short(kbd);
    let mut w = vec![0.0; frame * 2];
    match seq {
        Sequence::OnlyLong => {
            copy_half(&mut w, prev, curr);
        }
        Sequence::LongStart => {
            w[..frame].copy_from_slice(prev);
            for i in 0..point0 {
                w[frame + i] = 1.0;
            }
            for i in 0..sb {
                w[frame + point0 + i] = short[sb - 1 - i];
            }
        }
        Sequence::LongStop => {
            let prev_short = windows.short(prev_kbd);
            w[point0..point0 + sb].copy_from_slice(prev_short);
            for sample in w.iter_mut().take(frame).skip(point0 + sb) {
                *sample = 1.0;
            }
            for i in 0..frame {
                w[frame + i] = curr[frame - 1 - i];
            }
        }
        Sequence::EightShort => {}
    }
    w
}

fn copy_half(dest: &mut [f32], prev: &[f32], curr: &[f32]) {
    let frame = prev.len();
    dest[..frame].copy_from_slice(prev);
    for i in 0..frame {
        dest[frame + i] = curr[frame - 1 - i];
    }
}

fn short_window(windows: &Windows, first: bool, prev_kbd: bool, kbd: bool) -> Vec<f32> {
    let left = if first {
        windows.short(prev_kbd)
    } else {
        windows.short(kbd)
    };
    let right = windows.short(kbd);
    let sb = right.len();
    let mut w = Vec::with_capacity(sb * 2);
    w.extend_from_slice(left);
    for i in 0..sb {
        w.push(right[sb - 1 - i]);
    }
    w
}

pub fn mdct(input: &[f32], output: &mut [f32]) -> Result<(), CodecError> {
    let n = output.len();
    if n == 0 || input.len() != n * 2 {
        return Err(CodecError::InvalidApi {
            what: "mdct length",
        });
    }
    mdct_fft(input, output);
    Ok(())
}

pub fn imdct(spec: &[f32], output: &mut [f32]) -> Result<(), CodecError> {
    let n = spec.len();
    if n == 0 || output.len() != n * 2 {
        return Err(CodecError::InvalidApi {
            what: "imdct length",
        });
    }
    imdct_fft(spec, output);
    Ok(())
}

/// The analysis sum term by term, the oracle for [`mdct_fft`].
#[cfg(test)]
fn mdct_direct(input: &[f32], output: &mut [f32]) {
    let n = output.len();
    let n_f = n as f64;
    for (k, out) in output.iter_mut().enumerate() {
        let step = std::f64::consts::PI / n_f * (k as f64 + 0.5);
        let mut angle = step * (n_f / 2.0 + 0.5);
        let mut sum = 0.0;
        let mut t = 0;
        while t < input.len() {
            let mut cos = angle.cos();
            let mut sin = angle.sin();
            let cd = step.cos();
            let sd = step.sin();
            let end = (t + 32).min(input.len());
            while t < end {
                sum += f64::from(input[t]) * cos;
                let next_c = cos * cd - sin * sd;
                let next_s = sin * cd + cos * sd;
                cos = next_c;
                sin = next_s;
                t += 1;
            }
            angle = step * (t as f64 + n_f / 2.0 + 0.5);
        }
        *out = sum as f32;
    }
}

/// The synthesis sum term by term, the oracle for [`imdct_fft`].
#[cfg(test)]
fn imdct_direct(spec: &[f32], output: &mut [f32]) {
    let n = spec.len();
    let scale = 2.0 / n as f64;
    for (i, out) in output.iter_mut().enumerate() {
        let step = std::f64::consts::PI / n as f64 * (i as f64 + n as f64 / 2.0 + 0.5);
        let mut angle = step * 0.5;
        let mut sum = 0.0;
        let mut k = 0;
        while k < n {
            let mut cos = angle.cos();
            let mut sin = angle.sin();
            let cd = step.cos();
            let sd = step.sin();
            let end = (k + 32).min(n);
            while k < end {
                sum += f64::from(spec[k]) * cos;
                let next_c = cos * cd - sin * sd;
                let next_s = sin * cd + cos * sd;
                cos = next_c;
                sin = next_s;
                k += 1;
            }
            angle = step * (k as f64 + 0.5);
        }
        *out = (sum * scale) as f32;
    }
}

fn mdct_fft(input: &[f32], output: &mut [f32]) {
    let n = output.len();
    let mut re = vec![0.0; n * 2];
    let mut im = vec![0.0; n * 2];
    let n_f = n as f64;
    for (i, &sample) in input.iter().enumerate() {
        let angle = -std::f64::consts::PI / n_f * i as f64 * 0.5;
        let s = f64::from(sample);
        re[i] = s * angle.cos();
        im[i] = s * angle.sin();
    }
    fft(&mut re, &mut im);
    for (k, out) in output.iter_mut().enumerate() {
        let post = -std::f64::consts::PI / n_f * (n_f / 2.0 + 0.5) * (k as f64 + 0.5);
        let pr = post.cos();
        let pi = post.sin();
        *out = (re[k] * pr - im[k] * pi) as f32;
    }
}

fn imdct_fft(spec: &[f32], output: &mut [f32]) {
    let n = spec.len();
    let mut re = vec![0.0; n * 2];
    let mut im = vec![0.0; n * 2];
    let n_f = n as f64;
    for (k, &bin) in spec.iter().enumerate() {
        let angle = std::f64::consts::PI / n_f * (n_f / 2.0 + 0.5) * k as f64;
        let s = f64::from(bin);
        re[k] = s * angle.cos();
        im[k] = -s * angle.sin();
    }
    fft(&mut re, &mut im);
    let scale = 2.0 / n_f;
    for i in 0..output.len() {
        let tw = std::f64::consts::PI / n_f * (i as f64 + n_f / 2.0 + 0.5) * 0.5;
        let sr = re[i];
        let si = -im[i];
        let real = sr * tw.cos() - si * tw.sin();
        output[i] = (real * scale) as f32;
    }
}

/// Forward complex DFT in place, `X[k] = Σ x[j] e^(-2πi·jk/n)`.
///
/// A power of two goes straight to the radix-2 transform. Any other length
/// is `odd · m` with `m` a power of two (1920 = 15 · 128 for a 960-sample
/// frame, 240 = 15 · 16 for its short window), done as one Cooley-Tukey
/// step: `odd` radix-2 transforms of the samples `odd` apart, a twiddle, then
/// `m` direct transforms of length `odd`.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n == 0 {
        return;
    }
    let odd = n >> n.trailing_zeros();
    if odd == 1 {
        fft_radix2(re, im);
    } else {
        fft_mixed(re, im, odd);
    }
}

fn fft_mixed(re: &mut [f64], im: &mut [f64], odd: usize) {
    let n = re.len();
    let m = n / odd;
    let n_f = n as f64;
    // Row `j1` holds the transform of `x[odd·j2 + j1]` over `j2`, times
    // `e^(-2πi·j1·k1/n)` at column `k1`.
    let mut rows_re = vec![0.0; n];
    let mut rows_im = vec![0.0; n];
    for (j1, (row_re, row_im)) in rows_re
        .chunks_exact_mut(m)
        .zip(rows_im.chunks_exact_mut(m))
        .enumerate()
    {
        for (j2, (r, i)) in row_re.iter_mut().zip(row_im.iter_mut()).enumerate() {
            *r = re[odd * j2 + j1];
            *i = im[odd * j2 + j1];
        }
        fft_radix2(row_re, row_im);
        if j1 == 0 {
            continue;
        }
        // The twiddle steps by a rotation and is set exactly every 32
        // columns, as the direct sums do.
        let step = -2.0 * std::f64::consts::PI * j1 as f64 / n_f;
        let (sd, cd) = step.sin_cos();
        let (mut wr, mut wi) = (1.0, 0.0);
        for (k1, (r, i)) in row_re.iter_mut().zip(row_im.iter_mut()).enumerate() {
            if k1 % 32 == 0 {
                (wi, wr) = (step * k1 as f64).sin_cos();
            }
            let (xr, xi) = (*r, *i);
            *r = xr * wr - xi * wi;
            *i = xr * wi + xi * wr;
            (wr, wi) = (wr * cd - wi * sd, wr * sd + wi * cd);
        }
    }
    // `X[k1 + m·k2] = Σ_j1 row[j1][k1] · e^(-2πi·j1·k2/odd)`, column by
    // column. Rows `j` and `odd - j` share a cosine and have opposite sines,
    // so each column is summed over mirrored pairs: with
    // `a = Σ row·cos` and `b = Σ row·sin`, `X[k2] = a - i·b` and
    // `X[odd - k2] = a + i·b`.
    let roots: Vec<(f64, f64)> = (0..odd)
        .map(|r| {
            let (s, c) = (2.0 * std::f64::consts::PI * r as f64 / odd as f64).sin_cos();
            (c, s)
        })
        .collect();
    let half = odd / 2;
    let mut sum = vec![(0.0, 0.0); half + 1];
    let mut diff = vec![(0.0, 0.0); half + 1];
    for k1 in 0..m {
        let x0 = (rows_re[k1], rows_im[k1]);
        let mut total = x0;
        for j in 1..=half {
            let (ar, ai) = (rows_re[j * m + k1], rows_im[j * m + k1]);
            let (br, bi) = (rows_re[(odd - j) * m + k1], rows_im[(odd - j) * m + k1]);
            sum[j] = (ar + br, ai + bi);
            diff[j] = (ar - br, ai - bi);
            total.0 += ar + br;
            total.1 += ai + bi;
        }
        re[k1] = total.0;
        im[k1] = total.1;
        for k2 in 1..=half {
            let (mut ar, mut ai) = x0;
            let (mut br, mut bi) = (0.0, 0.0);
            let mut power = 0usize;
            for j in 1..=half {
                power += k2;
                if power >= odd {
                    power -= odd;
                }
                let (c, s) = roots[power];
                ar += sum[j].0 * c;
                ai += sum[j].1 * c;
                br += diff[j].0 * s;
                bi += diff[j].1 * s;
            }
            re[k1 + m * k2] = ar + bi;
            im[k1 + m * k2] = ai - br;
            re[k1 + m * (odd - k2)] = ar - bi;
            im[k1 + m * (odd - k2)] = ai + br;
        }
    }
}

fn fft_radix2(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        let mut start = 0usize;
        while start < n {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for k in 0..len / 2 {
                let right = start + k + len / 2;
                let vr = re[right] * wr - im[right] * wi;
                let vi = re[right] * wi + im[right] * wr;
                let left = start + k;
                let ur = re[left];
                let ui = im[left];
                re[left] = ur + vr;
                im[left] = ui + vi;
                re[right] = ur - vr;
                im[right] = ui - vi;
                let next_r = wr * wlen_re - wi * wlen_im;
                let next_i = wr * wlen_im + wi * wlen_re;
                wr = next_r;
                wi = next_i;
            }
            start += len;
        }
        len *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Overlap, Sequence, Windows, analyze, imdct, kbd_half, mdct, sine_half, synthesize,
        transition_ok,
    };

    fn rms(err: f64, n: usize) -> f64 {
        (err / n as f64).sqrt()
    }

    #[test]
    fn windows_are_power_complementary() {
        for n in [128usize, 960, 1024] {
            let sine = sine_half(n);
            let kbd = kbd_half(n, if n >= 256 { 4.0 } else { 6.0 });
            for half in [sine, kbd] {
                let worst = (0..n)
                    .map(|i| (half[i] * half[i] + half[n - 1 - i] * half[n - 1 - i] - 1.0).abs())
                    .fold(0.0f32, f32::max);
                assert!(worst < 1.0e-5, "{n} {worst}");
            }
        }
    }

    #[test]
    fn fast_mdct_matches_the_direct_sum() {
        let n = 32;
        let input: Vec<f32> = (0..n * 2).map(|i| ((i * 3) as f32 * 0.01).sin()).collect();
        let mut fast = vec![0.0; n];
        let mut slow = vec![0.0; n];
        mdct(&input, &mut fast).unwrap();
        super::mdct_direct(&input, &mut slow);
        let worst = fast
            .iter()
            .zip(slow.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1.0e-3, "{worst}");
        let mut back = vec![0.0; n * 2];
        imdct(&fast, &mut back).unwrap();
        let mut back_slow = vec![0.0; n * 2];
        super::imdct_direct(&slow, &mut back_slow);
        let worst = back
            .iter()
            .zip(back_slow)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1.0e-3, "{worst}");
    }

    /// Largest difference between two transforms, in units of the largest
    /// output of the direct sum.
    fn worst_relative(fast: &[f32], slow: &[f32]) -> f32 {
        let peak = slow.iter().fold(0.0f32, |max, v| max.max(v.abs()));
        assert!(peak > 0.0);
        fast.iter()
            .zip(slow)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / peak
    }

    /// Every block length of both frame sizes, against the direct sums on
    /// random input. The FFT path must agree to within two float steps of the peak.
    #[test]
    fn transforms_match_the_direct_sums_at_every_block_length() {
        let mut state = 0x2545_F491u32;
        let mut random = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        for n in [1024usize, 960, 128, 120] {
            for _ in 0..4 {
                let input: Vec<f32> = (0..n * 2).map(|_| random()).collect();
                let mut fast = vec![0.0; n];
                let mut slow = vec![0.0; n];
                mdct(&input, &mut fast).unwrap();
                super::mdct_direct(&input, &mut slow);
                let worst = worst_relative(&fast, &slow);
                assert!(worst < 2.0 * f32::EPSILON, "mdct {n}: {worst}");

                let spec: Vec<f32> = (0..n).map(|_| random()).collect();
                let mut fast = vec![0.0; n * 2];
                let mut slow = vec![0.0; n * 2];
                imdct(&spec, &mut fast).unwrap();
                super::imdct_direct(&spec, &mut slow);
                let worst = worst_relative(&fast, &slow);
                assert!(worst < 2.0 * f32::EPSILON, "imdct {n}: {worst}");
            }
        }
    }

    /// Lengths with an odd factor other than 15, down to an odd length
    /// alone, against a plain DFT.
    #[test]
    fn the_mixed_radix_fft_matches_a_plain_dft() {
        for n in [1usize, 3, 5, 6, 15, 30, 45, 240, 1920] {
            let re: Vec<f64> = (0..n).map(|j| ((j * 7 + 3) % 11) as f64 - 5.0).collect();
            let im: Vec<f64> = (0..n).map(|j| ((j * 5 + 1) % 13) as f64 - 6.0).collect();
            let mut got_re = re.clone();
            let mut got_im = im.clone();
            super::fft(&mut got_re, &mut got_im);
            for k in 0..n {
                let (mut want_re, mut want_im) = (0.0, 0.0);
                for j in 0..n {
                    let angle = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    want_re += re[j] * angle.cos() - im[j] * angle.sin();
                    want_im += re[j] * angle.sin() + im[j] * angle.cos();
                }
                let err = (got_re[k] - want_re).hypot(got_im[k] - want_im);
                assert!(err < 1.0e-9 * n as f64, "n {n} k {k}: {err}");
            }
        }
    }

    fn round_trip(frame: usize, seqs: &[Sequence], kbd: &[bool], signal: &[f32]) -> f64 {
        let windows = Windows::new(frame).unwrap();
        let mut history = vec![0.0; frame];
        let mut overlap = Overlap::new(frame);
        let mut out = Vec::new();
        let mut padded = signal.to_vec();
        padded.extend(std::iter::repeat_n(0.0, frame));
        for (index, seq) in seqs.iter().copied().enumerate() {
            let current = &padded[index * frame..(index + 1) * frame];
            let prev_kbd = overlap.prev_kbd;
            let spec = analyze(&windows, &history, current, seq, prev_kbd, kbd[index]).unwrap();
            history = current.to_vec();
            out.extend(synthesize(&windows, &mut overlap, &spec, seq, kbd[index]).unwrap());
        }
        let mut err = 0.0;
        for (i, sample) in signal.iter().enumerate() {
            let got = f64::from(out[i + frame]);
            let diff = got - f64::from(*sample);
            err += diff * diff;
        }
        rms(err, signal.len())
    }

    #[test]
    fn long_and_short_sequences_reconstruct() {
        for frame in [128usize, 256, 1024] {
            let signal: Vec<f32> = (0..frame * 4)
                .map(|i| (2.0 * std::f32::consts::PI * 3.0 * i as f32 / frame as f32).sin())
                .collect();
            let seqs = [
                Sequence::OnlyLong,
                Sequence::LongStart,
                Sequence::EightShort,
                Sequence::LongStop,
                Sequence::OnlyLong,
            ];
            let kbd = [false, false, true, true, false];
            let err = round_trip(frame, &seqs, &kbd, &signal);
            assert!(err < 1.0e-4, "frame {frame} err {err}");
        }
    }

    #[test]
    fn nine_sixty_sample_transition_reconstructs() {
        let frame = 960;
        let signal: Vec<f32> = (0..frame * 3)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / frame as f32).sin() * 0.5)
            .collect();
        let seqs = [
            Sequence::OnlyLong,
            Sequence::LongStart,
            Sequence::EightShort,
            Sequence::LongStop,
        ];
        let kbd = [true, true, true, false];
        let err = round_trip(frame, &seqs, &kbd, &signal);
        assert!(err < 1.0e-3, "{err}");
    }

    #[test]
    fn an_illegal_second_window_does_not_replace_overlap() {
        let windows = Windows::new(128).unwrap();
        let mut overlap = Overlap::new(128);
        let spec = vec![0.0; 128];
        synthesize(&windows, &mut overlap, &spec, Sequence::OnlyLong, false).unwrap();
        let before = overlap.delay.clone();
        let err =
            synthesize(&windows, &mut overlap, &spec, Sequence::EightShort, false).unwrap_err();
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Malformed {
                what: "window sequence",
                ..
            }
        ));
        assert_eq!(overlap.delay, before);
        assert!(!transition_ok(Sequence::OnlyLong, Sequence::EightShort));
        assert!(transition_ok(Sequence::OnlyLong, Sequence::LongStart));
    }
}
