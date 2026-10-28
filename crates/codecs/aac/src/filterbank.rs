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
}

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
    if n.is_power_of_two() {
        mdct_fft(input, output);
    } else {
        mdct_direct(input, output);
    }
    Ok(())
}

pub fn imdct(spec: &[f32], output: &mut [f32]) -> Result<(), CodecError> {
    let n = spec.len();
    if n == 0 || output.len() != n * 2 {
        return Err(CodecError::InvalidApi {
            what: "imdct length",
        });
    }
    if n.is_power_of_two() {
        imdct_fft(spec, output);
    } else {
        imdct_direct(spec, output);
    }
    Ok(())
}

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

fn fft(re: &mut [f64], im: &mut [f64]) {
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
