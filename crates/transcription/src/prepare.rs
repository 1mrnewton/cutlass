use cutlass_model::TimeRange;

/// Mono PCM. A source downmixes before it returns this.
#[derive(Clone, Debug, PartialEq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

/// PCM for one selected range. The range is microseconds on the media clock.
///
/// The file reader implements this. A later reader will use `container` and
/// `decode` and return the same [`Pcm`].
pub trait PcmSource: Send {
    fn read(&mut self, range: &TimeRange) -> Result<Pcm, PrepareError>;
}

/// Why a range did not become PCM the recognizer can accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareError {
    /// The range duration is not positive, or the range is outside the media.
    Range,
    /// A sample rate was zero.
    SampleRate,
    /// A sample was NaN or infinite.
    Samples,
    /// The resampled buffer does not fit in memory.
    TooLong,
    /// The path could not be opened.
    Open,
    /// The file has no audio stream.
    NoAudio,
    /// The packets could not be decoded.
    Decode,
}

/// Resample `pcm` to `sample_rate`. The same rate returns the buffer as it is.
pub fn fit(pcm: Pcm, sample_rate: u32) -> Result<Pcm, PrepareError> {
    if sample_rate == 0 || pcm.sample_rate == 0 {
        return Err(PrepareError::SampleRate);
    }
    if pcm.samples.iter().any(|sample| !sample.is_finite()) {
        return Err(PrepareError::Samples);
    }
    if pcm.sample_rate == sample_rate {
        return Ok(pcm);
    }
    let samples = resample(&pcm.samples, pcm.sample_rate, sample_rate)?;
    Ok(Pcm {
        sample_rate,
        samples,
    })
}

fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>, PrepareError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let samples = u128::try_from(input.len()).map_err(|_| PrepareError::TooLong)?;
    let from_u = u128::from(from);
    let scaled = samples
        .checked_mul(u128::from(to))
        .and_then(|scaled| scaled.checked_add(from_u - 1))
        .ok_or(PrepareError::TooLong)?;
    let out_len = usize::try_from(scaled / from_u).map_err(|_| PrepareError::TooLong)?;
    if out_len == 0 {
        return Err(PrepareError::TooLong);
    }
    let mut output = Vec::with_capacity(out_len);
    let from = f64::from(from);
    let to = f64::from(to);
    let last = input.len() - 1;
    for index in 0..out_len {
        let position = index as f64 * from / to;
        let left = position.floor() as usize;
        let left = left.min(last);
        let right = (left + 1).min(last);
        let fraction = (position - left as f64) as f32;
        let sample = input[left].mul_add(1.0 - fraction, input[right] * fraction);
        output.push(sample);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_rate_keeps_the_samples() {
        let pcm = Pcm {
            sample_rate: 16_000,
            samples: vec![0.25, -0.25],
        };
        let fitted = fit(pcm, 16_000).unwrap();
        assert_eq!(fitted.samples, [0.25, -0.25]);
    }

    #[test]
    fn doubling_the_rate_lerps_between_samples() {
        let pcm = Pcm {
            sample_rate: 8_000,
            samples: vec![0.0, 1.0],
        };
        let fitted = fit(pcm, 16_000).unwrap();
        assert_eq!(fitted.sample_rate, 16_000);
        assert_eq!(fitted.samples.len(), 4);
        assert!((fitted.samples[0] - 0.0).abs() < 1e-6);
        assert!((fitted.samples[1] - 0.5).abs() < 1e-6);
        assert!((fitted.samples[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_non_finite_sample_is_refused() {
        let pcm = Pcm {
            sample_rate: 16_000,
            samples: vec![f32::NAN],
        };
        assert_eq!(fit(pcm, 16_000).unwrap_err(), PrepareError::Samples);
    }

    #[test]
    fn a_zero_rate_is_refused() {
        let pcm = Pcm {
            sample_rate: 0,
            samples: vec![0.0],
        };
        assert_eq!(fit(pcm, 16_000).unwrap_err(), PrepareError::SampleRate);
    }
}
