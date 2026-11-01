//! Min and max peaks for a timeline lane.
//!
//! One pass folds interleaved frames into at most [`MAX_BUCKETS`] pairs.
//! A short buffer stays in the buckets it reached. Empty buckets are silence.

use crate::error::EngineError;

/// Peaks kept for one file. A lane draws this many min/max pairs.
pub const MAX_BUCKETS: usize = 64;

/// Channels one frame may carry. A wider layout is not a waveform.
const MAX_CHANNELS: usize = 32;

/// One bucket. Silence is `(0, 0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub min: f32,
    pub max: f32,
}

/// Peaks for the frames that were actually read.
#[derive(Clone, Debug, PartialEq)]
pub struct Waveform {
    peaks: Vec<Peak>,
    frames: u32,
    capped: bool,
}

impl Waveform {
    #[must_use]
    pub fn peaks(&self) -> &[Peak] {
        &self.peaks
    }

    #[must_use]
    pub const fn frames(&self) -> u32 {
        self.frames
    }

    /// The file continued past [`Self::frames`].
    #[must_use]
    pub const fn capped(&self) -> bool {
        self.capped
    }

    /// Fold interleaved `samples` into `buckets`.
    ///
    /// `channels` is the frame width. Each frame contributes its lowest and
    /// highest sample, so a stereo pair does not cancel. Values outside
    /// `-1.0..=1.0` clamp. A non-finite sample counts as silence. `capped`
    /// records that these frames are a prefix.
    pub(crate) fn from_interleaved(
        samples: &[f32],
        channels: usize,
        buckets: usize,
        frames: u32,
        capped: bool,
    ) -> Result<Self, EngineError> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(EngineError::Layout);
        }
        if buckets == 0 || buckets > MAX_BUCKETS {
            return Err(EngineError::Window);
        }
        let Ok(frames_usize) = usize::try_from(frames) else {
            return Err(EngineError::Layout);
        };
        let Some(needed) = frames_usize.checked_mul(channels) else {
            return Err(EngineError::Layout);
        };
        if needed != samples.len() {
            return Err(EngineError::Layout);
        }
        let mut peaks = vec![
            Peak {
                min: f32::INFINITY,
                max: f32::NEG_INFINITY,
            };
            buckets
        ];
        if frames_usize > 0 {
            for (index, frame) in samples.chunks_exact(channels).enumerate() {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for sample in frame {
                    let sample = if sample.is_finite() {
                        sample.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    lo = lo.min(sample);
                    hi = hi.max(sample);
                }
                let bucket = bucket_index(index, buckets, frames_usize)?;
                let peak = &mut peaks[bucket];
                peak.min = peak.min.min(lo);
                peak.max = peak.max.max(hi);
            }
        }
        for peak in &mut peaks {
            if !peak.min.is_finite() || !peak.max.is_finite() {
                *peak = Peak { min: 0.0, max: 0.0 };
            }
        }
        Ok(Self {
            peaks,
            frames,
            capped,
        })
    }
}

fn widen(value: usize) -> Result<u128, EngineError> {
    u64::try_from(value)
        .map(u128::from)
        .map_err(|_| EngineError::Window)
}

fn bucket_index(index: usize, buckets: usize, frames: usize) -> Result<usize, EngineError> {
    if frames == 0 || buckets == 0 {
        return Err(EngineError::Window);
    }
    let wide = widen(index)? * widen(buckets)? / widen(frames)?;
    let Ok(bucket) = usize::try_from(wide) else {
        return Err(EngineError::Window);
    };
    if bucket >= buckets {
        return Err(EngineError::Window);
    }
    Ok(bucket)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(peak: Peak, min: f32, max: f32) {
        assert_eq!(peak.min.to_bits(), min.to_bits());
        assert_eq!(peak.max.to_bits(), max.to_bits());
    }

    #[test]
    fn silence_stays_empty_and_a_short_prefix_is_not_spread() {
        let empty = Waveform::from_interleaved(&[], 1, 2, 0, false).expect("empty");
        assert_eq!(empty.frames(), 0);
        assert!(!empty.capped());
        bits(empty.peaks()[0], 0.0, 0.0);
        bits(empty.peaks()[1], 0.0, 0.0);

        let silence =
            Waveform::from_interleaved(&[0.0, 0.0, 0.0], 1, 3, 3, false).expect("silence");
        assert_eq!(silence.frames(), 3);
        assert!(!silence.capped());
        for peak in silence.peaks() {
            bits(*peak, 0.0, 0.0);
        }

        let prefix = Waveform::from_interleaved(&[1.0], 1, 4, 1, true).expect("prefix");
        assert!(prefix.capped());
        assert_eq!(prefix.frames(), 1);
        bits(prefix.peaks()[0], 1.0, 1.0);
        for peak in &prefix.peaks()[1..] {
            bits(*peak, 0.0, 0.0);
        }
    }

    #[test]
    fn channels_share_one_bucket_and_a_loud_sample_is_clamped() {
        let stereo = Waveform::from_interleaved(&[1.0, -1.0], 2, 1, 1, false).expect("stereo");
        bits(stereo.peaks()[0], -1.0, 1.0);

        let clamped =
            Waveform::from_interleaved(&[2.0, -2.0, f32::NAN], 1, 3, 3, false).expect("clamp");
        bits(clamped.peaks()[0], 1.0, 1.0);
        bits(clamped.peaks()[1], -1.0, -1.0);
        bits(clamped.peaks()[2], 0.0, 0.0);
    }

    #[test]
    fn a_peak_stays_in_its_bucket() {
        let mut samples = vec![0.0; 1_000];
        samples[500] = 0.5;
        let wave = Waveform::from_interleaved(&samples, 1, 4, 1_000, false).expect("fold");
        assert_eq!(wave.peaks().len(), 4);
        bits(wave.peaks()[2], 0.0, 0.5);
        bits(wave.peaks()[0], 0.0, 0.0);
        bits(wave.peaks()[1], 0.0, 0.0);
        bits(wave.peaks()[3], 0.0, 0.0);
    }

    #[test]
    fn a_bad_waveform_is_refused() {
        assert_eq!(
            Waveform::from_interleaved(&[], 0, 1, 0, false).expect_err("channels"),
            EngineError::Layout
        );
        assert_eq!(
            Waveform::from_interleaved(&[1.0, 2.0, 3.0], 2, 1, 1, false).expect_err("length"),
            EngineError::Layout
        );
        assert_eq!(
            Waveform::from_interleaved(&[], 1, 0, 0, false).expect_err("buckets"),
            EngineError::Window
        );
        assert_eq!(
            Waveform::from_interleaved(&[], 1, MAX_BUCKETS + 1, 0, false).expect_err("cap"),
            EngineError::Window
        );
        assert_eq!(
            Waveform::from_interleaved(&[0.0], 33, 1, 0, false).expect_err("wide"),
            EngineError::Layout
        );
    }
}
