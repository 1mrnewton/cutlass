//! Audio codec names, stream config, and decoded samples.

use crate::Timestamp;

/// An audio codec Cutlass decodes. `docs/codecs.md` lists which backend handles
/// each one on each OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodec {
    /// AAC, including the MP4 `mp4a` sample entry.
    Aac,
    /// MPEG-1 or MPEG-2 audio.
    Mp3,
    /// Integer or floating PCM. WAV and AIFF are containers for this.
    Pcm,
    /// FLAC.
    Flac,
    /// Opus.
    Opus,
    /// Vorbis, usually in Ogg.
    Vorbis,
    /// Apple Lossless.
    Alac,
    /// Dolby Digital.
    Ac3,
}

/// What an audio decoder needs from the container before the first packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioStreamConfig {
    pub codec: AudioCodec,
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel count. Stereo is left, then right.
    pub channels: u16,
    /// The codec configuration record exactly as the container stores it.
    /// For AAC this is the decoder-specific descriptor from `esds`. Empty when
    /// the codec has none.
    pub config_record: Vec<u8>,
}

/// Planar PCM at one sample rate.
///
/// Each sample is an `f32` in `-1.0..=1.0`. Plane `c` is [`Self::frames`]
/// samples. A decoder converts its native layout into this. The mix and the
/// resampler read the planes. A device or an encoder interleaves them.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSamples {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

impl AudioSamples {
    /// `samples` is channel-major. Its length is a multiple of `channels`.
    pub fn new(
        sample_rate: u32,
        channels: u16,
        samples: Vec<f32>,
    ) -> Result<Self, AudioSamplesError> {
        if sample_rate == 0 {
            return Err(AudioSamplesError::ZeroRate);
        }
        if channels == 0 {
            return Err(AudioSamplesError::ZeroChannels);
        }
        if !samples.len().is_multiple_of(usize::from(channels)) {
            return Err(AudioSamplesError::UnevenPlanes);
        }
        Ok(Self {
            sample_rate,
            channels,
            samples,
        })
    }

    /// Samples per second.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Channel count.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Samples in one plane.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    /// Plane `channel`, or `None` when `channel` is past [`Self::channels`].
    pub fn plane(&self, channel: u16) -> Option<&[f32]> {
        if channel >= self.channels {
            return None;
        }
        let frames = self.frames();
        let start = usize::from(channel) * frames;
        Some(&self.samples[start..start + frames])
    }
}

/// Why [`AudioSamples::new`] rejected a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSamplesError {
    /// The sample rate is zero.
    ZeroRate,
    /// The channel count is zero.
    ZeroChannels,
    /// The buffer length is not a multiple of the channel count.
    UnevenPlanes,
}

impl std::fmt::Display for AudioSamplesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroRate => "audio sample rate is zero",
            Self::ZeroChannels => "audio channel count is zero",
            Self::UnevenPlanes => "audio planes do not share one length",
        })
    }
}

impl std::error::Error for AudioSamplesError {}

/// A decoded audio frame.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    pub pts: Timestamp,
    pub duration: Option<Timestamp>,
    pub samples: AudioSamples,
}

#[cfg(test)]
mod tests {
    use super::{AudioSamples, AudioSamplesError};

    #[test]
    fn planes_split_by_channel() {
        let samples =
            AudioSamples::new(48_000, 2, vec![0.0, 1.0, -1.0, 0.5]).expect("two stereo frames");
        assert_eq!(samples.sample_rate(), 48_000);
        assert_eq!(samples.channels(), 2);
        assert_eq!(samples.frames(), 2);
        assert_eq!(samples.plane(0), Some(&[0.0, 1.0][..]));
        assert_eq!(samples.plane(1), Some(&[-1.0, 0.5][..]));
        assert_eq!(samples.plane(2), None);
    }

    #[test]
    fn an_empty_buffer_has_empty_planes() {
        let samples = AudioSamples::new(48_000, 2, Vec::new()).expect("no frames");
        assert_eq!(samples.frames(), 0);
        assert_eq!(samples.plane(0), Some(&[][..]));
        assert_eq!(samples.plane(1), Some(&[][..]));
    }

    #[test]
    fn a_zero_rate_is_rejected() {
        assert_eq!(
            AudioSamples::new(0, 1, vec![0.0]),
            Err(AudioSamplesError::ZeroRate)
        );
    }

    #[test]
    fn a_zero_channel_count_is_rejected() {
        assert_eq!(
            AudioSamples::new(48_000, 0, Vec::new()),
            Err(AudioSamplesError::ZeroChannels)
        );
    }

    #[test]
    fn uneven_planes_are_rejected() {
        assert_eq!(
            AudioSamples::new(48_000, 2, vec![0.0, 1.0, -1.0]),
            Err(AudioSamplesError::UnevenPlanes)
        );
    }
}
