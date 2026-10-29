use super::pixel::ColorTags;
use super::sample::SampleFormat;
use super::{CodecId, Timescale};

/// Video geometry copied from a sample entry. Pixel format stays in the bitstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoParams {
    pub width: u32,
    pub height: u32,
}

/// Audio geometry copied from a sample entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioParams {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub format: SampleFormat,
}

/// What a track carries. Data is a track this crate can frame and not classify.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Video(VideoParams),
    Audio(AudioParams),
    Data,
}

/// Parameters a demuxer reads from a sample description and a muxer writes back.
///
/// `duration_ticks` is in `timescale`. `extradata` is the codec configuration
/// record, not a decoded header. `sample_entry` is the four-character code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamParams {
    pub track_id: u32,
    pub codec: CodecId,
    pub kind: MediaKind,
    pub timescale: Timescale,
    pub language: [u8; 3],
    pub extradata: Vec<u8>,
    pub duration_ticks: u64,
    pub sample_entry: [u8; 4],
    pub color: Option<ColorTags>,
}

impl StreamParams {
    #[must_use]
    pub fn video(codec: CodecId, timescale: Timescale, width: u32, height: u32) -> Self {
        Self {
            track_id: 0,
            codec,
            kind: MediaKind::Video(VideoParams { width, height }),
            timescale,
            language: *b"und",
            extradata: Vec::new(),
            duration_ticks: 0,
            sample_entry: [0; 4],
            color: None,
        }
    }

    #[must_use]
    pub fn audio(
        codec: CodecId,
        timescale: Timescale,
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        format: SampleFormat,
    ) -> Self {
        Self {
            track_id: 0,
            codec,
            kind: MediaKind::Audio(AudioParams {
                sample_rate,
                channels,
                bits_per_sample,
                format,
            }),
            timescale,
            language: *b"und",
            extradata: Vec::new(),
            duration_ticks: 0,
            sample_entry: [0; 4],
            color: None,
        }
    }
}
