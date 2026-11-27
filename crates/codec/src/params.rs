//! Parameters a container reads from a stream and hands to a decoder.
//!
//! Pixel format and sample format are not here. A video stream may carry
//! the color its container states ([`StreamColor`]); the color the
//! bitstream states is the decoder's to read. This crate does not decode.

use crate::access::{ColorRange, ContentLight, MasteringDisplay};
use crate::id::StreamCodec;
use std::fmt;

/// Width and height of a video frame, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameSize {
    width: u32,
    height: u32,
}

/// A frame size of zero is not a picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSizeError {
    Empty,
}

impl fmt::Display for FrameSizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("frame size is empty"),
        }
    }
}

impl std::error::Error for FrameSizeError {}

impl FrameSize {
    /// Both edges must be greater than zero.
    pub const fn try_new(width: u32, height: u32) -> Result<Self, FrameSizeError> {
        if width == 0 || height == 0 {
            return Err(FrameSizeError::Empty);
        }
        Ok(Self { width, height })
    }

    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// A sample rate of zero is not a rate. Absence is [`None`] on the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateError {
    Zero,
}

impl fmt::Display for RateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str("sample rate is zero"),
        }
    }
}

impl std::error::Error for RateError {}

/// A channel count of zero is not a layout. Absence is [`None`] on the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelsError {
    Zero,
}

impl fmt::Display for ChannelsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str("channel count is zero"),
        }
    }
}

impl std::error::Error for ChannelsError {}

/// The color a container states for a video stream, beside its bitstream.
///
/// The three codes are ISO/IEC 23091-2 `colour_primaries`,
/// `transfer_characteristics` and `matrix_coefficients`. [`None`] is a code
/// the container does not state, states as 2 (unspecified), or states as a
/// value that table does not define. The light level and mastering display
/// are in the units [`ContentLight`] and [`MasteringDisplay`] name. The
/// default states nothing.
///
/// A bitstream that states its own color is not overruled by this: the
/// engine gives it to the decoder, and fills from it only what a decoded
/// picture leaves unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct StreamColor {
    pub primaries: Option<u8>,
    pub transfer: Option<u8>,
    pub matrix: Option<u8>,
    pub range: ColorRange,
    pub content_light: Option<ContentLight>,
    pub mastering_display: Option<MasteringDisplay>,
}

impl StreamColor {
    /// Whether the container states nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Codec configuration a demuxer found on one stream.
///
/// `extradata` is the decoder configuration blob. `size` is the video frame.
/// `sample_rate` and `channels` describe audio. `color` is the color the
/// container states for a picture. A missing fact stays [`None`] rather
/// than a zero sentinel. An unknown codec stays on the value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamParams {
    codec: StreamCodec,
    extradata: Vec<u8>,
    size: Option<FrameSize>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
    color: StreamColor,
}

impl StreamParams {
    /// A video stream. Sample rate and channels stay absent.
    #[must_use]
    pub fn video(codec: StreamCodec, size: FrameSize, extradata: Vec<u8>) -> Self {
        Self {
            codec,
            extradata,
            size: Some(size),
            sample_rate: None,
            channels: None,
            color: StreamColor::default(),
        }
    }

    /// An audio stream. Frame size stays absent.
    ///
    /// A zero rate or a zero channel count is refused. Callers that do not
    /// know the rate leave it off the value by not calling this constructor.
    pub fn try_audio(
        codec: StreamCodec,
        sample_rate: u32,
        channels: u16,
        extradata: Vec<u8>,
    ) -> Result<Self, AudioParamsError> {
        if sample_rate == 0 {
            return Err(AudioParamsError::Rate(RateError::Zero));
        }
        if channels == 0 {
            return Err(AudioParamsError::Channels(ChannelsError::Zero));
        }
        Ok(Self {
            codec,
            extradata,
            size: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            color: StreamColor::default(),
        })
    }

    /// A stream with no picture size and no audio rate.
    ///
    /// Timed metadata and an unclassified track use this. Absence is not a
    /// zero-sized picture and it is not a silent audio stream.
    #[must_use]
    pub fn opaque(codec: StreamCodec, extradata: Vec<u8>) -> Self {
        Self {
            codec,
            extradata,
            size: None,
            sample_rate: None,
            channels: None,
            color: StreamColor::default(),
        }
    }

    #[must_use]
    pub const fn codec(&self) -> &StreamCodec {
        &self.codec
    }

    #[must_use]
    pub fn extradata(&self) -> &[u8] {
        &self.extradata
    }

    #[must_use]
    pub const fn size(&self) -> Option<FrameSize> {
        self.size
    }

    #[must_use]
    pub const fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    #[must_use]
    pub const fn channels(&self) -> Option<u16> {
        self.channels
    }

    /// The same parameters with the color the container states.
    #[must_use]
    pub const fn with_color(mut self, color: StreamColor) -> Self {
        self.color = color;
        self
    }

    /// The color the container states. Empty when it states none.
    #[must_use]
    pub const fn color(&self) -> StreamColor {
        self.color
    }
}

/// Audio parameters that cannot describe a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioParamsError {
    Rate(RateError),
    Channels(ChannelsError),
}

impl fmt::Display for AudioParamsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rate(error) => write!(formatter, "{error}"),
            Self::Channels(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for AudioParamsError {}

#[cfg(test)]
mod tests {
    use super::{
        AudioParamsError, ChannelsError, FrameSize, FrameSizeError, RateError, StreamColor,
        StreamParams,
    };
    use crate::access::{ColorRange, ContentLight};
    use crate::id::{CodecId, StreamCodec, UnknownCodec};

    #[test]
    fn video_keeps_the_frame_and_leaves_audio_absent() {
        let size = FrameSize::try_new(1920, 1080).expect("size");
        let params = StreamParams::video(StreamCodec::Known(CodecId::H264), size, vec![1, 2, 3]);
        assert_eq!(params.codec(), &StreamCodec::Known(CodecId::H264));
        assert_eq!(params.extradata(), &[1, 2, 3]);
        assert_eq!(params.size().expect("frame").width(), 1920);
        assert_eq!(params.size().expect("frame").height(), 1080);
        assert_eq!(params.sample_rate(), None);
        assert_eq!(params.channels(), None);
        assert!(params.color().is_empty());
    }

    #[test]
    fn a_stream_keeps_the_color_its_container_states() {
        let size = FrameSize::try_new(64, 32).expect("size");
        let color = StreamColor {
            primaries: Some(9),
            transfer: Some(16),
            matrix: Some(9),
            range: ColorRange::Limited,
            content_light: Some(ContentLight {
                max_cll: 1000,
                max_fall: 400,
            }),
            mastering_display: None,
        };
        let params = StreamParams::video(StreamCodec::Known(CodecId::Vp9), size, Vec::new())
            .with_color(color);
        assert_eq!(params.color(), color);
        assert!(!params.color().is_empty());
        let plain = StreamParams::video(StreamCodec::Known(CodecId::Vp9), size, Vec::new());
        assert_ne!(params, plain);
    }

    #[test]
    fn audio_keeps_the_rate_and_leaves_the_frame_absent() {
        let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Aac), 48_000, 2, vec![9])
            .expect("audio");
        assert_eq!(params.sample_rate(), Some(48_000));
        assert_eq!(params.channels(), Some(2));
        assert_eq!(params.extradata(), &[9]);
        assert_eq!(params.size(), None);
    }

    #[test]
    fn opaque_leaves_size_and_rate_absent() {
        let params = StreamParams::opaque(StreamCodec::Known(CodecId::Jpeg), vec![4]);
        assert_eq!(params.codec(), &StreamCodec::Known(CodecId::Jpeg));
        assert_eq!(params.extradata(), &[4]);
        assert_eq!(params.size(), None);
        assert_eq!(params.sample_rate(), None);
        assert_eq!(params.channels(), None);
    }

    #[test]
    fn a_zero_rate_or_channel_count_is_refused() {
        let codec = StreamCodec::Known(CodecId::Pcm);
        assert_eq!(
            StreamParams::try_audio(codec.clone(), 0, 2, Vec::new()),
            Err(AudioParamsError::Rate(RateError::Zero))
        );
        assert_eq!(
            StreamParams::try_audio(codec, 48_000, 0, Vec::new()),
            Err(AudioParamsError::Channels(ChannelsError::Zero))
        );
    }

    #[test]
    fn an_empty_frame_is_refused() {
        assert_eq!(FrameSize::try_new(0, 10), Err(FrameSizeError::Empty));
        assert_eq!(FrameSize::try_new(10, 0), Err(FrameSizeError::Empty));
    }

    #[test]
    fn an_unknown_codec_stays_on_the_parameters() {
        let unknown = UnknownCodec::new(b"zz99".to_vec()).expect("tag");
        let size = FrameSize::try_new(16, 16).expect("size");
        let params = StreamParams::video(StreamCodec::Unknown(unknown), size, Vec::new());
        assert!(params.codec().is_unknown());
        assert_eq!(params.size().expect("frame").width(), 16);
    }
}
