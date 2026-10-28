use super::pixel::{ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer};
use super::sample::{ChannelLayout, SampleFormat};
use crate::CodecId;

/// Parameters a factory needs before it opens a codec.
///
/// `extra` is the codec configuration blob. For AAC it is an
/// `AudioSpecificConfig`. `bitrate` is bits per second of the compressed
/// stream; zero means the caller did not name a rate.
#[derive(Clone, Debug, PartialEq)]
pub struct CodecParams {
    pub codec: CodecId,
    pub extra: Vec<u8>,
    pub sample_rate: u32,
    pub layout: Option<ChannelLayout>,
    pub sample_format: Option<SampleFormat>,
    pub bitrate: u32,
    pub width: u32,
    pub height: u32,
    pub pixel: PixelFormat,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub range: ColorRange,
    pub matrix: MatrixCoefficients,
}

impl CodecParams {
    #[must_use]
    pub fn audio(codec: CodecId, sample_rate: u32, layout: ChannelLayout) -> Self {
        Self {
            codec,
            extra: Vec::new(),
            sample_rate,
            layout: Some(layout),
            sample_format: Some(SampleFormat::F32),
            bitrate: 0,
            width: 0,
            height: 0,
            pixel: PixelFormat::None,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            range: ColorRange::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
        }
    }

    /// A video stream. `bitrate` stays zero until the caller names a rate.
    #[must_use]
    pub fn video(codec: CodecId, width: u32, height: u32, pixel: PixelFormat) -> Self {
        Self {
            codec,
            extra: Vec::new(),
            sample_rate: 0,
            layout: None,
            sample_format: None,
            bitrate: 0,
            width,
            height,
            pixel,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            range: ColorRange::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pixel::PixelFormat;
    use super::super::sample::ChannelLayout;
    use super::CodecParams;
    use crate::CodecId;

    #[test]
    fn audio_params_leave_the_picture_unset() {
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::stereo());
        assert_eq!(params.codec, CodecId::Aac);
        assert_eq!(params.sample_rate, 48_000);
        assert_eq!(params.pixel, PixelFormat::None);
        assert!(params.extra.is_empty());
        assert_eq!(params.bitrate, 0);
    }

    #[test]
    fn video_params_leave_audio_absent() {
        let params = CodecParams::video(CodecId::Vp9, 320, 240, PixelFormat::Yuv420p);
        assert_eq!(params.width, 320);
        assert_eq!(params.height, 240);
        assert_eq!(params.pixel, PixelFormat::Yuv420p);
        assert_eq!(params.sample_rate, 0);
        assert!(params.layout.is_none());
    }
}
