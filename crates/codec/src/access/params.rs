use crate::CodecId;
use super::pixel::{ColorPrimaries, PixelFormat, Transfer};
use super::sample::{ChannelLayout, SampleFormat};

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
}
