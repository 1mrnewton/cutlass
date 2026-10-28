use crate::id::CodecId;

/// Parameters a factory needs before it opens a codec.
///
/// `extra` is the codec configuration blob. MP3 frames carry their own
/// header, so an MP3 stream leaves `extra` empty. `bitrate` is bits per
/// second of the compressed stream. Zero means the caller did not name a rate.
///
/// `encoder_delay` and `trailing_padding` are samples at `sample_rate`.
/// They are the trim a container found in the file. Zero means the caller
/// did not name a trim. A container does not drop those samples itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecParams {
    pub codec: CodecId,
    pub extra: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub bitrate: u32,
    pub width: u32,
    pub height: u32,
    pub encoder_delay: u32,
    pub trailing_padding: u32,
}

impl CodecParams {
    #[must_use]
    pub fn audio(codec: CodecId, sample_rate: u32, channels: u16) -> Self {
        Self {
            codec,
            extra: Vec::new(),
            sample_rate,
            channels,
            bitrate: 0,
            width: 0,
            height: 0,
            encoder_delay: 0,
            trailing_padding: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CodecParams;
    use crate::id::CodecId;

    #[test]
    fn audio_params_leave_the_picture_and_the_trim_unset() {
        let params = CodecParams::audio(CodecId::Mp3, 44_100, 2);
        assert_eq!(params.codec, CodecId::Mp3);
        assert_eq!(params.sample_rate, 44_100);
        assert_eq!(params.channels, 2);
        assert!(params.extra.is_empty());
        assert_eq!(params.width, 0);
        assert_eq!(params.height, 0);
        assert_eq!(params.encoder_delay, 0);
        assert_eq!(params.trailing_padding, 0);
        assert_eq!(params.bitrate, 0);
    }
}
