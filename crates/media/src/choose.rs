//! One place that picks a decoder. The system codec wins when it can do the job.

use cutlass_core::VideoCodec;

use crate::Error;

/// Which implementation will decode one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderKind {
    /// The operating-system codec.
    System,
    /// The Rust codec.
    Software,
}

/// Picks from answers the caller already has.
///
/// `system` means this machine's OS codec can decode it. `software` means the
/// Rust codec can. The system codec is used when both can.
pub fn choose(system: bool, software: bool, codec: VideoCodec) -> Result<DecoderKind, Error> {
    if system {
        Ok(DecoderKind::System)
    } else if software {
        Ok(DecoderKind::Software)
    } else {
        Err(Error::NoDecoder(codec))
    }
}

/// Asks the OS codec, then the Rust codec.
pub fn choose_video_decoder(codec: VideoCodec) -> Result<DecoderKind, Error> {
    choose(system_decodes(codec), software_decodes(codec), codec)
}

/// No Rust video decoder is connected yet. The match is exhaustive so a new
/// codec has to be considered here.
fn software_decodes(codec: VideoCodec) -> bool {
    match codec {
        VideoCodec::H264
        | VideoCodec::Hevc
        | VideoCodec::Vp8
        | VideoCodec::Vp9
        | VideoCodec::Av1
        | VideoCodec::Mpeg2
        | VideoCodec::Mpeg4Part2
        | VideoCodec::Mjpeg
        | VideoCodec::ProRes => false,
    }
}

#[cfg(target_vendor = "apple")]
fn system_decodes(codec: VideoCodec) -> bool {
    cutlass_videotoolbox::decodes(codec)
}

#[cfg(not(target_vendor = "apple"))]
fn system_decodes(codec: VideoCodec) -> bool {
    match codec {
        VideoCodec::H264
        | VideoCodec::Hevc
        | VideoCodec::Vp8
        | VideoCodec::Vp9
        | VideoCodec::Av1
        | VideoCodec::Mpeg2
        | VideoCodec::Mpeg4Part2
        | VideoCodec::Mjpeg
        | VideoCodec::ProRes => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_codec_wins_when_both_can_decode() {
        assert_eq!(
            choose(true, true, VideoCodec::H264),
            Ok(DecoderKind::System)
        );
        assert_eq!(
            choose(false, true, VideoCodec::H264),
            Ok(DecoderKind::Software)
        );
        assert_eq!(
            choose(false, false, VideoCodec::Vp9),
            Err(Error::NoDecoder(VideoCodec::Vp9))
        );
    }

    #[test]
    fn h264_follows_the_system_decoder() {
        let result = choose_video_decoder(VideoCodec::H264);
        #[cfg(target_vendor = "apple")]
        assert_eq!(result, Ok(DecoderKind::System));
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(result, Err(Error::NoDecoder(VideoCodec::H264)));
    }

    #[test]
    fn vp9_has_no_decoder_yet() {
        assert_eq!(
            choose_video_decoder(VideoCodec::Vp9),
            Err(Error::NoDecoder(VideoCodec::Vp9))
        );
    }
}
