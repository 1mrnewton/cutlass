//! MediaCodec MIME types. `docs/codecs.md` names which codecs Android lists.

use cutlass_core::{AudioCodec, VideoCodec};

/// Codecs MediaCodec listed, split by decode and encode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Capabilities {
    video_decode: Vec<VideoCodec>,
    video_encode: Vec<VideoCodec>,
    audio_decode: Vec<AudioCodec>,
    audio_encode: Vec<AudioCodec>,
}

impl Capabilities {
    /// Asks MediaCodec. A process with no Java VM is an empty list.
    pub fn query() -> Self {
        crate::jni::query()
    }

    /// True when a decoder info listed `codec`.
    pub fn decodes_video(&self, codec: VideoCodec) -> bool {
        self.video_decode.contains(&codec)
    }

    /// True when an encoder info listed `codec`.
    pub fn encodes_video(&self, codec: VideoCodec) -> bool {
        self.video_encode.contains(&codec)
    }

    /// True when a decoder info listed `codec`.
    pub fn decodes_audio(&self, codec: AudioCodec) -> bool {
        self.audio_decode.contains(&codec)
    }

    /// True when an encoder info listed `codec`.
    pub fn encodes_audio(&self, codec: AudioCodec) -> bool {
        self.audio_encode.contains(&codec)
    }

    pub(crate) fn from_lists(
        video_decode: Vec<VideoCodec>,
        video_encode: Vec<VideoCodec>,
        audio_decode: Vec<AudioCodec>,
        audio_encode: Vec<AudioCodec>,
    ) -> Self {
        Self {
            video_decode,
            video_encode,
            audio_decode,
            audio_encode,
        }
    }
}

pub(crate) fn video_mime(mime: &[u8]) -> Option<VideoCodec> {
    match mime {
        b"video/avc" => Some(VideoCodec::H264),
        b"video/hevc" => Some(VideoCodec::Hevc),
        b"video/x-vnd.on2.vp8" => Some(VideoCodec::Vp8),
        b"video/x-vnd.on2.vp9" => Some(VideoCodec::Vp9),
        b"video/av01" => Some(VideoCodec::Av1),
        b"video/mp4v-es" => Some(VideoCodec::Mpeg4Part2),
        b"video/mpeg2" => Some(VideoCodec::Mpeg2),
        b"video/mjpeg" => Some(VideoCodec::Mjpeg),
        _ => None,
    }
}

pub(crate) fn audio_mime(mime: &[u8]) -> Option<AudioCodec> {
    match mime {
        b"audio/mp4a-latm" => Some(AudioCodec::Aac),
        b"audio/mpeg" => Some(AudioCodec::Mp3),
        b"audio/raw" => Some(AudioCodec::Pcm),
        b"audio/flac" => Some(AudioCodec::Flac),
        b"audio/opus" => Some(AudioCodec::Opus),
        b"audio/vorbis" => Some(AudioCodec::Vorbis),
        b"audio/ac3" => Some(AudioCodec::Ac3),
        _ => None,
    }
}

pub(crate) fn push_video(list: &mut Vec<VideoCodec>, codec: VideoCodec) {
    if !list.contains(&codec) {
        list.push(codec);
    }
}

pub(crate) fn push_audio(list: &mut Vec<AudioCodec>, codec: AudioCodec) {
    if !list.contains(&codec) {
        list.push(codec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_mimes_match_the_codec_table() {
        assert_eq!(video_mime(b"video/avc"), Some(VideoCodec::H264));
        assert_eq!(video_mime(b"video/hevc"), Some(VideoCodec::Hevc));
        assert_eq!(video_mime(b"video/x-vnd.on2.vp8"), Some(VideoCodec::Vp8));
        assert_eq!(video_mime(b"video/x-vnd.on2.vp9"), Some(VideoCodec::Vp9));
        assert_eq!(video_mime(b"video/av01"), Some(VideoCodec::Av1));
        assert_eq!(video_mime(b"video/mp4v-es"), Some(VideoCodec::Mpeg4Part2));
        assert_eq!(video_mime(b"video/mpeg2"), Some(VideoCodec::Mpeg2));
        assert_eq!(video_mime(b"video/mjpeg"), Some(VideoCodec::Mjpeg));
        assert_eq!(video_mime(b"video/apch"), None);
        assert_eq!(video_mime(b"video/dolby-vision"), None);
    }

    #[test]
    fn audio_mimes_match_the_codec_table() {
        assert_eq!(audio_mime(b"audio/mp4a-latm"), Some(AudioCodec::Aac));
        assert_eq!(audio_mime(b"audio/mpeg"), Some(AudioCodec::Mp3));
        assert_eq!(audio_mime(b"audio/raw"), Some(AudioCodec::Pcm));
        assert_eq!(audio_mime(b"audio/flac"), Some(AudioCodec::Flac));
        assert_eq!(audio_mime(b"audio/opus"), Some(AudioCodec::Opus));
        assert_eq!(audio_mime(b"audio/vorbis"), Some(AudioCodec::Vorbis));
        assert_eq!(audio_mime(b"audio/ac3"), Some(AudioCodec::Ac3));
        assert_eq!(audio_mime(b"audio/alac"), None);
    }
}
