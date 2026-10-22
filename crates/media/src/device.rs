//! Codecs the device listed at startup. Pass the value around.

use cutlass_core::{AudioCodec, VideoCodec};

/// Codecs Android or Linux listed. Other systems are empty.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceCodecs {
    video_decode: Vec<VideoCodec>,
    video_encode: Vec<VideoCodec>,
    audio_decode: Vec<AudioCodec>,
    audio_encode: Vec<AudioCodec>,
}

impl DeviceCodecs {
    /// Asks the device once. Android reads MediaCodec. Linux reads VA-API,
    /// NVDEC, and NVENC. A missing library is skipped. Apple and Windows stay
    /// on their own queries, so this list is empty there.
    ///
    /// The chooser does not open these decoders yet.
    pub fn query() -> Self {
        query_device()
    }

    /// True when the device listed a decoder for `codec`.
    pub fn decodes_video(&self, codec: VideoCodec) -> bool {
        self.video_decode.contains(&codec)
    }

    /// True when the device listed an encoder for `codec`.
    pub fn encodes_video(&self, codec: VideoCodec) -> bool {
        self.video_encode.contains(&codec)
    }

    /// True when the device listed a decoder for `codec`.
    pub fn decodes_audio(&self, codec: AudioCodec) -> bool {
        self.audio_decode.contains(&codec)
    }

    /// True when the device listed an encoder for `codec`.
    pub fn encodes_audio(&self, codec: AudioCodec) -> bool {
        self.audio_encode.contains(&codec)
    }
}

#[cfg(target_os = "linux")]
fn query_device() -> DeviceCodecs {
    linux()
}

#[cfg(target_os = "android")]
fn query_device() -> DeviceCodecs {
    android()
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn query_device() -> DeviceCodecs {
    let mut codecs = DeviceCodecs::default();
    record_videos(&mut codecs, |_| false, |_| false);
    record_audio(&mut codecs, |_| false, |_| false);
    codecs
}

#[cfg(target_os = "linux")]
fn linux() -> DeviceCodecs {
    let va = cutlass_vaapi::Capabilities::query();
    let nvidia = cutlass_nvcodec::Capabilities::query();
    let mut codecs = DeviceCodecs::default();
    record_videos(
        &mut codecs,
        |codec| va.decodes(codec) || nvidia.decodes(codec),
        |codec| va.encodes(codec) || nvidia.encodes(codec),
    );
    // Linux has no system audio codecs.
    record_audio(&mut codecs, |_| false, |_| false);
    codecs
}

#[cfg(target_os = "android")]
fn android() -> DeviceCodecs {
    let listed = cutlass_mediacodec::Capabilities::query();
    let mut codecs = DeviceCodecs::default();
    record_videos(
        &mut codecs,
        |codec| listed.decodes_video(codec),
        |codec| listed.encodes_video(codec),
    );
    record_audio(
        &mut codecs,
        |codec| listed.decodes_audio(codec),
        |codec| listed.encodes_audio(codec),
    );
    codecs
}

fn record_videos(
    codecs: &mut DeviceCodecs,
    decode: impl Fn(VideoCodec) -> bool,
    encode: impl Fn(VideoCodec) -> bool,
) {
    for codec in [
        VideoCodec::H264,
        VideoCodec::Hevc,
        VideoCodec::Vp8,
        VideoCodec::Vp9,
        VideoCodec::Av1,
        VideoCodec::Mpeg2,
        VideoCodec::Mpeg4Part2,
        VideoCodec::Mjpeg,
        VideoCodec::ProRes,
    ] {
        note_video(codec);
        if decode(codec) {
            codecs.video_decode.push(codec);
        }
        if encode(codec) {
            codecs.video_encode.push(codec);
        }
    }
}

fn note_video(codec: VideoCodec) {
    match codec {
        VideoCodec::H264
        | VideoCodec::Hevc
        | VideoCodec::Vp8
        | VideoCodec::Vp9
        | VideoCodec::Av1
        | VideoCodec::Mpeg2
        | VideoCodec::Mpeg4Part2
        | VideoCodec::Mjpeg
        | VideoCodec::ProRes => {}
    }
}

fn record_audio(
    codecs: &mut DeviceCodecs,
    decode: impl Fn(AudioCodec) -> bool,
    encode: impl Fn(AudioCodec) -> bool,
) {
    for codec in [
        AudioCodec::Aac,
        AudioCodec::Mp3,
        AudioCodec::Pcm,
        AudioCodec::Flac,
        AudioCodec::Opus,
        AudioCodec::Vorbis,
        AudioCodec::Alac,
        AudioCodec::Ac3,
    ] {
        note_audio(codec);
        if decode(codec) {
            codecs.audio_decode.push(codec);
        }
        if encode(codec) {
            codecs.audio_encode.push(codec);
        }
    }
}

fn note_audio(codec: AudioCodec) {
    match codec {
        AudioCodec::Aac
        | AudioCodec::Mp3
        | AudioCodec::Pcm
        | AudioCodec::Flac
        | AudioCodec::Opus
        | AudioCodec::Vorbis
        | AudioCodec::Alac
        | AudioCodec::Ac3 => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_query_omits_prores() {
        let codecs = DeviceCodecs::query();
        assert!(!codecs.decodes_video(VideoCodec::ProRes));
        assert!(!codecs.encodes_video(VideoCodec::ProRes));
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn this_system_lists_only_its_own_codecs() {
        let codecs = DeviceCodecs::query();
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            assert!(!codecs.decodes_video(VideoCodec::H264));
            assert!(!codecs.encodes_video(VideoCodec::Hevc));
            assert!(!codecs.decodes_audio(AudioCodec::Aac));
            assert!(!codecs.encodes_audio(AudioCodec::Pcm));
        }
        #[cfg(target_os = "linux")]
        {
            assert!(!codecs.decodes_audio(AudioCodec::Aac));
            assert!(!codecs.encodes_audio(AudioCodec::Aac));
            assert!(!codecs.decodes_audio(AudioCodec::Mp3));
            assert!(!codecs.decodes_audio(AudioCodec::Pcm));
            assert!(!codecs.decodes_audio(AudioCodec::Flac));
            assert!(!codecs.decodes_audio(AudioCodec::Opus));
            assert!(!codecs.decodes_audio(AudioCodec::Vorbis));
            assert!(!codecs.decodes_audio(AudioCodec::Alac));
            assert!(!codecs.decodes_audio(AudioCodec::Ac3));
        }
    }
}
