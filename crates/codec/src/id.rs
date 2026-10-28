/// One id per codec the workspace names.
///
/// A variant existing here does not mean this build can decode it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecId {
    Aac,
    Ac3,
    Alac,
    Av1,
    Flac,
    Gif,
    H264,
    Hevc,
    Jpeg,
    Mjpeg,
    Mp3,
    Mpeg2,
    Mpeg4,
    Opus,
    Pcm,
    Png,
    Prores,
    Vorbis,
    Vp8,
    Vp9,
    Webp,
}

/// One id per container family the workspace names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContainerId {
    Adts,
    Aiff,
    Avi,
    Caf,
    Flac,
    Isobmff,
    Matroska,
    Mp3,
    MpegTs,
    Ogg,
    Riff,
    Still,
}

#[cfg(test)]
mod tests {
    use super::{CodecId, ContainerId};

    #[test]
    fn mp3_ids_are_distinct_from_the_other_families() {
        assert_ne!(CodecId::Mp3, CodecId::Aac);
        assert_ne!(ContainerId::Mp3, ContainerId::Adts);
        assert_ne!(ContainerId::Mp3, ContainerId::Isobmff);
    }
}
