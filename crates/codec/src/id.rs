/// One id per codec the workspace names.
///
/// A variant existing here does not mean this build can decode it. The
/// decoder factory accepts an id only after an implementation registers.
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
    Latm,
    Matroska,
    Mp3,
    MpegTs,
    Ogg,
    Riff,
    Still,
}
