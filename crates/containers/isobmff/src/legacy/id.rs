/// A payload this workspace can decode or encode, or a four-character code it cannot name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecId {
    H264,
    Hevc,
    Av1,
    Vp8,
    Vp9,
    Mpeg4,
    Mpeg2,
    Mjpeg,
    ProRes,
    Aac,
    Opus,
    Mp3,
    Flac,
    Pcm,
    Vorbis,
    Alac,
    Ac3,
    Jpeg,
    Png,
    Webp,
    Gif,
    /// Sample entry or codec tag this build does not map to a named codec.
    Unknown([u8; 4]),
}

impl CodecId {
    #[must_use]
    pub const fn is_named(self) -> bool {
        !matches!(self, Self::Unknown(_))
    }
}
