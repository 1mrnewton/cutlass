//! One id for each codec this workspace names, and one id for each
//! container family. Identifying a tag the workspace does not name is a
//! [`StreamCodec::Unknown`] value. It is not a panic, and it is not dropped.

use std::fmt;

/// A codec this workspace has a crate for.
///
/// A variant existing here does not mean this build can decode it. The
/// decoder factory accepts an id only after an implementation registers.
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
    Prores,
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
}

/// A container family this workspace has a crate for.
///
/// `Mp3` and `Flac` name the file families. [`CodecId::Mp3`] and
/// [`CodecId::Flac`] name the bitstreams those files carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContainerId {
    Isobmff,
    Matroska,
    MpegTs,
    Avi,
    Riff,
    Aiff,
    Still,
    Mp3,
    Adts,
    Ogg,
    Flac,
    Caf,
}

/// A tag from a file that does not match a [`CodecId`].
///
/// The bytes are the file's own codec tag. They are not a fourcc table
/// and they are not interpreted here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnknownCodec {
    tag: Vec<u8>,
}

/// The tag does not fit in an [`UnknownCodec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagError {
    /// Longer than [`UnknownCodec::MAX_TAG`] bytes.
    TooLong { len: usize },
}

impl fmt::Display for TagError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong { len } => {
                write!(
                    formatter,
                    "codec tag is {len} bytes, and the limit is {}",
                    UnknownCodec::MAX_TAG
                )
            }
        }
    }
}

impl std::error::Error for TagError {}

impl UnknownCodec {
    /// Matroska codec ids and four-character tags both fit.
    pub const MAX_TAG: usize = 64;

    /// Stores `tag` when it is within [`Self::MAX_TAG`].
    ///
    /// An empty tag is still unknown. The stream that carries it stays.
    pub fn new(tag: impl Into<Vec<u8>>) -> Result<Self, TagError> {
        let tag = tag.into();
        if tag.len() > Self::MAX_TAG {
            return Err(TagError::TooLong { len: tag.len() });
        }
        Ok(Self { tag })
    }

    #[must_use]
    pub fn tag(&self) -> &[u8] {
        &self.tag
    }
}

/// The codec a stream claimed.
///
/// [`StreamCodec::Unknown`] is a successful identification of a tag this
/// workspace does not name. Opening the file still returns that stream.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StreamCodec {
    Known(CodecId),
    Unknown(UnknownCodec),
}

impl StreamCodec {
    #[must_use]
    pub fn known(self) -> Option<CodecId> {
        match self {
            Self::Known(id) => Some(id),
            Self::Unknown(_) => None,
        }
    }

    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
}

#[cfg(test)]
mod tests {
    use super::{CodecId, ContainerId, StreamCodec, TagError, UnknownCodec};

    fn codec_label(id: CodecId) -> &'static str {
        match id {
            CodecId::H264 => "h264",
            CodecId::Hevc => "hevc",
            CodecId::Av1 => "av1",
            CodecId::Vp8 => "vp8",
            CodecId::Vp9 => "vp9",
            CodecId::Mpeg4 => "mpeg4",
            CodecId::Mpeg2 => "mpeg2",
            CodecId::Mjpeg => "mjpeg",
            CodecId::Prores => "prores",
            CodecId::Aac => "aac",
            CodecId::Opus => "opus",
            CodecId::Mp3 => "mp3",
            CodecId::Flac => "flac",
            CodecId::Pcm => "pcm",
            CodecId::Vorbis => "vorbis",
            CodecId::Alac => "alac",
            CodecId::Ac3 => "ac3",
            CodecId::Jpeg => "jpeg",
            CodecId::Png => "png",
            CodecId::Webp => "webp",
            CodecId::Gif => "gif",
        }
    }

    fn family_label(id: ContainerId) -> &'static str {
        match id {
            ContainerId::Isobmff => "isobmff",
            ContainerId::Matroska => "matroska",
            ContainerId::MpegTs => "mpegts",
            ContainerId::Avi => "avi",
            ContainerId::Riff => "riff",
            ContainerId::Aiff => "aiff",
            ContainerId::Still => "still",
            ContainerId::Mp3 => "mp3",
            ContainerId::Adts => "adts",
            ContainerId::Ogg => "ogg",
            ContainerId::Flac => "flac",
            ContainerId::Caf => "caf",
        }
    }

    #[test]
    fn workspace_codecs_and_families_each_have_an_id() {
        assert_eq!(codec_label(CodecId::H264), "h264");
        assert_eq!(codec_label(CodecId::Gif), "gif");
        assert_eq!(family_label(ContainerId::Isobmff), "isobmff");
        assert_eq!(family_label(ContainerId::Caf), "caf");
        assert_ne!(CodecId::Mp3, CodecId::Flac);
        assert_ne!(ContainerId::Mp3, ContainerId::Flac);
    }

    #[test]
    fn an_unknown_codec_is_a_value_and_not_a_known_id() {
        let unknown = UnknownCodec::new(b"zz99".to_vec()).expect("tag");
        let identified = StreamCodec::Unknown(unknown.clone());
        assert!(identified.is_unknown());
        assert_eq!(identified.known(), None);
        assert_eq!(unknown.tag(), b"zz99");
        assert_ne!(
            StreamCodec::Known(CodecId::H264),
            StreamCodec::Unknown(unknown)
        );
    }

    #[test]
    fn an_empty_unknown_tag_is_still_unknown() {
        let unknown = UnknownCodec::new(Vec::new()).expect("empty tag");
        assert!(unknown.tag().is_empty());
        assert!(StreamCodec::Unknown(unknown).is_unknown());
    }

    #[test]
    fn a_tag_past_the_limit_is_refused() {
        let tag = vec![0; UnknownCodec::MAX_TAG + 1];
        let error = UnknownCodec::new(tag).expect_err("too long");
        assert_eq!(
            error,
            TagError::TooLong {
                len: UnknownCodec::MAX_TAG + 1
            }
        );
    }
}
