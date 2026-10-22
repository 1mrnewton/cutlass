//! One media file. This crate stores the source string and does not open it.

use cutlass_core::Timestamp;

use crate::Error;
use crate::id::AssetId;
use crate::segment::Channels;

/// What a file contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    /// Picture, and audio when `has_audio` is set.
    Video {
        has_audio: bool,
        width: u32,
        height: u32,
    },
    /// Sound only.
    Audio,
    /// A still. No media duration.
    Image { width: u32, height: u32 },
}

/// One file in the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    id: AssetId,
    kind: AssetKind,
    source: String,
    duration: Option<Timestamp>,
}

impl Asset {
    pub fn video(
        source: impl Into<String>,
        duration: Timestamp,
        width: u32,
        height: u32,
        has_audio: bool,
    ) -> Result<Self, Error> {
        Self::media(
            source,
            duration,
            AssetKind::Video {
                has_audio,
                width,
                height,
            },
            width,
            height,
        )
    }

    pub fn audio(source: impl Into<String>, duration: Timestamp) -> Result<Self, Error> {
        check_duration(duration)?;
        Ok(Self {
            id: AssetId::raw(0),
            kind: AssetKind::Audio,
            source: source_string(source)?,
            duration: Some(duration),
        })
    }

    pub fn image(source: impl Into<String>, width: u32, height: u32) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::ZeroCanvas);
        }
        Ok(Self {
            id: AssetId::raw(0),
            kind: AssetKind::Image { width, height },
            source: source_string(source)?,
            duration: None,
        })
    }

    fn media(
        source: impl Into<String>,
        duration: Timestamp,
        kind: AssetKind,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::ZeroCanvas);
        }
        check_duration(duration)?;
        Ok(Self {
            id: AssetId::raw(0),
            kind,
            source: source_string(source)?,
            duration: Some(duration),
        })
    }

    pub const fn id(&self) -> AssetId {
        self.id
    }

    pub(crate) fn assign(&mut self, id: AssetId) {
        self.id = id;
    }

    pub const fn kind(&self) -> AssetKind {
        self.kind
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Replaces the source string. An empty string is refused and the old one stays.
    pub(crate) fn set_source(&mut self, source: impl Into<String>) -> Result<(), Error> {
        self.source = source_string(source)?;
        Ok(())
    }

    pub const fn duration(&self) -> Option<Timestamp> {
        self.duration
    }

    pub const fn is_video(&self) -> bool {
        matches!(self.kind, AssetKind::Video { .. })
    }

    pub const fn visual_size(&self) -> Option<(u32, u32)> {
        match self.kind {
            AssetKind::Video { width, height, .. } | AssetKind::Image { width, height } => {
                Some((width, height))
            }
            AssetKind::Audio => None,
        }
    }

    /// Video with audio defaults to both. Silent video and images are picture.
    /// Audio is sound.
    pub const fn default_channels(&self) -> Channels {
        match self.kind {
            AssetKind::Video {
                has_audio: true, ..
            } => Channels::Both,
            AssetKind::Video {
                has_audio: false, ..
            }
            | AssetKind::Image { .. } => Channels::Picture,
            AssetKind::Audio => Channels::Sound,
        }
    }
}

fn check_duration(duration: Timestamp) -> Result<(), Error> {
    if duration.value < 0 {
        return Err(Error::NegativeDuration);
    }
    if duration.value == 0 {
        return Err(Error::ZeroDuration);
    }
    Ok(())
}

fn source_string(source: impl Into<String>) -> Result<String, Error> {
    let source = source.into();
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scale;
    use std::num::NonZeroU32;

    fn length(value: i64) -> Timestamp {
        Timestamp::new(value, scale(30_000))
    }

    #[test]
    fn a_content_uri_is_stored_and_not_opened() {
        let asset = Asset::video("content://media/1", length(30_000), 1280, 720, true).unwrap();
        assert_eq!(asset.source(), "content://media/1");
        assert_eq!(asset.duration().unwrap().timescale, scale(30_000));
        assert_eq!(asset.visual_size(), Some((1280, 720)));
        assert!(asset.is_video());
    }

    #[test]
    fn an_image_has_no_media_duration() {
        let asset = Asset::image("file:///still.png", 100, 50).unwrap();
        assert!(asset.duration().is_none());
        assert_eq!(
            asset.kind(),
            AssetKind::Image {
                width: 100,
                height: 50
            }
        );
        assert_eq!(asset.default_channels(), Channels::Picture);
    }

    #[test]
    fn audio_is_sound_without_a_picture_size() {
        let asset = Asset::audio("file:///a.wav", length(48_000)).unwrap();
        assert_eq!(asset.kind(), AssetKind::Audio);
        assert_eq!(asset.visual_size(), None);
        assert_eq!(asset.default_channels(), Channels::Sound);
        assert_eq!(asset.duration().unwrap().value, 48_000);
    }

    #[test]
    fn a_silent_video_defaults_to_picture() {
        let asset = Asset::video("file:///v.mp4", length(30_000), 320, 180, false).unwrap();
        assert_eq!(asset.default_channels(), Channels::Picture);
    }

    #[test]
    fn a_video_with_audio_defaults_to_both() {
        let asset = Asset::video("file:///v.mp4", length(30_000), 320, 180, true).unwrap();
        assert_eq!(asset.default_channels(), Channels::Both);
    }

    #[test]
    fn an_empty_source_is_rejected() {
        assert_eq!(Asset::audio("", length(1)).unwrap_err(), Error::EmptySource);
    }

    #[test]
    fn a_zero_picture_size_is_rejected() {
        assert_eq!(
            Asset::image("file:///s.png", 0, 10).unwrap_err(),
            Error::ZeroCanvas
        );
        assert_eq!(
            Asset::video("file:///v.mp4", length(1), 0, 10, true).unwrap_err(),
            Error::ZeroCanvas
        );
    }

    #[test]
    fn a_zero_or_negative_media_duration_is_rejected() {
        assert_eq!(
            Asset::audio("file:///a.wav", Timestamp::new(0, NonZeroU32::MIN)).unwrap_err(),
            Error::ZeroDuration
        );
        assert_eq!(
            Asset::audio("file:///a.wav", Timestamp::new(-1, NonZeroU32::MIN)).unwrap_err(),
            Error::NegativeDuration
        );
    }
}
