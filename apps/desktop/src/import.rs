//! Turn an opened MP4 or MOV into an asset. The project does not open the file.
//! `docs/project.md`.

use std::path::{Path, PathBuf};

use cutlass_core::Timestamp;
use cutlass_media::{Content, Probe, TrackInfo};
use cutlass_project::Asset;

/// Why a file did not become an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImportError {
    /// The bytes are not a readable MP4 or MOV, or the picture has no size or length.
    Unread,
    /// The file has no picture track and no sound track.
    Empty,
}

/// The first enabled picture track, or the first enabled sound track.
pub(crate) fn asset_from_bytes(source: &str, bytes: &[u8]) -> Result<Asset, ImportError> {
    if source.is_empty() {
        return Err(ImportError::Unread);
    }
    let probe = Probe::parse(bytes).map_err(|_| ImportError::Unread)?;
    asset_from_tracks(source, probe.tracks())
}

fn asset_from_tracks(source: &str, tracks: &[TrackInfo]) -> Result<Asset, ImportError> {
    let video = tracks
        .iter()
        .find(|track| track.enabled && matches!(track.content, Content::Video { .. }));
    let audio = tracks
        .iter()
        .find(|track| track.enabled && matches!(track.content, Content::Audio(_)));
    if let Some(track) = video {
        let (width, height) = track.picture_size().ok_or(ImportError::Unread)?;
        return Asset::video(source, stamp(track)?, width, height, audio.is_some())
            .map_err(|_| ImportError::Unread);
    }
    if let Some(track) = audio {
        return Asset::audio(source, stamp(track)?).map_err(|_| ImportError::Unread);
    }
    Err(ImportError::Empty)
}

/// Reads one path. A missing file, or a path that is not Unicode, is unread.
pub(crate) fn from_path(path: &Path) -> Result<Asset, ImportError> {
    let Some(source) = path.to_str() else {
        return Err(ImportError::Unread);
    };
    let bytes = std::fs::read(path).map_err(|_| ImportError::Unread)?;
    asset_from_bytes(source, &bytes)
}

/// The assets that could be read, and one reason if any path was refused.
/// A refused path does not drop the assets that succeeded.
pub(crate) fn chosen(paths: &[PathBuf]) -> (Vec<Asset>, Option<ImportError>) {
    let mut assets = Vec::new();
    let mut unread = false;
    let mut empty = false;
    for path in paths {
        match from_path(path) {
            Ok(asset) => assets.push(asset),
            Err(ImportError::Unread) => unread = true,
            Err(ImportError::Empty) => empty = true,
        }
    }
    let failed = if unread {
        Some(ImportError::Unread)
    } else if empty {
        Some(ImportError::Empty)
    } else {
        None
    };
    (assets, failed)
}

/// The native picker. Cancel returns nothing. The label stays untranslated.
pub(crate) fn pick() -> Vec<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("MP4, MOV, M4A", &["mp4", "mov", "m4a"])
        .pick_files()
        .unwrap_or_default()
}

fn stamp(track: &TrackInfo) -> Result<Timestamp, ImportError> {
    let value = i64::try_from(track.duration).map_err(|_| ImportError::Unread)?;
    if value <= 0 {
        return Err(ImportError::Unread);
    }
    Ok(Timestamp::new(value, track.timescale))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::{AudioCodec, AudioStreamConfig, Timestamp, VideoCodec, VideoStreamConfig};
    use cutlass_media::{Content, FrameTiming, TrackInfo};
    use cutlass_mp4::Rotation;
    use cutlass_project::AssetKind;

    use super::{ImportError, asset_from_bytes, asset_from_tracks, chosen};

    fn scale(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("nonzero")
    }

    fn track(
        enabled: bool,
        duration: u64,
        timescale: u32,
        display: (u32, u32),
        rotation: Rotation,
        content: Content,
    ) -> TrackInfo {
        TrackInfo {
            id: 1,
            enabled,
            timescale: scale(timescale),
            duration,
            rotation,
            display_width: display.0,
            display_height: display.1,
            timing: FrameTiming::Constant(1),
            content,
        }
    }

    fn picture(width: u32, height: u32) -> Content {
        Content::Video {
            config: VideoStreamConfig {
                codec: VideoCodec::H264,
                width,
                height,
                config_record: Vec::new(),
            },
            color: None,
        }
    }

    fn sound() -> Content {
        Content::Audio(AudioStreamConfig {
            codec: AudioCodec::Aac,
            sample_rate: 48_000,
            channels: 2,
            config_record: Vec::new(),
        })
    }

    #[test]
    fn a_picture_keeps_the_display_size_and_a_sound_track_marks_audio() {
        let asset = asset_from_tracks(
            "file:///clip.mp4",
            &[
                track(
                    true,
                    400,
                    1000,
                    (180, 320),
                    Rotation::Clockwise90,
                    picture(320, 180),
                ),
                track(true, 48_000, 48_000, (0, 0), Rotation::Zero, sound()),
            ],
        )
        .expect("video");
        assert_eq!(asset.source(), "file:///clip.mp4");
        assert_eq!(
            asset.kind(),
            AssetKind::Video {
                has_audio: true,
                width: 180,
                height: 320,
            }
        );
        assert_eq!(asset.duration(), Some(Timestamp::new(400, scale(1000))));
    }

    #[test]
    fn a_missing_display_size_uses_the_encoded_size_and_the_rotation() {
        let asset = asset_from_tracks(
            "file:///clip.mp4",
            &[track(
                true,
                10,
                1000,
                (0, 0),
                Rotation::Clockwise270,
                picture(320, 180),
            )],
        )
        .expect("video");
        assert_eq!(
            asset.kind(),
            AssetKind::Video {
                has_audio: false,
                width: 180,
                height: 320,
            }
        );
    }

    #[test]
    fn sound_alone_is_audio_and_a_disabled_picture_does_not_win() {
        let asset = asset_from_tracks(
            "file:///clip.m4a",
            &[
                track(false, 10, 1000, (16, 16), Rotation::Zero, picture(16, 16)),
                track(true, 2048, 48_000, (0, 0), Rotation::Zero, sound()),
            ],
        )
        .expect("audio");
        assert_eq!(asset.kind(), AssetKind::Audio);
        assert_eq!(asset.duration(), Some(Timestamp::new(2048, scale(48_000))));
    }

    #[test]
    fn a_file_with_no_picture_or_sound_is_empty_and_junk_is_unread() {
        let timecode = track(
            true,
            1,
            1000,
            (0, 0),
            Rotation::Zero,
            Content::Other(cutlass_mp4::Fourcc::new(*b"tmcd")),
        );
        assert_eq!(
            asset_from_tracks("file:///clip.mp4", &[timecode]).unwrap_err(),
            ImportError::Empty
        );
        assert_eq!(
            asset_from_tracks(
                "file:///clip.mp4",
                &[track(
                    true,
                    0,
                    1000,
                    (16, 16),
                    Rotation::Zero,
                    picture(16, 16)
                )]
            )
            .unwrap_err(),
            ImportError::Unread
        );
        assert_eq!(
            asset_from_bytes("file:///clip.mp4", b"not a movie").unwrap_err(),
            ImportError::Unread
        );
        assert_eq!(
            asset_from_bytes("", b"not a movie").unwrap_err(),
            ImportError::Unread
        );
    }

    #[test]
    fn a_missing_file_is_unread_and_adds_nothing() {
        let (assets, failed) = chosen(&[std::path::PathBuf::from("/no/such/cutlass-import.mp4")]);
        assert!(assets.is_empty());
        assert_eq!(failed, Some(ImportError::Unread));
    }
}
