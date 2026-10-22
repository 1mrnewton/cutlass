//! Turn an opened MP4 or MOV into an asset. The project does not open the file.
//! A drop from Finder or Explorer is the same import. `docs/library.md`.

use std::path::{Path, PathBuf};

use eframe::egui::{self, Align2, Color32, FontFamily, FontId, Rect, vec2};

use cutlass_core::Timestamp;
use cutlass_media::{Content, Probe, TrackInfo};
use cutlass_project::{Asset, AssetKind};

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
        log::warn!("Could not import a file: the source is empty.");
        return Err(ImportError::Unread);
    }
    let probe = match Probe::parse(bytes) {
        Ok(probe) => probe,
        Err(error) => {
            log::warn!("Could not import {source}: {error}.");
            return Err(ImportError::Unread);
        }
    };
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
        let Some((width, height)) = track.picture_size() else {
            log::warn!("Could not import {source}: the picture has no size.");
            return Err(ImportError::Unread);
        };
        let Ok(time) = stamp(track) else {
            log::warn!("Could not import {source}: the track has no duration.");
            return Err(ImportError::Unread);
        };
        return match Asset::video(source, time, width, height, audio.is_some()) {
            Ok(asset) => Ok(asset),
            Err(error) => {
                log::warn!("Could not import {source}: {error}.");
                Err(ImportError::Unread)
            }
        };
    }
    if let Some(track) = audio {
        let Ok(time) = stamp(track) else {
            log::warn!("Could not import {source}: the track has no duration.");
            return Err(ImportError::Unread);
        };
        return match Asset::audio(source, time) {
            Ok(asset) => Ok(asset),
            Err(error) => {
                log::warn!("Could not import {source}: {error}.");
                Err(ImportError::Unread)
            }
        };
    }
    log::warn!("Could not import {source}: the file has no picture and no sound.");
    Err(ImportError::Empty)
}

/// Reads one path. A missing file, or a path that is not Unicode, is unread.
pub(crate) fn from_path(path: &Path) -> Result<Asset, ImportError> {
    let Some(source) = path.to_str() else {
        log::warn!("Could not import {}: the path is not text.", path.display());
        return Err(ImportError::Unread);
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::warn!("Could not import {}: {error}.", path.display());
            return Err(ImportError::Unread);
        }
    };
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

/// Paths the window received this frame. The integration clears the list after.
pub(crate) fn dropped(ui: &egui::Ui) -> Vec<PathBuf> {
    ui.input(|input| {
        input
            .raw
            .dropped_files
            .iter()
            .map(|file| file.path().to_path_buf())
            .collect()
    })
}

/// A file is over the window and has not been dropped or cancelled yet.
pub(crate) fn hovering(ui: &egui::Ui) -> bool {
    ui.input(|input| !input.raw.hovered_files.is_empty())
}

/// Dims the window and names the drop. A dialog that is already open does not call this.
pub(crate) fn show_hover(ui: &egui::Ui) {
    if !hovering(ui) {
        return;
    }
    let text = crate::words::of(ui, crate::words::Phrase::DropToImport);
    let screen = ui.ctx().content_rect();
    egui::Area::new(egui::Id::unique("import-drop"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .movable(false)
        .constrain(false)
        .fade_in(false)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(screen);
            ui.set_min_size(screen.size());
            ui.painter()
                .rect_filled(screen, 0.0, Color32::from_black_alpha(140));
            let font = FontId::new(14.0, FontFamily::Proportional);
            let galley =
                ui.painter()
                    .layout_no_wrap(text.to_owned(), font.clone(), crate::theme::INK);
            let plate = Rect::from_center_size(screen.center(), galley.size() + vec2(32.0, 16.0));
            ui.painter().rect_filled(plate, 10.0, crate::theme::BAR);
            ui.painter().text(
                plate.center(),
                Align2::CENTER_CENTER,
                text,
                font,
                crate::theme::INK,
            );
            crate::tip::read(ui, plate, ui.scope_id().with("import-drop-label"), text);
        });
}

/// A video stays a video, and sound stays sound. Size and audio inside a video may differ.
pub(crate) fn same_kind(current: AssetKind, replacement: AssetKind) -> bool {
    matches!(
        (current, replacement),
        (AssetKind::Video { .. }, AssetKind::Video { .. })
            | (AssetKind::Audio, AssetKind::Audio)
            | (AssetKind::Image { .. }, AssetKind::Image { .. })
    )
}

/// One file from the native picker. Cancel returns nothing.
pub(crate) fn pick_one() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("MP4, MOV, M4A", &["mp4", "mov", "m4a"])
        .pick_file()
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

    use super::{ImportError, asset_from_bytes, asset_from_tracks, chosen, same_kind};

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
            mastering: None,
            content_light: None,
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

    #[test]
    fn a_replacement_must_be_the_same_kind() {
        let picture = AssetKind::Video {
            has_audio: true,
            width: 16,
            height: 16,
        };
        let silent = AssetKind::Video {
            has_audio: false,
            width: 32,
            height: 8,
        };
        assert!(same_kind(picture, silent));
        assert!(same_kind(AssetKind::Audio, AssetKind::Audio));
        assert!(!same_kind(picture, AssetKind::Audio));
        assert!(!same_kind(
            AssetKind::Image {
                width: 1,
                height: 1
            },
            picture
        ));
    }

    #[test]
    fn a_folder_is_unread() {
        let (assets, failed) = chosen(&[std::env::temp_dir()]);
        assert!(assets.is_empty());
        assert_eq!(failed, Some(ImportError::Unread));
    }

    #[derive(Debug)]
    struct FakeFile {
        path: std::path::PathBuf,
    }

    impl eframe::egui::DroppedFile for FakeFile {
        fn path(&self) -> &std::path::Path {
            &self.path
        }

        fn bytes(&self) -> Result<Vec<u8>, String> {
            Err("the shell reads the path".into())
        }
    }

    #[test]
    fn a_hovered_file_is_named_and_a_drop_is_its_path() {
        let ctx = eframe::egui::Context::default();
        ctx.enable_accesskit();
        crate::words::install(&ctx, crate::words::Language::English);
        let screen = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(800.0, 600.0),
        );
        let path = std::path::PathBuf::from("/no/such/cutlass-drop.mp4");
        let mut time = 0.0;
        let mut pass = |input: eframe::egui::RawInput| {
            time += 0.05;
            let mut got = Vec::new();
            let mut input = input;
            input.screen_rect = Some(screen);
            input.time = Some(time);
            let mut output = ctx.run_ui(input, |ui| {
                super::show_hover(ui);
                got = super::dropped(ui);
            });
            let tree = output.platform_output.accesskit_update.take();
            output.textures_delta.clear();
            (tree.expect("the tree is built"), got)
        };
        let hover = eframe::egui::RawInput {
            hovered_files: vec![eframe::egui::HoveredFile {
                path: Some(path.clone()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _warm = pass(hover.clone());
        let (tree, got) = pass(hover);
        assert!(got.is_empty());
        assert!(tree.nodes.iter().any(|(_, node)| {
            node.label() == Some("Drop to import") || node.value() == Some("Drop to import")
        }));
        let (tree, got) = pass(eframe::egui::RawInput {
            dropped_files: vec![std::sync::Arc::new(FakeFile { path: path.clone() })],
            ..Default::default()
        });
        assert_eq!(got, vec![path]);
        assert!(tree.nodes.iter().all(|(_, node)| {
            node.label() != Some("Drop to import") && node.value() != Some("Drop to import")
        }));
    }
}
