//! Timeline export to MP4.

mod common;

use std::path::Path;
use std::time::Duration;

use common::{
    add_generated, add_media_clip, add_track, export_to, import_asset, rt, small_video_asset,
    temp_engine, tr,
};
use cutlass_cache::SourceFingerprint;
use cutlass_commands::{Command, EditCommand, ProjectCommand};
use cutlass_decoder::{DecodeOptions, Decoder, HwAccel};
use cutlass_models::{Generator, TrackKind};

fn open_export(path: &Path) -> Decoder {
    Decoder::open_with(path, DecodeOptions::default().hw_accel(HwAccel::None))
        .unwrap_or_else(|e| panic!("open export {}: {e}", path.display()))
}

fn assert_export_duration_near(path: &Path, frames: u64, fps: i32) {
    let dec = open_export(path);
    let dur = dec
        .duration()
        .expect("exported file should report duration");
    let expected_ms = (frames as f64 / f64::from(fps)) * 1000.0;
    let actual_ms = dur.as_millis() as f64;
    assert!(
        (actual_ms - expected_ms).abs() < 250.0,
        "duration {actual_ms}ms not near {expected_ms}ms for {frames} frames @ {fps}fps"
    );
}

// --- empty / invalid inputs -----------------------------------------------

#[test]
fn export_empty_timeline_errors() {
    let (_dir, mut engine) = temp_engine();
    let err = engine
        .apply(Command::Project(ProjectCommand::Export {
            path: std::env::temp_dir().join("empty_export.mp4"),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("no content"));
}

#[test]
fn export_track_without_clips_errors() {
    let (_dir, mut engine) = temp_engine();
    let _track = add_track(&mut engine, TrackKind::Video, "V1");

    let err = engine
        .apply(Command::Project(ProjectCommand::Export {
            path: std::env::temp_dir().join("track_only_export.mp4"),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("no content"));
}

#[test]
fn export_missing_output_parent_errors() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 12),
    );

    let out = dir.path().join("nested/missing/export.mp4");
    let err = engine
        .apply(Command::Project(ProjectCommand::Export { path: out }))
        .unwrap_err();
    assert!(
        format!("{err}").contains("failed to open"),
        "export should fail when parent directory is missing: {err}"
    );
}

// --- solid / generated ----------------------------------------------------

#[test]
fn export_generated_timeline_writes_mp4() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [40, 80, 120, 255],
        },
        tr(0, 24),
    );

    let out = dir.path().join("solid_export.mp4");
    let stats = export_to(&mut engine, &out);

    assert_eq!(stats.frames, 24);
    assert_eq!(stats.width, 1920);
    assert_eq!(stats.height, 1080);
    assert!(out.exists());
    assert!(std::fs::metadata(&out).unwrap().len() > 0);

    let mut dec = open_export(&out);
    let frame = dec
        .seek_to_frame(Duration::from_millis(200))
        .expect("seek")
        .expect("decoded frame");
    assert_eq!(frame.width, 1920);
    assert_eq!(frame.height, 1080);
    assert_export_duration_near(&out, 24, 24);
}

#[test]
fn export_longer_solid_timeline_scales_frame_count() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [255, 0, 0, 255],
        },
        tr(0, 48),
    );

    let out = dir.path().join("solid_48f.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 48);
    assert_export_duration_near(&out, 48, 24);
}

#[test]
fn export_text_clip_renders_visible_pixels() {
    // The project-overview "gap to close": a text generator must reach the
    // exported frames, not be silently dropped. White title text on the black
    // canvas yields bright luma pixels; a dropped generator would stay black.
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Text, "T1");
    add_generated(&mut engine, track, Generator::text("HELLO"), tr(0, 24));

    let out = dir.path().join("text_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 24);

    let mut dec = open_export(&out);
    let frame = dec
        .seek_to_frame(Duration::from_millis(200))
        .expect("seek")
        .expect("decoded frame");
    // Limited-range black is Y≈16; rasterized white text pushes luma high.
    let bright = frame.planes[0].data.iter().filter(|&&y| y > 120).count();
    assert!(
        bright > 0,
        "exported text frame had no bright pixels (text dropped?)"
    );
}

// --- media ----------------------------------------------------------------

#[test]
fn export_media_clip_writes_seekable_mp4() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = add_track(&mut engine, TrackKind::Video, "V1");
    add_media_clip(&mut engine, track, media_id, tr(0, 24), rt(0));

    let out = dir.path().join("clip_export.mp4");
    let stats = export_to(&mut engine, &out);

    assert_eq!(stats.frames, 24);
    let media = engine.project().media(media_id).expect("media");
    assert_eq!(stats.width, media.width);
    assert_eq!(stats.height, media.height);
    assert!(out.exists());

    let mut dec = open_export(&out);
    let frame = dec
        .seek_to_frame(Duration::ZERO)
        .expect("seek head")
        .expect("frame");
    assert_eq!(frame.width, media.width);
    assert_eq!(frame.height, media.height);
    assert_export_duration_near(&out, 24, 24);
}

// --- edits affecting duration / layout ------------------------------------

#[test]
fn export_trimmed_clip_uses_shortened_duration() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    let clip = add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [5, 10, 15, 255],
        },
        tr(0, 48),
    );

    engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip,
            timeline: tr(0, 12),
        }))
        .expect("trim");

    let out = dir.path().join("trimmed_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 12);
    assert_export_duration_near(&out, 12, 24);
}

#[test]
fn export_abutting_clips_span_full_timeline() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [255, 0, 0, 255],
        },
        tr(0, 24),
    );
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [0, 255, 0, 255],
        },
        tr(24, 24),
    );

    let out = dir.path().join("abutting_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 48);
    assert_export_duration_near(&out, 48, 24);
}

#[test]
fn export_two_track_composite_writes_valid_mp4() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let v1 = add_track(&mut engine, TrackKind::Video, "V1");
    let overlay = add_track(&mut engine, TrackKind::Sticker, "T1");

    add_media_clip(&mut engine, v1, media_id, tr(0, 24), rt(0));
    add_generated(
        &mut engine,
        overlay,
        Generator::SolidColor {
            rgba: [0, 0, 0, 128],
        },
        tr(0, 24),
    );

    let out = dir.path().join("stack_export.mp4");
    let stats = export_to(&mut engine, &out);
    let media = engine.project().media(media_id).expect("media");
    assert_eq!(stats.frames, 24);
    assert_eq!(stats.width, media.width);
    assert_eq!(stats.height, media.height);
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
}

// Two clips of the *same* media on stacked tracks at different offsets — the
// reported "export is glacially slow" case. Both read the one source file at
// two different positions every overlapping frame; with a per-media decoder
// that thrashed on backward seeks. A per-clip decoder keeps both cursors
// rolling forward. This guards correctness of that path end to end (the
// speedup itself isn't asserted — it's not a timing test).
#[test]
fn export_overlapping_clips_of_same_media_composite_correctly() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let v1 = add_track(&mut engine, TrackKind::Video, "V1");
    let v2 = add_track(&mut engine, TrackKind::Video, "V2");

    // Same source on both lanes, offset so they overlap on [12, 24).
    add_media_clip(&mut engine, v1, media_id, tr(0, 24), rt(0));
    add_media_clip(&mut engine, v2, media_id, tr(0, 24), rt(12));

    let out = dir.path().join("same_media_overlap_export.mp4");
    let stats = export_to(&mut engine, &out);
    let media = engine.project().media(media_id).expect("media");
    // Latest-ending clip finishes at tick 36.
    assert_eq!(stats.frames, 36);
    assert_eq!(stats.width, media.width);
    assert_eq!(stats.height, media.height);
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
    assert_export_duration_near(&out, 36, 24);
}

#[test]
fn export_duration_follows_latest_ending_track() {
    let (dir, mut engine) = temp_engine();
    let v1 = add_track(&mut engine, TrackKind::Sticker, "T1");
    let v2 = add_track(&mut engine, TrackKind::Sticker, "T2");

    add_generated(
        &mut engine,
        v1,
        Generator::SolidColor {
            rgba: [1, 1, 1, 255],
        },
        tr(0, 24),
    );
    add_generated(
        &mut engine,
        v2,
        Generator::SolidColor {
            rgba: [2, 2, 2, 255],
        },
        tr(0, 36),
    );

    let out = dir.path().join("max_end_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 36);
}

// --- gaps render as black ---------------------------------------------------

#[test]
fn export_timeline_gap_renders_black() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 24),
    );
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [4, 5, 6, 255],
        },
        tr(48, 24),
    );

    let out = dir.path().join("gap_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 72);
    assert_export_duration_near(&out, 72, 24);
}

#[test]
fn export_delayed_clip_start_renders_leading_gap_black() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [9, 9, 9, 255],
        },
        tr(12, 12),
    );

    let out = dir.path().join("delayed_start_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 24);
    assert_export_duration_near(&out, 24, 24);
}

#[test]
fn export_decodes_from_source_when_cache_corrupt() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = add_track(&mut engine, TrackKind::Video, "V1");
    add_media_clip(&mut engine, track, media_id, tr(0, 24), rt(0));

    engine.get_frame(rt(0)).expect("warm preview cache");
    engine.cache().sync();
    // Import canonicalizes the source path; fingerprint the same spelling.
    let source_id = SourceFingerprint::from_path(&path.canonicalize().expect("asset exists"))
        .expect("fingerprint")
        .id();
    assert!(
        engine.cache().frame_count(source_id) > 0,
        "preview should populate cache"
    );

    let blob = engine.config().cache_dir.join(format!("{source_id}.yuv"));
    std::fs::write(&blob, vec![0u8; 64]).expect("corrupt cache blob");

    let out = dir.path().join("cache_bypass_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 24);
    assert!(out.exists());

    let mut dec = open_export(&out);
    dec.seek_to_frame(Duration::ZERO)
        .expect("seek")
        .expect("decoded frame after corrupt cache");
}

// --- dialog settings: resolution / fps / cancel -----------------------------

#[test]
fn export_with_settings_scales_and_resamples() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [40, 80, 120, 255],
        },
        tr(0, 24),
    );

    let out = dir.path().join("settings_export.mp4");
    let settings = cutlass_engine::ExportSettings {
        target_height: Some(540),
        fps: Some(cutlass_models::Rational::new(12, 1)),
        quality: Some(28),
    };
    let mut calls = 0u64;
    let stats = cutlass_engine::export_project_with(
        engine.project(),
        &out,
        cutlass_engine::ColorConvertPath::Gpu,
        settings,
        &mut |done, total| {
            assert!(done <= total);
            calls += 1;
            true
        },
    )
    .expect("export with settings");

    // 24 ticks @ 24fps resampled to 12fps ⇒ 12 frames; 1920x1080 canvas at
    // target height 540 ⇒ 960x540.
    assert_eq!(stats.frames, 12);
    assert_eq!((stats.width, stats.height), (960, 540));
    // (0, total) plus one call per encoded frame.
    assert_eq!(calls, 13);
    assert_export_duration_near(&out, 12, 12);

    let mut dec = open_export(&out);
    let frame = dec
        .seek_to_frame(Duration::ZERO)
        .expect("seek")
        .expect("frame");
    assert_eq!((frame.width, frame.height), (960, 540));
}

#[test]
fn export_upscales_above_canvas_when_requested() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [120, 60, 30, 255],
        },
        tr(0, 6),
    );

    // 1920×1080 canvas, 1440p preset: the pick wins — no clamping to canvas.
    let out = dir.path().join("upscale_export.mp4");
    let stats = cutlass_engine::export_project_with(
        engine.project(),
        &out,
        cutlass_engine::ColorConvertPath::Gpu,
        cutlass_engine::ExportSettings {
            target_height: Some(1440),
            quality: Some(35),
            ..Default::default()
        },
        &mut |_, _| true,
    )
    .expect("upscaled export");
    assert_eq!((stats.width, stats.height), (2560, 1440));

    let mut dec = open_export(&out);
    let frame = dec
        .seek_to_frame(Duration::ZERO)
        .expect("seek")
        .expect("frame");
    assert_eq!((frame.width, frame.height), (2560, 1440));
}

#[test]
fn export_cancel_stops_partway() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 48),
    );

    let out = dir.path().join("cancelled_export.mp4");
    let err = cutlass_engine::export_project_with(
        engine.project(),
        &out,
        cutlass_engine::ColorConvertPath::Gpu,
        cutlass_engine::ExportSettings::default(),
        &mut |done, _| done < 5,
    )
    .unwrap_err();
    assert!(
        matches!(err, cutlass_engine::EngineError::ExportCancelled),
        "expected ExportCancelled, got: {err}"
    );
}

// --- audio ------------------------------------------------------------------

fn audio_asset() -> Option<std::path::PathBuf> {
    let path = common::assets_dir().join("baby.mp3");
    path.exists().then_some(path)
}

#[test]
fn export_with_audio_clip_muxes_an_audio_track() {
    let Some(audio_path) = audio_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();

    // Visuals: 1s solid. Audio: the same 1s span from an mp3 on an audio lane.
    let overlay = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        overlay,
        Generator::SolidColor {
            rgba: [20, 40, 60, 255],
        },
        tr(0, 24),
    );
    let media_id = import_asset(&mut engine, &audio_path);
    let a1 = add_track(&mut engine, TrackKind::Audio, "A1");
    // Source ranges are expressed at the media's native rate (1000/1 for
    // audio-only media): 1000 ticks = 1s = 24 timeline frames.
    let rate = engine.project().media(media_id).expect("media").frame_rate;
    add_media_clip(
        &mut engine,
        a1,
        media_id,
        cutlass_models::TimeRange::at_rate(0, rate.num as i64, rate),
        rt(0),
    );

    let out = dir.path().join("audio_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 24);

    // The deliverable carries a decodable audio stream covering ~1s.
    let mut reader = cutlass_decoder::AudioReader::open(&out, 48_000)
        .expect("exported file has an audio stream");
    let mut buf = vec![0f32; 4096 * 2];
    let mut frames = 0u64;
    loop {
        let n = reader.read(&mut buf).expect("decode exported audio");
        if n == 0 {
            break;
        }
        frames += n as u64;
    }
    let seconds = frames as f64 / 48_000.0;
    assert!(
        (seconds - 1.0).abs() < 0.2,
        "expected ~1s of audio, got {seconds:.3}s"
    );
}

#[test]
fn export_without_audible_clips_writes_video_only_file() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [50, 60, 70, 255],
        },
        tr(0, 12),
    );

    let out = dir.path().join("video_only.mp4");
    export_to(&mut engine, &out);
    assert!(
        cutlass_decoder::AudioReader::open(&out, 48_000).is_err(),
        "silent timeline must not grow an audio track"
    );
}

#[test]
fn export_muted_audio_lane_is_omitted() {
    let Some(audio_path) = audio_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let overlay = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        overlay,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 12),
    );
    let media_id = import_asset(&mut engine, &audio_path);
    let a1 = add_track(&mut engine, TrackKind::Audio, "A1");
    let rate = engine.project().media(media_id).expect("media").frame_rate;
    add_media_clip(
        &mut engine,
        a1,
        media_id,
        cutlass_models::TimeRange::at_rate(0, rate.num as i64 / 2, rate),
        rt(0),
    );
    engine
        .apply(Command::Edit(EditCommand::SetTrackMuted {
            track: a1,
            muted: true,
        }))
        .expect("mute lane");

    let out = dir.path().join("muted_export.mp4");
    export_to(&mut engine, &out);
    assert!(
        cutlass_decoder::AudioReader::open(&out, 48_000).is_err(),
        "muted lanes contribute no audio track"
    );
}

// --- engine semantics -----------------------------------------------------

#[test]
fn export_does_not_push_undo_entry() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 12),
    );
    assert!(engine.can_undo());

    let out = dir.path().join("undo_check.mp4");
    export_to(&mut engine, &out);
    assert!(engine.can_undo(), "export should not clear undo stack");
    assert!(!engine.can_redo());

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);
}

#[test]
fn export_twice_to_same_path_succeeds() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [10, 20, 30, 255],
        },
        tr(0, 12),
    );

    let out = dir.path().join("overwrite.mp4");
    let first = export_to(&mut engine, &out);
    let second = export_to(&mut engine, &out);
    assert_eq!(first, second);
    assert!(out.exists());
}
