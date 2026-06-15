//! Auto-captions end-to-end (M9 Phase 4): decode a real clip's audio, run it
//! through an injected (stub) transcribe backend, and assert the cues land as
//! subtitle text clips on a fresh lane in one undoable entry.
//!
//! Gated on a local audio asset (`local-assets/assets/baby.mp3`) since it
//! exercises the decode path; skips cleanly when the asset is absent, like the
//! other decode-dependent engine tests.

mod common;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::{assets_dir, import_asset, temp_engine};
use cutlass_engine::{ApplyOutcome, EditOutcome};
use cutlass_ml::{
    CaptionLayout, Segment, StubTranscriber, TranscribeOptions, Transcript, Word,
};
use cutlass_models::{ClipSource, Generator, Rational, TimeRange, TrackKind};

fn audio_asset() -> Option<std::path::PathBuf> {
    let path = assets_dir().join("baby.mp3");
    path.exists().then_some(path)
}

/// A two-cue transcript the stub returns for any audio.
fn two_cue_stub() -> StubTranscriber {
    StubTranscriber::new(Transcript {
        segments: vec![
            Segment {
                text: "first line.".into(),
                start: 0.0,
                end: 0.8,
                words: vec![
                    Word {
                        text: "first".into(),
                        start: 0.0,
                        end: 0.4,
                        confidence: None,
                    },
                    Word {
                        text: "line.".into(),
                        start: 0.4,
                        end: 0.8,
                        confidence: None,
                    },
                ],
            },
            Segment {
                text: "second line.".into(),
                start: 1.0,
                end: 1.8,
                words: vec![
                    Word {
                        text: "second".into(),
                        start: 1.0,
                        end: 1.4,
                        confidence: None,
                    },
                    Word {
                        text: "line.".into(),
                        start: 1.4,
                        end: 1.8,
                        confidence: None,
                    },
                ],
            },
        ],
        language: Some("en".into()),
    })
}

#[test]
fn captions_land_as_text_clips_on_a_fresh_lane_and_undo_cleanly() {
    let Some(audio) = audio_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    engine.set_transcriber(Arc::new(two_cue_stub()));

    // Place ~2s of the audio on an audio lane.
    let media_id = import_asset(&mut engine, &audio);
    let rate = engine.project().media(media_id).expect("media").frame_rate;
    let a1 = match engine
        .apply(cutlass_engine::Command::Edit(
            cutlass_engine::EditCommand::AddTrack {
                kind: TrackKind::Audio,
                name: "A1".into(),
                index: None,
            },
        ))
        .unwrap()
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    };
    let clip = match engine
        .apply(cutlass_engine::Command::Edit(
            cutlass_engine::EditCommand::AddClip {
                track: a1,
                media: media_id,
                source: TimeRange::at_rate(0, 2 * rate.num as i64, rate),
                start: cutlass_models::RationalTime::new(0, Rational::FPS_24),
            },
        ))
        .unwrap()
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    };

    let tracks_before = engine.project().timeline().order().len();

    let cancel = AtomicBool::new(false);
    let outcome = engine
        .generate_captions(
            clip,
            TranscribeOptions::default(),
            CaptionLayout::default(),
            &cancel,
            &mut |_| {},
        )
        .expect("captions generated");

    let caption_track = match outcome {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected a captions track, got {other:?}"),
    };

    // A fresh text lane carrying two subtitle text clips.
    assert_eq!(engine.project().timeline().order().len(), tracks_before + 1);
    let track = engine
        .project()
        .timeline()
        .track(caption_track)
        .expect("captions track");
    assert_eq!(track.kind, TrackKind::Text);
    let mut texts: Vec<String> = track
        .clips()
        .filter_map(|c| match &c.content {
            ClipSource::Generated(Generator::Text { content, .. }) => Some(content.clone()),
            _ => None,
        })
        .collect();
    texts.sort();
    assert_eq!(texts, vec!["first line.", "second line."]);

    // The whole pass is one undo entry: it removes the lane and both clips.
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().order().len(), tracks_before);
    assert!(engine.project().timeline().track(caption_track).is_none());

    // And redo brings them all back.
    assert!(engine.redo());
    assert_eq!(engine.project().timeline().order().len(), tracks_before + 1);
}

#[test]
fn captions_without_a_backend_error() {
    let (_dir, mut engine) = temp_engine();
    let cancel = AtomicBool::new(false);
    let err = engine
        .generate_captions(
            cutlass_models::ClipId::from_raw(1),
            TranscribeOptions::default(),
            CaptionLayout::default(),
            &cancel,
            &mut |_| {},
        )
        .unwrap_err();
    assert!(
        matches!(err, cutlass_engine::EngineError::Transcribe(_)),
        "expected a transcribe error, got: {err}"
    );
}
