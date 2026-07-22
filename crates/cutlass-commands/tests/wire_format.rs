//! Wire-format lock for the command protocol.
//!
//! The shells (iOS/Android FFI) and the future AI agent speak commands as
//! JSON, so the serialized shape is a public contract. These tests (a) golden
//! the exact JSON for representative commands and (b) roundtrip **every**
//! variant through the untagged [`Command`] surface. The `variant_name`
//! matches are deliberately non-wildcard: adding a variant fails compilation
//! here until a sample (and thus a roundtrip) is added.

use std::path::PathBuf;

use cutlass_commands::{
    AnimationRef, AnimationSlot, AudioRole, CanvasAspect, ChromaKey, ClipId, ClipParam,
    ClipTransform, ColorAdjustments, Command, CropRect, Easing, EditCommand, EditOutcome, Filter,
    Generator, Lut, MarkerColor, MarkerId, Mask, MaskKind, MediaId, Param, ParamValue,
    ProjectCommand, Rational, RationalTime, Replaceable, StabilizeLevel, TemplateMeta,
    TemplatePick, TimeRange, TrackId, TrackKind,
};
use serde_json::{Value, json};

const FPS: Rational = Rational::FPS_30;

fn t(value: i64) -> RationalTime {
    RationalTime::new(value, FPS)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS)
}

fn clip(raw: u64) -> ClipId {
    ClipId::from_raw(raw)
}

fn track(raw: u64) -> TrackId {
    TrackId::from_raw(raw)
}

fn media(raw: u64) -> MediaId {
    MediaId::from_raw(raw)
}

/// One sample per [`ProjectCommand`] variant.
fn project_samples() -> Vec<ProjectCommand> {
    vec![
        ProjectCommand::Import {
            path: PathBuf::from("/media/clip.mp4"),
        },
        ProjectCommand::RemoveMedia { media: media(3) },
        ProjectCommand::Save {
            path: PathBuf::from("/projects/cut.cutlass"),
        },
        ProjectCommand::Open {
            path: PathBuf::from("/projects/cut.cutlass"),
        },
        ProjectCommand::Load {
            path: PathBuf::from("/projects/cut.cutlass"),
        },
        ProjectCommand::RelinkMedia {
            media: media(3),
            path: PathBuf::from("/media/moved.mp4"),
        },
        ProjectCommand::Export {
            path: PathBuf::from("/out/final.mp4"),
        },
        ProjectCommand::SaveTemplate {
            path: PathBuf::from("/templates/intro.cutlasst"),
            meta: TemplateMeta::new("Intro"),
        },
        ProjectCommand::ApplyTemplate {
            path: PathBuf::from("/templates/intro.cutlasst"),
            picks: vec![
                TemplatePick {
                    path: PathBuf::from("/media/a.mp4"),
                    source_in: Some(t(30)),
                },
                TemplatePick {
                    path: PathBuf::from("/media/b.mp4"),
                    source_in: None,
                },
            ],
        },
    ]
}

/// One sample per [`EditCommand`] variant.
fn edit_samples() -> Vec<EditCommand> {
    vec![
        EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "V2".into(),
            index: Some(1),
            pinned: false,
        },
        EditCommand::AddClip {
            track: track(1),
            media: media(2),
            source: tr(0, 90),
            start: t(30),
        },
        EditCommand::ExtractAudio {
            clip: clip(4),
            to_track: Some(track(2)),
        },
        EditCommand::DuplicateClip {
            clip: clip(4),
            to_track: track(2),
            start: t(120),
        },
        EditCommand::FreezeFrame {
            clip: clip(4),
            at: t(45),
            duration: t(30),
        },
        EditCommand::AddGenerated {
            track: track(1),
            generator: Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            timeline: tr(0, 60),
        },
        EditCommand::SetGenerator {
            clip: clip(4),
            generator: Generator::Text {
                content: "Title".into(),
                style: Default::default(),
            },
        },
        EditCommand::SetClipMedia {
            clip: clip(4),
            media: media(2),
            source: tr(15, 45),
        },
        EditCommand::SetReplaceable {
            clip: clip(4),
            replaceable: Some(Replaceable::new(0)),
        },
        EditCommand::SetTextEditable {
            clip: clip(4),
            editable: true,
        },
        EditCommand::SetClipTransform {
            clip: clip(4),
            transform: ClipTransform {
                position: [0.25, -0.5],
                anchor_point: [0.5, 0.5],
                scale: 2.0,
                rotation: 90.0,
                opacity: 0.75,
            },
            at: Some(t(45)),
        },
        EditCommand::SetParamKeyframe {
            clip: clip(4),
            param: ClipParam::Scale,
            at: t(45),
            value: ParamValue::Scalar(1.5),
            easing: Easing::EaseInOut,
        },
        EditCommand::RemoveParamKeyframe {
            clip: clip(4),
            param: ClipParam::Position,
            at: t(45),
        },
        EditCommand::SetParamConstant {
            clip: clip(4),
            param: ClipParam::Opacity,
            value: ParamValue::Scalar(0.5),
        },
        EditCommand::SetClipSpeed {
            clip: clip(4),
            speed: Rational::new(2, 1),
            reversed: true,
        },
        EditCommand::SetSpeedCurve {
            clip: clip(4),
            curve: Some(Param::Constant(1.0)),
        },
        EditCommand::SetClipPitch {
            clip: clip(4),
            preserve_pitch: true,
        },
        EditCommand::SetClipCrop {
            clip: clip(4),
            crop: CropRect {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            flip_h: true,
            flip_v: false,
        },
        EditCommand::SetClipAudio {
            clip: clip(4),
            volume: Some(0.5),
            fade_in: t(15),
            fade_out: t(30),
        },
        EditCommand::SetClipDenoise {
            clip: clip(4),
            denoise: true,
        },
        EditCommand::SetClipMask {
            clip: clip(4),
            mask: Some(Mask {
                kind: MaskKind::Circle,
                feather: 0.5.into(),
                invert: true,
            }),
        },
        EditCommand::SetClipChroma {
            clip: clip(4),
            chroma: Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.7.into(),
                shadow: 0.2.into(),
            }),
        },
        EditCommand::SetClipStabilize {
            clip: clip(4),
            stabilize: Some(StabilizeLevel::MaxSmooth),
        },
        EditCommand::SetClipFilter {
            clip: clip(4),
            filter: Some(Filter {
                id: "vivid".into(),
                intensity: 0.6.into(),
            }),
        },
        EditCommand::SetClipLut {
            clip: clip(4),
            lut: Some(Lut {
                path: "/luts/teal-orange.cube".into(),
                intensity: 0.7.into(),
            }),
        },
        EditCommand::SetClipAdjustments {
            clip: clip(4),
            adjust: ColorAdjustments {
                brightness: 0.1.into(),
                contrast: (-0.2).into(),
                saturation: 0.3.into(),
                exposure: 0.0.into(),
                temperature: (-0.4).into(),
            },
        },
        EditCommand::SetClipAnimation {
            clip: clip(4),
            slot: AnimationSlot::In,
            animation: Some(AnimationRef::new("fade_in")),
        },
        EditCommand::SetAudioRole {
            clip: clip(4),
            role: Some(AudioRole::Voiceover),
        },
        EditCommand::AddEffect {
            clip: clip(4),
            effect_id: "gaussian_blur".into(),
        },
        EditCommand::RemoveEffect {
            clip: clip(4),
            index: 0,
        },
        EditCommand::MoveEffect {
            clip: clip(4),
            from_index: 0,
            to_index: 2,
        },
        EditCommand::SetEffectParam {
            clip: clip(4),
            index: 0,
            param: 0,
            value: 8.0,
        },
        EditCommand::AddTransition {
            clip: clip(4),
            transition_id: "crossfade".into(),
        },
        EditCommand::RemoveTransition { clip: clip(4) },
        EditCommand::SetTransition {
            clip: clip(4),
            duration: 15,
        },
        EditCommand::SplitClip {
            clip: clip(4),
            at: t(45),
        },
        EditCommand::TrimClip {
            clip: clip(4),
            timeline: tr(30, 60),
        },
        EditCommand::MoveClip {
            clip: clip(4),
            to_track: track(2),
            start: t(90),
        },
        EditCommand::RemoveClip { clip: clip(4) },
        EditCommand::RemoveTrack { track: track(2) },
        EditCommand::MoveTrack {
            track: track(2),
            index: 1,
        },
        EditCommand::SetTrackName {
            track: track(2),
            name: "Dialogue".into(),
        },
        EditCommand::SetTrackEnabled {
            track: track(2),
            enabled: false,
        },
        EditCommand::SetTrackMuted {
            track: track(2),
            muted: true,
        },
        EditCommand::SetTrackLocked {
            track: track(2),
            locked: true,
        },
        EditCommand::SetTrackDuckSource {
            track: track(2),
            duck_source: true,
        },
        EditCommand::RippleDelete { clip: clip(4) },
        EditCommand::ShiftClips {
            track: track(1),
            from: t(60),
            delta: t(-30),
        },
        EditCommand::RippleInsert {
            track: track(1),
            media: media(2),
            source: tr(0, 90),
            at: t(60),
        },
        EditCommand::LinkClips {
            clips: vec![clip(4), clip(5)],
        },
        EditCommand::UnlinkClips {
            clips: vec![clip(4), clip(5)],
        },
        EditCommand::DuckLanes {
            voice: vec![clip(4)],
            music: vec![clip(5)],
            threshold: 0.125,
            amount: 0.5,
            attack: 0.25,
            release: 0.5,
        },
        EditCommand::DetectBeats { clip: clip(4) },
        EditCommand::ClearBeats { clip: clip(4) },
        EditCommand::AddMarker {
            at: t(45),
            name: "Beat drop".into(),
            color: Some(MarkerColor::Teal),
        },
        EditCommand::RemoveMarker {
            marker: MarkerId::from_raw(7),
        },
        EditCommand::SetMarker {
            marker: MarkerId::from_raw(7),
            at: t(60),
            name: "Chorus".into(),
            color: MarkerColor::Pink,
        },
        EditCommand::SetCanvas {
            aspect: CanvasAspect::Tall9x16,
            background: [0, 0, 0],
        },
        EditCommand::SetProjectName {
            name: "Sunday cut".into(),
        },
    ]
}

/// Compile-time exhaustiveness: adding a variant breaks these matches until a
/// sample is added above.
fn project_variant_name(cmd: &ProjectCommand) -> &'static str {
    match cmd {
        ProjectCommand::Import { .. } => "Import",
        ProjectCommand::RemoveMedia { .. } => "RemoveMedia",
        ProjectCommand::Save { .. } => "Save",
        ProjectCommand::Open { .. } => "Open",
        ProjectCommand::Load { .. } => "Load",
        ProjectCommand::RelinkMedia { .. } => "RelinkMedia",
        ProjectCommand::Export { .. } => "Export",
        ProjectCommand::SaveTemplate { .. } => "SaveTemplate",
        ProjectCommand::ApplyTemplate { .. } => "ApplyTemplate",
    }
}

fn edit_variant_name(cmd: &EditCommand) -> &'static str {
    match cmd {
        EditCommand::AddTrack { .. } => "AddTrack",
        EditCommand::AddClip { .. } => "AddClip",
        EditCommand::ExtractAudio { .. } => "ExtractAudio",
        EditCommand::DuplicateClip { .. } => "DuplicateClip",
        EditCommand::FreezeFrame { .. } => "FreezeFrame",
        EditCommand::AddGenerated { .. } => "AddGenerated",
        EditCommand::SetGenerator { .. } => "SetGenerator",
        EditCommand::SetClipMedia { .. } => "SetClipMedia",
        EditCommand::SetReplaceable { .. } => "SetReplaceable",
        EditCommand::SetTextEditable { .. } => "SetTextEditable",
        EditCommand::SetClipTransform { .. } => "SetClipTransform",
        EditCommand::SetParamKeyframe { .. } => "SetParamKeyframe",
        EditCommand::RemoveParamKeyframe { .. } => "RemoveParamKeyframe",
        EditCommand::SetParamConstant { .. } => "SetParamConstant",
        EditCommand::SetClipSpeed { .. } => "SetClipSpeed",
        EditCommand::SetSpeedCurve { .. } => "SetSpeedCurve",
        EditCommand::SetClipPitch { .. } => "SetClipPitch",
        EditCommand::SetClipCrop { .. } => "SetClipCrop",
        EditCommand::SetClipAudio { .. } => "SetClipAudio",
        EditCommand::SetClipDenoise { .. } => "SetClipDenoise",
        EditCommand::SetClipMask { .. } => "SetClipMask",
        EditCommand::SetClipChroma { .. } => "SetClipChroma",
        EditCommand::SetClipStabilize { .. } => "SetClipStabilize",
        EditCommand::SetClipFilter { .. } => "SetClipFilter",
        EditCommand::SetClipLut { .. } => "SetClipLut",
        EditCommand::SetClipAdjustments { .. } => "SetClipAdjustments",
        EditCommand::SetClipAnimation { .. } => "SetClipAnimation",
        EditCommand::SetAudioRole { .. } => "SetAudioRole",
        EditCommand::AddEffect { .. } => "AddEffect",
        EditCommand::RemoveEffect { .. } => "RemoveEffect",
        EditCommand::MoveEffect { .. } => "MoveEffect",
        EditCommand::SetEffectParam { .. } => "SetEffectParam",
        EditCommand::AddTransition { .. } => "AddTransition",
        EditCommand::RemoveTransition { .. } => "RemoveTransition",
        EditCommand::SetTransition { .. } => "SetTransition",
        EditCommand::SplitClip { .. } => "SplitClip",
        EditCommand::TrimClip { .. } => "TrimClip",
        EditCommand::MoveClip { .. } => "MoveClip",
        EditCommand::RemoveClip { .. } => "RemoveClip",
        EditCommand::RemoveTrack { .. } => "RemoveTrack",
        EditCommand::MoveTrack { .. } => "MoveTrack",
        EditCommand::SetTrackName { .. } => "SetTrackName",
        EditCommand::SetTrackEnabled { .. } => "SetTrackEnabled",
        EditCommand::SetTrackMuted { .. } => "SetTrackMuted",
        EditCommand::SetTrackLocked { .. } => "SetTrackLocked",
        EditCommand::SetTrackDuckSource { .. } => "SetTrackDuckSource",
        EditCommand::RippleDelete { .. } => "RippleDelete",
        EditCommand::ShiftClips { .. } => "ShiftClips",
        EditCommand::RippleInsert { .. } => "RippleInsert",
        EditCommand::LinkClips { .. } => "LinkClips",
        EditCommand::UnlinkClips { .. } => "UnlinkClips",
        EditCommand::DuckLanes { .. } => "DuckLanes",
        EditCommand::DetectBeats { .. } => "DetectBeats",
        EditCommand::ClearBeats { .. } => "ClearBeats",
        EditCommand::AddMarker { .. } => "AddMarker",
        EditCommand::RemoveMarker { .. } => "RemoveMarker",
        EditCommand::SetMarker { .. } => "SetMarker",
        EditCommand::SetCanvas { .. } => "SetCanvas",
        EditCommand::SetProjectName { .. } => "SetProjectName",
    }
}

// --- roundtrips -------------------------------------------------------------

#[test]
fn command_variant_counts_are_locked() {
    assert_eq!(project_samples().len(), 9);
    assert_eq!(edit_samples().len(), 59);
}

#[test]
fn every_project_command_roundtrips_through_command() {
    for sample in project_samples() {
        let wrapped = Command::Project(sample.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(back, wrapped, "roundtrip failed for {json}");

        // The wire tag is exactly the variant name.
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], project_variant_name(&sample));
    }
}

#[test]
fn every_edit_command_roundtrips_through_command() {
    for sample in edit_samples() {
        let wrapped = Command::Edit(sample.clone());
        let json = serde_json::to_string(&wrapped).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(back, wrapped, "roundtrip failed for {json}");

        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], edit_variant_name(&sample));
    }
}

#[test]
fn command_layer_is_flat_json() {
    // The untagged `Command` never adds a Project/Edit envelope.
    let json =
        serde_json::to_value(Command::Edit(EditCommand::RemoveClip { clip: clip(9) })).unwrap();
    assert_eq!(json, json!({"type": "RemoveClip", "clip": 9}));
}

// --- goldens ----------------------------------------------------------------

#[test]
fn golden_import() {
    let cmd = Command::Project(ProjectCommand::Import {
        path: PathBuf::from("/media/clip.mp4"),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({"type": "Import", "path": "/media/clip.mp4"})
    );
}

#[test]
fn golden_add_clip() {
    let cmd = Command::Edit(EditCommand::AddClip {
        track: track(1),
        media: media(2),
        source: tr(0, 90),
        start: t(30),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "AddClip",
            "track": 1,
            "media": 2,
            "source": {
                "start": {"value": 0, "rate": {"num": 30, "den": 1}},
                "duration": {"value": 90, "rate": {"num": 30, "den": 1}},
            },
            "start": {"value": 30, "rate": {"num": 30, "den": 1}},
        })
    );
}

#[test]
fn golden_extract_audio() {
    let explicit = Command::Edit(EditCommand::ExtractAudio {
        clip: clip(4),
        to_track: Some(track(2)),
    });
    assert_eq!(
        serde_json::to_value(&explicit).unwrap(),
        json!({"type": "ExtractAudio", "clip": 4, "to_track": 2})
    );

    let automatic = Command::Edit(EditCommand::ExtractAudio {
        clip: clip(4),
        to_track: None,
    });
    assert_eq!(
        serde_json::to_value(&automatic).unwrap(),
        json!({"type": "ExtractAudio", "clip": 4, "to_track": null})
    );
}

#[test]
fn golden_duplicate_clip() {
    let cmd = Command::Edit(EditCommand::DuplicateClip {
        clip: clip(4),
        to_track: track(2),
        start: t(120),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "DuplicateClip",
            "clip": 4,
            "to_track": 2,
            "start": {"value": 120, "rate": {"num": 30, "den": 1}},
        })
    );
}

#[test]
fn golden_freeze_frame() {
    let cmd = Command::Edit(EditCommand::FreezeFrame {
        clip: clip(4),
        at: t(45),
        duration: t(30),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "FreezeFrame",
            "clip": 4,
            "at": {"value": 45, "rate": {"num": 30, "den": 1}},
            "duration": {"value": 30, "rate": {"num": 30, "den": 1}},
        })
    );
}

#[test]
fn golden_split_clip() {
    let cmd = Command::Edit(EditCommand::SplitClip {
        clip: clip(4),
        at: t(45),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "SplitClip",
            "clip": 4,
            "at": {"value": 45, "rate": {"num": 30, "den": 1}},
        })
    );
}

#[test]
fn golden_unlink_clips() {
    let cmd = Command::Edit(EditCommand::UnlinkClips {
        clips: vec![clip(4), clip(5)],
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({"type": "UnlinkClips", "clips": [4, 5]})
    );
}

#[test]
fn golden_move_effect() {
    let cmd = Command::Edit(EditCommand::MoveEffect {
        clip: clip(4),
        from_index: 0,
        to_index: 2,
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({"type": "MoveEffect", "clip": 4, "from_index": 0, "to_index": 2})
    );
}

#[test]
fn golden_set_param_keyframe() {
    let cmd = Command::Edit(EditCommand::SetParamKeyframe {
        clip: clip(4),
        param: ClipParam::Scale,
        at: t(45),
        value: ParamValue::Scalar(1.5),
        easing: Easing::EaseInOut,
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "SetParamKeyframe",
            "clip": 4,
            "param": "scale",
            "at": {"value": 45, "rate": {"num": 30, "den": 1}},
            "value": {"scalar": 1.5},
            "easing": "ease_in_out",
        })
    );
}

#[test]
fn golden_set_canvas() {
    let cmd = Command::Edit(EditCommand::SetCanvas {
        aspect: CanvasAspect::Tall9x16,
        background: [0, 0, 0],
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({"type": "SetCanvas", "aspect": "9:16", "background": [0, 0, 0]})
    );
}

#[test]
fn golden_add_generated_solid() {
    let cmd = Command::Edit(EditCommand::AddGenerated {
        track: track(1),
        generator: Generator::SolidColor {
            rgba: [255, 0, 0, 255],
        },
        timeline: tr(0, 60),
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({
            "type": "AddGenerated",
            "track": 1,
            "generator": {"SolidColor": {"rgba": [255, 0, 0, 255]}},
            "timeline": {
                "start": {"value": 0, "rate": {"num": 30, "den": 1}},
                "duration": {"value": 60, "rate": {"num": 30, "den": 1}},
            },
        })
    );
}

#[test]
fn golden_speed_curve_forms() {
    // A constant curve serializes as the bare value…
    let constant = Command::Edit(EditCommand::SetSpeedCurve {
        clip: clip(4),
        curve: Some(Param::Constant(1.0)),
    });
    assert_eq!(
        serde_json::to_value(&constant).unwrap(),
        json!({"type": "SetSpeedCurve", "clip": 4, "curve": 1.0})
    );

    // …and a keyframed curve as the `{"kf": [...]}` map (the project-file
    // format), roundtripping through `Command`.
    let keyframed: Command = serde_json::from_value(json!({
        "type": "SetSpeedCurve",
        "clip": 4,
        "curve": {"kf": [
            {"t": 0, "v": 1.0},
            {"t": 500, "v": 2.0, "e": "ease_in"},
        ]},
    }))
    .unwrap();
    let json = serde_json::to_value(&keyframed).unwrap();
    assert_eq!(json["curve"]["kf"][1]["v"], 2.0);
    let back: Command = serde_json::from_value(json).unwrap();
    assert_eq!(back, keyframed);
}

#[test]
fn golden_clear_speed_curve_is_null() {
    let cmd = Command::Edit(EditCommand::SetSpeedCurve {
        clip: clip(4),
        curve: None,
    });
    assert_eq!(
        serde_json::to_value(&cmd).unwrap(),
        json!({"type": "SetSpeedCurve", "clip": 4, "curve": null})
    );
}

#[test]
fn golden_look_commands() {
    // Snake_case ids on the wire; zero/default sub-fields elided.
    let mask = Command::Edit(EditCommand::SetClipMask {
        clip: clip(4),
        mask: Some(Mask::new(MaskKind::Circle)),
    });
    assert_eq!(
        serde_json::to_value(&mask).unwrap(),
        json!({"type": "SetClipMask", "clip": 4, "mask": {"kind": "circle"}})
    );

    let chroma = Command::Edit(EditCommand::SetClipChroma {
        clip: clip(4),
        chroma: Some(ChromaKey {
            rgb: [0, 255, 0],
            strength: 0.5.into(),
            shadow: 0.0.into(),
        }),
    });
    assert_eq!(
        serde_json::to_value(&chroma).unwrap(),
        json!({
            "type": "SetClipChroma",
            "clip": 4,
            "chroma": {"rgb": [0, 255, 0], "strength": 0.5},
        })
    );

    let stabilize = Command::Edit(EditCommand::SetClipStabilize {
        clip: clip(4),
        stabilize: Some(StabilizeLevel::MaxSmooth),
    });
    assert_eq!(
        serde_json::to_value(&stabilize).unwrap(),
        json!({"type": "SetClipStabilize", "clip": 4, "stabilize": "max_smooth"})
    );

    let filter = Command::Edit(EditCommand::SetClipFilter {
        clip: clip(4),
        filter: Some(Filter {
            id: "vivid".into(),
            intensity: 0.75.into(),
        }),
    });
    assert_eq!(
        serde_json::to_value(&filter).unwrap(),
        json!({
            "type": "SetClipFilter",
            "clip": 4,
            "filter": {"id": "vivid", "intensity": 0.75},
        })
    );

    let adjust = Command::Edit(EditCommand::SetClipAdjustments {
        clip: clip(4),
        adjust: ColorAdjustments {
            brightness: 0.25.into(),
            ..Default::default()
        },
    });
    assert_eq!(
        serde_json::to_value(&adjust).unwrap(),
        json!({
            "type": "SetClipAdjustments",
            "clip": 4,
            "adjust": {"brightness": 0.25},
        })
    );

    let animation = Command::Edit(EditCommand::SetClipAnimation {
        clip: clip(4),
        slot: AnimationSlot::Combo,
        animation: Some(AnimationRef::new("pulse")),
    });
    assert_eq!(
        serde_json::to_value(&animation).unwrap(),
        json!({
            "type": "SetClipAnimation",
            "clip": 4,
            "slot": "combo",
            "animation": {"id": "pulse"},
        })
    );

    // Clearing is an explicit null (same convention as SetSpeedCurve).
    let clear_role = Command::Edit(EditCommand::SetAudioRole {
        clip: clip(4),
        role: None,
    });
    assert_eq!(
        serde_json::to_value(&clear_role).unwrap(),
        json!({"type": "SetAudioRole", "clip": 4, "role": null})
    );
    let role = Command::Edit(EditCommand::SetAudioRole {
        clip: clip(4),
        role: Some(AudioRole::Sfx),
    });
    assert_eq!(
        serde_json::to_value(&role).unwrap(),
        json!({"type": "SetAudioRole", "clip": 4, "role": "sfx"})
    );
}

// --- outcomes ---------------------------------------------------------------

#[test]
fn edit_outcome_wire_format() {
    let samples: Vec<(EditOutcome, Value)> = vec![
        (
            EditOutcome::Created(clip(7)),
            json!({"type": "Created", "id": 7}),
        ),
        (
            EditOutcome::CreatedTrack(track(3)),
            json!({"type": "CreatedTrack", "id": 3}),
        ),
        (
            EditOutcome::Updated(clip(7)),
            json!({"type": "Updated", "id": 7}),
        ),
        (
            EditOutcome::Removed(clip(7)),
            json!({"type": "Removed", "id": 7}),
        ),
        (
            EditOutcome::RemovedTrack(track(3)),
            json!({"type": "RemovedTrack", "id": 3}),
        ),
        (
            EditOutcome::ShiftedTrack(track(3)),
            json!({"type": "ShiftedTrack", "id": 3}),
        ),
        (
            EditOutcome::UpdatedTrack(track(3)),
            json!({"type": "UpdatedTrack", "id": 3}),
        ),
        (
            EditOutcome::CreatedMarker(MarkerId::from_raw(9)),
            json!({"type": "CreatedMarker", "id": 9}),
        ),
        (
            EditOutcome::UpdatedMarker(MarkerId::from_raw(9)),
            json!({"type": "UpdatedMarker", "id": 9}),
        ),
        (
            EditOutcome::RemovedMarker(MarkerId::from_raw(9)),
            json!({"type": "RemovedMarker", "id": 9}),
        ),
        (EditOutcome::UpdatedCanvas, json!({"type": "UpdatedCanvas"})),
    ];
    for (outcome, expected) in samples {
        assert_eq!(serde_json::to_value(outcome).unwrap(), expected);
        let back: EditOutcome = serde_json::from_value(expected).unwrap();
        assert_eq!(back, outcome);
    }
}
