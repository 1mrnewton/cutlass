//! Resolve-time coverage for per-character text animations.

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    AnimationRef, AnimationSlot, Generator, Project, TextStyle as ModelTextStyle, TimeRange,
    TrackKind,
};

use crate::resolve::resolve;
use crate::scene::LayerSource;

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS_24)
}

#[test]
fn typewriter_combo_attaches_text_animation() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Hello".into(),
                style: ModelTextStyle::default(),
            },
            tr(0, 48),
        )
        .unwrap();
    project
        .set_clip_animation(
            clip,
            AnimationSlot::Combo,
            Some(AnimationRef::new("typewriter")),
        )
        .unwrap();

    let scene = resolve(&project, rt(6)).unwrap();
    let LayerSource::Text { animation, .. } = &scene.layers[0].source else {
        panic!("expected text layer");
    };
    let anim = animation.as_ref().expect("per-char animation");
    assert_eq!(anim.id, "typewriter");
    assert_eq!(anim.slot, AnimationSlot::Combo);
    assert!(anim.t > 0.0 && anim.t < 1.0);
    // Whole-layer opacity stays identity — the reveal is per glyph.
    assert!((scene.layers[0].opacity - 1.0).abs() < 1e-4);
}

#[test]
fn char_fade_in_samples_during_entrance_window() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Hi".into(),
                style: ModelTextStyle::default(),
            },
            tr(0, 48),
        )
        .unwrap();
    project
        .set_clip_animation(
            clip,
            AnimationSlot::In,
            Some(AnimationRef::new("char_fade_in")),
        )
        .unwrap();

    let start = resolve(&project, rt(0)).unwrap();
    let LayerSource::Text {
        animation: Some(anim),
        ..
    } = &start.layers[0].source
    else {
        panic!("expected char_fade_in at start");
    };
    assert_eq!(anim.id, "char_fade_in");
    assert!(anim.t < 0.05);

    let mid = resolve(&project, rt(24)).unwrap();
    let LayerSource::Text {
        animation: mid_anim,
        ..
    } = &mid.layers[0].source
    else {
        panic!("expected text");
    };
    // Past the entrance window the per-char sample drops away.
    assert!(mid_anim.is_none());
}
