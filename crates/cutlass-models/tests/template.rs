//! End-to-end tests for the CapCut-style template system, driving only the
//! public `cutlass_models` surface (authoring -> apply -> render-ready project).

mod common;

use common::*;
use cutlass_models::{
    ClipSource, ClipTransform, Generator, MediaId, MediaSource, ModelError, Pick, Project,
    Rational, Replaceable, SlotMedia, Template, TemplateCategory, TemplateMeta, TimeRange,
    TrackKind, speed_preset,
};

const MS_RATE: Rational = Rational::new(1000, 1);

/// IDs of the clips an authored template exposes, so assertions can target
/// exact slots.
struct Intro {
    template: Template,
    slot0: cutlass_models::ClipId,
    slot1: cutlass_models::ClipId,
    image_slot: cutlass_models::ClipId,
    title: cutlass_models::ClipId,
    music: cutlass_models::ClipId,
    sample_video: MediaId,
}

/// A finished 3-slot intro: two 1s video slots and one 1s image slot on V1, an
/// editable title on T1, and a swappable music clip on A1. A non-identity
/// transform rides slot 0 and an effect rides slot 1, so apply can be checked
/// to preserve the locked look.
fn intro_template() -> Intro {
    let mut p = Project::new("Trendy Intro", FPS_24);
    let sample_video = p.add_media(sample_media(FPS_24, 240));
    let sample_image = p.add_media(MediaSource::image("logo.png", 800, 600));
    let sample_music = p.add_media(MediaSource::new("beat.mp3", 0, 0, MS_RATE, 30_000, true));

    let v = p.add_track(TrackKind::Video, "V1");
    let t = p.add_track(TrackKind::Text, "T1");
    let a = p.add_track(TrackKind::Audio, "A1");

    let slot0 = p.add_clip(v, sample_video, tr(0, 24), rt(0)).unwrap();
    let slot1 = p.add_clip(v, sample_video, tr(0, 24), rt(24)).unwrap();
    let image_slot = p
        .add_clip(
            v,
            sample_image,
            TimeRange::at_rate(0, 1000, MS_RATE),
            rt(48),
        )
        .unwrap();
    let title = p
        .add_generated(t, Generator::text("Your Name"), tr(0, 72))
        .unwrap();
    let music = p
        .add_clip(
            a,
            sample_music,
            TimeRange::at_rate(0, 30_000, MS_RATE),
            rt(0),
        )
        .unwrap();

    // A locked look on the slots: a transform on slot 0, an effect on slot 1.
    p.set_transform(
        slot0,
        ClipTransform {
            position: [0.1, 0.2],
            anchor_point: [0.5, 0.5],
            scale: 1.5.into(),
            rotation: 15.0,
            opacity: 0.8,
        },
        None,
    )
    .unwrap();
    let effect_id = cutlass_models::effect_catalog()[0].id;
    p.add_effect(slot1, effect_id).unwrap();

    // Mark the fillable surface (slots authored out of order on purpose).
    p.set_replaceable(slot1, Some(Replaceable::new(1))).unwrap();
    p.set_replaceable(slot0, Some(Replaceable::new(0))).unwrap();
    p.set_replaceable(
        image_slot,
        Some(Replaceable::new(2).with_accepts(SlotMedia::ImageOnly)),
    )
    .unwrap();
    p.set_replaceable(
        music,
        Some(Replaceable::new(0).with_accepts(SlotMedia::AudioOnly)),
    )
    .unwrap();
    p.set_text_editable(title, true).unwrap();

    let meta = TemplateMeta::new("Trendy Intro")
        .with_author("cutlass")
        .with_category(TemplateCategory::Trending)
        .with_tags(["intro".to_string(), "fast".to_string()]);

    Intro {
        template: Template::from_project(p, meta),
        slot0,
        slot1,
        image_slot,
        title,
        music,
        sample_video,
    }
}

fn video_pick(name: &str, frames: i64) -> Pick {
    Pick::new(MediaSource::new(name, 1920, 1080, FPS_24, frames, true))
}

fn image_pick(name: &str) -> Pick {
    Pick::new(MediaSource::image(name, 1024, 768))
}

#[test]
fn derived_surface_lists_visual_slots_music_and_texts() {
    let intro = intro_template();
    let t = &intro.template;

    // Visual slots only, in ascending replaceable order; the music slot is
    // excluded from the ordered fill sequence.
    let slots: Vec<_> = t.slots().iter().map(|c| c.id).collect();
    assert_eq!(slots, vec![intro.slot0, intro.slot1, intro.image_slot]);
    assert_eq!(t.slot_count(), 3);

    assert_eq!(t.music().map(|c| c.id), Some(intro.music));
    let texts: Vec<_> = t.editable_texts().iter().map(|c| c.id).collect();
    assert_eq!(texts, vec![intro.title]);
}

#[test]
fn apply_fills_in_order_and_preserves_the_locked_look() {
    let intro = intro_template();
    let t = &intro.template;

    let out = t
        .apply(&[
            video_pick("a.mp4", 120),
            video_pick("b.mp4", 90),
            image_pick("c.png"),
        ])
        .unwrap();

    // Output is a fresh, distinct project (not aliasing the template's).
    assert_ne!(out.id, t.project().id);

    // Each slot now points at its pick, windowed to the locked duration, and
    // the timeline placement / transform / effects are untouched.
    let cases = [(intro.slot0, "a.mp4"), (intro.slot1, "b.mp4")];
    for (slot, name) in cases {
        let filled = out.clip(slot).unwrap();
        let authored = t.project().clip(slot).unwrap();
        assert_eq!(filled.timeline, authored.timeline, "duration stays locked");
        assert_eq!(filled.transform, authored.transform, "transform preserved");
        assert_eq!(filled.effects, authored.effects, "effects preserved");
        match &filled.content {
            ClipSource::Media { media, source } => {
                assert_eq!(source.duration.value, 24, "1s slot at 24fps");
                assert_eq!(out.media(*media).unwrap().path().to_str().unwrap(), name);
            }
            other => panic!("slot not filled with media: {other:?}"),
        }
    }
    // slot1 actually carried an effect, proving "preserved" is non-trivial.
    assert!(!out.clip(intro.slot1).unwrap().effects.is_empty());

    // The image slot drew from a still (windowed to the locked 1s).
    match &out.clip(intro.image_slot).unwrap().content {
        ClipSource::Media { media, source } => {
            assert!(out.media(*media).unwrap().is_image);
            assert_eq!(source.duration.value, 1000);
        }
        other => panic!("image slot not filled: {other:?}"),
    }
}

#[test]
fn unfilled_slots_keep_sample_media_for_preview() {
    let intro = intro_template();
    let t = &intro.template;

    // Only the first slot is picked; the rest preview the author's samples.
    let out = t.apply(&[video_pick("a.mp4", 120)]).unwrap();

    for slot in [intro.slot1, intro.image_slot] {
        match (
            &out.clip(slot).unwrap().content,
            &t.project().clip(slot).unwrap().content,
        ) {
            (ClipSource::Media { media: got, .. }, ClipSource::Media { media: sample, .. }) => {
                assert_eq!(got, sample, "unfilled slot keeps its sample media");
            }
            _ => unreachable!(),
        }
    }
    // And applying with no picks at all is a faithful preview of the template.
    let preview = t.apply(&[]).unwrap();
    match &preview.clip(intro.slot0).unwrap().content {
        ClipSource::Media { media, .. } => assert_eq!(*media, intro.sample_video),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn apply_rejects_media_type_mismatch() {
    let intro = intro_template();
    // The image slot is third; supply a video there.
    let err = intro
        .template
        .apply(&[
            video_pick("a.mp4", 120),
            video_pick("b.mp4", 120),
            video_pick("not-an-image.mp4", 120),
        ])
        .unwrap_err();
    assert!(
        matches!(err, ModelError::SlotMediaMismatch { slot, .. } if slot == intro.image_slot),
        "got {err:?}"
    );
}

#[test]
fn apply_rejects_extra_picks() {
    let intro = intro_template();
    // Four picks for a 3-slot template: refused, not silently truncated —
    // the AI agent fills templates too, and this catches its off-by-ones.
    let err = intro
        .template
        .apply(&[
            video_pick("a.mp4", 120),
            video_pick("b.mp4", 120),
            image_pick("c.png"),
            video_pick("extra.mp4", 120),
        ])
        .unwrap_err();
    assert!(
        matches!(err, ModelError::TooManyPicks { given: 4, slots: 3 }),
        "got {err:?}"
    );
}

#[test]
fn apply_refuses_reversed_and_speed_ramped_slots() {
    // Sizing a retimed slot's source window needs the curve integral; v1
    // refuses the pick instead of mis-windowing it.
    let mut p = Project::new("Reversed", FPS_24);
    let m = p.add_media(sample_media(FPS_24, 240));
    let v = p.add_track(TrackKind::Video, "V1");
    let slot = p.add_clip(v, m, tr(0, 24), rt(0)).unwrap();
    p.set_clip_speed(slot, Rational::new(1, 1), true).unwrap();
    p.set_replaceable(slot, Some(Replaceable::new(0))).unwrap();
    let t = Template::from_project(p, TemplateMeta::new("Reversed"));
    let err = t.apply(&[video_pick("a.mp4", 120)]).unwrap_err();
    assert!(
        matches!(err, ModelError::SlotRetimeUnsupported { slot: s } if s == slot),
        "got {err:?}"
    );

    let mut p = Project::new("Ramped", FPS_24);
    let m = p.add_media(sample_media(FPS_24, 240));
    let v = p.add_track(TrackKind::Video, "V1");
    let slot = p.add_clip(v, m, tr(0, 24), rt(0)).unwrap();
    p.set_clip_speed_curve(slot, Some(speed_preset("montage").unwrap()))
        .unwrap();
    p.set_replaceable(slot, Some(Replaceable::new(0))).unwrap();
    let t = Template::from_project(p, TemplateMeta::new("Ramped"));
    let err = t.apply(&[video_pick("a.mp4", 120)]).unwrap_err();
    assert!(
        matches!(err, ModelError::SlotRetimeUnsupported { .. }),
        "got {err:?}"
    );
}

#[test]
fn cross_rate_fill_rounds_source_window_up() {
    let mut p = Project::new("OddLen", FPS_24);
    let m = p.add_media(sample_media(FPS_24, 240));
    let v = p.add_track(TrackKind::Video, "V1");
    // 25 ticks at 24fps has no exact 30fps equivalent (31.25 ticks).
    let slot = p.add_clip(v, m, tr(0, 25), rt(0)).unwrap();
    p.set_replaceable(slot, Some(Replaceable::new(0))).unwrap();
    let t = Template::from_project(p, TemplateMeta::new("OddLen"));

    let r30 = Rational::new(30, 1);
    // Rounding to nearest (31) would under-cover the slot; the window must
    // round up to 32 so the last timeline frame still has source to read.
    let out = t
        .apply(&[Pick::new(MediaSource::new(
            "m.mp4", 1920, 1080, r30, 32, true,
        ))])
        .unwrap();
    match &out.clip(slot).unwrap().content {
        ClipSource::Media { source, .. } => assert_eq!(source.duration.value, 32),
        other => panic!("unexpected: {other:?}"),
    }
    // Exactly-31-frame media is refused rather than silently short.
    let err = t
        .apply(&[Pick::new(MediaSource::new(
            "short.mp4",
            1920,
            1080,
            r30,
            31,
            true,
        ))])
        .unwrap_err();
    assert!(
        matches!(err, ModelError::SlotDurationUnmet { .. }),
        "got {err:?}"
    );
}

#[test]
fn set_replaceable_validates_at_mark_time() {
    let mut p = Project::new("Marks", FPS_24);
    let m = p.add_media(sample_media(FPS_24, 240));
    let music_m = p.add_media(MediaSource::new("beat.mp3", 0, 0, MS_RATE, 30_000, true));
    let v = p.add_track(TrackKind::Video, "V1");
    let t = p.add_track(TrackKind::Text, "T1");
    let a = p.add_track(TrackKind::Audio, "A1");
    let video_clip = p.add_clip(v, m, tr(0, 24), rt(0)).unwrap();
    let title = p
        .add_generated(t, Generator::text("hi"), tr(0, 24))
        .unwrap();
    let audio_clip = p
        .add_clip(a, music_m, TimeRange::at_rate(0, 1000, MS_RATE), rt(0))
        .unwrap();

    // Generated content can never be refilled: refused when marked, not at
    // apply time with a track error.
    assert!(matches!(
        p.set_replaceable(title, Some(Replaceable::new(0))),
        Err(ModelError::InvalidParam(_))
    ));
    // The accepts restriction must match the lane.
    assert!(matches!(
        p.set_replaceable(
            video_clip,
            Some(Replaceable::new(0).with_accepts(SlotMedia::AudioOnly))
        ),
        Err(ModelError::InvalidParam(_))
    ));
    assert!(matches!(
        p.set_replaceable(audio_clip, Some(Replaceable::new(0))),
        Err(ModelError::InvalidParam(_))
    ));

    // Valid markings work; unmarking is always allowed.
    p.set_replaceable(video_clip, Some(Replaceable::new(0)))
        .unwrap();
    p.set_replaceable(
        audio_clip,
        Some(Replaceable::new(1).with_accepts(SlotMedia::AudioOnly)),
    )
    .unwrap();
    p.set_replaceable(video_clip, None).unwrap();
}

#[test]
fn apply_rejects_media_too_short_for_locked_slot() {
    let intro = intro_template();
    // slot0 is a locked 1s (24 frame) window; a 10-frame clip can't cover it.
    let err = intro
        .template
        .apply(&[video_pick("tiny.mp4", 10)])
        .unwrap_err();
    assert!(
        matches!(err, ModelError::SlotDurationUnmet { slot } if slot == intro.slot0),
        "got {err:?}"
    );
}

#[test]
fn pick_in_point_windows_the_chosen_source() {
    let intro = intro_template();
    // Start slot 0 from frame 50 of a long clip.
    let out = intro
        .template
        .apply(&[Pick::at(
            MediaSource::new("long.mp4", 1920, 1080, FPS_24, 300, true),
            rt(50),
        )])
        .unwrap();
    match &out.clip(intro.slot0).unwrap().content {
        ClipSource::Media { source, .. } => {
            assert_eq!(source.start.value, 50);
            assert_eq!(source.duration.value, 24);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn music_swaps_independently_of_visual_slots() {
    let intro = intro_template();
    let mut out = intro.template.apply(&[]).unwrap();

    // The music clip is a real, swappable clip in the applied project.
    let new_track_music = out.add_media(MediaSource::new(
        "new-song.mp3",
        0,
        0,
        MS_RATE,
        40_000,
        true,
    ));
    out.set_clip_media(
        intro.music,
        new_track_music,
        TimeRange::at_rate(0, 30_000, MS_RATE),
    )
    .unwrap();

    match &out.clip(intro.music).unwrap().content {
        ClipSource::Media { media, .. } => assert_eq!(*media, new_track_music),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn template_file_roundtrips_with_markers_and_meta() {
    let intro = intro_template();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("intro.cutlasst");

    intro.template.save_to_file(&path).unwrap();
    let loaded = Template::load_from_file(&path).unwrap();

    assert_eq!(loaded.meta.name, "Trendy Intro");
    assert_eq!(loaded.meta.author.as_deref(), Some("cutlass"));
    assert_eq!(loaded.meta.category, TemplateCategory::Trending);
    assert_eq!(loaded.slot_count(), 3);
    assert_eq!(loaded.music().map(|c| c.id), Some(intro.music));
    assert_eq!(loaded.editable_texts().len(), 1);

    // The reloaded template still applies.
    let out = loaded
        .apply(&[
            video_pick("a.mp4", 120),
            video_pick("b.mp4", 120),
            image_pick("c.png"),
        ])
        .unwrap();
    assert!(matches!(
        &out.clip(intro.slot0).unwrap().content,
        ClipSource::Media { .. }
    ));
}

#[test]
fn loading_a_plain_project_as_template_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.cutlasst");
    Project::new("plain", FPS_24).save_to_file(&path).unwrap();

    assert!(Template::load_from_file(&path).is_err());
}
