//! Bespoke live-preview lanes for motion blur and look-animation knobs.
//!
//! These are not [`ClipParam`]-addressable (preview-only; AI wire stays
//! structural `SetClipMotionBlur` / `SetClipAnimation`). Modeled on the
//! styles delta lane: UI sends one field key, the worker merges against
//! committed engine state, installs a session override, and release clears
//! on commit.

use super::*;

/// Point the engine's motion-blur override at `clip` for the next renders.
pub(super) fn apply_motion_blur_override(engine: &mut Engine, clip: &str, blur: MotionBlur) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_motion_blur_override(Some((id, blur))),
        None => error!(clip, "motion blur override ignored: unparsable clip id"),
    }
}

/// Point the engine's animation override at `(clip, slot)` for the next renders.
pub(super) fn apply_animation_override(
    engine: &mut Engine,
    clip: &str,
    slot: AnimationSlot,
    animation: AnimationRef,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_animation_override(Some((id, slot, animation))),
        None => error!(clip, "animation override ignored: unparsable clip id"),
    }
}

/// Merge one motion-blur field delta against the clip's committed settings.
/// Unknown keys / clips return `None` (dropped like style deltas).
pub(super) fn motion_blur_from_preview_delta(
    engine: &Engine,
    clip: &str,
    key: &str,
    value: f32,
) -> Option<MotionBlur> {
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let clip_ref = engine.project().clip(clip_id)?;
    let mut blur = clip_ref.motion_blur;
    // Sliders only show while enabled; keep the toggle on during a drag.
    blur.enabled = true;
    match key {
        "shutter" => {
            blur.shutter_deg = (value / 10.0).round() * 10.0;
            blur.shutter_deg = blur.shutter_deg.clamp(0.0, 360.0);
        }
        "samples" => {
            blur.samples = value.round().clamp(2.0, 16.0) as u32;
        }
        _ => return None,
    }
    blur.validate().ok()?;
    Some(blur)
}

/// Resolve a motion-blur field delta and install the session override.
pub(super) fn apply_motion_blur_preview_delta(
    engine: &mut Engine,
    clip: &str,
    key: &str,
    value: f32,
) {
    let Some(blur) = motion_blur_from_preview_delta(engine, clip, key, value) else {
        error!(clip, key, "motion blur preview delta ignored");
        return;
    };
    apply_motion_blur_override(engine, clip, blur);
}

/// Merge one animation-knob delta against the clip's committed slot.
pub(super) fn animation_from_preview_delta(
    engine: &Engine,
    clip: &str,
    slot: &str,
    key: &str,
    value: f32,
) -> Option<(AnimationSlot, AnimationRef)> {
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let clip_ref = engine.project().clip(clip_id)?;
    let animation_slot = parse_animation_slot(slot)?;
    let mut anim = match animation_slot {
        AnimationSlot::In => clip_ref.animation_in.clone(),
        AnimationSlot::Out => clip_ref.animation_out.clone(),
        AnimationSlot::Combo => clip_ref.animation_combo.clone(),
    }?;
    match key {
        "speed" => anim.speed = value.clamp(0.25, 4.0),
        "intensity" => anim.intensity = value.clamp(0.0, 2.0),
        "stagger" => anim.stagger = value.clamp(0.0, 2.0),
        _ => return None,
    }
    Some((animation_slot, anim))
}

/// Resolve an animation-knob delta and install the session override.
pub(super) fn apply_animation_preview_delta(
    engine: &mut Engine,
    clip: &str,
    slot: &str,
    key: &str,
    value: f32,
) {
    let Some((animation_slot, anim)) =
        animation_from_preview_delta(engine, clip, slot, key, value)
    else {
        error!(clip, slot, key, "animation preview delta ignored");
        return;
    };
    apply_animation_override(engine, clip, animation_slot, anim);
}

/// Coalesce a burst of motion-blur preview deltas: latest wins per clip.
#[allow(clippy::too_many_arguments)]
pub(super) fn coalesce_motion_blur_deltas(
    engine: &mut Engine,
    clipboard: &mut Option<Vec<ClipboardClip>>,
    main_magnet: &mut bool,
    linkage: &mut bool,
    mut clip: String,
    mut key: String,
    mut value: f32,
    mut tick: i64,
    req_rx: &Receiver<WorkerMsg>,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    fit: &FrameFit,
    cache: &FrameCache,
    sprite_mode: &Cell<bool>,
    export_state: &ExportJobState,
    ui: &UiSink,
) -> i64 {
    let mut pending = true;
    while let Ok(next) = req_rx.try_recv() {
        match next {
            WorkerMsg::Frame(latest) => tick = latest,
            WorkerMsg::PreviewMotionBlurDelta {
                clip: c,
                key: k,
                value: v,
                tick: at,
            } => {
                clip = c;
                key = k;
                value = v;
                tick = at;
                pending = true;
            }
            other => {
                if std::mem::take(&mut pending) {
                    apply_motion_blur_preview_delta(engine, &clip, &key, value);
                }
                dispatch(
                    engine,
                    clipboard,
                    main_magnet,
                    linkage,
                    other,
                    tl_rate,
                    preview_weak,
                    fit,
                    cache,
                    sprite_mode,
                    export_state,
                    ui,
                );
            }
        }
    }
    if pending {
        apply_motion_blur_preview_delta(engine, &clip, &key, value);
    }
    tick
}

/// Coalesce a burst of animation-knob preview deltas: latest wins per clip.
#[allow(clippy::too_many_arguments)]
pub(super) fn coalesce_animation_deltas(
    engine: &mut Engine,
    clipboard: &mut Option<Vec<ClipboardClip>>,
    main_magnet: &mut bool,
    linkage: &mut bool,
    mut clip: String,
    mut slot: String,
    mut key: String,
    mut value: f32,
    mut tick: i64,
    req_rx: &Receiver<WorkerMsg>,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    fit: &FrameFit,
    cache: &FrameCache,
    sprite_mode: &Cell<bool>,
    export_state: &ExportJobState,
    ui: &UiSink,
) -> i64 {
    let mut pending = true;
    while let Ok(next) = req_rx.try_recv() {
        match next {
            WorkerMsg::Frame(latest) => tick = latest,
            WorkerMsg::PreviewClipAnimationDelta {
                clip: c,
                slot: s,
                key: k,
                value: v,
                tick: at,
            } => {
                clip = c;
                slot = s;
                key = k;
                value = v;
                tick = at;
                pending = true;
            }
            other => {
                if std::mem::take(&mut pending) {
                    apply_animation_preview_delta(engine, &clip, &slot, &key, value);
                }
                dispatch(
                    engine,
                    clipboard,
                    main_magnet,
                    linkage,
                    other,
                    tl_rate,
                    preview_weak,
                    fit,
                    cache,
                    sprite_mode,
                    export_state,
                    ui,
                );
            }
        }
    }
    if pending {
        apply_animation_preview_delta(engine, &clip, &slot, &key, value);
    }
    tick
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_commands::{Command, EditCommand};
    use cutlass_models::{
        AnimationRef, AnimationSlot, MediaSource, MotionBlur, Project, Rational, RationalTime,
        TimeRange, TrackKind,
    };
    use cutlass_render::{ResolveOverrides, attach_motion_blur_passes, resolve, resolve_with};

    fn engine_with_blur_and_anim() -> (Engine, ClipId, String, Rational) {
        let r = Rational::FPS_24;
        let mut project = Project::new("look-preview", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/look-preview.mp4",
            1920,
            1080,
            r,
            1000,
            true,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip = project
            .add_clip(
                track,
                media,
                TimeRange::at_rate(0, 48, r),
                RationalTime::new(0, r),
            )
            .expect("clip");
        project
            .set_motion_blur(
                clip,
                MotionBlur {
                    enabled: true,
                    shutter_deg: 180.0,
                    samples: 8,
                },
            )
            .expect("blur");
        // Animate position so motion-blur supersampling has something to sample.
        project
            .set_param_keyframe(
                clip,
                ClipParam::Position,
                RationalTime::new(0, r),
                ParamValue::Vec2([-0.2, 0.0]),
                Easing::Linear,
                None,
            )
            .expect("kf0");
        project
            .set_param_keyframe(
                clip,
                ClipParam::Position,
                RationalTime::new(40, r),
                ParamValue::Vec2([0.2, 0.0]),
                Easing::Linear,
                None,
            )
            .expect("kf1");
        project
            .set_clip_animation(
                clip,
                AnimationSlot::In,
                Some(AnimationRef {
                    id: "fade_in".into(),
                    speed: 1.0,
                    intensity: 1.0,
                    stagger: 1.0,
                }),
            )
            .expect("anim");
        let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        (engine, clip, clip.raw().to_string(), r)
    }

    #[test]
    fn motion_blur_preview_then_commit_clears_override() {
        let (mut engine, clip, clip_s, r) = engine_with_blur_and_anim();
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        apply_motion_blur_preview_delta(&mut engine, &clip_s, "samples", 4.0);
        apply_motion_blur_preview_delta(&mut engine, &clip_s, "samples", 12.0);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);

        let live_blur = MotionBlur {
            enabled: true,
            shutter_deg: 180.0,
            samples: 12,
        };
        let mut scene = resolve_with(
            engine.project(),
            RationalTime::new(20, r),
            ResolveOverrides {
                motion_blur: Some((clip, live_blur)),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        attach_motion_blur_passes(engine.project(), &mut scene, Some((clip, live_blur)));
        let layer = scene.layers.iter().find(|l| l.clip == Some(clip)).unwrap();
        assert_eq!(layer.blur_passes.len(), 12);

        // Release: clear then one SetClipMotionBlur.
        engine.set_motion_blur_override(None);
        engine
            .apply(Command::Edit(EditCommand::SetClipMotionBlur {
                clip,
                motion_blur: MotionBlur {
                    enabled: true,
                    shutter_deg: 180.0,
                    samples: 12,
                },
            }))
            .expect("commit");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert!(engine.can_undo());

        let mut plain = resolve(engine.project(), RationalTime::new(20, r)).expect("committed");
        attach_motion_blur_passes(engine.project(), &mut plain, None);
        let layer = plain.layers.iter().find(|l| l.clip == Some(clip)).unwrap();
        assert_eq!(layer.blur_passes.len(), 12);
    }

    #[test]
    fn motion_blur_delta_coalesces_latest_wins() {
        let (engine, _clip, clip_s, _r) = engine_with_blur_and_anim();
        let early = motion_blur_from_preview_delta(&engine, &clip_s, "shutter", 90.0).unwrap();
        let late = motion_blur_from_preview_delta(&engine, &clip_s, "shutter", 270.0).unwrap();
        assert!((early.shutter_deg - 90.0).abs() < f32::EPSILON);
        assert!((late.shutter_deg - 270.0).abs() < f32::EPSILON);
        assert_eq!(late.samples, 8, "unrelated field kept from committed");
    }

    #[test]
    fn animation_preview_then_commit_clears_override() {
        let (mut engine, clip, clip_s, r) = engine_with_blur_and_anim();
        let rev_before = engine.revision();

        apply_animation_preview_delta(&mut engine, &clip_s, "in", "intensity", 0.5);
        apply_animation_preview_delta(&mut engine, &clip_s, "in", "intensity", 1.5);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);

        let anim = AnimationRef {
            id: "fade_in".into(),
            speed: 1.0,
            intensity: 1.5,
            stagger: 1.0,
        };
        // Mid-entrance: higher intensity pulls opacity further from 1.0.
        let mid = RationalTime::new(6, r);
        let live = resolve_with(
            engine.project(),
            mid,
            ResolveOverrides {
                animation: Some((clip, AnimationSlot::In, &anim)),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        let plain = resolve(engine.project(), mid).expect("plain");
        assert!(
            live.layers[0].opacity < plain.layers[0].opacity,
            "intensity 1.5 should dim fade_in mid-window more than intensity 1.0"
        );

        engine.set_animation_override(None);
        engine
            .apply(Command::Edit(EditCommand::SetClipAnimation {
                clip,
                slot: AnimationSlot::In,
                animation: Some(anim.clone()),
            }))
            .expect("commit");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);

        let committed = engine.project().clip(clip).unwrap();
        assert!((committed.animation_in.as_ref().unwrap().intensity - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn animation_delta_coalesces_latest_wins() {
        let (engine, _clip, clip_s, _r) = engine_with_blur_and_anim();
        let (_, early) =
            animation_from_preview_delta(&engine, &clip_s, "in", "speed", 0.5).unwrap();
        let (_, late) = animation_from_preview_delta(&engine, &clip_s, "in", "speed", 2.0).unwrap();
        assert!((early.speed - 0.5).abs() < f32::EPSILON);
        assert!((late.speed - 2.0).abs() < f32::EPSILON);
        assert!((late.intensity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn handle_emits_look_preview_messages() {
        let (tx, rx) = unbounded();
        let handle = WorkerHandle { tx };
        handle.preview_motion_blur_delta("7".into(), "shutter".into(), 270.0, 12);
        handle.clear_motion_blur_override(13);
        handle.preview_clip_animation_delta("7".into(), "in".into(), "speed".into(), 2.0, 14);
        handle.clear_animation_override(15);

        let WorkerMsg::PreviewMotionBlurDelta {
            clip,
            key,
            value,
            tick,
        } = rx.try_recv().unwrap()
        else {
            panic!("expected PreviewMotionBlurDelta");
        };
        assert_eq!(clip, "7");
        assert_eq!(key, "shutter");
        assert!((value - 270.0).abs() < f32::EPSILON);
        assert_eq!(tick, 12);

        let WorkerMsg::ClearMotionBlurOverride { tick } = rx.try_recv().unwrap() else {
            panic!("expected ClearMotionBlurOverride");
        };
        assert_eq!(tick, 13);

        let WorkerMsg::PreviewClipAnimationDelta {
            clip,
            slot,
            key,
            value,
            tick,
        } = rx.try_recv().unwrap()
        else {
            panic!("expected PreviewClipAnimationDelta");
        };
        assert_eq!(clip, "7");
        assert_eq!(slot, "in");
        assert_eq!(key, "speed");
        assert!((value - 2.0).abs() < f32::EPSILON);
        assert_eq!(tick, 14);

        let WorkerMsg::ClearAnimationOverride { tick } = rx.try_recv().unwrap() else {
            panic!("expected ClearAnimationOverride");
        };
        assert_eq!(tick, 15);
    }
}
