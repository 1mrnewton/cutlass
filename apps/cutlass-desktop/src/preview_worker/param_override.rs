//! Generic per-param live preview override lane.
//!
//! UI/inspector drags send [`WorkerMsg::ParamOverride`]; the worker stores
//! `(clip, param) → value` on the engine and re-renders. Volume/pan also
//! mirror onto the preview audio mixer so drag is audible without a project
//! republish. Release commits via `SetParamConstant` / `SetParamKeyframe`
//! and clears with [`WorkerMsg::ClearParamOverride`]. No history, revision,
//! or projection.

use super::*;

/// Point the engine's param-override map at `(clip, param)` for the next
/// renders. Volume/pan also update the preview audio mixer. Unparsable ids
/// are dropped (stale projection race).
pub(super) fn apply_param_override(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    value: ParamValue,
    audio: Option<&crate::audio::AudioHandle>,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => {
            engine.set_param_override(id, param, value);
            if let Some(audio) = audio {
                audio.set_param_override(id, param, value);
            }
        }
        None => error!(clip, "param override ignored: unparsable clip id"),
    }
}

/// Drop every live param override for `clip` (and any mirrored audio
/// volume/pan override).
pub(super) fn clear_param_overrides(
    engine: &mut Engine,
    clip: &str,
    audio: Option<&crate::audio::AudioHandle>,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => {
            engine.clear_param_overrides(id);
            if let Some(audio) = audio {
                audio.clear_param_overrides(id);
            }
        }
        None => error!(clip, "clear param override ignored: unparsable clip id"),
    }
}

/// Drop one live param override after that param is committed (and any
/// mirrored audio volume/pan override).
pub(super) fn clear_param_override(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    audio: Option<&crate::audio::AudioHandle>,
) {
    if let Some(id) = parse_raw_id(clip).map(ClipId::from_raw) {
        engine.clear_param_override(id, param);
        if let Some(audio) = audio {
            audio.clear_param_override(id, param);
        }
    }
}

/// Coalesce a burst of [`WorkerMsg::ParamOverride`] messages: latest value
/// per `(clip, param)` wins, then at most one frame build. Mutating messages
/// drained from the queue are dispatched in order; a pending override is
/// applied *before* a drained mutation that might clear it (same rule as
/// [`WorkerMsg::TransformOverride`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn coalesce_param_overrides(
    engine: &mut Engine,
    clipboard: &mut Option<Vec<ClipboardClip>>,
    main_magnet: &mut bool,
    linkage: &mut bool,
    clip: String,
    param: ClipParam,
    value: ParamValue,
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
    // Pending map: latest value per (clip, param). Seeded with the head msg.
    let mut pending: HashMap<(String, ClipParam), ParamValue> = HashMap::new();
    pending.insert((clip, param), value);
    let mut dirty = true;

    while let Ok(next) = req_rx.try_recv() {
        match next {
            WorkerMsg::Frame(latest) => tick = latest,
            WorkerMsg::ParamOverride {
                clip: c,
                param: p,
                value: v,
                tick: at,
            } => {
                pending.insert((c, p), v);
                tick = at;
                dirty = true;
            }
            other => {
                if std::mem::take(&mut dirty) {
                    flush_param_overrides(engine, &pending, Some(&ui.audio));
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

    if dirty {
        flush_param_overrides(engine, &pending, Some(&ui.audio));
    }
    tick
}

fn flush_param_overrides(
    engine: &mut Engine,
    pending: &HashMap<(String, ClipParam), ParamValue>,
    audio: Option<&crate::audio::AudioHandle>,
) {
    for ((clip, param), value) in pending {
        apply_param_override(engine, clip, *param, *value, audio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{
        ChromaKey, CropRect, LookParam, MediaSource, Project, Rational, RationalTime, TimeRange,
        TrackKind,
    };
    use cutlass_render::{ResolveOverrides, resolve, resolve_with};

    fn engine_with_chroma() -> (Engine, ClipId, String, Rational) {
        let r = Rational::FPS_24;
        let mut project = Project::new("param-lane", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/param-lane.mp4",
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
            .set_clip_chroma_key(
                clip,
                Some(ChromaKey {
                    rgb: [0, 255, 0],
                    strength: 0.2.into(),
                    shadow: 0.0.into(),
                }),
            )
            .expect("chroma");
        project
            .set_param_constant(
                clip,
                ClipParam::Crop,
                ParamValue::Rect([0.0, 0.0, 1.0, 1.0]),
            )
            .expect("crop");
        let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        (engine, clip, clip.raw().to_string(), r)
    }

    #[test]
    fn override_wins_over_stored_and_clears() {
        let (mut engine, clip, clip_s, r) = engine_with_chroma();
        let param = ClipParam::Look {
            param: LookParam::ChromaStrength,
        };
        let revision = engine.revision();
        let could_undo = engine.can_undo();

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.9), None);
        assert!(engine.has_live_overrides());
        let scene = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("resolve");
        assert!((scene.layers[0].chroma_key.unwrap().strength - 0.9).abs() < f32::EPSILON);
        assert_eq!(engine.revision(), revision);
        assert_eq!(engine.can_undo(), could_undo);

        clear_param_overrides(&mut engine, &clip_s, None);
        assert!(!engine.has_live_overrides());
        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("plain");
        assert!((plain.layers[0].chroma_key.unwrap().strength - 0.2).abs() < f32::EPSILON);
        let _ = clip;
    }

    #[test]
    fn override_wins_over_keyframed_param() {
        let (mut engine, clip, clip_s, r) = engine_with_chroma();
        let param = ClipParam::Look {
            param: LookParam::ChromaStrength,
        };
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param,
                at: RationalTime::new(0, r),
                value: ParamValue::Scalar(0.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("kf0");
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param,
                at: RationalTime::new(40, r),
                value: ParamValue::Scalar(1.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("kf1");

        let mid = resolve(engine.project(), RationalTime::new(20, r)).expect("mid");
        assert!((mid.layers[0].chroma_key.unwrap().strength - 0.5).abs() < 1e-5);

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.12), None);
        let live = resolve_with(
            engine.project(),
            RationalTime::new(20, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        assert!((live.layers[0].chroma_key.unwrap().strength - 0.12).abs() < f32::EPSILON);
    }

    #[test]
    fn same_param_latest_wins_and_multi_param_applies() {
        let (mut engine, _clip, clip_s, r) = engine_with_chroma();
        let chroma = ClipParam::Look {
            param: LookParam::ChromaStrength,
        };
        apply_param_override(&mut engine, &clip_s, chroma, ParamValue::Scalar(0.3), None);
        apply_param_override(&mut engine, &clip_s, chroma, ParamValue::Scalar(0.7), None);
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Crop,
            ParamValue::Rect([0.25, 0.0, 0.5, 1.0]),
            None,
        );

        assert_eq!(
            engine
                .param_overrides()
                .get(parse_raw_id(&clip_s).map(ClipId::from_raw).unwrap(), chroma),
            Some(ParamValue::Scalar(0.7))
        );

        let scene = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("resolve");
        assert!((scene.layers[0].chroma_key.unwrap().strength - 0.7).abs() < f32::EPSILON);
        assert!((scene.layers[0].uv[0] - 0.25).abs() < 1e-5);
        assert!((scene.layers[0].uv[2] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn handle_emits_param_override_and_clear_messages() {
        let (tx, rx) = unbounded();
        let handle = WorkerHandle { tx };
        let param = ClipParam::Look {
            param: LookParam::ChromaStrength,
        };
        handle.param_override("7".into(), param, ParamValue::Scalar(0.4), 12);
        handle.param_override("7".into(), param, ParamValue::Scalar(0.6), 13);
        handle.clear_param_override("7".into(), 14);

        let WorkerMsg::ParamOverride {
            clip,
            param: p,
            value,
            tick,
        } = rx.try_recv().unwrap()
        else {
            panic!("expected ParamOverride");
        };
        assert_eq!(clip, "7");
        assert_eq!(p, param);
        assert_eq!(value, ParamValue::Scalar(0.4));
        assert_eq!(tick, 12);

        let WorkerMsg::ParamOverride { value, tick, .. } = rx.try_recv().unwrap() else {
            panic!("expected second ParamOverride");
        };
        assert_eq!(value, ParamValue::Scalar(0.6));
        assert_eq!(tick, 13);

        let WorkerMsg::ClearParamOverride { clip, tick } = rx.try_recv().unwrap() else {
            panic!("expected ClearParamOverride");
        };
        assert_eq!(clip, "7");
        assert_eq!(tick, 14);
    }

    #[test]
    fn effect_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};

        let r = Rational::FPS_24;
        let mut project = Project::new("effect-lane", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/effect-lane.mp4",
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
        project.add_effect(clip, "gaussian_blur").expect("add blur");
        let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        let clip_s = clip.raw().to_string();
        let param = ClipParam::Effect {
            effect: 0,
            param: 0,
        };
        let rev_before = engine.revision();

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(12.0), None);
        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(24.0), None);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);

        let live = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        assert!((live.layers[0].effects[0].params[0] - 24.0).abs() < f32::EPSILON);

        clear_param_override(&mut engine, &clip_s, param, None);
        engine
            .apply(Command::Edit(EditCommand::SetEffectParam {
                clip,
                index: 0,
                param: 0,
                value: 24.0,
            }))
            .expect("commit effect");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);

        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        assert!((plain.layers[0].effects[0].params[0] - 24.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lut_intensity_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};
        use cutlass_models::Lut;

        let r = Rational::FPS_24;
        let mut project = Project::new("lut-lane", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/lut-lane.mp4",
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
            .set_clip_lut(
                clip,
                Some(Lut {
                    path: "/tmp/test.cube".into(),
                    intensity: 0.5.into(),
                }),
            )
            .expect("lut");
        let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        let clip_s = clip.raw().to_string();
        let param = ClipParam::Look {
            param: LookParam::LutIntensity,
        };
        let rev_before = engine.revision();

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.9), None);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);

        let live = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        assert!((live.layers[0].lut.as_ref().unwrap().intensity - 0.9).abs() < f32::EPSILON);

        clear_param_override(&mut engine, &clip_s, param, None);
        engine
            .apply(Command::Edit(EditCommand::SetClipLut {
                clip,
                lut: Some(Lut {
                    path: "/tmp/test.cube".into(),
                    intensity: 0.9.into(),
                }),
            }))
            .expect("commit lut");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);

        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        assert!((plain.layers[0].lut.as_ref().unwrap().intensity - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn chroma_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};

        let (mut engine, clip, clip_s, r) = engine_with_chroma();
        let param = ClipParam::Look {
            param: LookParam::ChromaStrength,
        };
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.4), None);
        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.85), None);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);

        let live = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        assert!((live.layers[0].chroma_key.unwrap().strength - 0.85).abs() < f32::EPSILON);

        // Release: clear then one SetParamConstant (set_param_constant_and_publish).
        clear_param_override(&mut engine, &clip_s, param, None);
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param,
                value: ParamValue::Scalar(0.85),
            }))
            .expect("commit chroma");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert!(engine.can_undo());

        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        assert!((plain.layers[0].chroma_key.unwrap().strength - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn crop_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};

        let (mut engine, clip, clip_s, r) = engine_with_chroma();
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        // Drag ticks: session override only.
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Crop,
            ParamValue::Rect([0.1, 0.05, 0.7, 0.8]),
            None,
        );
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Crop,
            ParamValue::Rect([0.2, 0.1, 0.6, 0.7]),
            None,
        );
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);

        let live = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        assert!((live.layers[0].uv[0] - 0.2).abs() < 1e-5);
        assert!((live.layers[0].uv[1] - 0.1).abs() < 1e-5);

        // Release: clear Crop override then one SetClipCrop (mirrors
        // set_clip_crop_and_publish ordering).
        let crop = CropRect {
            x: 0.2,
            y: 0.1,
            w: 0.6,
            h: 0.7,
        };
        clear_param_override(&mut engine, &clip_s, ClipParam::Crop, None);
        engine
            .apply(Command::Edit(EditCommand::SetClipCrop {
                clip,
                crop,
                flip_h: false,
                flip_v: false,
                at: Some(RationalTime::new(0, r)),
            }))
            .expect("commit crop");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert!(engine.can_undo());

        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        assert!((plain.layers[0].uv[0] - 0.2).abs() < 1e-5);
        assert!((plain.layers[0].uv[2] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn volume_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};

        let (mut engine, clip, clip_s, _r) = engine_with_chroma();
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Volume,
            ParamValue::Scalar(0.35),
            None,
        );
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Volume,
            ParamValue::Scalar(0.6),
            None,
        );
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);
        assert_eq!(
            engine.param_overrides().get(clip, ClipParam::Volume),
            Some(ParamValue::Scalar(0.6))
        );

        // Release: clear then one SetParamConstant (SetClipAudio clears
        // Volume the same way before its edit). Mix-time consultation of
        // this map is covered by cutlass-render export_audio tests.
        clear_param_override(&mut engine, &clip_s, ClipParam::Volume, None);
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param: ClipParam::Volume,
                value: ParamValue::Scalar(0.6),
            }))
            .expect("commit volume");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert!(engine.can_undo());
        assert_eq!(
            engine.project().clip(clip).unwrap().volume.constant(),
            Some(0.6)
        );
    }

    #[test]
    fn pan_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};

        let (mut engine, clip, clip_s, _r) = engine_with_chroma();
        let rev_before = engine.revision();

        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Pan,
            ParamValue::Scalar(-0.5),
            None,
        );
        assert_eq!(
            engine.param_overrides().get(clip, ClipParam::Pan),
            Some(ParamValue::Scalar(-0.5))
        );

        clear_param_override(&mut engine, &clip_s, ClipParam::Pan, None);
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param: ClipParam::Pan,
                value: ParamValue::Scalar(-0.5),
            }))
            .expect("commit pan");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert_eq!(
            engine.project().clip(clip).unwrap().pan.constant(),
            Some(-0.5)
        );
    }

    /// Graph-editor drag: move ticks send playhead-resampled overrides;
    /// release clears then writes one SetParamKeyframe (single undo).
    #[test]
    fn graph_keyframe_drag_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};
        use cutlass_models::{Easing, Keyframe, Param};

        let (mut engine, clip, clip_s, r) = engine_with_chroma();
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param: ClipParam::Opacity,
                at: RationalTime::new(0, r),
                value: ParamValue::Scalar(0.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("kf0");
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param: ClipParam::Opacity,
                at: RationalTime::new(40, r),
                value: ParamValue::Scalar(1.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("kf40");
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        // Hypothetical live curve after dragging start value 0→0.4, then 0.6.
        // Playhead at 20 ⇒ samples 0.7 then 0.8.
        let live_a = Param::Keyframed {
            keyframes: vec![
                Keyframe::new(0, 0.4, Easing::Linear),
                Keyframe::new(40, 1.0, Easing::Linear),
            ],
        };
        let live_b = Param::Keyframed {
            keyframes: vec![
                Keyframe::new(0, 0.6, Easing::Linear),
                Keyframe::new(40, 1.0, Easing::Linear),
            ],
        };
        let sample_a = live_a.sample(20);
        let sample_b = live_b.sample(20);
        assert!((sample_a - 0.7).abs() < 1e-5);
        assert!((sample_b - 0.8).abs() < 1e-5);

        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Opacity,
            ParamValue::Scalar(sample_a),
            None,
        );
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Opacity,
            ParamValue::Scalar(sample_b),
            None,
        );
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);
        assert_eq!(
            engine.param_overrides().get(clip, ClipParam::Opacity),
            Some(ParamValue::Scalar(sample_b))
        );

        // Release: clear then one SetParamKeyframe (set_param_keyframe_and_publish).
        clear_param_override(&mut engine, &clip_s, ClipParam::Opacity, None);
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip,
                param: ClipParam::Opacity,
                at: RationalTime::new(0, r),
                value: ParamValue::Scalar(0.6),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("commit opacity kf");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        assert!(engine.can_undo());
        let committed = engine
            .project()
            .clip(clip)
            .unwrap()
            .transform
            .opacity
            .sample(20);
        assert!((committed - 0.8).abs() < 1e-5);
    }

    #[test]
    fn text_spacing_preview_then_commit_clears_override() {
        use cutlass_commands::{Command, EditCommand};
        use cutlass_models::{Generator, TextParam, TextStyle, TrackKind};
        use cutlass_render::LayerSource;

        let r = Rational::FPS_24;
        let mut project = Project::new("spacing-lane", r);
        let track = project.add_track(TrackKind::Text, "T1");
        let clip = project
            .add_generated(
                track,
                Generator::Text {
                    content: "Hi".into(),
                    style: TextStyle::default(),
                },
                TimeRange::at_rate(0, 48, r),
            )
            .expect("text");
        let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        let clip_s = clip.raw().to_string();
        let letter = ClipParam::Text {
            param: TextParam::LetterSpacing,
        };
        let line = ClipParam::Text {
            param: TextParam::LineSpacing,
        };
        let rev_before = engine.revision();
        let could_undo = engine.can_undo();

        // Drag ticks: session overrides only (letter then line, latest wins
        // per param — same coalesce semantics as volume/crop).
        apply_param_override(&mut engine, &clip_s, letter, ParamValue::Scalar(4.0), None);
        apply_param_override(&mut engine, &clip_s, letter, ParamValue::Scalar(12.0), None);
        apply_param_override(&mut engine, &clip_s, line, ParamValue::Scalar(1.8), None);
        assert!(engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        assert_eq!(engine.can_undo(), could_undo);

        let live = resolve_with(
            engine.project(),
            RationalTime::new(0, r),
            ResolveOverrides {
                params: Some(engine.param_overrides()),
                ..ResolveOverrides::default()
            },
        )
        .expect("live");
        let LayerSource::Text { style, .. } = &live.layers[0].source else {
            panic!("expected text layer");
        };
        assert!((style.letter_spacing - 12.0).abs() < 1e-3);
        // line_height = font_size * line_spacing; default size 90 → 162.
        assert!((style.line_height - 90.0 * 1.8).abs() < 1e-2);

        // Release: clear then one SetParamConstant per field.
        clear_param_override(&mut engine, &clip_s, letter, None);
        clear_param_override(&mut engine, &clip_s, line, None);
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param: letter,
                value: ParamValue::Scalar(12.0),
            }))
            .expect("commit letter");
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param: line,
                value: ParamValue::Scalar(1.8),
            }))
            .expect("commit line");
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 2);
        assert!(engine.can_undo());

        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        let LayerSource::Text { style, .. } = &plain.layers[0].source else {
            panic!("expected text layer");
        };
        assert!((style.letter_spacing - 12.0).abs() < 1e-3);
        assert!((style.line_height - 90.0 * 1.8).abs() < 1e-2);
    }
}
