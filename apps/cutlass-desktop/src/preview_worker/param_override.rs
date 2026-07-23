//! Generic per-param live preview override lane.
//!
//! UI/inspector drags send [`WorkerMsg::ParamOverride`]; the worker stores
//! `(clip, param) → value` on the engine and re-renders. Release commits via
//! `SetParamConstant` / `SetParamKeyframe` and clears with
//! [`WorkerMsg::ClearParamOverride`]. No history, revision, or projection.

use super::*;

/// Point the engine's param-override map at `(clip, param)` for the next
/// renders. Unparsable ids are dropped (stale projection race).
pub(super) fn apply_param_override(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    value: ParamValue,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_param_override(id, param, value),
        None => error!(clip, "param override ignored: unparsable clip id"),
    }
}

/// Drop every live param override for `clip` and re-render `tick` from
/// committed state.
pub(super) fn clear_param_overrides(engine: &mut Engine, clip: &str) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.clear_param_overrides(id),
        None => error!(clip, "clear param override ignored: unparsable clip id"),
    }
}

/// Drop one live param override after that param is committed.
pub(super) fn clear_param_override(engine: &mut Engine, clip: &str, param: ClipParam) {
    if let Some(id) = parse_raw_id(clip).map(ClipId::from_raw) {
        engine.clear_param_override(id, param);
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
                    flush_param_overrides(engine, &pending);
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
        flush_param_overrides(engine, &pending);
    }
    tick
}

fn flush_param_overrides(engine: &mut Engine, pending: &HashMap<(String, ClipParam), ParamValue>) {
    for ((clip, param), value) in pending {
        apply_param_override(engine, clip, *param, *value);
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

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.9));
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

        clear_param_overrides(&mut engine, &clip_s);
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

        apply_param_override(&mut engine, &clip_s, param, ParamValue::Scalar(0.12));
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
        apply_param_override(&mut engine, &clip_s, chroma, ParamValue::Scalar(0.3));
        apply_param_override(&mut engine, &clip_s, chroma, ParamValue::Scalar(0.7));
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Crop,
            ParamValue::Rect([0.25, 0.0, 0.5, 1.0]),
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
        );
        apply_param_override(
            &mut engine,
            &clip_s,
            ClipParam::Crop,
            ParamValue::Rect([0.2, 0.1, 0.6, 0.7]),
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
        clear_param_override(&mut engine, &clip_s, ClipParam::Crop);
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
}
