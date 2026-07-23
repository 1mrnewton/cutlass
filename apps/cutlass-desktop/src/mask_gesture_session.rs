//! Mask gizmo / inspector ParamOverride session pairing.
//!
//! Canvas presses and inspector slider previews share the same lane:
//! begin once, coalesce overrides while dragging, then either one commit
//! (clear + SetParamConstant/Keyframe) or Escape (clear, no commit).

use cutlass_models::{ClipParam, LookParam, ParamValue};

use crate::preview_mask_gizmo::{
    HANDLE_BODY, HANDLE_CENTER, HANDLE_FEATHER, HANDLE_NONE, HANDLE_ROTATION, HANDLE_ROUNDNESS,
    HANDLE_SIZE_X, HANDLE_SIZE_Y,
};

/// Which look param a mask handle edits (size handles share MaskSize).
#[cfg_attr(not(test), allow(dead_code))]
pub fn handle_param(handle: i32) -> Option<ClipParam> {
    match handle {
        HANDLE_CENTER | HANDLE_BODY => Some(ClipParam::Look {
            param: LookParam::MaskCenter,
        }),
        HANDLE_SIZE_X | HANDLE_SIZE_Y => Some(ClipParam::Look {
            param: LookParam::MaskSize,
        }),
        HANDLE_ROTATION => Some(ClipParam::Look {
            param: LookParam::MaskRotation,
        }),
        HANDLE_FEATHER => Some(ClipParam::Look {
            param: LookParam::MaskFeather,
        }),
        HANDLE_ROUNDNESS => Some(ClipParam::Look {
            param: LookParam::MaskRoundness,
        }),
        _ => None,
    }
}

/// Map a [`ClipParam`] to a representative mask-handle id (for session begin).
pub fn clip_param_to_handle(param: ClipParam) -> Option<i32> {
    match param {
        ClipParam::Look {
            param: LookParam::MaskCenter,
        } => Some(HANDLE_CENTER),
        ClipParam::Look {
            param: LookParam::MaskSize,
        } => Some(HANDLE_SIZE_X),
        ClipParam::Look {
            param: LookParam::MaskRotation,
        } => Some(HANDLE_ROTATION),
        ClipParam::Look {
            param: LookParam::MaskFeather,
        } => Some(HANDLE_FEATHER),
        ClipParam::Look {
            param: LookParam::MaskRoundness,
        } => Some(HANDLE_ROUNDNESS),
        _ => None,
    }
}

/// Pack live mask fields into the ParamValue for `handle`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn handle_value(
    handle: i32,
    center: [f32; 2],
    size: [f32; 2],
    rotation: f32,
    feather: f32,
    roundness: f32,
) -> Option<ParamValue> {
    match handle {
        HANDLE_CENTER | HANDLE_BODY => Some(ParamValue::Vec2(center)),
        HANDLE_SIZE_X | HANDLE_SIZE_Y => {
            Some(ParamValue::Vec2([size[0].max(0.05), size[1].max(0.05)]))
        }
        HANDLE_ROTATION => Some(ParamValue::Scalar(rotation)),
        HANDLE_FEATHER => Some(ParamValue::Scalar(feather.clamp(0.0, 1.0))),
        HANDLE_ROUNDNESS => Some(ParamValue::Scalar(roundness.clamp(0.0, 1.0))),
        _ => None,
    }
}

/// Fresh [`Mask::new`] defaults for a double-click handle reset.
#[cfg_attr(not(test), allow(dead_code))]
pub fn default_value_for_handle(handle: i32) -> Option<ParamValue> {
    match handle {
        HANDLE_CENTER | HANDLE_BODY => Some(ParamValue::Vec2([0.0, 0.0])),
        HANDLE_SIZE_X | HANDLE_SIZE_Y => Some(ParamValue::Vec2([1.0, 1.0])),
        HANDLE_ROTATION | HANDLE_FEATHER | HANDLE_ROUNDNESS => Some(ParamValue::Scalar(0.0)),
        _ => None,
    }
}

/// Whether a mask-handle drag is paired with the worker override lane.
#[derive(Debug, Default, Clone)]
pub struct MaskGestureSession {
    active_clip: Option<String>,
    handle: i32,
}

impl MaskGestureSession {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn is_active(&self) -> bool {
        self.active_clip.is_some() && self.handle != HANDLE_NONE
    }

    #[cfg(test)]
    pub fn handle(&self) -> i32 {
        self.handle
    }

    /// Explicit begin (canvas press on a handle). Returns `true` when this
    /// is a new session (caller may seed live UI state).
    pub fn begin(&mut self, clip: &str, handle: i32) -> bool {
        if handle == HANDLE_NONE {
            return false;
        }
        if self.active_clip.as_deref() == Some(clip) && self.handle == handle {
            return false;
        }
        self.active_clip = Some(clip.to_string());
        self.handle = handle;
        true
    }

    /// First override of a slider drag auto-begins.
    pub fn preview(&mut self, clip: &str, handle: i32) -> bool {
        self.begin(clip, handle)
    }

    pub fn end(&mut self) {
        self.active_clip = None;
        self.handle = HANDLE_NONE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_commands::{Command, EditCommand};
    use cutlass_engine::{Engine, EngineConfig};
    use cutlass_models::{
        Mask, MaskKind, MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind,
    };
    use cutlass_render::{ResolveOverrides, resolve, resolve_with};

    fn engine_with_mask() -> (Engine, cutlass_models::ClipId, String, Rational) {
        let r = Rational::FPS_24;
        let mut project = Project::new("mask-gesture", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/mask-gesture.mp4",
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
            .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
            .expect("mask");
        let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        (engine, clip, clip.raw().to_string(), r)
    }

    #[test]
    fn begin_is_idempotent_for_same_handle() {
        let mut session = MaskGestureSession::new();
        assert!(session.begin("42", HANDLE_CENTER));
        assert!(!session.begin("42", HANDLE_CENTER));
        assert!(session.is_active());
        assert_eq!(session.handle(), HANDLE_CENTER);
    }

    #[test]
    fn handle_switch_rebegins() {
        let mut session = MaskGestureSession::new();
        assert!(session.begin("42", HANDLE_CENTER));
        assert!(session.begin("42", HANDLE_FEATHER));
        assert_eq!(session.handle(), HANDLE_FEATHER);
    }

    #[test]
    fn drag_overrides_then_commit_once() {
        let (mut engine, clip, clip_s, r) = engine_with_mask();
        let param = handle_param(HANDLE_FEATHER).unwrap();
        let rev_before = engine.revision();
        let mut session = MaskGestureSession::new();

        assert!(session.begin(&clip_s, HANDLE_FEATHER));
        engine.set_param_override(clip, param, ParamValue::Scalar(0.2));
        engine.set_param_override(clip, param, ParamValue::Scalar(0.55));
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
        assert!((live.layers[0].mask.unwrap().feather - 0.55).abs() < f32::EPSILON);

        // Release: clear then one commit.
        engine.clear_param_override(clip, param);
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param,
                value: ParamValue::Scalar(0.55),
            }))
            .expect("commit");
        session.end();

        assert!(!session.is_active());
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before + 1);
        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
        assert!((plain.layers[0].mask.unwrap().feather - 0.55).abs() < f32::EPSILON);
        let _ = clip_s;
    }

    #[test]
    fn escape_clears_override_without_commit() {
        let (mut engine, clip, clip_s, r) = engine_with_mask();
        let param = handle_param(HANDLE_CENTER).unwrap();
        let rev_before = engine.revision();
        let mut session = MaskGestureSession::new();

        assert!(session.begin(&clip_s, HANDLE_CENTER));
        engine.set_param_override(clip, param, ParamValue::Vec2([0.3, -0.2]));
        assert!(engine.has_live_overrides());

        // Escape: clear only.
        engine.clear_param_override(clip, param);
        session.end();

        assert!(!session.is_active());
        assert!(!engine.has_live_overrides());
        assert_eq!(engine.revision(), rev_before);
        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("plain");
        assert_eq!(plain.layers[0].mask.unwrap().center, [0.0, 0.0]);
    }

    #[test]
    fn handle_value_packs_size_and_center() {
        assert_eq!(
            handle_value(HANDLE_CENTER, [0.1, 0.2], [1.0, 1.0], 0.0, 0.0, 0.0),
            Some(ParamValue::Vec2([0.1, 0.2]))
        );
        assert_eq!(
            handle_value(HANDLE_SIZE_Y, [0.0, 0.0], [0.8, 0.4], 0.0, 0.0, 0.0),
            Some(ParamValue::Vec2([0.8, 0.4]))
        );
        assert_eq!(handle_param(HANDLE_NONE), None);
    }

    #[test]
    fn default_value_for_handle_matches_mask_new() {
        let fresh = Mask::new(MaskKind::Rectangle);
        assert_eq!(
            default_value_for_handle(HANDLE_CENTER),
            Some(ParamValue::Vec2([0.0, 0.0]))
        );
        assert_eq!(
            default_value_for_handle(HANDLE_SIZE_X),
            Some(ParamValue::Vec2([1.0, 1.0]))
        );
        assert_eq!(
            default_value_for_handle(HANDLE_FEATHER),
            Some(ParamValue::Scalar(0.0))
        );
        assert_eq!(fresh.center.constant(), Some([0.0, 0.0]));
        assert_eq!(fresh.size.constant(), Some([1.0, 1.0]));
        assert_eq!(fresh.feather.constant(), Some(0.0));
    }

    #[test]
    fn double_click_reset_commits_once() {
        let (mut engine, clip, clip_s, r) = engine_with_mask();
        let param = handle_param(HANDLE_FEATHER).unwrap();
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param,
                value: ParamValue::Scalar(0.7),
            }))
            .expect("seed");
        let rev_before = engine.revision();
        let mut session = MaskGestureSession::new();
        assert!(session.begin(&clip_s, HANDLE_FEATHER));
        // Simulate a prior drag override, then double-click reset path:
        // clear override + one constant commit to default.
        engine.set_param_override(clip, param, ParamValue::Scalar(0.4));
        engine.clear_param_override(clip, param);
        let default = default_value_for_handle(HANDLE_FEATHER).unwrap();
        engine
            .apply(Command::Edit(EditCommand::SetParamConstant {
                clip,
                param,
                value: default,
            }))
            .expect("reset");
        session.end();
        assert_eq!(engine.revision(), rev_before + 1);
        let plain = resolve(engine.project(), RationalTime::new(0, r)).expect("plain");
        assert!((plain.layers[0].mask.unwrap().feather - 0.0).abs() < f32::EPSILON);
    }
}
