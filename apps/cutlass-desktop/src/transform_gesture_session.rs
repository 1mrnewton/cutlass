//! Transform-gesture session pairing for the preview worker fast path.
//!
//! Canvas presses call [`TransformGestureSession::begin`] before the first
//! override so the worker can build partitioned sprite frames. Inspector
//! sliders (and any other override-first caller) skip that press — they only
//! emit overrides — so [`TransformGestureSession::preview`] auto-begins on the
//! first tick of a drag. Commit / clear / abandon end the session so the next
//! drag begins cleanly, including while Slint still mirrors
//! `gesture-commit-pending` after a successful commit.

use cutlass_models::ClipTransform;

use crate::preview_worker::WorkerHandle;

/// Whether a transform gesture is currently paired with the worker.
#[derive(Debug, Default, Clone)]
pub struct TransformGestureSession {
    active_clip: Option<String>,
    /// Last override values mirrored for on-canvas selection/handles.
    /// Cleared on commit / clear / abandon so Escape restores the pre-drag
    /// box from the (still frozen) projection.
    overlay: Option<(String, ClipTransform)>,
}

impl TransformGestureSession {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn is_active(&self) -> bool {
        self.active_clip.is_some()
    }

    #[cfg(test)]
    pub fn active_clip(&self) -> Option<&str> {
        self.active_clip.as_deref()
    }

    /// Live overlay mirror updated on every preview tick (inspector or canvas).
    pub fn overlay_mirror(&self) -> Option<(&str, &ClipTransform)> {
        self.overlay.as_ref().map(|(id, t)| (id.as_str(), t))
    }

    /// Explicit begin (canvas press). Returns `true` when the caller should
    /// send `BeginTransformGesture`. A second begin for the same clip is a
    /// no-op so canvas + inspector can't double-build sprites mid-drag.
    pub fn begin(&mut self, clip: &str) -> bool {
        if self.active_clip.as_deref() == Some(clip) {
            return false;
        }
        self.active_clip = Some(clip.to_string());
        true
    }

    /// First override of a drag auto-begins when the caller skipped press.
    /// Returns `true` when a begin should be sent before the override.
    pub fn preview(&mut self, clip: &str) -> bool {
        self.begin(clip)
    }

    fn mirror_overlay(&mut self, clip: &str, transform: ClipTransform) {
        self.overlay = Some((clip.to_string(), transform));
    }

    /// Commit, clear, or abandon — the next drag must begin again.
    pub fn end(&mut self) {
        self.active_clip = None;
        self.overlay = None;
    }
}

/// Drive the worker through an inspector-style (or canvas) preview tick:
/// begin once if needed, mirror into overlay state, then send the override.
pub fn preview_transform(
    session: &mut TransformGestureSession,
    handle: &WorkerHandle,
    clip: String,
    transform: ClipTransform,
    tick: i64,
) {
    if session.preview(&clip) {
        handle.begin_transform_gesture(clip.clone(), tick);
    }
    session.mirror_overlay(&clip, transform);
    handle.transform_override(clip, transform, tick);
}

/// Explicit canvas press begin.
pub fn begin_transform_gesture(
    session: &mut TransformGestureSession,
    handle: &WorkerHandle,
    clip: String,
    tick: i64,
) {
    if session.begin(&clip) {
        handle.begin_transform_gesture(clip, tick);
    }
}

/// Commit one undoable transform and end the session.
pub fn commit_transform(
    session: &mut TransformGestureSession,
    handle: &WorkerHandle,
    clip: String,
    transform: ClipTransform,
    tick: i64,
) {
    session.end();
    handle.set_transform(clip, transform, tick);
}

/// No-op / cancelled release: drop the override and end the session.
pub fn clear_transform_override(
    session: &mut TransformGestureSession,
    handle: &WorkerHandle,
    tick: i64,
) {
    session.end();
    handle.clear_transform_override(tick);
}

/// Press ended without a drag: drop prepared sprite frames.
pub fn abandon_transform_gesture(session: &mut TransformGestureSession, handle: &WorkerHandle) {
    session.end();
    handle.end_transform_gesture();
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_commands::{Command, EditCommand};
    use cutlass_engine::{Engine, EngineConfig};
    use cutlass_models::{
        ClipTransform, MediaSource, Project, Rational, RationalTime, Scale2, TimeRange, TrackKind,
    };

    fn identity_transform() -> ClipTransform {
        ClipTransform {
            position: [0.0, 0.0],
            anchor_point: [0.5, 0.5],
            scale: Scale2 { x: 1.0, y: 1.0 },
            rotation: 0.0,
            opacity: 1.0,
        }
    }

    fn moved_transform() -> ClipTransform {
        ClipTransform {
            position: [0.1, -0.2],
            ..identity_transform()
        }
    }

    #[test]
    fn same_clip_begin_is_idempotent() {
        let mut session = TransformGestureSession::new();
        assert!(session.begin("42"));
        assert!(!session.begin("42"));
        assert_eq!(session.active_clip(), Some("42"));
    }

    #[test]
    fn clip_switch_rebegins() {
        let mut session = TransformGestureSession::new();
        assert!(session.begin("a"));
        assert!(session.begin("b"));
        assert_eq!(session.active_clip(), Some("b"));
    }

    #[test]
    fn preview_auto_begins_then_stays_active() {
        let mut session = TransformGestureSession::new();
        assert!(session.preview("7"));
        assert!(!session.preview("7"));
        assert!(session.is_active());
        session.end();
        assert!(session.preview("7"), "next drag begins again after end");
    }

    #[test]
    fn overlay_mirror_clears_on_end() {
        let mut session = TransformGestureSession::new();
        session.begin("7");
        session.mirror_overlay("7", moved_transform());
        let (id, mirrored) = session.overlay_mirror().expect("mirrored");
        assert_eq!(id, "7");
        assert_eq!(mirrored.position, [0.1, -0.2]);
        session.end();
        assert!(
            session.overlay_mirror().is_none(),
            "cancel/commit clears overlay so handles return to projection"
        );
    }

    #[test]
    fn inspector_style_commit_is_exactly_one_undoable_edit() {
        let r = Rational::FPS_24;
        let mut project = Project::new("gesture-commit", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/gesture-commit.mp4",
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
        let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        assert!(!engine.can_undo());

        // Mirror the worker arms: live override (no history), then one commit.
        engine.set_transform_override(Some((clip, moved_transform())));
        assert!(engine.has_live_overrides());
        engine.set_transform_override(None);
        engine
            .apply(Command::Edit(EditCommand::SetClipTransform {
                clip,
                transform: moved_transform(),
                at: Some(RationalTime::new(12, r)),
            }))
            .expect("commit");

        assert!(engine.can_undo());
        assert!(engine.undo());
        assert!(!engine.can_undo());
    }

    #[test]
    fn inspector_style_abandon_leaves_no_undo_entry() {
        let r = Rational::FPS_24;
        let mut project = Project::new("gesture-abandon", r);
        let media = project.add_media(MediaSource::new(
            "/tmp/gesture-abandon.mp4",
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
        let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");

        engine.set_transform_override(Some((clip, moved_transform())));
        engine.set_transform_override(None);
        assert!(!engine.can_undo());
        assert!(!engine.has_live_overrides());
    }
}
