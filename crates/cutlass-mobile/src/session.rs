//! The session FFI: one opaque handle around the full editing engine.
//!
//! A `CutlassSession` owns a [`cutlass_engine::Engine`] — project state,
//! inverse undo/redo, and a persistent GPU renderer — so the shells get the
//! exact editing runtime the desktop uses. Commands and intents cross as JSON
//! (see [`crate::wire`]); preview pixels cross as the binary [`CutlassImage`].
//!
//! Threading: a session is **not** internally synchronized. The shell must
//! serialize all calls on one handle (the Swift bridge wraps it in an actor;
//! the engine is `Send`, so the actor's executor may hop threads).

use std::ffi::c_char;
use std::path::PathBuf;

use cutlass_commands::{Command, ProjectCommand};
use cutlass_engine::{Engine, EngineConfig, EngineError};
use cutlass_models::{Project, Rational, RationalTime, TrackKind, resample};

use crate::CutlassImage;
use crate::intents::Intent;
use crate::ui_state::{SessionMeta, UiState};
use crate::wire::{err_response, ok_response, protocol_err_response, str_arg, to_c_string};

/// An editing session: the engine plus nothing else (kept as a struct so the
/// FFI surface can grow session-scoped state without re-plumbing handles).
pub struct CutlassSession {
    engine: Engine,
}

impl CutlassSession {
    fn new(frame_rate: Rational) -> Result<Self, EngineError> {
        let mut project = Project::new("untitled", frame_rate);
        // Every editor session starts with the magnetic main track; the UI
        // renders it even when empty.
        project.add_track(TrackKind::Video, "Main");
        let engine = Engine::with_project(EngineConfig::default(), project)?;
        Ok(Self { engine })
    }

    fn open(path: &str) -> Result<Self, EngineError> {
        let mut engine = Engine::new(EngineConfig::default())?;
        // `Load` tolerates missing media paths (mobile sandboxes move), so an
        // edited-elsewhere project still opens and can be relinked.
        engine.apply(Command::Project(ProjectCommand::Load {
            path: PathBuf::from(path),
        }))?;
        Ok(Self { engine })
    }

    fn meta(&self) -> SessionMeta {
        SessionMeta {
            revision: self.engine.revision(),
            dirty: self.engine.is_dirty(),
            can_undo: self.engine.can_undo(),
            can_redo: self.engine.can_redo(),
        }
    }

    fn ui_state_json(&self) -> String {
        UiState::build(self.engine.project(), self.meta()).to_json()
    }

    fn apply_json(&mut self, json: &str) -> String {
        let command: Command = match serde_json::from_str(json) {
            Ok(command) => command,
            Err(e) => return protocol_err_response(format!("unparseable command: {e}")),
        };
        match self.engine.apply(command) {
            Ok(outcome) => match serde_json::to_value(&outcome) {
                Ok(payload) => ok_response(payload, self.engine.revision()),
                Err(e) => protocol_err_response(format!("unserializable outcome: {e}")),
            },
            Err(e) => err_response(&e),
        }
    }

    fn intent_json(&mut self, json: &str) -> String {
        let intent: Intent = match serde_json::from_str(json) {
            Ok(intent) => intent,
            Err(e) => return protocol_err_response(format!("unparseable intent: {e}")),
        };
        match crate::intents::run(&mut self.engine, intent) {
            Ok(payload) => ok_response(payload, self.engine.revision()),
            Err(e) => err_response(&e),
        }
    }

    fn duration_seconds(&self) -> f64 {
        self.engine.project().timeline().duration().seconds()
    }

    /// Render the frame nearest `seconds`, fit within `max_w`×`max_h`.
    fn render_fit(
        &mut self,
        seconds: f64,
        max_w: u32,
        max_h: u32,
    ) -> Option<cutlass_render::RgbaImage> {
        let rate = self.engine.project().timeline().frame_rate;
        let duration = resample(self.engine.project().timeline().duration(), rate);
        let max_frame = (duration.value - 1).max(0);
        let frame = crate::intents::frame_tick(seconds, rate).clamp(0, max_frame);
        match self
            .engine
            .get_frame_fit(RationalTime::new(frame, rate), max_w, max_h)
        {
            Ok(image) => Some(image),
            Err(e) => {
                log::error!("session render at frame {frame} failed: {e}");
                None
            }
        }
    }

    /// Direct engine access for in-process (rlib) consumers: the export job
    /// snapshots the project through this.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

/// Open a fresh session: an empty project at `fps_num/fps_den` (falls back to
/// 30 fps when degenerate) with a main track. Null when the GPU renderer
/// can't be brought up. Free with [`cutlass_session_close`].
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_session_new(fps_num: i32, fps_den: i32) -> *mut CutlassSession {
    let rate = if fps_num > 0 && fps_den > 0 {
        Rational::new(fps_num, fps_den)
    } else {
        Rational::FPS_30
    };
    match CutlassSession::new(rate) {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(e) => {
            log::error!("session new failed: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Open a session from a `.cutlass` project file (UTF-8 path, `path_len`
/// bytes). Missing media paths are tolerated. Null on failure.
///
/// # Safety
/// `path_utf8` must point to `path_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_open(
    path_utf8: *const u8,
    path_len: usize,
) -> *mut CutlassSession {
    // SAFETY: caller guarantees the byte range.
    let Some(path) = (unsafe { str_arg(path_utf8, path_len) }) else {
        return std::ptr::null_mut();
    };
    match CutlassSession::open(path) {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(e) => {
            log::error!("session open failed for {path}: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Release a session handle. Null is a no-op.
///
/// # Safety
/// `handle` must be null or a live pointer from a `cutlass_session_*`
/// constructor, not yet closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_close(handle: *mut CutlassSession) {
    if !handle.is_null() {
        // SAFETY: came from Box::into_raw, freed exactly once.
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Borrow the session behind an FFI handle for one call.
///
/// # Safety
/// Caller contract of every session fn: `handle` is a live, non-concurrent
/// session pointer.
unsafe fn session<'a>(handle: *mut CutlassSession) -> Option<&'a mut CutlassSession> {
    // SAFETY: per the caller contract above.
    unsafe { handle.as_mut() }
}

/// Apply one wire [`Command`] (`json_utf8`, `json_len` bytes — the flat
/// `{"type": …}` object). Returns a JSON response envelope; free it with
/// [`cutlass_string_free`].
///
/// # Safety
/// `handle` must be a live session used non-concurrently; `json_utf8` must
/// point to `json_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_apply(
    handle: *mut CutlassSession,
    json_utf8: *const u8,
    json_len: usize,
) -> *mut c_char {
    // SAFETY: caller contract.
    let (session, json) = unsafe { (session(handle), str_arg(json_utf8, json_len)) };
    let response = match (session, json) {
        (Some(session), Some(json)) => session.apply_json(json),
        (None, _) => protocol_err_response("null session handle"),
        (_, None) => protocol_err_response("command payload is not valid UTF-8"),
    };
    to_c_string(response)
}

/// Run one gesture-level [`Intent`] (`{"intent": …}` JSON). Multi-command
/// intents are grouped into a single undo step and roll back atomically on
/// failure. Returns a JSON response envelope; free with
/// [`cutlass_string_free`].
///
/// # Safety
/// Same contract as [`cutlass_session_apply`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_intent(
    handle: *mut CutlassSession,
    json_utf8: *const u8,
    json_len: usize,
) -> *mut c_char {
    // SAFETY: caller contract.
    let (session, json) = unsafe { (session(handle), str_arg(json_utf8, json_len)) };
    let response = match (session, json) {
        (Some(session), Some(json)) => session.intent_json(json),
        (None, _) => protocol_err_response("null session handle"),
        (_, None) => protocol_err_response("intent payload is not valid UTF-8"),
    };
    to_c_string(response)
}

/// The full UI presentation state (lanes, clips, canvas, durations) as JSON.
/// Free with [`cutlass_string_free`]. Null for a null handle.
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_ui_state(handle: *mut CutlassSession) -> *mut c_char {
    // SAFETY: caller contract.
    match unsafe { session(handle) } {
        Some(session) => to_c_string(session.ui_state_json()),
        None => std::ptr::null_mut(),
    }
}

/// Undo the most recent edit (or group). Returns whether anything changed.
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_undo(handle: *mut CutlassSession) -> bool {
    // SAFETY: caller contract.
    unsafe { session(handle) }.is_some_and(|s| s.engine.undo())
}

/// Redo the most recently undone edit. Returns whether anything changed.
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_redo(handle: *mut CutlassSession) -> bool {
    // SAFETY: caller contract.
    unsafe { session(handle) }.is_some_and(|s| s.engine.redo())
}

/// Whether an undo step is available.
///
/// # Safety
/// `handle` must be a live session (read-only is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_can_undo(handle: *mut CutlassSession) -> bool {
    // SAFETY: caller contract.
    unsafe { session(handle) }.is_some_and(|s| s.engine.can_undo())
}

/// Whether a redo step is available.
///
/// # Safety
/// `handle` must be a live session (read-only is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_can_redo(handle: *mut CutlassSession) -> bool {
    // SAFETY: caller contract.
    unsafe { session(handle) }.is_some_and(|s| s.engine.can_redo())
}

/// Monotonic session revision (bumps on every successful mutation). Shells
/// use it to invalidate caches (thumbnails, audio readers).
///
/// # Safety
/// `handle` must be a live session (read-only is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_revision(handle: *mut CutlassSession) -> u64 {
    // SAFETY: caller contract.
    unsafe { session(handle) }.map_or(0, |s| s.engine.revision())
}

/// Whether the session has edits not yet saved to its project file.
///
/// # Safety
/// `handle` must be a live session (read-only is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_is_dirty(handle: *mut CutlassSession) -> bool {
    // SAFETY: caller contract.
    unsafe { session(handle) }.is_some_and(|s| s.engine.is_dirty())
}

/// Open a history group: every command until `commit_group` folds into one
/// undo step (property-panel sessions, multi-command gestures driven from
/// the shell side).
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_begin_group(handle: *mut CutlassSession) {
    // SAFETY: caller contract.
    if let Some(session) = unsafe { session(handle) } {
        session.engine.begin_group();
    }
}

/// Close the open history group as one undo entry (no-op when empty).
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_commit_group(handle: *mut CutlassSession) {
    // SAFETY: caller contract.
    if let Some(session) = unsafe { session(handle) } {
        session.engine.commit_group();
    }
}

/// Abort the open history group, reverting its commands (panel Cancel).
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_rollback_group(handle: *mut CutlassSession) {
    // SAFETY: caller contract.
    if let Some(session) = unsafe { session(handle) } {
        session.engine.rollback_group();
    }
}

/// End of the timeline in seconds (0 for an empty project or null handle).
///
/// # Safety
/// `handle` must be a live session (read-only is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_duration_seconds(handle: *mut CutlassSession) -> f64 {
    // SAFETY: caller contract.
    unsafe { session(handle) }.map_or(0.0, |s| s.duration_seconds())
}

/// Render the timeline frame nearest `seconds` (snapped to the project frame
/// grid, clamped to the timeline), scaled to fit `max_width`×`max_height`.
/// On failure the returned [`CutlassImage`] has `data == null`; release every
/// non-null result with [`cutlass_image_free`].
///
/// # Safety
/// `handle` must be a live session used non-concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_render_fit(
    handle: *mut CutlassSession,
    seconds: f64,
    max_width: u32,
    max_height: u32,
) -> CutlassImage {
    // SAFETY: caller contract.
    let Some(session) = (unsafe { session(handle) }) else {
        return CutlassImage::empty();
    };
    if max_width == 0 || max_height == 0 {
        return CutlassImage::empty();
    }
    match session.render_fit(seconds, max_width, max_height) {
        Some(image) => CutlassImage::from_rgba(image),
        None => CutlassImage::empty(),
    }
}
