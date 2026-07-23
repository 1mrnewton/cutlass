//! Dedicated OS thread that owns the [`Engine`].
//!
//! `Engine` is not safely shared across tokio worker threads (platform
//! decoders / GPU state). Async MCP tool handlers therefore send a
//! [`HostRequest`] and await a oneshot reply instead of touching the engine
//! directly. When every [`EngineHost`] clone is dropped, the channel closes
//! and the host thread exits.

mod edits;
mod render;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use cutlass_commands::{Command, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{Project, Rational, TrackKind};
use serde::Serialize;
use tokio::sync::oneshot;

pub use edits::{AppliedBatch, AppliedEdit, UndoResult};
pub use render::{ExportDone, FrameGrab};

/// Readable when a tool needs a project and none has been opened yet.
///
/// Deliberate: agents that forgot `project_new` / `project_open` get told,
/// instead of silently editing an unsaved untitled project.
pub const NO_PROJECT: &str = "no project open — call project_new or project_open first";

/// Session flags + identity for the open project (mirrors mobile `SessionMeta`
/// plus name/path so agents can round-trip saves).
#[derive(Debug, Serialize, PartialEq)]
pub struct Meta {
    pub revision: u64,
    pub dirty: bool,
    pub can_undo: bool,
    pub can_redo: bool,
    pub name: String,
    pub path: Option<String>,
}

/// Compact project summary plus session meta — the same summary shape the
/// in-app agent sees via [`cutlass_ai::summarize`].
#[derive(Debug, Serialize)]
pub struct ProjectDoc {
    pub meta: Meta,
    pub project: serde_json::Value,
}

/// Per-path import outcome. Failures for one path do not abort the batch.
#[derive(Debug, Serialize)]
pub struct ImportedMedia {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

enum HostRequest {
    NewProject {
        name: String,
        fps: f64,
        reply: oneshot::Sender<Result<Meta, String>>,
    },
    OpenProject {
        path: PathBuf,
        reply: oneshot::Sender<Result<Meta, String>>,
    },
    SaveProject {
        path: Option<PathBuf>,
        reply: oneshot::Sender<Result<PathBuf, String>>,
    },
    GetProject {
        reply: oneshot::Sender<Result<ProjectDoc, String>>,
    },
    ImportMedia {
        paths: Vec<PathBuf>,
        reply: oneshot::Sender<Result<Vec<ImportedMedia>, String>>,
    },
    ApplyEdits {
        edits: Vec<serde_json::Value>,
        reply: oneshot::Sender<Result<AppliedBatch, String>>,
    },
    Undo {
        reply: oneshot::Sender<Result<UndoResult, String>>,
    },
    Redo {
        reply: oneshot::Sender<Result<UndoResult, String>>,
    },
    GetFrame {
        seconds: f64,
        max_dim: u32,
        reply: oneshot::Sender<Result<FrameGrab, String>>,
    },
    ExportVideo {
        path: PathBuf,
        reply: oneshot::Sender<Result<ExportDone, String>>,
    },
}

/// Cloneable handle to the engine host thread.
#[derive(Clone)]
pub struct EngineHost {
    tx: mpsc::Sender<HostRequest>,
}

impl EngineHost {
    /// Spawn the `"cutlass-mcp-engine"` thread with an empty engine slot.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<HostRequest>();
        thread::Builder::new()
            .name("cutlass-mcp-engine".into())
            .spawn(move || host_loop(rx))
            .expect("spawn cutlass-mcp-engine thread");
        Self { tx }
    }

    pub async fn new_project(&self, name: String, fps: f64) -> Result<Meta, String> {
        self.roundtrip(|reply| HostRequest::NewProject { name, fps, reply })
            .await
    }

    pub async fn open_project(&self, path: PathBuf) -> Result<Meta, String> {
        self.roundtrip(|reply| HostRequest::OpenProject { path, reply })
            .await
    }

    pub async fn save_project(&self, path: Option<PathBuf>) -> Result<PathBuf, String> {
        self.roundtrip(|reply| HostRequest::SaveProject { path, reply })
            .await
    }

    pub async fn get_project(&self) -> Result<ProjectDoc, String> {
        self.roundtrip(|reply| HostRequest::GetProject { reply })
            .await
    }

    pub async fn import_media(&self, paths: Vec<PathBuf>) -> Result<Vec<ImportedMedia>, String> {
        self.roundtrip(|reply| HostRequest::ImportMedia { paths, reply })
            .await
    }

    /// Validate and apply a batch of wire edits as one undo group.
    pub async fn apply_edits(&self, edits: Vec<serde_json::Value>) -> Result<AppliedBatch, String> {
        self.roundtrip(|reply| HostRequest::ApplyEdits { edits, reply })
            .await
    }

    pub async fn undo(&self) -> Result<UndoResult, String> {
        self.roundtrip(|reply| HostRequest::Undo { reply }).await
    }

    pub async fn redo(&self) -> Result<UndoResult, String> {
        self.roundtrip(|reply| HostRequest::Redo { reply }).await
    }

    /// Composite the timeline at `seconds`, fit-scaled to `max_dim`, as PNG.
    pub async fn get_frame(&self, seconds: f64, max_dim: u32) -> Result<FrameGrab, String> {
        self.roundtrip(|reply| HostRequest::GetFrame {
            seconds,
            max_dim,
            reply,
        })
        .await
    }

    /// Mux the whole timeline to an H.264/AAC MP4 at `path` (sync on host).
    pub async fn export_video(&self, path: PathBuf) -> Result<ExportDone, String> {
        self.roundtrip(|reply| HostRequest::ExportVideo { path, reply })
            .await
    }

    async fn roundtrip<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<Result<T, String>>) -> HostRequest,
    ) -> Result<T, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(make(reply_tx))
            .map_err(|_| "engine host thread stopped".to_string())?;
        reply_rx
            .await
            .map_err(|_| "engine host dropped reply".to_string())?
    }
}

fn host_loop(rx: mpsc::Receiver<HostRequest>) {
    let mut engine: Option<Engine> = None;
    while let Ok(req) = rx.recv() {
        match req {
            HostRequest::NewProject { name, fps, reply } => {
                dispatch_request(&mut engine, reply, |slot| do_new_project(slot, name, fps));
            }
            HostRequest::OpenProject { path, reply } => {
                dispatch_request(&mut engine, reply, |slot| do_open_project(slot, path));
            }
            HostRequest::SaveProject { path, reply } => {
                dispatch_request(&mut engine, reply, |slot| do_save_project(slot, path));
            }
            HostRequest::GetProject { reply } => {
                dispatch_request(&mut engine, reply, |slot| do_get_project(slot));
            }
            HostRequest::ImportMedia { paths, reply } => {
                dispatch_request(&mut engine, reply, |slot| do_import_media(slot, paths));
            }
            HostRequest::ApplyEdits { edits, reply } => {
                dispatch_request(&mut engine, reply, |slot| {
                    edits::do_apply_edits(slot, edits)
                });
            }
            HostRequest::Undo { reply } => {
                dispatch_request(&mut engine, reply, edits::do_undo);
            }
            HostRequest::Redo { reply } => {
                dispatch_request(&mut engine, reply, edits::do_redo);
            }
            HostRequest::GetFrame {
                seconds,
                max_dim,
                reply,
            } => {
                dispatch_request(&mut engine, reply, |slot| {
                    render::do_get_frame(slot, seconds, max_dim)
                });
            }
            HostRequest::ExportVideo { path, reply } => {
                dispatch_request(&mut engine, reply, |slot| {
                    render::do_export_video(slot, path)
                });
            }
        }
    }
}

/// Run one request handler under `catch_unwind`.
///
/// A long-lived MCP server must survive one bad edit: without this, a panic
/// inside a `do_*` handler unwinds the `cutlass-mcp-engine` thread, the MCP
/// connection stays up, and every later call returns "engine host thread
/// stopped" while the in-flight call gets the misleading "engine host dropped
/// reply". On panic we reply honestly, discard the (possibly corrupted) open
/// project — unsaved work is lost — and keep the loop alive.
fn dispatch_request<T, F>(
    engine: &mut Option<Engine>,
    reply: oneshot::Sender<Result<T, String>>,
    f: F,
) where
    F: FnOnce(&mut Option<Engine>) -> Result<T, String>,
{
    match catch_unwind(AssertUnwindSafe(|| f(engine))) {
        Ok(payload) => {
            let _ = reply.send(payload);
        }
        Err(panic) => {
            *engine = None;
            let msg = panic_message(&*panic);
            let _ = reply.send(Err(format!(
                "engine crashed while handling the request: {msg}; \
                 open project state was discarded — project_open or project_new to continue"
            )));
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .map(str::to_owned)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

fn do_new_project(slot: &mut Option<Engine>, name: String, fps: f64) -> Result<Meta, String> {
    let rate = resolve_fps(fps)?;
    let mut project = Project::new(name, rate);
    // Every editor session starts with the magnetic main track; the UI
    // (and agents) expect it even when empty.
    project.add_track(TrackKind::Video, "Main");
    let eng = Engine::with_project(EngineConfig::default(), project).map_err(eng_err)?;
    let meta = meta_of(&eng);
    *slot = Some(eng);
    Ok(meta)
}

fn do_open_project(slot: &mut Option<Engine>, path: PathBuf) -> Result<Meta, String> {
    let mut eng = Engine::new(EngineConfig::default()).map_err(eng_err)?;
    // `Load` tolerates missing media paths so a moved project still opens;
    // missing media shows up in the project summary for later relink.
    eng.apply(Command::Project(ProjectCommand::Load { path }))
        .map_err(eng_err)?;
    let meta = meta_of(&eng);
    *slot = Some(eng);
    Ok(meta)
}

fn do_save_project(slot: &mut Option<Engine>, path: Option<PathBuf>) -> Result<PathBuf, String> {
    let engine = require_engine_mut(slot)?;
    let path = match path {
        Some(p) => p,
        None => engine
            .project_path()
            .cloned()
            .ok_or_else(|| "project has no file yet — pass a path".to_string())?,
    };
    engine
        .apply(Command::Project(ProjectCommand::Save {
            path: path.clone(),
        }))
        .map_err(eng_err)?;
    Ok(path)
}

fn do_get_project(slot: &Option<Engine>) -> Result<ProjectDoc, String> {
    let engine = require_engine(slot)?;
    let summary = cutlass_ai::summarize(engine.project());
    let project = serde_json::to_value(&summary).map_err(|e| e.to_string())?;
    Ok(ProjectDoc {
        meta: meta_of(engine),
        project,
    })
}

fn do_import_media(
    slot: &mut Option<Engine>,
    paths: Vec<PathBuf>,
) -> Result<Vec<ImportedMedia>, String> {
    let engine = require_engine_mut(slot)?;
    let mut results = Vec::with_capacity(paths.len());
    for path in paths {
        let path_str = path.display().to_string();
        match engine.apply(Command::Project(ProjectCommand::Import {
            path: path.clone(),
        })) {
            Ok(ApplyOutcome::Imported { media }) => {
                let summary = cutlass_ai::summarize(engine.project());
                let media_val = summary
                    .media
                    .iter()
                    .find(|m| m.id == media.raw())
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or(serde_json::Value::Null);
                results.push(ImportedMedia {
                    path: path_str,
                    media: Some(media_val),
                    error: None,
                });
            }
            Ok(other) => {
                results.push(ImportedMedia {
                    path: path_str,
                    media: None,
                    error: Some(format!("unexpected import outcome: {other:?}")),
                });
            }
            Err(e) => {
                results.push(ImportedMedia {
                    path: path_str,
                    media: None,
                    error: Some(eng_err(e)),
                });
            }
        }
    }
    Ok(results)
}

fn require_engine(slot: &Option<Engine>) -> Result<&Engine, String> {
    slot.as_ref().ok_or_else(|| NO_PROJECT.to_string())
}

fn require_engine_mut(slot: &mut Option<Engine>) -> Result<&mut Engine, String> {
    slot.as_mut().ok_or_else(|| NO_PROJECT.to_string())
}

fn meta_of(engine: &Engine) -> Meta {
    Meta {
        revision: engine.revision(),
        dirty: engine.is_dirty(),
        can_undo: engine.can_undo(),
        can_redo: engine.can_redo(),
        name: engine.project().name.clone(),
        path: engine
            .project_path()
            .map(|p| p.to_string_lossy().into_owned()),
    }
}

fn eng_err(e: impl std::fmt::Display) -> String {
    format!("{e}")
}

/// Map a tool-facing fps float onto a named timeline rate.
///
/// Arbitrary rationals are rejected — the project frame rate is load-bearing
/// for timeline ticks, media resampling, and export.
fn resolve_fps(fps: f64) -> Result<Rational, String> {
    const EPS: f64 = 1e-3;
    const CANDIDATES: &[Rational] = &[
        Rational::FPS_24,
        Rational::FPS_23_976,
        Rational::FPS_25,
        Rational::FPS_30,
        Rational::FPS_29_97,
        Rational::FPS_50,
        Rational::FPS_60,
        Rational::FPS_59_94,
    ];
    for &rate in CANDIDATES {
        if (fps - rate.as_f64()).abs() < EPS {
            return Ok(rate);
        }
    }
    Err(format!(
        "unsupported frame rate {fps}: supported rates are \
         24, 23.976, 25, 30, 29.97, 50, 60, 59.94"
    ))
}

#[cfg(test)]
mod tests;
