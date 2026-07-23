//! Background preview rendering: engine and decode/composite stay off the UI thread.
//!
//! Ported from main's crates/cutlass-ui onto this branch's engine: engine
//! ownership, the full edit/project message set, debounced autosave, the
//! fit-sized preview pump, audio snapshots, thumbnail/strip registration,
//! export, live gesture/generator overrides, and the AI agent bridge.

mod agent_bridge;
mod clip_audio;
mod clip_look;
mod clip_place;
mod clip_retime;
mod clipboard;
mod dispatch;
mod edit_helpers;
mod effects;
mod export;
mod frame_cache;
mod frame_fit;
mod handle;
mod import_drop;
mod markers_tracks;
mod overrides;
mod project;
mod proxy;
mod publish;
mod render;
mod rpc;
#[cfg(test)]
mod tests;
mod timeline_ops;
mod types;
mod worker_loop;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, bounded, unbounded};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand, TemplatePick};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig, SeekPolicy};
use cutlass_models::{
    AnimatedTransform, BlendMode, ChromaKey, ClipId, ClipParam, ClipSource, ClipTransform,
    ColorAdjustments, CropRect, Easing, Filter, Generator, LayerStyles, LinkId, Lut, MAX_SPEED,
    MIN_SPEED, MarkerColor, MarkerId, Mask, MediaId, MotionBlur, Param, ParamValue,
    PiecewiseEasingPreset, Project, Rational, RationalTime, TimeRange, Track, TrackId, TrackKind,
    resample,
};
use cutlass_render::{ExportSettings, RenderError, Renderer};
use slint::{Rgba8Pixel, SharedPixelBuffer};
use tracing::{debug, error, info, warn};

use crate::agent::{AgentCreated, AgentPlanStep};
use crate::proxy::ProxyHandle;
use crate::strips::StripHandle;
use crate::thumbnails::{ThumbKind, ThumbnailHandle};
use crate::{EditorStore, ExportBackend, PreviewStore};

use agent_bridge::*;
use clip_audio::*;
use clip_look::*;
use clip_place::*;
use clip_retime::*;
use clipboard::*;
use dispatch::*;
use edit_helpers::*;
use effects::*;
use export::*;
use frame_cache::*;
use frame_fit::*;
use import_drop::*;
use markers_tracks::*;
use overrides::*;
use project::*;
use proxy::*;
use publish::*;
use render::*;
use rpc::*;
use timeline_ops::*;
use types::*;
// `PreviewWorker` (the other `pub` item in `worker_loop`) is re-exported
// explicitly below; this one is a testable seam exercised directly by
// `preview_worker::tests` and otherwise unused outside `worker_loop` itself.
#[allow(unused_imports)]
use worker_loop::message_invalidates_preview;

// Re-exported for `agent::tests`, which replays plans directly against a live
// engine; unused outside `#[cfg(test)]` builds since `agent_bridge` itself
// already reaches `agent_replay` through the glob import above.
#[allow(unused_imports)]
pub(crate) use agent_bridge::{STALE_PLAN_SEED_ERROR, agent_apply_with_seed, agent_replay};
pub(crate) use rpc::ProjectMaintenanceGuard;
pub(crate) use types::{
    ApplyTemplateRpcResult, ImportMediaRpcResult, NewProjectRpcResult, OpenProjectRpcResult,
    PreviewCacheStats, RelinkFolderRpcResult, RelinkMediaRpcResult, SaveProjectRpcResult,
};
pub use types::{
    ExportRequest, GroupMove, MoveParamKeyframeRequest, PreviewSession, TrackFlag, WorkerHandle,
};
pub use worker_loop::PreviewWorker;
