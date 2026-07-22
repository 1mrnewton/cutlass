//! Editor command layer: a closed set of structured, deterministic edits.
//!
//! Every edit is an explicit, serializable value — auditable, replayable, and
//! undo-friendly once wired through `cutlass-engine`.

mod command;

pub use command::{Command, EditCommand, EditOutcome, ProjectCommand, TemplatePick};
// Every model type a command field carries, so callers (shell FFI, the AI
// agent, tests) can build any command from this crate alone.
pub use cutlass_models::{
    AnimationRef, AnimationSlot, AudioRole, BlendMode, CanvasAspect, ChromaKey, ClipId, ClipParam,
    ClipTransform, ColorAdjustments, CropRect, Easing, Filter, Generator, LayerShadow, LayerStyles,
    Lut, MarkerColor, MarkerId, Mask, MaskKind, MediaId, Param, ParamValue, Rational, RationalTime,
    Replaceable, SpatialTangents, StabilizeLevel, TemplateMeta, TimeRange, TrackId, TrackKind,
};

use tracing::info;

pub fn init() {
    info!("cutlass-commands ready");
}
