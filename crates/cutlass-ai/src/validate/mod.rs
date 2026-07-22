//! Validation + lowering: wire commands → engine commands, against a live
//! project snapshot.
//!
//! This is the guardrail chokepoint. Every rejection carries a
//! model-readable message (including the ids that *do* exist) so the agent
//! loop can feed it straight back as a tool result and let the model correct
//! course. Checks here are for good error messages and the whitelist; the
//! engine remains the authority (overlaps, source bounds, rate math) and
//! re-validates everything on apply.

mod audio;
mod command;
mod look;
mod lookup;
mod lower;
#[cfg(test)]
mod tests;
mod time;

use std::collections::HashSet;

use cutlass_commands::{Command, EditCommand};
use cutlass_models::{
    AnimationRef, AnimationSlot, AudioRole, BlendMode, CanvasAspect, ChromaKey, Clip, ClipId,
    ClipParam, ClipTransform, CropRect, Easing, Filter, Generator, Marker, MarkerColor, MarkerId,
    Mask, MaskKind, MediaId, Param, ParamValue, Project, Rational, RationalTime, StabilizeLevel,
    TimeRange, TrackId, TrackKind, animation_spec, filter_catalog, filter_spec,
};

use crate::wire::{
    MAX_MULTI_CLIP_REFS, WireAnimationSlot, WireAudioRole, WireBlendMode, WireCanvasAspect,
    WireChromaKey, WireClipParam, WireCommand, WireEasing, WireGenerator, WireMarkerColor,
    WireMask, WireMaskKind, WireShape, WireStabilizeLevel, WireTrackKind,
};

use audio::*;
use look::*;
use lookup::*;
use lower::*;
use time::*;

/// A wire command the project as it stands cannot accept. The message is
/// written for the model: it names the problem and the valid alternatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub message: String,
}

impl Rejection {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Rejection {}

pub use command::validate;
