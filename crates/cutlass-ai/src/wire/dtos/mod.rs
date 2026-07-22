//! The command DTOs: every argument struct and enum in the wire vocabulary.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod look;
mod params;
mod timeline;

pub use look::*;
pub use params::*;
pub use timeline::*;

/// Model-facing clip lists stay small enough for deterministic validation and
/// useful rejection messages while covering realistic linked groups.
pub(crate) const MAX_MULTI_CLIP_REFS: usize = 64;

/// Track lane categories the agent may create or target.
///
/// The engine has more kinds (effect / filter / adjustment lanes); the agent
/// cannot create those yet — it applies effects, filters, and adjustments to
/// clips directly instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireTrackKind {
    /// Footage and other imported picture media.
    Video,
    /// Imported sound media.
    Audio,
    /// Titles and captions.
    Text,
    /// Graphic overlays: solid colors and shapes.
    Sticker,
}

/// Geometry of a generated shape clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireShape {
    Rectangle,
    Ellipse,
}

/// Synthetic clip content the agent may create.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireGenerator {
    /// A title / text layer (rendered with the default style; styling is
    /// preserved when replacing the text of an existing text clip).
    Text {
        /// The text to display.
        content: String,
    },
    /// A solid color fill covering the canvas.
    Solid {
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
    },
    /// A filled vector shape centered on the canvas.
    Shape {
        shape: WireShape,
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
        /// Width in reference pixels (1080px-tall canvas). Omit to keep the
        /// clip's current size when editing an existing shape.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        width: Option<f32>,
        /// Height in reference pixels. Omit to keep the clip's current size.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        height: Option<f32>,
    },
}

/// Add a track to the timeline stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddTrack {
    pub kind: WireTrackKind,
    /// Display name, e.g. "V2" or "Music".
    pub name: String,
    /// Stack position (0 = bottom layer, composited first). Omit to add on
    /// top of the stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

/// Place a trimmed range of imported media on a video or audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddClip {
    /// Target track id.
    pub track: u64,
    /// Media pool id of the source file.
    pub media: u64,
    /// In-point within the source media, in seconds.
    pub source_start: f64,
    /// Length of the source range to use, in seconds.
    pub source_duration: f64,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
}

/// Detach a video clip's embedded sound onto an explicit audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractAudio {
    /// Source media clip on a video track.
    pub clip: u64,
    /// Target unlocked audio track. This is required: call `add_track` first
    /// when no suitable audio lane exists, then use its returned id.
    pub track: u64,
}

/// Make a deep property-preserving copy of one clip at an explicit target
/// track and timeline start. The copy receives a fresh unlinked clip id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DuplicateClip {
    /// Source clip to copy.
    pub clip: u64,
    /// Explicit destination track id.
    pub to_track: u64,
    /// Explicit destination start in timeline seconds.
    pub start: f64,
}

/// Place a generated clip (text, solid color, shape) on a matching track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddGenerated {
    /// Target track id. Text goes on text tracks; solids and shapes go on
    /// sticker (overlay) tracks.
    pub track: u64,
    /// The content to generate, as a tagged object — e.g.
    /// `{"type": "text", "content": "Hello"}`,
    /// `{"type": "solid", "rgba": [0, 0, 0, 255]}`, or
    /// `{"type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255]}`.
    pub generator: WireGenerator,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
    /// Clip length on the timeline, in seconds.
    pub duration: f64,
}

/// Replace a generated clip's content (edit a title's text, recolor a
/// shape). Rejected for media-backed clips. Replacing the text of a text
/// clip keeps its current styling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetGenerator {
    /// The generated clip to modify.
    pub clip: u64,
    /// The replacement content, as a tagged object — e.g.
    /// `{"type": "text", "content": "Hello"}`,
    /// `{"type": "solid", "rgba": [0, 0, 0, 255]}`, or
    /// `{"type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255]}`.
    pub generator: WireGenerator,
}
