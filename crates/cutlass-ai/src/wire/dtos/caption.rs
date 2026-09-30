//! Caption DTOs: creating a caption group and restyling it.
//!
//! A caption cue *is* a text clip, so everything the agent already knows about
//! text clips (trim, move, keyframes, layer styles) works on a cue unchanged.
//! What needs its own vocabulary is the group: one call that lays down N lines,
//! and one call each to restyle, re-lay-out, or highlight all of them.
//!
//! Deliberately absent: word timings. They come from speech recognition (a
//! desktop job), not from a language model, and a cue without them simply
//! renders without karaoke highlighting.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Model-facing cue lists stay small enough that one `add_captions` call is
/// still a readable tool argument; longer scripts arrive as several calls.
pub(crate) const MAX_WIRE_CAPTION_CUES: usize = 200;

/// One caption line and when it shows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireCaptionCue {
    /// The line as it reads on screen; "\n" for a deliberate break.
    pub text: String,
    /// Timeline position, in seconds.
    pub start: f64,
    /// Time on screen, in seconds.
    pub duration: f64,
}

/// Lay a run of caption lines onto a text track as one group, so a later
/// restyle or removal is a single call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddCaptions {
    /// Text track id; add_track(kind="text") first if there is no text lane.
    pub track: u64,
    /// The lines, in timeline order and non-overlapping.
    pub cues: Vec<WireCaptionCue>,
    /// Name for the caption list ("Intro captions").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Template id styling every line: clean, bold_box, karaoke_pop, glow,
    /// outline, multiline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
}

/// Remove a caption group and every line it owns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveCaptions {
    /// Caption group id (see describe_project).
    pub group: u64,
}

/// Restyle a whole caption group from the template catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCaptionTemplate {
    pub group: u64,
    /// Template id: clean, bold_box, karaoke_pop, glow, outline, multiline.
    pub template: String,
}

/// Adjust a caption group's shared look. Omitted fields keep their current
/// value, so this patches whatever the template left behind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCaptionStyle {
    pub group: u64,
    /// Font family name ("" for the system default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font: Option<String>,
    /// Font size in reference px (1080p-tall canvas).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    /// Fill as [red, green, blue, alpha], each 0-255.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<[u8; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bold: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub italic: Option<bool>,
    /// Render every line in capitals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uppercase: Option<bool>,
    /// Vertical placement, canvas fractions down from the center: 0 centers
    /// the block, 0.32 is the default just above the safe area, 0.5 the edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_y: Option<f64>,
    /// Uniform scale on top of the font size (1.0 = the size as given).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Leave hand-styled lines alone (default false: restyle every line).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_overrides: Option<bool>,
}

/// Change how a caption group breaks speech into lines and where they sit.
/// These rules govern re-segmentation and the safe area; existing lines move
/// into the new safe area but are not re-split. Omitted fields keep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCaptionLayout {
    pub group: u64,
    /// Characters per line before a break (8–256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars_per_line: Option<u32>,
    /// Lines one cue may hold (1–6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_lines: Option<u32>,
    /// Shortest a line is held, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_duration: Option<f64>,
    /// Longest a line runs before it is split, in seconds (max 30).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_duration: Option<f64>,
    /// Gap between lines so they visibly change, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_gap: Option<f64>,
    /// Clearance from the canvas bottom, canvas fractions (0–0.5). The 0.18
    /// default clears the TikTok/Reels UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safe_area_bottom: Option<f64>,
}

/// How a caption group picks out the word being spoken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireCaptionHighlightMode {
    /// No highlight; every line renders in one color.
    Off,
    /// Recolor the word being spoken (karaoke).
    Word,
    /// Recolor every word up to the one being spoken (progressive fill).
    Line,
}

/// Set or clear a caption group's word highlight.
///
/// Highlighting needs per-word timings, which only auto-generated captions
/// carry — on hand-written or imported lines this setting is stored and simply
/// does not draw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCaptionHighlight {
    pub group: u64,
    /// "off" clears the highlight; "word" and "line" turn it on.
    pub mode: WireCaptionHighlightMode,
    /// Highlighted-word fill as [red, green, blue, alpha], each 0-255.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<[u8; 4]>,
    /// Card behind the highlighted word; alpha 0 for a plain color swap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plate: Option<[u8; 4]>,
    /// Card corner rounding, 0 (square) to 1 (pill).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plate_radius: Option<f64>,
    /// Size multiplier for the highlighted word (1.0 = none, 0.25–4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
}

/// Reword one caption line. Word timings ride along, remapped onto the new
/// text, so karaoke highlighting stays roughly in sync.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCaptionText {
    /// The caption cue clip to edit.
    pub clip: u64,
    /// The corrected line.
    pub text: String,
    /// Speaker label for the line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// Merge caption lines of one group into the earliest, joining their text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MergeCaptions {
    /// Two or more cue clips of the same caption group.
    pub clips: Vec<u64>,
}
