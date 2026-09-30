//! Captions: CapCut-style timed subtitle lines.
//!
//! A caption *cue* is an ordinary text clip — `Generator::Text` on a
//! [`TrackKind::Text`](crate::TrackKind) lane — carrying [`CaptionCue`]
//! metadata. That is deliberate: cues inherit typography, glyph treatments,
//! transform keyframes, look animations, persistence, undo, and the whole AI
//! command vocabulary from text clips, and stay individually trimmable,
//! movable, and splittable on the timeline.
//!
//! What captions add on top is the [`CaptionGroup`]: a sequence-scoped entity
//! holding the style template, segmentation rules ([`CaptionLayout`]),
//! provenance ([`CaptionSource`]), and playback highlighting
//! ([`CaptionHighlight`]) shared by a batch of cues. It is what makes "restyle
//! every caption" and "re-run recognition" single operations.
//!
//! ## Style is written through, not referenced
//!
//! A group's [`CaptionStyle`] is the *last applied template*; each cue clip
//! keeps its own `TextStyle` as the render truth. So the renderer, the template
//! system, and per-cue emphasis styling all work unchanged, and "Apply to all"
//! is an explicit command rather than an implicit lookup. Each cue's
//! [`CaptionCue::style_override`] flag records that a line was hand-tuned, so a
//! group restyle can skip it (see [`CaptionStyleScope`]).

mod cue;
mod group;
mod highlight;
mod layout;
mod spec;
mod template;

pub use cue::{CaptionCue, CaptionWord, MAX_CAPTION_WORDS};
pub use group::{CaptionFileFormat, CaptionGroup, CaptionSource, CaptionStyle, CaptionStyleScope};
pub use highlight::{
    CaptionHighlight, CaptionHighlightMode, MAX_HIGHLIGHT_SCALE, MIN_HIGHLIGHT_SCALE,
};
pub use layout::{
    CaptionLayout, DEFAULT_SAFE_AREA_BOTTOM, MAX_CAPTION_CHARS_PER_LINE, MAX_CAPTION_DURATION_MS,
    MAX_CAPTION_LINES, MIN_CAPTION_CHARS_PER_LINE,
};
pub use spec::{CaptionCueSpec, CaptionGroupSpec, MAX_CAPTION_CUES};
pub use template::{
    CaptionPlateSpec, CaptionShadowSpec, CaptionStrokeSpec, CaptionTemplateSpec,
    caption_template_catalog, caption_template_spec,
};
