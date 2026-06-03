use crate::ids::{ClipId, MediaId};
use crate::time::TimeRange;

/// What a clip draws. Either a trimmed range of imported media, or synthetic
/// content rendered by the engine (text, shapes, solids, ...).
#[derive(Debug, Clone, PartialEq)]
pub enum ClipSource {
    /// A trimmed portion of a [`MediaSource`](crate::MediaSource).
    ///
    /// `source` is the in/out within the media, in **source frames**. It is
    /// part of this variant because it is meaningless for generated content.
    Media { media: MediaId, source: TimeRange },
    /// Engine-generated content with no backing file.
    Generated(Generator),
}

/// A synthetic clip with no source media. Parameters are intentionally minimal
/// for now; richer styling (fonts, transforms, gradients) can be added per
/// variant without touching the timeline model.
#[derive(Debug, Clone, PartialEq)]
pub enum Generator {
    /// A title / text layer.
    Text { content: String },
    /// A solid fill (RGBA, 0-255).
    SolidColor { rgba: [u8; 4] },
    /// A vector shape.
    Shape { shape: Shape },
    /// A pass-through layer that only affects tracks beneath it.
    Adjustment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Rectangle,
    Ellipse,
}

/// A placement of some [`ClipSource`] on a track.
///
/// Every clip has a `timeline` range (where it sits, in **timeline frames**),
/// regardless of whether it is media-backed or generated.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub id: ClipId,
    pub content: ClipSource,
    pub timeline: TimeRange,
}

impl Clip {
    /// A clip backed by a trimmed range of imported media.
    pub fn from_media(media: MediaId, source: TimeRange, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Media { media, source },
            timeline,
        }
    }

    /// A generated clip (text, shape, solid, ...).
    pub fn generated(generator: Generator, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Generated(generator),
            timeline,
        }
    }

    /// First timeline frame occupied by the clip.
    pub fn start(&self) -> i64 {
        self.timeline.start
    }

    /// Exclusive last timeline frame.
    pub fn end(&self) -> i64 {
        self.timeline.end()
    }

    /// The media this clip references, or `None` for generated content.
    pub fn media(&self) -> Option<MediaId> {
        match &self.content {
            ClipSource::Media { media, .. } => Some(*media),
            ClipSource::Generated(_) => None,
        }
    }

    /// The source in/out range, or `None` for generated content.
    pub fn source_range(&self) -> Option<TimeRange> {
        match &self.content {
            ClipSource::Media { source, .. } => Some(*source),
            ClipSource::Generated(_) => None,
        }
    }

    pub fn is_generated(&self) -> bool {
        matches!(self.content, ClipSource::Generated(_))
    }

    /// Map a timeline frame to the corresponding source frame, for media clips.
    /// Returns `None` if the frame is outside the clip or the clip is generated.
    pub fn source_frame_at(&self, timeline_frame: i64) -> Option<i64> {
        if !self.timeline.contains(timeline_frame) {
            return None;
        }
        match &self.content {
            ClipSource::Media { source, .. } => {
                let offset = timeline_frame - self.timeline.start;
                Some(source.start + offset)
            }
            ClipSource::Generated(_) => None,
        }
    }
}
