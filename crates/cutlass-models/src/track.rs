use crate::Map;
use crate::clip::Clip;
use crate::ids::{ClipId, TrackId};
use crate::time::TimeRange;

/// Whether a track carries picture or sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Video,
    Audio,
}

/// A single lane of the timeline holding non-overlapping [`Clip`]s.
///
/// Clips are stored in a hash map keyed by [`ClipId`] for O(1) lookup. Order is
/// not stored; call [`clips_ordered`](Track::clips_ordered) to iterate by start
/// time. Overlap is enforced by the [`Timeline`](crate::Timeline) on insert.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: TrackId,
    pub kind: TrackKind,
    pub name: String,
    /// Video: whether the track contributes to the composite. Audio: unused.
    pub enabled: bool,
    /// Audio: whether the track is silenced. Video: unused.
    pub muted: bool,
    clips: Map<ClipId, Clip>,
}

impl Track {
    /// Create a track with a freshly allocated [`TrackId`].
    pub fn new(kind: TrackKind, name: impl Into<String>) -> Self {
        Self {
            id: TrackId::next(),
            kind,
            name: name.into(),
            enabled: true,
            muted: false,
            clips: Map::default(),
        }
    }

    pub fn clip(&self, id: ClipId) -> Option<&Clip> {
        self.clips.get(&id)
    }

    pub fn clip_mut(&mut self, id: ClipId) -> Option<&mut Clip> {
        self.clips.get_mut(&id)
    }

    pub fn clips(&self) -> impl Iterator<Item = &Clip> {
        self.clips.values()
    }

    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Clips sorted by their timeline start frame (ties broken by `ClipId`).
    pub fn clips_ordered(&self) -> Vec<&Clip> {
        let mut v: Vec<&Clip> = self.clips.values().collect();
        v.sort_by_key(|c| (c.timeline.start, c.id));
        v
    }

    /// The clip occupying `timeline_frame`, if any. (At most one, since clips on
    /// a track never overlap.)
    pub fn clip_at(&self, timeline_frame: i64) -> Option<&Clip> {
        self.clips
            .values()
            .find(|c| c.timeline.contains(timeline_frame))
    }

    /// Whether `range` would collide with any existing clip, optionally
    /// ignoring one clip (useful when re-placing an existing clip).
    pub fn has_overlap(&self, range: TimeRange, ignore: Option<ClipId>) -> bool {
        self.clips
            .values()
            .filter(|c| Some(c.id) != ignore)
            .any(|c| c.timeline.overlaps(range))
    }

    /// Exclusive end frame of the last clip (0 if empty).
    pub fn content_end(&self) -> i64 {
        self.clips.values().map(|c| c.end()).max().unwrap_or(0)
    }

    /// Insert without overlap checking. Returns the displaced clip, if any.
    /// Prefer [`Timeline::add_clip`](crate::Timeline::add_clip) which validates.
    pub(crate) fn insert_clip(&mut self, clip: Clip) -> Option<Clip> {
        self.clips.insert(clip.id, clip)
    }

    pub(crate) fn remove_clip(&mut self, id: ClipId) -> Option<Clip> {
        self.clips.remove(&id)
    }
}
