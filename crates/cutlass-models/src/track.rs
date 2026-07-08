use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::{Clip, ClipSource, Generator};
use crate::error::ModelError;
use crate::ids::{ClipId, TrackId};
use crate::time::{RationalTime, TimeRange};
use crate::transition::Transition;

/// Lane category on the timeline. Drives drag targeting, clip placement rules,
/// and compositor participation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackKind {
    /// Footage and other imported picture media.
    Video,
    /// Imported sound media.
    Audio,
    /// Titles and captions.
    Text,
    /// Stickers, shapes, and other graphic overlays.
    Sticker,
    /// Motion and composited effects.
    Effect,
    /// Blur, mask, and similar filters.
    Filter,
    /// Color grade / adjustment layers.
    Adjustment,
}

impl TrackKind {
    /// Picture stack lanes (excludes audio).
    pub const fn is_visual(self) -> bool {
        !matches!(self, Self::Audio)
    }

    /// Whether `content` may be placed on a track of this kind.
    pub fn accepts_content(self, content: &ClipSource) -> bool {
        matches!(
            (self, content),
            (Self::Video | Self::Audio, ClipSource::Media { .. })
                | (Self::Text, ClipSource::Generated(Generator::Text { .. }))
                | (
                    Self::Sticker,
                    ClipSource::Generated(
                        Generator::Sticker { .. }
                            | Generator::SolidColor { .. }
                            | Generator::Shape { .. },
                    ),
                )
                | (Self::Effect, ClipSource::Generated(Generator::Effect))
                | (Self::Filter, ClipSource::Generated(Generator::Filter))
                | (
                    Self::Adjustment,
                    ClipSource::Generated(Generator::Adjustment)
                )
        )
    }

    /// Whether `clip` may be placed on a track of this kind.
    pub fn accepts_clip(self, clip: &Clip) -> bool {
        self.accepts_content(&clip.content)
    }

    /// Track kind required for a generated clip variant.
    pub const fn for_generator(generator: &Generator) -> Option<Self> {
        match generator {
            Generator::Text { .. } => Some(Self::Text),
            Generator::SolidColor { .. } | Generator::Shape { .. } | Generator::Sticker { .. } => {
                Some(Self::Sticker)
            }
            Generator::Effect => Some(Self::Effect),
            Generator::Filter => Some(Self::Filter),
            Generator::Adjustment => Some(Self::Adjustment),
        }
    }
}

/// A single lane of the timeline holding non-overlapping [`Clip`]s.
///
/// Clips are stored in a hash map keyed by [`ClipId`] for O(1) lookup. Order is
/// not stored; call [`clips_ordered`](Track::clips_ordered) to iterate by start
/// time. Overlap is enforced by the [`Timeline`](crate::Timeline) on insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: TrackId,
    pub kind: TrackKind,
    pub name: String,
    /// Video: whether the track contributes to the composite. Audio: unused.
    pub enabled: bool,
    /// Audio: whether the track is silenced. Video: unused.
    pub muted: bool,
    /// Whether the lane is locked: its clips can't be selected, moved, or
    /// trimmed. Compositing/playback are unaffected (CapCut semantics).
    #[serde(default)]
    pub locked: bool,
    /// When true, an empty lane is not auto-removed by implicit cleanup —
    /// only an explicit user delete removes it (manual "Add track").
    #[serde(default)]
    pub pinned: bool,
    /// Audio: whether this lane is a sidechain "voice" source — its clips
    /// drive audio ducking (M8 Phase 4), so a "duck under voice" gesture dips
    /// music under the talkers on every lane flagged here. Additive: old
    /// projects load with none. Inert on non-audio lanes.
    #[serde(default)]
    pub duck_source: bool,
    #[serde(with = "crate::serde_map")]
    clips: Map<ClipId, Clip>,
    /// Transitions at clip junctions, keyed by the left (outgoing) clip id.
    /// Additive (M4): old projects load with none. A junction lives only
    /// while its pair still abuts; structural edits prune dead ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    transitions: Vec<Transition>,
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
            locked: false,
            pinned: false,
            duck_source: false,
            clips: Map::default(),
            transitions: Vec::new(),
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

    /// Mutable iteration over the track's clips (unordered). Used by ripple
    /// edits that shift many clips at once; callers must not introduce overlaps.
    pub fn clips_mut(&mut self) -> impl Iterator<Item = &mut Clip> {
        self.clips.values_mut()
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
        v.sort_by_key(|c| (c.timeline.start.value, c.id));
        v
    }

    /// The clip occupying `timeline_pos`, if any. (At most one, since clips on a
    /// track never overlap.)
    pub fn clip_at(&self, timeline_pos: RationalTime) -> Result<Option<&Clip>, ModelError> {
        for clip in self.clips.values() {
            if clip.timeline.contains(timeline_pos)? {
                return Ok(Some(clip));
            }
        }
        Ok(None)
    }

    /// Whether `range` would collide with any existing clip, optionally
    /// ignoring one clip (useful when re-placing an existing clip).
    pub fn has_overlap(
        &self,
        range: TimeRange,
        ignore: Option<ClipId>,
    ) -> Result<bool, ModelError> {
        for clip in self.clips.values().filter(|c| Some(c.id) != ignore) {
            if clip.timeline.overlaps(range)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Exclusive end tick of the last clip (0 if empty), at the timeline rate.
    pub fn content_end(&self) -> i64 {
        self.clips
            .values()
            .map(|c| c.timeline.end_tick())
            .max()
            .unwrap_or(0)
    }

    /// Insert without overlap checking. Returns the displaced clip, if any.
    /// Prefer [`Timeline::add_clip`](crate::Timeline::add_clip) which validates.
    pub(crate) fn insert_clip(&mut self, clip: Clip) -> Option<Clip> {
        self.clips.insert(clip.id, clip)
    }

    pub(crate) fn remove_clip(&mut self, id: ClipId) -> Option<Clip> {
        self.clips.remove(&id)
    }

    // --- transitions (M4) -------------------------------------------------

    /// All transitions on this track (junction order is not guaranteed).
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// Replace the whole transition set (used by undo/prune restore).
    pub(crate) fn set_transitions(&mut self, transitions: Vec<Transition>) {
        self.transitions = transitions;
    }

    /// The transition whose outgoing side is `left`, if any.
    pub fn transition_at(&self, left: ClipId) -> Option<&Transition> {
        self.transitions.iter().find(|t| t.left == left)
    }

    pub fn transition_at_mut(&mut self, left: ClipId) -> Option<&mut Transition> {
        self.transitions.iter_mut().find(|t| t.left == left)
    }

    /// Add or replace the transition at the `left` junction.
    pub(crate) fn upsert_transition(&mut self, transition: Transition) {
        if let Some(existing) = self.transition_at_mut(transition.left) {
            *existing = transition;
        } else {
            self.transitions.push(transition);
        }
    }

    /// Remove the transition at the `left` junction, returning it if present.
    pub(crate) fn remove_transition(&mut self, left: ClipId) -> Option<Transition> {
        let idx = self.transitions.iter().position(|t| t.left == left)?;
        Some(self.transitions.remove(idx))
    }

    /// Whether `left` and `right` abut on this track: `left` ends exactly
    /// where `right` starts and both are present.
    pub fn clips_abut(&self, left: ClipId, right: ClipId) -> bool {
        match (self.clip(left), self.clip(right)) {
            (Some(l), Some(r)) => l.timeline.end_tick() == r.timeline.start.value,
            _ => false,
        }
    }

    /// Drop transitions whose junction no longer abuts (a clip moved, trimmed,
    /// split, or was removed). Returns whether anything was pruned.
    pub(crate) fn prune_dead_transitions(&mut self) -> bool {
        let before = self.transitions.len();
        // Borrow-checker dance: collect survivors against an immutable view.
        let survivors: Vec<Transition> = self
            .transitions
            .iter()
            .filter(|t| self.clips_abut(t.left, t.right))
            .cloned()
            .collect();
        if survivors.len() != before {
            self.transitions = survivors;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, ClipSource, Generator};
    use crate::time::Rational;

    const R24: Rational = Rational::FPS_24;

    fn rt(value: i64) -> RationalTime {
        RationalTime::new(value, R24)
    }

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    fn generated_clip(start: i64, duration: i64) -> Clip {
        Clip::generated(Generator::Adjustment, tr(start, duration))
    }

    fn video_track(name: &str) -> Track {
        Track::new(TrackKind::Video, name)
    }

    // --- Track::new -------------------------------------------------------

    #[test]
    fn new_wires_kind_name_and_defaults() {
        let track = video_track("V1");
        assert_eq!(track.kind, TrackKind::Video);
        assert_eq!(track.name, "V1");
        assert!(track.enabled);
        assert!(!track.muted);
        assert!(!track.locked);
        assert!(!track.duck_source);
        assert!(track.is_empty());
        assert!(track.id.raw() >= 1);
    }

    #[test]
    fn new_accepts_owned_string_name() {
        let track = Track::new(TrackKind::Audio, String::from("A1"));
        assert_eq!(track.kind, TrackKind::Audio);
        assert_eq!(track.name, "A1");
    }

    #[test]
    fn new_assigns_distinct_track_ids() {
        let a = video_track("A");
        let b = video_track("B");
        assert_ne!(a.id, b.id);
    }

    // --- insert / remove / lookup -----------------------------------------

    #[test]
    fn insert_and_remove_clip() {
        let mut track = video_track("V1");
        let clip = generated_clip(0, 50);
        let id = clip.id;

        assert!(track.insert_clip(clip).is_none());
        assert_eq!(track.len(), 1);
        assert_eq!(track.clip(id).unwrap().timeline, tr(0, 50));

        let removed = track.remove_clip(id).unwrap();
        assert_eq!(removed.id, id);
        assert!(track.is_empty());
        assert!(track.clip(id).is_none());
    }

    #[test]
    fn insert_replaces_same_id() {
        let mut track = video_track("V1");
        let first = generated_clip(0, 10);
        let id = first.id;
        assert!(track.insert_clip(first).is_none());

        let replacement = Clip {
            id,
            content: ClipSource::Generated(Generator::Adjustment),
            timeline: tr(20, 30),
            ..generated_clip(20, 30)
        };
        let displaced = track.insert_clip(replacement).unwrap();
        assert_eq!(displaced.timeline, tr(0, 10));
        assert_eq!(track.clip(id).unwrap().timeline, tr(20, 30));
        assert_eq!(track.len(), 1);
    }

    #[test]
    fn clip_mut_updates_timeline() {
        let mut track = video_track("V1");
        let clip = generated_clip(0, 50);
        let id = clip.id;
        track.insert_clip(clip);
        track.clip_mut(id).unwrap().timeline = tr(10, 40);
        assert_eq!(track.clip(id).unwrap().timeline, tr(10, 40));
    }

    #[test]
    fn clips_and_clips_mut_iterate_all() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 10));
        track.insert_clip(generated_clip(20, 10));
        assert_eq!(track.clips().count(), 2);

        for clip in track.clips_mut() {
            clip.timeline = tr(clip.timeline.start.value + 1, clip.timeline.duration.value);
        }
        // `clips()` iterates the backing hash map — unordered by contract
        // (FxHash order shifts with the globally allocated clip ids).
        let mut starts: Vec<i64> = track.clips().map(|c| c.start().value).collect();
        starts.sort_unstable();
        assert_eq!(starts, vec![1, 21]);
    }

    // --- clips_ordered ----------------------------------------------------

    #[test]
    fn clips_ordered_sorts_by_start_then_id() {
        let mut track = video_track("V1");
        let late = generated_clip(100, 10);
        let early = generated_clip(0, 10);
        let early_id = early.id;
        let late_id = late.id;
        track.insert_clip(late);
        track.insert_clip(early);

        let ordered: Vec<ClipId> = track.clips_ordered().iter().map(|c| c.id).collect();
        assert_eq!(ordered, vec![early_id, late_id]);
    }

    #[test]
    fn clips_ordered_breaks_start_ties_by_clip_id() {
        let mut track = video_track("V1");
        let a = Clip {
            id: ClipId::from_raw(2),
            ..generated_clip(10, 5)
        };
        let b = Clip {
            id: ClipId::from_raw(1),
            ..generated_clip(10, 5)
        };
        track.insert_clip(a);
        track.insert_clip(b);

        let ordered: Vec<ClipId> = track.clips_ordered().iter().map(|c| c.id).collect();
        assert_eq!(ordered, vec![ClipId::from_raw(1), ClipId::from_raw(2)]);
    }

    // --- clip_at ----------------------------------------------------------

    #[test]
    fn clip_at_finds_occupant() {
        let mut track = video_track("V1");
        let clip = generated_clip(10, 20);
        let id = clip.id;
        track.insert_clip(clip);

        assert_eq!(track.clip_at(rt(10)).unwrap().map(|c| c.id), Some(id));
        assert_eq!(track.clip_at(rt(29)).unwrap().map(|c| c.id), Some(id));
        assert!(track.clip_at(rt(30)).unwrap().is_none());
        assert!(track.clip_at(rt(9)).unwrap().is_none());
    }

    #[test]
    fn clip_at_rate_mismatch_errors() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 10));
        assert_eq!(
            track
                .clip_at(RationalTime::new(5, Rational::FPS_30))
                .unwrap_err(),
            ModelError::RateMismatch {
                expected: Rational::FPS_30,
                got: R24,
            }
        );
    }

    // --- has_overlap ------------------------------------------------------

    #[test]
    fn has_overlap_detects_partial_overlap() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 50));
        assert!(track.has_overlap(tr(25, 50), None).unwrap());
    }

    #[test]
    fn has_overlap_touching_ranges_do_not_overlap() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 50));
        assert!(!track.has_overlap(tr(50, 50), None).unwrap());
        assert!(track.has_overlap(tr(0, 50), None).unwrap()); // identical placement collides
    }

    #[test]
    fn has_overlap_ignore_skips_clip() {
        let mut track = video_track("V1");
        let clip = generated_clip(0, 50);
        let id = clip.id;
        track.insert_clip(clip);
        assert!(!track.has_overlap(tr(0, 50), Some(id)).unwrap());
        assert!(track.has_overlap(tr(0, 50), None).unwrap());
    }

    #[test]
    fn has_overlap_rate_mismatch_errors() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 50));
        let bad = TimeRange::at_rate(0, 10, Rational::FPS_30);
        assert_eq!(
            track.has_overlap(bad, None).unwrap_err(),
            ModelError::RateMismatch {
                expected: Rational::FPS_30,
                got: R24,
            }
        );
    }

    // --- content_end ------------------------------------------------------

    #[test]
    fn content_end_empty_is_zero() {
        assert_eq!(video_track("V1").content_end(), 0);
    }

    #[test]
    fn content_end_is_max_exclusive_end() {
        let mut track = video_track("V1");
        track.insert_clip(generated_clip(0, 50));
        track.insert_clip(generated_clip(100, 30));
        assert_eq!(track.content_end(), 130);
    }

    // --- flags / Clone ----------------------------------------------------

    #[test]
    fn enabled_muted_and_locked_are_mutable() {
        let mut track = Track::new(TrackKind::Audio, "A1");
        track.enabled = false;
        track.muted = true;
        track.locked = true;
        track.duck_source = true;
        assert!(!track.enabled);
        assert!(track.muted);
        assert!(track.locked);
        assert!(track.duck_source);
    }

    #[test]
    fn duck_source_defaults_false_for_back_compat() {
        // A pre-Phase-4 track JSON (no duck_source field) loads as not a
        // voice source.
        let json = serde_json::json!({
            "id": 1,
            "kind": "Audio",
            "name": "A1",
            "enabled": true,
            "muted": false,
            "clips": []
        });
        let track: Track = serde_json::from_value(json).unwrap();
        assert!(!track.duck_source);
    }

    #[test]
    fn clone_preserves_clips_and_metadata() {
        let mut track = video_track("V1");
        let clip = generated_clip(0, 25);
        let id = clip.id;
        track.insert_clip(clip);
        let cloned = track.clone();
        assert_eq!(cloned.name, "V1");
        assert_eq!(cloned.clip(id).unwrap().timeline, tr(0, 25));
        assert_eq!(cloned.len(), 1);
    }

    #[test]
    fn track_kind_equality() {
        assert_eq!(TrackKind::Video, TrackKind::Video);
        assert_ne!(TrackKind::Video, TrackKind::Audio);
        assert_ne!(TrackKind::Text, TrackKind::Sticker);
    }

    #[test]
    fn track_kind_accepts_clip_by_lane() {
        use crate::clip::{Clip, Generator};

        let media = Clip::from_media(crate::ids::MediaId::next(), tr(0, 10), tr(0, 10));
        let text = Clip::generated(Generator::text("hi"), tr(0, 10));
        let sticker = Clip::generated(Generator::SolidColor { rgba: [1, 2, 3, 4] }, tr(0, 10));
        let adj = Clip::generated(Generator::Adjustment, tr(0, 10));

        assert!(TrackKind::Video.accepts_clip(&media));
        assert!(!TrackKind::Video.accepts_clip(&text));
        assert!(TrackKind::Text.accepts_clip(&text));
        assert!(!TrackKind::Text.accepts_clip(&sticker));
        assert!(TrackKind::Sticker.accepts_clip(&sticker));
        assert!(TrackKind::Adjustment.accepts_clip(&adj));
        assert!(!TrackKind::Adjustment.accepts_clip(&text));
        assert!(TrackKind::Audio.accepts_clip(&media));
    }
}
