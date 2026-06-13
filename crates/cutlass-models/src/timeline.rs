use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::Clip;
use crate::error::ModelError;
use crate::ids::{ClipId, MarkerId, TrackId};
use crate::time::{Rational, RationalTime, resample};
use crate::track::{Track, TrackKind};

/// Fixed marker flag palette (M1 markers). Serialized by name so project
/// files stay readable; [`rgba`](Self::rgba) gives the render color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarkerColor {
    Teal,
    Blue,
    Purple,
    Pink,
    Red,
    Orange,
    Yellow,
    Green,
}

impl MarkerColor {
    pub const ALL: [MarkerColor; 8] = [
        MarkerColor::Teal,
        MarkerColor::Blue,
        MarkerColor::Purple,
        MarkerColor::Pink,
        MarkerColor::Red,
        MarkerColor::Orange,
        MarkerColor::Yellow,
        MarkerColor::Green,
    ];

    /// The color for the `index`-th marker when none was chosen explicitly:
    /// cycle through the palette so neighboring markers stay distinguishable.
    pub fn cycle(index: usize) -> Self {
        Self::ALL[index % Self::ALL.len()]
    }

    /// Render color as `[r, g, b, a]`.
    pub fn rgba(self) -> [u8; 4] {
        match self {
            MarkerColor::Teal => [0x00, 0xE5, 0xC7, 0xFF],
            MarkerColor::Blue => [0x4A, 0x9E, 0xF5, 0xFF],
            MarkerColor::Purple => [0xA7, 0x7B, 0xF5, 0xFF],
            MarkerColor::Pink => [0xF5, 0x6F, 0xC0, 0xFF],
            MarkerColor::Red => [0xF5, 0x5A, 0x5A, 0xFF],
            MarkerColor::Orange => [0xF5, 0x9A, 0x3C, 0xFF],
            MarkerColor::Yellow => [0xF0, 0xD0, 0x4A, 0xFF],
            MarkerColor::Green => [0x6F, 0xD8, 0x5E, 0xFF],
        }
    }

    /// The serialized lowercase name ("teal", "blue", …).
    pub fn name(self) -> &'static str {
        match self {
            MarkerColor::Teal => "teal",
            MarkerColor::Blue => "blue",
            MarkerColor::Purple => "purple",
            MarkerColor::Pink => "pink",
            MarkerColor::Red => "red",
            MarkerColor::Orange => "orange",
            MarkerColor::Yellow => "yellow",
            MarkerColor::Green => "green",
        }
    }
}

/// Canvas aspect-ratio presets (M1 canvas settings, the CapCut ratio list).
/// Serialized by ratio name (`"16:9"`) so project files stay readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CanvasAspect {
    /// Follow the footage: canvas shape and size derive from the largest
    /// video media on the timeline (the pre-canvas-settings behavior).
    #[default]
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "16:9")]
    Wide16x9,
    #[serde(rename = "9:16")]
    Tall9x16,
    #[serde(rename = "1:1")]
    Square1x1,
    #[serde(rename = "4:5")]
    Portrait4x5,
    #[serde(rename = "21:9")]
    Cinema21x9,
}

impl CanvasAspect {
    pub const ALL: [CanvasAspect; 6] = [
        CanvasAspect::Auto,
        CanvasAspect::Wide16x9,
        CanvasAspect::Tall9x16,
        CanvasAspect::Square1x1,
        CanvasAspect::Portrait4x5,
        CanvasAspect::Cinema21x9,
    ];

    /// `(w, h)` ratio for fixed presets; `None` follows the footage.
    pub fn ratio(self) -> Option<(u32, u32)> {
        match self {
            CanvasAspect::Auto => None,
            CanvasAspect::Wide16x9 => Some((16, 9)),
            CanvasAspect::Tall9x16 => Some((9, 16)),
            CanvasAspect::Square1x1 => Some((1, 1)),
            CanvasAspect::Portrait4x5 => Some((4, 5)),
            CanvasAspect::Cinema21x9 => Some((21, 9)),
        }
    }

    /// The serialized name (`"auto"`, `"16:9"`, …) — also the UI label and
    /// the agent-facing identifier.
    pub fn name(self) -> &'static str {
        match self {
            CanvasAspect::Auto => "auto",
            CanvasAspect::Wide16x9 => "16:9",
            CanvasAspect::Tall9x16 => "9:16",
            CanvasAspect::Square1x1 => "1:1",
            CanvasAspect::Portrait4x5 => "4:5",
            CanvasAspect::Cinema21x9 => "21:9",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }
}

/// Per-project canvas settings (M1): aspect preset + background color.
/// The default (`Auto` + black) reproduces the pre-canvas-settings render
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CanvasSettings {
    #[serde(default, skip_serializing_if = "CanvasSettings::aspect_is_auto")]
    pub aspect: CanvasAspect,
    /// Opaque canvas background, `[r, g, b]`. Layers composite over it;
    /// uncovered canvas shows it in preview and export.
    #[serde(default, skip_serializing_if = "CanvasSettings::background_is_black")]
    pub background: [u8; 3],
}

impl CanvasSettings {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    fn aspect_is_auto(aspect: &CanvasAspect) -> bool {
        *aspect == CanvasAspect::Auto
    }

    fn background_is_black(background: &[u8; 3]) -> bool {
        *background == [0, 0, 0]
    }
}

/// A named, colored anchor point on the timeline ruler (M1 markers): the
/// agent aligns edits to them, beat-sync (M8) will emit them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub id: MarkerId,
    /// Position on the timeline, stored at the timeline frame rate
    /// ([`Timeline::add_marker`] resamples on insert).
    pub tick: RationalTime,
    /// Short label shown beside the flag. May be empty (unnamed marker).
    pub name: String,
    pub color: MarkerColor,
}

impl Marker {
    /// A fresh marker with a newly allocated id.
    pub fn new(tick: RationalTime, name: impl Into<String>, color: MarkerColor) -> Self {
        Self {
            id: MarkerId::next(),
            tick,
            name: name.into(),
            color,
        }
    }
}

/// The single sequence of a [`Project`](crate::Project): an ordered stack of
/// tracks plus a clip-location index.
///
/// - `tracks` is keyed by [`TrackId`] for O(1) lookup.
/// - `order` is the z-stack from bottom (index 0) to top; the topmost enabled
///   video track wins when compositing. The UI renders it top-first, so index 0
///   shows at the *bottom* of the lane list.
/// - **Audio-floor invariant:** every [`TrackKind::Audio`] lane sits below every
///   visual lane in `order`, so audio always renders at the bottom of the
///   timeline (CapCut behavior). Maintained by every add/insert/restore path.
/// - `clip_index` maps every [`ClipId`] to the track containing it, so a clip
///   can be found across the whole timeline in O(1) without scanning tracks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Editing/playback frame rate. Clip `timeline` ranges are in these frames.
    pub frame_rate: Rational,
    #[serde(with = "crate::serde_map")]
    tracks: Map<TrackId, Track>,
    order: Vec<TrackId>,
    #[serde(with = "crate::serde_map")]
    clip_index: Map<ClipId, TrackId>,
    /// Ruler markers in `(tick, id)` order. Optional + defaulted so pre-marker
    /// project files load unchanged and marker-free saves stay byte-identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    markers: Vec<Marker>,
    /// Canvas settings (M1): aspect preset + background color. Optional +
    /// defaulted so pre-canvas project files load unchanged and default
    /// saves stay byte-identical.
    #[serde(default, skip_serializing_if = "CanvasSettings::is_default")]
    canvas: CanvasSettings,
}

impl Timeline {
    pub fn new(frame_rate: Rational) -> Self {
        Self {
            frame_rate,
            tracks: Map::default(),
            order: Vec::new(),
            clip_index: Map::default(),
            markers: Vec::new(),
            canvas: CanvasSettings::default(),
        }
    }

    // --- canvas -------------------------------------------------------------

    /// Canvas settings: aspect preset + background color.
    pub fn canvas(&self) -> CanvasSettings {
        self.canvas
    }

    pub fn set_canvas(&mut self, settings: CanvasSettings) {
        self.canvas = settings;
    }

    // --- tracks -----------------------------------------------------------

    /// Append a track to the top of the stack. Returns its [`TrackId`].
    ///
    /// Audio lanes are sunk to the bottom of the stack to keep the audio-floor
    /// invariant (see [`Timeline`]), so a new audio track lands at the top of
    /// the audio block — still below every visual lane.
    pub fn add_track(&mut self, track: Track) -> TrackId {
        let id = track.id;
        self.tracks.insert(id, track);
        self.order.push(id);
        self.enforce_audio_floor();
        id
    }

    /// Insert a track at `order_index` in the stack (0 = bottom layer),
    /// clamped to the current stack height. Returns its [`TrackId`].
    ///
    /// The audio-floor invariant (see [`Timeline`]) is re-applied afterwards:
    /// a visual track requested below the audio block is lifted just above it,
    /// and an audio track requested above visual lanes sinks back down.
    pub fn insert_track(&mut self, track: Track, order_index: usize) -> TrackId {
        let id = track.id;
        self.tracks.insert(id, track);
        let idx = order_index.min(self.order.len());
        self.order.insert(idx, id);
        self.enforce_audio_floor();
        id
    }

    /// Stable-partition `order` so every audio lane sits below every visual
    /// lane (index 0 = bottom of the stack / bottom of the UI lane list).
    /// Relative order within the audio group and within the visual group is
    /// preserved, so the only movement is sinking audio under video.
    ///
    /// This is the single chokepoint that guarantees the audio-floor invariant
    /// for every add/insert/restore (and therefore for AI commands, drag-drop,
    /// and undo/redo). O(n) on the track count — a cold, per-track-edit path.
    fn enforce_audio_floor(&mut self) {
        let mut audio: Vec<TrackId> = Vec::with_capacity(self.order.len());
        let mut visual: Vec<TrackId> = Vec::with_capacity(self.order.len());
        for &id in &self.order {
            if self.tracks.get(&id).is_some_and(|t| t.kind == TrackKind::Audio) {
                audio.push(id);
            } else {
                visual.push(id);
            }
        }
        audio.extend(visual);
        self.order = audio;
    }

    pub fn track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.get(&id)
    }

    pub fn track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    /// Track IDs from bottom to top of the stack.
    pub fn order(&self) -> &[TrackId] {
        &self.order
    }

    /// Tracks in stacking order (bottom to top).
    pub fn tracks_ordered(&self) -> impl Iterator<Item = &Track> {
        self.order.iter().filter_map(move |id| self.tracks.get(id))
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Remove a track and all its clips (also purging the clip index).
    pub fn remove_track(&mut self, id: TrackId) -> Option<Track> {
        let track = self.tracks.remove(&id)?;
        self.order.retain(|t| *t != id);
        for clip in track.clips() {
            self.clip_index.remove(&clip.id);
        }
        Some(track)
    }

    /// Re-insert a removed track at its prior stack position (undo of [`remove_track`]).
    pub fn restore_track(
        &mut self,
        track: Track,
        order_index: usize,
    ) -> Result<TrackId, ModelError> {
        let id = track.id;
        if self.tracks.contains_key(&id) {
            return Err(ModelError::InvalidRange);
        }
        for clip in track.clips() {
            if self.clip_index.contains_key(&clip.id) {
                return Err(ModelError::InvalidRange);
            }
        }
        let clip_ids: Vec<ClipId> = track.clips().map(|c| c.id).collect();
        let idx = order_index.min(self.order.len());
        self.order.insert(idx, id);
        self.tracks.insert(id, track);
        self.enforce_audio_floor();
        for clip_id in clip_ids {
            self.clip_index.insert(clip_id, id);
        }
        Ok(id)
    }

    // --- clips ------------------------------------------------------------

    /// Place `clip` on `track_id`, rejecting unknown tracks and overlaps.
    pub fn add_clip(&mut self, track_id: TrackId, clip: Clip) -> Result<ClipId, ModelError> {
        let track = self
            .tracks
            .get(&track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?;
        if !track.kind.accepts_clip(&clip) {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind: track.kind,
            });
        }

        let track = self.tracks.get_mut(&track_id).expect("track exists");

        if track.has_overlap(clip.timeline, None)? {
            return Err(ModelError::Overlap(track_id));
        }

        let clip_id = clip.id;
        track.insert_clip(clip);
        self.clip_index.insert(clip_id, track_id);
        Ok(clip_id)
    }

    /// Remove a clip by ID from wherever it lives.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> Option<Clip> {
        let track_id = self.clip_index.remove(&clip_id)?;
        self.tracks.get_mut(&track_id)?.remove_clip(clip_id)
    }

    /// Find a clip by ID across all tracks in O(1).
    pub fn clip(&self, clip_id: ClipId) -> Option<&Clip> {
        let track_id = *self.clip_index.get(&clip_id)?;
        self.tracks.get(&track_id)?.clip(clip_id)
    }

    pub fn clip_mut(&mut self, clip_id: ClipId) -> Option<&mut Clip> {
        let track_id = *self.clip_index.get(&clip_id)?;
        self.tracks.get_mut(&track_id)?.clip_mut(clip_id)
    }

    /// The track that contains `clip_id`, if any.
    pub fn track_of(&self, clip_id: ClipId) -> Option<TrackId> {
        self.clip_index.get(&clip_id).copied()
    }

    pub fn clip_count(&self) -> usize {
        self.clip_index.len()
    }

    // --- markers ------------------------------------------------------------

    /// Ruler markers in `(tick, id)` order.
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    pub fn marker(&self, id: MarkerId) -> Option<&Marker> {
        self.markers.iter().find(|m| m.id == id)
    }

    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Insert a marker, keeping `(tick, id)` order. The tick is resampled to
    /// the timeline rate so every stored marker shares it. Rejects negative
    /// positions and duplicate ids (undo restores must not double-insert).
    pub fn add_marker(&mut self, mut marker: Marker) -> Result<MarkerId, ModelError> {
        if marker.tick.value < 0 || !marker.tick.rate.is_valid() {
            return Err(ModelError::InvalidRange);
        }
        if self.marker(marker.id).is_some() {
            return Err(ModelError::InvalidRange);
        }
        marker.tick = resample(marker.tick, self.frame_rate);
        let id = marker.id;
        let at = self
            .markers
            .partition_point(|m| (m.tick.value, m.id) <= (marker.tick.value, marker.id));
        self.markers.insert(at, marker);
        Ok(id)
    }

    /// Remove a marker by id, returning it for undo capture.
    pub fn remove_marker(&mut self, id: MarkerId) -> Option<Marker> {
        let index = self.markers.iter().position(|m| m.id == id)?;
        Some(self.markers.remove(index))
    }

    /// Move / rename / recolor a marker in one shot (the `SetMarker`
    /// command and its undo both funnel through here). Re-sorts on tick
    /// change; rejects unknown ids and negative positions.
    pub fn set_marker(
        &mut self,
        id: MarkerId,
        tick: RationalTime,
        name: String,
        color: MarkerColor,
    ) -> Result<(), ModelError> {
        let mut marker = self.remove_marker(id).ok_or(ModelError::UnknownMarker(id))?;
        let before = marker.clone();
        marker.tick = tick;
        marker.name = name;
        marker.color = color;
        match self.add_marker(marker) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Validation failed: put the original back untouched.
                let restored = self.add_marker(before);
                debug_assert!(restored.is_ok(), "re-inserting the removed marker");
                Err(e)
            }
        }
    }

    /// Total timeline length: the end of the last-ending clip at [`frame_rate`](Self::frame_rate).
    pub fn duration(&self) -> RationalTime {
        let tick = self
            .tracks
            .values()
            .map(Track::content_end)
            .max()
            .unwrap_or(0);
        RationalTime::new(tick, self.frame_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, Generator};
    use crate::time::TimeRange;
    use crate::track::{Track, TrackKind};

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

    fn timeline_with_track() -> (Timeline, TrackId) {
        let mut timeline = Timeline::new(R24);
        let track = timeline.add_track(Track::new(TrackKind::Adjustment, "FX"));
        (timeline, track)
    }

    // --- Timeline::new ----------------------------------------------------

    #[test]
    fn new_starts_empty_at_frame_rate() {
        let timeline = Timeline::new(R24);
        assert_eq!(timeline.frame_rate, R24);
        assert_eq!(timeline.track_count(), 0);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.order().is_empty());
        assert_eq!(timeline.duration(), rt(0));
    }

    // --- tracks -----------------------------------------------------------

    #[test]
    fn add_track_appends_visual_to_top_and_floors_audio() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
        let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));

        // Visual lanes stack bottom→top in insert order; the audio lane sinks
        // below them (index 0 = bottom of the stack / bottom of the UI).
        assert_eq!(timeline.order(), &[a1, v1, v2]);
        assert_eq!(timeline.track_count(), 3);
        assert_eq!(timeline.track(v1).unwrap().name, "V1");
        assert_eq!(timeline.track(a1).unwrap().kind, TrackKind::Audio);
    }

    #[test]
    fn audio_always_sinks_below_video_regardless_of_add_order() {
        // Interleave kinds and confirm every audio lane ends up below every
        // visual lane, with relative order preserved inside each group.
        let mut timeline = Timeline::new(R24);
        let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let a2 = timeline.add_track(Track::new(TrackKind::Audio, "A2"));
        let t1 = timeline.add_track(Track::new(TrackKind::Text, "T1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));

        assert_eq!(timeline.order(), &[a1, a2, v1, t1, v2]);
    }

    #[test]
    fn insert_track_audio_sinks_and_visual_clamps_above_audio() {
        let mut timeline = Timeline::new(R24);
        let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));

        // Requesting a visual track at the very bottom (index 0) lifts it just
        // above the audio block rather than under it.
        let v0 = timeline.insert_track(Track::new(TrackKind::Video, "V0"), 0);
        assert_eq!(timeline.order(), &[a1, v0, v1]);

        // Requesting an audio track at the top sinks back into the audio block.
        let a2 = timeline.insert_track(Track::new(TrackKind::Audio, "A2"), 99);
        assert_eq!(timeline.order(), &[a1, a2, v0, v1]);
    }

    #[test]
    fn insert_track_places_at_order_index_and_clamps() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));

        let bottom = timeline.insert_track(Track::new(TrackKind::Video, "V3"), 0);
        let middle = timeline.insert_track(Track::new(TrackKind::Video, "V4"), 2);
        let top = timeline.insert_track(Track::new(TrackKind::Video, "V5"), 99);

        assert_eq!(timeline.order(), &[bottom, v1, middle, v2, top]);
        assert_eq!(timeline.track_count(), 5);
    }

    #[test]
    fn tracks_ordered_yields_bottom_to_top() {
        let mut timeline = Timeline::new(R24);
        timeline.add_track(Track::new(TrackKind::Video, "bottom"));
        timeline.add_track(Track::new(TrackKind::Video, "top"));
        let names: Vec<&str> = timeline
            .tracks_ordered()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, ["bottom", "top"]);
    }

    #[test]
    fn track_mut_can_toggle_enabled() {
        let mut timeline = Timeline::new(R24);
        let id = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        timeline.track_mut(id).unwrap().enabled = false;
        assert!(!timeline.track(id).unwrap().enabled);
    }

    #[test]
    fn restore_track_reinserts_stack_position_and_clip_index() {
        let (mut timeline, track_id) = timeline_with_track();
        let clip_id = timeline
            .add_clip(track_id, generated_clip(0, 10))
            .expect("clip");
        let track = timeline.remove_track(track_id).expect("remove");
        assert_eq!(timeline.track_count(), 0);
        assert!(timeline.clip(clip_id).is_none());

        timeline
            .restore_track(track, 0)
            .expect("restore");
        assert_eq!(timeline.track_count(), 1);
        assert_eq!(timeline.track_of(clip_id), Some(track_id));
    }

    #[test]
    fn restore_track_keeps_audio_below_video() {
        // Undo of removing a video lane must not slip it under the audio floor.
        let mut timeline = Timeline::new(R24);
        let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
        assert_eq!(timeline.order(), &[a1, v1, v2]);

        let order_index = timeline.order().iter().position(|&t| t == v1).unwrap();
        let removed = timeline.remove_track(v1).expect("remove");
        assert_eq!(timeline.order(), &[a1, v2]);

        timeline.restore_track(removed, order_index).expect("restore");
        assert_eq!(timeline.order(), &[a1, v1, v2]);
    }

    #[test]
    fn remove_track_purges_clips_from_index() {
        let (mut timeline, track) = timeline_with_track();
        let clip = timeline
            .add_clip(track, generated_clip(0, 50))
            .unwrap();

        let removed = timeline.remove_track(track).unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(timeline.track_count(), 0);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.clip(clip).is_none());
        assert!(timeline.track_of(clip).is_none());
    }

    #[test]
    fn remove_unknown_track_returns_none() {
        let mut timeline = Timeline::new(R24);
        assert!(timeline.remove_track(TrackId::from_raw(99)).is_none());
    }

    // --- add_clip / clip index --------------------------------------------

    #[test]
    fn add_clip_registers_in_track_and_index() {
        let (mut timeline, track) = timeline_with_track();
        let clip = generated_clip(10, 40);
        let clip_id = clip.id;

        let returned = timeline.add_clip(track, clip).unwrap();
        assert_eq!(returned, clip_id);
        assert_eq!(timeline.clip_count(), 1);
        assert_eq!(timeline.track_of(clip_id), Some(track));
        assert_eq!(timeline.clip(clip_id).unwrap().timeline, tr(10, 40));
        assert_eq!(timeline.track(track).unwrap().len(), 1);
    }

    #[test]
    fn add_clip_unknown_track_errors() {
        let (mut timeline, _) = timeline_with_track();
        let missing = TrackId::from_raw(404);
        assert_eq!(
            timeline.add_clip(missing, generated_clip(0, 10)),
            Err(ModelError::UnknownTrack(missing))
        );
    }

    #[test]
    fn add_clip_rejects_overlap_on_same_track() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        assert_eq!(
            timeline.add_clip(track, generated_clip(25, 50)),
            Err(ModelError::Overlap(track))
        );
    }

    #[test]
    fn add_clip_allows_same_range_on_different_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));

        let c1 = timeline.add_clip(v1, generated_clip(0, 50)).unwrap();
        let c2 = timeline.add_clip(v2, generated_clip(0, 50)).unwrap();
        assert_ne!(c1, c2);
        assert_eq!(timeline.clip_count(), 2);
    }

    #[test]
    fn add_clip_allows_adjacent_non_overlapping_clips() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        let second = timeline.add_clip(track, generated_clip(50, 50)).unwrap();
        assert_eq!(timeline.clip_count(), 2);
        assert_eq!(timeline.clip(second).unwrap().start().value, 50);
    }

    // --- remove_clip / lookup ---------------------------------------------

    #[test]
    fn remove_clip_returns_clip_and_clears_index() {
        let (mut timeline, track) = timeline_with_track();
        let id = timeline.add_clip(track, generated_clip(0, 30)).unwrap();

        let removed = timeline.remove_clip(id).unwrap();
        assert_eq!(removed.id, id);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.clip(id).is_none());
        assert!(timeline.track_of(id).is_none());
        assert!(timeline.track(track).unwrap().is_empty());
    }

    #[test]
    fn remove_clip_unknown_returns_none() {
        let (mut timeline, _) = timeline_with_track();
        assert!(timeline.remove_clip(ClipId::from_raw(77)).is_none());
    }

    #[test]
    fn clip_mut_updates_timeline_range() {
        let (mut timeline, track) = timeline_with_track();
        let id = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

        timeline.clip_mut(id).unwrap().timeline = tr(10, 40);
        assert_eq!(timeline.clip(id).unwrap().timeline, tr(10, 40));
    }

    #[test]
    fn clip_lookup_finds_across_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));
        let on_v2 = timeline.add_clip(v2, generated_clip(100, 20)).unwrap();
        timeline.add_clip(v1, generated_clip(0, 10)).unwrap();

        assert_eq!(timeline.track_of(on_v2), Some(v2));
        assert_eq!(timeline.clip(on_v2).unwrap().start().value, 100);
    }

    // --- duration ---------------------------------------------------------

    #[test]
    fn duration_empty_timeline_is_zero() {
        let timeline = Timeline::new(R24);
        assert_eq!(timeline.duration(), rt(0));
    }

    #[test]
    fn duration_is_max_end_across_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Adjustment, "FX2"));
        timeline.add_clip(v1, generated_clip(0, 100)).unwrap();
        timeline.add_clip(v2, generated_clip(50, 200)).unwrap(); // ends at 250

        assert_eq!(timeline.duration().value, 250);
        assert_eq!(timeline.duration().rate, R24);
    }

    #[test]
    fn duration_ignores_gap_between_clips_on_same_track() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        timeline.add_clip(track, generated_clip(100, 30)).unwrap(); // ends 130

        assert_eq!(timeline.duration().value, 130);
    }

    // --- markers ------------------------------------------------------------

    fn marker_at(tick: i64) -> Marker {
        Marker::new(rt(tick), "", MarkerColor::Teal)
    }

    #[test]
    fn add_marker_keeps_tick_order_and_resamples() {
        let mut timeline = Timeline::new(R24);
        let late = timeline.add_marker(marker_at(100)).unwrap();
        let early = timeline.add_marker(marker_at(10)).unwrap();
        // 2 s at 48 ticks/s resamples to 24 ticks at the 24 fps timeline.
        let resampled = timeline
            .add_marker(Marker::new(
                RationalTime::new(96, Rational::new(48, 1)),
                "beat",
                MarkerColor::Red,
            ))
            .unwrap();

        let order: Vec<MarkerId> = timeline.markers().iter().map(|m| m.id).collect();
        assert_eq!(order, [early, resampled, late]);
        let beat = timeline.marker(resampled).unwrap();
        assert_eq!(beat.tick, rt(48));
        assert_eq!(beat.name, "beat");
        assert_eq!(beat.color, MarkerColor::Red);
    }

    #[test]
    fn add_marker_rejects_negative_tick_and_duplicate_id() {
        let mut timeline = Timeline::new(R24);
        assert_eq!(
            timeline.add_marker(marker_at(-1)),
            Err(ModelError::InvalidRange)
        );
        let marker = marker_at(5);
        let dup = marker.clone();
        timeline.add_marker(marker).unwrap();
        assert_eq!(timeline.add_marker(dup), Err(ModelError::InvalidRange));
        assert_eq!(timeline.marker_count(), 1);
    }

    #[test]
    fn remove_marker_returns_snapshot_for_undo() {
        let mut timeline = Timeline::new(R24);
        let id = timeline
            .add_marker(Marker::new(rt(7), "drop", MarkerColor::Pink))
            .unwrap();

        let removed = timeline.remove_marker(id).unwrap();
        assert_eq!(removed.name, "drop");
        assert_eq!(timeline.marker_count(), 0);
        assert!(timeline.remove_marker(id).is_none());

        // Restoring the snapshot keeps the same id (undo of remove).
        timeline.add_marker(removed).unwrap();
        assert_eq!(timeline.marker(id).unwrap().tick, rt(7));
    }

    #[test]
    fn set_marker_moves_renames_recolors_and_resorts() {
        let mut timeline = Timeline::new(R24);
        let a = timeline.add_marker(marker_at(10)).unwrap();
        let b = timeline.add_marker(marker_at(20)).unwrap();

        timeline
            .set_marker(a, rt(30), "outro".into(), MarkerColor::Green)
            .unwrap();
        let order: Vec<MarkerId> = timeline.markers().iter().map(|m| m.id).collect();
        assert_eq!(order, [b, a], "tick change re-sorts");
        let moved = timeline.marker(a).unwrap();
        assert_eq!((moved.tick, moved.name.as_str()), (rt(30), "outro"));
        assert_eq!(moved.color, MarkerColor::Green);

        assert_eq!(
            timeline.set_marker(MarkerId::from_raw(999), rt(0), String::new(), MarkerColor::Teal),
            Err(ModelError::UnknownMarker(MarkerId::from_raw(999)))
        );
        // A rejected move leaves the marker untouched.
        assert_eq!(
            timeline.set_marker(a, rt(-5), String::new(), MarkerColor::Teal),
            Err(ModelError::InvalidRange)
        );
        assert_eq!(timeline.marker(a).unwrap().tick, rt(30));
        assert_eq!(timeline.marker(a).unwrap().name, "outro");
    }

    #[test]
    fn markers_serialize_only_when_present() {
        let mut timeline = Timeline::new(R24);
        let empty = serde_json::to_value(&timeline).unwrap();
        assert!(
            empty.get("markers").is_none(),
            "marker-free timelines serialize without the field"
        );
        // Pre-marker files (no `markers` key) deserialize to an empty list.
        let loaded: Timeline = serde_json::from_value(empty).unwrap();
        assert_eq!(loaded.marker_count(), 0);

        timeline
            .add_marker(Marker::new(rt(12), "intro", MarkerColor::Blue))
            .unwrap();
        let json = serde_json::to_value(&timeline).unwrap();
        assert_eq!(json["markers"][0]["name"], "intro");
        assert_eq!(json["markers"][0]["color"], "blue");
        let back: Timeline = serde_json::from_value(json).unwrap();
        assert_eq!(back.markers(), timeline.markers());
    }

    #[test]
    fn marker_color_cycles_through_the_palette() {
        assert_eq!(MarkerColor::cycle(0), MarkerColor::Teal);
        assert_eq!(MarkerColor::cycle(7), MarkerColor::Green);
        assert_eq!(MarkerColor::cycle(8), MarkerColor::Teal);
        for color in MarkerColor::ALL {
            assert_eq!(color.rgba()[3], 0xFF, "palette colors are opaque");
            assert!(!color.name().is_empty());
        }
    }

    // --- canvas -------------------------------------------------------------

    #[test]
    fn canvas_defaults_to_auto_black_and_round_trips() {
        let mut timeline = Timeline::new(R24);
        assert!(timeline.canvas().is_default());
        assert_eq!(timeline.canvas().aspect, CanvasAspect::Auto);
        assert_eq!(timeline.canvas().background, [0, 0, 0]);

        // Default settings serialize without the field; pre-canvas files
        // (no `canvas` key) deserialize to the default.
        let json = serde_json::to_value(&timeline).unwrap();
        assert!(json.get("canvas").is_none());
        let loaded: Timeline = serde_json::from_value(json).unwrap();
        assert!(loaded.canvas().is_default());

        timeline.set_canvas(CanvasSettings {
            aspect: CanvasAspect::Tall9x16,
            background: [20, 30, 40],
        });
        let json = serde_json::to_value(&timeline).unwrap();
        assert_eq!(json["canvas"]["aspect"], "9:16");
        assert_eq!(json["canvas"]["background"], serde_json::json!([20, 30, 40]));
        let back: Timeline = serde_json::from_value(json).unwrap();
        assert_eq!(back.canvas(), timeline.canvas());
    }

    #[test]
    fn canvas_partial_fields_serialize_independently() {
        let mut timeline = Timeline::new(R24);
        // Only the aspect set: the black background stays off the wire.
        timeline.set_canvas(CanvasSettings {
            aspect: CanvasAspect::Square1x1,
            background: [0, 0, 0],
        });
        let json = serde_json::to_value(&timeline).unwrap();
        assert_eq!(json["canvas"]["aspect"], "1:1");
        assert!(json["canvas"].get("background").is_none());

        // Only the background set: auto aspect stays off the wire.
        timeline.set_canvas(CanvasSettings {
            aspect: CanvasAspect::Auto,
            background: [255, 255, 255],
        });
        let json = serde_json::to_value(&timeline).unwrap();
        assert!(json["canvas"].get("aspect").is_none());
        assert_eq!(json["canvas"]["background"], serde_json::json!([255, 255, 255]));
        let back: Timeline = serde_json::from_value(json).unwrap();
        assert_eq!(back.canvas().background, [255, 255, 255]);
        assert_eq!(back.canvas().aspect, CanvasAspect::Auto);
    }

    #[test]
    fn canvas_aspect_names_round_trip() {
        for aspect in CanvasAspect::ALL {
            assert_eq!(CanvasAspect::from_name(aspect.name()), Some(aspect));
            match aspect.ratio() {
                None => assert_eq!(aspect, CanvasAspect::Auto),
                Some((w, h)) => {
                    assert!(w > 0 && h > 0);
                    assert_eq!(aspect.name(), format!("{w}:{h}"));
                }
            }
        }
        assert_eq!(CanvasAspect::from_name("4:3"), None);
    }

    // --- Clone ------------------------------------------------------------

    #[test]
    fn clone_is_independent_snapshot() {
        let (mut timeline, track) = timeline_with_track();
        let clip = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

        let mut cloned = timeline.clone();
        cloned.remove_clip(clip);
        assert_eq!(cloned.clip_count(), 0);
        assert_eq!(timeline.clip_count(), 1);
    }
}
