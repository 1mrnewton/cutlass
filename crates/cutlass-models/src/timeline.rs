use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::Clip;
use crate::error::ModelError;
use crate::ids::{ClipId, MarkerId, TrackId};
use crate::time::{Rational, RationalTime, resample};
use crate::track::{Track, TrackKind};

/// Marker flag color (M1 markers). The eight named presets serialize as
/// lowercase strings (`"teal"`, …) so existing projects keep working;
/// [`Rgba`](Self::Rgba) serializes as `{"rgba":[r,g,b,a]}` (markers are
/// UI chrome — callers should keep alpha opaque).
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
    /// Custom color. Alpha is stored for serde symmetry with other RGBA
    /// fields; the desktop UI forces opaque (`0xFF`).
    Rgba([u8; 4]),
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

    /// Opaque custom color (alpha forced to `0xFF`).
    pub fn custom(r: u8, g: u8, b: u8) -> Self {
        MarkerColor::Rgba([r, g, b, 0xFF])
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
            MarkerColor::Rgba(rgba) => rgba,
        }
    }

    /// The serialized lowercase name for a palette preset (`"teal"`, …).
    /// `None` for [`Rgba`](Self::Rgba) — use [`token`](Self::token) for a
    /// round-trippable wire string.
    pub fn name(self) -> Option<&'static str> {
        match self {
            MarkerColor::Teal => Some("teal"),
            MarkerColor::Blue => Some("blue"),
            MarkerColor::Purple => Some("purple"),
            MarkerColor::Pink => Some("pink"),
            MarkerColor::Red => Some("red"),
            MarkerColor::Orange => Some("orange"),
            MarkerColor::Yellow => Some("yellow"),
            MarkerColor::Green => Some("green"),
            MarkerColor::Rgba(_) => None,
        }
    }

    /// Desktop / command wire token: palette name, or `#RRGGBB` for custom
    /// (opaque; alpha discarded).
    pub fn token(self) -> String {
        match self {
            MarkerColor::Rgba([r, g, b, _]) => format!("#{r:02X}{g:02X}{b:02X}"),
            preset => preset
                .name()
                .expect("palette colors always have a name")
                .to_string(),
        }
    }

    /// Inverse of [`token`](Self::token): palette names or `#RGB` /
    /// `#RRGGBB` / `#RRGGBBAA` (leading `#` optional). Custom parses force
    /// opaque alpha.
    pub fn parse_token(s: &str) -> Option<Self> {
        match s {
            "teal" => Some(MarkerColor::Teal),
            "blue" => Some(MarkerColor::Blue),
            "purple" => Some(MarkerColor::Purple),
            "pink" => Some(MarkerColor::Pink),
            "red" => Some(MarkerColor::Red),
            "orange" => Some(MarkerColor::Orange),
            "yellow" => Some(MarkerColor::Yellow),
            "green" => Some(MarkerColor::Green),
            other => parse_marker_hex(other).map(|[r, g, b]| MarkerColor::custom(r, g, b)),
        }
    }
}

/// Parse `#RGB` / `#RRGGBB` / `#RRGGBBAA` (leading `#` optional) into opaque
/// RGB. Invalid lengths / non-hex → `None`. Alpha in 8-digit form is ignored
/// (markers are UI chrome).
fn parse_marker_hex(input: &str) -> Option<[u8; 3]> {
    let s = input.trim();
    let s = s.strip_prefix('#').unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    match s.len() {
        3 => {
            let r = marker_nibble(s.as_bytes()[0])? * 0x11;
            let g = marker_nibble(s.as_bytes()[1])? * 0x11;
            let b = marker_nibble(s.as_bytes()[2])? * 0x11;
            Some([r, g, b])
        }
        6 | 8 => {
            let r = u8::from_str_radix(&s[0..2], 16).ok()?;
            let g = u8::from_str_radix(&s[2..4], 16).ok()?;
            let b = u8::from_str_radix(&s[4..6], 16).ok()?;
            Some([r, g, b])
        }
        _ => None,
    }
}

fn marker_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
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
/// - **Lane-zone invariant (CapCut):** `order` is always partitioned into four
///   zones, bottom to top: every [`TrackKind::Audio`] lane, then the main
///   video track ([`Track::main`]), then every other visual lane (overlay
///   video / sticker / effect / filter / adjustment), then every
///   [`TrackKind::Text`] lane. So audio renders at the bottom, text on top,
///   and nothing but audio ever sits below the main track. Maintained by
///   every add/insert/move/restore path.
/// - **Main-track invariant:** at most one track is flagged [`Track::main`],
///   it is always a video lane, and one exists whenever any video lane does
///   (the first video track added is designated automatically; removing the
///   main lane promotes the next bottom-most video lane).
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
    /// The lane-zone invariant (see [`Timeline`]) is re-applied afterwards,
    /// so an audio lane sinks to the audio floor, a text lane rises to the
    /// top, and the first video lane ever added becomes the main track and
    /// lands directly above the audio block.
    pub fn add_track(&mut self, track: Track) -> TrackId {
        let id = track.id;
        self.tracks.insert(id, track);
        self.order.push(id);
        self.designate_main();
        self.enforce_lane_zones();
        id
    }

    /// Insert a track at `order_index` in the stack (0 = bottom layer),
    /// clamped to the current stack height. Returns its [`TrackId`].
    ///
    /// The lane-zone invariant (see [`Timeline`]) is re-applied afterwards:
    /// the requested index only decides the lane's position *within its
    /// zone* — a visual track requested below the audio block or the main
    /// track is lifted just above them, a text lane rises above the overlay
    /// block, and an audio track requested above visual lanes sinks back
    /// down.
    pub fn insert_track(&mut self, track: Track, order_index: usize) -> TrackId {
        let id = track.id;
        self.tracks.insert(id, track);
        let idx = order_index.min(self.order.len());
        self.order.insert(idx, id);
        self.designate_main();
        self.enforce_lane_zones();
        id
    }

    /// The main track (CapCut's magnetic lane): the single video lane flagged
    /// [`Track::main`]. `None` until a video track exists.
    pub fn main_track(&self) -> Option<TrackId> {
        self.order
            .iter()
            .copied()
            .find(|id| self.tracks.get(id).is_some_and(|t| t.main))
    }

    /// Re-derive the lane-zone and main-track invariants from scratch —
    /// the chokepoint for projects loaded from disk (files written before
    /// the main-track flag existed, or edited externally).
    pub fn normalize_lanes(&mut self) {
        self.designate_main();
        self.enforce_lane_zones();
    }

    /// Keep the main-track invariant: clear the flag from non-video lanes,
    /// keep only the bottom-most flagged video lane when several claim it,
    /// and designate the bottom-most video lane when none does.
    fn designate_main(&mut self) {
        let mut seen_main = false;
        let mut first_video: Option<TrackId> = None;
        for id in self.order.clone() {
            let Some(track) = self.tracks.get_mut(&id) else {
                continue;
            };
            if track.kind != TrackKind::Video {
                track.main = false;
                continue;
            }
            if first_video.is_none() {
                first_video = Some(id);
            }
            if track.main {
                if seen_main {
                    track.main = false;
                } else {
                    seen_main = true;
                }
            }
        }
        if !seen_main && let Some(id) = first_video {
            self.tracks.get_mut(&id).expect("track exists").main = true;
        }
    }

    /// Stable-partition `order` into the four CapCut zones (bottom to top):
    /// audio, the main video track, other visual lanes, text. Relative order
    /// *within* each zone is preserved, so the only movement is lanes sinking
    /// or rising to their zone boundary.
    ///
    /// This is the single chokepoint that guarantees the lane-zone invariant
    /// for every add/insert/move/restore (and therefore for AI commands,
    /// drag-drop, and undo/redo). O(n log n) on the track count — a cold,
    /// per-track-edit path.
    fn enforce_lane_zones(&mut self) {
        let zone = |id: &TrackId| -> u8 {
            match self.tracks.get(id) {
                Some(t) if t.kind == TrackKind::Audio => 0,
                Some(t) if t.main => 1,
                Some(t) if t.kind == TrackKind::Text => 3,
                _ => 2,
            }
        };
        // Vec::sort_by_key is stable: ties keep their relative order.
        let mut order = std::mem::take(&mut self.order);
        order.sort_by_key(zone);
        self.order = order;
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
    /// Removing the main track promotes the next bottom-most video lane.
    pub fn remove_track(&mut self, id: TrackId) -> Option<Track> {
        let track = self.tracks.remove(&id)?;
        self.order.retain(|t| *t != id);
        for clip in track.clips() {
            self.clip_index.remove(&clip.id);
        }
        if track.main {
            self.designate_main();
            self.enforce_lane_zones();
        }
        Some(track)
    }

    /// Reorder a track within the stack. The lane-zone invariant is
    /// re-applied afterwards, so the move only takes effect within the
    /// track's zone — in particular, the main track never moves, and no
    /// visual lane can cross below it or below the audio floor.
    pub fn move_track(&mut self, id: TrackId, new_index: usize) -> Result<(), ModelError> {
        let current = self
            .order
            .iter()
            .position(|&t| t == id)
            .ok_or(ModelError::UnknownTrack(id))?;
        if current == new_index {
            return Ok(());
        }
        self.order.remove(current);
        let idx = new_index.min(self.order.len());
        self.order.insert(idx, id);
        self.enforce_lane_zones();
        Ok(())
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
        // A restored main lane may collide with an interim promotion (undo of
        // remove-main); `designate_main` keeps the bottom-most claim and
        // clears the other, so the invariant holds either way.
        self.designate_main();
        self.enforce_lane_zones();
        for clip_id in clip_ids {
            self.clip_index.insert(clip_id, id);
        }
        Ok(id)
    }

    // --- clips ------------------------------------------------------------

    /// Place `clip` on `track_id`, rejecting unknown tracks, duplicate ids,
    /// and overlaps.
    pub fn add_clip(&mut self, track_id: TrackId, clip: Clip) -> Result<ClipId, ModelError> {
        // Defense in depth against id collisions: a plain map insert would
        // silently overwrite an existing clip (and its index entry), which is
        // how a cross-session id clash used to destroy a clip. Reject instead.
        if self.clip_index.contains_key(&clip.id) {
            return Err(ModelError::DuplicateClip(clip.id));
        }
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

    /// Whether `clip_id`'s media audio is heard from its *own* lane.
    ///
    /// CapCut keeps a video's sound on the video clip itself; a dropped clip
    /// is audible without a second lane. `true` for audio-lane media clips and
    /// for video-lane media clips, with two exceptions: freeze-frame clips are
    /// semantically silent, and a video clip whose sound has been *detached*
    /// to a linked audio-lane partner (CapCut "separate audio") goes silent on
    /// its own lane — the partner carries it instead. `false` for lanes that
    /// never carry media audio (text/effect/etc.).
    ///
    /// Callers still layer mute / silence / `has_audio` on top; this answers
    /// only "does this clip's media audio belong to *this* lane?". The detach
    /// scan only runs for *linked* video clips, so undetached drops (the common
    /// case, `link == None`) short-circuit.
    pub fn carries_own_audio(&self, clip_id: ClipId) -> bool {
        if self.clip(clip_id).is_none_or(|clip| clip.freeze_frame) {
            return false;
        }
        let Some(track) = self.track_of(clip_id).and_then(|t| self.track(t)) else {
            return false;
        };
        match track.kind {
            TrackKind::Audio => true,
            TrackKind::Video => !self.detached_to_audio_lane(clip_id),
            _ => false,
        }
    }

    /// Whether `clip_id` shares a link group with a clip on an audio lane — the
    /// CapCut "separate audio" companion that took over its sound.
    pub fn detached_to_audio_lane(&self, clip_id: ClipId) -> bool {
        let Some(link) = self.clip(clip_id).and_then(|c| c.link) else {
            return false;
        };
        self.tracks_ordered().any(|t| {
            t.kind == TrackKind::Audio && t.clips_ordered().iter().any(|c| c.link == Some(link))
        })
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
        let mut marker = self
            .remove_marker(id)
            .ok_or(ModelError::UnknownMarker(id))?;
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
mod tests;
