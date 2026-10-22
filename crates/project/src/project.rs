//! The project: settings, assets, and one timeline.

use std::collections::BTreeSet;

use cutlass_core::Timestamp;

use crate::Error;
use crate::animation::AnimationKind;
use crate::asset::Asset;
use crate::attachment::Attachment;
use crate::command::{Command, Outcome};
use crate::id::{AssetId, ProjectId, SegmentId, TimelineId, TrackId};
use crate::segment::{Channels, Segment};
use crate::settings::{AspectRatio, Canvas, Settings};
use crate::time::{Speed, TimeRange};
use crate::timeline::Timeline;
use crate::track::{Band, Track, TrackKind};
use crate::transition::Transition;

/// Settings, assets, and one timeline. The shell loads and saves it.
#[derive(Debug, Clone)]
pub struct Project {
    id: ProjectId,
    next_id: u64,
    name: String,
    settings: Settings,
    assets: Vec<Asset>,
    timeline: Timeline,
}

impl Project {
    /// An empty project. The timeline is `timeline-1` and the main track is `track-2`.
    pub fn new(id: ProjectId, name: impl Into<String>) -> Self {
        Self {
            id,
            next_id: 3,
            name: name.into(),
            settings: Settings::default(),
            assets: Vec::new(),
            timeline: Timeline::new(TimelineId::raw(1), TrackId::raw(2)),
        }
    }

    pub(crate) fn restore(
        id: ProjectId,
        next_id: u64,
        name: String,
        settings: Settings,
        assets: Vec<Asset>,
        timeline: Timeline,
    ) -> Result<Self, Error> {
        check_restored(&assets, &timeline, next_id)?;
        Ok(Self {
            id,
            next_id,
            name,
            settings,
            assets,
            timeline,
        })
    }

    pub const fn id(&self) -> ProjectId {
        self.id
    }

    /// The next number the counter will assign.
    pub const fn next_id(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn keep_at_least(&mut self, next: u64) {
        if next > self.next_id {
            self.next_id = next;
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn assets(&self) -> &[Asset] {
        &self.assets
    }

    pub fn asset(&self, id: AssetId) -> Option<&Asset> {
        self.assets.iter().find(|asset| asset.id() == id)
    }

    pub const fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    /// The highest audio track with room for `range`.
    ///
    /// `None` means every audio track is taken at that time, so a place opens
    /// another audio track.
    pub fn audio_track_with_room(&self, range: TimeRange) -> Option<TrackId> {
        self.timeline
            .tracks()
            .iter()
            .rev()
            .find(|track| track.kind() == TrackKind::Audio && !track.overlaps_range(range))
            .map(Track::id)
    }

    pub fn duration(&self) -> cutlass_core::Timestamp {
        self.timeline.duration()
    }

    /// Version [`crate::FILE_VERSION`] of the project file, as JSON text.
    ///
    /// Undo and redo are not included. The crate does not open a path.
    pub fn to_json(&self) -> Result<String, Error> {
        crate::file::to_json(self)
    }

    /// Reads version [`crate::FILE_VERSION`].
    ///
    /// Any other version, or a file that does not match this version, loads
    /// nothing.
    pub fn from_json(text: &str) -> Result<Self, Error> {
        crate::file::from_json(text)
    }

    pub(crate) fn apply(&mut self, command: Command) -> Result<Outcome, Error> {
        let outcome = match command {
            Command::AddAsset(asset) => Ok(Outcome::Asset(self.add_asset_now(asset)?)),
            Command::RemoveAsset(id) => {
                self.remove_asset_now(id)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetSource { asset, source } => {
                self.set_source_now(asset, source)?;
                Ok(Outcome::Unchanged)
            }
            Command::Rename(name) => {
                self.name = name;
                Ok(Outcome::Unchanged)
            }
            Command::SetCanvas {
                width,
                height,
                ratio,
            } => {
                self.settings.set_canvas(Canvas::new(width, height, ratio)?);
                Ok(Outcome::Unchanged)
            }
            Command::SetRate(rate) => {
                self.settings.set_rate(rate);
                Ok(Outcome::Unchanged)
            }
            Command::SetColorSpace(color_space) => {
                self.settings.set_color_space(color_space);
                Ok(Outcome::Unchanged)
            }
            Command::SetBackground(background) => {
                self.settings.set_background(background);
                Ok(Outcome::Unchanged)
            }
            Command::SetLinkage(on) => {
                self.settings.set_linkage(on);
                Ok(Outcome::Unchanged)
            }
            Command::SetMagnet(on) => {
                self.set_magnet_now(on)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetLocked { track, on } => {
                self.timeline.track_mut(track)?.set_locked(on);
                Ok(Outcome::Unchanged)
            }
            Command::SetHidden { track, on } => {
                self.timeline.track_mut(track)?.set_hidden(on);
                Ok(Outcome::Unchanged)
            }
            Command::SetMuted { track, on } => {
                self.timeline.track_mut(track)?.set_muted(on);
                Ok(Outcome::Unchanged)
            }
            Command::Place { track, segment } => self.place_now(track, segment),
            Command::RemoveSegment(id) => {
                self.remove_now(id)?;
                Ok(Outcome::Unchanged)
            }
            Command::Trim { segment, source } => {
                self.trim_now(segment, source)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetSpeed { segment, speed } => {
                self.set_speed_now(segment, speed)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetCurve { segment, curve } => {
                self.set_curve_now(segment, curve)?;
                Ok(Outcome::Unchanged)
            }
            Command::Move {
                segment,
                track,
                start,
            } => {
                self.move_now(segment, track, start)?;
                Ok(Outcome::Unchanged)
            }
            Command::Split { segment, at } => Ok(Outcome::Segment(self.split_now(segment, at)?)),
            Command::SetTransform { segment, transform } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_transform(transform);
                Ok(Outcome::Unchanged)
            }
            Command::SetVolume { segment, volume } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_volume(volume)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetFades {
                segment,
                fade_in,
                fade_out,
            } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .set_fades(fade_in, fade_out)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetDenoise { segment, on } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_denoise(on)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetKeepPitch { segment, keep } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_keep_pitch(keep)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetCrop { segment, crop } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_crop(crop)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetBlend { segment, blend } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_blend(blend)?;
                Ok(Outcome::Unchanged)
            }
            Command::DetachAudio { segment } => self.detach_now(segment),
            Command::Freeze { segment, at, hold } => self.freeze_now(segment, at, hold),
            Command::Reverse { segment } => self.reverse_now(segment),
            Command::SetText { segment, words } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_words(words)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetSticker { segment, name } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_sticker(name)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetAttachment {
                segment,
                attachment,
            } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .set_attachment(attachment)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetTransition {
                segment,
                duration_us,
                overlap,
                attachment,
            } => {
                self.set_transition_now(segment, duration_us, overlap, attachment)?;
                Ok(Outcome::Unchanged)
            }
            Command::ClearTransition { segment } => {
                self.clear_transition_now(segment)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetAnimation {
                segment,
                kind,
                animation,
            } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .set_animation(kind, animation)?;
                Ok(Outcome::Unchanged)
            }
            Command::AddMarker { time_us } => {
                self.add_marker_now(time_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::RemoveMarker { time_us } => {
                self.remove_marker_now(time_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::AddClipMarker { segment, offset_us } => {
                self.add_clip_marker_now(segment, offset_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::RemoveClipMarker { segment, offset_us } => {
                self.remove_clip_marker_now(segment, offset_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::AddBeat { segment, offset_us } => {
                self.add_beat_now(segment, offset_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::RemoveBeat { segment, offset_us } => {
                self.remove_beat_now(segment, offset_us)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetVisible { segment, visible } => {
                self.refuse_lane(segment)?;
                self.timeline.segment_mut(segment)?.set_visible(visible);
                Ok(Outcome::Unchanged)
            }
            Command::AddKeyframe { segment, keyframe } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .push_keyframe(keyframe)?;
                Ok(Outcome::Unchanged)
            }
            Command::RemoveKeyframe {
                segment,
                offset_us,
                property,
            } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .remove_keyframe(offset_us, property)?;
                Ok(Outcome::Unchanged)
            }
            Command::SetEasing {
                segment,
                offset_us,
                property,
                easing,
            } => {
                self.refuse_lane(segment)?;
                self.timeline
                    .segment_mut(segment)?
                    .set_easing(offset_us, property, easing)?;
                Ok(Outcome::Unchanged)
            }
        };
        if outcome.is_ok() {
            self.repair_transitions();
            self.repair_marks();
        }
        outcome
    }

    fn add_marker_now(&mut self, time: i64) -> Result<(), Error> {
        if time < 0 {
            return Err(Error::NegativeTime);
        }
        let end = self.timeline.duration().value;
        if end <= 0 || time > end {
            return Err(Error::Marker);
        }
        self.timeline.insert_marker(time);
        Ok(())
    }

    fn remove_marker_now(&mut self, time: i64) -> Result<(), Error> {
        if time < 0 {
            return Err(Error::NegativeTime);
        }
        self.timeline.remove_marker(time);
        Ok(())
    }

    fn add_clip_marker_now(&mut self, id: SegmentId, offset: i64) -> Result<(), Error> {
        self.refuse_lane(id)?;
        self.offset_fits(id, offset)?;
        self.timeline.segment_mut(id)?.insert_marker(offset);
        Ok(())
    }

    fn remove_clip_marker_now(&mut self, id: SegmentId, offset: i64) -> Result<(), Error> {
        self.refuse_lane(id)?;
        if offset < 0 {
            return Err(Error::NegativeTime);
        }
        self.timeline.segment_mut(id)?.remove_marker(offset);
        Ok(())
    }

    fn add_beat_now(&mut self, id: SegmentId, offset: i64) -> Result<(), Error> {
        self.refuse_lane(id)?;
        self.require_audio(id)?;
        self.offset_fits(id, offset)?;
        self.timeline.segment_mut(id)?.insert_beat(offset);
        Ok(())
    }

    fn remove_beat_now(&mut self, id: SegmentId, offset: i64) -> Result<(), Error> {
        self.refuse_lane(id)?;
        self.require_audio(id)?;
        if offset < 0 {
            return Err(Error::NegativeTime);
        }
        self.timeline.segment_mut(id)?.remove_beat(offset);
        Ok(())
    }

    fn require_audio(&self, id: SegmentId) -> Result<(), Error> {
        let track = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        let kind = self
            .timeline
            .track(track)
            .ok_or(Error::UnknownTrack(track))?
            .kind();
        if kind == TrackKind::Audio {
            Ok(())
        } else {
            Err(Error::Marker)
        }
    }

    fn offset_fits(&self, id: SegmentId, offset: i64) -> Result<(), Error> {
        if offset < 0 {
            return Err(Error::NegativeTime);
        }
        let span = self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .timeline()
            .duration()
            .value;
        if offset >= span {
            Err(Error::Marker)
        } else {
            Ok(())
        }
    }

    fn repair_marks(&mut self) {
        let end = self.timeline.duration().value;
        self.timeline.retain_markers(end);
        for track in self.timeline.tracks_mut() {
            let audio = track.kind() == TrackKind::Audio;
            for segment in track.segments_mut() {
                segment.retain_stamps(audio);
            }
        }
    }

    fn set_transition_now(
        &mut self,
        segment: SegmentId,
        duration_us: i64,
        overlap: bool,
        attachment: Attachment,
    ) -> Result<(), Error> {
        self.refuse_lane(segment)?;
        let facts = self.touching_cut(segment)?;
        let transition = Transition::new(segment, facts.right, duration_us, overlap, attachment)?;
        if !transition.fits(facts.left_dur, facts.right_dur) {
            return Err(Error::Transition);
        }
        let Some((start, end)) = blend_range_checked(facts.cut, transition.sides()) else {
            return Err(Error::DurationOverflow);
        };
        if self.blend_hits(segment, start, end) {
            return Err(Error::Transition);
        }
        self.timeline.main_mut().upsert_transition(transition);
        Ok(())
    }

    fn clear_transition_now(&mut self, segment: SegmentId) -> Result<(), Error> {
        self.refuse_lane(segment)?;
        if self
            .timeline
            .main_track()
            .segments()
            .iter()
            .all(|item| item.id() != segment)
        {
            return Err(Error::Transition);
        }
        self.timeline.main_mut().clear_transition(segment);
        Ok(())
    }

    fn touching_cut(&self, segment: SegmentId) -> Result<CutFacts, Error> {
        let segments = self.timeline.main_track().segments();
        let Some(index) = segments.iter().position(|item| item.id() == segment) else {
            if self.timeline.segment(segment).is_none() {
                return Err(Error::UnknownSegment(segment));
            }
            return Err(Error::Transition);
        };
        let left = &segments[index];
        let Some(right) = segments.get(index + 1) else {
            return Err(Error::Transition);
        };
        if left.timeline().end().value != right.timeline().start().value {
            return Err(Error::Transition);
        }
        Ok(CutFacts {
            cut: left.timeline().end().value,
            left_dur: left.timeline().duration().value,
            right: right.id(),
            right_dur: right.timeline().duration().value,
        })
    }

    fn blend_hits(&self, left: SegmentId, start: i64, end: i64) -> bool {
        self.timeline
            .main_track()
            .transitions()
            .iter()
            .any(|existing| {
                if existing.left() == left {
                    return false;
                }
                let Some(facts) = self.cut_facts(existing.left()) else {
                    return true;
                };
                let Some((other_start, other_end)) =
                    blend_range_checked(facts.cut, existing.sides())
                else {
                    return true;
                };
                start < other_end && other_start < end
            })
    }

    fn cut_facts(&self, left: SegmentId) -> Option<CutFacts> {
        self.touching_cut(left).ok()
    }

    fn refuse_inside_transition(&self, id: SegmentId, at: i64) -> Result<(), Error> {
        for transition in self.timeline.main_track().transitions() {
            if transition.left() != id && transition.right() != id {
                continue;
            }
            let Some(facts) = self.cut_facts(transition.left()) else {
                continue;
            };
            if transition.covers(facts.cut, at) {
                return Err(Error::InsideTransition);
            }
        }
        Ok(())
    }

    fn repair_transitions(&mut self) {
        let segments: Vec<(SegmentId, i64, i64, i64)> = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .map(|segment| {
                let range = segment.timeline();
                (
                    segment.id(),
                    range.start().value,
                    range.end().value,
                    range.duration().value,
                )
            })
            .collect();
        let existing = self.timeline.main_track().transitions().to_vec();
        let mut kept = Vec::new();
        for transition in existing {
            let Some(index) = segments.iter().position(|item| item.0 == transition.left()) else {
                continue;
            };
            let Some(right) = segments.get(index + 1) else {
                continue;
            };
            if right.0 != transition.right() || segments[index].2 != right.1 {
                continue;
            }
            if !transition.fits(segments[index].3, right.3) {
                continue;
            }
            let (start, end) = match blend_range_checked(segments[index].2, transition.sides()) {
                Some(range) => range,
                None => continue,
            };
            let clashes = kept.iter().any(|other: &Transition| {
                let Some(other_index) = segments.iter().position(|item| item.0 == other.left())
                else {
                    return false;
                };
                let Some((other_start, other_end)) =
                    blend_range_checked(segments[other_index].2, other.sides())
                else {
                    return false;
                };
                start < other_end && other_start < end
            });
            if clashes {
                continue;
            }
            kept.push(transition);
        }
        kept.sort_by_key(|transition| {
            segments
                .iter()
                .find(|item| item.0 == transition.left())
                .map(|item| item.1)
                .unwrap_or(0)
        });
        self.timeline.main_mut().set_transitions(kept);
    }

    fn allocate(&mut self) -> Result<u64, Error> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(Error::IdsExhausted)?;
        if id == 0 {
            return Err(Error::IdsExhausted);
        }
        Ok(id)
    }

    fn set_source_now(&mut self, id: AssetId, source: String) -> Result<(), Error> {
        if self.asset(id).is_none() {
            return Err(Error::UnknownAsset(id));
        }
        let partner = self.reverse_partner(id);
        self.asset_mut(id)?.set_source(source.clone())?;
        if let Some(partner) = partner {
            self.asset_mut(partner)?.set_source(source)?;
        }
        Ok(())
    }

    fn asset_mut(&mut self, id: AssetId) -> Result<&mut Asset, Error> {
        self.assets
            .iter_mut()
            .find(|asset| asset.id() == id)
            .ok_or(Error::UnknownAsset(id))
    }

    fn reverse_partner(&self, id: AssetId) -> Option<AssetId> {
        if let Some(forward) = self.asset(id)?.reverses() {
            return Some(forward);
        }
        self.assets
            .iter()
            .find(|asset| asset.reverses() == Some(id))
            .map(Asset::id)
    }

    fn add_asset_now(&mut self, mut asset: Asset) -> Result<AssetId, Error> {
        let id = AssetId::raw(self.allocate()?);
        asset.assign(id);
        let first_video = asset.is_video()
            && self.settings.ratio() == AspectRatio::Original
            && !self.assets.iter().any(Asset::is_video);
        if first_video {
            if let Some((width, height)) = asset.visual_size() {
                self.settings.set_size(width, height)?;
            }
        }
        self.assets.push(asset);
        Ok(id)
    }

    fn remove_asset_now(&mut self, id: AssetId) -> Result<(), Error> {
        if self.asset(id).is_none() {
            return Err(Error::UnknownAsset(id));
        }
        if self.asset_in_use(id) {
            return Err(Error::AssetInUse);
        }
        if self.assets.iter().any(|asset| asset.reverses() == Some(id)) {
            return Err(Error::AssetInUse);
        }
        self.assets.retain(|asset| asset.id() != id);
        Ok(())
    }

    fn asset_in_use(&self, id: AssetId) -> bool {
        self.timeline.tracks().iter().any(|track| {
            track
                .segments()
                .iter()
                .any(|segment| segment.asset_id() == Some(id))
        })
    }

    fn set_magnet_now(&mut self, on: bool) -> Result<(), Error> {
        if self.timeline.main_track().locked() {
            return Err(Error::Locked);
        }
        if on && self.settings.linkage() && self.locked_follower() {
            return Err(Error::Locked);
        }
        self.settings.set_magnet(on);
        if on {
            self.pack_main()?;
        }
        Ok(())
    }

    fn place_now(
        &mut self,
        track_id: Option<TrackId>,
        mut segment: Segment,
    ) -> Result<Outcome, Error> {
        self.prepare(&segment)?;
        let target = match track_id {
            None => None,
            Some(track_id) => {
                let (kind, main, locked, overlaps) = {
                    let track = self
                        .timeline
                        .track(track_id)
                        .ok_or(Error::UnknownTrack(track_id))?;
                    (
                        track.kind(),
                        track.is_main(),
                        track.locked(),
                        track.overlaps_range(segment.timeline()),
                    )
                };
                if !accepts(kind, segment.channels(), body(&segment)) {
                    return Err(Error::WrongTrack);
                }
                if locked {
                    return Err(Error::Locked);
                }
                if main && self.settings.magnet() && self.packing_hits_a_lock() {
                    return Err(Error::Locked);
                }
                Some((track_id, kind, main, overlaps))
            }
        };
        let segment_id = SegmentId::raw(self.allocate()?);
        segment.assign(segment_id);
        let Some((track_id, kind, main, overlaps)) = target else {
            let track = self.spill(kind_for(&segment), segment)?;
            return Ok(Outcome::Opened {
                segment: segment_id,
                track,
            });
        };
        if main && self.settings.magnet() {
            // Followers are chosen before the new segment exists, so a clip that
            // already sits on one main segment keeps following that segment.
            let old = self.timeline.main_track().ranges();
            let index = magnet_index(
                self.timeline.main_track().segments(),
                segment.timeline().start().value,
            );
            self.timeline.main_mut().insert_at(index, segment);
            self.pack_from(&old)?;
            return Ok(Outcome::Segment(segment_id));
        }
        if overlaps {
            let track = self.spill(kind, segment)?;
            return Ok(Outcome::Opened {
                segment: segment_id,
                track,
            });
        }
        self.timeline.track_mut(track_id)?.insert(segment)?;
        Ok(Outcome::Segment(segment_id))
    }

    fn detach_now(&mut self, id: SegmentId) -> Result<Outcome, Error> {
        self.refuse_lane(id)?;
        let (channels, range) = {
            let segment = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
            (segment.channels(), segment.timeline())
        };
        match channels {
            Channels::Picture => return Err(Error::NoSound),
            Channels::Sound => return Err(Error::NoPicture),
            Channels::Both => {}
        }
        let home = self.audio_track_with_room(range);
        if home.is_some_and(|track| self.track_is_locked(track)) {
            return Err(Error::Locked);
        }
        let sound = self.timeline.segment_mut(id)?.take_sound()?;
        self.place_now(home, sound)
    }

    fn remove_now(&mut self, id: SegmentId) -> Result<(), Error> {
        if self.timeline.segment(id).is_none() {
            return Err(Error::UnknownSegment(id));
        }
        self.refuse_edit(id)?;
        let on_main = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        if on_main && self.settings.linkage() {
            let followers = self.follower_ids(id);
            for follower in followers {
                self.timeline.remove_segment(follower)?;
            }
        }
        self.timeline.remove_segment(id)?;
        self.timeline.drop_empty();
        if on_main && self.settings.magnet() {
            self.pack_main()?;
        }
        Ok(())
    }

    fn trim_now(&mut self, id: SegmentId, source: TimeRange) -> Result<(), Error> {
        let asset_id = self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .asset_id();
        self.refuse_edit(id)?;
        let Some(asset_id) = asset_id else {
            return Err(Error::Text);
        };
        let on_main = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        let asset = self
            .asset(asset_id)
            .ok_or(Error::UnknownAsset(asset_id))?
            .clone();
        self.timeline.segment_mut(id)?.trim_source(&asset, source)?;
        self.reflow(id, on_main)
    }

    fn set_speed_now(&mut self, id: SegmentId, speed: crate::Speed) -> Result<(), Error> {
        let on_main = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        if self.timeline.segment(id).is_none() {
            return Err(Error::UnknownSegment(id));
        }
        self.refuse_edit(id)?;
        self.timeline.segment_mut(id)?.change_speed(speed)?;
        self.reflow(id, on_main)
    }

    fn set_curve_now(
        &mut self,
        id: SegmentId,
        curve: Option<crate::SpeedCurve>,
    ) -> Result<(), Error> {
        let on_main = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        if self.timeline.segment(id).is_none() {
            return Err(Error::UnknownSegment(id));
        }
        self.refuse_edit(id)?;
        self.timeline.segment_mut(id)?.set_curve(curve)?;
        self.reflow(id, on_main)
    }

    fn move_now(&mut self, id: SegmentId, track: Option<TrackId>, start: i64) -> Result<(), Error> {
        if start < 0 {
            return Err(Error::NegativeTime);
        }
        let from = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        let to = track.unwrap_or(from);
        let segment = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
        let channels = segment.channels();
        let body = body(segment);
        let duration = segment.timeline().duration().value;
        let old_start = segment.timeline().start().value;
        let (to_main, dest_locked) = {
            let dest = self.timeline.track(to).ok_or(Error::UnknownTrack(to))?;
            if !accepts(dest.kind(), channels, body) {
                return Err(Error::WrongTrack);
            }
            (dest.is_main(), dest.locked())
        };
        let from_main = self.timeline.track(from).is_some_and(Track::is_main);
        if start.checked_add(duration).is_none() {
            return Err(Error::DurationOverflow);
        }
        self.refuse_edit(id)?;
        if dest_locked {
            return Err(Error::Locked);
        }

        if to_main && self.settings.magnet() {
            let old = self.timeline.main_track().ranges();
            let segment = self.timeline.take_segment(id)?;
            self.timeline.drop_empty();
            let index = magnet_index(self.timeline.main_track().segments(), start);
            self.timeline.main_mut().insert_at(index, segment);
            return self.pack_from(&old);
        }

        if self.collides(to, id, start, duration)? {
            return Err(Error::Overlap);
        }
        let delta = start
            .checked_sub(old_start)
            .ok_or(Error::DurationOverflow)?;
        let old_main = self.timeline.main_track().ranges();
        if from == to {
            self.timeline.track_mut(from)?.set_start(id, start)?;
            self.timeline.track_mut(from)?.sort_by_start();
        } else {
            let mut segment = self.timeline.take_segment(id)?;
            self.timeline.drop_empty();
            if from_main && self.settings.magnet() {
                self.pack_from(&old_main)?;
            }
            segment.set_start(start)?;
            self.timeline.track_mut(to)?.insert(segment)?;
        }
        if from_main && self.settings.linkage() && delta != 0 {
            self.shift_followers(&old_main, &[(id, delta)])?;
        }
        Ok(())
    }

    fn split_now(&mut self, id: SegmentId, at: i64) -> Result<SegmentId, Error> {
        self.refuse_edit(id)?;
        self.split_at(id, at, true)
    }

    fn split_at(&mut self, id: SegmentId, at: i64, pack: bool) -> Result<SegmentId, Error> {
        let segment = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
        let fade_in = segment.fade_in();
        let fade_out = segment.fade_out();
        let range = segment.timeline();
        if at <= range.start().value || at >= range.end().value {
            return Err(Error::SplitOutside);
        }
        let from = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        let on_main = self.timeline.track(from).is_some_and(Track::is_main);
        if on_main {
            self.refuse_inside_transition(id, at)?;
        }
        let asset_id = segment.asset_id();
        let mut left_curve = None;
        let mut right_curve = None;
        let (left_source, left_dur, right_source, right_dur) = match segment.source() {
            None => {
                let left = at
                    .checked_sub(range.start().value)
                    .ok_or(Error::DurationOverflow)?;
                let right = range
                    .end()
                    .value
                    .checked_sub(at)
                    .ok_or(Error::DurationOverflow)?;
                (None, left, None, right)
            }
            Some(source) => {
                let left_us = at
                    .checked_sub(range.start().value)
                    .ok_or(Error::DurationOverflow)?;
                let ticks = segment.source_offset(left_us)?;
                if ticks < 1 || ticks >= source.duration().value {
                    return Err(Error::SplitOutside);
                }
                let scale = source.duration().timescale;
                let left_source = TimeRange::new(source.start(), Timestamp::new(ticks, scale))?;
                let right_start = source
                    .start()
                    .value
                    .checked_add(ticks)
                    .ok_or(Error::DurationOverflow)?;
                let right_ticks = source
                    .duration()
                    .value
                    .checked_sub(ticks)
                    .ok_or(Error::DurationOverflow)?;
                let right_source = TimeRange::new(
                    Timestamp::new(right_start, scale),
                    Timestamp::new(right_ticks, scale),
                )?;
                let parts = segment.split_speed(ticks)?;
                left_curve = parts.left;
                right_curve = parts.right;
                (
                    Some(left_source),
                    parts.left_us,
                    Some(right_source),
                    parts.right_us,
                )
            }
        };
        let total = left_dur
            .checked_add(right_dur)
            .ok_or(Error::DurationOverflow)?;
        if self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .keyframes()
            .iter()
            .any(|key| key.offset().value > total)
        {
            return Err(Error::KeyframeOutOfRange);
        }
        let right_start = range
            .start()
            .value
            .checked_add(left_dur)
            .ok_or(Error::DurationOverflow)?;
        if self.collides(from, id, right_start, right_dur)? {
            return Err(Error::Overlap);
        }

        let asset = match asset_id {
            Some(id) if left_source.is_some() => {
                Some(self.asset(id).ok_or(Error::UnknownAsset(id))?.clone())
            }
            None if left_source.is_some() => return Err(Error::Text),
            _ => None,
        };
        let right_id = SegmentId::raw(self.allocate()?);
        let mut right = self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .clone();
        self.timeline
            .segment_mut(id)?
            .clear_animation(AnimationKind::Out);
        let keys = self.timeline.segment_mut(id)?.split_keyframes(left_dur)?;
        let (right_markers, right_beats) = self.timeline.segment_mut(id)?.split_stamps(left_dur);
        match left_source {
            Some(source) => {
                let asset = asset.as_ref().ok_or(Error::Text)?;
                if let Some(curve) = left_curve {
                    self.timeline.segment_mut(id)?.install_curve(curve);
                }
                self.timeline.segment_mut(id)?.trim_source(asset, source)?;
            }
            None => self
                .timeline
                .segment_mut(id)?
                .set_timeline_duration(left_dur)?,
        }
        right.replace_keyframes(keys);
        right.set_marks(right_markers, right_beats);
        right.assign(right_id);
        right.clear_animation(AnimationKind::In);
        match right_source {
            Some(source) => {
                let asset = asset.as_ref().ok_or(Error::Text)?;
                if let Some(curve) = right_curve {
                    right.install_curve(curve);
                }
                right.trim_source(asset, source)?;
                right.set_start(right_start)?;
            }
            None => {
                right.set_timeline_duration(right_dur)?;
                right.set_start(right_start)?;
            }
        }
        let (left_in, left_out, right_in, right_out) =
            crate::segment::parted_fades(fade_in, fade_out, left_dur, right_dur);
        self.timeline
            .segment_mut(id)?
            .set_fades(left_in, left_out)?;
        right.set_fades(right_in, right_out)?;
        self.timeline.track_mut(from)?.insert(right)?;
        if on_main {
            self.timeline.main_mut().retarget_left(id, right_id);
        }
        if pack && on_main && self.settings.magnet() {
            self.pack_main()?;
        }
        Ok(right_id)
    }

    fn freeze_now(&mut self, id: SegmentId, at: i64, hold: i64) -> Result<Outcome, Error> {
        if hold < 0 {
            return Err(Error::NegativeTime);
        }
        if hold == 0 {
            return Err(Error::ZeroDuration);
        }
        {
            let segment = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
            segment.refuse_no_file()?;
            if segment.channels() == Channels::Sound {
                return Err(Error::NoPicture);
            }
            if segment.source().is_none() {
                return Err(Error::ImageSource);
            }
        }
        self.refuse_edit(id)?;
        let from = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        let right_id = self.split_at(id, at, false)?;
        let on_main = self.timeline.track(from).is_some_and(Track::is_main);
        let before = if on_main {
            Some(self.timeline.main_track().ranges())
        } else {
            None
        };
        let right = self
            .timeline
            .segment(right_id)
            .ok_or(Error::UnknownSegment(right_id))?
            .clone();
        let cut = right.timeline().start().value;
        let source = right.source().ok_or(Error::MediaSource)?;
        let frame = TimeRange::new(source.start(), Timestamp::new(1, source.start().timescale))?;
        let asset_id = right.asset_id().ok_or(Error::MediaSource)?;
        let asset = self
            .asset(asset_id)
            .ok_or(Error::UnknownAsset(asset_id))?
            .clone();
        let (visible, crop, blend, transform) = {
            let left = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
            (left.visible(), left.crop(), left.blend(), left.transform())
        };
        let mut still = Segment::media(&asset, frame, cut, Channels::Picture, Speed::ONE)?;
        still.set_visible(visible);
        still.set_crop(crop)?;
        still.set_blend(blend)?;
        still.set_transform(transform);
        still.hold_frame(hold)?;

        let movers: Vec<(SegmentId, i64)> = self
            .timeline
            .track(from)
            .ok_or(Error::UnknownTrack(from))?
            .segments()
            .iter()
            .filter(|segment| segment.timeline().start().value >= cut)
            .map(|segment| (segment.id(), segment.timeline().start().value))
            .collect();
        for (segment_id, start) in movers {
            let next = start.checked_add(hold).ok_or(Error::DurationOverflow)?;
            self.timeline.track_mut(from)?.set_start(segment_id, next)?;
        }
        self.timeline.track_mut(from)?.sort_by_start();

        let freeze_id = SegmentId::raw(self.allocate()?);
        still.assign(freeze_id);
        self.prepare(&still)?;
        self.timeline.track_mut(from)?.insert(still)?;

        if on_main && self.settings.magnet() {
            self.pack_main()?;
        }
        if let Some(before) = before.filter(|_| self.settings.linkage()) {
            let after = self.timeline.main_track().ranges();
            let deltas = moved_starts(&before, &after)?;
            self.shift_followers(&before, &deltas)?;
        }
        Ok(Outcome::Segment(freeze_id))
    }

    fn reverse_now(&mut self, id: SegmentId) -> Result<Outcome, Error> {
        let segment = self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .clone();
        segment.refuse_no_file()?;
        if segment.frozen() {
            return Err(Error::Freeze);
        }
        let source = segment.source().ok_or(Error::ImageSource)?;
        self.refuse_lane(id)?;
        let current_id = segment.asset_id().ok_or(Error::ImageSource)?;
        let current = self
            .asset(current_id)
            .ok_or(Error::UnknownAsset(current_id))?
            .clone();
        let target_id = if let Some(forward) = current.reverses() {
            forward
        } else if let Some(existing) = self
            .assets
            .iter()
            .find(|asset| asset.reverses() == Some(current_id))
        {
            existing.id()
        } else {
            self.add_asset_now(current.reversed()?)?
        };
        let target = self
            .asset(target_id)
            .ok_or(Error::UnknownAsset(target_id))?
            .clone();
        let media = target.duration().ok_or(Error::ImageSource)?;
        let mirrored = mirror_source(source, media)?;
        let segment = self.timeline.segment_mut(id)?;
        segment.set_asset(target_id);
        segment.trim_source(&target, mirrored)?;
        Ok(Outcome::Asset(target_id))
    }

    fn collides(
        &self,
        track: TrackId,
        id: SegmentId,
        start: i64,
        duration: i64,
    ) -> Result<bool, Error> {
        let range = TimeRange::sequence(start, duration)?;
        let track = self
            .timeline
            .track(track)
            .ok_or(Error::UnknownTrack(track))?;
        Ok(track
            .segments()
            .iter()
            .any(|segment| segment.id() != id && segment.timeline().overlaps(range)))
    }

    fn reflow(&mut self, id: SegmentId, on_main: bool) -> Result<(), Error> {
        if on_main && self.settings.magnet() {
            return self.pack_main();
        }
        if self.timeline.track_has_overlap(id) {
            return Err(Error::Overlap);
        }
        Ok(())
    }

    fn prepare(&self, segment: &Segment) -> Result<(), Error> {
        if segment.asset_id().is_none() {
            return Ok(());
        }
        let id = segment.asset_id().ok_or(Error::Text)?;
        let asset = self.asset(id).ok_or(Error::UnknownAsset(id))?;
        segment.recheck(asset)
    }

    fn spill(&mut self, kind: TrackKind, segment: Segment) -> Result<TrackId, Error> {
        let id = TrackId::raw(self.allocate()?);
        let track = Track::occupied(id, kind, segment)?;
        self.timeline.insert(track)?;
        Ok(id)
    }

    fn pack_main(&mut self) -> Result<(), Error> {
        let old = self.timeline.main_track().ranges();
        self.pack_from(&old)
    }

    fn pack_from(&mut self, old: &[(SegmentId, TimeRange)]) -> Result<(), Error> {
        let deltas = self.timeline.pack_main()?;
        if self.settings.linkage() {
            self.shift_followers(old, &deltas)?;
        }
        Ok(())
    }

    fn follower_ids(&self, owner: SegmentId) -> Vec<SegmentId> {
        let mains = self.timeline.main_track().ranges();
        let mut ids = Vec::new();
        for track in self.timeline.tracks() {
            if track.is_main() {
                continue;
            }
            for segment in track.segments() {
                if sole_owner(segment.timeline(), &mains) == Some(owner) {
                    ids.push(segment.id());
                }
            }
        }
        ids
    }

    fn shift_followers(
        &mut self,
        old_main: &[(SegmentId, TimeRange)],
        deltas: &[(SegmentId, i64)],
    ) -> Result<(), Error> {
        let mut starts = Vec::new();
        for track in self.timeline.tracks() {
            if track.is_main() {
                continue;
            }
            for segment in track.segments() {
                if deltas.iter().any(|(owner, _)| *owner == segment.id()) {
                    continue;
                }
                let Some(owner) = sole_owner(segment.timeline(), old_main) else {
                    continue;
                };
                let Some(delta) = deltas
                    .iter()
                    .find(|(id, _)| *id == owner)
                    .map(|(_, delta)| *delta)
                else {
                    continue;
                };
                if delta == 0 {
                    continue;
                }
                let new_start = segment
                    .timeline()
                    .start()
                    .value
                    .checked_add(delta)
                    .ok_or(Error::DurationOverflow)?;
                if new_start < 0 {
                    return Err(Error::NegativeTime);
                }
                starts.push((segment.id(), new_start));
            }
        }
        for track in self.timeline.tracks() {
            preview_overlap(track, &starts)?;
        }
        self.timeline.assign_starts(&starts)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Body {
    Media,
    Text,
    Sticker,
    Effect,
    Filter,
    Adjust,
}

fn body(segment: &Segment) -> Body {
    if segment.is_text() {
        Body::Text
    } else if segment.is_sticker() {
        Body::Sticker
    } else if segment.is_effect() {
        Body::Effect
    } else if segment.is_filter() {
        Body::Filter
    } else if segment.is_adjust() {
        Body::Adjust
    } else {
        Body::Media
    }
}

fn restored_kind(segment: &Segment) -> Option<TrackKind> {
    if segment.is_text() {
        Some(TrackKind::Text)
    } else if segment.is_sticker() {
        Some(TrackKind::Sticker)
    } else if segment.is_effect() {
        Some(TrackKind::Effect)
    } else if segment.is_filter() {
        Some(TrackKind::Filter)
    } else if segment.is_adjust() {
        Some(TrackKind::Adjust)
    } else {
        None
    }
}

fn kind_for(segment: &Segment) -> TrackKind {
    match body(segment) {
        Body::Text => TrackKind::Text,
        Body::Sticker => TrackKind::Sticker,
        Body::Effect => TrackKind::Effect,
        Body::Filter => TrackKind::Filter,
        Body::Adjust => TrackKind::Adjust,
        Body::Media => match segment.channels() {
            Channels::Sound => TrackKind::Audio,
            Channels::Picture | Channels::Both => TrackKind::Video,
        },
    }
}

fn accepts(kind: TrackKind, channels: Channels, body: Body) -> bool {
    match body {
        Body::Text => return kind == TrackKind::Text,
        Body::Sticker => return kind == TrackKind::Sticker,
        Body::Effect => return kind == TrackKind::Effect,
        Body::Filter => return kind == TrackKind::Filter,
        Body::Adjust => return kind == TrackKind::Adjust,
        Body::Media => {}
    }
    match (kind, channels) {
        (TrackKind::Video, Channels::Picture | Channels::Both) => true,
        (TrackKind::Video, Channels::Sound) => false,
        (TrackKind::Audio, Channels::Sound) => true,
        (TrackKind::Audio, Channels::Picture | Channels::Both) => false,
        (
            TrackKind::Adjust
            | TrackKind::Filter
            | TrackKind::Effect
            | TrackKind::Text
            | TrackKind::Sticker
            | TrackKind::PluginEffect,
            Channels::Picture | Channels::Sound | Channels::Both,
        ) => false,
    }
}

fn magnet_index(segments: &[Segment], start: i64) -> usize {
    segments
        .iter()
        .position(|segment| segment.timeline().start().value >= start)
        .unwrap_or(segments.len())
}

fn mirror_source(source: TimeRange, media: Timestamp) -> Result<TimeRange, Error> {
    if source.start().timescale != media.timescale {
        return Err(Error::SourceTimescale {
            asset: media.timescale.get(),
            source_scale: source.start().timescale.get(),
        });
    }
    let start = media
        .value
        .checked_sub(source.end().value)
        .ok_or(Error::DurationOverflow)?;
    if start < 0 {
        return Err(Error::SourceOutsideAsset);
    }
    TimeRange::new(Timestamp::new(start, media.timescale), source.duration())
}

fn moved_starts(
    before: &[(SegmentId, TimeRange)],
    after: &[(SegmentId, TimeRange)],
) -> Result<Vec<(SegmentId, i64)>, Error> {
    let mut deltas = Vec::new();
    for (id, range) in before {
        let Some((_, later)) = after.iter().find(|(later_id, _)| later_id == id) else {
            continue;
        };
        let delta = later
            .start()
            .value
            .checked_sub(range.start().value)
            .ok_or(Error::DurationOverflow)?;
        if delta != 0 {
            deltas.push((*id, delta));
        }
    }
    Ok(deltas)
}

impl Project {
    fn track_is_locked(&self, id: TrackId) -> bool {
        self.timeline.track(id).is_some_and(Track::locked)
    }

    /// The segment's own track is locked.
    fn refuse_lane(&self, id: SegmentId) -> Result<(), Error> {
        let track = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        if self.track_is_locked(track) {
            return Err(Error::Locked);
        }
        Ok(())
    }

    /// A main edit would move a clip on a locked track.
    fn ripples_locked(&self, id: SegmentId) -> bool {
        let on_main = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| segment.id() == id);
        if !on_main {
            return false;
        }
        let magnet = self.settings.magnet();
        let linkage = self.settings.linkage();
        if !magnet && !linkage {
            return false;
        }
        let mains: Vec<_> = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .map(|segment| (segment.id(), segment.timeline()))
            .collect();
        for track in self.timeline.tracks() {
            if track.is_main() || !track.locked() {
                continue;
            }
            for segment in track.segments() {
                let owner = sole_owner(segment.timeline(), &mains);
                let follows = (magnet && owner.is_some()) || (linkage && owner == Some(id));
                if follows {
                    return true;
                }
            }
        }
        false
    }

    fn refuse_edit(&self, id: SegmentId) -> Result<(), Error> {
        self.refuse_lane(id)?;
        if self.ripples_locked(id) {
            return Err(Error::Locked);
        }
        Ok(())
    }

    /// Magnet on the main track would slide a clip on a locked track.
    fn packing_hits_a_lock(&self) -> bool {
        self.timeline
            .main_track()
            .segments()
            .iter()
            .any(|segment| self.ripples_locked(segment.id()))
    }

    /// A locked lane holds a clip that follows one main clip.
    fn locked_follower(&self) -> bool {
        let mains: Vec<_> = self
            .timeline
            .main_track()
            .segments()
            .iter()
            .map(|segment| (segment.id(), segment.timeline()))
            .collect();
        self.timeline.tracks().iter().any(|track| {
            !track.is_main()
                && track.locked()
                && track
                    .segments()
                    .iter()
                    .any(|segment| sole_owner(segment.timeline(), &mains).is_some())
        })
    }
}

fn sole_owner(range: TimeRange, mains: &[(SegmentId, TimeRange)]) -> Option<SegmentId> {
    let mut found = None;
    for (id, main) in mains {
        if range.overlaps(*main) {
            if found.is_some() {
                return None;
            }
            found = Some(*id);
        }
    }
    found
}

fn preview_overlap(track: &Track, starts: &[(SegmentId, i64)]) -> Result<(), Error> {
    let mut ranges = Vec::new();
    for segment in track.segments() {
        let start = starts
            .iter()
            .find(|(id, _)| *id == segment.id())
            .map(|(_, start)| *start)
            .unwrap_or(segment.timeline().start().value);
        let end = start
            .checked_add(segment.timeline().duration().value)
            .ok_or(Error::DurationOverflow)?;
        ranges.push((segment.id(), start, end));
    }
    ranges.sort_by_key(|(_, start, _)| *start);
    for pair in ranges.windows(2) {
        if pair[1].1 < pair[0].2 {
            let moved = [pair[0].0, pair[1].0]
                .into_iter()
                .find(|id| starts.iter().any(|(planned, _)| planned == id))
                .unwrap_or(pair[1].0);
            return Err(Error::LinkageOverlap(moved));
        }
    }
    Ok(())
}

fn check_restored(assets: &[Asset], timeline: &Timeline, next_id: u64) -> Result<(), Error> {
    let mut seen = BTreeSet::new();
    claim(&mut seen, timeline.id().get())?;
    let mut mains = 0;
    let mut previous: Option<Band> = None;
    for track in timeline.tracks() {
        claim(&mut seen, track.id().get())?;
        if track.is_main() {
            mains += 1;
            if track.kind() != TrackKind::Video {
                return Err(Error::WrongTrack);
            }
        } else if track.segments().is_empty() {
            return Err(Error::EmptyTrack);
        }
        if previous.is_some_and(|band| band > track.band()) {
            return Err(Error::TrackOrder);
        }
        previous = Some(track.band());
        for segment in track.segments() {
            claim(&mut seen, segment.id().get())?;
            if let Some(expected) = restored_kind(segment) {
                if track.kind() != expected {
                    return Err(Error::WrongTrack);
                }
                continue;
            }
            let asset_id = segment.asset_id().ok_or(Error::Text)?;
            let asset = assets
                .iter()
                .find(|asset| asset.id() == asset_id)
                .ok_or(Error::UnknownAsset(asset_id))?;
            segment.recheck(asset)?;
            if !accepts(track.kind(), segment.channels(), Body::Media) {
                return Err(Error::WrongTrack);
            }
        }
    }
    if mains != 1 {
        return Err(if mains == 0 {
            Error::MissingMain
        } else {
            Error::SecondMainTrack
        });
    }
    for asset in assets {
        claim(&mut seen, asset.id().get())?;
    }
    let Some(max) = seen.iter().next_back().copied() else {
        return Err(Error::MissingMain);
    };
    if next_id <= max {
        return Err(Error::IdCounter { next_id });
    }
    Ok(())
}

struct CutFacts {
    cut: i64,
    left_dur: i64,
    right: SegmentId,
    right_dur: i64,
}

fn blend_range_checked(cut: i64, sides: (i64, i64)) -> Option<(i64, i64)> {
    Some((cut.checked_sub(sides.0)?, cut.checked_add(sides.1)?))
}

fn claim(seen: &mut BTreeSet<u64>, id: u64) -> Result<(), Error> {
    if id == 0 {
        return Err(Error::BadId);
    }
    if !seen.insert(id) {
        return Err(Error::RepeatedId(id));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;
    use crate::history::History;
    use crate::id::ProjectId;
    use crate::segment::Channels;
    use crate::time::{Speed, TimeRange};
    use cutlass_core::Timestamp;
    use std::num::NonZeroU32;

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([1; 16]), "cut")
    }

    #[test]
    fn an_empty_project_starts_with_the_main_track() {
        let project = project();
        assert_eq!(project.name(), "cut");
        assert_eq!(project.duration().value, 0);
        assert_eq!(project.duration().timescale.get(), 1_000_000);
        assert_eq!(project.timeline().id().get(), 1);
        assert_eq!(project.timeline().tracks().len(), 1);
        assert!(project.timeline().main_track().is_main());
        assert_eq!(project.timeline().main_track().id().get(), 2);
        assert_eq!(project.next_id(), 3);
        assert!(project.settings().magnet());
        assert!(project.settings().linkage());
    }

    #[test]
    fn rename_replaces_the_name() {
        let mut history = History::new();
        let mut project = Project::new(ProjectId::from_bytes([1; 16]), "");
        assert_eq!(project.name(), "");
        history
            .apply(&mut project, &[Command::Rename("spot".to_string())])
            .unwrap();
        assert_eq!(project.name(), "spot");
    }

    #[test]
    fn set_source_replaces_the_path_and_leaves_the_clip() {
        let mut history = History::new();
        let mut project = project();
        let video = Asset::video(
            "file:///v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).unwrap()),
            320,
            180,
            true,
        )
        .unwrap();
        let id = match history
            .apply(&mut project, &[Command::AddAsset(video)])
            .unwrap()
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let stored = project.asset(id).unwrap();
        let duration = stored.duration().unwrap();
        let source = TimeRange::new(
            Timestamp::new(0, duration.timescale),
            Timestamp::new(duration.value, duration.timescale),
        )
        .unwrap();
        let segment = Segment::media(stored, source, 0, Channels::Both, Speed::ONE).unwrap();
        let main = project.timeline().main_track().id();
        history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .unwrap();
        history
            .apply(
                &mut project,
                &[Command::SetSource {
                    asset: id,
                    source: "file:///other.mp4".to_string(),
                }],
            )
            .unwrap();
        let asset = project.asset(id).unwrap();
        assert_eq!(asset.source(), "file:///other.mp4");
        assert_eq!(asset.duration().unwrap().value, 30_000);
        assert_eq!(asset.visual_size(), Some((320, 180)));
        assert_eq!(
            project.timeline().main_track().segments()[0]
                .source()
                .unwrap()
                .duration()
                .value,
            30_000
        );
        history.undo(&mut project).unwrap();
        assert_eq!(project.asset(id).unwrap().source(), "file:///v.mp4");
        let empty = history.apply(
            &mut project,
            &[Command::SetSource {
                asset: id,
                source: String::new(),
            }],
        );
        assert_eq!(empty.unwrap_err(), Error::EmptySource);
        assert_eq!(project.asset(id).unwrap().source(), "file:///v.mp4");
        let missing = crate::id::AssetId::raw(99);
        let unknown = history.apply(
            &mut project,
            &[Command::SetSource {
                asset: missing,
                source: "file:///x.mp4".to_string(),
            }],
        );
        assert_eq!(unknown.unwrap_err(), Error::UnknownAsset(missing));
    }

    #[test]
    fn a_segment_is_checked_against_the_asset_in_the_project() {
        let mut history = History::new();
        let mut project = project();
        let short = Asset::video(
            "file:///v.mp4",
            Timestamp::new(30_000, NonZeroU32::new(30_000).unwrap()),
            320,
            180,
            true,
        )
        .unwrap();
        let asset = match history
            .apply(&mut project, &[Command::AddAsset(short)])
            .unwrap()
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let mut long = Asset::video(
            "file:///v.mp4",
            Timestamp::new(90_000, NonZeroU32::new(30_000).unwrap()),
            320,
            180,
            true,
        )
        .unwrap();
        long.assign(asset);
        let source = TimeRange::new(
            Timestamp::new(0, NonZeroU32::new(30_000).unwrap()),
            Timestamp::new(60_000, NonZeroU32::new(30_000).unwrap()),
        )
        .unwrap();
        let segment = Segment::media(&long, source, 0, Channels::Both, Speed::ONE).unwrap();
        let main = project.timeline().main_track().id();
        let error = history
            .apply(
                &mut project,
                &[Command::Place {
                    track: Some(main),
                    segment,
                }],
            )
            .unwrap_err();
        assert_eq!(error, Error::SourceOutsideAsset);
        assert!(project.timeline().main_track().segments().is_empty());
        assert_eq!(project.next_id(), 4);
    }
}
