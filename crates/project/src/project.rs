//! The project: settings, assets, and one timeline.

use cutlass_core::Timestamp;

use crate::Error;
use crate::asset::Asset;
use crate::command::{Command, Outcome};
use crate::id::{AssetId, ProjectId, SegmentId, TimelineId, TrackId};
use crate::segment::{Channels, Segment};
use crate::settings::{AspectRatio, Canvas, Settings};
use crate::time::{TimeRange, source_ticks, timeline_duration};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};

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

    pub fn duration(&self) -> cutlass_core::Timestamp {
        self.timeline.duration()
    }

    pub(crate) fn apply(&mut self, command: Command) -> Result<Outcome, Error> {
        match command {
            Command::AddAsset(asset) => Ok(Outcome::Asset(self.add_asset_now(asset)?)),
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
                self.timeline.segment_mut(segment)?.set_transform(transform);
                Ok(Outcome::Unchanged)
            }
        }
    }

    fn allocate(&mut self) -> Result<u64, Error> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(Error::IdsExhausted)?;
        if id == 0 {
            return Err(Error::IdsExhausted);
        }
        Ok(id)
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

    fn set_magnet_now(&mut self, on: bool) -> Result<(), Error> {
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
                let track = self
                    .timeline
                    .track(track_id)
                    .ok_or(Error::UnknownTrack(track_id))?;
                if !accepts(track.kind(), segment.channels()) {
                    return Err(Error::WrongTrack);
                }
                Some((
                    track_id,
                    track.kind(),
                    track.is_main(),
                    track.overlaps_range(segment.timeline()),
                ))
            }
        };
        let segment_id = SegmentId::raw(self.allocate()?);
        segment.assign(segment_id);
        let Some((track_id, kind, main, overlaps)) = target else {
            let track = self.spill(kind_for(segment.channels()), segment)?;
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

    fn remove_now(&mut self, id: SegmentId) -> Result<(), Error> {
        if self.timeline.segment(id).is_none() {
            return Err(Error::UnknownSegment(id));
        }
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
        self.timeline.segment_mut(id)?.change_speed(speed)?;
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
        let duration = segment.timeline().duration().value;
        let old_start = segment.timeline().start().value;
        let dest = self.timeline.track(to).ok_or(Error::UnknownTrack(to))?;
        if !accepts(dest.kind(), channels) {
            return Err(Error::WrongTrack);
        }
        let to_main = dest.is_main();
        let from_main = self.timeline.track(from).is_some_and(Track::is_main);
        if start.checked_add(duration).is_none() {
            return Err(Error::DurationOverflow);
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
        let segment = self.timeline.segment(id).ok_or(Error::UnknownSegment(id))?;
        let range = segment.timeline();
        if at <= range.start().value || at >= range.end().value {
            return Err(Error::SplitOutside);
        }
        let from = self
            .timeline
            .track_of(id)
            .ok_or(Error::UnknownSegment(id))?;
        let on_main = self.timeline.track(from).is_some_and(Track::is_main);
        let asset_id = segment.asset_id();
        let speed = segment.speed();
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
                let ticks = source_ticks(left_us, source.duration().timescale, speed)?;
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
                let left_dur = timeline_duration(left_source.duration(), speed)?.value;
                let right_dur = timeline_duration(right_source.duration(), speed)?.value;
                (Some(left_source), left_dur, Some(right_source), right_dur)
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

        let asset = self
            .asset(asset_id)
            .ok_or(Error::UnknownAsset(asset_id))?
            .clone();
        let right_id = SegmentId::raw(self.allocate()?);
        let mut right = self
            .timeline
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .clone();
        let keys = self.timeline.segment_mut(id)?.split_keyframes(left_dur)?;
        match left_source {
            Some(source) => self.timeline.segment_mut(id)?.trim_source(&asset, source)?,
            None => self
                .timeline
                .segment_mut(id)?
                .set_timeline_duration(left_dur)?,
        }
        right.replace_keyframes(keys);
        right.assign(right_id);
        match right_source {
            Some(source) => {
                right.trim_source(&asset, source)?;
                right.set_start(right_start)?;
            }
            None => {
                right.set_timeline_duration(right_dur)?;
                right.set_start(right_start)?;
            }
        }
        self.timeline.track_mut(from)?.insert(right)?;
        if on_main && self.settings.magnet() {
            self.pack_main()?;
        }
        Ok(right_id)
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
        let asset = self
            .asset(segment.asset_id())
            .ok_or(Error::UnknownAsset(segment.asset_id()))?;
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

fn kind_for(channels: Channels) -> TrackKind {
    match channels {
        Channels::Sound => TrackKind::Audio,
        Channels::Picture | Channels::Both => TrackKind::Video,
    }
}

fn accepts(kind: TrackKind, channels: Channels) -> bool {
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
