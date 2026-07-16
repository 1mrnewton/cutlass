#![allow(unused_imports)]

use std::path::Path;

use crate::clip::{
    Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, ParamValue, Replaceable,
    SlotMedia, look_animation_combo_period_ticks, look_animation_window_ticks, split_speed_curve,
};
use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
use crate::look::{
    AnimationRef, AnimationSlot, AudioRole, ChromaKey, ColorAdjustments, Filter, Lut, Mask,
    StabilizeLevel, animation_spec,
};
use crate::media::MediaSource;
use crate::metadata::ProjectMetadata;
use crate::param::{Easing, Param};
use crate::schema::ProjectSchema;
use crate::time::{
    Rational, RationalTime, TimeRange, check_same_rate, resample, time_add, time_sub,
};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};
use crate::transition::Transition;

use super::Project;

impl Project {
    /// Move a clip to `to_track` at `new_start`, preserving duration and source.
    pub fn move_clip(
        &mut self,
        clip_id: ClipId,
        to_track: TrackId,
        new_start: RationalTime,
    ) -> Result<(), ModelError> {
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(new_start.rate, tl_rate)?;
        if new_start.value < 0 {
            return Err(ModelError::InvalidRange);
        }

        let from_track = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip_content = &self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .content;

        let dest = self
            .timeline
            .track(to_track)
            .ok_or(ModelError::UnknownTrack(to_track))?;

        if !dest.kind.accepts_content(clip_content) {
            return Err(ModelError::IncompatibleTrackKind {
                track: to_track,
                kind: dest.kind,
            });
        }

        let duration = self
            .timeline
            .clip(clip_id)
            .expect("clip is on a track")
            .timeline
            .duration
            .value;
        let new_tl = TimeRange::at_rate(new_start.value, duration, tl_rate);

        let ignore = (from_track == to_track).then_some(clip_id);
        if dest.has_overlap(new_tl, ignore)? {
            return Err(ModelError::Overlap(to_track));
        }

        if from_track == to_track {
            self.timeline
                .clip_mut(clip_id)
                .expect("clip is on a track")
                .timeline = new_tl;
        } else {
            let mut clip = self
                .timeline
                .remove_clip(clip_id)
                .expect("clip is on a track");
            clip.timeline = new_tl;
            self.timeline.add_clip(to_track, clip)?;
        }
        Ok(())
    }

    /// Shift every clip on `track_id` whose start is at or after `from` by
    /// `delta` ticks (ripple primitive: opens a hole for an insert when
    /// positive, closes a gap when negative).
    ///
    /// Validated atomically: shifting left must not collide with the nearest
    /// unshifted clip or push the first shifted clip below tick 0. Relative
    /// spacing among the shifted clips is preserved, so no other overlap can
    /// arise. Returns the new start of the first shifted clip, or `None` when
    /// no clip starts at/after `from` (a no-op).
    pub fn shift_clips(
        &mut self,
        track_id: TrackId,
        from: RationalTime,
        delta: RationalTime,
    ) -> Result<Option<RationalTime>, ModelError> {
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(from.rate, tl_rate)?;
        check_same_rate(delta.rate, tl_rate)?;
        if delta.value == 0 {
            return Err(ModelError::InvalidRange);
        }

        let track = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?;

        // Clips never overlap, so the shifted set is a contiguous suffix in
        // start order; only its first member can collide when moving left.
        let mut first_shifted: Option<i64> = None;
        let mut prev_end: i64 = 0;
        for clip in track.clips() {
            let start = clip.timeline.start.value;
            if start >= from.value {
                first_shifted = Some(first_shifted.map_or(start, |s| s.min(start)));
            } else {
                prev_end = prev_end.max(clip.timeline.end_tick());
            }
        }
        let Some(first) = first_shifted else {
            return Ok(None);
        };

        let new_first = first + delta.value;
        if new_first < 0 {
            return Err(ModelError::InvalidRange);
        }
        if delta.value < 0 && new_first < prev_end {
            return Err(ModelError::Overlap(track_id));
        }

        let track = self
            .timeline
            .track_mut(track_id)
            .expect("track existence checked above");
        for clip in track.clips_mut() {
            if clip.timeline.start.value >= from.value {
                clip.timeline.start = time_add(&clip.timeline.start, &delta)?;
            }
        }
        Ok(Some(RationalTime::new(new_first, tl_rate)))
    }

    /// Delete a clip and slide later clips on its track left to close the gap.
    pub fn ripple_delete(&mut self, clip_id: ClipId) -> Result<Clip, ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let removed = self
            .timeline
            .remove_clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let gap_start = removed.timeline.start;
        let gap = removed.timeline.duration;

        let track = self
            .timeline
            .track_mut(track_id)
            .expect("track existence checked above");
        for clip in track.clips_mut() {
            if clip.timeline.start.value >= gap_start.value {
                clip.timeline.start = time_sub(&clip.timeline.start, &gap)?;
            }
        }
        Ok(removed)
    }
}
