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
    // --- transitions (M4) -------------------------------------------------

    /// Add (or replace) a transition at the junction where `left` abuts the
    /// next clip on its track. The catalog id must exist and `left` must abut
    /// a following clip. Uses the default window length.
    pub fn add_transition(&mut self, left: ClipId, transition_id: &str) -> Result<(), ModelError> {
        if crate::transition::transition_spec(transition_id).is_none() {
            return Err(ModelError::InvalidParam(format!(
                "unknown transition '{transition_id}'"
            )));
        }
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownClip(left))?
            .kind;
        if !kind.supports_transitions() {
            return Err(ModelError::InvalidParam(format!(
                "transitions aren't supported on {kind:?} lanes"
            )));
        }
        let right = self.right_neighbor(track_id, left).ok_or_else(|| {
            ModelError::InvalidParam("clip has no abutting clip to its right".into())
        })?;
        let transition = Transition::new(
            left,
            right,
            transition_id,
            crate::transition::DEFAULT_TRANSITION_TICKS,
        );
        self.timeline
            .track_mut(track_id)
            .ok_or(ModelError::UnknownClip(left))?
            .upsert_transition(transition);
        Ok(())
    }

    /// Remove the transition at the `left` junction. Errors if none exists.
    pub fn remove_transition(&mut self, left: ClipId) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let removed = self
            .timeline
            .track_mut(track_id)
            .and_then(|t| t.remove_transition(left));
        if removed.is_none() {
            return Err(ModelError::InvalidParam(
                "clip has no transition at its right junction".into(),
            ));
        }
        Ok(())
    }

    /// Set the window length (timeline ticks) of an existing transition.
    pub fn set_transition_duration(
        &mut self,
        left: ClipId,
        duration: i64,
    ) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let transition = self
            .timeline
            .track_mut(track_id)
            .and_then(|t| t.transition_at_mut(left))
            .ok_or_else(|| {
                ModelError::InvalidParam("clip has no transition at its right junction".into())
            })?;
        transition.duration = duration.max(1);
        Ok(())
    }

    /// The clip on `track` whose start abuts the end of `left`, if any.
    fn right_neighbor(&self, track_id: TrackId, left: ClipId) -> Option<ClipId> {
        let track = self.timeline.track(track_id)?;
        let left_end = track.clip(left)?.timeline.end_tick();
        track
            .clips()
            .find(|c| c.id != left && c.timeline.start.value == left_end)
            .map(|c| c.id)
    }

    /// Whether any track carries a transition (cheap guard for prune).
    pub fn has_transitions(&self) -> bool {
        self.timeline
            .tracks_ordered()
            .any(|t| !t.transitions().is_empty())
    }

    /// Snapshot every track's transitions (for undo of structural edits that
    /// prune dead junctions).
    pub fn transitions_snapshot(&self) -> Vec<(TrackId, Vec<Transition>)> {
        self.timeline
            .tracks_ordered()
            .map(|t| (t.id, t.transitions().to_vec()))
            .collect()
    }

    /// Restore a [`Self::transitions_snapshot`] across tracks.
    pub fn restore_transitions(&mut self, snapshot: Vec<(TrackId, Vec<Transition>)>) {
        for (track_id, transitions) in snapshot {
            if let Some(track) = self.timeline.track_mut(track_id) {
                track.set_transitions(transitions);
            }
        }
    }

    /// Drop transitions whose junction no longer abuts, across all tracks.
    /// Returns whether anything was pruned.
    pub fn prune_dead_transitions(&mut self) -> bool {
        let track_ids: Vec<TrackId> = self.timeline.tracks_ordered().map(|t| t.id).collect();
        let mut pruned = false;
        for id in track_ids {
            if let Some(track) = self.timeline.track_mut(id) {
                pruned |= track.prune_dead_transitions();
            }
        }
        pruned
    }
}
