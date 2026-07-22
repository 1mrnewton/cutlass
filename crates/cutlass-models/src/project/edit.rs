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
    // --- editing primitives ----------------------------------------------

    pub fn remove_clip(&mut self, clip_id: ClipId) -> Option<Clip> {
        self.timeline.remove_clip(clip_id)
    }

    /// Split the clip at timeline position `at` into two abutting clips.
    pub fn split_clip(&mut self, clip_id: ClipId, at: RationalTime) -> Result<ClipId, ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .cloned()
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let tl = clip.timeline;
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(at.rate, tl_rate)?;

        if at.value <= tl.start.value || at.value >= tl.end_tick() {
            return Err(ModelError::InvalidRange);
        }

        let left_tl = TimeRange::at_rate(tl.start.value, at.value - tl.start.value, tl_rate);
        let right_tl = TimeRange::at_rate(at.value, tl.end_tick() - at.value, tl_rate);
        super::helpers::validate_split_render_continuity(
            &clip,
            left_tl.duration.value,
            right_tl.duration.value,
            tl_rate,
        )?;
        let split_fraction = left_tl.duration.value as f64 / tl.duration.value as f64;
        let (left_speed_curve, right_speed_curve) =
            split_speed_curve(&clip.speed_curve, split_fraction)?;

        let new_left_source = match clip.content.clone() {
            ClipSource::Media { media, source } => {
                let media_fps = self
                    .media
                    .get(&media)
                    .ok_or(ModelError::UnknownMedia(media))?
                    .frame_rate;
                if clip.freeze_frame {
                    let held = TimeRange::at_rate(source.start.value, 1, media_fps);
                    Some((held, held))
                } else {
                    if source.duration.value < 2 {
                        return Err(ModelError::InvalidRange);
                    }
                    // Use the original clip's actual source positions on both
                    // sides of the cut. This includes exact rational base
                    // speed, integrated speed ramps, mixed source/timeline
                    // rates, and reversal. If adjacent timeline frames resolve
                    // to the same source frame, disjoint non-empty source
                    // windows cannot preserve both halves, reject the cut.
                    let boundary = clip.source_time_at(at)?.ok_or(ModelError::InvalidRange)?;
                    let previous_at = RationalTime::new(
                        at.value.checked_sub(1).ok_or(ModelError::TimeOverflow)?,
                        tl_rate,
                    );
                    let previous = clip
                        .source_time_at(previous_at)?
                        .ok_or(ModelError::InvalidRange)?;
                    let source_last = source.start.value + source.duration.value - 1;
                    let left_src_dur = if clip.reversed {
                        if previous.value <= boundary.value {
                            return Err(ModelError::InvalidRange);
                        }
                        source_last - boundary.value
                    } else {
                        if previous.value >= boundary.value {
                            return Err(ModelError::InvalidRange);
                        }
                        boundary.value - source.start.value
                    };
                    if left_src_dur <= 0 || left_src_dur >= source.duration.value {
                        return Err(ModelError::InvalidRange);
                    }
                    // A reversed clip plays its window backward: the timeline's
                    // left half shows the source window's TOP, so the split
                    // hands the window bottom to the right clip.
                    let (left_src_start, right_src_start) = if clip.reversed {
                        (
                            source.start.value + source.duration.value - left_src_dur,
                            source.start.value,
                        )
                    } else {
                        (source.start.value, source.start.value + left_src_dur)
                    };
                    let left_source = TimeRange::at_rate(left_src_start, left_src_dur, media_fps);
                    let right_source = TimeRange::at_rate(
                        right_src_start,
                        source.duration.value - left_src_dur,
                        media_fps,
                    );
                    Some((left_source, right_source))
                }
            }
            ClipSource::Generated(_) => None,
        };

        // Clone first so newly-added serde-default fields automatically survive
        // future splits. Only split identity/placement/linkage and the fields
        // whose domains are anchored to clip time are changed below.
        let mut new_clip = clip.clone();
        new_clip.id = ClipId::next();
        new_clip.timeline = right_tl;
        // Existing policy: the shell decides whether split tails should form a
        // new link group (linked multi-clip splits relink them explicitly).
        new_clip.link = None;
        if let (Some((_, right_source)), ClipSource::Media { source, .. }) =
            (new_left_source, &mut new_clip.content)
        {
            *source = right_source;
        }
        // Ordinary animation keyframes are clip-relative. Moving every tail
        // keyframe left by the split offset makes sampling tail tick `t`
        // identical to sampling original tick `split + t`.
        new_clip.shift_timeline_params(-left_tl.duration.value)?;
        new_clip.speed_curve = right_speed_curve;
        // Edge-anchored properties stay on the corresponding outer edge.
        new_clip.fade_in = 0;
        if new_clip.animation_combo.is_none() {
            new_clip.animation_in = None;
        }

        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;

        {
            let left = self
                .timeline
                .clip_mut(clip_id)
                .expect("clip existence checked above");
            left.timeline = left_tl;
            // The tail owns properties anchored to the original right edge.
            left.fade_out = 0;
            if left.animation_combo.is_none() {
                left.animation_out = None;
            }
            if let (Some((left_source, _)), ClipSource::Media { source, .. }) =
                (new_left_source, &mut left.content)
            {
                *source = left_source;
            }
            left.speed_curve = left_speed_curve;
        }
        self.timeline.add_clip(track_id, new_clip)
    }

    /// Set the clip's timeline placement to `new_timeline` (trim/extend).
    pub fn trim_clip(
        &mut self,
        clip_id: ClipId,
        new_timeline: TimeRange,
    ) -> Result<(), ModelError> {
        if new_timeline.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let old_tl = clip.timeline;
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(new_timeline.start.rate, tl_rate)?;

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let new_source = match clip.content.clone() {
            ClipSource::Media { media: _, source } if clip.freeze_frame => {
                Some(TimeRange::at_rate(source.start.value, 1, source.start.rate))
            }
            ClipSource::Media { media, source } => {
                let media = self
                    .media
                    .get(&media)
                    .ok_or(ModelError::UnknownMedia(media))?;
                // Source ticks consumed per timeline tick scale with the
                // clip's speed (1:1 for never-retimed clips).
                let head_delta = clip.scale_by_speed(
                    resample(
                        RationalTime::new(new_timeline.start.value - old_tl.start.value, tl_rate),
                        media.frame_rate,
                    )
                    .value,
                );
                let new_src_dur = clip
                    .scale_by_speed(resample(new_timeline.duration, media.frame_rate).value)
                    .max(1);
                // A reversed clip plays its window backward, so the
                // timeline head shows the window's END: a head trim drops
                // source from the top, a tail trim from the bottom —
                // mirror-image of the forward case.
                let new_src_start = if clip.reversed {
                    source.start.value + source.duration.value - new_src_dur - head_delta
                } else {
                    source.start.value + head_delta
                };
                // Stills extend freely past the pool's default 5s window —
                // the one frame repeats and decode ignores the window, so
                // the source range is duration bookkeeping only. Clamp the
                // start to 0 so extensions stay canonical.
                if media.is_image {
                    Some(TimeRange::at_rate(
                        new_src_start.max(0),
                        new_src_dur,
                        media.frame_rate,
                    ))
                } else {
                    if new_src_start < 0 || new_src_start + new_src_dur > media.duration.value {
                        return Err(ModelError::SourceOutOfBounds);
                    }
                    Some(TimeRange::at_rate(
                        new_src_start,
                        new_src_dur,
                        media.frame_rate,
                    ))
                }
            }
            ClipSource::Generated(_) => None,
        };

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.timeline = new_timeline;
        if let (Some(src), ClipSource::Media { source, .. }) = (new_source, &mut clip.content) {
            *source = src;
        }
        Ok(())
    }

    /// Retime a media clip (CapCut speed, M1): keep its timeline start and
    /// source window, set `speed`/`reversed`, and re-derive the timeline
    /// duration (source duration ÷ speed — faster clips occupy less
    /// timeline). Rejected on generated clips (no source to retime), on
    /// non-positive speeds, and when the retimed extent would overlap a
    /// neighbor.
    pub fn set_clip_speed(
        &mut self,
        clip_id: ClipId,
        speed: Rational,
        reversed: bool,
    ) -> Result<(), ModelError> {
        if speed.num <= 0 || speed.den <= 0 {
            return Err(ModelError::InvalidParam("speed must be positive".into()));
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.freeze_frame {
            return Err(ModelError::InvalidParam(
                "freeze-frame clips cannot be retimed".into(),
            ));
        }
        let Some(source) = clip.source_range() else {
            return Err(ModelError::InvalidParam(
                "speed requires a media-backed clip".into(),
            ));
        };
        let tl_rate = self.timeline.frame_rate;
        let src_dur_tl = resample(source.duration, tl_rate).value;
        // Faster average ⇒ less timeline. A flat ramp keeps the exact integer
        // path (no f64 drift); any active ramp folds in its average.
        let new_dur = super::helpers::retimed_duration(
            src_dur_tl,
            speed,
            clip.speed_curve_average(),
            clip.has_speed_curve(),
        );
        let new_timeline = TimeRange::at_rate(clip.timeline.start.value, new_dur, tl_rate);

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.speed = speed;
        clip.reversed = reversed;
        clip.timeline = new_timeline;
        Ok(())
    }

    /// Set (or clear) a media clip's playback-rate ramp (CapCut speed curves,
    /// M2): keep its timeline start, base `speed`, and source window; store
    /// the normalized `curve` (`None` clears it to a flat unit ramp); and
    /// re-derive the timeline duration from `source ÷ (base_speed ×
    /// average_curve)`. Rejected on generated clips, malformed curves, and
    /// when the retimed extent would overlap a neighbor.
    pub fn set_clip_speed_curve(
        &mut self,
        clip_id: ClipId,
        curve: Option<Param<f32>>,
    ) -> Result<(), ModelError> {
        if let Some(curve) = &curve {
            crate::clip::validate_speed_curve(curve)?;
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.freeze_frame {
            return Err(ModelError::InvalidParam(
                "freeze-frame clips cannot use speed curves".into(),
            ));
        }
        let Some(source) = clip.source_range() else {
            return Err(ModelError::InvalidParam(
                "speed ramps require a media-backed clip".into(),
            ));
        };
        let new_curve = curve.unwrap_or(Param::Constant(1.0));
        let has_curve = !matches!(&new_curve, Param::Constant(v) if *v == 1.0);
        let average = match &new_curve {
            Param::Constant(v) => f64::from(*v),
            Param::Keyframed { .. } => {
                // Reuse the clip's integral over the candidate curve.
                let mut probe = clip.clone();
                probe.speed_curve = new_curve.clone();
                probe.speed_curve_average()
            }
        };

        let tl_rate = self.timeline.frame_rate;
        let src_dur_tl = resample(source.duration, tl_rate).value;
        let new_dur = super::helpers::retimed_duration(src_dur_tl, clip.speed, average, has_curve);
        let new_timeline = TimeRange::at_rate(clip.timeline.start.value, new_dur, tl_rate);

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.speed_curve = new_curve;
        clip.timeline = new_timeline;
        Ok(())
    }

    /// Toggle whether a retimed media clip preserves pitch while it plays
    /// (CapCut's "pitch" switch, M8 Phase 3): `true` time-stretches so the
    /// audio keeps its pitch, `false` lets pitch ride the speed ("chipmunk").
    /// Pure audio property — it changes no duration, so there is no overlap
    /// check. Rejected on generated clips (nothing to hear).
    pub fn set_clip_pitch(
        &mut self,
        clip_id: ClipId,
        preserve_pitch: bool,
    ) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.source_range().is_none() {
            return Err(ModelError::InvalidParam(
                "pitch lock requires a media-backed clip".into(),
            ));
        }
        clip.preserve_pitch = preserve_pitch;
        Ok(())
    }

    /// Toggle noise reduction on a media clip (CapCut "Reduce noise", M8
    /// Phase 5), returning the previous flag for the inverse. The mixers run
    /// the clip's audio through RNNoise when set. Rejected on generated clips
    /// (no source audio to clean).
    pub fn set_clip_denoise(&mut self, clip_id: ClipId, denoise: bool) -> Result<bool, ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.source_range().is_none() {
            return Err(ModelError::InvalidParam(
                "noise reduction requires a media-backed clip".into(),
            ));
        }
        Ok(std::mem::replace(&mut clip.denoise, denoise))
    }

    /// Set a media clip's audio mix (CapCut volume + fades): `volume` is
    /// `Some` to set a flat gain (`0` mutes, `1` unchanged, up to
    /// [`crate::MAX_CLIP_VOLUME`]× boost), overwriting any M8 envelope
    /// (CapCut's basic slider), or `None` to keep the current gain and only
    /// update the fades — so a fade edit never flattens an envelope. Fades
    /// are linear in/out durations at the timeline rate. Rejected on
    /// generated clips (nothing to hear), out-of-range volume, negative
    /// fades, and fades longer than the clip.
    pub fn set_clip_audio(
        &mut self,
        clip_id: ClipId,
        volume: Option<f32>,
        fade_in: RationalTime,
        fade_out: RationalTime,
    ) -> Result<(), ModelError> {
        if let Some(volume) = volume {
            crate::clip::validate_volume(volume)?;
        }
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(fade_in.rate, tl_rate)?;
        check_same_rate(fade_out.rate, tl_rate)?;

        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.is_generated() {
            return Err(ModelError::InvalidParam(
                "audio parameters require a media-backed clip".into(),
            ));
        }
        let duration = clip.timeline.duration.value;
        for (name, fade) in [("fade_in", fade_in.value), ("fade_out", fade_out.value)] {
            if fade < 0 {
                return Err(ModelError::InvalidParam(format!("{name} must be ≥ 0")));
            }
            if fade > duration {
                return Err(ModelError::InvalidParam(format!(
                    "{name} ({fade} ticks) is longer than the clip ({duration} ticks)"
                )));
            }
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        // `Some` is the basic volume slider: a flat level that flattens any
        // envelope (CapCut). `None` keeps the gain (constant or envelope) and
        // touches only the fades, so a fade edit never destroys automation;
        // envelopes are otherwise drawn through the volume keyframe commands
        // (`ClipParam::Volume`).
        if let Some(volume) = volume {
            clip.volume = Param::Constant(volume);
        }
        clip.fade_in = fade_in.value;
        clip.fade_out = fade_out.value;
        Ok(())
    }

    /// Set a clip's framing (CapCut crop, M1): the normalized kept region
    /// plus horizontal/vertical mirroring. Visual clips only — audio has no
    /// frame to crop. Rejected on a degenerate or out-of-frame crop rect.
    ///
    /// `at` composes with animation CapCut-style (same as [`Self::set_transform`]):
    /// `Some(playhead)` writes a keyframe when crop is already animated;
    /// `None` (or a never-animated crop) flattens to a constant.
    pub fn set_clip_crop(
        &mut self,
        clip_id: ClipId,
        crop: CropRect,
        flip_h: bool,
        flip_v: bool,
        at: Option<RationalTime>,
    ) -> Result<(), ModelError> {
        crop.validate()?;
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        if !kind.is_visual() {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        if let Some(at) = at {
            check_same_rate(at.rate, self.timeline.frame_rate)?;
        }
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match at {
            Some(at) if clip.crop.is_animated() => {
                let tick = clip.animation_tick(at.value);
                clip.crop.set_keyframe(tick, crop, Easing::Linear);
            }
            _ => clip.crop.set_constant(crop),
        }
        clip.flip_h = flip_h;
        clip.flip_v = flip_v;
        Ok(())
    }

    /// Replace a clip's detected beat markers (M8 Phase 6), returning the
    /// previous list for the inverse. Beats are source ticks at the media
    /// frame rate; this only stores what detection (or a clear) produced —
    /// the engine owns the analysis. Media clips only (generated content has
    /// no audio to analyze). Stored sorted + de-duplicated and clamped to the
    /// source window so a stale list can't snap to phantom positions.
    pub fn set_clip_beats(
        &mut self,
        clip_id: ClipId,
        beats: Vec<i64>,
    ) -> Result<Vec<i64>, ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let source = clip
            .source_range()
            .ok_or_else(|| ModelError::InvalidParam("beats require a media-backed clip".into()))?;
        let (lo, hi) = (
            source.start.value,
            source.start.value + source.duration.value,
        );
        let mut beats: Vec<i64> = beats
            .into_iter()
            .filter(|&b| (lo..hi).contains(&b))
            .collect();
        beats.sort_unstable();
        beats.dedup();

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        Ok(std::mem::replace(&mut clip.beats, beats))
    }
}
