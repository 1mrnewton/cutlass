//! Generalized ripple-cut: remove a set of absolute timeline tick ranges from a
//! single track, splitting any clip that straddles a range edge and
//! ripple-deleting the clips that fall inside each range so the tail closes up.
//!
//! This is the structural core shared by silence removal (M9 Phase 1 — ranges
//! come from a DSP pass) and transcript editing (M9 Phase 3 — ranges come from
//! the words the user struck out). Unlike a single-clip cut it resolves the
//! affected clips by *timeline position*, so it works after earlier edits have
//! already split the region into several clips — exactly the transcript case,
//! where each delete narrows a region built from many surviving spans.
//!
//! The forward pass reuses the [`split_clip`] and [`ripple_delete`] primitives
//! (so source-window trimming and the gap-close stay correct), but the inverse
//! is a single track-clips snapshot ([`SetTrackClipsAction`]) rather than a
//! composition of those primitives' own inverses: composing them re-mints clip
//! ids on redo, which strands a later chained delete on a stale id. A snapshot
//! swap restores the exact clips (ids included) and oscillates cleanly.

use cutlass_models::{Clip, ModelError, Project, Rational, RationalTime, TrackId};

use crate::action::edit::{ripple_delete, split_clip};
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Ripple-delete `ranges` (absolute timeline ticks) from `track`. Ranges need
/// not be sorted and empties are ignored. Snapshots the track up front and
/// returns a [`SetTrackClipsAction`] restoring it exactly (clip ids included)
/// for a clean one-entry undo that oscillates. A no-op cut still returns a
/// valid — trivially oscillating — inverse.
pub fn cut_ranges(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    fps: Rational,
    ranges: &[(i64, i64)],
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = track_clips(ctx.project, track)?;
    // Process back-to-front so earlier ranges' tick coordinates stay valid as
    // each later (downstream) cut ripples the tail left.
    let mut sorted: Vec<(i64, i64)> = ranges.iter().copied().filter(|&(a, b)| b > a).collect();
    sorted.sort_unstable();
    for (a, b) in sorted.into_iter().rev() {
        cut_one(ctx, track, fps, a, b)?;
    }
    Ok(Box::new(SetTrackClipsAction {
        track,
        clips: before,
    }))
}

/// Cut a single range `[a, b)`: split the clips straddling each edge so the
/// interior is whole clips, then ripple-delete those interior clips
/// highest-start first (so each delete only ripples clips downstream of the
/// range, leaving the remaining interior clips' positions valid).
fn cut_one(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    fps: Rational,
    a: i64,
    b: i64,
) -> Result<(), EngineError> {
    split_at(ctx, track, fps, b)?;
    split_at(ctx, track, fps, a)?;

    let mut interior: Vec<(i64, cutlass_models::ClipId)> = ctx
        .project
        .timeline()
        .track(track)
        .ok_or(ModelError::UnknownTrack(track))?
        .clips()
        .filter(|c| c.timeline.start.value >= a && c.timeline.end_tick() <= b)
        .map(|c| (c.timeline.start.value, c.id))
        .collect();
    interior.sort_unstable_by(|x, y| y.0.cmp(&x.0));
    for (_, id) in interior {
        ripple_delete::execute(ctx, id)?;
    }
    Ok(())
}

/// Split the clip straddling tick `t` (its start strictly before `t`) at `t`.
/// A `t` on a clip boundary or in a gap needs no split.
fn split_at(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    fps: Rational,
    t: i64,
) -> Result<(), EngineError> {
    let pos = RationalTime::new(t, fps);
    let straddler = ctx
        .project
        .timeline()
        .track(track)
        .ok_or(ModelError::UnknownTrack(track))?
        .clip_at(pos)?
        .filter(|c| c.timeline.start.value < t)
        .map(|c| c.id);
    if let Some(id) = straddler {
        split_clip::execute(ctx, id, pos)?;
    }
    Ok(())
}

/// Clone the clips currently on `track`.
pub fn track_clips(project: &Project, track: TrackId) -> Result<Vec<Clip>, EngineError> {
    let track = project
        .timeline()
        .track(track)
        .ok_or(ModelError::UnknownTrack(track))?;
    Ok(track.clips().cloned().collect())
}

/// Replace a track's clips wholesale with a saved set, returning the inverse
/// (the clips it displaced). `add_clip` preserves each clip's id, so this
/// restores the exact pre-cut layout and oscillates as one undo entry.
pub struct SetTrackClipsAction {
    pub track: TrackId,
    pub clips: Vec<Clip>,
}

impl EditAction for SetTrackClipsAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let previous = track_clips(ctx.project, self.track)?;
        for clip in &previous {
            ctx.project.timeline_mut().remove_clip(clip.id);
        }
        for clip in self.clips {
            ctx.project.timeline_mut().add_clip(self.track, clip)?;
        }
        Ok(Box::new(SetTrackClipsAction {
            track: self.track,
            clips: previous,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::History;
    use cutlass_cache::FrameCache;
    use cutlass_models::{Clip, Generator, TimeRange, TrackKind};

    const R24: Rational = Rational::FPS_24;

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    fn setup() -> (tempfile::TempDir, Project, FrameCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = FrameCache::new(dir.path().join("cache"), 1024 * 1024).unwrap();
        let project = Project::new("ripple-cut", R24);
        (dir, project, cache)
    }

    #[test]
    fn cuts_a_middle_span_and_ripples_downstream() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        // C [0,48), then D [48,68) downstream on the same track.
        let c = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 48)))
            .unwrap();
        let d = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(48, 20)))
            .unwrap();

        let mut path = None;
        let mut history = History::new(32);
        let mut ctx = ApplyContext {
            project: &mut project,
            cache: &cache,
            project_path: &mut path,
            history: &mut history,
        };

        // Cut [12,24): C shrinks to [0,12), its tail [24,48) shifts to [12,36),
        // and D shifts left by 12 to [36,56).
        let inverse = cut_ranges(&mut ctx, track, R24, &[(12, 24)]).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().timeline, tr(0, 12));
        assert_eq!(ctx.project.clip(d).unwrap().start().value, 36);
        assert_eq!(ctx.project.timeline().clip_count(), 3);

        // Undo restores the original layout (clip ids included).
        let redo = inverse.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().timeline, tr(0, 48));
        assert_eq!(ctx.project.clip(d).unwrap().timeline, tr(48, 20));
        assert_eq!(ctx.project.timeline().clip_count(), 2);

        // Redo cuts again and oscillates.
        let _ = redo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().timeline, tr(0, 12));
        assert_eq!(ctx.project.clip(d).unwrap().start().value, 36);
    }

    #[test]
    fn cuts_leading_and_trailing_span() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let c = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 48)))
            .unwrap();

        let mut path = None;
        let mut history = History::new(32);
        let mut ctx = ApplyContext {
            project: &mut project,
            cache: &cache,
            project_path: &mut path,
            history: &mut history,
        };

        // Trim [0,12) off the front and [36,48) off the back: only [12,36)
        // survives, ripple-anchored back to tick 0 → [0,24).
        let inverse = cut_ranges(&mut ctx, track, R24, &[(0, 12), (36, 48)]).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        let survivor = ctx
            .project
            .timeline()
            .track(track)
            .unwrap()
            .clips()
            .next()
            .unwrap();
        assert_eq!(survivor.timeline, tr(0, 24));

        let _ = inverse.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().timeline, tr(0, 48));
        assert_eq!(ctx.project.timeline().clip_count(), 1);
    }

    #[test]
    fn cuts_a_span_crossing_two_existing_clips() {
        // After an earlier edit the region is two abutting clips C[0,24) D[24,48).
        // A cut [12,36) crosses the C/D boundary: split both, delete the two
        // interior halves, leaving C[0,12) + D's tail rippled to [12,24).
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 24)))
            .unwrap();
        project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(24, 24)))
            .unwrap();

        let mut path = None;
        let mut history = History::new(32);
        let mut ctx = ApplyContext {
            project: &mut project,
            cache: &cache,
            project_path: &mut path,
            history: &mut history,
        };

        let inverse = cut_ranges(&mut ctx, track, R24, &[(12, 36)]).unwrap();
        let spans: Vec<TimeRange> = ctx
            .project
            .timeline()
            .track(track)
            .unwrap()
            .clips_ordered()
            .iter()
            .map(|c| c.timeline)
            .collect();
        assert_eq!(spans, vec![tr(0, 12), tr(12, 12)]);

        let _ = inverse.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);
    }
}
