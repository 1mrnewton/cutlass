//! Silence removal / AutoCut (AI media roadmap M9 Phase 1).
//!
//! "Cut the silences out of this": decode a clip's audio, find the pauses
//! ([`cutlass_decoder::detect_silences`]), and ripple-delete each silent span
//! so the remaining speech closes up. Like the ducking and beat passes the DSP
//! is pure and lives in the decoder; this module owns the decode, the
//! seconds → timeline-tick mapping, and the structural edit.
//!
//! The forward cut reuses the [`split_clip`] and [`ripple_delete`] primitives
//! (so source-window trimming and the gap-close stay correct), but the inverse
//! is a single track-clips snapshot ([`SetTrackClipsAction`]) rather than a
//! composition of the primitives' own inverses: composing those re-mints clip
//! ids on redo, which strands the chained ripple-delete on a stale id. A
//! snapshot swap restores the exact clips (ids included) and oscillates
//! cleanly. The pure plan ([`plan_silence_cuts`]) is split from the structural
//! apply so the tricky parts unit-test without decode.
//!
//! Deliberate gaps (tracked in `docs/ai-media-roadmap.md`): retimed clips are
//! rejected (the seconds → tick mapping is linear only at 1×), and the cut
//! ripples the target clip's own track — linked A/V companions and a
//! whole-timeline magnet ripple ride a follow-up.

use cutlass_decoder::{AUDIO_CHANNELS, AudioReader, SilenceSettings, detect_silences};
use cutlass_models::{Clip, ClipId, ModelError, Project, Rational, RationalTime, TrackId};

use crate::action::edit::{ripple_delete, split_clip};
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Decode rate for silence analysis. Silence detection only needs broadband
/// level over time, so 16 kHz mono (matching the ducking sidechain) is ample
/// and decodes far less than the export rate.
const ANALYSIS_RATE: u32 = 16_000;

/// Detect a clip's silent spans and ripple-delete them. Returns the clip's
/// track (for the edit outcome) and a snapshot inverse restoring the track's
/// clips exactly as they were.
pub fn remove(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    threshold: f32,
    min_silence: f32,
    padding: f32,
) -> Result<(TrackId, Box<dyn EditAction>), EngineError> {
    let target = ctx
        .project
        .clip(clip)
        .ok_or(ModelError::UnknownClip(clip))?;
    if target.is_retimed() {
        return Err(
            ModelError::InvalidParam("AutoCut does not yet support retimed clips".into()).into(),
        );
    }
    let track = ctx
        .project
        .timeline()
        .track_of(clip)
        .ok_or(ModelError::UnknownClip(clip))?;
    let fps = ctx.project.timeline().frame_rate;
    let span = target.timeline;

    let settings = SilenceSettings {
        threshold,
        min_silence,
        keep_padding: padding,
    };

    // Decode + analyze against an immutable view, then mutate.
    let silences = {
        let project: &Project = ctx.project;
        detect_clip_silences(project, clip, settings)?
    };
    let ranges = plan_silence_cuts(span.start.value, span.end_tick(), fps, &silences);
    let inverse = cut_clip(ctx, clip, track, fps, &ranges)?;
    Ok((track, inverse))
}

/// Snapshot the clip's track, apply the silent-span cuts, and return a
/// [`SetTrackClipsAction`] that restores the snapshot. (No-op cut still
/// returns a valid — trivially oscillating — inverse.)
fn cut_clip(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    track: TrackId,
    fps: Rational,
    ranges: &[(i64, i64)],
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = track_clips(ctx.project, track)?;
    apply_cuts(ctx, clip, fps, ranges)?;
    Ok(Box::new(SetTrackClipsAction {
        track,
        clips: before,
    }))
}

/// Map detected silence seconds (from the clip's window start) to absolute
/// timeline tick ranges to ripple-delete. Clamps each span to the clip's own
/// `[start, end)`, drops empties, and merges spans that abut or overlap after
/// frame-rounding. Returns sorted, disjoint ranges. Pure — the linear
/// seconds → ticks mapping holds because retimed clips are rejected upstream.
fn plan_silence_cuts(
    clip_start: i64,
    clip_end: i64,
    fps: Rational,
    silences: &[(f64, f64)],
) -> Vec<(i64, i64)> {
    if fps.num <= 0 || fps.den <= 0 || clip_end <= clip_start {
        return Vec::new();
    }
    let fps_f = f64::from(fps.num) / f64::from(fps.den);
    let mut cuts: Vec<(i64, i64)> = Vec::new();
    for &(s0, s1) in silences {
        if s1 <= s0 {
            continue;
        }
        let a = ((clip_start as f64 + s0 * fps_f).round() as i64).max(clip_start);
        let b = ((clip_start as f64 + s1 * fps_f).round() as i64).min(clip_end);
        if b <= a {
            continue;
        }
        match cuts.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => cuts.push((a, b)),
        }
    }
    cuts
}

/// Ripple-delete the given timeline tick `ranges` (sorted ascending, disjoint,
/// within the clip's span) from `clip`, back to front so earlier ranges'
/// positions stay valid as later ones shift the track left. Each range is the
/// silent middle: split off the right tail (unless it ends at the clip's right
/// edge), split off the middle (unless it starts at the clip's left edge),
/// then ripple-delete the middle. Forward only — the caller owns the inverse.
fn apply_cuts(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    fps: Rational,
    ranges: &[(i64, i64)],
) -> Result<(), EngineError> {
    let span = ctx
        .project
        .clip(clip)
        .ok_or(ModelError::UnknownClip(clip))?
        .timeline;
    let orig_start = span.start.value;
    let mut current_end = span.end_tick();

    for &(a, b) in ranges.iter().rev() {
        if b <= a || a < orig_start || b > current_end {
            continue; // defensive: planner guarantees this never trips
        }
        if b < current_end {
            split_clip::execute(ctx, clip, RationalTime::new(b, fps))?;
        }
        if a > orig_start {
            let (middle, _inverse) = split_clip::execute(ctx, clip, RationalTime::new(a, fps))?;
            ripple_delete::execute(ctx, middle)?;
            current_end = a;
        } else {
            // The working clip [orig_start, b) is entirely silent. This is the
            // earliest range (smallest start), so the clip is consumed last.
            ripple_delete::execute(ctx, clip)?;
            break;
        }
    }
    Ok(())
}

/// Clone the clips currently on `track`, in timeline order.
fn track_clips(project: &Project, track: TrackId) -> Result<Vec<Clip>, EngineError> {
    let track = project
        .timeline()
        .track(track)
        .ok_or(ModelError::UnknownTrack(track))?;
    Ok(track.clips().cloned().collect())
}

/// Replace a track's clips wholesale with a saved set, returning the inverse
/// (the clips it displaced). `add_clip` preserves each clip's id, so this
/// restores the exact pre-cut layout and oscillates as one undo entry.
struct SetTrackClipsAction {
    track: TrackId,
    clips: Vec<Clip>,
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

/// Decode the clip's source window at [`ANALYSIS_RATE`] and run silence
/// detection, returning silent spans in seconds from the window start. Rejects
/// generated clips and media without audio.
fn detect_clip_silences(
    project: &Project,
    clip_id: ClipId,
    settings: SilenceSettings,
) -> Result<Vec<(f64, f64)>, EngineError> {
    let clip = project
        .clip(clip_id)
        .ok_or(ModelError::UnknownClip(clip_id))?;
    let media_id = clip
        .media()
        .ok_or_else(|| ModelError::InvalidParam("AutoCut requires a media clip".into()))?;
    let media = project
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;
    if !media.has_audio {
        return Err(ModelError::InvalidParam("clip media has no audio to analyze".into()).into());
    }
    let source = clip
        .source_range()
        .ok_or_else(|| ModelError::InvalidParam("AutoCut requires a media clip".into()))?;
    let src_start = ticks_to_samples(source.start.value, source.start.rate);
    let src_frames = ticks_to_samples(source.duration.value, source.start.rate);
    if src_frames <= 0 {
        return Ok(Vec::new());
    }
    let mono = read_mono(media.path(), src_start, src_frames)?;
    Ok(detect_silences(&mono, ANALYSIS_RATE, &settings))
}

/// Read `frames` output frames from `src_start` of `path` at the analysis
/// rate, downmixed to mono. Streams in blocks so memory stays bounded.
fn read_mono(path: &std::path::Path, src_start: i64, frames: i64) -> Result<Vec<f32>, EngineError> {
    const BLOCK: usize = 16_384;
    let mut reader = AudioReader::open(path, ANALYSIS_RATE)?;
    reader.seek_to_frame(src_start)?;
    let mut mono = Vec::with_capacity(frames as usize);
    let mut buf = vec![0.0f32; BLOCK * AUDIO_CHANNELS];
    let mut remaining = frames as usize;
    while remaining > 0 {
        let want = remaining.min(BLOCK);
        let got = reader.read(&mut buf[..want * AUDIO_CHANNELS])?;
        if got == 0 {
            break;
        }
        for f in 0..got {
            mono.push((buf[f * AUDIO_CHANNELS] + buf[f * AUDIO_CHANNELS + 1]) * 0.5);
        }
        remaining -= got;
    }
    Ok(mono)
}

/// `value` ticks at `fps` → output frames at [`ANALYSIS_RATE`] (exact i128,
/// floored).
fn ticks_to_samples(value: i64, fps: Rational) -> i64 {
    if fps.num <= 0 || fps.den <= 0 {
        return 0;
    }
    let frames =
        i128::from(value) * i128::from(fps.den) * i128::from(ANALYSIS_RATE) / i128::from(fps.num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::History;
    use cutlass_cache::FrameCache;
    use cutlass_models::{Generator, TimeRange, TrackKind};

    const R24: Rational = Rational::FPS_24;

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    // --- plan_silence_cuts -------------------------------------------------

    #[test]
    fn maps_seconds_to_ticks_clamped_to_the_clip() {
        // Clip [0,48) at 24 fps. A silence at [0.5,1.0) s → ticks [12,24).
        let cuts = plan_silence_cuts(0, 48, R24, &[(0.5, 1.0)]);
        assert_eq!(cuts, vec![(12, 24)]);
    }

    #[test]
    fn anchors_at_the_clip_start_and_clamps_the_tail() {
        // Clip [24,72). A silence at [0.0,0.5) s → [24,36); a silence running
        // past the clip end clamps to 72.
        let cuts = plan_silence_cuts(24, 72, R24, &[(0.0, 0.5), (1.5, 9.0)]);
        assert_eq!(cuts, vec![(24, 36), (60, 72)]);
    }

    #[test]
    fn merges_abutting_spans() {
        // Two spans that round to abutting tick ranges fold into one cut.
        let cuts = plan_silence_cuts(0, 96, R24, &[(0.5, 1.0), (1.0, 1.5)]);
        assert_eq!(cuts, vec![(12, 36)]);
    }

    #[test]
    fn drops_empty_and_bad_input() {
        assert!(plan_silence_cuts(0, 48, R24, &[(1.0, 1.0)]).is_empty());
        assert!(plan_silence_cuts(0, 48, R24, &[(2.0, 1.0)]).is_empty());
        assert!(plan_silence_cuts(0, 0, R24, &[(0.0, 1.0)]).is_empty());
    }

    // --- cut_clip (structural compose + snapshot inverse) ------------------

    fn setup() -> (tempfile::TempDir, Project, FrameCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = FrameCache::new(dir.path().join("cache"), 1024 * 1024).unwrap();
        let project = Project::new("autocut", R24);
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
        let inverse = cut_clip(&mut ctx, c, track, R24, &[(12, 24)]).unwrap();
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
    fn cuts_leading_and_trailing_silence() {
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
        let inverse = cut_clip(&mut ctx, c, track, R24, &[(0, 12), (36, 48)]).unwrap();
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
}
