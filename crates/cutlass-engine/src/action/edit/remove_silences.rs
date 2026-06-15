//! Silence removal / AutoCut (AI media roadmap M9 Phase 1).
//!
//! "Cut the silences out of this": decode a clip's audio, find the pauses
//! ([`cutlass_decoder::detect_silences`]), and ripple-delete each silent span
//! so the remaining speech closes up. Like the ducking and beat passes the DSP
//! is pure and lives in the decoder, and the decode is shared
//! ([`crate::clip_audio`]); this module owns only the seconds → timeline-tick
//! mapping. The structural ripple-delete is the shared
//! [`ripple_cut`](crate::action::edit::ripple_cut) primitive (also used by
//! transcript editing). The pure plan ([`plan_silence_cuts`]) is split from the
//! cut so the tricky parts unit-test without decode.
//!
//! Deliberate gaps (tracked in `docs/ai-media-roadmap.md`): retimed clips are
//! rejected (the seconds → tick mapping is linear only at 1×), and the cut
//! ripples the target clip's own track — linked A/V companions and a
//! whole-timeline magnet ripple ride a follow-up.

use cutlass_decoder::{SilenceSettings, detect_silences};
use cutlass_models::{ClipId, ModelError, Project, Rational, TrackId};

use crate::action::edit::ripple_cut;
use crate::action::{ApplyContext, EditAction};
use crate::clip_audio::{self, ANALYSIS_RATE};
use crate::error::EngineError;

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
    let inverse = ripple_cut::cut_ranges(ctx, track, fps, &ranges)?;
    Ok((track, inverse))
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

/// Decode the clip's source window at the analysis rate and run silence
/// detection, returning silent spans in seconds from the window start. Rejects
/// generated clips and media without audio.
fn detect_clip_silences(
    project: &Project,
    clip_id: ClipId,
    settings: SilenceSettings,
) -> Result<Vec<(f64, f64)>, EngineError> {
    let mono = clip_audio::decode_clip_mono(project, clip_id)?;
    Ok(detect_silences(&mono, ANALYSIS_RATE, &settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    const R24: Rational = Rational::FPS_24;

    // --- plan_silence_cuts (the seconds → ticks mapping; the structural cut
    // it feeds is tested in `ripple_cut`) -------------------------------------

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
}
