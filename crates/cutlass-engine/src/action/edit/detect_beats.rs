//! Beat detection (audio roadmap M8 Phase 6).
//!
//! "Snap my cuts to the beat": decode a clip's audio, run onset/tempo analysis
//! ([`cutlass_decoder::detect_beats`]), and store the resulting grid on the
//! clip as source-tick beat markers the timeline magnet snaps to. Like the
//! ducking pass, the DSP is pure and lives in the decoder; this module owns the
//! decode and the seconds→source-tick mapping, and commits through
//! [`Project::set_clip_beats`] with a clip-snapshot inverse.

use cutlass_models::{ClipId, ModelError, Project, Rational};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

#[cfg(not(engine_native_decode))]
use cutlass_decoder::{AUDIO_CHANNELS, AudioReader, detect_beats as detect_beats_dsp};

/// Decode rate for beat analysis. 22.05 kHz resolves percussive transients
/// well while decoding less than half the export rate — onset detection only
/// needs to know *when* energy arrives, not reproduce full-bandwidth audio.
#[cfg(not(engine_native_decode))]
const ANALYSIS_RATE: u32 = 22_050;

/// Detect a media clip's beats and store them. Returns a clip-snapshot inverse.
#[cfg(not(engine_native_decode))]
pub fn detect(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
) -> Result<Box<dyn EditAction>, EngineError> {
    let beats = analyze(ctx.project, clip)?;
    commit(ctx, clip, beats)
}

/// Native build: audio decode (and thus beat analysis) isn't wired up yet.
#[cfg(engine_native_decode)]
pub fn detect(
    _ctx: &mut ApplyContext<'_>,
    _clip: ClipId,
) -> Result<Box<dyn EditAction>, EngineError> {
    Err(ModelError::InvalidParam("beat detection is unavailable in this build".into()).into())
}

/// Clear a clip's beats. Returns a clip-snapshot inverse.
pub fn clear(ctx: &mut ApplyContext<'_>, clip: ClipId) -> Result<Box<dyn EditAction>, EngineError> {
    commit(ctx, clip, Vec::new())
}

/// Snapshot the clip, set its beats (sorted/clamped by the model), and return
/// the snapshot as the undo inverse.
fn commit(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    beats: Vec<i64>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    ctx.project.set_clip_beats(clip, beats)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

/// Decode the clip's source window and run beat detection, returning beat
/// positions as source ticks at the media frame rate. Rejects generated clips
/// and media without audio (nothing to analyze).
#[cfg(not(engine_native_decode))]
fn analyze(project: &Project, clip_id: ClipId) -> Result<Vec<i64>, EngineError> {
    let clip = project
        .clip(clip_id)
        .ok_or(ModelError::UnknownClip(clip_id))?;
    let media_id = clip
        .media()
        .ok_or_else(|| ModelError::InvalidParam("beat detection requires a media clip".into()))?;
    let media = project
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;
    if !media.has_audio {
        return Err(ModelError::InvalidParam("clip media has no audio to analyze".into()).into());
    }
    let source = clip
        .source_range()
        .ok_or_else(|| ModelError::InvalidParam("beat detection requires a media clip".into()))?;

    let src_start = ticks_to_samples(source.start.value, source.start.rate);
    let src_frames = ticks_to_samples(source.duration.value, source.start.rate);
    if src_frames <= 0 {
        return Ok(Vec::new());
    }
    let mono = read_mono(media.path(), src_start, src_frames)?;
    let seconds = detect_beats_dsp(&mono, ANALYSIS_RATE);
    Ok(beats_to_source_ticks(
        &seconds,
        source.start.value,
        source.start.rate,
    ))
}

/// Map detected beat seconds (from the window start) to source ticks at the
/// media frame rate, anchored at the clip's source in-point. Pure.
#[cfg(not(engine_native_decode))]
fn beats_to_source_ticks(seconds: &[f64], source_start: i64, fps: Rational) -> Vec<i64> {
    if fps.num <= 0 || fps.den <= 0 {
        return Vec::new();
    }
    let fps_f = f64::from(fps.num) / f64::from(fps.den);
    seconds
        .iter()
        .map(|&s| source_start + (s * fps_f).round() as i64)
        .collect()
}

/// Read `frames` output frames from `src_start` of `path` at the analysis
/// rate, downmixed to mono. Streams in blocks so memory stays bounded.
#[cfg(not(engine_native_decode))]
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
#[cfg(not(engine_native_decode))]
fn ticks_to_samples(value: i64, fps: Rational) -> i64 {
    if fps.num <= 0 || fps.den <= 0 {
        return 0;
    }
    let frames =
        i128::from(value) * i128::from(fps.den) * i128::from(ANALYSIS_RATE) / i128::from(fps.num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(all(test, not(engine_native_decode)))]
mod tests {
    use super::*;

    #[test]
    fn beats_to_source_ticks_anchors_and_scales() {
        // At 24 fps, beats at 0/0.5/1.0 s from a window starting at source
        // tick 100 land at 100, 112, 124.
        let ticks = beats_to_source_ticks(&[0.0, 0.5, 1.0], 100, Rational::FPS_24);
        assert_eq!(ticks, vec![100, 112, 124]);
    }

    #[test]
    fn beats_to_source_ticks_handles_empty_and_bad_rate() {
        assert!(beats_to_source_ticks(&[], 0, Rational::FPS_24).is_empty());
        assert!(beats_to_source_ticks(&[1.0], 0, Rational::new(0, 1)).is_empty());
    }

    #[test]
    fn ticks_to_samples_is_exact_for_integer_rates() {
        // 24 ticks (1 s at 24 fps) = ANALYSIS_RATE frames.
        assert_eq!(
            ticks_to_samples(24, Rational::FPS_24),
            i64::from(ANALYSIS_RATE)
        );
        assert_eq!(ticks_to_samples(0, Rational::FPS_24), 0);
    }
}
