//! Sidechain audio ducking (audio roadmap M8 Phase 4).
//!
//! "Duck the music under the narration": measure speech-band energy on the
//! chosen *voice* clips, then dip each *music* clip's volume while the voice
//! is present. The result is written as **ordinary M8 volume keyframes** —
//! exactly what the inspector draws and the agent edits — so ducking is
//! inspectable and tunable after the fact, not a hidden processor. The whole
//! pass is one undo entry: a [`CompoundAction`] of per-clip restores.
//!
//! Split in two: [`plan_ducking`] is the pure timeline math (assemble a shared
//! energy track, run the ducker, thin each music clip's gain slice into
//! keyframes) and takes a closure for "decode a clip's energy", so it tests
//! with synthetic input; [`duck`] supplies the real decode (an
//! [`AudioReader`] at a reduced analysis rate) and commits the plan.

use cutlass_decoder::{
    AUDIO_CHANNELS, AudioReader, CONTROL_HZ, DuckSettings, duck_gain, reduce_curve,
    speech_band_energy,
};
use cutlass_models::{
    Clip, ClipId, Easing, Keyframe, MAX_CLIP_VOLUME, ModelError, Param, Project, Rational,
};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, CompoundAction, EditAction};
use crate::error::EngineError;

/// Decode rate for ducking analysis. The speech band tops out at 3.4 kHz, so
/// 16 kHz (Nyquist 8 kHz) captures it with headroom while decoding far less
/// than the 48 kHz export rate — the analysis only needs to know *when* the
/// voice is loud, not reproduce it.
const ANALYSIS_RATE: u32 = 16_000;

/// Keyframe-thinning tolerance, in gain units. ~0.02 keeps the duck shape's
/// corners while collapsing the long unity stretches between phrases.
const KEYFRAME_TOLERANCE: f32 = 0.02;

/// A music clip is left untouched when its deepest dip is shallower than this
/// (the voice never crosses the threshold over its span): no point writing a
/// flat-unity envelope or burning an undo slot.
const MIN_DUCK_DEPTH: f32 = 1.0e-3;

/// Analyze the `voice` clips and write ducking volume keyframes onto the
/// `music` clips. Returns the first music clip (for the edit outcome) and a
/// compound inverse restoring every clip the pass actually changed.
pub fn duck(
    ctx: &mut ApplyContext<'_>,
    voice: &[ClipId],
    music: &[ClipId],
    threshold: f32,
    amount: f32,
    attack: f32,
    release: f32,
) -> Result<(ClipId, Box<dyn EditAction>), EngineError> {
    let representative = *music
        .first()
        .ok_or_else(|| ModelError::InvalidParam("ducking needs at least one music clip".into()))?;
    if voice.is_empty() {
        return Err(
            ModelError::InvalidParam("ducking needs at least one voice clip".into()).into(),
        );
    }

    let settings = DuckSettings {
        threshold,
        amount,
        attack,
        release,
    };

    // Plan against an immutable view (decode happens here), then commit.
    let plans = {
        let project: &Project = ctx.project;
        plan_ducking(project, voice, music, settings, |clip| {
            clip_energy(clip, project)
        })?
    };

    let mut inverses: Vec<Box<dyn EditAction>> = Vec::with_capacity(plans.len());
    for (id, new_volume) in plans {
        let before = ctx
            .project
            .clip(id)
            .cloned()
            .ok_or(ModelError::UnknownClip(id))?;
        let slot = ctx
            .project
            .timeline_mut()
            .clip_mut(id)
            .ok_or(ModelError::UnknownClip(id))?;
        slot.volume = new_volume;
        inverses.push(Box::new(RestoreClipAction { clip: before }));
    }
    Ok((
        representative,
        Box::new(CompoundAction { actions: inverses }),
    ))
}

/// Pure ducking plan: returns the `(clip, new volume envelope)` for each music
/// clip the voice actually ducks. `energy_of` yields a control-rate
/// ([`CONTROL_HZ`]) speech-band energy envelope for a clip, in the clip's own
/// (source) time; placement onto the timeline, the ducker, and keyframe
/// thinning all live here so the decode stays swappable for tests.
pub(crate) fn plan_ducking(
    project: &Project,
    voice: &[ClipId],
    music: &[ClipId],
    settings: DuckSettings,
    mut energy_of: impl FnMut(&Clip) -> Result<Vec<f32>, EngineError>,
) -> Result<Vec<(ClipId, Param<f32>)>, EngineError> {
    let fps = project.timeline().frame_rate;

    // The shared energy track spans tick 0 to the last edge of any clip we
    // touch — so the ducker's attack/release run continuously across gaps.
    let mut total_steps = 0i64;
    for &id in voice.iter().chain(music) {
        let clip = project.clip(id).ok_or(ModelError::UnknownClip(id))?;
        total_steps = total_steps.max(tick_to_step(clip.timeline.end_tick(), fps));
    }
    if total_steps <= 0 {
        return Ok(Vec::new());
    }
    let mut energy = vec![0.0f32; total_steps as usize];

    // Composite each voice clip's energy onto the track by loudest-wins, so
    // overlapping talkers reinforce rather than cancel.
    for &id in voice {
        let clip = project.clip(id).ok_or(ModelError::UnknownClip(id))?;
        let clip_energy = energy_of(clip)?;
        if clip_energy.is_empty() {
            continue;
        }
        let start = tick_to_step(clip.timeline.start.value, fps).max(0);
        let span = (tick_to_step(clip.timeline.end_tick(), fps) - start).max(0);
        place_energy(&mut energy, start as usize, span as usize, &clip_energy);
    }

    let gain = duck_gain(&energy, &settings);

    let mut plans = Vec::new();
    for &id in music {
        let clip = project.clip(id).ok_or(ModelError::UnknownClip(id))?;
        let start = tick_to_step(clip.timeline.start.value, fps).max(0);
        let end = tick_to_step(clip.timeline.end_tick(), fps).max(start);
        let slice = &gain[(start as usize).min(gain.len())..(end as usize).min(gain.len())];
        if slice.is_empty() {
            continue;
        }
        let deepest = slice.iter().copied().fold(1.0f32, f32::min);
        if deepest > 1.0 - MIN_DUCK_DEPTH {
            continue; // voice never ducks this clip
        }

        // Thin the dip to its corners, then rebase each onto a clip-relative
        // tick and scale by the clip's own level there (so a set volume or a
        // prior envelope is preserved, dipped — not overwritten flat).
        let mut keyframes: Vec<Keyframe<f32>> = Vec::new();
        for (idx, g) in reduce_curve(slice, KEYFRAME_TOLERANCE) {
            let abs_step = start + idx as i64;
            let tick = (step_to_tick(abs_step, fps) - clip.timeline.start.value).max(0);
            let base = clip.volume.sample(tick);
            let value = (base * g).clamp(0.0, MAX_CLIP_VOLUME);
            match keyframes.last_mut() {
                // Control rate can outrun the tick grid; fold collisions.
                Some(last) if last.tick == tick => last.value = value,
                _ => keyframes.push(Keyframe {
                    tick,
                    value,
                    easing: Easing::Linear,
                tangents: None,
            }),
            }
        }
        let volume = match keyframes.len() {
            0 => continue,
            1 => Param::Constant(keyframes[0].value),
            _ => Param::Keyframed { keyframes },
        };
        plans.push((id, volume));
    }
    Ok(plans)
}

/// Loudest-wins placement of a clip's (source-time) energy envelope onto the
/// shared timeline track: stretch `clip_energy` across the clip's `span`
/// control steps starting at `start`. For an un-retimed clip `span` ≈
/// `clip_energy.len()`, so this is essentially a copy.
fn place_energy(track: &mut [f32], start: usize, span: usize, clip_energy: &[f32]) {
    if span == 0 || clip_energy.is_empty() {
        return;
    }
    let n = clip_energy.len();
    for k in 0..span {
        let dst = start + k;
        if dst >= track.len() {
            break;
        }
        let src = if span == 1 {
            0.0
        } else {
            k as f32 / (span - 1) as f32 * (n - 1) as f32
        };
        track[dst] = track[dst].max(sample_linear(clip_energy, src));
    }
}

/// Linear interpolation into `samples` at fractional index `x` (clamped).
fn sample_linear(samples: &[f32], x: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let x = x.clamp(0.0, (samples.len() - 1) as f32);
    let i = x.floor() as usize;
    if i + 1 >= samples.len() {
        return samples[samples.len() - 1];
    }
    let frac = x - i as f32;
    samples[i] + (samples[i + 1] - samples[i]) * frac
}

/// Decode a clip's source window at [`ANALYSIS_RATE`] and reduce it to a
/// speech-band energy envelope. Empty (skipped) for clips with no audio media;
/// fail-loud on a decode error, like export.
fn clip_energy(clip: &Clip, project: &Project) -> Result<Vec<f32>, EngineError> {
    let Some(media_id) = clip.media() else {
        return Ok(Vec::new());
    };
    let Some(media) = project.media(media_id) else {
        return Ok(Vec::new());
    };
    if !media.has_audio {
        return Ok(Vec::new());
    }
    let Some(source) = clip.source_range() else {
        return Ok(Vec::new());
    };
    let src_start = ticks_to_samples(
        source.start.value,
        source.start.rate.num,
        source.start.rate.den,
    );
    let src_frames = ticks_to_samples(
        source.duration.value,
        source.start.rate.num,
        source.start.rate.den,
    );
    if src_frames <= 0 {
        return Ok(Vec::new());
    }
    let mono = read_mono(media.path(), src_start, src_frames)?;
    Ok(speech_band_energy(&mono, ANALYSIS_RATE))
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

/// `tick` (timeline rate `fps`) → analysis control step at [`CONTROL_HZ`].
fn tick_to_step(tick: i64, fps: Rational) -> i64 {
    if fps.num <= 0 || fps.den <= 0 {
        return 0;
    }
    let secs = tick as f64 * f64::from(fps.den) / f64::from(fps.num);
    (secs * f64::from(CONTROL_HZ)).round() as i64
}

/// Control step → timeline `tick` (inverse of [`tick_to_step`]).
fn step_to_tick(step: i64, fps: Rational) -> i64 {
    if fps.num <= 0 || fps.den <= 0 {
        return 0;
    }
    let secs = step as f64 / f64::from(CONTROL_HZ);
    (secs * f64::from(fps.num) / f64::from(fps.den)).round() as i64
}

/// `value` ticks at `num/den` fps → output frames at [`ANALYSIS_RATE`] (exact
/// i128, floored).
fn ticks_to_samples(value: i64, num: i32, den: i32) -> i64 {
    if num <= 0 || den <= 0 {
        return 0;
    }
    let frames = i128::from(value) * i128::from(den) * i128::from(ANALYSIS_RATE) / i128::from(num);
    frames.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{MediaSource, RationalTime, TimeRange, TrackKind};

    /// Project with one voice lane + one music lane, each carrying a 2 s clip
    /// (48 ticks at 24 fps) of audio-bearing media over `[0, 2)` s.
    fn voice_and_music() -> (Project, ClipId, ClipId) {
        let mut project = Project::new("duck", Rational::FPS_24);
        let media = project.add_media(MediaSource::new(
            "/tmp/a.wav",
            0,
            0,
            Rational::FPS_24,
            48,
            true,
        ));
        let v_lane = project.add_track(TrackKind::Audio, "V");
        let m_lane = project.add_track(TrackKind::Audio, "M");
        let voice = project
            .add_clip(
                v_lane,
                media,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let music = project
            .add_clip(
                m_lane,
                media,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        (project, voice, music)
    }

    fn loud_settings() -> DuckSettings {
        DuckSettings {
            threshold: 0.02,
            amount: 0.7,
            attack: 0.05,
            release: 0.2,
        }
    }

    #[test]
    fn loud_voice_writes_a_dipped_envelope_on_the_music() {
        let (project, voice, music) = voice_and_music();
        // Voice is loud across its whole span; music sits under it.
        let plans = plan_ducking(&project, &[voice], &[music], loud_settings(), |_clip| {
            Ok(vec![0.5f32; 200])
        })
        .unwrap();

        assert_eq!(plans.len(), 1, "the music clip ducks");
        let (id, volume) = &plans[0];
        assert_eq!(*id, music);
        assert!(volume.is_animated(), "ducking writes a volume envelope");
        // The deepest keyframe approaches the floor (1 - amount = 0.3).
        let deepest = volume
            .keyframes()
            .iter()
            .map(|k| k.value)
            .fold(f32::INFINITY, f32::min);
        assert!(deepest < 0.5, "music dips, got {deepest}");
        // First keyframe sits at clip-relative tick 0; it never goes negative.
        assert_eq!(volume.keyframes()[0].tick, 0);
        assert!(volume.keyframes().iter().all(|k| k.value >= 0.0));
        // Strictly sorted, valid envelope.
        assert!(volume.validate_shape().is_ok());
    }

    #[test]
    fn quiet_voice_leaves_music_untouched() {
        let (project, voice, music) = voice_and_music();
        // Energy below threshold: nothing ducks, no plan.
        let plans = plan_ducking(&project, &[voice], &[music], loud_settings(), |_clip| {
            Ok(vec![0.001f32; 200])
        })
        .unwrap();
        assert!(plans.is_empty(), "no dip → no keyframes written");
    }

    #[test]
    fn ducking_scales_the_clips_existing_level() {
        let (mut project, voice, music) = voice_and_music();
        // Music already sits at 0.5: the ducked floor rides on top of it.
        project
            .set_clip_audio(
                music,
                Some(0.5),
                RationalTime::new(0, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let plans = plan_ducking(&project, &[voice], &[music], loud_settings(), |_clip| {
            Ok(vec![0.5f32; 200])
        })
        .unwrap();
        let (_, volume) = &plans[0];
        let peak = volume
            .keyframes()
            .iter()
            .map(|k| k.value)
            .fold(0.0f32, f32::max);
        assert!(
            peak <= 0.5 + 1e-4,
            "never exceeds the clip's own level: {peak}"
        );
        let deepest = volume
            .keyframes()
            .iter()
            .map(|k| k.value)
            .fold(f32::INFINITY, f32::min);
        assert!(
            deepest < 0.5 * 0.5 + 0.05,
            "dips below the base level: {deepest}"
        );
    }

    #[test]
    fn non_overlapping_voice_does_not_duck() {
        let mut project = Project::new("duck", Rational::FPS_24);
        let media = project.add_media(MediaSource::new(
            "/tmp/a.wav",
            0,
            0,
            Rational::FPS_24,
            48,
            true,
        ));
        let v_lane = project.add_track(TrackKind::Audio, "V");
        let m_lane = project.add_track(TrackKind::Audio, "M");
        // Voice in [0,2)s, music in [4,6)s — no temporal overlap.
        let voice = project
            .add_clip(
                v_lane,
                media,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let music = project
            .add_clip(
                m_lane,
                media,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(96, Rational::FPS_24),
            )
            .unwrap();
        let plans = plan_ducking(&project, &[voice], &[music], loud_settings(), |_clip| {
            Ok(vec![0.5f32; 200])
        })
        .unwrap();
        assert!(plans.is_empty(), "music outside the voice keeps its level");
    }
}
