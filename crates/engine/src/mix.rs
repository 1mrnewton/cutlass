//! One mix of decoded PCM.
//!
//! Each audible clip is resampled, stretched, gained, and faded through
//! `cutlass-audio`. A muted track contributes no samples: mute is not a gain
//! of zero. Text and shapes are refused before any of that runs. The limiter
//! runs on the sum.

use cutlass_audio::{FadeCurve, fade, limit, resample, stretch};
use cutlass_codec::access::{ChannelLayout, Pcm};
use cutlass_model::{Draft, MaterialRef, Micros, Segment, Track, TrackAttribute};

use crate::clock::ClockKind;
use crate::error::EngineError;
use crate::map;

/// Whether the mix may hear this clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipRole {
    Sound,
    Text,
    Shape,
}

/// PCM for one clip, or a hole the clock has not filled.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipPcm {
    Ready(Pcm),
    Missing,
}

/// One clip the mix can add.
#[derive(Clone, Debug, PartialEq)]
pub struct MixClip {
    pub pcm: ClipPcm,
    pub speed: f64,
    pub pitch_follows_speed: bool,
    pub volume: f64,
    pub fade_in_us: i64,
    pub fade_out_us: i64,
    pub fade_curve: i32,
    pub muted: bool,
    pub role: ClipRole,
}

/// Mix rate and limiter kind. Kind 0 leaves the sum alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mix {
    pub rate: u32,
    pub limiter: i32,
}

/// The summed buffer and how many clips added samples.
///
/// A muted clip is not counted. A clip at volume zero is counted: it went
/// through the gain path and added silence of that length.
#[derive(Clone, Debug, PartialEq)]
pub struct MixResult {
    pub pcm: Pcm,
    pub contributed: usize,
}

/// Ready PCM, or nothing when playback has not received the samples.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioSample {
    Ready(Pcm),
    Missing,
}

/// Segments that may enter the mix at `time`. Text and shapes are left out.
#[must_use]
pub fn mix_inputs(draft: &Draft, time: Micros) -> Vec<MixClip> {
    let mut clips = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if map::is_graphic(draft, segment) {
                continue;
            }
            if !covers(&segment.target, time) {
                continue;
            }
            clips.push(clip_for(draft, track, segment, time));
        }
    }
    clips
}

/// Volume at a keyframe offset. An empty list uses `base`.
pub fn volume_at(base: f64, keys: &[(Micros, f64)], at: Micros) -> Result<f64, EngineError> {
    if !base.is_finite() {
        return Err(EngineError::Audio("non-finite volume".to_owned()));
    }
    if keys.is_empty() {
        return Ok(base);
    }
    let mut ordered: Vec<(i64, f64, usize)> = keys
        .iter()
        .enumerate()
        .map(|(index, (time, value))| (time.0, *value, index))
        .collect();
    ordered.sort_by(|left, right| left.0.cmp(&right.0).then(left.2.cmp(&right.2)));
    let mut collapsed: Vec<(i64, f64)> = Vec::new();
    for (time, value, _) in ordered {
        if !value.is_finite() {
            return Err(EngineError::Audio("non-finite volume".to_owned()));
        }
        if let Some(last) = collapsed.last_mut()
            && last.0 == time
        {
            last.1 = value;
            continue;
        }
        collapsed.push((time, value));
    }
    let Some((first_time, first)) = collapsed.first().copied() else {
        return Ok(base);
    };
    if at.0 <= first_time {
        return Ok(first);
    }
    let Some((last_time, last)) = collapsed.last().copied() else {
        return Ok(base);
    };
    if at.0 >= last_time {
        return Ok(last);
    }
    for pair in collapsed.windows(2) {
        let (left_time, left) = pair[0];
        let (right_time, right) = pair[1];
        if at.0 >= left_time && at.0 <= right_time {
            let span = (right_time - left_time) as f64;
            if span == 0.0 {
                return Ok(right);
            }
            let t = (at.0 - left_time) as f64 / span;
            return Ok(left + (right - left) * t);
        }
    }
    Ok(last)
}

/// Resample, stretch, gain, fade, sum, then limit.
///
/// Playback skips a missing clip. Export refuses the mix until every audible
/// clip is ready. Mute skips the clip before the audio crate sees it.
pub fn render(mix: &Mix, clips: &[MixClip], clock: ClockKind) -> Result<MixResult, EngineError> {
    if mix.rate == 0 {
        return Err(EngineError::NoClock);
    }
    let mut buffers = Vec::new();
    for clip in clips {
        if clip.role != ClipRole::Sound {
            return Err(EngineError::NotMedia);
        }
        if clip.muted {
            continue;
        }
        match &clip.pcm {
            ClipPcm::Missing if clock == ClockKind::Export => return Err(EngineError::NotReady),
            ClipPcm::Missing => continue,
            ClipPcm::Ready(_) => buffers.push(process(mix, clip)?),
        }
    }
    let summed = sum(mix.rate, &buffers)?;
    let pcm = limit(&summed, mix.limiter)?;
    Ok(MixResult {
        contributed: buffers.len(),
        pcm,
    })
}

fn covers(range: &cutlass_model::TimeRange, time: Micros) -> bool {
    time.0 >= range.start.0 && time.0 < range.start.0.saturating_add(range.duration.0)
}

fn clip_for(draft: &Draft, track: &Track, segment: &Segment, time: Micros) -> MixClip {
    let offset = Micros(time.0.saturating_sub(segment.target.start.0));
    let volume = volume_from(segment, offset).unwrap_or(segment.volume);
    let (fade_in_us, fade_out_us, fade_curve) = fade_of(draft, segment);
    let speed = map::placement_of(draft, segment)
        .and_then(|placement| map::speed_at(&placement, time))
        .unwrap_or(1.0);
    MixClip {
        pcm: ClipPcm::Missing,
        speed,
        pitch_follows_speed: segment.is_tone_modify,
        volume,
        fade_in_us,
        fade_out_us,
        fade_curve,
        muted: track_is_muted(track),
        role: ClipRole::Sound,
    }
}

fn volume_from(segment: &Segment, offset: Micros) -> Result<f64, EngineError> {
    let mut keys = Vec::new();
    for group in &segment.common_keyframes {
        if group.property_type != "KFTypeVolume" && group.property_type != "volume" {
            continue;
        }
        for key in &group.keyframe_list {
            let Some(value) = key.values.first().copied() else {
                continue;
            };
            keys.push((key.time_offset, value));
        }
    }
    volume_at(segment.volume, &keys, offset)
}

fn fade_of(draft: &Draft, segment: &Segment) -> (i64, i64, i32) {
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Fade(fade)) = draft.materials.get(id.as_str()) {
            return (
                fade.fade_in_duration.0,
                fade.fade_out_duration.0,
                fade.fade_type,
            );
        }
    }
    (0, 0, 0)
}

fn track_is_muted(track: &Track) -> bool {
    track.attribute.contains(TrackAttribute::MUTE)
        || track.attribute.contains(TrackAttribute::MUTE_BY_SOLO)
}

fn process(mix: &Mix, clip: &MixClip) -> Result<Pcm, EngineError> {
    let ClipPcm::Ready(pcm) = &clip.pcm else {
        return Err(EngineError::NotReady);
    };
    let pcm = resample(pcm, mix.rate)?;
    let pcm = stretch(&pcm, clip.speed, !clip.pitch_follows_speed)?;
    let pcm = apply_gain(&pcm, clip.volume)?;
    let curve = FadeCurve::from_type(clip.fade_curve)?;
    Ok(fade(&pcm, clip.fade_in_us, clip.fade_out_us, curve)?)
}

fn apply_gain(pcm: &Pcm, volume: f64) -> Result<Pcm, EngineError> {
    if !volume.is_finite() {
        return Err(EngineError::Audio("non-finite volume".to_owned()));
    }
    if volume == 1.0 {
        return Ok(pcm.clone());
    }
    let Some(samples) = pcm.as_f32() else {
        return Err(EngineError::Layout);
    };
    let gain = volume as f32;
    let scaled = samples.iter().map(|sample| sample * gain).collect();
    Pcm::from_f32(pcm.rate, pcm.layout.clone(), scaled).map_err(|_| EngineError::Layout)
}

fn sum(rate: u32, buffers: &[Pcm]) -> Result<Pcm, EngineError> {
    if buffers.is_empty() {
        return Pcm::from_f32(rate, ChannelLayout::mono(), Vec::new())
            .map_err(|_| EngineError::Layout);
    }
    let layout = buffers[0].layout.clone();
    let channels = layout.len();
    let frames = buffers.iter().map(|pcm| pcm.samples).max().unwrap_or(0);
    let len = usize::try_from(frames)
        .ok()
        .and_then(|frames| frames.checked_mul(channels))
        .ok_or(EngineError::Overflow)?;
    let mut mixed = vec![0.0_f32; len];
    for pcm in buffers {
        if pcm.layout != layout || pcm.rate != rate {
            return Err(EngineError::Layout);
        }
        let Some(samples) = pcm.as_f32() else {
            return Err(EngineError::Layout);
        };
        for (index, sample) in samples.iter().enumerate() {
            if let Some(slot) = mixed.get_mut(index) {
                *slot += sample;
            }
        }
    }
    Pcm::from_f32(rate, layout, mixed).map_err(|_| EngineError::Layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_model::{
        Draft, MaterialAudioFade, MaterialId, Segment, TimeRange, Track, TrackAttribute, TrackType,
    };

    fn mono(rate: u32, samples: Vec<f32>) -> Pcm {
        Pcm::from_f32(rate, ChannelLayout::mono(), samples).expect("pcm")
    }

    fn sound(pcm: Pcm) -> MixClip {
        MixClip {
            pcm: ClipPcm::Ready(pcm),
            speed: 1.0,
            pitch_follows_speed: false,
            volume: 1.0,
            fade_in_us: 0,
            fade_out_us: 0,
            fade_curve: 0,
            muted: false,
            role: ClipRole::Sound,
        }
    }

    #[test]
    fn mute_adds_nothing_and_volume_zero_is_still_a_gain() {
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        let mut muted = sound(mono(8_000, vec![1.0; 4]));
        muted.muted = true;
        let silent = render(&mix, &[muted], ClockKind::Playback).expect("mute");
        assert_eq!(silent.contributed, 0);
        assert_eq!(silent.pcm.samples, 0);

        let mut quiet = sound(mono(8_000, vec![1.0; 4]));
        quiet.volume = 0.0;
        let gained = render(&mix, &[quiet], ClockKind::Playback).expect("gain");
        assert_eq!(gained.contributed, 1);
        assert_eq!(gained.pcm.as_f32(), Some(&[0.0, 0.0, 0.0, 0.0][..]));
    }

    #[test]
    fn clips_sum_after_resample_stretch_fade_and_the_limiter() {
        let mix = Mix {
            rate: 16_000,
            limiter: 1,
        };
        let mut left = sound(mono(8_000, vec![0.25; 4]));
        left.speed = 2.0;
        let right = sound(mono(16_000, vec![0.25; 4]));
        let summed = render(&mix, &[left, right], ClockKind::Playback).expect("sum");
        assert_eq!(summed.contributed, 2);
        let samples = summed.pcm.as_f32().expect("f32");
        assert!(samples.iter().any(|sample| *sample > 0.2 && *sample < 0.6));
        assert!(samples.iter().all(|sample| sample.abs() <= 1.0));

        let mut peaked = sound(mono(16_000, vec![2.0, 2.0]));
        peaked.fade_curve = 2;
        peaked.fade_in_us = 1_000_000;
        let faded = render(&mix, &[peaked], ClockKind::Playback).expect("fade");
        let samples = faded.pcm.as_f32().expect("f32");
        assert!(samples[0].abs() < 0.01);
        assert!(samples.iter().all(|sample| *sample <= 1.0));
    }

    #[test]
    fn text_does_not_enter_and_a_missing_clip_follows_the_clock() {
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        let mut text = sound(mono(8_000, vec![1.0]));
        text.role = ClipRole::Text;
        assert_eq!(
            render(&mix, &[text], ClockKind::Playback),
            Err(EngineError::NotMedia)
        );
        let missing = MixClip {
            pcm: ClipPcm::Missing,
            ..sound(mono(8_000, vec![1.0]))
        };
        let playback =
            render(&mix, std::slice::from_ref(&missing), ClockKind::Playback).expect("skip");
        assert_eq!(playback.contributed, 0);
        assert_eq!(
            render(&mix, &[missing], ClockKind::Export),
            Err(EngineError::NotReady)
        );
    }

    #[test]
    fn volume_keys_are_linear_and_a_muted_track_is_listed_without_samples() {
        let keys = [(Micros(0), 0.0), (Micros(1_000), 1.0)];
        let mid = volume_at(1.0, &keys, Micros(500)).expect("mid");
        assert!((mid - 0.5).abs() < 1e-9);

        let mut draft = Draft::empty("D");
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE);
        let mut segment = Segment::new("a", "audio", TimeRange::new(0, 1_000));
        segment.volume = 0.5;
        segment.is_tone_modify = true;
        segment.extra_material_refs.push(MaterialId::new("fade"));
        draft.tracks[0].segments.push(segment);
        draft.materials.audio_fades.push(MaterialAudioFade {
            id: MaterialId::new("fade"),
            fade_type: 2,
            fade_in_duration: Micros(10),
            fade_out_duration: Micros(20),
            ..MaterialAudioFade::default()
        });
        draft.tracks.push(Track::main("text"));
        draft.tracks[1].kind = TrackType::Text;
        draft.tracks[1]
            .segments
            .push(Segment::new("t", "words", TimeRange::new(0, 1_000)));
        let clips = mix_inputs(&draft, Micros(0));
        assert_eq!(clips.len(), 1);
        assert!(clips[0].muted);
        assert!(clips[0].pitch_follows_speed);
        assert_eq!(clips[0].fade_in_us, 10);
        assert_eq!(clips[0].fade_curve, 2);
        assert!(matches!(clips[0].pcm, ClipPcm::Missing));
    }
}
