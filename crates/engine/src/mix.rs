//! One mix of decoded PCM.
//!
//! The mix is mono or stereo. A clip with more channels, or two channels
//! that are not left and right, is first folded into stereo by
//! `cutlass-audio`'s downmix; a mono or stereo clip is passed on as it is.
//! Each audible clip is then resampled, stretched, gained, and faded through
//! `cutlass-audio`. The mix is asked for one window of the timeline at a
//! time, so a clip's buffer is a window of that clip: its fades are placed by
//! where the window sits in the clip, and a window clear of both fades is
//! not touched by them. Each sample of a window is the clips that cover that
//! sample's own time: a clip that starts inside the window is silent up to
//! its first sample, and one that ends inside it is silent from its end on.
//! A muted track contributes no samples: mute is not a gain of zero. Text
//! and shapes are refused before any of that runs. The limiter runs on the
//! sum.

use cutlass_audio::{FadeCurve, FadeSpan, downmix, fade_window, limit, resample, stretch};
use cutlass_codec::access::{ChannelLayout, Pcm};
use cutlass_model::{Draft, KeyframeGraph, MaterialRef, Micros, Segment, Track, TrackAttribute};

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

/// The stretch of the timeline one mix covers: `frames` samples at `rate`,
/// the first of them at `time`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixSpan {
    pub time: Micros,
    pub frames: u32,
    pub rate: u32,
}

/// One clip the mix can add.
///
/// `pcm` is the clip's part of the window being mixed: from the clip's
/// first sample in the window, for `frames` of the mix. `lead_frames` and
/// `tail_frames` are the window's samples before and after that part, where
/// the clip is not on the timeline. `offset_us` and `duration_us` say where
/// the part sits in the clip, which is what places the fades at the clip's
/// two edges.
#[derive(Clone, Debug, PartialEq)]
pub struct MixClip {
    pub pcm: ClipPcm,
    /// Window samples before the clip's first. Zero unless the clip starts
    /// inside the window.
    pub lead_frames: u32,
    /// Window samples the clip covers. A longer buffer is cut there.
    pub frames: u32,
    /// Window samples after the clip's last. Zero unless the clip ends
    /// inside the window.
    pub tail_frames: u32,
    /// From the clip's start on the timeline to its first sample in the
    /// window.
    pub offset_us: i64,
    /// The clip's length on the timeline.
    pub duration_us: i64,
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

/// Segments that may enter the mix of `span`: those that cover the time of
/// at least one of its samples. Text and shapes are left out.
#[must_use]
pub fn mix_inputs(draft: &Draft, span: MixSpan) -> Vec<MixClip> {
    let mut clips = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if !map::contributes_sound(draft, segment) {
                continue;
            }
            let Some(part) = clip_part(&segment.target, span) else {
                continue;
            };
            clips.push(clip_for(draft, track, segment, part));
        }
    }
    clips
}

/// The samples of one mix window that a clip covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipPart {
    /// Window samples before the clip's first.
    pub lead_frames: u32,
    /// Window samples the clip covers. Never zero.
    pub frames: u32,
    /// Window samples after the clip's last.
    pub tail_frames: u32,
    /// From the clip's start to the time of its first sample in the window.
    pub offset_us: i64,
}

/// The part of `span` a clip placed at `target` sounds in, or `None` when
/// it covers none of the samples.
///
/// Sample `i` of the window is at `span.time` plus `i` sample periods, and
/// it is the clip's when that time is inside `target`. So two clips that
/// meet at a cut share no sample and leave none between them, and a clip
/// that covers `span.time` always has the window's first sample.
#[must_use]
pub fn clip_part(target: &cutlass_model::TimeRange, span: MixSpan) -> Option<ClipPart> {
    if span.rate == 0 {
        return None;
    }
    let rate = i128::from(span.rate);
    let start = i128::from(target.start.0) - i128::from(span.time.0);
    let end = start + i128::from(target.duration.0);
    let frames = i128::from(span.frames);
    let first = frames_up_to(start, rate);
    let past = frames_up_to(end, rate).min(frames);
    if first >= past {
        return None;
    }
    // The time of sample `first`, rounded down to a microsecond: at or
    // after the clip's start, and the window's own start when `first` is 0.
    let offset = first * MICROS / rate - start;
    Some(ClipPart {
        lead_frames: u32::try_from(first).ok()?,
        frames: u32::try_from(past - first).ok()?,
        tail_frames: u32::try_from(frames - past).ok()?,
        offset_us: i64::try_from(offset).ok()?,
    })
}

const MICROS: i128 = 1_000_000;

/// The index of the first sample at or after `micros`, counting the
/// samples at `rate` from the one at zero. Before zero that is the first.
fn frames_up_to(micros: i128, rate: i128) -> i128 {
    if micros <= 0 {
        return 0;
    }
    (micros * rate + (MICROS - 1)) / MICROS
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
            ClipPcm::Ready(_) => buffers.push(place(process(mix, clip)?, clip)?),
        }
    }
    let summed = sum(mix.rate, &buffers)?;
    let pcm = limit(&summed, mix.limiter)?;
    Ok(MixResult {
        contributed: buffers.len(),
        pcm,
    })
}

fn clip_for(draft: &Draft, track: &Track, segment: &Segment, part: ClipPart) -> MixClip {
    let offset = Micros(part.offset_us);
    // The clip's first sample in the window, on the timeline.
    let time = Micros(segment.target.start.0.saturating_add(part.offset_us));
    let volume = clip_volume(segment, &draft.keyframe_graph_list, offset);
    let (fade_in_us, fade_out_us, fade_curve) = fade_of(draft, segment);
    let speed = map::placement_of(draft, segment)
        .and_then(|placement| map::speed_at(&placement, time))
        .unwrap_or(1.0);
    MixClip {
        pcm: ClipPcm::Missing,
        lead_frames: part.lead_frames,
        frames: part.frames,
        tail_frames: part.tail_frames,
        offset_us: offset.0,
        duration_us: segment.target.duration.0,
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

/// The clip's gain `offset` into it, its volume keyframes applied with
/// their curves and `graphs`, as every other keyed setting is played. 1 is
/// the file as recorded. Keys that do not read leave the stored volume.
#[must_use]
pub fn clip_volume(segment: &Segment, graphs: &[KeyframeGraph], offset: Micros) -> f64 {
    let keyed = cutlass_model::sample(
        &segment.common_keyframes,
        graphs,
        &[cutlass_model::VOLUME_KEY, "volume"],
        offset.0,
    );
    match keyed {
        Some(Some(value)) if value.is_finite() => value,
        _ => segment.volume,
    }
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
    // Before the resample, so it runs on two channels instead of eight.
    let folded;
    let pcm = if pcm.layout.len() == 1 || pcm.layout == ChannelLayout::stereo() {
        pcm
    } else {
        folded = downmix(pcm, &ChannelLayout::stereo())?;
        &folded
    };
    let pcm = resample(pcm, mix.rate)?;
    let pcm = stretch(&pcm, clip.speed, !clip.pitch_follows_speed)?;
    let pcm = apply_gain(&pcm, clip.volume)?;
    let curve = FadeCurve::from_type(clip.fade_curve)?;
    // After the stretch, so the buffer is on the timeline's clock, as the
    // fade durations and the clip's length are.
    let span = FadeSpan {
        start_us: clip.offset_us,
        clip_us: clip.duration_us,
    };
    Ok(fade_window(
        &pcm,
        span,
        clip.fade_in_us,
        clip.fade_out_us,
        curve,
    )?)
}

/// `pcm` where its clip sounds in the window: after the clip's lead of
/// silence, cut at the frames the clip covers, and, when the clip ends
/// inside the window, followed by silence to the window's end. A clip that
/// covers the whole window and is no longer than it is passed on as it is.
fn place(pcm: Pcm, clip: &MixClip) -> Result<Pcm, EngineError> {
    let held = pcm.samples.min(clip.frames);
    if clip.lead_frames == 0 && clip.tail_frames == 0 && held == pcm.samples {
        return Ok(pcm);
    }
    let Some(samples) = pcm.as_f32() else {
        return Err(EngineError::Layout);
    };
    let channels = pcm.layout.len();
    let len = |frames: u32| {
        usize::try_from(frames)
            .ok()
            .and_then(|frames| frames.checked_mul(channels))
            .ok_or(EngineError::Overflow)
    };
    let lead = len(clip.lead_frames)?;
    let sound = samples.get(..len(held)?).ok_or(EngineError::Layout)?;
    let mut placed = vec![0.0_f32; lead];
    placed.extend_from_slice(sound);
    if clip.tail_frames > 0 {
        let window = clip
            .lead_frames
            .checked_add(clip.frames)
            .and_then(|frames| frames.checked_add(clip.tail_frames))
            .ok_or(EngineError::Overflow)?;
        placed.resize(len(window)?, 0.0);
    }
    Pcm::from_f32(pcm.rate, pcm.layout.clone(), placed).map_err(|_| EngineError::Layout)
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

/// Every buffer added together in the widest layout among them. A mono
/// buffer is heard on both sides of a stereo mix, as a clip with no sound
/// of its own (silence) sits beside music. Other differing layouts are
/// refused.
fn sum(rate: u32, buffers: &[Pcm]) -> Result<Pcm, EngineError> {
    if buffers.is_empty() {
        return Pcm::from_f32(rate, ChannelLayout::mono(), Vec::new())
            .map_err(|_| EngineError::Layout);
    }
    let Some(layout) = buffers
        .iter()
        .map(|pcm| pcm.layout.clone())
        .max_by_key(ChannelLayout::len)
    else {
        return Err(EngineError::Layout);
    };
    let channels = layout.len();
    let frames = buffers.iter().map(|pcm| pcm.samples).max().unwrap_or(0);
    let len = usize::try_from(frames)
        .ok()
        .and_then(|frames| frames.checked_mul(channels))
        .ok_or(EngineError::Overflow)?;
    let mut mixed = vec![0.0_f32; len];
    for pcm in buffers {
        if pcm.rate != rate {
            return Err(EngineError::Layout);
        }
        let Some(samples) = pcm.as_f32() else {
            return Err(EngineError::Layout);
        };
        let spread = if pcm.layout == layout {
            1
        } else if pcm.layout.len() == 1 {
            channels
        } else {
            return Err(EngineError::Layout);
        };
        for (index, sample) in samples.iter().enumerate() {
            for side in 0..spread {
                if let Some(slot) = mixed.get_mut(index * spread + side) {
                    *slot += sample;
                }
            }
        }
    }
    Pcm::from_f32(rate, layout, mixed).map_err(|_| EngineError::Layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_model::{
        Draft, LinkPair, MaterialAudio, MaterialAudioFade, MaterialId, MaterialVideo, Relationship,
        Segment, SegmentId, TimeRange, Track, TrackAttribute, TrackType,
    };

    fn mono(rate: u32, samples: Vec<f32>) -> Pcm {
        Pcm::from_f32(rate, ChannelLayout::mono(), samples).expect("pcm")
    }

    #[test]
    fn a_mono_clip_is_heard_on_both_sides_of_a_stereo_mix() {
        let stereo = Pcm::from_f32(4, ChannelLayout::stereo(), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let summed = sum(4, &[mono(4, vec![0.5, 0.0]), stereo.clone()]).unwrap();
        assert_eq!(summed.layout, ChannelLayout::stereo());
        assert_eq!(summed.as_f32().unwrap(), [0.6, 0.7, 0.3, 0.4]);
        // The same either way round.
        let other = sum(4, &[stereo, mono(4, vec![0.5, 0.0])]).unwrap();
        assert_eq!(other, summed);
        assert_eq!(sum(8, &[mono(4, vec![0.5])]), Err(EngineError::Layout));
    }

    fn surround(rate: u32, frames: Vec<[f32; 6]>) -> Pcm {
        let layout = ChannelLayout::standard(cutlass_codec::access::StandardLayout::FiveOne);
        Pcm::from_f32(rate, layout, frames.concat()).expect("5.1")
    }

    #[test]
    fn a_surround_clip_is_folded_into_stereo_beside_mono_and_stereo() {
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        // FL, FR, FC, LFE, BL, BR.
        let five_one = surround(
            8_000,
            vec![
                [0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.5],
            ],
        );
        let alone = render(&mix, &[sound(five_one.clone())], ClockKind::Playback).expect("alone");
        assert_eq!(alone.pcm.layout, ChannelLayout::stereo());
        let c = std::f32::consts::FRAC_1_SQRT_2;
        let got = alone.pcm.as_f32().expect("f32");
        let want = [0.5 * c, 0.5 * c, 0.25, 0.5 * c];
        for (got, want) in got.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{got} is not {want}");
        }

        let stereo =
            Pcm::from_f32(8_000, ChannelLayout::stereo(), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let mono = mono(8_000, vec![0.5, 0.0]);
        let all = render(
            &mix,
            &[sound(five_one), sound(stereo), sound(mono)],
            ClockKind::Playback,
        )
        .expect("three layouts mix");
        assert_eq!(all.contributed, 3);
        assert_eq!(all.pcm.layout, ChannelLayout::stereo());
        let got = all.pcm.as_f32().expect("f32");
        let want = [0.5 * c + 0.6, 0.5 * c + 0.7, 0.55, 0.5 * c + 0.4];
        for (got, want) in got.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{got} is not {want}");
        }
    }

    #[test]
    fn mono_and_stereo_clips_reach_the_mix_bit_for_bit() {
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        let mono_pcm = mono(8_000, vec![0.1, -0.7, f32::MIN_POSITIVE]);
        let heard = render(&mix, &[sound(mono_pcm.clone())], ClockKind::Playback).expect("mono");
        assert_eq!(heard.pcm, mono_pcm);
        let stereo =
            Pcm::from_f32(8_000, ChannelLayout::stereo(), vec![0.3, -0.0, 1.5, -1.25]).unwrap();
        let heard = render(&mix, &[sound(stereo.clone())], ClockKind::Playback).expect("stereo");
        assert_eq!(heard.pcm, stereo);
    }

    /// A clip that is all of `pcm`: the window starts the clip and ends it,
    /// and nothing is cut from the buffer.
    fn sound(pcm: Pcm) -> MixClip {
        MixClip {
            lead_frames: 0,
            frames: u32::MAX,
            tail_frames: 0,
            offset_us: 0,
            duration_us: i64::from(pcm.samples) * 1_000_000 / i64::from(pcm.rate),
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

    /// One sample of a 48 kHz mix at `time`.
    fn at(time: i64) -> MixSpan {
        MixSpan {
            time: Micros(time),
            frames: 1,
            rate: 48_000,
        }
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
        let clips = mix_inputs(&draft, at(0));
        assert_eq!(clips.len(), 1);
        assert!(clips[0].muted);
        assert!(clips[0].pitch_follows_speed);
        assert_eq!(clips[0].fade_in_us, 10);
        assert_eq!(clips[0].fade_curve, 2);
        assert!(matches!(clips[0].pcm, ClipPcm::Missing));
    }

    #[test]
    fn a_fade_is_placed_by_where_the_window_sits_in_the_clip() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("a", "audio", TimeRange::new(2_000_000, 1_000_000));
        segment.extra_material_refs.push(MaterialId::new("fade"));
        draft.tracks[0].segments.push(segment);
        draft.materials.audio_fades.push(MaterialAudioFade {
            id: MaterialId::new("fade"),
            fade_type: 2,
            fade_in_duration: Micros(20_000),
            fade_out_duration: Micros(60_000),
            ..MaterialAudioFade::default()
        });
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        // 10 ms of full-scale tone, shorter than either fade.
        let window = mono(8_000, vec![1.0; 80]);
        let heard = |time: i64| {
            let span = MixSpan {
                time: Micros(time),
                frames: 80,
                rate: 8_000,
            };
            let mut clips = mix_inputs(&draft, span);
            assert_eq!(clips.len(), 1);
            assert_eq!(clips[0].offset_us, time - 2_000_000);
            assert_eq!(clips[0].duration_us, 1_000_000);
            clips[0].pcm = ClipPcm::Ready(window.clone());
            render(&mix, &clips, ClockKind::Export).expect("mix").pcm
        };

        // The clip's first window is the first half of the fade in.
        let first = heard(2_000_000);
        let first = first.as_f32().expect("f32");
        assert_eq!(first[0], 0.0);
        assert_eq!(first[40], 0.25);
        assert_eq!(first[79], 79.0 / 160.0);
        // Its middle is the tone, bit for bit, in every window.
        for time in [2_020_000, 2_500_000, 2_930_000] {
            assert_eq!(heard(time), window, "at {time}");
        }
        // Its last window is the end of the fade out.
        let last = heard(2_990_000);
        let last = last.as_f32().expect("f32");
        assert_eq!(last[0], 79.0 / 480.0);
        assert_eq!(last[79], 0.0);
        assert!(last.windows(2).all(|pair| pair[1] < pair[0]));
    }

    fn part_of(start: i64, duration: i64, time: i64, frames: u32) -> Option<ClipPart> {
        let span = MixSpan {
            time: Micros(time),
            frames,
            rate: 48_000,
        };
        clip_part(&TimeRange::new(start, duration), span)
    }

    #[test]
    fn a_clip_has_the_window_samples_whose_times_it_covers() {
        let part = |lead_frames, frames, tail_frames, offset_us| {
            Some(ClipPart {
                lead_frames,
                frames,
                tail_frames,
                offset_us,
            })
        };
        // 480 frames at 48 kHz are 10 ms. A clip over the whole window has
        // all of it, and the window sits where its start is in the clip.
        assert_eq!(part_of(0, 1_000_000, 0, 480), part(0, 480, 0, 0));
        assert_eq!(
            part_of(0, 1_000_000, 250_000, 480),
            part(0, 480, 0, 250_000)
        );
        // One that starts 2.5 ms in is silent for 120 samples, then plays
        // from its own first sample.
        assert_eq!(part_of(2_500, 1_000_000, 0, 480), part(120, 360, 0, 0));
        // One that ends 2.5 ms in has 120 samples, then 360 of silence.
        assert_eq!(
            part_of(0, 1_002_500, 1_000_000, 480),
            part(0, 120, 360, 1_000_000)
        );
        // One inside the window has both.
        assert_eq!(part_of(2_500, 5_000, 0, 480), part(120, 240, 120, 0));
        // A start between two samples takes the next one: sample 121 is
        // 2520.83 us in, 10 us after the clip starts.
        assert_eq!(part_of(2_510, 1_000_000, 0, 480), part(121, 359, 0, 10));

        // A window's last sample is before its end, and its first is at its
        // start: a clip is in neither window it only touches.
        assert_eq!(part_of(10_000, 1_000_000, 0, 480), None);
        assert_eq!(part_of(0, 10_000, 10_000, 480), None);
        assert_eq!(part_of(0, 10_000, 9_990, 480), part(0, 1, 479, 9_990));
        // A clip between two sample times has none of them.
        assert_eq!(part_of(2_505, 10, 0, 480), None);
        // No samples, no rate, or no length: nothing sounds.
        assert_eq!(part_of(0, 1_000_000, 0, 0), None);
        assert_eq!(part_of(0, 0, 0, 480), None);
        assert_eq!(part_of(0, -5, 0, 480), None);
        let no_rate = MixSpan {
            time: Micros(0),
            frames: 480,
            rate: 0,
        };
        assert_eq!(clip_part(&TimeRange::new(0, 1_000_000), no_rate), None);
        // Times at the far ends of the clock do not wrap.
        assert_eq!(part_of(i64::MIN, i64::MAX, i64::MAX, 480), None);
        assert_eq!(
            part_of(i64::MAX - 5_000, i64::MAX, i64::MAX - 10_000, 480),
            part(240, 240, 0, 0)
        );
    }

    #[test]
    fn clips_that_meet_at_a_cut_share_no_sample_and_leave_none_out() {
        // Every cut from 0 to 10 ms, on a sample or between two.
        for cut in (0..=10_000).step_by(7) {
            let first = part_of(-1_000_000, 1_000_000 + cut, 0, 480);
            let second = part_of(cut, 1_000_000, 0, 480);
            let before = first.map_or(0, |part| part.frames);
            let after = second.map_or(0, |part| part.frames);
            assert_eq!(before + after, 480, "cut at {cut}");
            if let Some(second) = second {
                assert_eq!(second.lead_frames, before, "cut at {cut}");
                assert!((0..21).contains(&second.offset_us), "cut at {cut}");
            }
            if let Some(first) = first {
                assert_eq!(first.lead_frames, 0, "cut at {cut}");
                assert_eq!(first.tail_frames, after, "cut at {cut}");
            }
        }
    }

    #[test]
    fn a_clip_is_placed_after_its_lead_cut_at_its_end_and_followed_by_silence() {
        let mix = Mix {
            rate: 8_000,
            limiter: 0,
        };
        let heard = |clips: &[MixClip]| {
            let mixed = render(&mix, clips, ClockKind::Export).expect("mix");
            mixed.pcm.as_f32().expect("f32").to_vec()
        };
        // A clip that starts two samples into the window.
        let mut late = sound(mono(8_000, vec![0.5; 4]));
        late.lead_frames = 2;
        late.frames = 4;
        assert_eq!(heard(&[late.clone()]), [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]);
        // One that ends three samples into a window of six: the buffer goes
        // on, and the mix stops at the clip's end and fills the window.
        let mut early = sound(mono(8_000, vec![0.25; 6]));
        early.frames = 3;
        early.tail_frames = 3;
        assert_eq!(heard(&[early.clone()]), [0.25, 0.25, 0.25, 0.0, 0.0, 0.0]);
        // A buffer that stops short of the clip's end is still followed by
        // silence to the window's end.
        let mut short = sound(mono(8_000, vec![0.25; 2]));
        short.frames = 3;
        short.tail_frames = 3;
        assert_eq!(heard(&[short]), [0.25, 0.25, 0.0, 0.0, 0.0, 0.0]);
        // The two at a cut, beside a clip that covers the window.
        let under = sound(mono(8_000, vec![0.125; 6]));
        late.lead_frames = 3;
        late.frames = 3;
        assert_eq!(
            heard(&[early, late.clone(), under]),
            [0.375, 0.375, 0.375, 0.625, 0.625, 0.625]
        );
        // A stereo clip keeps its sides after the lead.
        let stereo = Pcm::from_f32(8_000, ChannelLayout::stereo(), vec![0.1, 0.2, 0.3, 0.4]);
        let mut stereo = sound(stereo.expect("stereo"));
        stereo.lead_frames = 1;
        stereo.frames = 1;
        stereo.tail_frames = 1;
        assert_eq!(heard(&[stereo]), [0.0, 0.0, 0.1, 0.2, 0.0, 0.0]);
        // A missing clip still adds nothing in playback, lead or not.
        late.pcm = ClipPcm::Missing;
        let skipped = render(&mix, &[late], ClockKind::Playback).expect("skip");
        assert_eq!(skipped.pcm.samples, 0);
    }

    #[test]
    fn a_clip_that_starts_inside_the_window_is_listed_from_its_own_start() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("a", "audio", TimeRange::new(2_500, 5_000));
        cutlass_model::put_key(&mut segment, cutlass_model::VOLUME_KEY, Micros(0), 0.25);
        cutlass_model::put_key(&mut segment, cutlass_model::VOLUME_KEY, Micros(5_000), 0.75);
        draft.tracks[0].segments.push(segment);
        let span = |time, frames| MixSpan {
            time: Micros(time),
            frames,
            rate: 48_000,
        };
        // The window starts before the clip: the clip is listed from its
        // first sample, at the volume its first key gives.
        let clips = mix_inputs(&draft, span(0, 480));
        assert_eq!(clips.len(), 1);
        assert_eq!(
            (clips[0].lead_frames, clips[0].frames, clips[0].tail_frames),
            (120, 240, 120)
        );
        assert_eq!(clips[0].offset_us, 0);
        assert_eq!(clips[0].duration_us, 5_000);
        assert_eq!(clips[0].volume, 0.25);
        // A window that starts inside the clip is as far into it.
        let clips = mix_inputs(&draft, span(5_000, 480));
        assert_eq!(
            (clips[0].lead_frames, clips[0].frames, clips[0].tail_frames),
            (0, 120, 360)
        );
        assert_eq!(clips[0].offset_us, 2_500);
        assert_eq!(clips[0].volume, 0.5);
        // A window that stops before the clip, and one after it, list nothing.
        assert!(mix_inputs(&draft, span(0, 120)).is_empty());
        assert!(mix_inputs(&draft, span(7_500, 480)).is_empty());
    }

    #[test]
    fn a_volume_key_eases_along_its_graph_as_the_inspector_shows_it() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("a", "audio", TimeRange::new(0, 1_000));
        cutlass_model::put_key(&mut segment, cutlass_model::VOLUME_KEY, Micros(0), 0.0);
        cutlass_model::put_key(&mut segment, cutlass_model::VOLUME_KEY, Micros(1_000), 1.0);
        assert!((clip_volume(&segment, &[], Micros(500)) - 0.5).abs() < 1e-9);
        let ease_in = cutlass_model::GRAPH_PRESETS
            .iter()
            .find(|preset| preset.name == "Ease In")
            .expect("ease in");
        segment.common_keyframes[0].keyframe_list[1].graph_id = "ease".to_owned();
        draft
            .keyframe_graph_list
            .push(KeyframeGraph::of("ease", ease_in));
        let eased = clip_volume(&segment, &draft.keyframe_graph_list, Micros(500));
        let shown = cutlass_model::sample(
            &segment.common_keyframes,
            &draft.keyframe_graph_list,
            &[cutlass_model::VOLUME_KEY],
            500,
        )
        .flatten()
        .expect("shown");
        assert!(eased < 0.4, "{eased}");
        assert_eq!(eased, shown);
        draft.tracks[0].segments.push(segment);
        let clips = mix_inputs(&draft, at(500));
        assert_eq!(clips[0].volume, eased);
        // A key that does not read leaves the stored volume.
        let mut broken = Segment::new("b", "audio", TimeRange::new(0, 1_000));
        broken.volume = 0.25;
        cutlass_model::put_key(&mut broken, cutlass_model::VOLUME_KEY, Micros(0), f64::NAN);
        assert_eq!(clip_volume(&broken, &[], Micros(0)), 0.25);
    }

    #[test]
    fn a_separated_picture_sound_is_not_mixed_twice() {
        let mut draft = Draft::empty("D");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("sound"),
            kind: "video_original_sound".to_owned(),
            path: "clip.mp4".to_owned(),
            video_id: MaterialId::new("picture"),
            ..MaterialAudio::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "other",
            "picture",
            TimeRange::new(2_000_000, 1_000_000),
        ));
        let mut lane = Track::main("sound");
        lane.kind = TrackType::Audio;
        lane.segments
            .push(Segment::new("sound", "sound", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(lane);
        let both = mix_inputs(&draft, at(0));
        assert_eq!(both.len(), 2);

        draft.relationships.push(Relationship {
            kind: "linkage".to_owned(),
            id_to_id: vec![LinkPair {
                owner: SegmentId::new("picture"),
                dependent: SegmentId::new("sound"),
            }],
            ..Relationship::default()
        });
        assert_eq!(mix_inputs(&draft, at(0)).len(), 1);
        assert_eq!(mix_inputs(&draft, at(2_000_000)).len(), 1);
    }
}
