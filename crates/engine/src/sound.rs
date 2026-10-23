//! The clips under the playhead, summed at their volume.
//!
//! The shell decodes each piece and plays the mix. This module does not open
//! a file, a decoder, or a device. A fade, a mute, and a scrub are not heard.

use cutlass_core::{AudioSamples, Timestamp};
use cutlass_project::{Channels, Project, Segment, SegmentId};

use crate::clock::Motion;

/// What the playhead should do while play is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lead {
    /// The cut has no sound the device can play. The wall clock moves it.
    Silent,
    /// The first samples are still opening. Hold the playhead.
    Waiting,
    /// The device clock, snapped to a frame.
    Running(i64),
    /// The clock reached the end of the cut.
    Finished(i64),
}

impl From<Motion> for Lead {
    fn from(motion: Motion) -> Self {
        match motion {
            Motion::Running(us) => Self::Running(us),
            Motion::Finished(us) => Self::Finished(us),
            Motion::Idle => Self::Silent,
        }
    }
}

/// Sequence time of a device clock that started at `origin_us`.
pub fn clock_us(origin_us: i64, time: Timestamp) -> i64 {
    let scale = i128::from(time.timescale.get());
    if scale <= 0 {
        return origin_us;
    }
    let us = i128::from(time.value).saturating_mul(1_000_000) / scale;
    origin_us.saturating_add(i64::try_from(us).unwrap_or(i64::MAX))
}

/// One audible clip on the timeline.
pub struct Cue {
    segment: SegmentId,
    source: String,
    start_us: i64,
    end_us: i64,
    source_start: i64,
    source_scale: u32,
}

impl Cue {
    /// The segment this clip came from.
    pub fn segment(&self) -> SegmentId {
        self.segment
    }

    /// The asset source the shell opens.
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// The source window for one fill, already placed on the timeline.
pub struct Piece {
    /// The asset source the shell opens.
    pub source: String,
    /// Timeline start of the clip, so one decode can cover later fills.
    pub clip_start_us: i64,
    /// Segment volume, from 0 to 1.
    pub volume: f64,
    /// Timeline time where these samples start.
    pub origin_us: i64,
    /// How long these samples last on the timeline.
    pub span_us: i64,
    /// First source tick inside the window.
    pub first_tick: i64,
    /// Source tick where the window ends.
    pub last_tick: i64,
    /// Timescale of `first_tick` and `last_tick`.
    pub tick_scale: u32,
}

/// Sample index of `ticks` at `scale`, for a stream of `rate` samples per second.
pub fn ticks_to_samples(ticks: i64, scale: u32, rate: u32) -> Option<u64> {
    if scale == 0 || rate == 0 || ticks <= 0 {
        return Some(0);
    }
    let samples = u128::from(ticks as u64).saturating_mul(u128::from(rate)) / u128::from(scale);
    u64::try_from(samples).ok()
}

/// Audible clips on the timeline. A picture-only segment is left out.
pub fn cues(project: &Project) -> Vec<Cue> {
    let mut found = Vec::new();
    for track in project.timeline().tracks() {
        for segment in track.segments() {
            if let Some(cue) = cue_of(project, segment) {
                found.push(cue);
            }
        }
    }
    found
}

/// The pieces of `cues` that overlap `[from_us, until_us)`.
pub fn pieces(project: &Project, cues: &[Cue], from_us: i64, until_us: i64) -> Vec<Piece> {
    let mut asks = Vec::new();
    for cue in cues {
        let Some(segment) = project.timeline().segment(cue.segment) else {
            continue;
        };
        if let Some(ask) = piece_of(segment, cue, from_us, until_us) {
            asks.push(ask);
        }
    }
    asks
}

fn cue_of(project: &Project, segment: &Segment) -> Option<Cue> {
    if segment.channels() == Channels::Picture || !segment.visible() {
        return None;
    }
    let asset = project.asset(segment.asset_id()?)?;
    let source = segment.source()?;
    let timeline = segment.timeline();
    Some(Cue {
        segment: segment.id(),
        source: asset.source().to_owned(),
        start_us: timeline.start().value,
        end_us: timeline.end().value,
        source_start: source.start().value,
        source_scale: source.start().timescale.get(),
    })
}

fn piece_of(segment: &Segment, cue: &Cue, from_us: i64, until_us: i64) -> Option<Piece> {
    if !segment.visible() || segment.channels() == Channels::Picture {
        return None;
    }
    let origin = from_us.max(cue.start_us);
    let end = until_us.min(cue.end_us);
    if end <= origin {
        return None;
    }
    let into = origin.checked_sub(cue.start_us)?;
    let span = end.checked_sub(cue.start_us)?;
    let first = cue
        .source_start
        .checked_add(segment.source_offset(into).ok()?)?;
    let last = cue
        .source_start
        .checked_add(segment.source_offset(span).ok()?)?;
    if last <= first {
        return None;
    }
    Some(Piece {
        source: cue.source.clone(),
        clip_start_us: cue.start_us,
        volume: segment.volume(),
        origin_us: origin,
        span_us: end - origin,
        first_tick: first,
        last_tick: last,
        tick_scale: cue.source_scale,
    })
}

/// One decoded clip, spread across its timeline span.
pub struct MixClip {
    origin_us: i64,
    span_us: i64,
    volume: f32,
    channels: u16,
    /// Channel-major planes.
    planar: Vec<f32>,
}

impl MixClip {
    /// `planar` is channel-major. A mono clip is copied onto every output channel.
    pub fn new(origin_us: i64, span_us: i64, volume: f32, channels: u16, planar: Vec<f32>) -> Self {
        Self {
            origin_us,
            span_us,
            volume,
            channels,
            planar,
        }
    }
}

/// Samples for `[from_us, until_us)` at `rate`, in `channels` planes.
///
/// Clips add together and each sample stops at full scale. A mono clip is
/// copied to every output channel.
pub fn mix(
    clips: &[MixClip],
    from_us: i64,
    until_us: i64,
    rate: u32,
    channels: u16,
) -> AudioSamples {
    let frames = frames_in(from_us, until_us, rate);
    let channels_n = usize::from(channels);
    let mut planar = vec![0f32; frames.saturating_mul(channels_n)];
    if frames == 0 || channels == 0 || rate == 0 {
        return empty_samples(rate, channels);
    }
    for clip in clips {
        if clip.span_us <= 0 || clip.volume == 0.0 || clip.channels == 0 {
            continue;
        }
        let clip_frames = clip.planar.len() / usize::from(clip.channels);
        if clip_frames == 0 {
            continue;
        }
        for frame in 0..frames {
            let time = from_us.saturating_add(frame_us(frame, rate));
            let rel = time.saturating_sub(clip.origin_us);
            if rel < 0 || rel >= clip.span_us {
                continue;
            }
            let pos = (rel as f64) / (clip.span_us as f64) * (clip_frames as f64);
            for channel in 0..channels_n {
                let sample = clip_sample(clip, channel, pos) * clip.volume;
                let slot = channel * frames + frame;
                planar[slot] = (planar[slot] + sample).clamp(-1.0, 1.0);
            }
        }
    }
    AudioSamples::new(rate, channels, planar)
        .unwrap_or_else(|_| zeroed(rate, channels, frames, channels_n))
}

fn empty_samples(rate: u32, channels: u16) -> AudioSamples {
    let rate = rate.max(1);
    let channels = channels.max(1);
    AudioSamples::new(rate, channels, Vec::new()).unwrap_or_else(|_| {
        AudioSamples::new(48_000, 1, Vec::new()).expect("an empty buffer is valid")
    })
}

fn zeroed(rate: u32, channels: u16, frames: usize, channels_n: usize) -> AudioSamples {
    AudioSamples::new(rate, channels, vec![0.0; frames * channels_n])
        .expect("a zeroed buffer matches the frame count")
}

fn frames_in(from_us: i64, until_us: i64, rate: u32) -> usize {
    let span = until_us.saturating_sub(from_us).max(0) as u128;
    let frames = span.saturating_mul(u128::from(rate)) / 1_000_000;
    usize::try_from(frames).unwrap_or(0)
}

fn frame_us(frame: usize, rate: u32) -> i64 {
    if rate == 0 {
        return 0;
    }
    let us = (frame as u128).saturating_mul(1_000_000) / u128::from(rate);
    i64::try_from(us).unwrap_or(i64::MAX)
}

fn clip_sample(clip: &MixClip, channel: usize, pos: f64) -> f32 {
    let source = usize::from(clip.channels);
    let frames = clip.planar.len() / source;
    let index = if source == 1 {
        0
    } else if channel < source {
        channel
    } else {
        0
    };
    let plane = &clip.planar[index * frames..(index + 1) * frames];
    sample_at(plane, pos)
}

fn sample_at(plane: &[f32], pos: f64) -> f32 {
    if plane.is_empty() || !pos.is_finite() {
        return 0.0;
    }
    if pos <= 0.0 {
        return plane[0];
    }
    let last = plane.len() - 1;
    if pos >= last as f64 {
        return plane[last];
    }
    let index = pos.floor() as usize;
    let frac = (pos - index as f64) as f32;
    plane[index] * (1.0 - frac) + plane[index + 1] * frac
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::Timestamp;
    use cutlass_project::{
        Asset, Channels, Command, History, Outcome, Project, ProjectId, Segment, Speed, TimeRange,
    };

    use super::{MixClip, cue_of, cues, frames_in, mix, piece_of};

    fn project() -> Project {
        Project::new(ProjectId::from_bytes([2; 16]), "Untitled")
    }

    fn scale(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("scale")
    }

    fn place(channels: Channels, speed: Speed) -> (Project, cutlass_project::SegmentId) {
        let mut history = History::new();
        let mut cut = project();
        let audio = Asset::audio("file:///cuts/a.m4a", Timestamp::new(96_000, scale(48_000)))
            .expect("audio");
        let id = match history
            .apply(&mut cut, &[Command::AddAsset(audio)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let segment = {
            let stored = cut.asset(id).expect("asset");
            let source = TimeRange::new(
                Timestamp::new(0, scale(48_000)),
                Timestamp::new(96_000, scale(48_000)),
            )
            .expect("source");
            Segment::media(stored, source, 0, channels, speed).expect("segment")
        };
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: None,
                    segment,
                }],
            )
            .expect("place");
        let segment = cut
            .timeline()
            .tracks()
            .iter()
            .flat_map(|track| track.segments())
            .next()
            .expect("segment")
            .id();
        (cut, segment)
    }

    #[test]
    fn a_picture_is_not_a_voice_and_a_faster_clip_covers_more_source() {
        let (cut, sound) = place(Channels::Sound, Speed::new(2, 1).expect("speed"));
        let video = Asset::video(
            "file:///cuts/v.mp4",
            Timestamp::new(30_000, scale(30_000)),
            320,
            180,
            false,
        )
        .expect("video");
        let mut history = History::new();
        let mut cut = cut;
        let video_id = match history
            .apply(&mut cut, &[Command::AddAsset(video)])
            .expect("add")
            .as_slice()
        {
            [Outcome::Asset(id)] => *id,
            other => panic!("expected an asset, got {other:?}"),
        };
        let picture = {
            let stored = cut.asset(video_id).expect("asset");
            let source = TimeRange::new(
                Timestamp::new(0, scale(30_000)),
                Timestamp::new(30_000, scale(30_000)),
            )
            .expect("source");
            Segment::media(stored, source, 0, Channels::Picture, Speed::ONE).expect("segment")
        };
        let main = cut.timeline().main_track().id();
        history
            .apply(
                &mut cut,
                &[Command::Place {
                    track: Some(main),
                    segment: picture,
                }],
            )
            .expect("place");
        let found = cues(&cut);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].segment(), sound);
        let segment = cut.timeline().segment(sound).expect("sound");
        let ask = piece_of(segment, &found[0], 0, 250_000).expect("window");
        assert_eq!(ask.first_tick, 0);
        assert_eq!(ask.last_tick, 24_000);
        assert!(
            cue_of(
                &cut,
                cut.timeline()
                    .tracks()
                    .iter()
                    .flat_map(|track| track.segments())
                    .find(|segment| segment.channels() == Channels::Picture)
                    .expect("picture")
            )
            .is_none()
        );
    }

    #[test]
    fn volume_scales_and_a_later_clip_stays_silent_until_it_starts() {
        let tone = MixClip::new(0, 1_000_000, 0.5, 1, vec![1.0; 8]);
        let mixed = mix(&[tone], 0, 1_000, 1_000, 2);
        assert_eq!(mixed.frames(), 1);
        assert!((mixed.plane(0).expect("left")[0] - 0.5).abs() < 1e-4);
        assert!((mixed.plane(1).expect("right")[0] - 0.5).abs() < 1e-4);

        let later = MixClip::new(1_000, 1_000, 1.0, 1, vec![1.0]);
        let mixed = mix(&[later], 0, 2_000, 1_000, 1);
        assert_eq!(frames_in(0, 2_000, 1_000), 2);
        assert_eq!(mixed.plane(0).expect("mono")[0], 0.0);
        assert!((mixed.plane(0).expect("mono")[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn overlapping_clips_sum_and_stop_at_full_scale() {
        let clip = |volume: f32| MixClip::new(0, 1_000, volume, 1, vec![1.0]);
        let mixed = mix(&[clip(0.25), clip(0.25)], 0, 1_000, 1_000, 1);
        assert!((mixed.plane(0).expect("sum")[0] - 0.5).abs() < 1e-4);
        let mixed = mix(&[clip(1.0), clip(1.0)], 0, 1_000, 1_000, 1);
        assert_eq!(mixed.plane(0).expect("clamped")[0], 1.0);
    }
}
