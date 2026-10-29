//! The mixed PCM at one timeline time.
//!
//! Clips come from the engine mix. This module does not resample, fade, or
//! limit on its own. A muted track contributes nothing because the mix skips
//! it. Export passes [`ClockKind::Export`], so a missing clip is an error.
//! Playback can pass [`ClockKind::Playback`] and skip that clip.

use cutlass_engine::{ClipPcm, ClockKind, Mix, MixResult, is_graphic, mix_inputs, render};
use cutlass_model::{Draft, Micros, Segment, SegmentId};

use crate::Snapshot;

/// Why the mix did not return samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioError {
    /// Export is still waiting on samples for an audible clip.
    NotReady,
    /// The clip is text, a shape, or a solid color.
    NotMedia,
    /// The mix rate is zero.
    NoClock,
    /// The clip list and the draft did not match, or the mix refused the buffer.
    Mix,
    /// The sample source failed before the mix ran.
    Source,
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::NotReady => "export is waiting for audio samples",
            Self::NotMedia => "text and shapes are not audio",
            Self::NoClock => "the mix has no sample rate",
            Self::Mix => "the mix refused the samples",
            Self::Source => "the audio source failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for AudioError {}

/// Sample rate, limiter kind, and how many PCM frames the source should return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixWindow {
    pub rate: u32,
    pub limiter: i32,
    pub frames: u32,
}

/// Mixed PCM for clips that cover `time`.
///
/// `fill` is asked once per segment the mix will hear. A muted track is
/// skipped first, so its source is not read. The mix then applies volume and
/// fades. The draft is not written.
pub fn audio_slice<F>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    window: MixWindow,
    clock: ClockKind,
    mut fill: F,
) -> Result<MixResult, AudioError>
where
    F: FnMut(&SegmentId, u32, u32) -> Result<ClipPcm, AudioError>,
{
    if window.rate == 0 {
        return Err(AudioError::NoClock);
    }
    let draft = snapshot.draft();
    let mut clips = mix_inputs(draft, time);
    let segments = audible(draft, time);
    if clips.len() != segments.len() {
        return Err(AudioError::Mix);
    }
    for (clip, segment) in clips.iter_mut().zip(segments) {
        if clip.muted {
            continue;
        }
        clip.pcm = fill(&segment.id, window.frames, window.rate)?;
    }
    let mix = Mix {
        rate: window.rate,
        limiter: window.limiter,
    };
    render(&mix, &clips, clock).map_err(|error| match error {
        cutlass_engine::EngineError::NotReady => AudioError::NotReady,
        cutlass_engine::EngineError::NotMedia => AudioError::NotMedia,
        cutlass_engine::EngineError::NoClock => AudioError::NoClock,
        _ => AudioError::Mix,
    })
}

fn audible(draft: &Draft, time: Micros) -> Vec<&Segment> {
    let mut segments = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if is_graphic(draft, segment) {
                continue;
            }
            let end = segment
                .target
                .start
                .0
                .saturating_add(segment.target.duration.0);
            if time.0 >= segment.target.start.0 && time.0 < end {
                segments.push(segment);
            }
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use cutlass_codec::access::{ChannelLayout, Pcm};
    use cutlass_engine::{ClipPcm, ClockKind};
    use cutlass_model::{
        Draft, Micros, Segment, SegmentId, TimeRange, Track, TrackAttribute, TrackType,
    };

    use super::{AudioError, MixWindow, audio_slice};
    use crate::Snapshot;

    fn tone(value: f32) -> Pcm {
        Pcm::from_f32(4_000, ChannelLayout::mono(), vec![value; 4]).expect("pcm")
    }

    fn draft_with(volume: f64, attribute: TrackAttribute) -> (Draft, SegmentId) {
        let mut draft = Draft::empty("draft");
        let mut track = Track::main("audio");
        track.kind = TrackType::Audio;
        track.attribute = attribute;
        let mut segment = Segment::new("clip", "audio", TimeRange::new(0, 1_000_000));
        segment.volume = volume;
        let id = segment.id.clone();
        track.segments.push(segment);
        draft.tracks = vec![track];
        (draft, id)
    }

    #[test]
    fn volume_is_the_mix_gain_and_a_missing_export_clip_waits() {
        let (draft, id) = draft_with(0.5, TrackAttribute::default());
        let before = draft.clone();
        let window = MixWindow {
            rate: 4_000,
            limiter: 0,
            frames: 4,
        };
        let mixed = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Export,
            |segment, frames, rate| {
                assert_eq!(segment, &id);
                assert_eq!(frames, 4);
                assert_eq!(rate, 4_000);
                Ok(ClipPcm::Ready(tone(1.0)))
            },
        )
        .expect("mix");
        assert_eq!(mixed.contributed, 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[0.5, 0.5, 0.5, 0.5][..]));
        assert_eq!(draft, before);

        let missing = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Export,
            |_, _, _| Ok(ClipPcm::Missing),
        );
        assert_eq!(missing, Err(AudioError::NotReady));

        let skipped = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Playback,
            |_, _, _| Ok(ClipPcm::Missing),
        )
        .expect("playback skip");
        assert_eq!(skipped.contributed, 0);
        assert_eq!(skipped.pcm.samples, 0);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_muted_track_contributes_no_samples() {
        let (draft, _) = draft_with(1.0, TrackAttribute(TrackAttribute::MUTE));
        let window = MixWindow {
            rate: 4_000,
            limiter: 0,
            frames: 4,
        };
        let mut asked = false;
        let mixed = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Export,
            |_, _, _| {
                asked = true;
                Ok(ClipPcm::Missing)
            },
        )
        .expect("mute");
        assert!(!asked);
        assert_eq!(mixed.contributed, 0);
        assert_eq!(mixed.pcm.samples, 0);
    }
}
