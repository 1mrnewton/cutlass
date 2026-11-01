//! The mixed PCM at one timeline time.
//!
//! Clips come from the engine mix. This module does not resample, fade, or
//! limit on its own. A muted track contributes nothing because the mix skips
//! it. A compound segment is the child draft's mix at the mapped time, and
//! the parent then applies that segment's volume, speed, and fades. A cycle,
//! a chain past the depth limit, or a material with no inlined draft
//! contributes no samples. Export passes [`ClockKind::Export`], so a missing
//! clip is an error. Playback can pass [`ClockKind::Playback`] and skip that
//! clip.

use cutlass_engine::{
    ClipPcm, ClockKind, Mix, MixResult, contributes_sound, map_timeline, mix_inputs, placement_of,
    render,
};
use cutlass_model::{Draft, DraftId, MaterialRef, Micros, Segment, SegmentId};

use crate::nest::{descend, skips_child};

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
/// `fill` is asked once per media segment the mix will hear, at the time
/// that mix is using. Inside a compound segment that time is the mapped
/// child time. A muted track is skipped first, so its source is not read.
/// A compound segment is filled from the child mix instead. The mix then
/// applies volume and fades. The draft is not written.
pub fn audio_slice<F>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    window: MixWindow,
    clock: ClockKind,
    mut fill: F,
) -> Result<MixResult, AudioError>
where
    F: FnMut(&SegmentId, Micros, u32, u32) -> Result<ClipPcm, AudioError>,
{
    if window.rate == 0 {
        return Err(AudioError::NoClock);
    }
    mix_at(snapshot, time, window, clock, &mut fill, &[])
}

fn mix_at<F>(
    snapshot: &Snapshot<'_>,
    time: Micros,
    window: MixWindow,
    clock: ClockKind,
    fill: &mut F,
    ancestors: &[DraftId],
) -> Result<MixResult, AudioError>
where
    F: FnMut(&SegmentId, Micros, u32, u32) -> Result<ClipPcm, AudioError>,
{
    let draft = snapshot.draft();
    let mut clips = mix_inputs(draft, time);
    let segments = audible(draft, time);
    if clips.len() != segments.len() {
        return Err(AudioError::Mix);
    }
    let mut keep = Vec::with_capacity(clips.len());
    for (clip, segment) in clips.iter_mut().zip(segments) {
        if clip.muted {
            keep.push(true);
            continue;
        }
        match compound_pcm(draft, segment, time, window, clock, fill, ancestors)? {
            Compound::Media => {
                clip.pcm = fill(&segment.id, time, window.frames, window.rate)?;
                keep.push(true);
            }
            Compound::Slice(mixed) => {
                clip.pcm = ClipPcm::Ready(mixed.pcm);
                keep.push(true);
            }
            Compound::Drop => keep.push(false),
        }
    }
    let clips: Vec<cutlass_engine::MixClip> = clips
        .into_iter()
        .zip(keep)
        .filter(|(_, heard)| *heard)
        .map(|(clip, _)| clip)
        .collect();
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

enum Compound {
    Media,
    Slice(MixResult),
    Drop,
}

fn compound_pcm<F>(
    draft: &Draft,
    segment: &Segment,
    time: Micros,
    window: MixWindow,
    clock: ClockKind,
    fill: &mut F,
    ancestors: &[DraftId],
) -> Result<Compound, AudioError>
where
    F: FnMut(&SegmentId, Micros, u32, u32) -> Result<ClipPcm, AudioError>,
{
    let Some(MaterialRef::Draft(material)) = draft.materials.get(segment.material_id.as_str())
    else {
        return Ok(Compound::Media);
    };
    let Some(child) = material.draft.as_deref() else {
        return Ok(Compound::Drop);
    };
    if skips_child(draft, child, ancestors) {
        return Ok(Compound::Drop);
    }
    let Ok(map) = placement_of(draft, segment) else {
        return Ok(Compound::Drop);
    };
    let Ok(mapped) = map_timeline(&map, time) else {
        return Ok(Compound::Drop);
    };
    let chain = descend(ancestors, &draft.id);
    let mixed = mix_at(&Snapshot::new(child), mapped, window, clock, fill, &chain)?;
    if mixed.pcm.samples == 0 {
        Ok(Compound::Drop)
    } else {
        Ok(Compound::Slice(mixed))
    }
}

fn audible(draft: &Draft, time: Micros) -> Vec<&Segment> {
    let mut segments = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if !contributes_sound(draft, segment) {
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
    use cutlass_engine::{ClipPcm, ClockKind, MixResult};
    use cutlass_model::{
        Draft, LinkPair, MaterialAudio, MaterialAudioFade, MaterialDraft, MaterialId,
        MaterialVideo, Micros, Relationship, Segment, SegmentId, TimeRange, Track, TrackAttribute,
        TrackType,
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
            |segment, when, frames, rate| {
                assert_eq!(segment, &id);
                assert_eq!(when, Micros(0));
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
            |_, _, _, _| Ok(ClipPcm::Missing),
        );
        assert_eq!(missing, Err(AudioError::NotReady));

        let skipped = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Playback,
            |_, _, _, _| Ok(ClipPcm::Missing),
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
            |_, _, _, _| {
                asked = true;
                Ok(ClipPcm::Missing)
            },
        )
        .expect("mute");
        assert!(!asked);
        assert_eq!(mixed.contributed, 0);
        assert_eq!(mixed.pcm.samples, 0);
    }

    #[test]
    fn a_separated_picture_sound_is_heard_once() {
        let mut draft = Draft::empty("draft");
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
        let mut lane = Track::main("sound");
        lane.kind = TrackType::Audio;
        lane.segments
            .push(Segment::new("sound", "sound", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(lane);
        draft.relationships.push(Relationship {
            kind: "linkage".to_owned(),
            id_to_id: vec![LinkPair {
                owner: SegmentId::new("picture"),
                dependent: SegmentId::new("sound"),
            }],
            ..Relationship::default()
        });
        let window = MixWindow {
            rate: 4_000,
            limiter: 0,
            frames: 4,
        };
        let mut asked = 0;
        let mixed = audio_slice(
            &Snapshot::new(&draft),
            Micros(0),
            window,
            ClockKind::Playback,
            |segment, _, _, _| {
                asked += 1;
                assert_eq!(segment.as_str(), "sound");
                Ok(ClipPcm::Ready(tone(1.0)))
            },
        )
        .expect("mix");
        assert_eq!(asked, 1);
        assert_eq!(mixed.contributed, 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[1.0, 1.0, 1.0, 1.0][..]));
    }

    fn window() -> MixWindow {
        MixWindow {
            rate: 4_000,
            limiter: 0,
            frames: 4,
        }
    }

    fn child_tone(id: &str, volume: f64, start: i64, duration: i64) -> Draft {
        let mut draft = Draft::empty(id);
        let mut track = Track::main("child-track");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("child-clip", "tone", TimeRange::new(start, duration));
        segment.volume = volume;
        track.segments.push(segment);
        draft.tracks = vec![track];
        draft
    }

    fn compound_audio(
        id: &str,
        child: Draft,
        volume: f64,
        source: TimeRange,
        target: TimeRange,
    ) -> Draft {
        let mut draft = Draft::empty(id);
        draft.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut track = Track::main("parent-track");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("compound", "nest", target);
        segment.volume = volume;
        segment.source_range = Some(source);
        track.segments.push(segment);
        draft.tracks = vec![track];
        draft
    }

    fn span(start: i64, duration: i64) -> TimeRange {
        TimeRange::new(start, duration)
    }

    fn asked(draft: &Draft, time: i64) -> (MixResult, Vec<(SegmentId, Micros)>) {
        let mut heard = Vec::new();
        let mixed = audio_slice(
            &Snapshot::new(draft),
            Micros(time),
            window(),
            ClockKind::Export,
            |segment, when, frames, rate| {
                assert_eq!((frames, rate), (4, 4_000));
                heard.push((segment.clone(), when));
                Ok(ClipPcm::Ready(tone(1.0)))
            },
        )
        .expect("mix");
        (mixed, heard)
    }

    #[test]
    fn a_nested_draft_is_one_slice_with_the_parent_gain_and_fade() {
        let full = span(0, 1_000_000);
        let draft = compound_audio(
            "parent",
            child_tone("child", 0.5, 0, 1_000_000),
            0.5,
            full.clone(),
            full.clone(),
        );
        let before = draft.clone();
        let (mixed, heard) = asked(&draft, 0);
        assert_eq!(heard.len(), 1);
        assert_eq!(heard[0].0.as_str(), "child-clip");
        assert_eq!(mixed.contributed, 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[0.25; 4][..]));
        assert_eq!(draft, before);

        let mapped = compound_audio(
            "parent",
            child_tone("child", 1.0, 1_000, 1_000),
            1.0,
            span(1_000, 1_000),
            span(0, 1_000),
        );
        let (mixed, heard) = asked(&mapped, 0);
        assert_eq!(heard.len(), 1);
        assert_eq!(heard[0].1, Micros(1_000));
        assert_eq!(mixed.pcm.as_f32(), Some(&[1.0; 4][..]));

        let mut faded = compound_audio(
            "parent",
            child_tone("child", 1.0, 0, 1_000_000),
            1.0,
            full.clone(),
            full,
        );
        faded.materials.audio_fades.push(MaterialAudioFade {
            id: "fade".into(),
            fade_type: 2,
            fade_in_duration: Micros(1_000_000),
            ..MaterialAudioFade::default()
        });
        faded.tracks[0].segments[0]
            .extra_material_refs
            .push("fade".into());
        let (mixed, _) = asked(&faded, 0);
        let samples = mixed.pcm.as_f32().expect("f32");
        assert!(samples[0].abs() < 1.0e-6);
        assert!((samples[3] - 0.75).abs() < 1.0e-5);
    }

    #[test]
    fn speed_reverse_and_a_cycle_choose_whether_the_child_is_heard() {
        let mut reversed = compound_audio(
            "parent",
            child_tone("child", 1.0, 0, 100),
            1.0,
            span(0, 1_000),
            span(0, 1_000),
        );
        reversed.tracks[0].segments[0].reverse = true;
        let (silent, heard) = asked(&reversed, 0);
        assert!(heard.is_empty());
        assert_eq!(silent.contributed, 0);
        let (mixed, heard) = asked(&reversed, 900);
        assert_eq!(heard.len(), 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[1.0; 4][..]));

        let cycle = compound_audio(
            "parent",
            child_tone("parent", 1.0, 0, 1_000_000),
            1.0,
            span(0, 1_000_000),
            span(0, 1_000_000),
        );
        let (mixed, heard) = asked(&cycle, 0);
        assert!(heard.is_empty());
        assert_eq!(mixed.contributed, 0);

        let echoed = compound_audio(
            "mid",
            child_tone("outer", 1.0, 0, 1_000_000),
            1.0,
            span(0, 1_000_000),
            span(0, 1_000_000),
        );
        let outer = compound_audio("outer", echoed, 1.0, span(0, 1_000_000), span(0, 1_000_000));
        let (mixed, heard) = asked(&outer, 0);
        assert!(heard.is_empty());
        assert_eq!(mixed.contributed, 0);

        let mut stalled = compound_audio(
            "parent",
            child_tone("child", 1.0, 0, 1_000_000),
            1.0,
            span(0, 1_000_000),
            span(0, 1_000_000),
        );
        stalled.tracks[0].segments[0].speed = 0.0;
        let mut sibling = Track::main("sibling-track");
        sibling.kind = TrackType::Audio;
        sibling
            .segments
            .push(Segment::new("sibling", "tone", span(0, 1_000_000)));
        stalled.tracks.push(sibling);
        let (mixed, heard) = asked(&stalled, 0);
        assert_eq!(heard.len(), 1);
        assert_eq!(heard[0].0.as_str(), "sibling");
        assert_eq!(mixed.pcm.as_f32(), Some(&[1.0; 4][..]));
    }

    #[test]
    fn a_missing_child_a_mute_and_a_deep_chain_add_no_samples() {
        let mut path = Draft::empty("parent");
        path.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft_file_path: "missing.json".into(),
            ..MaterialDraft::default()
        });
        let mut track = Track::main("parent-track");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("compound", "nest", span(0, 1_000_000));
        segment.source_range = Some(span(0, 1_000_000));
        track.segments.push(segment);
        path.tracks = vec![track];
        let (mixed, heard) = asked(&path, 0);
        assert!(heard.is_empty());
        assert_eq!(mixed.contributed, 0);

        let mut muted = compound_audio(
            "parent",
            child_tone("child", 1.0, 0, 1_000_000),
            1.0,
            span(0, 1_000_000),
            span(0, 1_000_000),
        );
        muted.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE);
        let (mixed, heard) = asked(&muted, 0);
        assert!(heard.is_empty());
        assert_eq!(mixed.contributed, 0);

        let waiting = audio_slice(
            &Snapshot::new(&compound_audio(
                "parent",
                child_tone("child", 1.0, 0, 1_000_000),
                1.0,
                span(0, 1_000_000),
                span(0, 1_000_000),
            )),
            Micros(0),
            window(),
            ClockKind::Export,
            |_, _, _, _| Ok(ClipPcm::Missing),
        );
        assert_eq!(waiting, Err(AudioError::NotReady));

        let mut deep = child_tone("tone", 1.0, 0, 1_000_000);
        for index in 0..33 {
            deep = compound_audio(
                &format!("level-{index}"),
                deep,
                1.0,
                span(0, 1_000_000),
                span(0, 1_000_000),
            );
        }
        let (mixed, heard) = asked(&deep, 0);
        assert!(heard.is_empty());
        assert_eq!(mixed.contributed, 0);

        let nested = compound_audio(
            "outer",
            compound_audio(
                "inner",
                child_tone("tone", 1.0, 0, 1_000_000),
                1.0,
                span(0, 1_000_000),
                span(0, 1_000_000),
            ),
            1.0,
            span(0, 1_000_000),
            span(0, 1_000_000),
        );
        let (mixed, heard) = asked(&nested, 0);
        assert_eq!(heard.len(), 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[1.0; 4][..]));
    }
}
