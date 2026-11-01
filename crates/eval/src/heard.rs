//! PCM for one media segment at one timeline time.
//!
//! The bytes stay at the file rate. The mix resamples, stretches, and applies
//! volume. A muted track never reaches this module. A path that leaves the
//! project folder is not opened. A missing file is a hole the playback clock
//! skips.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use cutlass_engine::{
    ClipPcm, EngineError, MapError, Media, ReaderFactory, map_timeline, media_path, placement_of,
    speed_at,
};
use cutlass_model::{Draft, DraftId, Micros, Segment, SegmentId};

use crate::AudioError;

/// Source frames one preview window may ask a file for.
const MAX_SOURCE_FRAMES: u32 = 480_000;

/// What [`Heard::clip`] needs from the mix.
pub struct ClipAsk<'a> {
    pub readers: &'a ReaderFactory,
    pub draft: &'a Draft,
    pub revision: u64,
    pub segment: &'a SegmentId,
    pub time: Micros,
    pub frames: u32,
    pub rate: u32,
    pub root: Option<&'a Path>,
}

/// Open audio files for one draft revision.
pub struct Heard {
    revision: u64,
    bound: bool,
    open: HashMap<String, Media>,
}

impl Default for Heard {
    fn default() -> Self {
        Self::new()
    }
}

impl Heard {
    #[must_use]
    pub fn new() -> Self {
        Self {
            revision: 0,
            bound: false,
            open: HashMap::new(),
        }
    }

    /// Drops every open file.
    pub fn clear(&mut self) {
        self.open.clear();
        self.revision = 0;
        self.bound = false;
    }

    /// Source-rate PCM for `ask.segment` at `ask.time`.
    ///
    /// The frame count covers `ask.frames` of mix audio after the mix stretches
    /// by the segment speed. A file that cannot be opened is [`ClipPcm::Missing`].
    pub fn clip(&mut self, ask: ClipAsk<'_>) -> Result<ClipPcm, AudioError> {
        if ask.rate == 0 {
            return Err(AudioError::NoClock);
        }
        if ask.frames == 0 {
            return Ok(ClipPcm::Missing);
        }
        self.prepare(ask.revision);
        let mut seen = Vec::new();
        let Some((owner, segment)) = find(ask.draft, ask.segment, &mut seen) else {
            return Ok(ClipPcm::Missing);
        };
        let Some(stored) = media_path(owner, segment) else {
            return Ok(ClipPcm::Missing);
        };
        let Some(path) = media_file(ask.root, stored) else {
            return Ok(ClipPcm::Missing);
        };
        let (source, speed) = source_at(owner, segment, ask.time);
        let Some(source) = source else {
            return Ok(ClipPcm::Missing);
        };
        let Ok(media) = self.cached(ask.readers, &path) else {
            return Ok(ClipPcm::Missing);
        };
        let file_rate = media
            .info()
            .streams
            .iter()
            .find_map(|stream| stream.sample_rate)
            .filter(|rate| *rate > 0)
            .unwrap_or(ask.rate);
        let count = source_frames(ask.frames, ask.rate, file_rate, speed);
        if count == 0 {
            return Ok(ClipPcm::Missing);
        }
        match media.audio_from(source, count) {
            Ok(pcm) => Ok(ClipPcm::Ready(pcm)),
            Err(_) => Ok(ClipPcm::Missing),
        }
    }

    fn prepare(&mut self, revision: u64) {
        if self.bound && self.revision == revision {
            return;
        }
        self.open.clear();
        self.revision = revision;
        self.bound = true;
    }

    fn cached(&mut self, readers: &ReaderFactory, path: &Path) -> Result<&mut Media, EngineError> {
        let key = path.to_string_lossy().into_owned();
        if self.open.contains_key(&key) {
            return self.open.get_mut(&key).ok_or(EngineError::MissingSegment);
        }
        let media = readers.open(path)?;
        self.open.insert(key.clone(), media);
        self.open.get_mut(&key).ok_or(EngineError::MissingSegment)
    }
}

fn find<'a>(
    draft: &'a Draft,
    id: &SegmentId,
    seen: &mut Vec<DraftId>,
) -> Option<(&'a Draft, &'a Segment)> {
    if seen.iter().any(|seen_id| seen_id == &draft.id) {
        return None;
    }
    seen.push(draft.id.clone());
    for track in &draft.tracks {
        if let Some(segment) = track.segments.iter().find(|segment| &segment.id == id) {
            return Some((draft, segment));
        }
    }
    for material in &draft.materials.drafts {
        if let Some(child) = material.draft.as_deref()
            && let Some(found) = find(child, id, seen)
        {
            return Some(found);
        }
    }
    None
}

fn source_at(draft: &Draft, segment: &Segment, time: Micros) -> (Option<Micros>, f64) {
    match placement_of(draft, segment) {
        Ok(map) => match (speed_at(&map, time), map_timeline(&map, time)) {
            (Ok(speed), Ok(source)) => (Some(source), speed),
            _ => (None, 1.0),
        },
        Err(MapError::NoSource) => {
            let elapsed = time.0.saturating_sub(segment.target.start.0).max(0);
            (Some(Micros(elapsed)), 1.0)
        }
        Err(_) => (None, 1.0),
    }
}

fn source_frames(timeline: u32, mix_rate: u32, file_rate: u32, speed: f64) -> u32 {
    if timeline == 0 || mix_rate == 0 || file_rate == 0 || !speed.is_finite() || speed <= 0.0 {
        return 0;
    }
    let count = (f64::from(timeline) * speed * f64::from(file_rate) / f64::from(mix_rate)).round();
    if !count.is_finite() || count < 1.0 {
        return 0;
    }
    if count >= f64::from(MAX_SOURCE_FRAMES) {
        return MAX_SOURCE_FRAMES;
    }
    // `count` is finite and inside `1..MAX_SOURCE_FRAMES`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        count as u32
    }
}

fn media_file(root: Option<&Path>, stored: &str) -> Option<PathBuf> {
    if stored.is_empty() {
        return None;
    }
    let path = Path::new(stored);
    if path.is_absolute() {
        return Some(path.to_path_buf());
    }
    let root = root?;
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return None;
    }
    let joined = root.join(path);
    let root = root.canonicalize().ok()?;
    let canonical = joined.canonicalize().ok()?;
    canonical.starts_with(&root).then_some(canonical)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use cutlass_container::{Demux, OpenError};
    use cutlass_engine::{ClipPcm, ClockKind};
    use cutlass_model::{
        Draft, MaterialAudio, MaterialDraft, MaterialId, Micros, Segment, TimeRange, Track,
        TrackAttribute, TrackType,
    };

    use super::{ClipAsk, Heard};
    use crate::{MixWindow, Snapshot, audio_slice};

    fn open_wav(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        let file = File::open(path).map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?;
        let length = file
            .metadata()
            .map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?
            .len();
        Ok(Box::new(cutlass_riff::WavDemux::open(file, length)?))
    }

    fn readers() -> cutlass_engine::ReaderFactory {
        let mut readers = cutlass_engine::ReaderFactory::new();
        readers.register_container(cutlass_riff::probe, open_wav);
        cutlass_pcm::register_decoders(readers.decoders_mut());
        readers
    }

    fn write_tone(path: &Path, samples: &[i16]) {
        let mut payload = Vec::new();
        for sample in samples {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        let data_bytes = u32::try_from(payload.len()).expect("size");
        let rate = 48_000_u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).expect("wav");
    }

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cutlass-heard-{name}-{}", std::process::id()))
    }

    fn audio_draft(path: &str, volume: f64, mute: bool) -> Draft {
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(1_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: path.to_owned(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("audio");
        track.kind = TrackType::Audio;
        if mute {
            track.attribute = TrackAttribute(TrackAttribute::MUTE);
        }
        let mut segment = Segment::new("clip", "tone", TimeRange::new(0, 1_000_000));
        segment.volume = volume;
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        track.segments.push(segment);
        draft.tracks = vec![track];
        draft
    }

    fn mix_at(draft: &Draft, root: Option<&Path>) -> cutlass_engine::MixResult {
        let readers = readers();
        let mut heard = Heard::new();
        let before = draft.clone();
        let mixed = audio_slice(
            &Snapshot::new(draft),
            Micros(0),
            MixWindow {
                rate: 48_000,
                limiter: 0,
                frames: 4,
            },
            ClockKind::Playback,
            |segment, time, frames, rate| {
                heard.clip(ClipAsk {
                    readers: &readers,
                    draft,
                    revision: 1,
                    segment,
                    time,
                    frames,
                    rate,
                    root,
                })
            },
        )
        .expect("mix");
        assert_eq!(draft, &before);
        mixed
    }

    #[test]
    fn volume_scales_the_file_and_a_mute_does_not_read_it() {
        let path = temp("tone");
        write_tone(&path, &[32_767; 4]);
        let draft = audio_draft(&path.display().to_string(), 0.5, false);
        let mixed = mix_at(&draft, None);
        assert_eq!(mixed.contributed, 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[0.5; 4][..]));

        let muted = audio_draft(&path.display().to_string(), 1.0, true);
        let mixed = mix_at(&muted, None);
        assert_eq!(mixed.contributed, 0);
        assert_eq!(mixed.pcm.samples, 0);

        let missing = audio_draft("/no/such/cutlass-heard-tone.wav", 1.0, false);
        let mixed = mix_at(&missing, None);
        assert_eq!(mixed.contributed, 0);
        assert!(matches!(heard_pcm(&missing), ClipPcm::Missing));
        let _ = std::fs::remove_file(&path);
    }

    fn heard_pcm(draft: &Draft) -> ClipPcm {
        let readers = readers();
        let mut heard = Heard::new();
        let mut pcm = ClipPcm::Missing;
        let _ = audio_slice(
            &Snapshot::new(draft),
            Micros(0),
            MixWindow {
                rate: 48_000,
                limiter: 0,
                frames: 4,
            },
            ClockKind::Playback,
            |segment, time, frames, rate| {
                pcm = heard.clip(ClipAsk {
                    readers: &readers,
                    draft,
                    revision: 1,
                    segment,
                    time,
                    frames,
                    rate,
                    root: None,
                })?;
                Ok(ClipPcm::Missing)
            },
        );
        pcm
    }

    #[test]
    fn a_relative_path_stays_in_the_project_and_a_parent_is_silent() {
        let root = temp("project");
        std::fs::create_dir_all(&root).expect("root");
        let tone = root.join("tone.wav");
        write_tone(&tone, &[32_767; 4]);
        let draft = audio_draft("tone.wav", 0.5, false);
        let mixed = mix_at(&draft, Some(&root));
        assert_eq!(mixed.pcm.as_f32(), Some(&[0.5; 4][..]));

        let name = format!("cutlass-heard-escape-{}.wav", std::process::id());
        let outside = root.join("..").join(&name);
        write_tone(&outside, &[32_767; 4]);
        let escaped = audio_draft(&format!("../{name}"), 1.0, false);
        let mixed = mix_at(&escaped, Some(&root));
        assert_eq!(mixed.contributed, 0);
        assert_eq!(mixed.pcm.samples, 0);

        let _ = std::fs::remove_file(&tone);
        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir(&root);
    }

    #[test]
    fn a_nested_file_keeps_the_child_samples_and_the_parent_gain() {
        let path = temp("nested");
        write_tone(&path, &[32_767; 4]);
        let child = audio_draft(&path.display().to_string(), 1.0, false);
        let mut draft = Draft::empty("parent");
        draft.duration = Micros(1_000_000);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("nest"),
            draft: Some(Box::new(child)),
            ..MaterialDraft::default()
        });
        let mut track = Track::main("parent");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("compound", "nest", TimeRange::new(0, 1_000_000));
        segment.volume = 0.5;
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        track.segments.push(segment);
        draft.tracks = vec![track];
        let mixed = mix_at(&draft, None);
        assert_eq!(mixed.contributed, 1);
        assert_eq!(mixed.pcm.as_f32(), Some(&[0.5; 4][..]));
        let _ = std::fs::remove_file(&path);
    }
}
