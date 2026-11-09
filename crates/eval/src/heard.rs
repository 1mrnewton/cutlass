//! PCM for one media segment at one timeline time.
//!
//! The bytes stay at the file rate. The mix resamples, stretches, and applies
//! volume. A muted track never reaches this module. A path that leaves the
//! project folder is not opened. A missing file is a hole the playback clock
//! skips.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use cutlass_codec::access::{ChannelLayout, Pcm};
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
        let Some(at) = source_at(owner, segment, ask.time) else {
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
        let count = source_frames(ask.frames, ask.rate, file_rate, at.speed);
        if count == 0 {
            return Ok(ClipPcm::Missing);
        }
        let read = match at.read {
            Read::Silent => silence(file_rate, count),
            Read::Forward => media.audio_from(at.source, count).ok(),
            Read::Backward(first) => backward(media, at.source, first, count, file_rate),
        };
        Ok(read.map_or(ClipPcm::Missing, ClipPcm::Ready))
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

/// How the file is read for one window.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Read {
    /// From the source time on.
    Forward,
    /// Back from the source time, no earlier than the first source
    /// microsecond, then played last sample first.
    Backward(Micros),
    /// A frozen picture holds still and makes no sound.
    Silent,
}

/// Where a window starts in the file, how fast it is read, and which way.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SourceAt {
    source: Micros,
    speed: f64,
    read: Read,
}

fn source_at(draft: &Draft, segment: &Segment, time: Micros) -> Option<SourceAt> {
    match placement_of(draft, segment) {
        Ok(map) => {
            let speed = speed_at(&map, time).ok()?;
            let source = map_timeline(&map, time).ok()?;
            let read = if map.freeze.is_some() {
                Read::Silent
            } else if map.reverse {
                // `map_timeline` gives the last microsecond still to play;
                // the window ends just after it.
                Read::Backward(map.source.start)
            } else {
                Read::Forward
            };
            let source = match read {
                Read::Backward(_) => Micros(source.0.saturating_add(1)),
                Read::Forward | Read::Silent => source,
            };
            Some(SourceAt {
                source,
                speed,
                read,
            })
        }
        Err(MapError::NoSource) => {
            let elapsed = time.0.saturating_sub(segment.target.start.0).max(0);
            Some(SourceAt {
                source: Micros(elapsed),
                speed: 1.0,
                read: Read::Forward,
            })
        }
        Err(_) => None,
    }
}

/// `count` frames of mono silence, so an export still has every clip ready.
fn silence(rate: u32, count: u32) -> Option<Pcm> {
    let frames = usize::try_from(count).ok()?;
    Pcm::from_f32(rate, ChannelLayout::mono(), vec![0.0; frames]).ok()
}

/// The `count` frames that end at `end`, last frame first. Near the start of
/// the source range the window is shorter, and the mix reads it as ending
/// early.
fn backward(media: &mut Media, end: Micros, first: Micros, count: u32, rate: u32) -> Option<Pcm> {
    let span = i64::from(count)
        .checked_mul(1_000_000)?
        .checked_div(i64::from(rate))?;
    let (start, frames) = if end.0.saturating_sub(span) >= first.0 {
        (Micros(end.0.saturating_sub(span)), count)
    } else {
        let available = i128::from(end.0.saturating_sub(first.0).max(0));
        let frames = available
            .checked_mul(i128::from(rate))?
            .checked_div(1_000_000)?;
        (first, u32::try_from(frames).ok()?.min(count))
    };
    if frames == 0 {
        return silence(rate, 1);
    }
    let pcm = media.audio_from(start, frames).ok()?;
    reversed(&pcm)
}

/// `pcm` with its frames in the opposite order; each frame keeps its
/// channels in place.
fn reversed(pcm: &Pcm) -> Option<Pcm> {
    let channels = pcm.layout.len();
    if channels == 0 {
        return None;
    }
    if let Some(samples) = pcm.as_f32() {
        let flipped = samples
            .chunks_exact(channels)
            .rev()
            .flatten()
            .copied()
            .collect();
        return Pcm::from_f32(pcm.rate, pcm.layout.clone(), flipped).ok();
    }
    let samples = pcm.as_i16()?;
    let flipped = samples
        .chunks_exact(channels)
        .rev()
        .flatten()
        .copied()
        .collect();
    Pcm::from_i16(pcm.rate, pcm.layout.clone(), flipped).ok()
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

    use cutlass_codec::access::ChannelLayout;
    use cutlass_container::{Demux, OpenError};
    use cutlass_engine::{ClipPcm, ClockKind};
    use cutlass_model::{
        Draft, FreezeInfo, MaterialAudio, MaterialDraft, MaterialId, MaterialVideo, Micros,
        Segment, TimeRange, Track, TrackAttribute, TrackType,
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

    fn ramp(name: &str) -> PathBuf {
        let path = temp(name);
        let samples: Vec<i16> = (0..48).map(|index| index * 100).collect();
        write_tone(&path, &samples);
        path
    }

    fn ask_at(draft: &Draft, time: i64, frames: u32) -> ClipPcm {
        let readers = readers();
        Heard::new()
            .clip(ClipAsk {
                readers: &readers,
                draft,
                revision: 1,
                segment: &draft.tracks[0].segments[0].id,
                time: Micros(time),
                frames,
                rate: 48_000,
                root: None,
            })
            .expect("clip")
    }

    fn values(pcm: &ClipPcm) -> Vec<i32> {
        let ClipPcm::Ready(pcm) = pcm else {
            panic!("not ready: {pcm:?}");
        };
        if let Some(samples) = pcm.as_i16() {
            return samples.iter().map(|sample| i32::from(*sample)).collect();
        }
        let samples = pcm.as_f32().expect("f32");
        #[allow(clippy::cast_possible_truncation)]
        samples
            .iter()
            .map(|sample| (sample * 32_768.0).round() as i32)
            .collect()
    }

    #[test]
    fn a_reversed_clip_plays_its_file_backward_from_the_end() {
        let path = ramp("reverse");
        let mut draft = audio_draft(&path.display().to_string(), 1.0, false);
        draft.tracks[0].segments[0].target = TimeRange::new(0, 1_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 1_000));
        assert_eq!(values(&ask_at(&draft, 0, 4)), vec![0, 100, 200, 300]);

        draft.tracks[0].segments[0].reverse = true;
        assert_eq!(
            values(&ask_at(&draft, 0, 4)),
            vec![4_700, 4_600, 4_500, 4_400]
        );
        // The last window stops at the first sample instead of reading past it.
        let last = values(&ask_at(&draft, 980, 4));
        assert_eq!(last.last(), Some(&0), "{last:?}");
        assert!(last.windows(2).all(|pair| pair[0] > pair[1]), "{last:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_frozen_clip_is_ready_silence() {
        let path = ramp("freeze");
        let mut draft = audio_draft(&path.display().to_string(), 1.0, false);
        let mut video = MaterialVideo::new("tone");
        video.path = path.display().to_string();
        video.freeze = Some(FreezeInfo {
            timestamp: Micros(500),
            ..FreezeInfo::default()
        });
        draft.materials.audios.clear();
        draft.materials.videos.push(video);
        let heard = values(&ask_at(&draft, 0, 4));
        assert_eq!(heard, vec![0; 4]);
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

    // --- Surround files against `ffmpeg -ac 2` ------------------------------

    fn has_ffmpeg() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn ffmpeg(args: &[&str]) -> bool {
        std::process::Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(args)
            .status()
            .is_ok_and(|status| status.success())
    }

    fn f32le(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .expect("raw")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    /// `frames` of the playback mix at `time`, read through [`Heard`].
    fn heard_window(draft: &Draft, time: i64, frames: u32) -> cutlass_engine::MixResult {
        let readers = readers();
        let mut heard = Heard::new();
        audio_slice(
            &Snapshot::new(draft),
            Micros(time),
            MixWindow {
                rate: 48_000,
                limiter: 0,
                frames,
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
                    root: None,
                })
            },
        )
        .expect("mix")
    }

    /// The playback mix of `wav` beside ffmpeg's default stereo downmix of
    /// the same file. `scale` is what one of our samples is in ffmpeg's
    /// units: an s16 file is full scale at 32767 here and 32768 there.
    fn worst_against_ffmpeg(name: &str, wav: &Path, scale: f32) -> f32 {
        let reference = temp(&format!("{name}-reference"));
        assert!(ffmpeg(&[
            "-i",
            wav.to_str().unwrap(),
            "-ac",
            "2",
            "-f",
            "f32le",
            reference.to_str().unwrap(),
        ]));
        let theirs = f32le(&reference);
        let _ = std::fs::remove_file(&reference);
        let draft = audio_draft(&wav.display().to_string(), 1.0, false);
        // A tenth of a second in, so the window is not the file's start.
        let mixed = heard_window(&draft, 100_000, 4_800);
        assert_eq!(mixed.contributed, 1, "{name}");
        assert_eq!(mixed.pcm.layout, ChannelLayout::stereo(), "{name}");
        let ours = mixed.pcm.as_f32().expect("f32");
        assert_eq!(ours.len(), 9_600, "{name}");
        let theirs = &theirs[9_600..19_200];
        let worst = ours
            .iter()
            .zip(theirs)
            .map(|(ours, theirs)| (ours * scale - theirs).abs())
            .fold(0.0_f32, f32::max);
        let peak = theirs.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
        eprintln!("{name}: peak {peak:.4}, worst difference from ffmpeg {worst:e}");
        worst
    }

    /// 5.1 and 7.1 wave files from ffmpeg play through the mix as the same
    /// stereo `ffmpeg -ac 2` makes of them. The 5.1 file is 16-bit with
    /// side surrounds (mask 0x60F); the 7.1 file is float (mask 0x63F).
    #[test]
    fn a_surround_wave_plays_as_ffmpeg_folds_it() {
        let six = temp("five-one.wav");
        let eight = temp("seven-one.wav");
        let tones = |channels: usize| {
            (0..channels)
                .map(|index| format!("0.{}*sin(2*PI*{}*t)", 9 - index, 110 * (index + 2)))
                .collect::<Vec<_>>()
                .join("|")
        };
        let five_one = format!("aevalsrc=exprs={}:c=5.1(side):s=48000:d=1", tones(6));
        let seven_one = format!("aevalsrc=exprs={}:c=7.1:s=48000:d=1", tones(8));
        if !has_ffmpeg() {
            eprintln!("not available: ffmpeg");
            return;
        }
        assert!(ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            &five_one,
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            six.to_str().unwrap(),
        ]));
        assert!(ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            &seven_one,
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            eight.to_str().unwrap(),
        ]));
        let tolerance = 1e-6;
        assert!(worst_against_ffmpeg("5.1 s16", &six, 32_767.0 / 32_768.0) <= tolerance);
        assert!(worst_against_ffmpeg("7.1 f32", &eight, 1.0) <= tolerance);
        let _ = std::fs::remove_file(&six);
        let _ = std::fs::remove_file(&eight);
    }

    /// Real sound from `../local-files`, spread by ffmpeg over 5.1, folds
    /// back the way ffmpeg folds it.
    #[test]
    fn real_sound_spread_over_five_one_folds_as_ffmpeg_does() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !source.is_file() {
            eprintln!("skipped: {} is missing", source.display());
            return;
        }
        if !has_ffmpeg() {
            eprintln!("not available: ffmpeg");
            return;
        }
        let wav = temp("real-five-one.wav");
        let spread = "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=c0|BL=0.8*c0|BR=0.8*c1";
        assert!(ffmpeg(&[
            "-ss",
            "30",
            "-t",
            "1",
            "-i",
            source.to_str().unwrap(),
            "-af",
            spread,
            "-ar",
            "48000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            wav.to_str().unwrap(),
        ]));
        assert!(worst_against_ffmpeg("real 5.1", &wav, 1.0) <= 1e-6);
        let _ = std::fs::remove_file(&wav);
    }
}
