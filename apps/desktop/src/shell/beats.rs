//! Auto beats: mark the beats of the selected audio clip.
//!
//! The clip's sound is read, as the player reads it, and heard by the beat model as a job. When it
//! lands, every beat found is stored on the clip as a beat mark, in one
//! undo step. Marks already on the clip stay, and a found beat that sits
//! on one of them is left out.

use std::path::PathBuf;

use cutlass_codec::access::{ChannelLayout, Pcm};
use cutlass_engine::{Media, ReaderFactory};
use cutlass_model::{
    MaterialId, Micros, beat_timeline_time, segment_beats, track_holds_beats, user_beats,
};
use cutlass_session::Request;
use cutlass_transcription::HeardClip;
use cutlass_vision::{BEATS_MAX_SAMPLES, BEATS_MIN_SAMPLES, BEATS_RATE, BEATS_WEIGHTS, BeatModel};

use super::enhance::model_dir;
use super::{Shell, ShellError, placed};
use crate::jobs::Progress;

/// The job's label.
pub(crate) const BEATS_JOB: &str = "Auto beats";

/// The folder in `Models` holding the beat model.
pub(super) const BEATS_FOLDER: &str = "beats";

/// The sound resampled at a time: half a minute, so a long file is never
/// held whole at its own rate.
const PIECE: i64 = 30_000_000;

/// The most sample frames asked of the file in one read: the engine's own
/// limit is 480,000, ten seconds at 48 kHz.
const WINDOW: u32 = 240_000;

/// The share of the job that is reading the file; the rest is the model.
const READING: f32 = 0.2;

/// A found beat this close to a mark already on the clip is that mark.
/// The model answers every 20 ms and a beat is rarely under 200 ms long.
const NEAR: i64 = 70_000;

/// The samples `micros` of sound take at `rate`.
fn samples_at(micros: i64, rate: u32) -> usize {
    let micros = u64::try_from(micros).unwrap_or(0);
    usize::try_from(micros.saturating_mul(u64::from(rate)) / 1_000_000).unwrap_or(usize::MAX)
}

/// The samples `micros` of sound take at the model's rate.
fn samples_in(micros: i64) -> usize {
    samples_at(micros, BEATS_RATE)
}

/// `samples` at `rate`, at the model's rate.
fn at_model_rate(rate: u32, samples: Vec<f32>) -> Option<Vec<f32>> {
    if rate == BEATS_RATE {
        return Some(samples);
    }
    let pcm = Pcm::from_f32(rate, ChannelLayout::mono(), samples).ok()?;
    let fitted = cutlass_audio::resample(&pcm, BEATS_RATE).ok()?;
    fitted.as_f32().map(<[f32]>::to_vec)
}

/// `pcm` as one channel: the mean of each frame's channels.
fn mono(pcm: &Pcm) -> Option<Vec<f32>> {
    let channels = pcm.layout.len();
    let samples = pcm.as_f32()?;
    if channels == 0 {
        return None;
    }
    Some(
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
    )
}

/// The stretch of its file `clip` plays, mono at the model's rate.
///
/// The file is read as the player reads it, so a beat is found where it
/// is heard: a decoder's start-up samples are already gone. Sound that
/// cannot be read is silence, so the rest keeps its time. `None` when
/// none of it could be read, or the job was cancelled.
fn clip_sound(media: &mut Media, clip: &HeardClip, progress: &Progress) -> Option<Vec<f32>> {
    let rate = media
        .info()
        .streams
        .iter()
        .find_map(|stream| stream.sample_rate.filter(|rate| *rate > 0))?;
    let length = clip.range.duration.0;
    let total = samples_at(length, rate);
    let piece = samples_at(PIECE, rate).max(1);
    let mut sound: Vec<f32> = Vec::with_capacity(samples_in(length));
    // Sound at the file's rate, waiting to be fitted to the model's.
    let mut waiting: Vec<f32> = Vec::with_capacity(piece.min(total));
    let mut heard = false;
    let mut read = 0_usize;
    while read < total {
        if progress.cancelled() {
            return None;
        }
        let frames = u32::try_from(total - read).unwrap_or(WINDOW).min(WINDOW);
        // Each read starts on the sample the last one ended on, which the
        // engine continues without a seek.
        let offset = i64::try_from(read as u128 * 1_000_000 / u128::from(rate)).ok()?;
        let start = Micros(clip.range.start.0.saturating_add(offset));
        let window = media
            .audio_from(start, frames)
            .ok()
            .filter(|pcm| pcm.rate == rate)
            .and_then(|pcm| mono(&pcm));
        let end = waiting.len() + frames as usize;
        if let Some(samples) = window {
            heard = true;
            waiting.extend_from_slice(&samples[..samples.len().min(frames as usize)]);
        }
        waiting.resize(end, 0.0);
        read += frames as usize;
        if waiting.len() >= piece || read >= total {
            // Each piece ends on the sample its time ends on, so rounding
            // never adds up along the clip.
            let done = i64::try_from(read as u128 * 1_000_000 / u128::from(rate)).ok()?;
            let end = if read >= total {
                samples_in(length)
            } else {
                samples_in(done)
            };
            if let Some(fitted) = at_model_rate(rate, std::mem::take(&mut waiting)) {
                let room = end.saturating_sub(sound.len());
                sound.extend_from_slice(&fitted[..fitted.len().min(room)]);
            }
            sound.resize(end, 0.0);
        }
        progress.fraction(READING * read as f32 / total as f32);
    }
    heard.then_some(sound)
}

/// What the job found: the beats, in microseconds from the start of the
/// sound the clip plays; `None` when it was cancelled; or why it failed.
type Found = Result<Option<Vec<i64>>, String>;

fn find_beats(
    readers: &ReaderFactory,
    file: &std::path::Path,
    dir: &std::path::Path,
    kernels: &std::path::Path,
    clip: &HeardClip,
    progress: &Progress,
) -> Found {
    progress.stage("Listening to the clip");
    let sound = readers
        .open(file)
        .ok()
        .and_then(|mut media| clip_sound(&mut media, clip, progress));
    let Some(sound) = sound else {
        if progress.cancelled() {
            return Ok(None);
        }
        return Err("the clip's sound could not be read".to_owned());
    };
    progress.stage("Loading the model");
    let compute = crate::gpu_trust::compute();
    if let Err(error) = cutlass_vision::keep_kernels_in(kernels) {
        cutlass_logger::warn!("kernel cache: {error}");
    }
    let model = BeatModel::load(dir, compute).map_err(|error| error.to_string())?;
    progress.stage("Finding the beats");
    let found = model
        .find(&sound, &mut |done| {
            progress.fraction(READING + (1.0 - READING) * done);
            !progress.cancelled()
        })
        .map_err(|error| error.to_string())?;
    Ok(found.map(|found| found.beats))
}

impl Shell {
    /// The selected clip, when its beats can be found: a clip on an audio
    /// track, with a file, playing at one speed, of a length the model
    /// hears.
    fn beat_clip(&self) -> Result<HeardClip, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let on_audio = placed(draft, &segment_id).is_some_and(|(_, kind)| track_holds_beats(kind));
        if !on_audio {
            return Err(ShellError::Unavailable(
                "a beat belongs on audio".to_owned(),
            ));
        }
        self.refuse_locked(&segment_id)?;
        let clip = self.heard_clip(&segment_id)?;
        if !clip.speed_holds {
            return Err(ShellError::Unavailable(
                "the clip does not play at one speed".to_owned(),
            ));
        }
        let samples = samples_in(clip.range.duration.0);
        if samples < BEATS_MIN_SAMPLES {
            return Err(ShellError::Unavailable(
                "the clip is too short to have beats".to_owned(),
            ));
        }
        if samples > BEATS_MAX_SAMPLES {
            return Err(ShellError::Unavailable(
                "the clip is too long to find its beats".to_owned(),
            ));
        }
        Ok(clip)
    }

    /// Whether the selected clip's beats can be found.
    #[must_use]
    pub fn auto_beats_enabled(&self) -> bool {
        self.beat_clip().is_ok()
    }

    /// Find the selected audio clip's beats and mark them, as a job. The
    /// marks land in one undo step, unless the clip changed meanwhile.
    pub fn auto_beats(&mut self) -> Result<(), ShellError> {
        self.auto_beats_with(model_dir(BEATS_FOLDER, BEATS_WEIGHTS))
    }

    /// [`Self::auto_beats`] with the model in folder `dir`.
    fn auto_beats_with(&mut self, dir: PathBuf) -> Result<(), ShellError> {
        let clip = self.beat_clip()?;
        let file = self.sound_file(&clip.path)?;
        if !dir.join(BEATS_WEIGHTS).is_file() {
            return Err(ShellError::Unavailable(format!(
                "the {BEATS_JOB} model is not on this machine"
            )));
        }
        let kernels = Shell::home().cache().join("kernels");
        let readers = self.readers.clone();
        self.start_job(BEATS_JOB, move |progress| {
            let found = find_beats(&readers, &file, &dir, &kernels, &clip, progress);
            Box::new(move |shell: &mut Shell| match found {
                Ok(None) => Ok(()),
                Ok(Some(beats)) => shell.land_beats(&clip, &beats),
                Err(message) => Err(ShellError::Unavailable(message)),
            })
        })
    }

    /// Mark `beats` on the clip they were found in, as one undo step, and
    /// say how many. Each is a time in microseconds from the start of the
    /// sound the clip plays.
    fn land_beats(&mut self, clip: &HeardClip, beats: &[i64]) -> Result<(), ShellError> {
        self.still_heard(clip)?;
        self.refuse_locked(&clip.source_id)?;
        let draft = self.session_draft()?;
        let Some((segment, _)) = placed(draft, &clip.source_id) else {
            return Err(ShellError::Unavailable(
                "the clip changed while it was heard".to_owned(),
            ));
        };
        let mut marked = segment_beats(&draft.materials, segment)
            .map(user_beats)
            .unwrap_or_default();
        let mut sources = Vec::with_capacity(beats.len());
        for beat in beats {
            // A clip played backwards meets its last beat first.
            let offset = if clip.reverse {
                clip.range.duration.0.saturating_sub(*beat)
            } else {
                *beat
            };
            let Some(source) = clip.range.start.0.checked_add(offset) else {
                continue;
            };
            let near = marked
                .iter()
                .any(|mark| mark.abs_diff(source) < NEAR.unsigned_abs());
            if near || beat_timeline_time(segment, source).is_none() {
                continue;
            }
            marked.push(source);
            sources.push(source);
        }
        if sources.is_empty() {
            self.set_notice(if beats.is_empty() {
                "No beats were found in the clip"
            } else {
                "The clip's beats are already marked"
            });
            return Ok(());
        }
        // The first beat makes the clip's beats material when it has none.
        let material_id = MaterialId::new(self.fresh("beats"));
        let requests: Vec<Request> = sources
            .iter()
            .map(|source| Request::AddBeat {
                segment_id: clip.source_id.clone(),
                source: Micros(*source),
                material_id: material_id.clone(),
            })
            .collect();
        self.apply_requests(&requests)?;
        let count = sources.len();
        self.set_notice(format!(
            "Marked {count} beat{}",
            if count == 1 { "" } else { "s" }
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_model::{
        Draft, MaterialAudio, MaterialId, Micros, Segment, SegmentId, TimeRange, Track, TrackType,
    };
    use cutlass_store::Folder;
    use cutlass_vision::BEATS_WEIGHTS;

    use super::{BEATS_RATE, NEAR};
    use crate::shell::tests::{Temp, land_job, shell};

    const RATE: u32 = 22_050;
    const SECONDS: u32 = 20;

    /// A drum every half second from 0.25 s: a low thump that dies away.
    fn drums() -> Vec<f32> {
        let total = (SECONDS * RATE) as usize;
        let mut sound = vec![0.0_f32; total];
        let mut at = RATE as usize / 4;
        while at < total {
            for (index, sample) in sound
                .iter_mut()
                .skip(at)
                .take(RATE as usize / 4)
                .enumerate()
            {
                let time = index as f32 / RATE as f32;
                let tone =
                    (std::f32::consts::TAU * (55.0 + 90.0 * (-time * 40.0).exp()) * time).sin();
                *sample += 0.8 * tone * (-time * 18.0).exp();
            }
            at += RATE as usize / 2;
        }
        sound
    }

    fn write_wav(path: &Path, rate: u32, samples: &[f32]) {
        let data = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 30_000.0) as i16).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// A picture clip on the main track; on a sound lane, `song` playing
    /// its file from two seconds in for ten seconds, placed at five
    /// seconds, and `blip`, a clip of a hundredth of a second.
    fn project(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        write_wav(&dir.join("song.wav"), RATE, &drums());
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.videos.push(cutlass_model::MaterialVideo {
            id: MaterialId::new("camera"),
            path: "song.wav".to_owned(),
            duration: Micros(20_000_000),
            ..cutlass_model::MaterialVideo::default()
        });
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: "song.wav".to_owned(),
            duration: Micros(20_000_000),
            ..MaterialAudio::default()
        });
        let clip = |id: &str, material: &str, at: i64, from: i64, length: i64| {
            let mut segment = Segment::new(id, material, TimeRange::new(at, length));
            segment.source_range = Some(TimeRange::new(from, length));
            segment
        };
        draft.tracks[0]
            .segments
            .push(clip("video", "camera", 0, 0, 20_000_000));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(clip("song", "song", 5_000_000, 2_000_000, 10_000_000));
        sound
            .segments
            .push(clip("blip", "song", 16_000_000, 0, 10_000));
        draft.tracks.push(sound);
        Folder::new(draft).save(dir).unwrap();
    }

    fn opened(temp: &Temp) -> crate::shell::Shell {
        project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    /// The song clip as the job hears it.
    fn song(desktop: &crate::shell::Shell) -> cutlass_transcription::HeardClip {
        desktop.heard_clip(&SegmentId::new("song")).unwrap()
    }

    /// Where the song clip's marks are drawn on the timeline.
    fn marks(desktop: &crate::shell::Shell) -> Vec<i64> {
        let mut times: Vec<i64> = desktop
            .beats()
            .iter()
            .filter(|beat| beat.segment_id.as_str() == "song")
            .map(|beat| beat.time.0)
            .collect();
        times.sort_unstable();
        times
    }

    /// The beat model's folder in `local-files`, beside whichever checkout
    /// this is, when the weights were fetched.
    fn fetched_model() -> Option<PathBuf> {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let model = dir.join("local-files/models/beats");
            if model.join(BEATS_WEIGHTS).is_file() {
                return Some(model);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    #[test]
    fn only_an_audio_clip_of_a_length_the_model_hears_can_have_its_beats_found() {
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        assert!(!desktop.auto_beats_enabled());
        assert_eq!(
            desktop.auto_beats().unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("video")));
        assert!(!desktop.auto_beats_enabled());
        assert_eq!(
            desktop.auto_beats().unwrap_err().to_string(),
            "a beat belongs on audio"
        );
        desktop.select(Some(SegmentId::new("blip")));
        assert!(!desktop.auto_beats_enabled());
        assert_eq!(
            desktop.auto_beats().unwrap_err().to_string(),
            "the clip is too short to have beats"
        );
        desktop.select(Some(SegmentId::new("song")));
        assert!(desktop.auto_beats_enabled());
    }

    #[test]
    fn a_machine_without_the_model_is_told_so_and_starts_no_job() {
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        desktop.select(Some(SegmentId::new("song")));
        let empty = temp.path.join("no-models");
        assert_eq!(
            desktop.auto_beats_with(empty).unwrap_err().to_string(),
            "the Auto beats model is not on this machine"
        );
        assert!(desktop.job().is_none());
    }

    #[test]
    fn found_beats_land_at_their_source_times_in_one_undo_step() {
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        let clip = song(&desktop);
        // The clip plays from two seconds into its file and sits at five.
        // A beat past the clip's end is not one of its beats.
        desktop
            .land_beats(
                &clip,
                &[250_000, 750_000, 9_750_000, 10_000_000, 12_000_000],
            )
            .unwrap();
        assert_eq!(desktop.notice(), "Marked 3 beats");
        assert_eq!(marks(&desktop), [5_250_000, 5_750_000, 14_750_000]);
        let stored: Vec<i64> = desktop.beats().iter().map(|beat| beat.source).collect();
        assert!(stored.contains(&2_250_000), "{stored:?}");

        desktop.undo().unwrap();
        assert!(marks(&desktop).is_empty());
        desktop.redo().unwrap();
        assert_eq!(marks(&desktop).len(), 3);
    }

    #[test]
    fn a_beat_on_a_mark_already_there_is_left_out() {
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        let clip = song(&desktop);
        desktop.land_beats(&clip, &[1_000_000]).unwrap();
        assert_eq!(desktop.notice(), "Marked 1 beat");
        // The same beat a little off, one just past the reach, and two
        // found beats closer together than the reach.
        desktop
            .land_beats(
                &clip,
                &[1_000_000 + NEAR - 1, 1_000_000 + NEAR, 4_000_000, 4_010_000],
            )
            .unwrap();
        assert_eq!(desktop.notice(), "Marked 2 beats");
        assert_eq!(marks(&desktop), [6_000_000, 6_000_000 + NEAR, 9_000_000]);

        desktop.land_beats(&clip, &[1_000_000, 4_000_000]).unwrap();
        assert_eq!(desktop.notice(), "The clip's beats are already marked");
        desktop.land_beats(&clip, &[]).unwrap();
        assert_eq!(desktop.notice(), "No beats were found in the clip");
        assert_eq!(marks(&desktop).len(), 3);
    }

    #[test]
    fn beats_found_in_a_clip_that_changed_meanwhile_are_dropped() {
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        let mut clip = song(&desktop);
        clip.range = TimeRange::new(0, 10_000_000);
        assert_eq!(
            desktop
                .land_beats(&clip, &[1_000_000])
                .unwrap_err()
                .to_string(),
            "the clip changed while it was heard"
        );
        assert!(marks(&desktop).is_empty());
    }

    #[test]
    fn a_clip_is_heard_at_the_models_rate_whatever_its_files_rate() {
        let temp = Temp::new();
        let desktop = opened(&temp);
        let clip = song(&desktop);
        // The same drums at 48 kHz: each lasts as long at the model's rate.
        let fast: Vec<f32> = (0..SECONDS * 48_000)
            .map(|index| if index % 24_000 == 12_000 { 0.9 } else { 0.0 })
            .collect();
        let file = temp.path.join("fast.wav");
        write_wav(&file, 48_000, &fast);
        let progress = crate::jobs::Progress::default();
        let mut media = desktop.readers.open(&file).unwrap();
        let sound = super::clip_sound(&mut media, &clip, &progress).unwrap();
        assert_eq!(sound.len(), 10 * BEATS_RATE as usize);
        // The clip starts two seconds in, so its first click, at 2.25 s of
        // the file, is a quarter second into what the model hears.
        let loudest = sound
            .iter()
            .take(BEATS_RATE as usize / 2)
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(index, _)| index)
            .unwrap();
        assert!(loudest.abs_diff(BEATS_RATE as usize / 4) <= 2, "{loudest}");

        // A clip that runs past the end of its file is silent from there.
        let mut long = clip.clone();
        long.range = TimeRange::new(15_000_000, 10_000_000);
        let mut media = desktop.readers.open(&file).unwrap();
        let sound = super::clip_sound(&mut media, &long, &progress).unwrap();
        assert_eq!(sound.len(), 10 * BEATS_RATE as usize);
        assert!(
            sound[6 * BEATS_RATE as usize..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert!(
            sound[..5 * BEATS_RATE as usize]
                .iter()
                .any(|sample| sample.abs() > 0.1)
        );
    }

    #[test]
    fn the_job_marks_a_drum_every_half_second() {
        // The weights are fetched by tools/vision-models/scripts/fetch_beats.sh.
        let Some(model) = fetched_model() else {
            return;
        };
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        desktop.select(Some(SegmentId::new("song")));
        desktop.auto_beats_with(model).unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Auto beats".to_owned())
        );
        land_job(&mut desktop);
        // Ten seconds of a drum every half second, the first a quarter
        // second into the clip, which sits at five seconds. The model
        // answers every 20 ms. A bare drum is not music, and the model is
        // unsure of a few of them: every mark is on a drum, and most
        // drums are marked.
        let found = marks(&desktop);
        assert!(found.len() >= 16, "{}: {found:?}", desktop.notice());
        for time in &found {
            let off = (time - 5_250_000).rem_euclid(500_000);
            assert!(off.min(500_000 - off) <= 40_000, "{time}: {found:?}");
        }
        assert_eq!(desktop.notice(), format!("Marked {} beats", found.len()));
        desktop.undo().unwrap();
        assert!(marks(&desktop).is_empty());
    }

    /// The real-media check: `CUTLASS_BEATS_SONG=<file> cargo test -p
    /// cutlass a_song_of_your_own -- --ignored --nocapture` prints the
    /// beats found in the file's first minute.
    #[test]
    #[ignore = "needs CUTLASS_BEATS_SONG and the fetched model"]
    fn a_song_of_your_own_has_its_beats_marked() {
        let song = PathBuf::from(std::env::var("CUTLASS_BEATS_SONG").unwrap());
        let model = fetched_model().unwrap();
        let temp = Temp::new();
        std::fs::create_dir_all(&temp.path).unwrap();
        let mut draft = Draft::empty("draft");
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: song.to_string_lossy().into_owned(),
            duration: Micros(60_000_000),
            ..MaterialAudio::default()
        });
        let mut segment = Segment::new("song", "song", TimeRange::new(0, 60_000_000));
        segment.source_range = Some(TimeRange::new(0, 60_000_000));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound.segments.push(segment);
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("song")));
        let began = std::time::Instant::now();
        desktop.auto_beats_with(model).unwrap();
        land_job(&mut desktop);
        let found = marks(&desktop);
        println!("{} in {:?}", desktop.notice(), began.elapsed());
        println!("{found:?}");
        assert!(!found.is_empty());
    }

    #[test]
    fn a_cancelled_job_marks_nothing_and_an_unreadable_file_says_so() {
        let Some(model) = fetched_model() else {
            return;
        };
        let temp = Temp::new();
        let mut desktop = opened(&temp);
        desktop.select(Some(SegmentId::new("song")));
        desktop.auto_beats_with(model.clone()).unwrap();
        desktop.cancel_job();
        land_job(&mut desktop);
        assert!(marks(&desktop).is_empty());

        std::fs::write(temp.path.join("song.wav"), b"not sound").unwrap();
        desktop.auto_beats_with(model).unwrap();
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "the clip's sound could not be read");
        assert!(marks(&desktop).is_empty());
    }
}
