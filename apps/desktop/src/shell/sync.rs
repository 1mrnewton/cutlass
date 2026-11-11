//! Sync video and audio: line the picked clips up by their sound.
//!
//! One picture clip stays where it is; every other picked clip with sound
//! moves so the same sound plays at the same time in both. The files are
//! read as a job, and the clips move in one undo step when it lands.

use cutlass_audio::{Envelope, align};
use cutlass_model::{SegmentId, TimeRange, TrackId};
use cutlass_transcription::{FileSource, HeardClip, PcmSource};

use super::{Shell, ShellError};
use crate::jobs::Progress;

/// The sound read at a time: half a minute, so a long file is never held
/// whole.
const PIECE: i64 = 30_000_000;

/// How a clip's sound rises and falls over the stretch of its file it
/// plays. `None` when none of it could be read, or the job was cancelled.
fn clip_envelope(
    source: &mut dyn PcmSource,
    clip: &HeardClip,
    progress: &Progress,
) -> Option<Vec<f32>> {
    let end = clip.range.start.0.saturating_add(clip.range.duration.0);
    let mut envelope: Option<Envelope> = None;
    // Sound that could not be read before any could: its length is owed
    // as silence once the rate is known.
    let mut owed = 0_i64;
    let mut at = clip.range.start.0;
    while at < end {
        if progress.cancelled() {
            return None;
        }
        let length = PIECE.min(end - at);
        match source.read(&TimeRange::new(at, length)) {
            Ok(pcm) => {
                let rate = pcm.sample_rate;
                if envelope.is_none() {
                    let mut made = Envelope::new(rate)?;
                    made.skip(samples_in(owed, rate));
                    envelope = Some(made);
                }
                if let Some(envelope) = envelope.as_mut() {
                    envelope.push(&pcm.samples);
                    // A piece that came back short still took its time.
                    let wanted = samples_in(length, rate);
                    envelope.skip(wanted.saturating_sub(pcm.samples.len() as u64));
                }
            }
            Err(_) => match envelope.as_mut() {
                Some(envelope) => envelope.skip(samples_in(length, envelope_rate(envelope))),
                None => owed = owed.saturating_add(length),
            },
        }
        at = at.saturating_add(length);
    }
    envelope.map(Envelope::finish)
}

/// The samples `micros` of sound take at `rate`.
fn samples_in(micros: i64, rate: u32) -> u64 {
    let micros = u64::try_from(micros).unwrap_or(0);
    micros.saturating_mul(u64::from(rate)) / 1_000_000
}

fn envelope_rate(envelope: &Envelope) -> u32 {
    envelope.rate()
}

impl Shell {
    /// Whether the picked clips can be synced: a video clip and at least
    /// one more clip with sound, each playing forwards at its own speed.
    #[must_use]
    pub fn sync_enabled(&self) -> bool {
        self.draft()
            .is_some_and(|draft| cutlass_editor::sync_plan(draft, self.selection()).is_ok())
    }

    /// Line the picked clips up by their sound, as a job. The video clip on
    /// the main track, else the earliest video clip, stays; the others move
    /// to where their sound matches its sound. A clip whose sound matches
    /// nowhere stays where it is and is named when the job lands.
    pub fn start_sync(&mut self) -> Result<(), ShellError> {
        let draft = self.session_draft()?;
        let plan = cutlass_editor::sync_plan(draft, self.selection())
            .map_err(|miss| ShellError::Unavailable(miss.to_string()))?;
        for id in &plan.others {
            self.refuse_locked(id)?;
        }
        let draft = self.session_draft()?;
        let heard = |id: &SegmentId| -> Result<(HeardClip, std::path::PathBuf), ShellError> {
            let clip = cutlass_transcription::heard_clip(draft, id)
                .map_err(|_| ShellError::Unavailable("the clip has no sound".to_owned()))?;
            let file = self.sound_file(&clip.path)?;
            Ok((clip, file))
        };
        let reference = heard(&plan.reference)?;
        let others: Vec<(HeardClip, std::path::PathBuf)> =
            plan.others.iter().filter_map(|id| heard(id).ok()).collect();
        if others.is_empty() {
            return Err(ShellError::Unavailable(
                cutlass_editor::SyncMiss::TooFew.to_string(),
            ));
        }
        let anchor = plan.reference.clone();
        self.start_job("Sync", move |progress| {
            progress.stage("Listening to the video");
            let total = others.len() as f32 + 1.0;
            let lead = clip_envelope(&mut FileSource::new(&reference.1), &reference.0, progress);
            let mut shifts: Vec<(SegmentId, i64)> = Vec::new();
            let mut unmatched = 0_usize;
            if let Some(lead) = &lead {
                for (index, (clip, file)) in others.iter().enumerate() {
                    progress.fraction((index as f32 + 1.0) / total);
                    progress.stage("Matching the sound");
                    let found = clip_envelope(&mut FileSource::new(file), clip, progress)
                        .and_then(|envelope| align(lead, &envelope));
                    match found {
                        Some(found) => shifts.push((clip.source_id.clone(), found.micros())),
                        None => unmatched += 1,
                    }
                }
            }
            let cancelled = progress.cancelled();
            let silent = lead.is_none();
            Box::new(move |shell: &mut Shell| {
                if cancelled {
                    return Ok(());
                }
                if silent {
                    return Err(ShellError::Unavailable(
                        "the video's sound could not be read".to_owned(),
                    ));
                }
                shell.land_sync(&anchor, &shifts, unmatched)
            })
        })
    }

    /// Move the clips to the shifts a sync found, as one undo step, and say
    /// what happened.
    fn land_sync(
        &mut self,
        reference: &SegmentId,
        shifts: &[(SegmentId, i64)],
        unmatched: usize,
    ) -> Result<(), ShellError> {
        if shifts.is_empty() {
            return Err(ShellError::Unavailable(
                "the clips' sound does not match the video's".to_owned(),
            ));
        }
        // The lane names are taken before the draft is borrowed; one that
        // is not needed is simply not used.
        let mut lanes: Vec<TrackId> = (0..shifts.len())
            .map(|_| TrackId::new(self.fresh("track")))
            .collect();
        let draft = self.session_draft()?;
        let before: Vec<Option<i64>> = shifts
            .iter()
            .map(|(id, _)| {
                draft
                    .tracks
                    .iter()
                    .flat_map(|track| &track.segments)
                    .find(|segment| segment.id == *id)
                    .map(|segment| segment.target.start.0)
            })
            .collect();
        let (requests, left) = cutlass_editor::sync_requests(draft, reference, shifts, || {
            lanes.pop().unwrap_or_else(|| TrackId::new("track-sync"))
        });
        let moved = requests
            .iter()
            .filter(|request| matches!(request, cutlass_session::Request::Move { .. }))
            .count();
        if !requests.is_empty() {
            self.apply_requests(&requests)?;
        }
        let stayed = unmatched + left.len();
        let in_place = before.len().saturating_sub(moved + left.len());
        let mut said = match (moved, in_place) {
            (0, 0) => String::new(),
            (0, _) => "The clips were already in sync".to_owned(),
            (1, _) => "Synced 1 clip to the video".to_owned(),
            (moved, _) => format!("Synced {moved} clips to the video"),
        };
        if stayed > 0 {
            if !said.is_empty() {
                said.push_str("; ");
            }
            said.push_str(&match stayed {
                1 => "1 clip did not match and stayed".to_owned(),
                stayed => format!("{stayed} clips did not match and stayed"),
            });
        }
        if moved == 0 && in_place == 0 {
            return Err(ShellError::Unavailable(
                "the clips' sound does not match the video's".to_owned(),
            ));
        }
        self.set_notice(said);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cutlass_model::{
        Draft, MaterialAudio, MaterialId, MaterialVideo, Micros, Segment, SegmentId, TimeRange,
        Track, TrackType,
    };
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, land_job, shell};

    const RATE: u32 = 8_000;

    /// A repeatable noise source.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// `seconds` of a room: bursts of sound at times `seed` places.
    fn room(seed: u64, seconds: u32) -> Vec<f32> {
        let mut noise = Noise(seed);
        let mut events = Noise(seed.wrapping_mul(31).wrapping_add(7));
        let total = (seconds * RATE) as usize;
        let mut sound = vec![0.0_f32; total];
        let mut at = 0_usize;
        while at < total {
            let length = ((events.next().abs() * 0.4 + 0.1) * RATE as f32) as usize;
            let loud = events.next().abs() * 0.6 + 0.2;
            for sample in sound.iter_mut().skip(at).take(length) {
                *sample = noise.next() * loud;
            }
            at += length + ((events.next().abs() * 0.6 + 0.05) * RATE as f32) as usize;
        }
        sound
    }

    fn write_wav(path: &Path, samples: &[f32]) {
        let data = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&RATE.to_le_bytes());
        bytes.extend_from_slice(&(RATE * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 30_000.0) as i16).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// A camera clip on the main track from one second, and on a sound
    /// lane a recorder's take of the same room, started 6.2 seconds into
    /// the camera's, and a take of another room.
    fn project(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let camera = room(11, 24);
        write_wav(&dir.join("camera.wav"), &camera);
        let from = (6.2 * RATE as f32) as usize;
        let take: Vec<f32> = camera[from..from + 8 * RATE as usize]
            .iter()
            .map(|sample| sample * 0.4)
            .collect();
        write_wav(&dir.join("take.wav"), &take);
        write_wav(&dir.join("elsewhere.wav"), &room(5, 8));

        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.videos.push(MaterialVideo {
            id: MaterialId::new("camera"),
            path: "camera.wav".to_owned(),
            duration: Micros(24_000_000),
            ..MaterialVideo::default()
        });
        for id in ["take", "elsewhere"] {
            draft.materials.audios.push(MaterialAudio {
                id: MaterialId::new(id),
                path: format!("{id}.wav"),
                duration: Micros(8_000_000),
                ..MaterialAudio::default()
            });
        }
        let whole = |id: &str, material: &str, start: i64, length: i64| {
            let mut segment = Segment::new(id, material, TimeRange::new(start, length));
            segment.source_range = Some(TimeRange::new(0, length));
            segment
        };
        draft.tracks[0]
            .segments
            .push(whole("video", "camera", 1_000_000, 24_000_000));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound.segments.push(whole("take", "take", 0, 8_000_000));
        sound
            .segments
            .push(whole("elsewhere", "elsewhere", 30_000_000, 8_000_000));
        draft.tracks.push(sound);
        Folder::new(draft).save(dir).unwrap();
    }

    fn start_of(desktop: &crate::shell::Shell, id: &str) -> i64 {
        desktop
            .draft()
            .unwrap()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap()
            .target
            .start
            .0
    }

    #[test]
    fn a_take_moves_to_where_its_sound_is_in_the_video_and_a_stranger_stays() {
        let temp = Temp::new();
        project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();

        // One clip, or two with no video among them, cannot be synced.
        desktop.select(Some(SegmentId::new("video")));
        assert!(!desktop.sync_enabled());
        assert_eq!(
            desktop.start_sync().unwrap_err().to_string(),
            "pick a video clip and the sound that goes with it"
        );
        desktop.select(Some(SegmentId::new("take")));
        desktop.toggle_selected(SegmentId::new("elsewhere"));
        assert!(!desktop.sync_enabled());
        assert!(desktop.start_sync().is_err());

        desktop.toggle_selected(SegmentId::new("video"));
        assert!(desktop.sync_enabled());
        desktop.start_sync().unwrap();
        assert_eq!(desktop.job().map(|job| job.label), Some("Sync".to_owned()));
        land_job(&mut desktop);
        // The video starts at one second and the take belongs 6.2 seconds
        // into it. The step is five milliseconds.
        let landed = start_of(&desktop, "take");
        assert!((landed - 7_200_000).abs() <= 10_000, "{landed}");
        assert_eq!(start_of(&desktop, "video"), 1_000_000);
        assert_eq!(start_of(&desktop, "elsewhere"), 30_000_000);
        assert_eq!(
            desktop.notice(),
            "Synced 1 clip to the video; 1 clip did not match and stayed"
        );

        // Again, it is already in place; one undo puts the take back.
        desktop.start_sync().unwrap();
        land_job(&mut desktop);
        assert_eq!(start_of(&desktop, "take"), landed);
        assert_eq!(
            desktop.notice(),
            "The clips were already in sync; 1 clip did not match and stayed"
        );
        desktop.undo().unwrap();
        assert_eq!(start_of(&desktop, "take"), 0);
    }

    #[test]
    fn sound_that_matches_nowhere_or_cannot_be_read_moves_nothing() {
        let temp = Temp::new();
        project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let before = desktop.draft().unwrap().clone();
        desktop.select(Some(SegmentId::new("video")));
        desktop.toggle_selected(SegmentId::new("elsewhere"));
        desktop.start_sync().unwrap();
        land_job(&mut desktop);
        assert_eq!(
            desktop.notice(),
            "the clips' sound does not match the video's"
        );
        assert_eq!(desktop.draft().unwrap(), &before);

        // The video's file is gone: nothing can be matched to it.
        std::fs::remove_file(temp.path.join("camera.wav")).unwrap();
        desktop.toggle_selected(SegmentId::new("take"));
        desktop.start_sync().unwrap();
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "the video's sound could not be read");
        assert_eq!(desktop.draft().unwrap(), &before);

        // A cancelled sync moves nothing and says nothing new.
        std::fs::write(temp.path.join("camera.wav"), b"not sound").unwrap();
        desktop.start_sync().unwrap();
        desktop.cancel_job();
        land_job(&mut desktop);
        assert_eq!(desktop.draft().unwrap(), &before);
    }
}
