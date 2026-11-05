//! Speech for the assistant and `cutlass mcp`: hear a stretch of the
//! timeline as a job and answer with its words on the timeline clock, one
//! line each: start and end in microseconds, then the word.

use cutlass_model::TrackAttribute;
use cutlass_transcription::{FileSource, HeardClip};

use super::{Shell, ShellError, heard_words, load_speech, speech_model};

impl Shell {
    /// Start hearing `[start_us, start_us + duration_us)`: every clip with
    /// sound that plays there, on a track that is not muted. Returns the
    /// job's id for [`Self::hearing`].
    pub fn start_hearing(&mut self, start_us: i64, duration_us: i64) -> Result<String, ShellError> {
        if start_us < 0 || duration_us <= 0 {
            return Err(ShellError::Unavailable(
                "the stretch to hear is empty".to_owned(),
            ));
        }
        let end = start_us.saturating_add(duration_us);
        let draft = self.session_draft()?;
        let mut clips: Vec<HeardClip> = Vec::new();
        for track in &draft.tracks {
            if track.attribute.contains(TrackAttribute::MUTE) {
                continue;
            }
            for segment in &track.segments {
                let (from, to) = (
                    segment.target.start.0,
                    segment
                        .target
                        .start
                        .0
                        .saturating_add(segment.target.duration.0),
                );
                if !segment.visible || to <= start_us || from >= end {
                    continue;
                }
                if let Ok(clip) = cutlass_transcription::heard_clip(draft, &segment.id) {
                    clips.push(clip);
                }
            }
        }
        if clips.is_empty() {
            return Err(ShellError::Unavailable(
                "nothing with sound plays in that stretch".to_owned(),
            ));
        }
        self.refuse_busy()?;
        let loaded = self.speech.borrow_mut().take();
        if loaded.is_none() && !speech_model::speech_model_here() {
            self.wants_speech_model = true;
            return Err(ShellError::Unavailable(
                "the speech model is not on this machine".to_owned(),
            ));
        }
        self.hearings = self.hearings.saturating_add(1);
        let id = format!("speech-{}", self.hearings);
        let landed = id.clone();
        self.start_job("Transcribe", move |progress| {
            let recognizer = match loaded {
                Some(recognizer) => Ok(recognizer),
                None => {
                    progress.stage("Loading the speech model");
                    load_speech()
                }
            };
            let mut lines: Vec<(i64, i64, String)> = Vec::new();
            let mut failed = None;
            if let Ok(recognizer) = &recognizer {
                progress.stage("Listening");
                for clip in &clips {
                    match heard_words(recognizer, &mut FileSource::new(&clip.path), clip) {
                        Ok(words) => {
                            let base = clip.target.start.0;
                            for ((start, end_ms), text) in
                                words.start_time.iter().zip(&words.end_time).zip(words.text)
                            {
                                let from = base.saturating_add(start.saturating_mul(1000));
                                let to = base.saturating_add(end_ms.saturating_mul(1000));
                                if to > start_us && from < end {
                                    lines.push((from, to, text));
                                }
                            }
                        }
                        Err(error) => failed = Some(error.to_string()),
                    }
                }
            }
            lines.sort();
            let heard = match &recognizer {
                Err(error) => Err(error.to_string()),
                Ok(_) if lines.is_empty() => {
                    Err(failed.unwrap_or_else(|| "nothing was heard".to_owned()))
                }
                Ok(_) => Ok(lines
                    .iter()
                    .map(|(from, to, text)| format!("{from} {to} {text}"))
                    .collect::<Vec<_>>()
                    .join("\n")),
            };
            let kept = recognizer.ok();
            Box::new(move |shell: &mut Shell| {
                if kept.is_some() {
                    shell.speech.replace(kept);
                }
                shell.heard.insert(landed, heard);
                Ok(())
            })
        })?;
        Ok(id)
    }

    /// What job `id` heard: `None` while it is still listening.
    pub fn hearing(&mut self, id: &str) -> Result<Option<String>, ShellError> {
        self.poll_job();
        match self.heard.get(id) {
            Some(Ok(words)) => Ok(Some(words.clone())),
            Some(Err(error)) => Err(ShellError::Unavailable(error.clone())),
            None if self.job().is_some_and(|job| job.label == "Transcribe") => Ok(None),
            None => Err(ShellError::Unavailable(format!("no transcription {id}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cutlass_agent::{ApprovalAction, ToolCall};
    use cutlass_model::{Draft, MaterialAudio, Micros, Segment, TimeRange};
    use cutlass_store::Folder;

    use crate::shell::tests::{Temp, shell};

    #[test]
    fn hearing_needs_a_stretch_with_sound_and_names_its_job() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.materials.audios.push(MaterialAudio {
            id: "song".into(),
            path: temp.path.join("missing.wav").display().to_string(),
            ..MaterialAudio::default()
        });
        let mut track = cutlass_model::Track::main("a");
        track.kind = cutlass_model::TrackType::Audio;
        track
            .segments
            .push(Segment::new("sound", "song", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(track);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.start_hearing(0, 0).is_err(), "an empty stretch");
        let error = desktop.start_hearing(2_000_000, 500_000).unwrap_err();
        assert!(error.to_string().contains("nothing with sound"), "{error}");
        assert!(desktop.hearing("speech-9").is_err(), "an unknown job");
        // With sound there, it runs where the model is and says so where not.
        match desktop.start_hearing(0, 500_000) {
            Ok(id) => {
                assert_eq!(id, "speech-1");
                crate::shell::tests::land_job(&mut desktop);
                // The file is missing, so the job answers with why.
                assert!(desktop.hearing(&id).is_err());
            }
            Err(error) => {
                assert!(error.to_string().contains("speech model"), "{error}");
                // The window offers the download, until the user says no.
                assert!(desktop.wants_speech_model());
                desktop.decline_speech_model();
                assert!(!desktop.wants_speech_model());
            }
        }
        assert!(!desktop.wants_speech_model() || !super::speech_model::speech_model_here());
    }

    /// The assistant hears four seconds of a talk placed one second into the
    /// timeline, with the real model, and polls until the words land.
    #[test]
    fn the_assistant_hears_a_stretch_with_the_real_model() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file()
            || !super::speech_model::speech_model_dir()
                .join(super::speech_model::SPEECH.file)
                .is_file()
        {
            cutlass_transcription::weights_missing("the sound or the speech model");
            eprintln!("skipped: the sound or the speech model is missing");
            return;
        }
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.materials.audios.push(MaterialAudio {
            id: "talk".into(),
            path: sound.display().to_string(),
            duration: Micros(6_000_000),
            ..MaterialAudio::default()
        });
        let mut track = cutlass_model::Track::main("sound");
        track.kind = cutlass_model::TrackType::Audio;
        let mut segment = Segment::new("talk", "talk", TimeRange::new(1_000_000, 6_000_000));
        segment.source_range = Some(TimeRange::new(0, 6_000_000));
        track.segments.push(segment);
        draft.tracks.push(track);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let call = |name: &str, arguments: &str| ToolCall {
            id: "1".to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        };
        let allowed: &[ApprovalAction] = &[];
        let started = desktop.run_agent_tool(&call("speech.start", "2000000 4000000"), allowed);
        assert!(!started.outcome.is_error, "{}", started.outcome.content);
        assert!(
            started.outcome.content.starts_with("job=speech-1"),
            "{}",
            started.outcome.content
        );
        let until = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let words = loop {
            let polled = desktop.run_agent_tool(&call("speech.poll", "speech-1"), allowed);
            assert!(!polled.outcome.is_error, "{}", polled.outcome.content);
            if polled.outcome.content != "pending" {
                break polled.outcome.content;
            }
            assert!(
                std::time::Instant::now() < until,
                "the hearing took too long"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        let words = words.strip_prefix("words=").expect("words");
        let lines: Vec<&str> = words.lines().collect();
        assert!(lines.len() >= 3, "{words}");
        for line in lines {
            let mut parts = line.splitn(3, ' ');
            let from: i64 = parts.next().unwrap().parse().unwrap();
            let to: i64 = parts.next().unwrap().parse().unwrap();
            assert!(!parts.next().unwrap().is_empty());
            assert!(
                to > 2_000_000 && from < 6_000_000,
                "{line} is outside the stretch"
            );
            assert!(from >= 1_000_000, "{line} is before the clip");
        }
    }
}
