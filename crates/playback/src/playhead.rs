//! The playhead clock.
//!
//! The editor sends a microsecond. Seeking asks eval for that time. Play
//! stops at the stored duration and does not rewrite it. The audio clock
//! advances on its own: a late picture does not move it backward.

use std::fmt;

use cutlass_gpu::Device;
use cutlass_model::Micros;
use cutlass_session::Session;

use crate::apple::{self, AudioOutput, CopyOutcome};
use crate::present::{self, Cache, PresentError, Presentation};

/// CPU pixels for one presented frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readback {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The playhead could not move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayheadError {
    /// The time is negative or past the stored duration.
    Time,
    /// The stored duration is negative. The draft was not rewritten.
    Duration,
    /// The audio rate is zero, so samples are not a clock.
    NoClock,
    /// The sample ended before one frame, or the queue buffer is shorter.
    Truncated,
    /// The sample is larger than the queue will copy.
    Hostile,
}

impl fmt::Display for PlayheadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Time => "the playhead is outside the stored duration",
            Self::Duration => "the stored duration is negative",
            Self::NoClock => "the audio rate is not a clock",
            Self::Truncated => "the audio sample ended early",
            Self::Hostile => "the audio sample is past the limit",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for PlayheadError {}

/// One step of play.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tick {
    /// The playhead moved and the picture was shown.
    Playing(Readback),
    /// The playhead is at the stored duration.
    Stopped,
    /// The picture missed the audio clock. The clock stayed where it was.
    Dropped { audio_time: Micros },
}

/// The preview session. It borrows the open session and the caller's device.
pub struct Player {
    time: Micros,
    playing: bool,
    audio_time: Micros,
    cache: Cache,
    shown: Option<Readback>,
    output: AudioOutput,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

impl Player {
    /// The clock starts at zero. Audio is the software buffer, so tests can
    /// run with no OS output device.
    #[must_use]
    pub fn new() -> Self {
        Self {
            time: Micros(0),
            playing: false,
            audio_time: Micros(0),
            cache: Cache::new(),
            shown: None,
            output: AudioOutput::software(),
        }
    }

    #[must_use]
    pub fn playhead(&self) -> Micros {
        self.time
    }

    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    #[must_use]
    pub fn audio_time(&self) -> Micros {
        self.audio_time
    }

    #[must_use]
    pub fn output(&self) -> &AudioOutput {
        &self.output
    }

    /// The last picture that was shown. A dropped frame leaves this in place.
    #[must_use]
    pub fn readback(&self) -> Option<&[u8]> {
        self.shown.as_ref().map(|frame| frame.rgba.as_slice())
    }

    /// Move the playhead and ask eval for that time.
    pub fn seek<D: Device>(
        &mut self,
        session: &Session,
        device: &mut D,
        time: Micros,
    ) -> Result<Readback, PresentError<D::Error>> {
        self.evaluate_at(session, device, time, true)
    }

    /// The same evaluation as play, at another time.
    pub fn scrub<D: Device>(
        &mut self,
        session: &Session,
        device: &mut D,
        time: Micros,
    ) -> Result<Readback, PresentError<D::Error>> {
        self.seek(session, device, time)
    }

    /// Present the current playhead. A picture later than the audio clock is
    /// dropped and the clock stays put.
    pub fn present<D: Device>(
        &mut self,
        session: &Session,
        device: &mut D,
    ) -> Result<Presentation, PresentError<D::Error>> {
        let time = self.time;
        present::check_time(session.draft().duration, time)?;
        let readback = present::evaluate(&mut self.cache, session, device, time)?;
        self.output.set_slice(&[]);
        let presentation = present::accept(self.audio_time, time, readback);
        if let Presentation::Shown(ref frame) = presentation {
            self.shown = Some(frame.clone());
        }
        Ok(presentation)
    }

    /// Start play when the playhead is inside the stored duration.
    pub fn play(&mut self, session: &Session) -> Result<(), PlayheadError> {
        let duration = session.draft().duration;
        if duration.0 < 0 {
            return Err(PlayheadError::Duration);
        }
        self.playing = self.time.0 < duration.0;
        Ok(())
    }

    /// Advance toward the stored duration. The draft's duration field stays.
    pub fn tick<D: Device>(
        &mut self,
        session: &Session,
        device: &mut D,
        step: Micros,
    ) -> Result<Tick, PresentError<D::Error>> {
        let duration = session.draft().duration;
        if duration.0 < 0 {
            return Err(PlayheadError::Duration.into());
        }
        if !self.playing {
            return Ok(Tick::Stopped);
        }
        if step.0 < 0 {
            return Err(PlayheadError::Time.into());
        }
        let next = self.time.0.saturating_add(step.0);
        if next >= duration.0 {
            self.time = duration;
            self.playing = false;
            return Ok(Tick::Stopped);
        }
        let frame = self.evaluate_at(session, device, Micros(next), false)?;
        match present::accept(self.audio_time, Micros(next), frame) {
            Presentation::Shown(readback) => {
                self.time = Micros(next);
                self.shown = Some(readback.clone());
                Ok(Tick::Playing(readback))
            }
            Presentation::Dropped { audio_time } => {
                self.time = Micros(next);
                Ok(Tick::Dropped { audio_time })
            }
        }
    }

    /// Move the audio clock forward. This does not ask eval and does not wait
    /// for a picture.
    pub fn advance_audio(&mut self, frames: u32, rate: u32) -> Result<Micros, PlayheadError> {
        if rate == 0 {
            return Err(PlayheadError::NoClock);
        }
        let step = i128::from(frames) * 1_000_000 / i128::from(rate);
        let step = i64::try_from(step).unwrap_or(i64::MAX);
        self.audio_time.0 = self.audio_time.0.saturating_add(step);
        Ok(self.audio_time)
    }

    /// Choose the output for this host. Linux stays silent. A missing device
    /// is named and does not panic.
    pub fn open_output(&mut self, host: apple::AudioHost) {
        self.output = AudioOutput::open(host);
    }

    /// Copy one pipeline sample into `buffer` and advance the audio clock by
    /// that sample. A late picture is not required and is not awaited.
    pub fn copy_pipeline_sample(
        &mut self,
        sample: &[u8],
        buffer: &mut [u8],
    ) -> Result<CopyOutcome, PlayheadError> {
        let outcome = self.output.copy_one(sample, buffer)?;
        if let CopyOutcome::Copied(_) = outcome {
            let _ = self.advance_audio(1, apple::RATE);
        }
        Ok(outcome)
    }

    fn evaluate_at<D: Device>(
        &mut self,
        session: &Session,
        device: &mut D,
        time: Micros,
        commit_time: bool,
    ) -> Result<Readback, PresentError<D::Error>> {
        present::check_time(session.draft().duration, time)?;
        let readback = present::evaluate(&mut self.cache, session, device, time)?;
        self.output.set_slice(&[]);
        if commit_time {
            self.time = time;
            self.shown = Some(readback.clone());
        }
        Ok(readback)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_eval::EvalError;
    use cutlass_model::{Draft, Micros};
    use cutlass_renderer_software::SoftwareDevice;
    use cutlass_session::Session;

    use super::*;
    use crate::color_session;
    use crate::present::{PresentError, Presentation};

    const COLOR: [u8; 4] = [0x33, 0x66, 0x99, 0xFF];

    #[test]
    fn seeking_asks_eval_for_that_time() {
        let session = color_session(1_000_000, 1_000_000);
        let before = session.draft().clone();
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        let frame = player
            .seek(&session, &mut device, Micros(10))
            .expect("seek");
        assert_eq!(player.playhead(), Micros(10));
        assert_eq!(frame.rgba, COLOR.repeat(2));
        assert_eq!(player.readback(), Some(frame.rgba.as_slice()));
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_negative_time_or_duration_is_an_error() {
        let session = color_session(1_000_000, 1_000_000);
        let before = session.draft().clone();
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        let error = player
            .seek(&session, &mut device, Micros(-1))
            .expect_err("negative");
        assert_eq!(error, PresentError::Playhead(PlayheadError::Time));
        assert_eq!(player.playhead(), Micros(0));
        assert_eq!(session.draft(), &before);

        let past = color_session(1_000, 1_000);
        let before_past = past.draft().clone();
        let error = player
            .seek(&past, &mut device, Micros(1_001))
            .expect_err("past");
        assert_eq!(error, PresentError::Playhead(PlayheadError::Time));
        assert_eq!(player.playhead(), Micros(0));
        assert_eq!(past.draft(), &before_past);

        let negative = color_session(-5, 1_000);
        let before_negative = negative.draft().clone();
        let error = player
            .seek(&negative, &mut device, Micros(0))
            .expect_err("duration");
        assert_eq!(error, PresentError::Playhead(PlayheadError::Duration));
        assert_eq!(negative.draft().duration, Micros(-5));
        assert_eq!(negative.draft(), &before_negative);
    }

    #[test]
    fn a_hostile_canvas_returns_eval_and_does_not_panic() {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = -1;
        draft.duration = Micros(1_000);
        let session = Session::open(draft);
        let before = session.draft().clone();
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        let error = player
            .seek(&session, &mut device, Micros(0))
            .expect_err("canvas");
        assert!(matches!(error, PresentError::Picture(EvalError::Canvas)));
        assert_eq!(session.draft(), &before);
        assert_eq!(player.playhead(), Micros(0));
        assert!(player.readback().is_none());
    }

    #[test]
    fn play_stops_at_the_stored_duration_without_rewriting_it() {
        let session = color_session(1_000_000, 2_000_000);
        let before = session.draft().clone();
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        player.play(&session).expect("play");
        let first = player
            .tick(&session, &mut device, Micros(600_000))
            .expect("tick");
        assert!(matches!(first, Tick::Playing(_)));
        assert_eq!(player.playhead(), Micros(600_000));
        let second = player
            .tick(&session, &mut device, Micros(600_000))
            .expect("stop");
        assert_eq!(second, Tick::Stopped);
        let third = player
            .tick(&session, &mut device, Micros(1))
            .expect("stays stopped");
        assert_eq!(third, Tick::Stopped);
        assert!(!player.is_playing());
        assert_eq!(player.playhead(), Micros(1_000_000));
        assert_eq!(session.draft().duration, Micros(1_000_000));
        assert_eq!(
            session.draft().tracks[0].segments[0].target.duration,
            Micros(2_000_000)
        );
        assert_eq!(session.draft(), &before);

        let longer = color_session(3_000_000, 1_000_000);
        let before_longer = longer.draft().clone();
        let mut player = Player::new();
        player.play(&longer).expect("play");
        player
            .tick(&longer, &mut device, Micros(2_000_000))
            .expect("past the segment");
        assert!(player.is_playing());
        assert_eq!(player.playhead(), Micros(2_000_000));
        assert_eq!(player.readback(), Some([0, 0, 0, 0, 0, 0, 0, 0].as_slice()));
        assert_eq!(longer.draft(), &before_longer);
    }

    #[test]
    fn a_negative_duration_refuses_play_and_leaves_the_draft() {
        let session = color_session(-1, 1_000);
        let before = session.draft().clone();
        let mut player = Player::new();
        let error = player.play(&session).expect_err("duration");
        assert_eq!(error, PlayheadError::Duration);
        assert!(!player.is_playing());
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_late_picture_is_dropped_and_the_audio_clock_stays() {
        let session = color_session(1_000_000, 100);
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        let shown = player.seek(&session, &mut device, Micros(0)).expect("seek");
        assert_eq!(shown.rgba, COLOR.repeat(2));
        let clock = player.advance_audio(48_000, 48_000).expect("clock");
        assert_eq!(clock, Micros(1_000_000));
        let presented = player.present(&session, &mut device).expect("late present");
        assert_eq!(presented, Presentation::Dropped { audio_time: clock });
        assert_eq!(player.audio_time(), clock);
        assert_eq!(player.readback(), Some(shown.rgba.as_slice()));
        player.play(&session).expect("play");
        let tick = player
            .tick(&session, &mut device, Micros(500))
            .expect("late");
        assert_eq!(tick, Tick::Dropped { audio_time: clock });
        assert_eq!(player.audio_time(), clock);
        assert_eq!(player.playhead(), Micros(500));
        assert_eq!(player.readback(), Some(shown.rgba.as_slice()));
        let again = player.advance_audio(0, 48_000).expect("still");
        assert_eq!(again, clock);
    }

    #[test]
    fn a_zero_audio_rate_is_an_error_and_does_not_panic() {
        let mut player = Player::new();
        let error = player.advance_audio(1, 0).expect_err("rate");
        assert_eq!(error, PlayheadError::NoClock);
        assert_eq!(player.audio_time(), Micros(0));
        let mut guard = 0;
        while player.audio_time().0 < i64::MAX && guard < 10_000 {
            player.advance_audio(u32::MAX, 1).expect("advance");
            guard += 1;
        }
        assert_eq!(player.audio_time(), Micros(i64::MAX));
        assert!(guard < 10_000);
    }

    #[test]
    fn a_scrubbed_frame_matches_a_played_frame() {
        let session = color_session(1_000_000, 1_000_000);
        let mut player = Player::new();
        let mut device = SoftwareDevice::new();
        let played = player
            .seek(&session, &mut device, Micros(10))
            .expect("play");
        let scrubbed = player
            .scrub(&session, &mut device, Micros(10))
            .expect("scrub");
        assert_eq!(played.rgba, scrubbed.rgba);
        assert_eq!(played.rgba, COLOR.repeat(2));

        let played_late = player
            .seek(&session, &mut device, Micros(1_000_000))
            .expect("end");
        let scrubbed_late = player
            .scrub(&session, &mut device, Micros(1_000_000))
            .expect("scrub end");
        assert_eq!(played_late.rgba, scrubbed_late.rgba);
        assert_eq!(played_late.rgba, [0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
