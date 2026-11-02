//! Playhead play and audio output.
//!
//! Seeking a picture needs a device this crate cannot construct. The player has no pause.

use cutlass_playback::{AudioGap, AudioHost, CopyOutcome, Player};
use cutlass_session::{Micros, Session};

use crate::FfiError;

/// The phone playhead and its audio output.
#[derive(Default)]
pub struct PhonePlayer {
    player: Player,
}

impl PhonePlayer {
    /// Software audio, playhead at zero, not playing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            player: Player::new(),
        }
    }

    /// The playhead, in microseconds.
    #[must_use]
    pub fn playhead(&self) -> Micros {
        self.player.playhead()
    }

    /// Play is running inside the stored duration.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.player.is_playing()
    }

    /// The audio clock, in microseconds.
    #[must_use]
    pub fn audio_time(&self) -> Micros {
        self.player.audio_time()
    }

    /// The last picture that was shown.
    #[must_use]
    pub fn readback(&self) -> Option<&[u8]> {
        self.player.readback()
    }

    /// Silence named by the host, when this output has no device.
    #[must_use]
    pub fn gap(&self) -> Option<AudioGap> {
        self.player.output().gap()
    }

    /// Start play when the playhead is inside the stored duration.
    ///
    /// A negative duration is refused. The draft is left as it is.
    pub fn play(&mut self, session: &Session) -> Result<(), FfiError> {
        self.player.play(session)?;
        Ok(())
    }

    /// Choose the output. Linux is silence, not an error.
    pub fn open_output(&mut self, host: AudioHost) {
        self.player.open_output(host);
    }

    /// Move the audio clock by `frames` at `rate`. A zero rate is refused.
    pub fn advance_audio(&mut self, frames: u32, rate: u32) -> Result<Micros, FfiError> {
        Ok(self.player.advance_audio(frames, rate)?)
    }

    /// Copy one stereo frame into `buffer` and advance the audio clock.
    pub fn copy_sample(
        &mut self,
        sample: &[u8],
        buffer: &mut [u8],
    ) -> Result<CopyOutcome, FfiError> {
        Ok(self.player.copy_pipeline_sample(sample, buffer)?)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_playback::{AudioGap, AudioHost, CopyOutcome, PlayheadError};

    use super::PhonePlayer;
    use crate::FfiError;

    fn session(duration: i64) -> cutlass_session::Session {
        let mut draft = cutlass_session::Draft::empty("draft");
        draft.duration = cutlass_session::Micros(duration);
        cutlass_session::Session::open(draft)
    }

    #[test]
    fn a_new_player_is_at_zero_and_not_playing() {
        let player = PhonePlayer::new();
        assert_eq!(player.playhead(), cutlass_session::Micros(0));
        assert!(!player.is_playing());
        assert_eq!(player.audio_time(), cutlass_session::Micros(0));
        assert!(player.readback().is_none());
        assert_eq!(player.gap(), None);
    }

    #[test]
    fn play_starts_without_rewriting_the_duration() {
        let session = session(1_000_000);
        let mut player = PhonePlayer::new();
        player.play(&session).unwrap();
        assert!(player.is_playing());
        assert_eq!(player.playhead(), cutlass_session::Micros(0));
        assert_eq!(session.draft().duration, cutlass_session::Micros(1_000_000));
    }

    #[test]
    fn a_negative_duration_refuses_play() {
        let session = session(-1);
        let mut player = PhonePlayer::new();
        let error = player.play(&session).unwrap_err();
        assert!(matches!(error, FfiError::Playhead(PlayheadError::Duration)));
        assert!(!player.is_playing());
        assert_eq!(session.draft().duration, cutlass_session::Micros(-1));
    }

    #[test]
    fn a_zero_rate_does_not_move_audio_time() {
        let mut player = PhonePlayer::new();
        let error = player.advance_audio(1, 0).unwrap_err();
        assert!(matches!(error, FfiError::Playhead(PlayheadError::NoClock)));
        assert_eq!(player.audio_time(), cutlass_session::Micros(0));
    }

    #[test]
    fn linux_names_a_gap_and_software_has_none() {
        let mut player = PhonePlayer::new();
        player.open_output(AudioHost::Linux);
        assert_eq!(player.gap(), Some(AudioGap::Linux));
        player.open_output(AudioHost::Software);
        assert_eq!(player.gap(), None);
    }

    #[test]
    fn a_stereo_frame_is_copied_and_the_clock_advances() {
        let mut player = PhonePlayer::new();
        let sample = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut buffer = [0_u8; 32];
        let outcome = player.copy_sample(&sample, &mut buffer).unwrap();
        assert_eq!(outcome, CopyOutcome::Copied(8));
        assert_eq!(&buffer[..8], &sample[..8]);
        assert_eq!(
            player.audio_time(),
            cutlass_session::Micros(1_000_000 / 48_000)
        );
    }

    #[test]
    fn an_empty_sample_copies_eight_zero_bytes() {
        let mut player = PhonePlayer::new();
        let mut buffer = [9_u8; 32];
        let outcome = player.copy_sample(&[], &mut buffer).unwrap();
        assert_eq!(outcome, CopyOutcome::Copied(8));
        assert_eq!(&buffer[..8], &[0_u8; 8]);
        assert_eq!(
            player.audio_time(),
            cutlass_session::Micros(1_000_000 / 48_000)
        );
    }
}
