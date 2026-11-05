//! Play, pause, seek, and the tick that moves the playhead with the sound.
//!
//! The audio goes through the shared [`Transport`], the one the desktop
//! plays through. A phone opens its device output once; tests keep the
//! software buffer.

use cutlass_playback::{AudioGap, AudioHost, PlayheadError, Program, Transport};
use cutlass_session::Micros;

use crate::FfiError;

/// The phone playhead and its audio output.
#[derive(Default)]
pub struct PhonePlayer {
    transport: Transport,
    playhead: Micros,
    playing: bool,
}

impl PhonePlayer {
    /// Software audio, playhead at zero, not playing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The playhead, in microseconds.
    #[must_use]
    pub fn playhead(&self) -> Micros {
        self.playhead
    }

    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Silence named by the host, when this output has no device.
    #[must_use]
    pub fn gap(&self) -> Option<AudioGap> {
        self.transport.gap()
    }

    /// Choose the output. Linux is silence, not an error.
    pub fn open_output(&mut self, host: AudioHost) {
        self.playing = false;
        self.transport.open(host);
    }

    /// Play from the playhead, or from the start when it is at `end`. An
    /// empty cut does not play. A negative `end` is refused.
    pub fn play(&mut self, program: Program<'_>, end: Micros) -> Result<(), FfiError> {
        if end.0 < 0 {
            return Err(PlayheadError::Duration.into());
        }
        if end.0 == 0 {
            self.playing = false;
            return Ok(());
        }
        if self.playhead >= end {
            self.playhead = Micros(0);
        }
        self.playing = true;
        self.transport.start(program, self.playhead, end);
        Ok(())
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.transport.silence();
    }

    /// Move the playhead to `at`, held inside the cut. Playing goes on from
    /// there.
    pub fn seek(&mut self, program: Program<'_>, at: Micros, end: Micros) {
        self.playhead = Micros(at.0.clamp(0, end.0.max(0)));
        if self.playing {
            self.transport.start(program, self.playhead, end);
        }
    }

    /// Move a playing playhead by `elapsed` microseconds, toward the heard
    /// audio. Play stops at `end`. A paused playhead stays where it is.
    pub fn tick(&mut self, program: Program<'_>, elapsed: i64, end: Micros) -> Micros {
        if !self.playing {
            return self.playhead;
        }
        let ticked = self
            .transport
            .tick(program, self.playhead, elapsed, end, false);
        self.playhead = ticked.playhead;
        self.playing = ticked.playing;
        self.playhead
    }
}

#[cfg(test)]
mod tests {
    use cutlass_eval::ReaderFactory;
    use cutlass_playback::{AudioGap, AudioHost, PlayheadError, Program};
    use cutlass_session::{Draft, Micros};

    use super::PhonePlayer;
    use crate::FfiError;

    fn program<'a>(draft: &'a Draft, readers: &'a ReaderFactory) -> Program<'a> {
        Program {
            draft,
            revision: 0,
            root: None,
            readers,
        }
    }

    #[test]
    fn a_new_player_is_at_zero_and_not_playing() {
        let player = PhonePlayer::new();
        assert_eq!(player.playhead(), Micros(0));
        assert!(!player.is_playing());
        assert_eq!(player.gap(), None);
    }

    #[test]
    fn play_ticks_to_the_end_and_stops_there() {
        let draft = Draft::empty("draft");
        let readers = ReaderFactory::default();
        let end = Micros(100_000);
        let mut player = PhonePlayer::new();
        player.play(program(&draft, &readers), end).unwrap();
        assert!(player.is_playing());
        assert_eq!(
            player.tick(program(&draft, &readers), 40_000, end),
            Micros(40_000)
        );
        assert_eq!(player.tick(program(&draft, &readers), 90_000, end), end);
        assert!(!player.is_playing());
        // Play again from the end starts over.
        player.play(program(&draft, &readers), end).unwrap();
        assert_eq!(player.playhead(), Micros(0));
    }

    #[test]
    fn pause_holds_the_playhead_and_seek_stays_inside_the_cut() {
        let draft = Draft::empty("draft");
        let readers = ReaderFactory::default();
        let end = Micros(100_000);
        let mut player = PhonePlayer::new();
        player.play(program(&draft, &readers), end).unwrap();
        player.tick(program(&draft, &readers), 30_000, end);
        player.pause();
        assert!(!player.is_playing());
        assert_eq!(
            player.tick(program(&draft, &readers), 30_000, end),
            Micros(30_000)
        );
        player.seek(program(&draft, &readers), Micros(500_000), end);
        assert_eq!(player.playhead(), end);
        player.seek(program(&draft, &readers), Micros(-5), end);
        assert_eq!(player.playhead(), Micros(0));
    }

    #[test]
    fn an_empty_cut_does_not_play_and_a_negative_one_is_refused() {
        let draft = Draft::empty("draft");
        let readers = ReaderFactory::default();
        let mut player = PhonePlayer::new();
        player.play(program(&draft, &readers), Micros(0)).unwrap();
        assert!(!player.is_playing());
        let error = player
            .play(program(&draft, &readers), Micros(-1))
            .unwrap_err();
        assert!(matches!(error, FfiError::Playhead(PlayheadError::Duration)));
        assert!(!player.is_playing());
    }

    #[test]
    fn a_host_with_no_device_names_a_gap_and_software_has_none() {
        let mut player = PhonePlayer::new();
        player.open_output(AudioHost::Unnamed);
        assert_eq!(player.gap(), Some(AudioGap::NoDevice));
        player.open_output(AudioHost::Software);
        assert_eq!(player.gap(), None);
    }
}
