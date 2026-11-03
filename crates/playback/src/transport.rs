//! Play with sound: the playhead follows the heard audio.
//!
//! A host with a device stream mixes on the [`Mixer`] thread, and the
//! playhead leans toward the frames the device has played, so the picture
//! stays with the sound. A host with no stream clock queues one mixed slice
//! per tick and the playhead follows the wall clock. Every shell plays
//! through this, so the desktop and the phones hear the same thing.

use std::path::Path;

use cutlass_eval::{Heard, ReaderFactory};
use cutlass_model::{Draft, Micros};

use crate::apple::{AudioGap, AudioHost, AudioOutput, RATE};
use crate::mixer::{MixJob, Mixer};
use crate::preview::{PLAY_FRAMES, PreviewMix, preview_frames, queue_mix};

/// How far the picture may drift from the heard audio before it jumps to it.
const AUDIO_SNAP: i64 = 60_000;
/// The share of the drift corrected each tick. Small enough that the device's
/// buffer-sized clock steps do not reach the picture.
const AUDIO_PULL: i64 = 16;
/// How long the playhead waits for the device to play the first frame before
/// it gives up on the audio clock and follows the wall.
const DEAF_GRACE: i64 = 300_000;

/// What is playing: the draft at one revision, the folder its media paths
/// resolve against, and the readers that open them.
#[derive(Clone, Copy)]
pub struct Program<'a> {
    pub draft: &'a Draft,
    pub revision: u64,
    pub root: Option<&'a Path>,
    pub readers: &'a ReaderFactory,
}

/// The playhead after one tick, and whether play is still running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticked {
    pub playhead: Micros,
    pub playing: bool,
}

/// The audio stream playback follows.
#[derive(Clone, Copy, Debug)]
struct Stream {
    /// Timeline time of the first streamed frame.
    start: Micros,
    /// The device's played count when this stream started.
    played: u64,
    /// The device has played this stream's first frame.
    heard: bool,
    /// Wall time spent waiting for the first frame.
    waited: i64,
    /// The device never played in time, so the wall moves the playhead.
    deaf: bool,
}

impl Stream {
    fn time_at(self, frames: u64) -> Micros {
        let micros = i128::from(frames) * 1_000_000 / i128::from(RATE);
        Micros(
            self.start
                .0
                .saturating_add(i64::try_from(micros).unwrap_or(i64::MAX)),
        )
    }
}

/// The audio side of play, pause, seek, and tick.
pub struct Transport {
    output: AudioOutput,
    heard: Heard,
    stream: Option<Stream>,
    mixer: Option<Mixer>,
}

impl Default for Transport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport {
    /// The software buffer: no device, nothing heard.
    #[must_use]
    pub fn new() -> Self {
        Self {
            output: AudioOutput::software(),
            heard: Heard::new(),
            stream: None,
            mixer: None,
        }
    }

    /// Replace the output with the one for `host`.
    pub fn open(&mut self, host: AudioHost) {
        self.stream = None;
        self.mixer = None;
        self.output = AudioOutput::open(host);
    }

    #[must_use]
    pub fn output(&self) -> &AudioOutput {
        &self.output
    }

    /// Silence the host names, or `None` when it has an output.
    #[must_use]
    pub fn gap(&self) -> Option<AudioGap> {
        self.output.gap()
    }

    /// Close the files opened for the scrub slice, as for another project.
    pub fn forget_files(&mut self) {
        self.heard.clear();
    }

    /// Start a new mixer with other readers on the next play.
    pub fn forget_mixer(&mut self) {
        self.stream = None;
        self.mixer = None;
    }

    /// Stop every sound: the stream and the scrub slice.
    pub fn silence(&mut self) {
        self.stream = None;
        if let Some(mixer) = &self.mixer {
            mixer.stop();
        }
        self.output.clear_stream();
        self.output.queue(&[]);
    }

    /// Stream `program` from `at` to `end` on the mixer thread. A host with
    /// no stream clock queues one slice instead, and then one per tick.
    pub fn start(&mut self, program: Program<'_>, at: Micros, end: Micros) {
        self.output.clear_stream();
        let Some(clock) = self.output.clock() else {
            self.stream = None;
            self.queue(program, at, PLAY_FRAMES);
            return;
        };
        if self.mixer.is_none() {
            self.mixer = Mixer::start(&self.output, program.readers.clone());
        }
        let Some(mixer) = &self.mixer else {
            self.stream = None;
            self.queue(program, at, PLAY_FRAMES);
            return;
        };
        mixer.play(MixJob {
            draft: program.draft.clone(),
            revision: program.revision,
            root: program.root.map(Path::to_path_buf),
            start: at,
            end,
        });
        self.stream = Some(Stream {
            start: at,
            played: clock.played,
            heard: false,
            waited: 0,
            deaf: false,
        });
    }

    /// Queue `frames` of the mix at `at`, for a scrub or a host with no
    /// stream.
    pub fn queue(&mut self, program: Program<'_>, at: Micros, frames: u32) {
        queue_mix(
            &mut self.output,
            PreviewMix {
                draft: program.draft,
                revision: program.revision,
                time: at,
                frames,
                readers: program.readers,
                heard: &mut self.heard,
                root: program.root,
            },
        );
    }

    /// Move a running playhead by `elapsed` microseconds.
    ///
    /// With a stream the playhead holds until the device plays the first
    /// frame, lands on the heard time, and then leans toward it. Past `end`
    /// play stops there, or wraps to the start when `looping`. An empty
    /// timeline stops at zero. A draft changed while playing restarts the
    /// stream at the playhead, so the edit is heard.
    pub fn tick(
        &mut self,
        program: Program<'_>,
        playhead: Micros,
        elapsed: i64,
        end: Micros,
        looping: bool,
    ) -> Ticked {
        if end.0 <= 0 {
            self.silence();
            return Ticked {
                playhead: Micros(0),
                playing: false,
            };
        }
        if elapsed <= 0 {
            return Ticked {
                playhead,
                playing: true,
            };
        }
        let wall = playhead.0.saturating_add(elapsed);
        let mut next = match self.streamed_playhead(wall, elapsed) {
            Some(streamed) => streamed.max(playhead.0),
            None => wall,
        };
        let mut playing = true;
        let mut wrapped = false;
        if next >= end.0 {
            if looping {
                next %= end.0;
                wrapped = true;
            } else {
                next = end.0;
                playing = false;
            }
        }
        let playhead = Micros(next);
        if !playing {
            self.silence();
        } else if wrapped && self.stream.is_some() {
            self.start(program, playhead, end);
        } else if self.stream.is_some() {
            if self.mix_revision() != Some(program.revision) {
                self.start(program, playhead, end);
            }
        } else {
            self.queue(program, playhead, preview_frames(elapsed));
        }
        Ticked { playhead, playing }
    }

    /// The stream is running on a device clock.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// The draft revision the mixer is playing, if any.
    #[must_use]
    pub fn mix_revision(&self) -> Option<u64> {
        self.mixer.as_ref().and_then(Mixer::revision)
    }

    /// Frames of the stream the device has played.
    #[must_use]
    pub fn heard_frames(&self) -> Option<u64> {
        let stream = self.stream?;
        let clock = self.output.clock()?;
        let mixed = self.mixer.as_ref()?.mixed();
        Some(clock.heard().saturating_sub(stream.played).min(mixed))
    }

    /// Where the stream's heard frames reach, or `None` without a stream.
    #[must_use]
    pub fn heard_time(&self) -> Option<Micros> {
        Some(self.stream?.time_at(self.heard_frames()?))
    }

    /// The playhead the heard audio gives, or `None` to follow the wall.
    fn streamed_playhead(&mut self, wall: i64, elapsed: i64) -> Option<i64> {
        let heard_frames = self.heard_frames()?;
        self.stream.as_mut()?.playhead(heard_frames, wall, elapsed)
    }
}

impl Stream {
    /// The playhead once the device has played `heard_frames`: held at the
    /// start until the first frame is heard, on it then, and leaning toward
    /// it after. `None` once the device has played nothing for
    /// [`DEAF_GRACE`], as when another app holds the audio, so the wall
    /// moves the playhead instead of a picture frozen at the start.
    fn playhead(&mut self, heard_frames: u64, wall: i64, elapsed: i64) -> Option<i64> {
        if self.deaf {
            return None;
        }
        let heard = self.time_at(heard_frames).0;
        if heard_frames == 0 {
            self.waited = self.waited.saturating_add(elapsed);
            if self.waited > DEAF_GRACE {
                self.deaf = true;
                return None;
            }
            return Some(self.start.0);
        }
        if !self.heard {
            self.heard = true;
            return Some(heard);
        }
        Some(follow_audio(wall, heard))
    }
}

/// The playhead after one tick: moved by the frame's elapsed time, then
/// pulled a little toward the heard audio, or onto it when far off.
fn follow_audio(wall: i64, heard: i64) -> i64 {
    let drift = heard.saturating_sub(wall);
    if drift.abs() > AUDIO_SNAP {
        heard
    } else {
        wall.saturating_add(drift / AUDIO_PULL)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_eval::ReaderFactory;
    use cutlass_model::Micros;

    use super::{Program, Stream, Ticked, Transport, follow_audio};
    use crate::apple::AudioHost;
    use crate::color_session;

    #[test]
    fn the_playhead_moves_by_elapsed_time_and_leans_toward_the_audio() {
        assert_eq!(follow_audio(1_000_000, 1_000_000), 1_000_000);
        assert_eq!(follow_audio(1_000_000, 1_016_000), 1_001_000);
        assert_eq!(follow_audio(1_000_000, 984_000), 999_000);
        assert_eq!(follow_audio(1_000_000, 1_100_000), 1_100_000);
        let stream = Stream {
            start: Micros(2_000_000),
            played: 0,
            heard: true,
            waited: 0,
            deaf: false,
        };
        assert_eq!(
            stream.time_at(u64::from(crate::MIX_WINDOW)),
            Micros(2_020_000)
        );
    }

    #[test]
    fn a_device_that_never_plays_hands_the_playhead_to_the_wall() {
        let mut stream = Stream {
            start: Micros(1_000_000),
            played: 0,
            heard: false,
            waited: 0,
            deaf: false,
        };
        // Held at the start while the device warms up.
        assert_eq!(stream.playhead(0, 1_016_000, 16_000), Some(1_000_000));
        let mut wall = 1_016_000;
        let mut last = Some(1_000_000);
        while last.is_some() {
            wall += 16_000;
            last = stream.playhead(0, wall, 16_000);
            assert!(wall < 1_400_000, "still held at {wall}");
        }
        assert!(stream.deaf);
        // Sound arriving late does not pull the picture back.
        assert_eq!(stream.playhead(4_800, wall, 16_000), None);
    }

    #[test]
    fn the_first_heard_frame_lands_the_playhead_on_it() {
        let mut stream = Stream {
            start: Micros(0),
            played: 0,
            heard: false,
            waited: 0,
            deaf: false,
        };
        assert_eq!(stream.playhead(0, 16_000, 16_000), Some(0));
        assert_eq!(stream.playhead(480, 32_000, 16_000), Some(10_000));
        assert!(stream.heard && !stream.deaf);
        assert_eq!(stream.playhead(960, 26_000, 16_000), Some(25_625));
    }

    fn program<'a>(
        session: &'a cutlass_session::Session,
        readers: &'a ReaderFactory,
    ) -> Program<'a> {
        Program {
            draft: session.draft(),
            revision: session.revision(),
            root: None,
            readers,
        }
    }

    #[test]
    fn without_a_stream_the_playhead_follows_the_wall() {
        let session = color_session(1_000_000, 1_000_000);
        let readers = ReaderFactory::default();
        let mut transport = Transport::new();
        transport.start(program(&session, &readers), Micros(0), Micros(1_000_000));
        assert!(!transport.is_streaming());
        assert_eq!(transport.heard_time(), None);
        let ticked = transport.tick(
            program(&session, &readers),
            Micros(0),
            20_000,
            Micros(1_000_000),
            false,
        );
        assert_eq!(
            ticked,
            Ticked {
                playhead: Micros(20_000),
                playing: true
            }
        );
        // A draft with no sound mixes to silence.
        assert!(transport.output().queued().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn play_stops_at_the_end_or_wraps_when_looping() {
        let session = color_session(1_000_000, 1_000_000);
        let readers = ReaderFactory::default();
        let mut transport = Transport::new();
        let end = Micros(1_000_000);
        let stopped = transport.tick(
            program(&session, &readers),
            Micros(990_000),
            20_000,
            end,
            false,
        );
        assert_eq!(
            stopped,
            Ticked {
                playhead: end,
                playing: false
            }
        );
        assert!(transport.output().queued().is_empty(), "silent at the end");
        let wrapped = transport.tick(
            program(&session, &readers),
            Micros(990_000),
            20_000,
            end,
            true,
        );
        assert_eq!(
            wrapped,
            Ticked {
                playhead: Micros(10_000),
                playing: true
            }
        );
    }

    #[test]
    fn an_empty_timeline_stops_at_zero_and_no_time_holds_the_playhead() {
        let session = color_session(1_000_000, 1_000_000);
        let readers = ReaderFactory::default();
        let mut transport = Transport::new();
        let empty = transport.tick(
            program(&session, &readers),
            Micros(500),
            20_000,
            Micros(0),
            false,
        );
        assert_eq!(
            empty,
            Ticked {
                playhead: Micros(0),
                playing: false
            }
        );
        let held = transport.tick(
            program(&session, &readers),
            Micros(500),
            0,
            Micros(1_000_000),
            false,
        );
        assert_eq!(
            held,
            Ticked {
                playhead: Micros(500),
                playing: true
            }
        );
    }

    #[test]
    fn a_silent_host_names_its_gap_and_never_streams() {
        let session = color_session(1_000_000, 1_000_000);
        let readers = ReaderFactory::default();
        let mut transport = Transport::new();
        transport.open(AudioHost::Linux);
        assert!(transport.gap().is_some());
        transport.start(program(&session, &readers), Micros(0), Micros(1_000_000));
        assert!(!transport.is_streaming());
        assert!(transport.output().queued().is_empty());
    }

    #[test]
    #[ignore = "needs an audio output device"]
    fn a_device_stream_moves_the_playhead_with_the_heard_audio() {
        let session = color_session(2_000_000, 2_000_000);
        let readers = ReaderFactory::default();
        let mut transport = Transport::new();
        transport.open(AudioHost::current());
        if transport.output().clock().is_none() {
            return;
        }
        let end = Micros(2_000_000);
        transport.start(program(&session, &readers), Micros(0), end);
        assert!(transport.is_streaming());
        let mut playhead = Micros(0);
        let mut last = std::time::Instant::now();
        for _ in 0..60 {
            std::thread::sleep(std::time::Duration::from_micros(16_667));
            let now = std::time::Instant::now();
            let elapsed = i64::try_from(now.duration_since(last).as_micros()).unwrap_or(0);
            last = now;
            playhead = transport
                .tick(program(&session, &readers), playhead, elapsed, end, false)
                .playhead;
        }
        let heard = transport.heard_time().unwrap_or(Micros(0));
        assert!(playhead.0 > 500_000, "playback moved to {}", playhead.0);
        assert!((playhead.0 - heard.0).abs() < 10_000);
    }
}
