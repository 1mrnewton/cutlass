//! Reading individual source frames out of one media file.
//!
//! The decoder is *sequential and stateful*: cheap to step forward one frame,
//! expensive to jump (a seek flushes buffers and re-decodes from a keyframe). A
//! [`MediaReader`] hides that by tracking the decoder's current position and
//! choosing between **stepping forward** (the target is just ahead) and
//! **seeking** (a backward or far-forward jump). Frame indices are the media's
//! *native* source frames; the engine converts timeline frames to source frames
//! before calling in.

use std::time::Duration;

use cutlass_decode::{DecodeOptions, DecodedFrame, Decoder};
use cutlass_models::{MediaId, MediaSource, Rational};

use crate::error::EngineError;

/// Produces a decoded frame for a given source-frame index.
///
/// Abstracts the real ffmpeg-backed [`MediaReader`] so the [`MediaPool`] cache
/// routing can be tested with a deterministic fake.
///
/// [`MediaPool`]: crate::MediaPool
pub trait FrameReader {
    fn read(&mut self, source_frame: i64) -> Result<DecodedFrame, EngineError>;
}

/// How many frames ahead of the current position we will step (decode forward)
/// rather than seek. Stepping avoids a buffer flush + keyframe re-decode, so for
/// nearby forward targets — the playback and short-scrub case — it wins. Far
/// jumps seek instead. This is a heuristic; a keyframe index could later make it
/// exact (step only within the current GOP).
const MAX_STEP_AHEAD: i64 = 48;

/// Sequential frame reader over one decoded media file.
pub struct MediaReader {
    media: MediaId,
    decoder: Decoder,
    /// Native frame rate, used to map source-frame index <-> presentation time.
    fps: Rational,
    /// Seconds per stream tick, precomputed so we never name ffmpeg's `Rational`.
    secs_per_tick: f64,
    /// Total source length in frames; targets at/after this are out of range.
    duration_frames: i64,
    /// Index of the last frame handed out, if any (the decoder's position).
    current: Option<i64>,
}

impl MediaReader {
    /// Open `media`'s file for decoding, using default decode options.
    pub fn open(media: &MediaSource) -> Result<Self, EngineError> {
        Self::open_with(media, DecodeOptions::default())
    }

    /// Open `media`'s file with explicit decode options (e.g. forcing software).
    pub fn open_with(media: &MediaSource, options: DecodeOptions) -> Result<Self, EngineError> {
        let decoder = Decoder::open_with(media.path(), options)?;
        Ok(Self::from_decoder(
            media.id,
            decoder,
            media.frame_rate,
            media.duration,
        ))
    }

    fn from_decoder(
        media: MediaId,
        decoder: Decoder,
        fps: Rational,
        duration_frames: i64,
    ) -> Self {
        // One tick is tiny; scale up then divide to keep float precision.
        let secs_per_tick =
            cutlass_decode::ticks_to_duration(decoder.info().time_base, 1_000_000).as_secs_f64()
                / 1_000_000.0;
        Self {
            media,
            decoder,
            fps,
            secs_per_tick,
            duration_frames,
            current: None,
        }
    }

    /// The source-frame index a decoded frame's PTS corresponds to.
    fn pts_to_index(&self, pts_ticks: i64) -> i64 {
        let seconds = pts_ticks as f64 * self.secs_per_tick;
        (seconds * self.fps.as_f64()).round() as i64
    }

    /// Presentation time at the start of source frame `index`.
    fn index_to_time(&self, index: i64) -> Duration {
        let seconds = index.max(0) as f64 * self.fps.seconds_per_frame();
        Duration::from_secs_f64(seconds.max(0.0))
    }

    /// Whether the target is reachable by stepping forward from `current`.
    fn can_step_to(&self, target: i64) -> bool {
        match self.current {
            Some(cur) => target >= cur && target - cur <= MAX_STEP_AHEAD,
            None => false,
        }
    }
}

impl FrameReader for MediaReader {
    fn read(&mut self, source_frame: i64) -> Result<DecodedFrame, EngineError> {
        let media = self.media;
        let out_of_range = move || EngineError::FrameOutOfRange {
            media,
            frame: source_frame,
        };
        if self.duration_frames > 0 && source_frame >= self.duration_frames {
            return Err(out_of_range());
        }

        if self.can_step_to(source_frame) {
            // Decode forward until we reach (or just pass) the target frame.
            while let Some(frame) = self.decoder.next_frame()? {
                let idx = self.pts_to_index(frame.pts_ticks);
                if idx >= source_frame {
                    self.current = Some(idx);
                    return Ok(frame);
                }
            }
            return Err(out_of_range());
        }

        // Backward or far jump: seek to the keyframe and decode up to the target.
        match self.decoder.seek_to_frame(self.index_to_time(source_frame))? {
            Some(frame) => {
                self.current = Some(self.pts_to_index(frame.pts_ticks));
                Ok(frame)
            }
            None => Err(out_of_range()),
        }
    }
}

/// Map `source_frame` to the wall-clock time at the start of that frame.
///
/// Free function mirroring [`MediaReader::index_to_time`] for callers that only
/// have a [`Rational`] frame rate (e.g. timeline-to-source planning).
pub fn frame_to_time(source_frame: i64, fps: Rational) -> Duration {
    if !fps.is_valid() {
        return Duration::ZERO;
    }
    let seconds = source_frame.max(0) as f64 * fps.seconds_per_frame();
    Duration::from_secs_f64(seconds.max(0.0))
}

/// Round a wall-clock time to the nearest source-frame index at `fps`.
pub fn time_to_frame(time: Duration, fps: Rational) -> i64 {
    if !fps.is_valid() {
        return 0;
    }
    (time.as_secs_f64() * fps.as_f64()).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_time_roundtrips_at_integer_rate() {
        let fps = Rational::FPS_30;
        for frame in [0, 1, 29, 30, 100, 1000] {
            let t = frame_to_time(frame, fps);
            assert_eq!(time_to_frame(t, fps), frame, "frame {frame}");
        }
    }

    #[test]
    fn frame_time_roundtrips_at_ntsc_rate() {
        let fps = Rational::FPS_23_976;
        for frame in [0, 1, 24, 240, 2400] {
            let t = frame_to_time(frame, fps);
            assert_eq!(time_to_frame(t, fps), frame, "frame {frame}");
        }
    }

    #[test]
    fn invalid_rate_is_zero() {
        let bad = Rational::new(0, 0);
        assert_eq!(frame_to_time(10, bad), Duration::ZERO);
        assert_eq!(time_to_frame(Duration::from_secs(1), bad), 0);
    }

    #[test]
    fn negative_frame_clamps_to_zero_time() {
        assert_eq!(frame_to_time(-5, Rational::FPS_24), Duration::ZERO);
    }
}
