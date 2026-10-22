//! The preview clock.
//!
//! Play runs the playhead from where it sits to the end of the cut, then
//! stops on the last frame. The wall clock moves a cut with no sound. When
//! the device is playing, the audio clock moves it instead.

use std::time::Duration;

use cutlass_project::FrameRate;

use crate::ruler;

/// One run of play, anchored at the playhead where it started.
pub(crate) struct Run {
    from_us: i64,
    started: std::time::Instant,
}

impl Run {
    fn at(from_us: i64) -> Self {
        Self {
            from_us,
            started: std::time::Instant::now(),
        }
    }

    pub(crate) fn origin_us(&self) -> i64 {
        self.from_us
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Where the playhead is after `elapsed` of play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Motion {
    /// The cut has no length.
    Idle,
    /// Still running, at this sequence time.
    Running(i64),
    /// Stopped on the last frame.
    Finished(i64),
}

/// Sequence time after `elapsed` from `from_us`. The result is a frame time.
pub(crate) fn advance(
    from_us: i64,
    elapsed: Duration,
    duration_us: i64,
    rate: FrameRate,
) -> Motion {
    let end = duration_us.max(0);
    if end == 0 {
        return Motion::Idle;
    }
    let last = last_frame(end, rate);
    let elapsed_us = i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX);
    let raw = from_us.max(0).saturating_add(elapsed_us);
    if raw >= end {
        return Motion::Finished(last);
    }
    let snapped = ruler::seek(raw, rate, end);
    if snapped >= end {
        Motion::Finished(last)
    } else {
        Motion::Running(snapped)
    }
}

/// Starts at `playhead`, or at the beginning when it is already on the last
/// frame. A second call pauses and leaves the playhead where it is. An empty
/// cut does not start.
pub(crate) fn toggle(
    play: &mut Option<Run>,
    playhead: i64,
    duration_us: i64,
    rate: FrameRate,
) -> i64 {
    if play.take().is_some() {
        return playhead;
    }
    let end = duration_us.max(0);
    if end == 0 {
        return playhead;
    }
    let last = last_frame(end, rate);
    let from = if playhead >= last { 0 } else { playhead.max(0) };
    *play = Some(Run::at(from));
    from
}

/// How long one frame of `rate` lasts. Play asks for the next paint then.
pub(crate) fn frame_wait(rate: FrameRate) -> Duration {
    let numer = u64::from(rate.numerator()).max(1);
    let nanos = u64::from(rate.denominator()).saturating_mul(1_000_000_000) / numer;
    Duration::from_nanos(nanos.max(1))
}

fn last_frame(duration_us: i64, rate: FrameRate) -> i64 {
    ruler::seek(duration_us.saturating_sub(1), rate, duration_us)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use cutlass_project::FrameRate;

    use super::{Motion, advance, frame_wait, toggle};

    #[test]
    fn play_runs_to_the_last_frame_and_stops() {
        let rate = FrameRate::Fps30;
        let duration = 1_000_000;
        assert_eq!(
            advance(0, Duration::ZERO, duration, rate),
            Motion::Running(0)
        );
        assert_eq!(
            advance(0, Duration::from_millis(500), duration, rate),
            Motion::Running(crate::ruler::seek(500_000, rate, duration))
        );
        let last = crate::ruler::seek(duration - 1, rate, duration);
        assert!(last < duration);
        assert_eq!(
            advance(0, Duration::from_secs(2), duration, rate),
            Motion::Finished(last)
        );
    }

    #[test]
    fn an_empty_cut_does_not_play() {
        assert_eq!(
            advance(0, Duration::from_secs(1), 0, FrameRate::Fps30),
            Motion::Idle
        );
        let mut play = None;
        assert_eq!(toggle(&mut play, 0, 0, FrameRate::Fps30), 0);
        assert!(play.is_none());
    }

    #[test]
    fn pause_keeps_the_playhead_and_the_end_starts_over() {
        let rate = FrameRate::Fps30;
        let duration = 1_000_000;
        let mut play = None;
        assert_eq!(toggle(&mut play, 100_000, duration, rate), 100_000);
        assert!(play.is_some());
        assert_eq!(toggle(&mut play, 100_000, duration, rate), 100_000);
        assert!(play.is_none());
        let last = crate::ruler::seek(duration - 1, rate, duration);
        assert_eq!(toggle(&mut play, last, duration, rate), 0);
        assert_eq!(play.as_ref().map(super::Run::origin_us), Some(0));
    }

    #[test]
    fn one_frame_is_the_gap_between_paints() {
        assert_eq!(
            frame_wait(FrameRate::Fps30),
            Duration::from_nanos(1_000_000_000 / 30)
        );
        assert_eq!(
            frame_wait(FrameRate::Fps23976),
            Duration::from_nanos(1_001 * 1_000_000_000 / 24_000)
        );
    }
}
