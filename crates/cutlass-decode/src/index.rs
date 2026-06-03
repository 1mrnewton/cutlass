//! Keyframe index for fast, predictable seeking.
//!
//! [`KeyframeIndex::build`] demuxes a file once (no decode) and records the
//! presentation timestamp of every keyframe (I-frame) on the best video stream.
//! Given a target time, the nearest keyframe at or before it is the only valid
//! entry point for decode, so this index lets callers know exactly where a seek
//! must start — and how many frames lie between that keyframe and the target.

use std::path::Path;
use std::time::Duration;

use ffmpeg_next::format::{self};
use ffmpeg_next::media::Type;
use ffmpeg_next::packet::Packet;
use ffmpeg_next::{Error as FfmpegError, Rational};
use tracing::debug;

use crate::error::DecodeError;

/// Sorted keyframe timestamps for one video stream.
///
/// Timestamps are stored in the stream's `time_base` ticks for exactness;
/// use [`ticks_to_duration`](Self::ticks_to_duration) /
/// [`duration_to_ticks`](Self::duration_to_ticks) to convert at the boundary.
#[derive(Debug, Clone)]
pub struct KeyframeIndex {
    time_base: Rational,
    /// Ascending, de-duplicated keyframe PTS values (time_base ticks).
    keyframes: Vec<i64>,
}

impl KeyframeIndex {
    /// Demux `path` once and collect every keyframe's presentation timestamp.
    ///
    /// This does not decode any frames — it only inspects packet flags, so the
    /// cost is dominated by I/O and is suitable to run at import time.
    pub fn build(path: &Path) -> Result<Self, DecodeError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;

        let mut input = format::input(path_str).map_err(DecodeError::Open)?;

        let (stream_index, time_base) = {
            let stream = input
                .streams()
                .best(Type::Video)
                .ok_or_else(|| DecodeError::unsupported("no video stream found"))?;
            (stream.index(), stream.time_base())
        };

        let mut keyframes = Vec::new();
        let mut packet = Packet::empty();
        loop {
            match packet.read(&mut input) {
                Ok(()) => {
                    if packet.stream() == stream_index && packet.is_key() {
                        if let Some(pts) = packet.pts().or_else(|| packet.dts()) {
                            keyframes.push(pts);
                        }
                    }
                }
                Err(FfmpegError::Eof) => break,
                Err(e) => return Err(DecodeError::Io(e)),
            }
        }

        // Packets arrive in decode (DTS) order; keyframe PTS values can be
        // slightly out of order around B-frames, so normalize.
        keyframes.sort_unstable();
        keyframes.dedup();

        debug!(
            count = keyframes.len(),
            tb_num = time_base.numerator(),
            tb_den = time_base.denominator(),
            "built keyframe index"
        );

        Ok(Self {
            time_base,
            keyframes,
        })
    }

    /// The stream `time_base` these ticks are expressed in.
    pub fn time_base(&self) -> Rational {
        self.time_base
    }

    /// Keyframe presentation timestamps in ascending `time_base` ticks.
    pub fn keyframe_ticks(&self) -> &[i64] {
        &self.keyframes
    }

    /// Keyframe presentation times as wall-clock [`Duration`]s, ascending.
    pub fn keyframe_times(&self) -> impl Iterator<Item = Duration> + '_ {
        let tb = self.time_base;
        self.keyframes
            .iter()
            .map(move |&ticks| ticks_to_duration(tb, ticks))
    }

    /// Number of indexed keyframes.
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// True when no keyframes were found.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Convert `time_base` ticks into a wall-clock [`Duration`].
    pub fn ticks_to_duration(&self, ticks: i64) -> Duration {
        ticks_to_duration(self.time_base, ticks)
    }

    /// Convert a wall-clock [`Duration`] into `time_base` ticks.
    pub fn duration_to_ticks(&self, target: Duration) -> i64 {
        duration_to_ticks(self.time_base, target)
    }
}

/// Convert `time_base` ticks into a wall-clock [`Duration`] (clamped at zero).
pub fn ticks_to_duration(time_base: Rational, ticks: i64) -> Duration {
    let num = time_base.numerator();
    let den = time_base.denominator();
    if num <= 0 || den <= 0 || ticks <= 0 {
        return Duration::ZERO;
    }
    // seconds = ticks * (num / den)
    let secs = ticks as f64 * (f64::from(num) / f64::from(den));
    Duration::from_secs_f64(secs.max(0.0))
}

/// Convert a wall-clock [`Duration`] into `time_base` ticks.
pub fn duration_to_ticks(time_base: Rational, target: Duration) -> i64 {
    let num = time_base.numerator();
    let den = time_base.denominator();
    if num <= 0 || den <= 0 {
        return 0;
    }
    // ticks = seconds * (den / num)
    (target.as_secs_f64() * (f64::from(den) / f64::from(num))) as i64
}
