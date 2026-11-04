//! The level meter: how loud the heard audio is, on every platform alike.
//!
//! The mixer keeps the peak of each window it streams, and playback reads
//! the one the device is playing now. [`LevelMeter`] turns those peaks into
//! a reading: it rises at once, falls at [`FALL_DB_PER_SECOND`], and holds
//! each channel's peak for [`HOLD`] before the hold falls too.

use crate::apple::FRAME_BYTES;

/// The quietest level drawn. Anything below reads as the floor.
pub const METER_FLOOR_DB: f32 = -60.0;
/// How fast the bar falls once the sound gets quieter.
pub const FALL_DB_PER_SECOND: f32 = 24.0;
/// How long a peak mark stays before it falls, in microseconds.
pub const HOLD: i64 = 1_000_000;

/// The left and right peaks of interleaved little-endian f32 stereo bytes,
/// as amplitudes. An odd tail is ignored.
#[must_use]
pub fn window_peak(bytes: &[u8]) -> [f32; 2] {
    let mut peak = [0.0_f32; 2];
    for frame in bytes.chunks_exact(FRAME_BYTES) {
        for (channel, sample) in frame.chunks_exact(4).enumerate().take(2) {
            let value = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]).abs();
            if value.is_finite() && value > peak[channel] {
                peak[channel] = value;
            }
        }
    }
    peak
}

/// An amplitude in dB full scale, never under [`METER_FLOOR_DB`].
#[must_use]
pub fn level_db(amplitude: f32) -> f32 {
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return METER_FLOOR_DB;
    }
    (20.0 * amplitude.log10()).max(METER_FLOOR_DB)
}

/// Where `db` sits on a meter from the floor (0) to full scale (1).
#[must_use]
pub fn meter_fraction(db: f32) -> f32 {
    if !db.is_finite() {
        return 0.0;
    }
    ((db - METER_FLOOR_DB) / -METER_FLOOR_DB).clamp(0.0, 1.0)
}

/// One channel as drawn now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelReading {
    /// The bar, in dB full scale.
    pub level: f32,
    /// The held peak mark, in dB full scale.
    pub hold: f32,
    /// A sample reached full scale within the hold.
    pub clipped: bool,
}

impl ChannelReading {
    const FLOOR: Self = Self {
        level: METER_FLOOR_DB,
        hold: METER_FLOOR_DB,
        clipped: false,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Channel {
    reading: ChannelReading,
    /// How long the hold has stood.
    held: i64,
    /// How long since the last sample at full scale.
    since_clip: i64,
}

/// The meter's left and right bars between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelMeter {
    channels: [Channel; 2],
}

impl Default for LevelMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl LevelMeter {
    #[must_use]
    pub const fn new() -> Self {
        let channel = Channel {
            reading: ChannelReading::FLOOR,
            held: 0,
            since_clip: HOLD,
        };
        Self {
            channels: [channel; 2],
        }
    }

    /// Move the meter on by `elapsed` microseconds. `peak` is what is heard
    /// now, or `None` for silence, as when paused.
    pub fn update(&mut self, peak: Option<[f32; 2]>, elapsed: i64) {
        let elapsed = elapsed.max(0);
        let fall = FALL_DB_PER_SECOND * elapsed as f32 / 1_000_000.0;
        let peak = peak.unwrap_or([0.0; 2]);
        for (channel, amplitude) in self.channels.iter_mut().zip(peak) {
            let heard = level_db(amplitude);
            let reading = &mut channel.reading;
            reading.level = (reading.level - fall).max(heard).max(METER_FLOOR_DB);
            channel.held = channel.held.saturating_add(elapsed);
            if heard >= reading.hold {
                reading.hold = heard;
                channel.held = 0;
            } else if channel.held > HOLD {
                reading.hold = (reading.hold - fall).max(reading.level);
            }
            channel.since_clip = if amplitude >= 1.0 {
                0
            } else {
                channel.since_clip.saturating_add(elapsed)
            };
            reading.clipped = channel.since_clip < HOLD;
        }
    }

    /// The left and right channels as drawn now.
    #[must_use]
    pub fn reading(&self) -> [ChannelReading; 2] {
        self.channels.map(|channel| channel.reading)
    }

    /// Every bar is at the floor, so nothing moves until sound comes.
    #[must_use]
    pub fn is_resting(&self) -> bool {
        self.channels
            .iter()
            .all(|channel| channel.reading == ChannelReading::FLOOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(frames: &[[f32; 2]]) -> Vec<u8> {
        frames
            .iter()
            .flat_map(|[left, right]| {
                left.to_le_bytes()
                    .into_iter()
                    .chain(right.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn a_window_peak_is_each_channels_loudest_sample() {
        let peak = window_peak(&bytes(&[[0.1, -0.7], [-0.5, 0.2], [f32::NAN, 0.0]]));
        assert_eq!(peak, [0.5, 0.7]);
        assert_eq!(window_peak(&[1, 2, 3]), [0.0, 0.0]);
    }

    #[test]
    fn decibels_run_from_the_floor_to_full_scale() {
        assert_eq!(level_db(1.0), 0.0);
        assert!((level_db(0.5) + 6.0206).abs() < 0.001);
        assert_eq!(level_db(0.0), METER_FLOOR_DB);
        assert_eq!(level_db(1e-9), METER_FLOOR_DB);
        assert_eq!(meter_fraction(0.0), 1.0);
        assert_eq!(meter_fraction(METER_FLOOR_DB), 0.0);
        assert_eq!(meter_fraction(-30.0), 0.5);
        assert_eq!(meter_fraction(6.0), 1.0);
    }

    #[test]
    fn the_bar_rises_at_once_falls_steadily_and_the_hold_waits() {
        let mut meter = LevelMeter::new();
        assert!(meter.is_resting());
        meter.update(Some([1.0, 0.1]), 10_000);
        let [left, right] = meter.reading();
        assert_eq!((left.level, left.hold, left.clipped), (0.0, 0.0, true));
        assert_eq!(right.level, -20.0);
        assert!(!right.clipped);
        // Half a second of silence: the bar falls 12 dB, the hold stays.
        meter.update(None, 500_000);
        let [left, _] = meter.reading();
        assert_eq!((left.level, left.hold), (-12.0, 0.0));
        // Past the hold the mark falls too, never under the bar.
        meter.update(None, 600_000);
        let [left, _] = meter.reading();
        assert!((left.level + 26.4).abs() < 0.01);
        assert!((left.hold + 14.4).abs() < 0.01);
        assert!(!left.clipped);
        meter.update(None, 10_000_000);
        assert!(meter.is_resting());
    }
}
