use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// An integer count of microseconds.
///
/// Both the source range and the target range use this clock. The file
/// timescale is not stored on a segment.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Micros(pub i64);

/// A half-open span on one clock: `[start, start + duration)`.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Micros,
    pub duration: Micros,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// A range is not usable as an interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeError {
    /// Duration is zero or negative. Overlap is not tested.
    NonPositiveDuration,
    /// `start + duration` does not fit in an `i64`.
    EndOverflow,
}

/// A frame index could not be decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    NonPositiveDuration,
    EndOverflow,
    /// The rate is empty, or the index or time is negative.
    NotAFrame,
}

impl TimeRange {
    pub fn new(start: i64, duration: i64) -> Self {
        Self {
            start: Micros(start),
            duration: Micros(duration),
            rest: Map::new(),
        }
    }

    /// The end of the span. Addition saturates at the `i64` limits.
    #[must_use]
    pub fn end(&self) -> Micros {
        Micros(self.start.0.saturating_add(self.duration.0))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.duration.0 <= 0
    }

    /// Two positive ranges overlap when each starts before the other ends.
    /// Touching at an endpoint does not overlap. A zero or negative duration
    /// is refused before that comparison.
    pub fn overlaps(&self, other: &Self) -> Result<bool, RangeError> {
        if self.duration.0 <= 0 || other.duration.0 <= 0 {
            return Err(RangeError::NonPositiveDuration);
        }
        let self_end = self
            .start
            .0
            .checked_add(self.duration.0)
            .ok_or(RangeError::EndOverflow)?;
        let other_end = other
            .start
            .0
            .checked_add(other.duration.0)
            .ok_or(RangeError::EndOverflow)?;
        Ok(self.start.0 < other_end && other.start.0 < self_end)
    }
}

/// A rational frame rate. Stored draft time stays microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

/// A display label. Drop frame changes this label, not the stored microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimecodeLabel {
    pub hours: u64,
    pub minutes: u64,
    pub seconds: u64,
    pub frames: u64,
    pub drop_frame: bool,
}

impl FrameRate {
    pub const FPS_24: Self = Self {
        numerator: 24,
        denominator: 1,
    };
    pub const FPS_25: Self = Self {
        numerator: 25,
        denominator: 1,
    };
    pub const FPS_30: Self = Self {
        numerator: 30,
        denominator: 1,
    };
    pub const FPS_50: Self = Self {
        numerator: 50,
        denominator: 1,
    };
    pub const FPS_60: Self = Self {
        numerator: 60,
        denominator: 1,
    };
    pub const FPS_120: Self = Self {
        numerator: 120,
        denominator: 1,
    };
    pub const FPS_23_976: Self = Self {
        numerator: 24_000,
        denominator: 1001,
    };
    pub const FPS_29_97: Self = Self {
        numerator: 30_000,
        denominator: 1001,
    };
    pub const FPS_59_94: Self = Self {
        numerator: 60_000,
        denominator: 1001,
    };

    /// The frame a non-negative microsecond belongs to.
    ///
    /// `micros * numerator / (denominator * 1_000_000)`, with a wide product.
    #[must_use]
    pub fn frame_index(self, time: Micros) -> Option<i64> {
        if time.0 < 0 || self.numerator == 0 || self.denominator == 0 {
            return None;
        }
        let micros = u128::from(time.0 as u64);
        let numerator = u128::from(self.numerator);
        let denominator = u128::from(self.denominator);
        let index = micros * numerator / (denominator * 1_000_000);
        i64::try_from(index).ok()
    }

    /// The smallest microsecond that maps to `index`.
    #[must_use]
    pub fn frame_start(self, index: i64) -> Option<Micros> {
        if index < 0 || self.numerator == 0 || self.denominator == 0 {
            return None;
        }
        let index = u128::try_from(index).ok()?;
        let numerator = u128::from(self.numerator);
        let denominator = u128::from(self.denominator);
        let product = index.checked_mul(denominator)?.checked_mul(1_000_000)?;
        let start = product.div_ceil(numerator);
        i64::try_from(start).ok().map(Micros)
    }

    /// A half-open range includes a frame when that frame's start is inside it.
    pub fn contains_frame(self, range: &TimeRange, index: i64) -> Result<bool, FrameError> {
        if range.duration.0 <= 0 {
            return Err(FrameError::NonPositiveDuration);
        }
        let frame_start = self.frame_start(index).ok_or(FrameError::NotAFrame)?;
        let end = range
            .start
            .0
            .checked_add(range.duration.0)
            .ok_or(FrameError::EndOverflow)?;
        Ok(frame_start.0 >= range.start.0 && frame_start.0 < end)
    }

    /// Hours, minutes, seconds, and frames, counted at the nominal rate:
    /// 24 for 23.976, 30 for 29.97, and 60 for 59.94. Drop frame applies
    /// on 29.97 and 59.94 only. It skips 2 or 4 frame numbers at the start
    /// of each minute, except every tenth, and puts a semicolon before the
    /// frames. Other rates ignore the flag.
    #[must_use]
    pub fn label(self, time: Micros, drop_frame: bool) -> Option<TimecodeLabel> {
        let index = self.frame_index(time)?;
        let nominal = i64::from(self.nominal()?);
        let skip = if drop_frame {
            self.dropped_per_minute()
        } else {
            0
        };
        let counted = if skip > 0 {
            drop_frame_count(index, nominal, skip)
        } else {
            index
        };
        let frames = u64::try_from(counted % nominal).ok()?;
        let total_seconds = u64::try_from(counted / nominal).ok()?;
        Some(TimecodeLabel {
            hours: total_seconds / 3600,
            minutes: (total_seconds / 60) % 60,
            seconds: total_seconds % 60,
            frames,
            drop_frame: skip > 0,
        })
    }

    /// Whole frames a timecode second counts: the rate rounded, so 24 for
    /// 23.976 and 30 for 29.97.
    #[must_use]
    pub fn nominal(self) -> Option<u32> {
        if self.numerator == 0 || self.denominator == 0 {
            return None;
        }
        let rounded = (u64::from(self.numerator) + u64::from(self.denominator) / 2)
            / u64::from(self.denominator);
        u32::try_from(rounded).ok().filter(|rate| *rate > 0)
    }

    /// Frame numbers drop frame skips each minute: 2 at 29.97, 4 at 59.94,
    /// and none at any other rate.
    #[must_use]
    pub fn dropped_per_minute(self) -> i64 {
        match (self.numerator, self.denominator) {
            (30_000, 1001) => 2,
            (60_000, 1001) => 4,
            _ => 0,
        }
    }

    /// Read the draft's stored `fps` number. The drop-frame flag is not applied here.
    #[must_use]
    pub fn from_stored_fps(fps: f64) -> Option<Self> {
        if !fps.is_finite() || fps <= 0.0 {
            return None;
        }
        const FRACTIONS: [(u32, u32); 3] = [(24_000, 1001), (30_000, 1001), (60_000, 1001)];
        for (numerator, denominator) in FRACTIONS {
            let exact = f64::from(numerator) / f64::from(denominator);
            if (fps - exact).abs() < 0.000_5 {
                return Some(Self {
                    numerator,
                    denominator,
                });
            }
        }
        let rounded = fps.round();
        if (fps - rounded).abs() < 0.000_5 && rounded >= 1.0 && rounded <= f64::from(u32::MAX) {
            return Some(Self {
                numerator: rounded as u32,
                denominator: 1,
            });
        }
        None
    }
}

/// The frame number SMPTE drop frame shows for frame `index`.
fn drop_frame_count(index: i64, nominal: i64, skip: i64) -> i64 {
    let per_minute = nominal * 60 - skip;
    let per_ten = nominal * 600 - skip * 9;
    let (tens, within) = (index / per_ten, index % per_ten);
    let mut counted = index + skip * 9 * tens;
    if within > skip {
        counted += skip * ((within - skip) / per_minute);
    }
    counted
}

impl TimecodeLabel {
    #[must_use]
    pub fn format(self) -> String {
        let frames = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{frames}{:02}",
            self.hours, self.minutes, self.seconds, self.frames
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_ends_at_start_plus_duration() {
        let range = TimeRange::new(1_000, 5_000_000);
        assert_eq!(range.end(), Micros(5_001_000));
        let json = serde_json::to_value(&range).unwrap();
        assert_eq!(json["start"], 1_000);
        assert_eq!(json["duration"], 5_000_000);
        let back: TimeRange = serde_json::from_value(json).unwrap();
        assert_eq!(back, range);
    }

    #[test]
    fn a_range_keeps_unnamed_keys() {
        let range: TimeRange =
            serde_json::from_str(r#"{"start":1,"duration":2,"future_range":true}"#).unwrap();
        assert_eq!(range.end(), Micros(3));
        assert_eq!(serde_json::to_value(range).unwrap()["future_range"], true);
    }

    #[test]
    fn touching_ranges_do_not_overlap() {
        let left = TimeRange::new(0, 100);
        let right = TimeRange::new(100, 50);
        assert_eq!(left.overlaps(&right), Ok(false));
        assert_eq!(left.overlaps(&TimeRange::new(50, 50)), Ok(true));
    }

    #[test]
    fn a_non_positive_duration_is_refused_before_overlap() {
        assert_eq!(
            TimeRange::new(0, 0).overlaps(&TimeRange::new(0, 10)),
            Err(RangeError::NonPositiveDuration)
        );
        assert_eq!(
            TimeRange::new(0, 10).overlaps(&TimeRange::new(0, -1)),
            Err(RangeError::NonPositiveDuration)
        );
        assert_eq!(
            TimeRange::new(1, i64::MAX).overlaps(&TimeRange::new(0, 1)),
            Err(RangeError::EndOverflow)
        );
    }

    #[test]
    fn frame_boundaries_follow_the_rational() {
        for rate in [
            FrameRate::FPS_30,
            FrameRate::FPS_29_97,
            FrameRate::FPS_23_976,
        ] {
            for index in 0..120 {
                let start = rate.frame_start(index).unwrap();
                assert_eq!(rate.frame_index(start), Some(index));
                if index > 0 {
                    assert_eq!(rate.frame_index(Micros(start.0 - 1)), Some(index - 1));
                }
            }
        }
        assert_eq!(FrameRate::FPS_30.frame_index(Micros(-1)), None);
    }

    #[test]
    fn a_half_open_second_includes_thirty_frames() {
        let range = TimeRange::new(0, 1_000_000);
        assert_eq!(FrameRate::FPS_30.contains_frame(&range, 0), Ok(true));
        assert_eq!(FrameRate::FPS_30.contains_frame(&range, 29), Ok(true));
        assert_eq!(FrameRate::FPS_30.contains_frame(&range, 30), Ok(false));
        assert_eq!(
            FrameRate::FPS_30.contains_frame(&TimeRange::new(0, 0), 0),
            Err(FrameError::NonPositiveDuration)
        );
    }

    #[test]
    fn stored_fps_numbers_become_rationals() {
        assert_eq!(FrameRate::from_stored_fps(30.0), Some(FrameRate::FPS_30));
        assert_eq!(
            FrameRate::from_stored_fps(29.97),
            Some(FrameRate::FPS_29_97)
        );
        assert_eq!(
            FrameRate::from_stored_fps(30_000.0 / 1001.0),
            Some(FrameRate::FPS_29_97)
        );
        assert_eq!(
            FrameRate::from_stored_fps(23.976),
            Some(FrameRate::FPS_23_976)
        );
        assert_eq!(
            FrameRate::from_stored_fps(59.94),
            Some(FrameRate::FPS_59_94)
        );
        assert_eq!(
            FrameRate::from_stored_fps(48.0),
            Some(FrameRate {
                numerator: 48,
                denominator: 1
            })
        );
        assert_eq!(FrameRate::from_stored_fps(23.98), None);
        assert_eq!(FrameRate::from_stored_fps(0.0), None);
        assert_eq!(FrameRate::from_stored_fps(f64::NAN), None);
    }

    #[test]
    fn drop_frame_changes_the_label_only() {
        assert_eq!(
            FrameRate::FPS_30
                .label(Micros(1_000_000), false)
                .unwrap()
                .format(),
            "00:00:01:00"
        );
        assert_eq!(
            FrameRate::FPS_29_97
                .label(Micros(0), true)
                .unwrap()
                .format(),
            "00:00:00;00"
        );
        let one_second = FrameRate::FPS_23_976.frame_start(24).unwrap();
        assert_eq!(
            FrameRate::FPS_23_976
                .label(one_second, true)
                .unwrap()
                .format(),
            "00:00:01:00"
        );
        assert_eq!(
            FrameRate::FPS_30.label(Micros(0), true).unwrap().format(),
            "00:00:00:00"
        );
    }

    #[test]
    fn a_fractional_rate_counts_nominal_frames_without_drop_frame() {
        let rate = FrameRate::FPS_29_97;
        let at = |index| rate.label(rate.frame_start(index).unwrap(), false);
        assert_eq!(at(29).unwrap().format(), "00:00:00:29");
        assert_eq!(at(30).unwrap().format(), "00:00:01:00");
        assert_eq!(at(1800).unwrap().format(), "00:01:00:00");
        assert_eq!(FrameRate::FPS_23_976.nominal(), Some(24));
        assert_eq!(FrameRate::FPS_59_94.nominal(), Some(60));
    }

    #[test]
    fn drop_frame_skips_numbers_each_minute_but_every_tenth() {
        let label = |rate: FrameRate, index| {
            rate.label(rate.frame_start(index).unwrap(), true)
                .unwrap()
                .format()
        };
        let ntsc = FrameRate::FPS_29_97;
        assert_eq!(label(ntsc, 1799), "00:00:59;29");
        assert_eq!(label(ntsc, 1800), "00:01:00;02");
        assert_eq!(label(ntsc, 3597), "00:01:59;29");
        assert_eq!(label(ntsc, 3598), "00:02:00;02");
        assert_eq!(label(ntsc, 17_981), "00:09:59;29");
        assert_eq!(label(ntsc, 17_982), "00:10:00;00");
        // An hour of 29.97 is 107,892 frames.
        assert_eq!(label(ntsc, 107_892), "01:00:00;00");
        let double = FrameRate::FPS_59_94;
        assert_eq!(label(double, 3596), "00:00:59;56");
        assert_eq!(label(double, 3600), "00:01:00;04");
        assert_eq!(label(double, 35_964), "00:10:00;00");
    }
}
