//! The sequence clock, a start-plus-duration range, and speed.

use std::num::NonZeroU32;

use cutlass_core::Timestamp;

use crate::Error;

/// One sequence tick is one microsecond.
pub const SEQUENCE_TIMESCALE: NonZeroU32 = NonZeroU32::new(1_000_000).expect("nonzero");

/// How fast source time plays on the sequence.
///
/// `2/1` is twice as fast, so the timeline span is half the source span.
/// `1/1` is the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speed {
    numerator: u32,
    denominator: NonZeroU32,
}

impl Speed {
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: NonZeroU32::MIN,
    };

    pub fn new(numerator: u32, denominator: u32) -> Result<Self, Error> {
        if numerator == 0 {
            return Err(Error::ZeroSpeed);
        }
        let denominator = NonZeroU32::new(denominator).ok_or(Error::ZeroSpeed)?;
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    pub const fn denominator(self) -> NonZeroU32 {
        self.denominator
    }
}

/// A half-open range. `start` and `duration` share a timescale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    start: Timestamp,
    duration: Timestamp,
}

impl TimeRange {
    pub fn new(start: Timestamp, duration: Timestamp) -> Result<Self, Error> {
        if start.timescale != duration.timescale {
            return Err(Error::TimescaleMismatch {
                start: start.timescale.get(),
                duration: duration.timescale.get(),
            });
        }
        if duration.value < 0 {
            return Err(Error::NegativeDuration);
        }
        if duration.value == 0 {
            return Err(Error::ZeroDuration);
        }
        if start.value.checked_add(duration.value).is_none() {
            return Err(Error::DurationOverflow);
        }
        Ok(Self { start, duration })
    }

    /// A range on the sequence clock. `start` is zero or later.
    pub fn sequence(start: i64, duration: i64) -> Result<Self, Error> {
        if start < 0 {
            return Err(Error::NegativeTime);
        }
        Self::new(
            Timestamp::new(start, SEQUENCE_TIMESCALE),
            Timestamp::new(duration, SEQUENCE_TIMESCALE),
        )
    }

    pub const fn start(self) -> Timestamp {
        self.start
    }

    pub const fn duration(self) -> Timestamp {
        self.duration
    }

    pub fn end(self) -> Timestamp {
        Timestamp::new(
            self.start
                .value
                .checked_add(self.duration.value)
                .expect("range end fits"),
            self.start.timescale,
        )
    }

    /// True when the ranges share any time. Touching at an endpoint does not.
    pub fn overlaps(self, other: Self) -> bool {
        self.start < other.end() && other.start < self.end()
    }
}

/// Source duration divided by `speed`, rounded to the nearest microsecond.
///
/// An exact half rounds away from zero.
pub fn timeline_duration(source: Timestamp, speed: Speed) -> Result<Timestamp, Error> {
    if source.value < 0 {
        return Err(Error::NegativeDuration);
    }
    if source.value == 0 {
        return Err(Error::ZeroDuration);
    }
    let numer = i128::from(source.value)
        * i128::from(SEQUENCE_TIMESCALE.get())
        * i128::from(speed.denominator.get());
    let denom = i128::from(source.timescale.get()) * i128::from(speed.numerator);
    let value = round_half_away(numer, denom)?;
    if value == 0 {
        return Err(Error::RoundsToZero);
    }
    Ok(Timestamp::new(value, SEQUENCE_TIMESCALE))
}

fn round_half_away(numer: i128, denom: i128) -> Result<i64, Error> {
    if denom <= 0 {
        return Err(Error::ZeroSpeed);
    }
    let half = denom / 2;
    let rounded = if numer >= 0 {
        numer.checked_add(half).ok_or(Error::DurationOverflow)? / denom
    } else {
        numer.checked_sub(half).ok_or(Error::DurationOverflow)? / denom
    };
    i64::try_from(rounded).map_err(|_| Error::DurationOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(value: i64, timescale: u32) -> Timestamp {
        Timestamp::new(value, NonZeroU32::new(timescale).unwrap())
    }

    #[test]
    fn sequence_clock_is_one_microsecond() {
        assert_eq!(SEQUENCE_TIMESCALE.get(), 1_000_000);
    }

    #[test]
    fn range_rejects_mixed_timescales() {
        let error = TimeRange::new(ts(1, 1), ts(1_000, 1_000)).unwrap_err();
        assert_eq!(
            error,
            Error::TimescaleMismatch {
                start: 1,
                duration: 1_000
            }
        );
    }

    #[test]
    fn range_rejects_a_negative_or_zero_duration() {
        assert_eq!(
            TimeRange::new(ts(0, 30), ts(-1, 30)).unwrap_err(),
            Error::NegativeDuration
        );
        assert_eq!(
            TimeRange::new(ts(0, 30), ts(0, 30)).unwrap_err(),
            Error::ZeroDuration
        );
    }

    #[test]
    fn range_rejects_an_end_that_overflows() {
        let error = TimeRange::new(ts(i64::MAX, 30), ts(1, 30)).unwrap_err();
        assert_eq!(error, Error::DurationOverflow);
    }

    #[test]
    fn sequence_range_rejects_a_negative_start() {
        assert_eq!(
            TimeRange::sequence(-1, 10).unwrap_err(),
            Error::NegativeTime
        );
    }

    #[test]
    fn end_is_start_plus_duration() {
        let range = TimeRange::new(ts(10, 30), ts(5, 30)).unwrap();
        assert_eq!(range.end(), ts(15, 30));
    }

    #[test]
    fn overlap_uses_exact_time_not_raw_ticks() {
        let first = TimeRange::new(ts(0, 1), ts(1, 1)).unwrap();
        let inside = TimeRange::new(ts(1, 2), ts(1, 2)).unwrap();
        let touches = TimeRange::new(ts(2, 2), ts(1, 2)).unwrap();
        assert!(first.overlaps(inside));
        assert!(!first.overlaps(touches));
    }

    #[test]
    fn touching_endpoints_do_not_overlap() {
        let left = TimeRange::sequence(0, 10).unwrap();
        let right = TimeRange::sequence(10, 10).unwrap();
        assert!(!left.overlaps(right));
    }

    #[test]
    fn speed_rejects_zero() {
        assert_eq!(Speed::new(0, 1).unwrap_err(), Error::ZeroSpeed);
        assert_eq!(Speed::new(1, 0).unwrap_err(), Error::ZeroSpeed);
        assert_eq!(Speed::ONE.numerator(), 1);
        assert_eq!(Speed::ONE.denominator().get(), 1);
    }

    #[test]
    fn one_second_of_source_is_one_million_microseconds() {
        let duration = timeline_duration(ts(15_360, 15_360), Speed::ONE).unwrap();
        assert_eq!(duration, ts(1_000_000, 1_000_000));
    }

    #[test]
    fn double_speed_halves_the_timeline_duration() {
        let duration = timeline_duration(ts(1_000_000, 1_000_000), Speed::new(2, 1).unwrap());
        assert_eq!(duration.unwrap(), ts(500_000, 1_000_000));
    }

    #[test]
    fn half_speed_doubles_the_timeline_duration() {
        let duration = timeline_duration(ts(1_000_000, 1_000_000), Speed::new(1, 2).unwrap());
        assert_eq!(duration.unwrap(), ts(2_000_000, 1_000_000));
    }

    #[test]
    fn a_third_of_a_second_rounds_down() {
        // 1/3 second = 333_333.333... microseconds. The fraction is below a half.
        let duration = timeline_duration(ts(1, 3), Speed::ONE).unwrap();
        assert_eq!(duration.value, 333_333);
    }

    #[test]
    fn an_exact_half_microsecond_rounds_away_from_zero() {
        let away = timeline_duration(ts(1, 2_000_000), Speed::ONE).unwrap();
        assert_eq!(away.value, 1);
        assert_eq!(
            timeline_duration(ts(1, 3_000_000), Speed::ONE).unwrap_err(),
            Error::RoundsToZero
        );
        let above = timeline_duration(ts(2, 3_000_000), Speed::ONE).unwrap();
        assert_eq!(above.value, 1);
    }

    #[test]
    fn a_negative_half_rounds_away_from_zero() {
        assert_eq!(round_half_away(-1_000_000, 2_000_000).unwrap(), -1);
        assert_eq!(round_half_away(-1, 3).unwrap(), 0);
        assert_eq!(round_half_away(-2, 3).unwrap(), -1);
    }

    #[test]
    fn a_negative_source_duration_is_rejected() {
        assert_eq!(
            timeline_duration(ts(-1, 30), Speed::ONE).unwrap_err(),
            Error::NegativeDuration
        );
    }

    #[test]
    fn a_duration_that_does_not_fit_the_sequence_clock_is_rejected() {
        let error = timeline_duration(ts(i64::MAX, 1), Speed::ONE).unwrap_err();
        assert_eq!(error, Error::DurationOverflow);
    }
}
