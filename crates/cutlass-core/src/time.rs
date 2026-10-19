//! Time primitives aligned with OpenTimelineIO: exact [`Rational`] rates and
//! [`RationalTime`] positions/durations.
//!
//! A [`RationalTime`] is `value` ticks at `rate` ticks-per-second, so NTSC rates
//! (23.976 = 24000/1001) stay exact and a stream's PTS (`value` in its own
//! time base) is representable without rounding. [`TimeRange`] is half-open
//! `[start, start + duration)` with both endpoints carrying the same rate.

use core::cmp::Ordering;

/// Errors from rational-time arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TimeError {
    /// Two operands carried different rates where the same rate was required.
    #[error("rate mismatch: expected {expected:?}, got {got:?}")]
    RateMismatch { expected: Rational, got: Rational },
    /// Tick arithmetic overflowed `i64`.
    #[error("time arithmetic overflow")]
    Overflow,
}

/// An exact frame rate (or tick rate) as `num/den` units per second.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    pub const FPS_24: Rational = Rational::new(24, 1);
    pub const FPS_23_976: Rational = Rational::new(24000, 1001);
    pub const FPS_25: Rational = Rational::new(25, 1);
    pub const FPS_30: Rational = Rational::new(30, 1);
    pub const FPS_29_97: Rational = Rational::new(30000, 1001);
    pub const FPS_50: Rational = Rational::new(50, 1);
    pub const FPS_60: Rational = Rational::new(60, 1);
    pub const FPS_59_94: Rational = Rational::new(60000, 1001);

    pub const fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    /// Approximate units-per-second as a float (display/UI only).
    pub fn as_f64(self) -> f64 {
        if self.den == 0 {
            return 0.0;
        }
        f64::from(self.num) / f64::from(self.den)
    }

    /// Seconds-per-unit as a float.
    pub fn seconds_per_unit(self) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        f64::from(self.den) / f64::from(self.num)
    }

    pub fn is_valid(self) -> bool {
        self.num > 0 && self.den > 0
    }

    /// This rate in lowest terms, e.g. `600/25` → `24/1`.
    ///
    /// Rate equality in this crate is structural ([`rate_eq`]), but container
    /// timescales are arbitrary (a 24 fps file may report `600/25`). Reducing a
    /// probed rate before it enters the model lets it match the `FPS_*`
    /// constants and any source range authored at the canonical rate. Invalid
    /// rates (non-positive parts) are returned unchanged.
    pub fn reduced(self) -> Rational {
        if !self.is_valid() {
            return self;
        }
        let g = gcd(self.num, self.den);
        Rational::new(self.num / g, self.den / g)
    }
}

/// Greatest common divisor of two `i32`s (Euclid), never zero so it is safe to
/// divide by.
fn gcd(a: i32, b: i32) -> i32 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// A time position or duration as integer ticks at an exact rate.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RationalTime {
    pub value: i64,
    pub rate: Rational,
}

impl RationalTime {
    pub const fn new(value: i64, rate: Rational) -> Self {
        Self { value, rate }
    }

    pub fn zero(rate: Rational) -> Self {
        Self::new(0, rate)
    }

    /// Position in seconds as a float (display/UI; not for exact comparisons).
    pub fn seconds(self) -> f64 {
        if self.rate.num == 0 {
            return 0.0;
        }
        (self.value as f64) * f64::from(self.rate.den) / f64::from(self.rate.num)
    }

    /// Exact ordering against another time, **even at a different rate** — the
    /// primitive the scrub/playback path uses to ask "is this decoded frame at
    /// or after the requested timeline position?" without float drift.
    ///
    /// Compares `value · rate.den` cross-multiplied (rates are units/second with
    /// positive `num`), in `i128` so it never overflows for real media.
    pub fn compare(self, other: RationalTime) -> Ordering {
        let lhs = i128::from(self.value) * i128::from(self.rate.den) * i128::from(other.rate.num);
        let rhs = i128::from(other.value) * i128::from(other.rate.den) * i128::from(self.rate.num);
        lhs.cmp(&rhs)
    }
}

/// Half-open range `[start, start + duration)`.
///
/// `start` and `duration` must share the same `rate`; use [`TimeRange::at_rate`]
/// when constructing from tick counts at one rate.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    pub start: RationalTime,
    pub duration: RationalTime,
}

impl TimeRange {
    pub fn new(start: RationalTime, duration: RationalTime) -> Result<Self, TimeError> {
        check_same_rate(start.rate, duration.rate)?;
        Ok(Self { start, duration })
    }

    /// Build a range from tick counts at a single rate.
    pub fn at_rate(start: i64, duration: i64, rate: Rational) -> Self {
        Self {
            start: RationalTime::new(start, rate),
            duration: RationalTime::new(duration, rate),
        }
    }

    /// Exclusive end tick (`start + duration`) at the range's rate.
    pub fn end_tick(self) -> i64 {
        self.start.value + self.duration.value
    }

    /// Exclusive end as a [`RationalTime`].
    pub fn end(self) -> Result<RationalTime, TimeError> {
        time_add(&self.start, &self.duration)
    }

    pub fn is_empty(self) -> bool {
        self.duration.value <= 0
    }

    /// True if `t` lies within `[start, end)`.
    pub fn contains(self, t: RationalTime) -> Result<bool, TimeError> {
        check_same_rate(self.start.rate, t.rate)?;
        Ok(t.value >= self.start.value && t.value < self.end_tick())
    }

    /// True if the two ranges share at least one tick.
    pub fn overlaps(self, other: Self) -> Result<bool, TimeError> {
        check_same_rate(self.start.rate, other.start.rate)?;
        if self.is_empty() || other.is_empty() {
            return Ok(false);
        }
        Ok(self.start.value < other.end_tick() && other.start.value < self.end_tick())
    }

    /// The overlapping region, if any.
    pub fn intersection(self, other: Self) -> Result<Option<Self>, TimeError> {
        if !self.overlaps(other)? {
            return Ok(None);
        }
        let start = self.start.value.max(other.start.value);
        let end = self.end_tick().min(other.end_tick());
        Ok(Some(Self::at_rate(start, end - start, self.start.rate)))
    }
}

/// Structural equality on the rate pair (no gcd reduction).
pub fn rate_eq(a: Rational, b: Rational) -> bool {
    a.num == b.num && a.den == b.den
}

/// Errors when `actual` does not match `expected`.
pub fn check_same_rate(actual: Rational, expected: Rational) -> Result<(), TimeError> {
    if rate_eq(actual, expected) {
        Ok(())
    } else {
        Err(TimeError::RateMismatch {
            expected,
            got: actual,
        })
    }
}

/// Same-rate addition with checked overflow.
pub fn time_add(a: &RationalTime, b: &RationalTime) -> Result<RationalTime, TimeError> {
    check_same_rate(a.rate, b.rate)?;
    let value = a.value.checked_add(b.value).ok_or(TimeError::Overflow)?;
    Ok(RationalTime::new(value, a.rate))
}

/// Same-rate subtraction with checked overflow.
pub fn time_sub(a: &RationalTime, b: &RationalTime) -> Result<RationalTime, TimeError> {
    check_same_rate(a.rate, b.rate)?;
    let value = a.value.checked_sub(b.value).ok_or(TimeError::Overflow)?;
    Ok(RationalTime::new(value, a.rate))
}

/// Resample `time` to `to`, rounding to the nearest tick.
pub fn resample(time: RationalTime, to: Rational) -> RationalTime {
    if !time.rate.is_valid() || !to.is_valid() {
        return RationalTime::zero(to);
    }
    if rate_eq(time.rate, to) {
        return time;
    }
    let numer = i128::from(time.value) * i128::from(to.num) * i128::from(time.rate.den);
    let denom = i128::from(to.den) * i128::from(time.rate.num);
    if denom == 0 {
        return RationalTime::zero(to);
    }
    let half = denom.abs() / 2;
    let magnitude = (numer.abs() + half) / denom.abs();
    let q = if (numer >= 0) == (denom >= 0) {
        magnitude
    } else {
        -magnitude
    };
    RationalTime::new(q as i64, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    const R24: Rational = Rational::FPS_24;
    const R30: Rational = Rational::FPS_30;

    fn rt(value: i64, rate: Rational) -> RationalTime {
        RationalTime::new(value, rate)
    }

    #[test]
    fn rational_fps_constants_are_canonical() {
        assert_eq!(Rational::FPS_23_976, Rational::new(24000, 1001));
        assert_eq!(Rational::FPS_29_97, Rational::new(30000, 1001));
        assert_eq!(Rational::FPS_59_94, Rational::new(60000, 1001));
    }

    #[test]
    fn rational_as_f64_approximates_ntsc() {
        assert!((Rational::FPS_23_976.as_f64() - 23.976).abs() < 0.001);
        assert!((Rational::FPS_29_97.as_f64() - 29.97).abs() < 0.001);
    }

    #[test]
    fn rational_is_valid() {
        assert!(Rational::new(24, 1).is_valid());
        assert!(!Rational::new(0, 1).is_valid());
        assert!(!Rational::new(24, 0).is_valid());
        assert!(!Rational::new(-24, 1).is_valid());
    }

    #[test]
    fn rational_reduced_canonicalizes_container_timescales() {
        // A 24 fps file commonly reports its rate against a 600 timescale.
        assert_eq!(Rational::new(600, 25).reduced(), R24);
        assert_eq!(Rational::new(60, 2).reduced(), R30);
        assert_eq!(Rational::new(24000, 1000).reduced(), R24);
        // Already-canonical NTSC rates are coprime and survive untouched.
        assert_eq!(Rational::FPS_29_97.reduced(), Rational::FPS_29_97);
        assert_eq!(R24.reduced(), R24);
        // Invalid rates pass through rather than dividing by a zero gcd.
        assert_eq!(Rational::new(24, 0).reduced(), Rational::new(24, 0));
    }

    #[test]
    fn rational_time_seconds() {
        assert!((rt(48, R24).seconds() - 2.0).abs() < f64::EPSILON);
        // One 29.97 frame is value=1 at rate 30000/1001 -> exactly 1001/30000 s.
        assert!((rt(1, Rational::FPS_29_97).seconds() - 1001.0 / 30000.0).abs() < 1e-12);
    }

    #[test]
    fn compare_is_exact_across_rates() {
        // 1 second expressed two ways must compare equal.
        assert_eq!(rt(24, R24).compare(rt(30, R30)), Ordering::Equal);
        // One 24fps frame (1/24 s) is later than one 30fps frame (1/30 s).
        assert_eq!(rt(1, R24).compare(rt(1, R30)), Ordering::Greater);
        assert_eq!(rt(1, R30).compare(rt(1, R24)), Ordering::Less);
    }

    #[test]
    fn time_add_same_rate() {
        let sum = time_add(&rt(30, R30), &rt(45, R30)).unwrap();
        assert_eq!(sum.value, 75);
        assert_eq!(sum.rate, R30);
    }

    #[test]
    fn time_add_rate_mismatch_errors() {
        // time_add(a, b) checks check_same_rate(a.rate, b.rate): got = a.rate
        // (what was found), expected = b.rate.
        assert_eq!(
            time_add(&rt(1, R30), &rt(1, R24)),
            Err(TimeError::RateMismatch {
                expected: R24,
                got: R30,
            })
        );
    }

    #[test]
    fn time_add_overflow_is_checked() {
        assert_eq!(
            time_add(&rt(i64::MAX, R30), &rt(1, R30)),
            Err(TimeError::Overflow)
        );
    }

    #[test]
    fn resample_rounds_to_nearest_tick() {
        // 1 tick at 24fps -> nearest 30fps tick is 1 (1.25 rounds to 1).
        assert_eq!(resample(rt(1, R24), R30), rt(1, R30));
        // 2 ticks at 24fps (1/12 s) -> 2.5 at 30fps, rounds to 3 (half-up).
        assert_eq!(resample(rt(2, R24), R30), rt(3, R30));
        // Same rate is identity.
        assert_eq!(resample(rt(7, R30), R30), rt(7, R30));
    }

    #[test]
    fn resample_handles_negative_and_degenerate() {
        assert_eq!(resample(rt(-2, R24), R30).value, -3);
        assert_eq!(resample(rt(5, Rational::new(0, 1)), R30), rt(0, R30));
    }

    #[test]
    fn time_range_contains_and_overlaps() {
        let a = TimeRange::at_rate(0, 30, R30);
        assert!(a.contains(rt(0, R30)).unwrap());
        assert!(a.contains(rt(29, R30)).unwrap());
        assert!(!a.contains(rt(30, R30)).unwrap()); // half-open
        let b = TimeRange::at_rate(15, 30, R30);
        assert!(a.overlaps(b).unwrap());
        let c = TimeRange::at_rate(30, 30, R30);
        assert!(!a.overlaps(c).unwrap());
    }

    #[test]
    fn time_range_intersection() {
        let a = TimeRange::at_rate(0, 30, R30);
        let b = TimeRange::at_rate(15, 30, R30);
        assert_eq!(
            a.intersection(b).unwrap(),
            Some(TimeRange::at_rate(15, 15, R30))
        );
        let c = TimeRange::at_rate(100, 10, R30);
        assert_eq!(a.intersection(c).unwrap(), None);
    }
}
