//! Time primitives: rational frame rates and frame-based ranges.
//!
//! Frame rates are stored as exact rationals so NTSC rates (23.976 = 24000/1001,
//! 29.97 = 30000/1001) survive round-trips without drift. Positions/durations
//! are integer frame counts; whether they are *timeline* frames or *source*
//! frames is determined by context (see [`Clip`](crate::Clip)).

/// An exact frame rate as `num/den` frames per second.
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

    /// Approximate frames-per-second as a float (for display/UI only).
    pub fn as_f64(self) -> f64 {
        if self.den == 0 {
            return 0.0;
        }
        f64::from(self.num) / f64::from(self.den)
    }

    /// Seconds-per-frame as a float.
    pub fn seconds_per_frame(self) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        f64::from(self.den) / f64::from(self.num)
    }

    pub fn is_valid(self) -> bool {
        self.num > 0 && self.den > 0
    }
}

/// A half-open range of frames `[start, start + duration)`.
///
/// `duration` is always treated as `>= 0`; a zero-duration range is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// First frame in the range.
    pub start: i64,
    /// Number of frames; `0` means empty.
    pub duration: i64,
}

impl TimeRange {
    pub const fn new(start: i64, duration: i64) -> Self {
        Self { start, duration }
    }

    /// Exclusive end frame (`start + duration`).
    pub const fn end(self) -> i64 {
        self.start + self.duration
    }

    pub const fn is_empty(self) -> bool {
        self.duration <= 0
    }

    /// True if `frame` lies within `[start, end)`.
    pub const fn contains(self, frame: i64) -> bool {
        frame >= self.start && frame < self.end()
    }

    /// True if the two ranges share at least one frame.
    pub const fn overlaps(self, other: TimeRange) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.start < other.end()
            && other.start < self.end()
    }

    /// The overlapping region of the two ranges, if any.
    pub fn intersection(self, other: TimeRange) -> Option<TimeRange> {
        if !self.overlaps(other) {
            return None;
        }
        let start = self.start.max(other.start);
        let end = self.end().min(other.end());
        Some(TimeRange::new(start, end - start))
    }
}

/// Convert a frame count from one frame rate to another, rounding to nearest.
///
/// `value` frames at rate `from` correspond to the returned frames at rate `to`.
/// Uses 128-bit intermediate math to avoid overflow on long durations.
pub fn convert_frames(value: i64, from: Rational, to: Rational) -> i64 {
    if !from.is_valid() || !to.is_valid() {
        return 0;
    }
    // value_to = value * (to / from) = value * to.num * from.den / (to.den * from.num)
    let numer = i128::from(value) * i128::from(to.num) * i128::from(from.den);
    let denom = i128::from(to.den) * i128::from(from.num);
    if denom == 0 {
        return 0;
    }
    let half = denom.abs() / 2;
    let magnitude = (numer.abs() + half) / denom.abs();
    let q = if (numer >= 0) == (denom >= 0) {
        magnitude
    } else {
        -magnitude
    };
    q as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntsc_rate_is_exact() {
        assert_eq!(Rational::FPS_23_976, Rational::new(24000, 1001));
        assert!((Rational::FPS_23_976.as_f64() - 23.976).abs() < 0.001);
    }

    #[test]
    fn range_contains_and_end() {
        let r = TimeRange::new(10, 5); // [10, 15)
        assert_eq!(r.end(), 15);
        assert!(r.contains(10));
        assert!(r.contains(14));
        assert!(!r.contains(15));
        assert!(!r.contains(9));
    }

    #[test]
    fn overlap_logic() {
        let a = TimeRange::new(0, 10); // [0,10)
        let b = TimeRange::new(10, 5); // [10,15) -> touching, not overlapping
        let c = TimeRange::new(5, 10); // [5,15) -> overlaps a
        assert!(!a.overlaps(b));
        assert!(a.overlaps(c));
        assert_eq!(a.intersection(c), Some(TimeRange::new(5, 5)));
        assert_eq!(a.intersection(b), None);
    }

    #[test]
    fn empty_range_never_overlaps() {
        let empty = TimeRange::new(5, 0);
        assert!(empty.is_empty());
        assert!(!empty.overlaps(TimeRange::new(0, 100)));
    }

    #[test]
    fn convert_frames_between_rates() {
        // 100 frames at 30fps == 80 frames at 24fps (same wall-clock duration).
        assert_eq!(
            convert_frames(100, Rational::FPS_30, Rational::FPS_24),
            80
        );
        // Identity.
        assert_eq!(convert_frames(50, Rational::FPS_24, Rational::FPS_24), 50);
        // 1001 frames @ 24000/1001 (~41.75s) -> 1002 frames @ 24 (same wall-clock).
        assert_eq!(
            convert_frames(1001, Rational::FPS_23_976, Rational::FPS_24),
            1002
        );
    }
}
