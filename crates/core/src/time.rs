//! Media timestamps as exact fractions of a second.

use std::cmp::Ordering;
use std::num::NonZeroU32;

/// A point or span in media time: `value / timescale` seconds.
///
/// Containers count time in their own units (an MP4 track's timescale, 1/1000 s
/// in MKV), so a timestamp keeps the units it came with. Comparisons are exact
/// across timescales.
#[derive(Debug, Clone, Copy)]
pub struct Timestamp {
    pub value: i64,
    pub timescale: NonZeroU32,
}

impl Timestamp {
    pub const fn new(value: i64, timescale: NonZeroU32) -> Self {
        Self { value, timescale }
    }

    /// The time in seconds. Lossy, so use it for display and logs, not math.
    pub fn as_secs_f64(self) -> f64 {
        self.value as f64 / f64::from(self.timescale.get())
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Timestamp {}

impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = i128::from(self.value) * i128::from(other.timescale.get());
        let rhs = i128::from(other.value) * i128::from(self.timescale.get());
        lhs.cmp(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(value: i64, timescale: u32) -> Timestamp {
        Timestamp::new(value, NonZeroU32::new(timescale).unwrap())
    }

    #[test]
    fn same_instant_in_different_timescales_is_equal() {
        assert_eq!(ts(1001, 30000), ts(1001 * 3, 90000));
    }

    #[test]
    fn orders_across_timescales() {
        assert!(ts(1, 30) < ts(1, 24));
        assert!(ts(-1, 1000) < ts(0, 1));
    }

    #[test]
    fn extreme_values_compare_without_overflow() {
        assert!(ts(i64::MAX, 1) > ts(i64::MAX, u32::MAX));
        assert!(ts(i64::MIN, 1) < ts(i64::MIN, u32::MAX));
    }
}
