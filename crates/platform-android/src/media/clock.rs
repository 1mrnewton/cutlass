//! MediaCodec counts time in microseconds; a packet in its stream's ticks.
//!
//! [`Clock`] remembers which tick each microsecond value came from, so a
//! picture leaves with the exact time its packet had, in any order, and
//! falls back to rounding when the codec hands back a time it was not given.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

use std::collections::BTreeMap;

const MICROS: i128 = 1_000_000;

/// Times waiting for their picture. Past this many the oldest are dropped,
/// so a codec that swallows units does not grow the map.
const MAX_WAITING: usize = 512;

/// `ticks` at `timescale` in microseconds, rounded half away from zero.
pub fn to_micros(ticks: i64, timescale: u32) -> i64 {
    let scaled = i128::from(ticks) * MICROS;
    clamp(round_div(scaled, i128::from(timescale.max(1))))
}

/// Microseconds back to ticks at `timescale`, rounded half away from zero.
pub fn from_micros(micros: i64, timescale: u32) -> i64 {
    let scaled = i128::from(micros) * i128::from(timescale.max(1));
    clamp(round_div(scaled, MICROS))
}

fn clamp(value: i128) -> i64 {
    i64::try_from(value).unwrap_or(if value < 0 { i64::MIN } else { i64::MAX })
}

fn round_div(value: i128, divisor: i128) -> i128 {
    let half = divisor / 2;
    if value >= 0 {
        (value + half) / divisor
    } else {
        (value - half) / divisor
    }
}

/// The ticks behind each microsecond value handed to a codec.
#[derive(Debug, Default)]
pub struct Clock<T> {
    waiting: BTreeMap<i64, Vec<T>>,
    count: usize,
}

impl<T> Clock<T> {
    pub fn new() -> Self {
        Self {
            waiting: BTreeMap::new(),
            count: 0,
        }
    }

    pub fn insert(&mut self, micros: i64, value: T) {
        self.waiting.entry(micros).or_default().push(value);
        self.count += 1;
        while self.count > MAX_WAITING {
            let Some(mut oldest) = self.waiting.first_entry() else {
                break;
            };
            oldest.get_mut().remove(0);
            if oldest.get().is_empty() {
                oldest.remove();
            }
            self.count -= 1;
        }
    }

    /// The value handed in at `micros`, first in first out. `None` when the
    /// codec returned a time it was never given.
    pub fn take(&mut self, micros: i64) -> Option<T> {
        let mut entry = match self.waiting.entry(micros) {
            std::collections::btree_map::Entry::Occupied(entry) => entry,
            std::collections::btree_map::Entry::Vacant(_) => return None,
        };
        let value = entry.get_mut().remove(0);
        if entry.get().is_empty() {
            entry.remove();
        }
        self.count -= 1;
        Some(value)
    }

    pub fn clear(&mut self) {
        self.waiting.clear();
        self.count = 0;
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.count
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microseconds_round_trip_at_common_timescales() {
        for timescale in [1_000, 24_000, 30_000, 44_100, 90_000, 600] {
            for ticks in [0_i64, 1, 1001, 3003, 12_345_678] {
                let micros = to_micros(ticks, timescale);
                let back = from_micros(micros, timescale);
                assert!(
                    (back - ticks).abs() <= 1,
                    "{ticks} at {timescale} came back {back}"
                );
            }
        }
        assert_eq!(to_micros(1, 0), 1_000_000, "a zero timescale reads as one");
        assert_eq!(to_micros(i64::MAX, 1), i64::MAX, "overflow saturates");
        assert_eq!(to_micros(-3, 2), -1_500_000);
    }

    #[test]
    fn a_picture_takes_back_its_own_ticks_in_any_order() {
        let mut clock = Clock::new();
        for (micros, ticks) in [(0, 0_i64), (33_367, 1001), (66_733, 2002)] {
            clock.insert(micros, ticks);
        }
        assert_eq!(clock.take(66_733), Some(2002));
        assert_eq!(clock.take(0), Some(0));
        assert_eq!(clock.take(5), None, "a time never handed in");
        assert_eq!(clock.take(33_367), Some(1001));
        assert!(clock.is_empty());
    }

    #[test]
    fn repeated_times_leave_first_in_first_out_and_the_map_stays_bounded() {
        let mut clock = Clock::new();
        clock.insert(10, 'a');
        clock.insert(10, 'b');
        assert_eq!(clock.take(10), Some('a'));
        assert_eq!(clock.take(10), Some('b'));
        let mut clock = Clock::new();
        for micros in 0..(MAX_WAITING as i64 + 10) {
            clock.insert(micros, micros);
        }
        assert_eq!(clock.len(), MAX_WAITING);
        assert_eq!(clock.take(0), None, "the oldest were dropped");
        clock.clear();
        assert!(clock.is_empty());
    }
}
