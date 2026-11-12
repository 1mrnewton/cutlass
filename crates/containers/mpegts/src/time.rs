//! The 33-bit 90 kHz clock of PTS and DTS, and the 27 MHz PCR, unwrapped
//! onto one line (ITU-T H.222.0 clause 2.4.3.7 and 2.4.2.2).
//!
//! A timestamp wraps every 2^33 ticks, about 26.5 hours. Each one read is
//! placed on the copy of the 33-bit circle nearest a reference: the last
//! timestamp of the same program. Jumps of less than half the circle,
//! about 13.25 hours, either way come out right.

/// Ticks in one turn of the 33-bit clock.
pub const WRAP: i64 = 1 << 33;

/// The value congruent to `raw` modulo 2^33 that is nearest `reference`.
#[must_use]
pub const fn unwrap(raw: u64, reference: i64) -> i64 {
    let raw = (raw & (WRAP as u64 - 1)) as i64;
    let base = reference.div_euclid(WRAP) * WRAP + raw;
    let mut best = base;
    let candidates = [base - WRAP, base + WRAP];
    let mut index = 0;
    while index < candidates.len() {
        let candidate = candidates[index];
        if candidate.abs_diff(reference) < best.abs_diff(reference) {
            best = candidate;
        }
        index += 1;
    }
    best
}

/// A program's running reference for unwrapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unwrapper {
    reference: Option<i64>,
}

impl Unwrapper {
    /// Unwraps `raw` and moves the reference to it.
    pub fn place(&mut self, raw: u64) -> i64 {
        let value = match self.reference {
            Some(reference) => unwrap(raw, reference),
            None => (raw & (WRAP as u64 - 1)) as i64,
        };
        self.reference = Some(value);
        value
    }

    /// Unwraps `raw` near the reference without moving it.
    #[must_use]
    pub fn peek(&self, raw: u64) -> i64 {
        match self.reference {
            Some(reference) => unwrap(raw, reference),
            None => (raw & (WRAP as u64 - 1)) as i64,
        }
    }

    /// Sets the reference, after a seek.
    pub const fn set(&mut self, reference: Option<i64>) {
        self.reference = reference;
    }

    #[must_use]
    pub const fn reference(&self) -> Option<i64> {
        self.reference
    }
}

#[cfg(test)]
mod tests {
    use super::{Unwrapper, WRAP, unwrap};

    #[test]
    fn a_timestamp_lands_on_the_nearest_turn() {
        assert_eq!(unwrap(100, 0), 100);
        // Just after the wrap, read near the end of the first turn.
        assert_eq!(unwrap(10, WRAP - 50), WRAP + 10);
        // Just before the wrap, read after it: a B frame shown earlier.
        assert_eq!(unwrap((WRAP - 20) as u64, WRAP + 5), WRAP - 20);
        // Far into later turns.
        assert_eq!(unwrap(7, 5 * WRAP + 3), 5 * WRAP + 7);
        // Only 33 bits count.
        assert_eq!(unwrap((1 << 40) | 9, 0), 9);
    }

    #[test]
    fn a_run_across_two_wraps_keeps_rising() {
        let mut clock = Unwrapper::default();
        let mut raw = (WRAP - 3 * 3600) as u64;
        let mut last = clock.place(raw);
        for _ in 0..(2 * WRAP / 3600 / 1000) {
            raw = (raw + 3600 * 1000) & (WRAP as u64 - 1);
            let value = clock.place(raw);
            assert_eq!(value - last, 3600 * 1000);
            last = value;
        }
        assert!(last > 2 * WRAP);
    }
}
