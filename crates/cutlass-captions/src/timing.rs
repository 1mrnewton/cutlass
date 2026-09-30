// --- Milliseconds ↔ timeline ticks ----------------------------------------------------

use cutlass_models::{Rational, TimeRange};

/// Where caption times land on the timeline.
///
/// Captions arrive in milliseconds — recognizers and subtitle files both speak
/// wall-clock — while the timeline is exact ticks at a rational rate. This is
/// the single place that conversion happens, so every caption source snaps the
/// same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// The rate the cue clips are placed at (the timeline's frame rate).
    pub rate: Rational,
    /// Timeline tick that caption time zero maps to. Non-zero when captioning
    /// one asset that starts partway into the sequence.
    pub offset_ticks: i64,
}

impl Placement {
    pub const fn new(rate: Rational, offset_ticks: i64) -> Self {
        Self { rate, offset_ticks }
    }

    /// Placement at the sequence start.
    pub const fn at_rate(rate: Rational) -> Self {
        Self::new(rate, 0)
    }

    /// The nearest tick to `ms`, excluding the offset.
    pub fn ticks(self, ms: u32) -> i64 {
        let denominator = 1000_i128 * i128::from(self.rate.den);
        if denominator <= 0 || self.rate.num <= 0 {
            return 0;
        }
        let numerator = i128::from(ms) * i128::from(self.rate.num);
        // Round half away from zero; `ms` is unsigned so this is half-up.
        let ticks = (2 * numerator + denominator) / (2 * denominator);
        i64::try_from(ticks).unwrap_or(i64::MAX)
    }

    /// The millisecond position of `ticks`, excluding the offset. Negative
    /// ticks clamp to zero: caption times never run before their own cue.
    pub fn ms(self, ticks: i64) -> u32 {
        if self.rate.num <= 0 || self.rate.den <= 0 || ticks <= 0 {
            return 0;
        }
        let numerator = i128::from(ticks) * 1000 * i128::from(self.rate.den);
        let denominator = i128::from(self.rate.num);
        let ms = (2 * numerator + denominator) / (2 * denominator);
        u32::try_from(ms).unwrap_or(u32::MAX)
    }
}

/// Pull each span's end back so the next one starts at least `min_gap_ms`
/// later.
///
/// Span *starts* are never moved: they come from real speech, or from a file
/// someone already timed. The hold on the previous line is the soft rule that
/// yields, which is why a fast exchange ends up with short lines rather than
/// lines that lag the audio.
pub(crate) fn separate_spans(spans: &mut [(u32, u32)], min_gap_ms: u32) {
    for index in 1..spans.len() {
        let latest_end = spans[index].0.saturating_sub(min_gap_ms);
        let previous = &mut spans[index - 1];
        if previous.1 > latest_end {
            previous.1 = latest_end.max(previous.0.saturating_add(1));
        }
    }
}

/// Snap millisecond spans to whole ticks, guaranteeing every range is at least
/// one tick long and no range starts before the previous one ends.
///
/// Rounding two nearby cues to the same frame is common at 24 fps with 40 ms
/// gaps, so collisions are resolved by giving the frame to the earlier cue and
/// pushing the later one forward — the alternative (shortening from the front)
/// would drop the first word of a line.
pub(crate) fn snap_spans(placement: Placement, spans: &[(u32, u32)]) -> Vec<TimeRange> {
    let mut ranges = Vec::with_capacity(spans.len());
    let mut floor = i64::MIN;
    for &(start_ms, end_ms) in spans {
        let mut start = placement.ticks(start_ms);
        if start < floor {
            start = floor;
        }
        let end = placement.ticks(end_ms.max(start_ms)).max(start + 1);
        floor = end;
        ranges.push(TimeRange::at_rate(
            start.saturating_add(placement.offset_ticks),
            end - start,
            placement.rate,
        ));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_round_to_the_nearest_frame() {
        let at_30 = Placement::at_rate(Rational::FPS_30);
        assert_eq!(at_30.ticks(0), 0);
        assert_eq!(at_30.ticks(33), 1, "33 ms is one frame at 30 fps");
        assert_eq!(at_30.ticks(1_000), 30);
        // 16 ms is just under half a frame, 17 ms just over.
        assert_eq!(at_30.ticks(16), 0);
        assert_eq!(at_30.ticks(17), 1);
    }

    #[test]
    fn ticks_stay_exact_at_ntsc_rates() {
        let at_23_976 = Placement::at_rate(Rational::FPS_23_976);
        // 1001 ms is exactly 24 ticks at 24000/1001.
        assert_eq!(at_23_976.ticks(1_001), 24);
        assert_eq!(at_23_976.ms(24), 1_001);
    }

    #[test]
    fn ms_and_ticks_round_trip_within_a_frame() {
        let placement = Placement::at_rate(Rational::FPS_60);
        for ms in [0_u32, 1, 500, 1_234, 60_000] {
            let back = placement.ms(placement.ticks(ms));
            assert!(back.abs_diff(ms) <= 17, "{ms} ms drifted to {back} ms");
        }
    }

    #[test]
    fn an_invalid_rate_degrades_instead_of_dividing_by_zero() {
        let broken = Placement::at_rate(Rational::new(0, 0));
        assert_eq!(broken.ticks(1_000), 0);
        assert_eq!(broken.ms(30), 0);
    }

    #[test]
    fn snapping_keeps_every_span_non_empty_and_ordered() {
        let placement = Placement::at_rate(Rational::FPS_24);
        // Two cues 5 ms apart both round to tick 0 at 24 fps.
        let ranges = snap_spans(placement, &[(0, 4), (5, 9), (1_000, 2_000)]);
        assert_eq!(ranges[0].start.value, 0);
        assert_eq!(ranges[0].duration.value, 1);
        assert_eq!(ranges[1].start.value, 1, "pushed clear of the first cue");
        assert_eq!(ranges[1].duration.value, 1);
        assert_eq!(ranges[2].start.value, 24);
    }

    #[test]
    fn separating_shortens_the_earlier_span_only() {
        let mut spans = [(0, 1_000), (800, 1_500)];
        separate_spans(&mut spans, 40);
        assert_eq!(spans[0], (0, 760));
        assert_eq!(spans[1], (800, 1_500), "starts are never moved");
    }

    #[test]
    fn separating_never_empties_a_span() {
        let mut spans = [(100, 900), (110, 900)];
        separate_spans(&mut spans, 500);
        assert_eq!(spans[0], (100, 101), "kept non-empty for snapping to fix");
    }

    #[test]
    fn snapping_applies_the_offset() {
        let placement = Placement::new(Rational::FPS_30, 90);
        let ranges = snap_spans(placement, &[(0, 1_000)]);
        assert_eq!(ranges[0].start.value, 90);
        assert_eq!(ranges[0].duration.value, 30);
    }
}
