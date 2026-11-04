//! A timeline let go mid-swipe keeps sliding and slows to a stop.
//!
//! The speed falls by the same share each millisecond, as a phone's own
//! lists do, so a hard swipe travels far and a gentle one settles quickly.
//! Every shell steps the same coast, so the strip feels the same everywhere.

/// The share of its speed a coast keeps each millisecond.
pub const FLING_DECAY: f64 = 0.998;

/// Below this speed, in points a second, the coast stops.
pub const FLING_REST: f64 = 12.0;

/// A swipe slower than this, in points a second, does not coast at all.
pub const FLING_START: f64 = 80.0;

/// The fastest a coast starts, in points a second.
pub const FLING_LIMIT: f64 = 8_000.0;

/// One step of a coast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlingStep {
    /// Points moved this step, in the direction of the speed.
    pub distance: f64,
    /// The speed after the step, in points a second. Zero once at rest.
    pub velocity: f64,
}

/// The speed a coast starts at for a finger let go at `velocity` points a
/// second: held to [`FLING_LIMIT`], or zero when slower than
/// [`FLING_START`] or not a number.
#[must_use]
pub fn fling_start(velocity: f64) -> f64 {
    if !velocity.is_finite() || velocity.abs() < FLING_START {
        return 0.0;
    }
    velocity.clamp(-FLING_LIMIT, FLING_LIMIT)
}

/// Move a coast at `velocity` points a second on by `elapsed` seconds.
///
/// The distance is the exact integral of the decaying speed, so the coast
/// travels as far at 30 steps a second as at 120. A step that reaches
/// [`FLING_REST`] stops there. A bad speed or time stops at once.
#[must_use]
pub fn fling_step(velocity: f64, elapsed: f64) -> FlingStep {
    let rest = FlingStep {
        distance: 0.0,
        velocity: 0.0,
    };
    if !velocity.is_finite() || !elapsed.is_finite() || elapsed <= 0.0 {
        return rest;
    }
    if velocity.abs() < FLING_REST {
        return rest;
    }
    // Speed v·k^t with t in milliseconds; its integral over seconds.
    let rate = FLING_DECAY.ln() * 1_000.0;
    let kept = (rate * elapsed.min(1.0)).exp();
    let next = velocity * kept;
    let distance = (next - velocity) / rate;
    FlingStep {
        distance,
        velocity: if next.abs() < FLING_REST { 0.0 } else { next },
    }
}

#[cfg(test)]
mod tests {
    use super::{FLING_LIMIT, FLING_REST, fling_start, fling_step};

    /// Coast from `velocity` in steps of `step` seconds until at rest.
    fn coast(velocity: f64, step: f64) -> (f64, u32) {
        let mut velocity = fling_start(velocity);
        let (mut travelled, mut steps) = (0.0, 0);
        while velocity != 0.0 {
            let moved = fling_step(velocity, step);
            travelled += moved.distance;
            velocity = moved.velocity;
            steps += 1;
            assert!(steps < 10_000, "a coast always stops");
        }
        (travelled, steps)
    }

    #[test]
    fn a_coast_slows_and_stops_the_same_way_at_any_frame_rate() {
        let (at_60, steps) = coast(2_000.0, 1.0 / 60.0);
        let (at_120, _) = coast(2_000.0, 1.0 / 120.0);
        // About v / ln(1/k): a 2,000 point swipe slides about 1,000 points.
        assert!((990.0..1_000.0).contains(&at_60), "{at_60}");
        assert!((at_60 - at_120).abs() < 2.0);
        // It settles in a few seconds.
        assert!((120..200).contains(&steps), "{steps}");
        let (back, _) = coast(-2_000.0, 1.0 / 60.0);
        assert!((back + at_60).abs() < 1e-6, "left mirrors right");
    }

    #[test]
    fn a_slow_or_bad_swipe_does_not_coast_and_a_wild_one_is_held() {
        assert_eq!(fling_start(40.0), 0.0);
        assert_eq!(fling_start(f64::NAN), 0.0);
        assert_eq!(fling_start(f64::INFINITY), 0.0);
        assert_eq!(fling_start(90_000.0), FLING_LIMIT);
        assert_eq!(fling_start(-90_000.0), -FLING_LIMIT);
        assert_eq!(fling_step(f64::NAN, 0.016).velocity, 0.0);
        assert_eq!(fling_step(500.0, -1.0).distance, 0.0);
        assert_eq!(fling_step(500.0, f64::NAN).velocity, 0.0);
        assert_eq!(fling_step(FLING_REST / 2.0, 0.016).velocity, 0.0);
        // A long stall moves at most a second's worth.
        let stalled = fling_step(1_000.0, 60.0);
        assert!(stalled.distance > 0.0 && stalled.distance < 1_000.0);
        assert!(stalled.velocity < 150.0);
    }
}
