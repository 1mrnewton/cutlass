//! The curves behind the volume slider and the speed scale.
//!
//! A position is a fraction of the control, 0 at the left and 1 at the
//! right. Volume puts unity in the middle: the left half runs from silence
//! to unity and the right half from unity to ten times. Speed puts its marks
//! the same distance apart and runs straight between two marks.

/// Highest gain the volume slider reaches: 1000%.
pub const MAX_VOLUME: f64 = 10.0;

/// The speed marks, left to right, the same distance apart.
pub const SPEED_MARKS: [f64; 6] = [0.1, 1.0, 2.0, 5.0, 10.0, 100.0];

/// The gain at slider `position`. A position outside 0 to 1, or one that is
/// not finite, is held at the nearer end.
#[must_use]
pub fn volume_at(position: f64) -> f64 {
    let position = fraction(position);
    if position <= 0.5 {
        position * 2.0
    } else {
        1.0 + (position - 0.5) * 2.0 * (MAX_VOLUME - 1.0)
    }
}

/// The slider position that shows `volume`. A gain above the top shows at
/// the right end; a negative or unfinite one at the left.
#[must_use]
pub fn volume_position(volume: f64) -> f64 {
    if !volume.is_finite() || volume <= 0.0 {
        return 0.0;
    }
    if volume <= 1.0 {
        volume / 2.0
    } else {
        (0.5 + (volume - 1.0) / (2.0 * (MAX_VOLUME - 1.0))).min(1.0)
    }
}

/// The speed at scale `position`, to two decimals. A position outside 0 to
/// 1, or one that is not finite, is held at the nearer end.
#[must_use]
pub fn speed_at(position: f64) -> f64 {
    let gaps = (SPEED_MARKS.len() - 1) as f64;
    let along = fraction(position) * gaps;
    let index = (along.floor() as usize).min(SPEED_MARKS.len() - 2);
    let (low, high) = (SPEED_MARKS[index], SPEED_MARKS[index + 1]);
    let speed = low + (high - low) * (along - index as f64);
    (speed * 100.0).round() / 100.0
}

/// The scale position that shows `speed`. A speed past either end shows at
/// that end.
#[must_use]
pub fn speed_position(speed: f64) -> f64 {
    let gaps = (SPEED_MARKS.len() - 1) as f64;
    if !speed.is_finite() || speed <= SPEED_MARKS[0] {
        return 0.0;
    }
    for (index, pair) in SPEED_MARKS.windows(2).enumerate() {
        if speed <= pair[1] {
            return (index as f64 + (speed - pair[0]) / (pair[1] - pair[0])) / gaps;
        }
    }
    1.0
}

fn fraction(position: f64) -> f64 {
    if position.is_nan() {
        0.0
    } else {
        position.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_sits_in_the_middle_of_the_volume_slider() {
        assert_eq!(volume_at(0.0), 0.0);
        assert_eq!(volume_at(0.25), 0.5);
        assert_eq!(volume_at(0.5), 1.0);
        assert_eq!(volume_at(1.0), MAX_VOLUME);
        assert_eq!(volume_position(1.0), 0.5);
        assert_eq!(volume_position(MAX_VOLUME), 1.0);
        for position in [0.0, 0.1, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0] {
            let back = volume_position(volume_at(position));
            assert!((back - position).abs() < 1e-12, "{position} -> {back}");
        }
    }

    #[test]
    fn the_volume_slider_holds_bad_input_at_its_ends() {
        assert_eq!(volume_at(-3.0), 0.0);
        assert_eq!(volume_at(7.0), MAX_VOLUME);
        assert_eq!(volume_at(f64::NAN), 0.0);
        assert_eq!(volume_at(f64::INFINITY), MAX_VOLUME);
        assert_eq!(volume_position(-1.0), 0.0);
        assert_eq!(volume_position(f64::NAN), 0.0);
        assert_eq!(volume_position(50.0), 1.0);
    }

    #[test]
    fn speed_marks_sit_the_same_distance_apart() {
        for (index, mark) in SPEED_MARKS.iter().enumerate() {
            let position = index as f64 / 5.0;
            assert_eq!(speed_at(position), *mark);
            assert!((speed_position(*mark) - position).abs() < 1e-12);
        }
        // Halfway from 1x to 2x is 1.5x.
        assert_eq!(speed_at(0.3), 1.5);
        assert!((speed_position(1.5) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn the_speed_scale_holds_bad_input_at_its_ends_and_stays_positive() {
        assert_eq!(speed_at(-1.0), 0.1);
        assert_eq!(speed_at(2.0), 100.0);
        assert_eq!(speed_at(f64::NAN), 0.1);
        assert_eq!(speed_position(0.0), 0.0);
        assert_eq!(speed_position(-4.0), 0.0);
        assert_eq!(speed_position(f64::NAN), 0.0);
        assert_eq!(speed_position(400.0), 1.0);
        for step in 0..=100 {
            assert!(speed_at(f64::from(step) / 100.0) > 0.0);
        }
    }
}
