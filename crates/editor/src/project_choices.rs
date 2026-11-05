//! The choices a project's settings offer: frame rates, color spaces and
//! canvas shapes, with their labels and the step to the next one.

use cutlass_model::FrameRate;
use cutlass_session::{CANVAS_SHAPES, CanvasShape};

/// The project frame rates, slowest first.
pub const PROJECT_RATES: [FrameRate; 9] = [
    FrameRate::FPS_23_976,
    FrameRate::FPS_24,
    FrameRate::FPS_25,
    FrameRate::FPS_29_97,
    FrameRate::FPS_30,
    FrameRate::FPS_50,
    FrameRate::FPS_59_94,
    FrameRate::FPS_60,
    FrameRate::FPS_120,
];

/// The stored color-space ints and their names.
pub const COLOR_SPACES: [(i32, &str); 5] = [
    (-1, "Unknown"),
    (0, "None"),
    (2, "SDR 2020"),
    (67, "HLG"),
    (2084, "PQ"),
];

/// The rate after `current`; an unknown rate steps to the first.
#[must_use]
pub fn next_rate(current: Option<FrameRate>) -> FrameRate {
    let index = current
        .and_then(|rate| PROJECT_RATES.iter().position(|item| *item == rate))
        .map_or(0, |index| index.saturating_add(1));
    PROJECT_RATES[index % PROJECT_RATES.len()]
}

/// Drop-frame timecode exists for 29.97 and 59.94.
#[must_use]
pub fn is_drop_rate(rate: FrameRate) -> bool {
    matches!(
        (rate.numerator, rate.denominator),
        (30_000, 1001) | (60_000, 1001)
    )
}

/// The frames per second the draft stores for `rate`.
#[must_use]
pub fn rate_fps(rate: FrameRate) -> f64 {
    f64::from(rate.numerator) / f64::from(rate.denominator)
}

/// `29.97 fps`, `30 fps`.
#[must_use]
pub fn rate_label(rate: FrameRate) -> String {
    let name = match (rate.numerator, rate.denominator) {
        (24_000, 1001) => "23.976",
        (30_000, 1001) => "29.97",
        (60_000, 1001) => "59.94",
        (numerator, 1) => return format!("{numerator} fps"),
        _ => "Frame rate",
    };
    format!("{name} fps")
}

/// The name of a stored color space; an int outside the list shows as itself.
#[must_use]
pub fn color_space_name(space: i32) -> String {
    COLOR_SPACES
        .iter()
        .find(|item| item.0 == space)
        .map_or_else(|| space.to_string(), |item| item.1.to_owned())
}

/// The color space after `current`; an unknown int steps to the first.
#[must_use]
pub fn next_color_space(current: i32) -> i32 {
    let index = COLOR_SPACES
        .iter()
        .position(|item| item.0 == current)
        .map_or(0, |index| index.saturating_add(1));
    COLOR_SPACES[index % COLOR_SPACES.len()].0
}

/// The canvas shape after the one this frame matches. A new project is
/// 1920×1080, so the next shape is 9:16.
#[must_use]
pub fn next_canvas(width: i32, height: i32) -> CanvasShape {
    let index = CANVAS_SHAPES
        .iter()
        .position(|shape| shape.width == width && shape.height == height)
        .map_or(0, |index| index.saturating_add(1));
    CANVAS_SHAPES[index % CANVAS_SHAPES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_step_and_wrap_and_label() {
        assert_eq!(next_rate(None), FrameRate::FPS_23_976);
        assert_eq!(next_rate(Some(FrameRate::FPS_29_97)), FrameRate::FPS_30);
        assert_eq!(next_rate(Some(FrameRate::FPS_120)), FrameRate::FPS_23_976);
        assert_eq!(rate_label(FrameRate::FPS_59_94), "59.94 fps");
        assert_eq!(rate_label(FrameRate::FPS_25), "25 fps");
        assert!(is_drop_rate(FrameRate::FPS_29_97));
        assert!(!is_drop_rate(FrameRate::FPS_23_976));
        assert!((rate_fps(FrameRate::FPS_29_97) - 29.97).abs() < 0.001);
    }

    #[test]
    fn color_spaces_and_canvas_step() {
        assert_eq!(color_space_name(67), "HLG");
        assert_eq!(color_space_name(9), "9");
        assert_eq!(next_color_space(2084), -1);
        assert_eq!(next_color_space(9), -1);
        let after = next_canvas(1920, 1080);
        assert_eq!((after.width, after.height), (1080, 1920));
    }
}
