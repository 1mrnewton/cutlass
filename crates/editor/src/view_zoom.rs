//! The preview's zoom (fit, or a step of canvas pixels to screen pixels)
//! and its safe-area guides.

/// The zoom steps, as screen pixels per canvas pixel.
pub const PREVIEW_ZOOMS: [f64; 9] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];

/// The step after `scale` toward bigger (`up`) or smaller, from the scale
/// the canvas shows at now, fitted or not. `None` past the last step.
#[must_use]
pub fn preview_zoom_step(scale: f64, up: bool) -> Option<f64> {
    // A hair of slack, so a step lands on the next step and not on itself.
    let slack = 1e-6;
    if up {
        PREVIEW_ZOOMS
            .iter()
            .copied()
            .find(|step| *step > scale + slack)
    } else {
        PREVIEW_ZOOMS
            .iter()
            .rev()
            .copied()
            .find(|step| *step < scale - slack)
    }
}

/// The share of the canvas action is kept inside.
pub const ACTION_SAFE: f64 = 0.9;
/// The share of the canvas titles are kept inside.
pub const TITLE_SAFE: f64 = 0.8;

/// The box, centred, that holds `share` of a canvas `width` by `height`
/// on each side: its left, top, width and height in the same units.
#[must_use]
pub fn safe_box(width: f64, height: f64, share: f64) -> (f64, f64, f64, f64) {
    let (inner_width, inner_height) = (width * share, height * share);
    (
        (width - inner_width) / 2.0,
        (height - inner_height) / 2.0,
        inner_width,
        inner_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_safe_box_is_centred_and_takes_its_share_of_each_side() {
        assert_eq!(
            safe_box(1920.0, 1080.0, TITLE_SAFE),
            (192.0, 108.0, 1536.0, 864.0)
        );
        let (left, top, width, height) = safe_box(1080.0, 1920.0, ACTION_SAFE);
        assert!((left - 54.0).abs() < 1e-9 && (top - 96.0).abs() < 1e-9);
        assert!((width - 972.0).abs() < 1e-9 && (height - 1728.0).abs() < 1e-9);
    }

    #[test]
    fn a_step_goes_to_the_next_size_from_wherever_the_fit_left_it() {
        assert_eq!(preview_zoom_step(0.37, true), Some(0.5));
        assert_eq!(preview_zoom_step(0.37, false), Some(0.25));
        assert_eq!(preview_zoom_step(1.0, true), Some(1.5));
        assert_eq!(preview_zoom_step(1.0, false), Some(0.75));
        assert_eq!(preview_zoom_step(4.0, true), None);
        assert_eq!(preview_zoom_step(0.1, false), None);
    }
}
