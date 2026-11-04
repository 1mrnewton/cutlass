//! The preview's zoom: fit, or a step of canvas pixels to screen pixels.

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

#[cfg(test)]
mod tests {
    use super::*;

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
