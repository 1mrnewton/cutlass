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

/// The smallest and biggest zoom a wheel or a pinch reaches, the same as
/// the first and last step.
pub const PREVIEW_ZOOM_RANGE: (f64, f64) =
    (PREVIEW_ZOOMS[0], PREVIEW_ZOOMS[PREVIEW_ZOOMS.len() - 1]);

/// A preview zoom by `factor` (above one bigger) about `anchor`, from
/// `scale` with the canvas moved by `pan`. `anchor` and `pan` are screen
/// points from the middle of the preview, so the canvas spot under the
/// anchor stays under it. The zoom is held to [`PREVIEW_ZOOM_RANGE`].
/// Answers the new zoom and pan, or `None` when nothing changes: a factor
/// of one, a factor or a scale that is not finite and above zero, or a
/// zoom already at the end it is pushed toward.
#[must_use]
pub fn preview_zoom_at(
    scale: f64,
    factor: f64,
    anchor: (f64, f64),
    pan: (f64, f64),
) -> Option<(f64, (f64, f64))> {
    if !(scale.is_finite() && scale > 0.0 && factor.is_finite() && factor > 0.0) {
        return None;
    }
    let (low, high) = PREVIEW_ZOOM_RANGE;
    let zoom = (scale * factor).clamp(low, high);
    if (zoom - scale).abs() <= f64::EPSILON * scale {
        return None;
    }
    // The canvas middle sits at `pan`; the anchor's distance from it grows
    // with the zoom, so the middle moves away from the anchor by as much.
    let grow = zoom / scale;
    let pan = (
        anchor.0 - (anchor.0 - pan.0) * grow,
        anchor.1 - (anchor.1 - pan.1) * grow,
    );
    Some((zoom, pan))
}

/// `pan` held to what keeps a canvas `shown` big over the middle of a
/// preview `frame` big: a canvas no bigger than the preview on a side
/// stays centred on that side. Every size is in screen points.
#[must_use]
pub fn clamp_preview_pan(pan: (f64, f64), shown: (f64, f64), frame: (f64, f64)) -> (f64, f64) {
    let room = |shown: f64, frame: f64| ((shown - frame) / 2.0).max(0.0);
    let (x, y) = (room(shown.0, frame.0), room(shown.1, frame.1));
    let hold = |value: f64, room: f64| {
        if value.is_finite() {
            value.clamp(-room, room)
        } else {
            0.0
        }
    };
    (hold(pan.0, x), hold(pan.1, y))
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

    fn near(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
    }

    #[test]
    fn a_zoom_keeps_the_canvas_spot_under_the_anchor() {
        let (scale, pan, anchor) = (0.5, (10.0, -20.0), (120.0, 80.0));
        let (zoom, moved) = preview_zoom_at(scale, 2.0, anchor, pan).expect("zooms");
        assert!((zoom - 1.0).abs() < 1e-12);
        // The canvas pixel under the anchor before and after.
        let before = ((anchor.0 - pan.0) / scale, (anchor.1 - pan.1) / scale);
        let after = ((anchor.0 - moved.0) / zoom, (anchor.1 - moved.1) / zoom);
        assert!(near(before, after));
        // About the middle, the pan only scales.
        let (_, about_middle) = preview_zoom_at(1.0, 1.5, (0.0, 0.0), (4.0, 6.0)).expect("zooms");
        assert!(near(about_middle, (6.0, 9.0)));
    }

    #[test]
    fn a_zoom_is_held_to_the_range_and_a_pinned_end_is_no_change() {
        let (high, _) = preview_zoom_at(3.0, 10.0, (0.0, 0.0), (0.0, 0.0)).expect("zooms");
        assert!((high - PREVIEW_ZOOM_RANGE.1).abs() < 1e-12);
        let (low, _) = preview_zoom_at(0.2, 0.01, (0.0, 0.0), (0.0, 0.0)).expect("zooms");
        assert!((low - PREVIEW_ZOOM_RANGE.0).abs() < 1e-12);
        assert_eq!(
            preview_zoom_at(PREVIEW_ZOOM_RANGE.1, 1.2, (5.0, 5.0), (0.0, 0.0)),
            None
        );
        assert_eq!(
            preview_zoom_at(PREVIEW_ZOOM_RANGE.0, 0.8, (5.0, 5.0), (0.0, 0.0)),
            None
        );
    }

    #[test]
    fn a_zoom_refuses_a_factor_or_scale_that_is_not_a_size() {
        for (scale, factor) in [
            (1.0, 1.0),
            (1.0, 0.0),
            (1.0, -2.0),
            (1.0, f64::NAN),
            (1.0, f64::INFINITY),
            (0.0, 2.0),
            (f64::NAN, 2.0),
        ] {
            assert_eq!(preview_zoom_at(scale, factor, (1.0, 1.0), (0.0, 0.0)), None);
        }
    }

    #[test]
    fn a_pan_keeps_the_canvas_over_the_middle() {
        // 1000 wide in an 800 frame has 100 of room each way; 300 tall in a
        // 600 frame has none.
        assert_eq!(
            clamp_preview_pan((250.0, 40.0), (1000.0, 300.0), (800.0, 600.0)),
            (100.0, 0.0)
        );
        assert_eq!(
            clamp_preview_pan((-50.0, 0.0), (1000.0, 300.0), (800.0, 600.0)),
            (-50.0, 0.0)
        );
        assert_eq!(
            clamp_preview_pan((f64::NAN, 1.0), (1000.0, 900.0), (800.0, 600.0)),
            (0.0, 1.0)
        );
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
