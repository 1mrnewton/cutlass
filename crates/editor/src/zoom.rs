use cutlass_model::{FrameRate, Micros};

/// Pixels the two-frame accuracy step occupies.
///
/// Pixel size is pixels per microsecond. The last accuracy entry is a
/// two-frame step, so its pixel size is this width divided by two frames
/// of the draft rate.
const TWO_FRAME_STEP_PIXELS: f64 = 400.0;

/// The lane width and the cut length a zoom is clamped against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomBounds {
    /// Width of the lane, in pixels.
    pub lane_width: f64,
    /// Cut length, in microseconds. Zero has no quarter-lane term.
    pub duration: Micros,
    pub rate: FrameRate,
}

/// Timeline zoom. The value is a pixel size, in pixels per microsecond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zoom {
    pixel_size: f64,
}

impl Zoom {
    /// A finite pixel size greater than zero.
    #[must_use]
    pub fn new(pixel_size: f64) -> Option<Self> {
        if pixel_size.is_finite() && pixel_size > 0.0 {
            Some(Self { pixel_size })
        } else {
            None
        }
    }

    #[must_use]
    pub fn pixel_size(self) -> f64 {
        self.pixel_size
    }

    /// Double the pixel size, then clamp it to the lane.
    #[must_use]
    pub fn zoom_in(self, bounds: ZoomBounds) -> Option<Self> {
        Self::apply(self.pixel_size * 2.0, bounds)
    }

    /// Halve the pixel size, then clamp it to the lane.
    #[must_use]
    pub fn zoom_out(self, bounds: ZoomBounds) -> Option<Self> {
        Self::apply(self.pixel_size * 0.5, bounds)
    }

    /// Fit the cut to three quarters of the lane, then clamp it.
    #[must_use]
    pub fn zoom_to_fit(self, bounds: ZoomBounds) -> Option<Self> {
        let requested = match fit_pixel_size(bounds.lane_width, bounds.duration) {
            Some(size) => size,
            None if bounds.duration.0 <= 0 => last_accuracy_step(bounds.rate)?,
            None => return None,
        };
        Self::apply(requested, bounds)
    }

    fn apply(requested: f64, bounds: ZoomBounds) -> Option<Self> {
        let size = clamp_pixel_size(requested, bounds)?;
        Some(Self { pixel_size: size })
    }
}

/// The last accuracy step: two frames of `rate` across [`TWO_FRAME_STEP_PIXELS`].
#[must_use]
pub fn last_accuracy_step(rate: FrameRate) -> Option<f64> {
    if rate.numerator == 0 || rate.denominator == 0 {
        return None;
    }
    let step = TWO_FRAME_STEP_PIXELS * f64::from(rate.numerator)
        / (2.0 * 1_000_000.0 * f64::from(rate.denominator));
    finite_positive(step)
}

/// A quarter of the lane divided by the cut length.
///
/// The length is microseconds, the same clock as the pixel size.
#[must_use]
pub fn quarter_lane(lane_width: f64, duration: Micros) -> Option<f64> {
    if duration.0 <= 0 {
        return None;
    }
    scale_lane(lane_width, duration, 0.25)
}

/// Three quarters of the lane divided by the cut length.
#[must_use]
pub fn fit_pixel_size(lane_width: f64, duration: Micros) -> Option<f64> {
    if duration.0 <= 0 {
        return None;
    }
    scale_lane(lane_width, duration, 0.75)
}

/// The smaller of a quarter of the lane over the duration and the last accuracy step.
///
/// A duration of zero has no quarter term, so the floor is the accuracy step.
#[must_use]
pub fn minimum_pixel_size(bounds: ZoomBounds) -> Option<f64> {
    let step = last_accuracy_step(bounds.rate)?;
    if !bounds.lane_width.is_finite() || bounds.lane_width <= 0.0 {
        return None;
    }
    match quarter_lane(bounds.lane_width, bounds.duration) {
        Some(quarter) => finite_positive(quarter.min(step)),
        None => finite_positive(step),
    }
}

fn scale_lane(lane_width: f64, duration: Micros, fraction: f64) -> Option<f64> {
    if !lane_width.is_finite() || lane_width <= 0.0 {
        return None;
    }
    let size = fraction * lane_width / duration_pixels(duration)?;
    finite_positive(size)
}

fn duration_pixels(duration: Micros) -> Option<f64> {
    if duration.0 <= 0 {
        return None;
    }
    let length = duration.0 as f64;
    if length.is_finite() && length > 0.0 {
        Some(length)
    } else {
        None
    }
}

fn clamp_pixel_size(requested: f64, bounds: ZoomBounds) -> Option<f64> {
    let low = minimum_pixel_size(bounds)?;
    let high = last_accuracy_step(bounds.rate)?;
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    finite_positive(clamp_range(requested, low, high))
}

fn clamp_range(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

fn finite_positive(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(lane_width: f64, duration: i64) -> ZoomBounds {
        ZoomBounds {
            lane_width,
            duration: Micros(duration),
            rate: FrameRate::FPS_30,
        }
    }

    #[test]
    fn zoom_in_doubles_and_zoom_out_halves() {
        let limits = bounds(800.0, 10_000_000);
        let start = minimum_pixel_size(limits).unwrap();
        let zoom = Zoom::new(start).unwrap();
        let closer = zoom.zoom_in(limits).unwrap();
        assert_eq!(closer.pixel_size(), start * 2.0);
        let back = closer.zoom_out(limits).unwrap();
        assert_eq!(back.pixel_size(), start);
    }

    #[test]
    fn zoom_to_fit_uses_three_quarters_of_the_lane() {
        let limits = bounds(800.0, 10_000_000);
        let fitted = Zoom::new(1.0)
            .unwrap()
            .zoom_to_fit(limits)
            .unwrap()
            .pixel_size();
        assert_eq!(fitted, 0.75 * 800.0 / 10_000_000.0);
        assert!(fitted > minimum_pixel_size(limits).unwrap());
        assert!(fitted < last_accuracy_step(FrameRate::FPS_30).unwrap());
    }

    #[test]
    fn zoom_out_stops_at_the_minimum() {
        let limits = bounds(800.0, 10_000_000);
        let floor = minimum_pixel_size(limits).unwrap();
        let quarter = quarter_lane(800.0, Micros(10_000_000)).unwrap();
        let step = last_accuracy_step(FrameRate::FPS_30).unwrap();
        assert_eq!(floor, quarter.min(step));
        assert!(quarter < step);

        let mut zoom = Zoom::new(floor * 4.0).unwrap();
        zoom = zoom.zoom_out(limits).unwrap();
        assert_eq!(zoom.pixel_size(), floor * 2.0);
        zoom = zoom.zoom_out(limits).unwrap();
        assert_eq!(zoom.pixel_size(), floor);
        zoom = zoom.zoom_out(limits).unwrap();
        assert_eq!(zoom.pixel_size(), floor);
    }

    #[test]
    fn a_short_cut_floors_and_fits_at_the_accuracy_step() {
        let limits = bounds(1_000.0, 1_000);
        let step = last_accuracy_step(FrameRate::FPS_30).unwrap();
        let quarter = quarter_lane(1_000.0, Micros(1_000)).unwrap();
        assert!(quarter > step);
        assert_eq!(minimum_pixel_size(limits).unwrap(), step);
        let zoom = Zoom::new(step).unwrap();
        assert_eq!(zoom.zoom_out(limits).unwrap().pixel_size(), step);
        assert_eq!(zoom.zoom_in(limits).unwrap().pixel_size(), step);
        assert_eq!(zoom.zoom_to_fit(limits).unwrap().pixel_size(), step);
    }

    #[test]
    fn an_empty_cut_uses_the_accuracy_step() {
        let limits = bounds(800.0, 0);
        let step = last_accuracy_step(FrameRate::FPS_30).unwrap();
        assert_eq!(minimum_pixel_size(limits).unwrap(), step);
        assert!(quarter_lane(800.0, Micros(0)).is_none());
        assert!(fit_pixel_size(800.0, Micros(0)).is_none());
        let zoom = Zoom::new(step * 0.5).unwrap();
        assert_eq!(zoom.zoom_to_fit(limits).unwrap().pixel_size(), step);
    }

    #[test]
    fn a_new_duration_does_not_move_the_stored_pixel_size() {
        let short = bounds(800.0, 1_000_000);
        let zoom = Zoom::new(minimum_pixel_size(short).unwrap())
            .unwrap()
            .zoom_in(short)
            .unwrap();
        let stored = zoom.pixel_size();
        let longer = bounds(800.0, 20_000_000);
        assert!(minimum_pixel_size(longer).unwrap() < stored);
        assert_eq!(zoom.pixel_size(), stored);
    }

    #[test]
    fn the_two_frame_step_follows_the_rate() {
        let thirty = last_accuracy_step(FrameRate::FPS_30).unwrap();
        let sixty = last_accuracy_step(FrameRate::FPS_60).unwrap();
        assert_eq!(sixty, thirty * 2.0);
        assert_eq!(thirty, 400.0 * 30.0 / (2.0 * 1_000_000.0));
        assert!(
            last_accuracy_step(FrameRate {
                numerator: 0,
                denominator: 1
            })
            .is_none()
        );
        assert!(Zoom::new(0.0).is_none());
        assert!(Zoom::new(f64::NAN).is_none());
        assert!(minimum_pixel_size(bounds(0.0, 1_000_000)).is_none());
    }
}
