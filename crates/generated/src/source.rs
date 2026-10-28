use cutlass_model::Micros;
use cutlass_picture::{Picture, PictureSource, Sample, rgba_len};

use crate::color::{FillError, fill};

/// A solid color over a half-open microsecond span.
///
/// `sample` is ready inside `[start, start + duration)` and missing outside
/// it. A non-positive duration, or an end that does not fit in `i64`, is
/// always a miss. The bytes come from [`fill`](crate::fill).
#[derive(Debug)]
pub struct SolidColor {
    width: u32,
    height: u32,
    packed: u32,
    span: Option<(i64, i64)>,
}

impl SolidColor {
    pub fn new(
        width: u32,
        height: u32,
        packed: u32,
        start: Micros,
        duration: Micros,
    ) -> Result<Self, FillError> {
        if rgba_len(width, height).is_none() {
            return Err(FillError::Extent);
        }
        Ok(Self {
            width,
            height,
            packed,
            span: exclusive_end(start.0, duration.0),
        })
    }
}

fn exclusive_end(start: i64, duration: i64) -> Option<(i64, i64)> {
    if duration <= 0 {
        return None;
    }
    Some((start, start.checked_add(duration)?))
}

impl PictureSource for SolidColor {
    fn sample(&self, time: Micros) -> Sample {
        let Some((start, end)) = self.span else {
            return Sample::Miss;
        };
        if time.0 < start || time.0 >= end {
            return Sample::Miss;
        }
        match fill(self.width, self.height, self.packed) {
            Ok(bytes) => Sample::Ready(Picture::Cpu(bytes)),
            Err(_) => Sample::Miss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_solid_source_is_ready_inside_its_span_and_missing_outside() {
        let solid = SolidColor::new(1, 1, 0xFF00_00FF, Micros(0), Micros(1_000)).unwrap();
        let Sample::Ready(Picture::Cpu(bytes)) = solid.sample(Micros(0)) else {
            panic!("solid frame was not ready");
        };
        assert_eq!(bytes.rgba(), &[0x00, 0x00, 0xFF, 0xFF]);
        assert!(matches!(solid.sample(Micros(1_000)), Sample::Miss));
        assert!(matches!(solid.sample(Micros(-1)), Sample::Miss));
    }

    #[test]
    fn a_non_positive_or_overflowing_span_is_a_miss() {
        let zero = SolidColor::new(1, 1, 0xFF00_0000, Micros(0), Micros(0)).unwrap();
        assert!(matches!(zero.sample(Micros(0)), Sample::Miss));
        let overflow = SolidColor::new(1, 1, 0xFF00_0000, Micros(i64::MAX), Micros(1)).unwrap();
        assert!(matches!(overflow.sample(Micros(i64::MAX)), Sample::Miss));
    }
}
