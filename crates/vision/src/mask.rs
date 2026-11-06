//! The model's low-resolution logits, turned into a mask at the frame's size.
//!
//! The decoder answers on a [`LOW_RES`] square that covers the model's
//! padded 1024 square, four model pixels to each logit. Only the part over
//! the resized frame is real; the rest is padding and is never sampled.

use crate::VisionError;
use crate::frame::MODEL_SIDE;

/// The side of the decoder's square of logits.
pub(crate) const LOW_RES: usize = 256;

/// Which pixels of a frame are selected.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    /// One byte per pixel, row by row: 255 selected, 0 not.
    pub alpha: Vec<u8>,
    /// The model's own estimate of how well the mask fits, about 0 to 1.
    pub score: f32,
}

impl Mask {
    /// How many pixels are selected.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.alpha.iter().filter(|a| **a != 0).count()
    }
}

/// Samples `logits` bilinearly at each source pixel's centre and keeps the
/// pixels above zero. `resized` is the size the model read the frame at.
pub(crate) fn upscale(
    logits: &[f32],
    resized: (u32, u32),
    source: (u32, u32),
    score: f32,
) -> Result<Mask, VisionError> {
    if logits.len() != LOW_RES * LOW_RES {
        return Err(VisionError::Model {
            message: format!(
                "a mask has {} logits, not {}",
                logits.len(),
                LOW_RES * LOW_RES
            ),
        });
    }
    let (width, height) = source;
    if width == 0 || height == 0 || resized.0 == 0 || resized.1 == 0 {
        return Err(VisionError::Model {
            message: "a mask for an empty frame".into(),
        });
    }
    let columns = taps(resized.0, width);
    let rows = taps(resized.1, height);
    let mut alpha = Vec::with_capacity(width as usize * height as usize);
    for &(y0, y1, fy) in &rows {
        let (Some(top), Some(bottom)) = (
            logits.get(y0 * LOW_RES..(y0 + 1) * LOW_RES),
            logits.get(y1 * LOW_RES..(y1 + 1) * LOW_RES),
        ) else {
            return Err(VisionError::Model {
                message: "a mask row outside the logits".into(),
            });
        };
        for &(x0, x1, fx) in &columns {
            let at = |row: &[f32], x: usize| row.get(x).copied().unwrap_or(f32::NEG_INFINITY);
            let upper = at(top, x0) + (at(top, x1) - at(top, x0)) * fx;
            let lower = at(bottom, x0) + (at(bottom, x1) - at(bottom, x0)) * fx;
            let value = upper + (lower - upper) * fy;
            alpha.push(if value > 0.0 { 255 } else { 0 });
        }
    }
    Ok(Mask {
        width,
        height,
        alpha,
        score,
    })
}

/// For each of `source` pixels along one side, the two logits around its
/// centre and the weight of the second. `resized` is that side in the model.
fn taps(resized: u32, source: u32) -> Vec<(usize, usize, f32)> {
    let per_logit = MODEL_SIDE as usize / LOW_RES;
    // The logits over the resized frame; the rest of the square is padding.
    let valid = (resized as usize).div_ceil(per_logit).clamp(1, LOW_RES);
    let last = valid - 1;
    #[allow(clippy::cast_precision_loss)]
    let ratio = resized as f32 / source as f32 / per_logit as f32;
    (0..source)
        .map(|s| {
            #[allow(clippy::cast_precision_loss)]
            let at = ((s as f32 + 0.5) * ratio - 0.5).max(0.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let low = (at.floor() as usize).min(last);
            let high = (low + 1).min(last);
            #[allow(clippy::cast_precision_loss)]
            let weight = if high == low { 0.0 } else { at - low as f32 };
            (low, high, weight.clamp(0.0, 1.0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        (0..LOW_RES * LOW_RES)
            .map(|i| f(i % LOW_RES, i / LOW_RES))
            .collect()
    }

    #[test]
    fn all_positive_logits_select_every_pixel() {
        let mask = upscale(&square(|_, _| 3.0), (1024, 576), (1920, 1080), 0.9).unwrap();
        assert_eq!((mask.width, mask.height), (1920, 1080));
        assert_eq!(mask.alpha.len(), 1920 * 1080);
        assert_eq!(mask.selected(), 1920 * 1080);
        assert_eq!(mask.score, 0.9);
    }

    #[test]
    fn the_left_half_of_the_frame_maps_to_the_left_half_of_the_mask() {
        // A landscape frame fills all 256 columns: the left 128 are selected.
        let logits = square(|x, _| if x < 128 { 5.0 } else { -5.0 });
        let mask = upscale(&logits, (1024, 576), (1920, 1080), 1.0).unwrap();
        let row = &mask.alpha[540 * 1920..541 * 1920];
        assert!(row[..955].iter().all(|a| *a == 255));
        assert!(row[965..].iter().all(|a| *a == 0));
    }

    #[test]
    fn the_padding_below_a_landscape_frame_is_never_sampled() {
        // 576 model rows are 144 logits; everything below is padding.
        let logits = square(|_, y| if y < 144 { -5.0 } else { 5.0 });
        let mask = upscale(&logits, (1024, 576), (1920, 1080), 1.0).unwrap();
        assert_eq!(mask.selected(), 0);
    }

    #[test]
    fn the_padding_right_of_a_portrait_frame_is_never_sampled() {
        let logits = square(|x, _| if x < 144 { -5.0 } else { 5.0 });
        let mask = upscale(&logits, (576, 1024), (1080, 1920), 1.0).unwrap();
        assert_eq!(mask.selected(), 0);
    }

    #[test]
    fn a_frame_smaller_than_the_logits_still_gets_its_size() {
        let mask = upscale(&square(|_, _| 1.0), (1024, 1024), (3, 2), 0.5).unwrap();
        assert_eq!((mask.width, mask.height, mask.alpha.len()), (3, 2, 6));
    }

    #[test]
    fn logits_of_the_wrong_size_are_refused() {
        assert!(matches!(
            upscale(&[1.0; 10], (1024, 576), (1920, 1080), 1.0),
            Err(VisionError::Model { .. })
        ));
    }

    #[test]
    fn upscaling_reads_each_pixel_once() {
        // The cost per click: one pass over the output, with taps per side.
        let columns = taps(576, 1080);
        let rows = taps(1024, 1920);
        assert_eq!((columns.len(), rows.len()), (1080, 1920));
        assert!(columns.iter().chain(&rows).all(|(lo, hi, w)| {
            *lo < LOW_RES && *hi < LOW_RES && *hi >= *lo && (0.0..=1.0).contains(w)
        }));
    }
}
