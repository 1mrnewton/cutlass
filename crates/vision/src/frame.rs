//! A frame resized the way SAM-family models read it: the longest side is
//! [`MODEL_SIDE`], the other keeps the aspect, and the values stay 0 to 255.

use image::imageops::{self, FilterType};
use image::{ImageBuffer, Rgb};

use crate::{Image, VisionError};

/// The longest side the model reads.
pub(crate) const MODEL_SIDE: u32 = 1024;

/// The longest side a frame may have. An 8K frame is 7680.
pub const MAX_SIDE: u32 = 16_384;

/// RGB as `f32`, row by row, at the size the model reads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Resized {
    pub hwc: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// The resized size over the source size.
    pub scale: f32,
}

/// Checks `image`, then resizes it with a triangle filter.
pub(crate) fn resize(image: &Image<'_>) -> Result<Resized, VisionError> {
    let refused = || VisionError::Image {
        width: image.width,
        height: image.height,
        bytes: image.rgb.len(),
    };
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(refused());
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3));
    if expected != Some(image.rgb.len()) {
        return Err(refused());
    }
    let (resized_width, resized_height) = model_size(width, height);
    #[allow(clippy::cast_precision_loss)]
    let scale = MODEL_SIDE as f32 / width.max(height) as f32;
    let hwc = if (resized_width, resized_height) == (width, height) {
        image.rgb.iter().copied().map(f32::from).collect()
    } else {
        let source: ImageBuffer<Rgb<u8>, &[u8]> =
            ImageBuffer::from_raw(width, height, image.rgb).ok_or_else(refused)?;
        imageops::resize(&source, resized_width, resized_height, FilterType::Triangle)
            .into_raw()
            .into_iter()
            .map(f32::from)
            .collect()
    };
    Ok(Resized {
        hwc,
        width: resized_width,
        height: resized_height,
        scale,
    })
}

/// The size the model reads a `width` by `height` frame at. Neither side is zero.
pub(crate) fn model_size(width: u32, height: u32) -> (u32, u32) {
    let longest = u64::from(width.max(height).max(1));
    let side = |s: u32| {
        // Rounded to the nearest pixel, as the reference does.
        let scaled = (u64::from(s) * u64::from(MODEL_SIDE) * 2 + longest) / (longest * 2);
        u32::try_from(scaled.clamp(1, u64::from(MODEL_SIDE))).unwrap_or(MODEL_SIDE)
    };
    (side(width), side(height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; width as usize * height as usize * 3]
    }

    #[test]
    fn the_longest_side_becomes_1024_and_the_aspect_stays() {
        assert_eq!(model_size(1920, 1080), (1024, 576));
        assert_eq!(model_size(1080, 1920), (576, 1024));
        assert_eq!(model_size(3840, 2160), (1024, 576));
        assert_eq!(model_size(512, 512), (1024, 1024));
        assert_eq!(model_size(1024, 1), (1024, 1));
        assert_eq!(model_size(16_384, 1), (1024, 1));
    }

    #[test]
    fn a_frame_is_resized_and_keeps_its_values() {
        let rgb = gray(1920, 1080, 200);
        let image = Image {
            width: 1920,
            height: 1080,
            rgb: &rgb,
        };
        let resized = resize(&image).unwrap();
        assert_eq!((resized.width, resized.height), (1024, 576));
        assert_eq!(resized.hwc.len(), 1024 * 576 * 3);
        assert!(resized.hwc.iter().all(|v| (*v - 200.0).abs() < 0.5));
        assert!((resized.scale - 1024.0 / 1920.0).abs() < 1e-6);
    }

    #[test]
    fn a_frame_already_at_the_model_size_is_not_filtered() {
        let mut rgb = gray(1024, 4, 0);
        rgb[5] = 255;
        let image = Image {
            width: 1024,
            height: 4,
            rgb: &rgb,
        };
        let resized = resize(&image).unwrap();
        assert_eq!(resized.hwc[5], 255.0);
        assert_eq!(resized.hwc[4], 0.0);
    }

    #[test]
    fn a_bad_size_or_buffer_is_refused() {
        let rgb = gray(4, 4, 0);
        for (width, height, bytes) in [
            (0, 4, 0),
            (4, 0, 0),
            (4, 4, 47),
            (4, 4, 49),
            (MAX_SIDE + 1, 1, (MAX_SIDE as usize + 1) * 3),
        ] {
            let big = vec![0; bytes];
            let image = Image {
                width,
                height,
                rgb: if bytes == 48 { &rgb } else { &big },
            };
            assert_eq!(
                resize(&image),
                Err(VisionError::Image {
                    width,
                    height,
                    bytes
                })
            );
        }
    }
}
