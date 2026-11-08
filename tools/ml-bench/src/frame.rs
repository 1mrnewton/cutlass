//! A frame resized the way SAM expects: the longest side is 1024.

use std::path::Path;

use image::imageops::FilterType;

use crate::engine::{BenchError, IMAGE_SIZE};

/// RGB, 0 to 255, row by row, in the resized size.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub hwc: Vec<f32>,
    pub height: usize,
    pub width: usize,
    /// Resized size over the source size.
    pub scale: f32,
}

impl Frame {
    /// Reads a PNG or JPEG and resizes it with a bilinear filter.
    pub fn open(path: &Path) -> Result<Self, BenchError> {
        let source = image::open(path)
            .map_err(|error| BenchError::Io(format!("{}: {error}", path.display())))?
            .to_rgb8();
        let (width, height) = source.dimensions();
        let longest = width.max(height);
        if longest == 0 {
            return Err(BenchError::Shape(format!("{} is empty", path.display())));
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = IMAGE_SIZE as f32 / longest as f32;
        let resize = |side: u32| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let scaled = (side as f32 * scale + 0.5) as u32;
            scaled.clamp(1, u32::try_from(IMAGE_SIZE).unwrap_or(u32::MAX))
        };
        let resized =
            image::imageops::resize(&source, resize(width), resize(height), FilterType::Triangle);
        let (width, height) = resized.dimensions();
        Ok(Self {
            hwc: resized.into_raw().into_iter().map(f32::from).collect(),
            height: height as usize,
            width: width as usize,
            scale,
        })
    }

    /// A frame already resized elsewhere, such as the one in a reference file.
    pub fn from_hwc(
        hwc: Vec<f32>,
        height: usize,
        width: usize,
        scale: f32,
    ) -> Result<Self, BenchError> {
        if height == 0 || width == 0 || height > IMAGE_SIZE || width > IMAGE_SIZE {
            return Err(BenchError::Shape(format!(
                "frame {width}x{height} is outside 1..=1024"
            )));
        }
        if hwc.len() != height * width * 3 {
            return Err(BenchError::Shape(format!(
                "frame has {} values, {width}x{height} RGB needs {}",
                hwc.len(),
                height * width * 3
            )));
        }
        Ok(Self {
            hwc,
            height,
            width,
            scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_with_the_wrong_length_is_refused() {
        assert!(Frame::from_hwc(vec![0.0; 5], 2, 2, 1.0).is_err());
    }

    #[test]
    fn a_frame_larger_than_the_encoder_is_refused() {
        assert!(Frame::from_hwc(vec![0.0; 1025 * 3], 1, 1025, 1.0).is_err());
    }
}
