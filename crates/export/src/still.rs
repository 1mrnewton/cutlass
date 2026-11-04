//! Write the picture at one time as a PNG or JPEG file.
//!
//! The picture is the one a movie export draws at that time, at the canvas
//! size. Like a movie, it is written under a hidden part name and moved into
//! place once whole.

use std::fs;
use std::path::Path;

use cutlass_model::Micros;
use cutlass_renderer_software::SoftwareDevice;
use image::{ExtendedColorType, ImageEncoder};

use crate::ExportError;
use crate::movie::{MovieSources, part_path};
use crate::run::still_rgba;

/// The file a still is written as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StillFormat {
    /// Lossless, and keeps a clear canvas clear.
    #[default]
    Png,
    /// Smaller; a clear canvas is laid on black.
    Jpeg,
}

impl StillFormat {
    /// The file extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }
}

/// JPEG quality out of 100.
const JPEG_QUALITY: u8 = 92;

/// Write the picture at `time` to `destination`. Answers the picture's
/// width and height.
pub fn write_still(
    sources: &MovieSources<'_>,
    time: Micros,
    format: StillFormat,
    destination: &Path,
) -> Result<(u32, u32), ExportError> {
    let mut device = SoftwareDevice::new();
    let (width, height, rgba) = still_rgba(
        sources.draft,
        &mut device,
        time,
        sources.readers,
        sources.root,
        sources.fonts,
        sources.renderer,
    )?;
    let bytes = encoded(width, height, &rgba, format)?;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = fs::write(&part, bytes).and_then(|()| fs::rename(&part, destination));
    if let Err(error) = written {
        let _ = fs::remove_file(&part);
        return Err(ExportError::File(error.to_string()));
    }
    Ok((width, height))
}

/// `rgba` as the bytes of a `format` file.
fn encoded(
    width: u32,
    height: u32,
    rgba: &[u8],
    format: StillFormat,
) -> Result<Vec<u8>, ExportError> {
    let mut bytes = Vec::new();
    let result = match format {
        StillFormat::Png => image::codecs::png::PngEncoder::new(&mut bytes).write_image(
            rgba,
            width,
            height,
            ExtendedColorType::Rgba8,
        ),
        StillFormat::Jpeg => {
            // JPEG has no alpha: lay each pixel on black.
            let rgb: Vec<u8> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| {
                    let alpha = u16::from(pixel[3]);
                    [0, 1, 2].map(|channel| {
                        let over = u16::from(pixel[channel]) * alpha / 255;
                        u8::try_from(over).unwrap_or(u8::MAX)
                    })
                })
                .collect();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
                .write_image(&rgb, width, height, ExtendedColorType::Rgb8)
        }
    };
    result.map_err(|error| ExportError::Picture(error.to_string()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_format_writes_a_file_its_decoder_reads_back() {
        // A 2x1 picture: opaque red, then half-clear white.
        let rgba = [255, 0, 0, 255, 255, 255, 255, 128];
        let png = encoded(2, 1, &rgba, StillFormat::Png).unwrap();
        let read = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .unwrap()
            .to_rgba8();
        assert_eq!(read.as_raw().as_slice(), rgba.as_slice());
        let jpeg = encoded(2, 1, &rgba, StillFormat::Jpeg).unwrap();
        assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
        let read = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb8();
        assert_eq!(read.dimensions(), (2, 1));
    }
}
