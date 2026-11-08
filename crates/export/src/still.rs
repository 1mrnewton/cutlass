//! Write the picture at one time as a PNG, JPEG or WebP file.
//!
//! The picture is the one a movie export draws at that time, at the canvas
//! size. Like a movie, it is written under a hidden part name and moved into
//! place once whole.

use std::fs;
use std::path::Path;

use cutlass_encode::{
    BitDepth, CodecId, CodecParams, ColorType, EncodeFactory, PictureInput, PixelFormat, StillImage,
};
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
    /// Lossy with lossless alpha, by the WebP encoder the sources
    /// register, at its default settings.
    Webp,
}

impl StillFormat {
    /// The file extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
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
    let bytes = encoded(width, height, &rgba, format, sources.encoders)?;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = write_synced(&part, &bytes).and_then(|()| fs::rename(&part, destination));
    if let Err(error) = written {
        let _ = fs::remove_file(&part);
        return Err(ExportError::File(error.to_string()));
    }
    Ok((width, height))
}

/// `rgba` as the bytes of a `format` file. WebP comes from `encoders`.
fn encoded(
    width: u32,
    height: u32,
    rgba: &[u8],
    format: StillFormat,
    encoders: &EncodeFactory,
) -> Result<Vec<u8>, ExportError> {
    let mut bytes = Vec::new();
    let result = match format {
        StillFormat::Webp => return webp(width, height, rgba, encoders),
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

/// One WebP file: the encoder answers a still picture with one packet.
fn webp(
    width: u32,
    height: u32,
    rgba: &[u8],
    encoders: &EncodeFactory,
) -> Result<Vec<u8>, ExportError> {
    let params = CodecParams::with_format(CodecId::Webp, width, height, PixelFormat::Rgba8);
    let mut encoder = encoders.open(&params).map_err(ExportError::Encoder)?;
    let samples = rgba.iter().map(|&sample| u16::from(sample)).collect();
    let image = StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples)
        .checked()
        .map_err(|error| ExportError::Picture(error.to_string()))?;
    let mut packets = encoder
        .push_picture(&PictureInput { image })
        .map_err(ExportError::Encoder)?;
    packets.extend(encoder.drain().map_err(ExportError::Encoder)?);
    match <[_; 1]>::try_from(packets) {
        Ok([packet]) => Ok(packet.data),
        Err(_) => Err(ExportError::Picture(
            "the webp encoder did not answer one file".to_owned(),
        )),
    }
}

/// `bytes` into `path`, on the disk before the call returns, so a crash
/// after the rename cannot leave an empty picture.
fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::Decoder;

    #[test]
    fn each_format_writes_a_file_its_decoder_reads_back() {
        // A 2x1 picture: opaque red, then half-clear white.
        let rgba = [255, 0, 0, 255, 255, 255, 255, 128];
        let mut encoders = EncodeFactory::new();
        cutlass_webp::register_encoders(&mut encoders);
        let png = encoded(2, 1, &rgba, StillFormat::Png, &encoders).unwrap();
        let read = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .unwrap()
            .to_rgba8();
        assert_eq!(read.as_raw().as_slice(), rgba.as_slice());
        let jpeg = encoded(2, 1, &rgba, StillFormat::Jpeg, &encoders).unwrap();
        assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
        let read = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb8();
        assert_eq!(read.dimensions(), (2, 1));
        // WebP keeps alpha exactly; the colour is lossy.
        let webp = encoded(2, 1, &rgba, StillFormat::Webp, &encoders).unwrap();
        assert_eq!((&webp[..4], &webp[8..12]), (&b"RIFF"[..], &b"WEBP"[..]));
        let mut decoder =
            cutlass_webp::WebpDecoder::open(&CodecParams::video(CodecId::Webp, 2, 1)).unwrap();
        let packet = cutlass_decode::Packet::video(webp, 0, 0, 0, true);
        let pictures = decoder.decode_picture(&packet).unwrap();
        assert_eq!(pictures.len(), 1);
        let image = &pictures[0].image;
        assert_eq!(
            (image.width, image.height, image.color),
            (2, 1, ColorType::Rgba)
        );
        assert_eq!((image.samples[3], image.samples[7]), (255, 128));
        // The two pixels share their chroma: red stays the strongest.
        let red = &image.samples[..3];
        assert!(red[0] > red[1] + 40 && red[0] > red[2] + 40, "{red:?}");
        // Without a WebP encoder the still is refused.
        assert!(encoded(2, 1, &rgba, StillFormat::Webp, &EncodeFactory::new()).is_err());
    }
}
