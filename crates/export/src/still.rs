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

/// The short side, in pixels, of a picture drawn to be looked at; a
/// smaller canvas is not made larger.
pub const LOOK_SHORT_SIDE: u32 = 480;

/// The longest side of a picture drawn to be looked at, which a very wide
/// or very tall canvas reaches first.
pub const LOOK_LONG_SIDE: u32 = 1024;

/// JPEG quality of a picture drawn to be looked at: where things are and
/// how they read, at a small size to send.
const LOOK_QUALITY: u8 = 80;

/// Write the picture at `time` to `destination`. Answers the picture's
/// width and height.
pub fn write_still(
    sources: &MovieSources<'_>,
    time: Micros,
    format: StillFormat,
    destination: &Path,
) -> Result<(u32, u32), ExportError> {
    let (width, height, rgba) = still_picture(sources, time)?;
    let bytes = encoded(width, height, &rgba, format, sources.encoders)?;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = write_synced(&part, &bytes).and_then(|()| fs::rename(&part, destination));
    if let Err(error) = written {
        let _ = fs::remove_file(&part);
        return Err(ExportError::File(error.to_string()));
    }
    Ok((width, height))
}

/// The picture at `time` that [`write_still`] would write, before it is
/// encoded: canvas-sized, straight RGBA, with its width and height.
pub fn still_picture(
    sources: &MovieSources<'_>,
    time: Micros,
) -> Result<(u32, u32, Vec<u8>), ExportError> {
    let mut device = SoftwareDevice::new();
    still_rgba(
        sources.draft,
        &mut device,
        time,
        sources.readers,
        sources.root,
        sources.fonts,
        sources.renderer,
    )
}

/// The picture at `time`, scaled down to be looked at instead of kept: a
/// JPEG no larger than [`LOOK_SHORT_SIDE`] by [`LOOK_LONG_SIDE`], with its
/// width and height. It is the picture [`write_still`] writes, smaller.
pub fn look_picture(
    sources: &MovieSources<'_>,
    time: Micros,
) -> Result<(u32, u32, Vec<u8>), ExportError> {
    let (width, height, rgba) = still_picture(sources, time)?;
    let (small_width, small_height) = look_size(width, height);
    let rgba = if (small_width, small_height) == (width, height) {
        rgba
    } else {
        let full = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| ExportError::Picture("the frame is not its size".to_owned()))?;
        image::imageops::resize(
            &full,
            small_width,
            small_height,
            image::imageops::FilterType::Triangle,
        )
        .into_raw()
    };
    let bytes = jpeg(small_width, small_height, &rgba, LOOK_QUALITY)?;
    Ok((small_width, small_height, bytes))
}

/// The size a `width` by `height` canvas is looked at: scaled down, never
/// up, until its short side is within [`LOOK_SHORT_SIDE`] and its long
/// side within [`LOOK_LONG_SIDE`], and at least a pixel each way.
fn look_size(width: u32, height: u32) -> (u32, u32) {
    let (short, long) = (width.min(height), width.max(height));
    if short <= LOOK_SHORT_SIDE && long <= LOOK_LONG_SIDE {
        return (width, height);
    }
    // The tighter of the two limits, as a fraction: `over` / `under`.
    let (over, under) = if u64::from(LOOK_SHORT_SIDE) * u64::from(long)
        <= u64::from(LOOK_LONG_SIDE) * u64::from(short)
    {
        (LOOK_SHORT_SIDE, short)
    } else {
        (LOOK_LONG_SIDE, long)
    };
    let scaled = |side: u32| {
        let rounded = (u64::from(side) * u64::from(over) + u64::from(under) / 2) / u64::from(under);
        u32::try_from(rounded).unwrap_or(u32::MAX).max(1)
    };
    (scaled(width), scaled(height))
}

/// `rgba` as a JPEG file at `quality` out of 100.
fn jpeg(width: u32, height: u32, rgba: &[u8], quality: u8) -> Result<Vec<u8>, ExportError> {
    // The encoder panics on a buffer that is not the picture's size.
    let samples = u64::from(width) * u64::from(height) * 4;
    if u64::try_from(rgba.len()).ok() != Some(samples) {
        return Err(ExportError::Picture("the frame is not its size".to_owned()));
    }
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
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality)
        .write_image(&rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|error| ExportError::Picture(error.to_string()))?;
    Ok(bytes)
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
        StillFormat::Jpeg => return jpeg(width, height, rgba, JPEG_QUALITY),
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
    fn a_look_is_scaled_down_to_its_limits_and_never_up() {
        for (canvas, looked) in [
            // The short side is the limit for the usual canvases.
            ((1920, 1080), (853, 480)),
            ((1080, 1920), (480, 853)),
            ((3840, 2160), (853, 480)),
            ((1280, 720), (853, 480)),
            ((1080, 1080), (480, 480)),
            // A strip reaches the long side first.
            ((4000, 480), (1024, 123)),
            ((480, 4000), (123, 1024)),
            ((5000, 10), (1024, 2)),
            // A side never scales away.
            ((100_000, 1), (1024, 1)),
            // Within both limits already: as it is.
            ((640, 360), (640, 360)),
            ((1024, 480), (1024, 480)),
            ((1, 1), (1, 1)),
        ] {
            assert_eq!(look_size(canvas.0, canvas.1), looked, "{canvas:?}");
        }
    }

    #[test]
    fn a_look_is_a_jpeg_laid_on_black() {
        // Opaque red, then clear white, which reads as black.
        let rgba = [255, 0, 0, 255, 255, 255, 255, 0];
        let bytes = jpeg(2, 1, &rgba, LOOK_QUALITY).unwrap();
        assert_eq!(&bytes[..2], &[0xff, 0xd8]);
        let read = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb8();
        assert_eq!(read.dimensions(), (2, 1));
        // A bad size is an error, not a picture.
        assert!(jpeg(3, 1, &rgba, LOOK_QUALITY).is_err());
    }

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
