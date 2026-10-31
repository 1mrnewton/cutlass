//! `PictureSource` over decoded pictures.
//!
//! Text and shapes never open a decoder. A platform buffer stays a platform
//! buffer. Coded video goes through `cutlass-color`. A still is scaled to
//! 8-bit RGBA and premultiplied. Gamma, chromaticity, and a profile are not
//! applied. A layout neither path can convert is a named miss.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use cutlass_codec::access::{
    BitDepth, ColorPrimaries, ColorRange, ColorType, MatrixCoefficients, PixelFormat, Transfer,
    Transparency, Yuv420p,
};
use cutlass_color::{Input, Matrix, Outcome, Picture as ColorPicture, Request};
use cutlass_decode::StillImage;
use cutlass_gpu::Extent;
use cutlass_model::{Draft, Micros, SegmentId};
use cutlass_picture::{CpuBytes, Picture, PictureSource, PlatformBuffer, Sample, rgba_len};

use crate::cache::FrameWindow;
use crate::clock::{ClockKind, MediaClock};
use crate::error::EngineError;
use crate::map::{self, TimelineMap};
use crate::reader::{DecodedFrame, Media, MediaInfo, ReaderFactory};

/// Decoded pictures for one segment.
pub struct VideoSource {
    path: PathBuf,
    media: RefCell<Media>,
    map: TimelineMap,
    cache: RefCell<FrameWindow<DecodedFrame>>,
    clock: RefCell<MediaClock>,
    decodes: Cell<u32>,
}

impl VideoSource {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn info(&self) -> MediaInfo {
        self.media.borrow().info().clone()
    }

    #[must_use]
    pub fn decodes(&self) -> u32 {
        self.decodes.get()
    }

    #[must_use]
    pub fn cached(&self) -> usize {
        self.cache.borrow().len()
    }

    pub fn advance_audio(&self, frames: u32, rate: u32) -> Result<Micros, EngineError> {
        self.clock.borrow_mut().advance_audio(frames, rate)
    }

    /// Decode into the window unless playback is already past `time`.
    pub fn fill(&self, time: Micros) -> Result<Sample, EngineError> {
        let source = map::map_timeline(&self.map, time)?;
        let held = self.decode(source)?;
        let clock = self.clock.borrow();
        let late = clock.kind() == ClockKind::Playback && clock.audio_time().0 > time.0;
        drop(clock);
        if late {
            return Ok(Sample::Miss);
        }
        let frame = held.frame.clone();
        self.cache
            .borrow_mut()
            .insert(held.start.0, held.end.0, held.frame);
        self.deliver(&frame)
    }

    /// The picture at `time`. Playback misses without decoding. Export decodes and waits.
    pub fn picture_at(&self, time: Micros) -> Result<Picture, EngineError> {
        let source = map::map_timeline(&self.map, time)?;
        let cached = self.cache.borrow().get(source.0).cloned();
        if let Some(frame) = cached {
            return to_picture(&frame);
        }
        match self.clock.borrow().kind() {
            ClockKind::Playback => Err(EngineError::NotReady),
            ClockKind::Export => {
                let held = self.decode(source)?;
                let frame = held.frame.clone();
                self.cache
                    .borrow_mut()
                    .insert(held.start.0, held.end.0, held.frame);
                to_picture(&frame)
            }
        }
    }

    fn decode(&self, source: Micros) -> Result<crate::reader::HeldFrame, EngineError> {
        self.decodes.set(self.decodes.get().saturating_add(1));
        self.media.borrow_mut().decode_at(source, &self.map.source)
    }

    fn deliver(&self, frame: &DecodedFrame) -> Result<Sample, EngineError> {
        to_picture(frame).map(Sample::Ready)
    }
}

impl PictureSource for VideoSource {
    fn sample(&self, time: Micros) -> Sample {
        match self.picture_at(time) {
            Ok(picture) => Sample::Ready(picture),
            Err(_) => Sample::Miss,
        }
    }
}

/// Opens the segment's file. The draft is only borrowed.
pub fn open_segment(
    factory: &ReaderFactory,
    draft: &Draft,
    id: &SegmentId,
    clock: ClockKind,
    window: usize,
) -> Result<VideoSource, EngineError> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .find(|segment| segment.id == *id)
        .ok_or(EngineError::MissingSegment)?;
    if map::is_graphic(draft, segment) {
        return Err(EngineError::NotMedia);
    }
    let Some(path) = map::media_path(draft, segment) else {
        return Err(EngineError::MissingFile {
            path: PathBuf::new(),
        });
    };
    let media = factory.open(Path::new(path))?;
    let map = map::placement_of(draft, segment)?;
    Ok(VideoSource {
        path: PathBuf::from(path),
        media: RefCell::new(media),
        map,
        cache: RefCell::new(FrameWindow::new(window)?),
        clock: RefCell::new(MediaClock::new(clock)),
        decodes: Cell::new(0),
    })
}

fn to_picture(frame: &DecodedFrame) -> Result<Picture, EngineError> {
    match frame {
        DecodedFrame::Platform {
            width,
            height,
            token,
        } => Ok(Picture::Platform(PlatformBuffer {
            extent: Extent {
                width: *width,
                height: *height,
            },
            token: *token,
        })),
        DecodedFrame::Still(image) => still_picture(image),
        DecodedFrame::Coded {
            width,
            height,
            pixel,
            primaries,
            transfer,
            bytes,
        } => coded_bytes(*pixel, *width, *height, *primaries, *transfer, bytes),
    }
}

fn coded_bytes(
    pixel: PixelFormat,
    width: u32,
    height: u32,
    primaries: ColorPrimaries,
    transfer: Transfer,
    bytes: &[u8],
) -> Result<Picture, EngineError> {
    let layout = pixel_layout(pixel);
    if pixel != PixelFormat::Yuv420p {
        return Err(EngineError::ColorMiss {
            layout: layout.to_owned(),
        });
    }
    let Some(yuv) = yuv420(width, height, bytes) else {
        return Err(EngineError::ColorMiss {
            layout: layout.to_owned(),
        });
    };
    let Ok(matrix) = Matrix::from_coefficients(MatrixCoefficients::Bt709) else {
        return Err(EngineError::ColorMiss {
            layout: layout.to_owned(),
        });
    };
    let picture = ColorPicture::from_yuv420p(
        &yuv,
        ColorRange::Limited,
        matrix,
        primaries,
        cutlass_color::curve_from_codec(transfer),
    );
    let working = match cutlass_color::convert(Input::Cpu(picture), &Request::draft(0)) {
        Ok(Outcome::Working(working)) => working,
        Ok(Outcome::Platform { .. }) | Err(_) => {
            return Err(EngineError::ColorMiss {
                layout: layout.to_owned(),
            });
        }
    };
    let cpu = CpuBytes::new(width, height, working.rgba).map_err(|_| EngineError::ColorMiss {
        layout: layout.to_owned(),
    })?;
    Ok(Picture::Cpu(cpu))
}

/// Packed 4:2:0 is Y, then U, then V, with no stride. That is the byte order
/// `PictureOutput` stores for [`PixelFormat::Yuv420p`].
fn yuv420(width: u32, height: u32, bytes: &[u8]) -> Option<Yuv420p> {
    let y_len = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    let chroma_width = usize::try_from(width / 2).ok()?;
    let chroma_height = usize::try_from(height / 2).ok()?;
    let chroma_len = chroma_width.checked_mul(chroma_height)?;
    let u_end = y_len.checked_add(chroma_len)?;
    let v_end = u_end.checked_add(chroma_len)?;
    if bytes.len() != v_end {
        return None;
    }
    Yuv420p::try_new(
        width,
        height,
        bytes[..y_len].to_vec(),
        bytes[y_len..u_end].to_vec(),
        bytes[u_end..v_end].to_vec(),
    )
    .ok()
}

/// Coded samples become premultiplied 8-bit RGBA.
///
/// A depth below 8 bits and a 16-bit sample use `sample * 255 / max`.
/// Palette alpha past the table is opaque. A color key compares the coded
/// samples. Partial alpha is `(channel * alpha + 127) / 255`.
fn still_picture(image: &StillImage) -> Result<Picture, EngineError> {
    let Some(rgba) = still_rgba(image) else {
        return Err(EngineError::ColorMiss {
            layout: still_layout(image).to_owned(),
        });
    };
    let cpu =
        CpuBytes::new(image.width, image.height, rgba).map_err(|_| EngineError::ColorMiss {
            layout: still_layout(image).to_owned(),
        })?;
    Ok(Picture::Cpu(cpu))
}

fn still_rgba(image: &StillImage) -> Option<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return None;
    }
    let channels = image.color.channels();
    if channels == 0 {
        return None;
    }
    let pixels = usize::try_from(image.width)
        .ok()?
        .checked_mul(usize::try_from(image.height).ok()?)?;
    if image.samples.len() != pixels.checked_mul(channels)? {
        return None;
    }
    let mut rgba = vec![0_u8; rgba_len(image.width, image.height)?];
    for (pixel, chunk) in image.samples.chunks(channels).enumerate() {
        let (red, green, blue, alpha) = straight(image, chunk)?;
        let dest = pixel.checked_mul(4)?;
        let slot = rgba.get_mut(dest..dest + 4)?;
        slot[0] = premultiply(red, alpha);
        slot[1] = premultiply(green, alpha);
        slot[2] = premultiply(blue, alpha);
        slot[3] = alpha;
    }
    Some(rgba)
}

fn straight(image: &StillImage, samples: &[u16]) -> Option<(u8, u8, u8, u8)> {
    let max = image.depth.max_sample();
    match image.color {
        ColorType::Gray => {
            let sample = *samples.first()?;
            let gray = scale_sample(sample, max)?;
            let alpha = match &image.transparency {
                Transparency::None => 255,
                Transparency::ColorKey(key) if key.len() == 1 && key[0] == sample => 0,
                Transparency::ColorKey(key) if key.len() == 1 => 255,
                Transparency::ColorKey(_) | Transparency::PaletteAlpha(_) => return None,
            };
            Some((gray, gray, gray, alpha))
        }
        ColorType::GrayAlpha => {
            if !matches!(image.transparency, Transparency::None) {
                return None;
            }
            let gray = scale_sample(*samples.first()?, max)?;
            let alpha = scale_sample(*samples.get(1)?, max)?;
            Some((gray, gray, gray, alpha))
        }
        ColorType::Rgb => {
            let red_s = *samples.first()?;
            let green_s = *samples.get(1)?;
            let blue_s = *samples.get(2)?;
            let alpha = match &image.transparency {
                Transparency::None => 255,
                Transparency::ColorKey(key)
                    if key.len() == 3
                        && key[0] == red_s
                        && key[1] == green_s
                        && key[2] == blue_s =>
                {
                    0
                }
                Transparency::ColorKey(key) if key.len() == 3 => 255,
                Transparency::ColorKey(_) | Transparency::PaletteAlpha(_) => return None,
            };
            Some((
                scale_sample(red_s, max)?,
                scale_sample(green_s, max)?,
                scale_sample(blue_s, max)?,
                alpha,
            ))
        }
        ColorType::Rgba => {
            if !matches!(image.transparency, Transparency::None) {
                return None;
            }
            Some((
                scale_sample(*samples.first()?, max)?,
                scale_sample(*samples.get(1)?, max)?,
                scale_sample(*samples.get(2)?, max)?,
                scale_sample(*samples.get(3)?, max)?,
            ))
        }
        ColorType::Indexed => {
            let index = usize::from(*samples.first()?);
            let entry = image.palette.get(index)?;
            let alpha = match &image.transparency {
                Transparency::None => 255,
                Transparency::PaletteAlpha(table) => table.get(index).copied().unwrap_or(255),
                Transparency::ColorKey(_) => return None,
            };
            Some((entry.red, entry.green, entry.blue, alpha))
        }
    }
}

fn scale_sample(sample: u16, max: u16) -> Option<u8> {
    if max == 0 || sample > max {
        return None;
    }
    let scaled = u32::from(sample)
        .checked_mul(255)?
        .checked_div(u32::from(max))?;
    u8::try_from(scaled).ok()
}

fn premultiply(channel: u8, alpha: u8) -> u8 {
    if alpha == 255 {
        return channel;
    }
    let mixed = u16::from(channel) * u16::from(alpha) + 127;
    u8::try_from(mixed / 255).unwrap_or(0)
}

fn still_layout(image: &StillImage) -> &'static str {
    match (image.color, image.depth) {
        (ColorType::Gray, BitDepth::Eight) => "gray8",
        (ColorType::Rgb, BitDepth::Eight) => "rgb8",
        (ColorType::Rgba, BitDepth::Eight) => "rgba8",
        (ColorType::GrayAlpha, BitDepth::Eight) => "gray-alpha8",
        (ColorType::Indexed, BitDepth::Eight) => "indexed8",
        _ => "still",
    }
}

fn pixel_layout(pixel: PixelFormat) -> &'static str {
    match pixel {
        PixelFormat::Rgba8 => "rgba8",
        PixelFormat::Yuv420p => "yuv420p",
        PixelFormat::None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_codec::access::Rgb8;

    fn row(color: ColorType, depth: BitDepth, samples: Vec<u16>) -> StillImage {
        let channels = u32::try_from(color.channels()).unwrap_or(1);
        let width = u32::try_from(samples.len()).unwrap_or(0) / channels;
        StillImage::from_samples(width, 1, color, depth, samples)
    }

    fn rgba(image: StillImage) -> Vec<u8> {
        let Picture::Cpu(bytes) = still_picture(&image).expect("picture") else {
            panic!("cpu");
        };
        bytes.into_rgba()
    }

    #[test]
    fn eight_bit_color_keeps_its_samples_and_partial_alpha_is_premultiplied() {
        assert_eq!(
            rgba(row(
                ColorType::Rgb,
                BitDepth::Eight,
                vec![10, 20, 30, 40, 50, 60]
            )),
            vec![10, 20, 30, 255, 40, 50, 60, 255]
        );
        assert_eq!(
            rgba(row(ColorType::Gray, BitDepth::Eight, vec![7])),
            vec![7, 7, 7, 255]
        );
        assert_eq!(
            rgba(row(
                ColorType::Rgba,
                BitDepth::Eight,
                vec![200, 10, 20, 128]
            )),
            vec![100, 5, 10, 128]
        );
        assert_eq!(
            rgba(row(ColorType::GrayAlpha, BitDepth::Eight, vec![100, 128])),
            vec![50, 50, 50, 128]
        );
    }

    #[test]
    fn a_palette_a_color_key_and_a_narrow_sample_scale_to_eight_bits() {
        let mut indexed = row(ColorType::Indexed, BitDepth::Eight, vec![0, 1]);
        indexed.palette = vec![
            Rgb8 {
                red: 255,
                green: 0,
                blue: 0,
            },
            Rgb8 {
                red: 0,
                green: 255,
                blue: 0,
            },
        ];
        indexed.transparency = Transparency::PaletteAlpha(vec![0, 255]);
        assert_eq!(rgba(indexed), vec![0, 0, 0, 0, 0, 255, 0, 255]);

        let mut keyed = row(ColorType::Gray, BitDepth::Eight, vec![4, 5]);
        keyed.transparency = Transparency::ColorKey(vec![4]);
        assert_eq!(rgba(keyed), vec![0, 0, 0, 0, 5, 5, 5, 255]);

        assert_eq!(
            rgba(row(ColorType::Gray, BitDepth::One, vec![0, 1])),
            vec![0, 0, 0, 255, 255, 255, 255, 255]
        );
        assert_eq!(
            rgba(row(
                ColorType::Rgb,
                BitDepth::Sixteen,
                vec![65535, 0, 32896]
            )),
            vec![255, 0, 128, 255]
        );
    }

    #[test]
    fn gamma_and_a_profile_do_not_change_the_samples() {
        let mut tagged = row(ColorType::Rgb, BitDepth::Eight, vec![1, 2, 3]);
        tagged.gamma_times_100000 = Some(45_455);
        tagged.icc_profile = vec![1, 2, 3, 4];
        assert_eq!(rgba(tagged), vec![1, 2, 3, 255]);
    }

    #[test]
    fn a_short_or_out_of_range_still_is_a_color_miss() {
        let short = StillImage::from_samples(2, 1, ColorType::Gray, BitDepth::Eight, vec![1]);
        assert!(matches!(
            still_picture(&short),
            Err(EngineError::ColorMiss { .. })
        ));
        let high = row(ColorType::Gray, BitDepth::Eight, vec![256]);
        assert!(matches!(
            still_picture(&high),
            Err(EngineError::ColorMiss { .. })
        ));
    }
}
