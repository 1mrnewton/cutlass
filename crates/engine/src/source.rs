//! `PictureSource` over decoded pictures.
//!
//! Text and shapes never open a decoder. A platform buffer stays a platform
//! buffer. Coded video goes through `cutlass-color`. A still is scaled to
//! 8-bit, mapped through its gamma or profile, then premultiplied. A layout
//! neither path can convert is a named miss.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use cutlass_codec::access::{
    BitDepth, ColorPrimaries, ColorRange, ColorType, MatrixCoefficients, PixelFormat, Transfer,
    Transparency, Yuv420p,
};
use cutlass_color::{Input, Matrix, Outcome, Picture as ColorPicture, Request};
use cutlass_container::DisplayQuarter;
use cutlass_decode::StillImage;
use cutlass_gpu::Extent;
use cutlass_model::{Draft, Micros, SegmentId};
use cutlass_picture::{CpuBytes, Picture, PictureSource, PlatformBuffer, Sample, rgba_len};

use crate::ahead::ReadAhead;
use crate::cache::FrameWindow;
use crate::clock::{ClockKind, MediaClock};
use crate::error::EngineError;
use crate::filmstrip::{self, StripCell};
use crate::map::{self, TimelineMap};
use crate::preload::Preload;
use crate::reader::{DecodedFrame, MAX_AUDIO_FRAMES, Media, MediaInfo, ReaderFactory};
use crate::waveform::{MAX_BUCKETS, Waveform};

/// Decoded pictures for one segment.
pub struct VideoSource {
    path: PathBuf,
    media: RefCell<Media>,
    map: TimelineMap,
    cache: RefCell<FrameWindow<DecodedFrame>>,
    clock: RefCell<MediaClock>,
    preload: RefCell<Preload>,
    decodes: Cell<u32>,
    display: DisplayQuarter,
    /// A decode thread an export-clock read waits on instead of decoding here.
    ahead: Option<ReadAhead>,
    /// Off, a read whose picture the thread has not decoded yet returns the
    /// picture this clip showed last instead of waiting. Preview turns it off.
    patient: Cell<bool>,
    /// A read returned an earlier picture than the one asked for.
    stale: Cell<bool>,
}

impl VideoSource {
    /// Decode this clip on its own thread, `frames` pictures ahead.
    ///
    /// The thread opens the file again through `factory`. An export-clock
    /// read then waits on it instead of decoding on the caller's thread, and
    /// moving forward finds the next picture already decoded. Playback-clock
    /// reads, filmstrip cells, and waveforms are unchanged.
    pub fn read_ahead(
        &mut self,
        factory: &ReaderFactory,
        frames: usize,
    ) -> Result<(), EngineError> {
        let media = factory.open(&self.path)?;
        self.ahead = Some(ReadAhead::start(media, self.map.source.clone(), frames)?);
        Ok(())
    }

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

    /// Clockwise quarter turn from the file. [`DisplayQuarter::Zero`] leaves
    /// the decoded picture unchanged.
    #[must_use]
    pub const fn display(&self) -> DisplayQuarter {
        self.display
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
        to_picture(&self.frame_at(time)?)
    }

    /// Wait for the picture asked for, or not. See [`Self::take_stale`].
    pub fn set_patient(&self, patient: bool) {
        self.patient.set(patient);
    }

    /// Whether a read since the last call returned an earlier picture than
    /// the one asked for, because the clip's thread was still decoding it.
    /// The flag clears.
    pub fn take_stale(&self) -> bool {
        self.stale.replace(false)
    }

    /// Start decoding the picture at `time` on the clip's thread without
    /// waiting. A clip without a thread, or a time outside the clip, is left
    /// alone.
    pub fn prefetch(&self, time: Micros) {
        if let (Some(ahead), Ok(source)) = (&self.ahead, map::map_timeline(&self.map, time)) {
            ahead.prefetch(source);
        }
    }

    /// The decoded picture at `time`, before color conversion, for a pass
    /// that converts on the GPU. It is cached and decoded like
    /// [`Self::picture_at`].
    pub fn decoded_at(&self, time: Micros) -> Result<DecodedFrame, EngineError> {
        self.frame_at(time)
    }

    /// The decoded picture at `time`. A clip that is not patient and whose
    /// thread has not decoded it yet returns the stored picture nearest it
    /// and marks itself stale. A clip with nothing stored waits.
    fn frame_at(&self, time: Micros) -> Result<DecodedFrame, EngineError> {
        let source = map::map_timeline(&self.map, time)?;
        let cached = self.cache.borrow().get(source.0).cloned();
        if let Some(frame) = cached {
            return Ok(frame);
        }
        if self.clock.borrow().kind() == ClockKind::Playback {
            return Err(EngineError::NotReady);
        }
        let held = match &self.ahead {
            Some(ahead) if !self.patient.get() => match ahead.try_picture(source)? {
                Some(held) => held,
                None => {
                    let shown = ahead
                        .nearest(source)
                        .map(|held| held.frame)
                        .or_else(|| self.cache.borrow().nearest(source.0).cloned());
                    if let Some(shown) = shown {
                        self.stale.set(true);
                        return Ok(shown);
                    }
                    ahead.picture(source)?
                }
            },
            Some(ahead) => ahead.picture(source)?,
            None => self.decode(source)?,
        };
        let frame = held.frame.clone();
        self.cache
            .borrow_mut()
            .insert(held.start.0, held.end.0, held.frame);
        Ok(frame)
    }

    /// The filmstrip cell for `time`, if a fill already stored one.
    ///
    /// A miss is [`EngineError::NotReady`] and does not decode. A time outside
    /// the segment is [`EngineError::Outside`].
    pub fn strip_at(&self, time: Micros) -> Result<StripCell, EngineError> {
        let source = map::map_timeline(&self.map, time)?;
        self.preload.borrow().strip(source)
    }

    /// Decode `time` and store one filmstrip cell.
    ///
    /// The cell's long side is at most [`crate::MAX_EDGE`]. A platform buffer is
    /// [`EngineError::NotReady`] and is not stored. This does not wait on the
    /// audio clock and does not fill the playback window.
    pub fn fill_strip(&self, time: Micros) -> Result<(), EngineError> {
        let source = map::map_timeline(&self.map, time)?;
        let held = self.decode(source)?;
        let picture = decoded_picture(&held.frame)?;
        let Picture::Cpu(bytes) = picture else {
            return Err(EngineError::NotReady);
        };
        let cell = filmstrip::fit(held.start, bytes.width(), bytes.height(), bytes.rgba())?;
        self.preload.borrow_mut().store_picture(cell);
        Ok(())
    }

    /// The waveform already stored for this file.
    ///
    /// A miss is [`EngineError::NotReady`] and does not decode.
    pub fn waveform(&self) -> Result<Waveform, EngineError> {
        self.preload.borrow().waveform()
    }

    /// Read a bounded prefix of the file and store its peaks.
    ///
    /// A file with no audio is [`EngineError::NotMedia`]. The peaks cover the
    /// frames that were read, at most the mix window. A full window sets
    /// [`Waveform::capped`] when another buffer remains, and that buffer is
    /// not folded into the peaks. This does not decode a picture.
    pub fn fill_waveform(&self) -> Result<(), EngineError> {
        let wave = {
            let mut media = self.media.borrow_mut();
            peaks_from(&mut media, MAX_AUDIO_FRAMES)?
        };
        self.preload.borrow_mut().store_waveform(wave);
        Ok(())
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
///
/// A relative path is opened from the process directory. [`open_segment_in`]
/// joins that path to a project folder instead.
pub fn open_segment(
    factory: &ReaderFactory,
    draft: &Draft,
    id: &SegmentId,
    clock: ClockKind,
    window: usize,
) -> Result<VideoSource, EngineError> {
    open_segment_in(factory, draft, id, clock, window, None)
}

/// Opens the segment's file. A relative path stays inside `root`.
///
/// An absolute path is opened as stored. Without `root`, a relative path is
/// opened from the process directory. `..` is refused when `root` is set.
pub fn open_segment_in(
    factory: &ReaderFactory,
    draft: &Draft,
    id: &SegmentId,
    clock: ClockKind,
    window: usize,
    root: Option<&Path>,
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
    let Some(stored) = map::media_path(draft, segment) else {
        return Err(EngineError::MissingFile {
            path: PathBuf::new(),
        });
    };
    let path = located_media(root, stored).ok_or_else(|| EngineError::MissingFile {
        path: PathBuf::from(stored),
    })?;
    let media = factory.open(&path)?;
    let display = media.info().display;
    let map = map::placement_of(draft, segment)?;
    Ok(VideoSource {
        path,
        media: RefCell::new(media),
        map,
        cache: RefCell::new(FrameWindow::new(window)?),
        clock: RefCell::new(MediaClock::new(clock)),
        preload: RefCell::new(Preload::new()),
        decodes: Cell::new(0),
        display,
        ahead: None,
        patient: Cell::new(true),
        stale: Cell::new(false),
    })
}

/// A relative media path joined to `root`. `..` and a path outside `root` are refused.
#[must_use]
pub fn media_in_project(root: &Path, stored: &str) -> Option<PathBuf> {
    let path = Path::new(stored);
    if stored.is_empty() || path.is_absolute() {
        return None;
    }
    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return None;
    }
    let joined = root.join(path);
    let root = root.canonicalize().ok()?;
    let canonical = joined.canonicalize().ok()?;
    canonical.starts_with(&root).then_some(canonical)
}

fn located_media(root: Option<&Path>, stored: &str) -> Option<PathBuf> {
    if stored.is_empty() {
        return None;
    }
    let path = Path::new(stored);
    if path.is_absolute() {
        return Some(path.to_path_buf());
    }
    match root {
        Some(root) => media_in_project(root, stored),
        None => Some(PathBuf::from(stored)),
    }
}

/// Peaks for at most `frames` from the start of `media`.
///
/// A shorter file is not marked capped. A file that continues past `frames`
/// is capped, and the next buffer is discarded.
pub(crate) fn peaks_from(media: &mut Media, frames: u32) -> Result<Waveform, EngineError> {
    let pcm = media.audio_from(Micros(0), frames)?;
    let capped = if pcm.samples == frames {
        media.pull_audio()?.is_some()
    } else {
        false
    };
    let Some(samples) = pcm.as_f32() else {
        return Err(EngineError::Layout);
    };
    Waveform::from_interleaved(samples, pcm.layout.len(), MAX_BUCKETS, pcm.samples, capped)
}

/// The CPU or platform picture for one decoded frame.
pub fn decoded_picture(frame: &DecodedFrame) -> Result<Picture, EngineError> {
    to_picture(frame)
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
/// samples. The straight sample is mapped, then partial alpha is
/// `(channel * alpha + 127) / 255`. A gamma of zero, with no sRGB intent and
/// no readable profile, is a color miss.
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
    let map = cutlass_color::StillMap::prepare(&cutlass_color::StillTags {
        gamma_times_100000: image.gamma_times_100000,
        chromaticity: image.chromaticity,
        srgb_intent: image.srgb_intent,
        icc_profile: &image.icc_profile,
    })
    .ok()?;
    let mut rgba = vec![0_u8; rgba_len(image.width, image.height)?];
    for (pixel, chunk) in image.samples.chunks(channels).enumerate() {
        let (red, green, blue, alpha) = straight(image, chunk)?;
        let [red, green, blue] = map.map_rgb([red, green, blue]);
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
        PixelFormat::Yuv422p => "yuv422p",
        PixelFormat::Yuv440p => "yuv440p",
        PixelFormat::Yuv444p => "yuv444p",
        PixelFormat::Gbrp => "gbrp",
        PixelFormat::Yuv420p10 => "yuv420p10",
        PixelFormat::Yuv420p12 => "yuv420p12",
        PixelFormat::Yuv422p10 => "yuv422p10",
        PixelFormat::Yuv422p12 => "yuv422p12",
        PixelFormat::Yuv440p10 => "yuv440p10",
        PixelFormat::Yuv440p12 => "yuv440p12",
        PixelFormat::Yuv444p10 => "yuv444p10",
        PixelFormat::Yuv444p12 => "yuv444p12",
        PixelFormat::Gbrp10 => "gbrp10",
        PixelFormat::Gbrp12 => "gbrp12",
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
    fn a_relative_path_stays_inside_the_project_and_a_parent_is_refused() {
        let root = std::env::temp_dir().join(format!("cutlass-media-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Resources")).expect("root");
        std::fs::write(root.join("Resources/clip.png"), b"png").expect("file");
        let resolved = media_in_project(&root, "Resources/clip.png").expect("inside");
        assert!(resolved.ends_with("Resources/clip.png"));
        assert!(media_in_project(&root, "../clip.png").is_none());
        assert!(media_in_project(&root, "/tmp/clip.png").is_none());
        assert!(media_in_project(&root, "").is_none());
        let _ = std::fs::remove_dir_all(&root);
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
    fn reference_gamma_and_an_unreadable_profile_keep_the_samples() {
        let mut tagged = row(ColorType::Rgb, BitDepth::Eight, vec![1, 2, 3]);
        tagged.gamma_times_100000 = Some(45_455);
        tagged.icc_profile = vec![1, 2, 3, 4];
        assert_eq!(rgba(tagged), vec![1, 2, 3, 255]);
    }

    #[test]
    fn linear_gamma_is_encoded_before_premultiply() {
        let mut gray = row(ColorType::Gray, BitDepth::Eight, vec![128]);
        gray.gamma_times_100000 = Some(100_000);
        assert_eq!(rgba(gray), vec![186, 186, 186, 255]);

        let mut partial = row(ColorType::GrayAlpha, BitDepth::Eight, vec![100, 128]);
        partial.gamma_times_100000 = Some(100_000);
        assert_eq!(rgba(partial), vec![84, 84, 84, 128]);

        let mut wide = row(ColorType::Gray, BitDepth::Sixteen, vec![32_768]);
        wide.gamma_times_100000 = Some(100_000);
        assert_eq!(rgba(wide), vec![186, 186, 186, 255]);

        let mut intent = row(ColorType::Rgb, BitDepth::Eight, vec![128, 10, 20]);
        intent.gamma_times_100000 = Some(100_000);
        intent.srgb_intent = Some(0);
        assert_eq!(rgba(intent), vec![128, 10, 20, 255]);

        let mut zero_intent = row(ColorType::Rgb, BitDepth::Eight, vec![128, 10, 20]);
        zero_intent.gamma_times_100000 = Some(0);
        zero_intent.srgb_intent = Some(0);
        assert_eq!(rgba(zero_intent), vec![128, 10, 20, 255]);

        let mut broken = row(ColorType::Gray, BitDepth::Eight, vec![128]);
        broken.gamma_times_100000 = Some(100_000);
        broken.icc_profile = vec![1, 2, 3, 4];
        assert_eq!(rgba(broken), vec![186, 186, 186, 255]);

        let mut zero = row(ColorType::Gray, BitDepth::Eight, vec![128]);
        zero.gamma_times_100000 = Some(0);
        assert!(matches!(
            still_picture(&zero),
            Err(EngineError::ColorMiss { .. })
        ));
    }

    #[test]
    fn chromaticity_and_a_profile_move_the_straight_sample() {
        let mut shifted = row(ColorType::Rgb, BitDepth::Eight, vec![180, 80, 40]);
        shifted.gamma_times_100000 = Some(45_455);
        shifted.chromaticity = Some([
            31_270, 32_900, 50_000, 30_000, 30_000, 60_000, 15_000, 6_000,
        ]);
        assert_eq!(rgba(shifted), vec![180, 93, 86, 255]);

        let mut ignored = row(ColorType::Gray, BitDepth::Eight, vec![128]);
        ignored.gamma_times_100000 = Some(100_000);
        ignored.chromaticity = Some([31_270, 0, 64_000, 33_000, 30_000, 60_000, 15_000, 6_000]);
        assert_eq!(rgba(ignored), vec![186, 186, 186, 255]);

        let mut profiled = row(ColorType::Gray, BitDepth::Eight, vec![128]);
        profiled.gamma_times_100000 = Some(45_455);
        profiled.srgb_intent = Some(1);
        profiled.icc_profile = linear_bt709_profile();
        assert_eq!(rgba(profiled), vec![186, 186, 186, 255]);

        let mut indexed = row(ColorType::Indexed, BitDepth::Eight, vec![0]);
        indexed.palette = vec![Rgb8 {
            red: 128,
            green: 128,
            blue: 128,
        }];
        indexed.gamma_times_100000 = Some(100_000);
        assert_eq!(rgba(indexed), vec![186, 186, 186, 255]);
    }

    fn linear_bt709_profile() -> Vec<u8> {
        let columns = [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ];
        let illuminant = [0.95047, 1.0, 1.08883];
        let signatures: [[u8; 4]; 6] = [*b"rXYZ", *b"gXYZ", *b"bXYZ", *b"rTRC", *b"gTRC", *b"bTRC"];
        let mut tags = Vec::new();
        for column in columns {
            tags.push(xyz_tag(column));
        }
        for _ in 0..3 {
            tags.push(curv_tag(256));
        }
        let mut body = vec![0_u8; 128];
        body[16..20].copy_from_slice(b"RGB ");
        body[20..24].copy_from_slice(b"XYZ ");
        body[36..40].copy_from_slice(b"acsp");
        for (index, value) in illuminant.iter().enumerate() {
            let at = 68 + index * 4;
            body[at..at + 4].copy_from_slice(&fixed(*value).to_be_bytes());
        }
        let mut cursor = 132 + signatures.len() * 12;
        let mut entries = Vec::new();
        for (signature, tag) in signatures.iter().zip(&tags) {
            entries.push((*signature, cursor, tag.len()));
            cursor = cursor.saturating_add(tag.len()).div_ceil(4) * 4;
        }
        body.extend(
            u32::try_from(signatures.len())
                .expect("count")
                .to_be_bytes(),
        );
        for (signature, offset, size) in &entries {
            body.extend(signature);
            body.extend(u32::try_from(*offset).expect("offset").to_be_bytes());
            body.extend(u32::try_from(*size).expect("size").to_be_bytes());
        }
        for tag in tags {
            let pad = (4 - tag.len() % 4) % 4;
            body.extend(tag);
            body.extend(std::iter::repeat_n(0, pad));
        }
        let size = u32::try_from(body.len()).expect("size");
        body[..4].copy_from_slice(&size.to_be_bytes());
        body
    }

    fn xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
        let mut tag = Vec::new();
        tag.extend(b"XYZ ");
        tag.extend([0, 0, 0, 0]);
        for value in xyz {
            tag.extend(fixed(value).to_be_bytes());
        }
        tag
    }

    fn curv_tag(gamma_8_8: u16) -> Vec<u8> {
        let mut tag = Vec::new();
        tag.extend(b"curv");
        tag.extend([0, 0, 0, 0]);
        tag.extend(1_u32.to_be_bytes());
        tag.extend(gamma_8_8.to_be_bytes());
        tag
    }

    fn fixed(value: f64) -> i32 {
        value.mul_add(65_536.0, 0.0).round() as i32
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
