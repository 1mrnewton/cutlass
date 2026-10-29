//! `PictureSource` over decoded pictures.
//!
//! Text and shapes never open a decoder. A platform buffer stays a platform
//! buffer. CPU pixels go through `cutlass-color`; a layout it cannot convert
//! is a named miss.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use cutlass_codec::access::{
    BitDepth, ColorPrimaries, ColorRange, ColorType, MatrixCoefficients, PixelFormat, Transfer,
    Yuv420p,
};
use cutlass_color::{Input, Matrix, Outcome, Picture as ColorPicture, Request};
use cutlass_decode::StillImage;
use cutlass_gpu::Extent;
use cutlass_model::{Draft, Micros, SegmentId};
use cutlass_picture::{CpuBytes, Picture, PictureSource, PlatformBuffer, Sample};

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
        DecodedFrame::Still(image) => Err(EngineError::ColorMiss {
            layout: still_layout(image).to_owned(),
        }),
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
