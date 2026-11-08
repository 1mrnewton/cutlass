//! A movie made of pictures handed in one at a time, not of a timeline:
//! what a job that changes every frame of a clip, such as Enhance quality,
//! writes its result as.
//!
//! The movie is H.264 in MP4, pictures only, at one size and rate. It is
//! written beside its destination under a hidden part name and moved there
//! only once it is finished, as [`crate::write_movie`] does.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use cutlass_codec::{CodecId, FrameSize, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_encode::{CodecParams, CpuFrame, EncodeFactory, FrameInput, PixelFormat};
use cutlass_isobmff::{FileBrand, Mp4Mux};
use cutlass_model::FrameRate;

use crate::config::{ExportError, MAX_EDGE, MAX_FRAMES};
use crate::movie::part_path;
use crate::run::{stop, video_time, write_packets};

/// The shape of a picture movie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PictureMovie {
    /// Even, as H.264 codes it, and at most [`MAX_EDGE`].
    pub width: u32,
    pub height: u32,
    pub frame_rate: FrameRate,
    /// Bits a second.
    pub bitrate: u32,
    /// Pictures in all, at least one.
    pub frames: u64,
}

/// Write `movie` to `destination`, picture `k` from `picture(k)` as RGBA
/// `width` by `height`. `progress` hears the pictures done and the
/// pictures in all after each, and stops the movie by returning false;
/// a stopped or failed movie leaves nothing behind.
pub fn write_pictures(
    encoders: &EncodeFactory,
    movie: &PictureMovie,
    destination: &Path,
    picture: &mut dyn FnMut(u64) -> Result<Vec<u8>, ExportError>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), ExportError> {
    check(movie)?;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = write_part(encoders, movie, &part, picture, progress)
        .and_then(|()| fs::rename(&part, destination).map_err(io));
    if written.is_err() {
        let _ = fs::remove_file(&part);
    }
    written
}

fn check(movie: &PictureMovie) -> Result<(), ExportError> {
    let PictureMovie {
        width,
        height,
        frame_rate,
        frames,
        ..
    } = *movie;
    let even = |side: u32| side >= 2 && side.is_multiple_of(2) && side <= MAX_EDGE;
    if !even(width) || !even(height) {
        return Err(ExportError::Canvas);
    }
    if frame_rate.numerator == 0 || frame_rate.denominator == 0 {
        return Err(ExportError::FrameRate);
    }
    if frames == 0 || frames > MAX_FRAMES {
        return Err(ExportError::Limit);
    }
    Ok(())
}

fn write_part(
    encoders: &EncodeFactory,
    movie: &PictureMovie,
    part: &Path,
    picture: &mut dyn FnMut(u64) -> Result<Vec<u8>, ExportError>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), ExportError> {
    let mut params =
        CodecParams::with_format(CodecId::H264, movie.width, movie.height, PixelFormat::Rgba8);
    params.bitrate = movie.bitrate;
    let mut encoder = encoders.open(&params).map_err(ExportError::Encoder)?;
    let file = File::create(part).map_err(io)?;
    let mut mux = Mp4Mux::with_brand(BufWriter::new(file), FileBrand::Mp4);
    let size = FrameSize::try_new(movie.width, movie.height).map_err(|_| ExportError::Canvas)?;
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(CodecId::H264), size, Vec::new()),
            timescale: Timescale::new(movie.frame_rate.numerator),
            language: None,
        })
        .map_err(ExportError::Mux)?;
    let mut ticks = 0;
    let encoded = (|| {
        let len = usize::try_from(u64::from(movie.width) * u64::from(movie.height) * 4)
            .map_err(|_| ExportError::Limit)?;
        for index in 0..movie.frames {
            let rgba = picture(index)?;
            if rgba.len() != len {
                return Err(ExportError::Picture(format!(
                    "picture {index} is {} bytes, not {len}",
                    rgba.len()
                )));
            }
            let at = i64::try_from(index).map_err(|_| ExportError::Limit)?;
            let frame = CpuFrame::new(
                movie.width,
                movie.height,
                PixelFormat::Rgba8,
                rgba,
                video_time(movie.frame_rate, at)?,
            )
            .map_err(ExportError::Encoder)?;
            let packets = encoder
                .push_frame(&FrameInput::Cpu(frame))
                .map_err(ExportError::Encoder)?;
            write_packets(&mut mux, track.get(), &packets, &mut ticks)?;
            if !progress(index + 1, movie.frames) {
                return Err(ExportError::Cancelled);
            }
        }
        let packets = encoder.drain().map_err(ExportError::Encoder)?;
        write_packets(&mut mux, track.get(), &packets, &mut ticks)?;
        mux.interleave().map_err(ExportError::Mux)?;
        mux.finish(&[TrackDuration {
            track,
            timescale: Timescale::new(movie.frame_rate.numerator),
            ticks,
        }])
        .map_err(|failed| ExportError::Mux(failed.reason().clone()))
    })();
    if let Err(error) = encoded {
        return Err(stop(&mut mux, error));
    }
    let mut writer = mux.into_writer().map_err(ExportError::Mux)?;
    writer.flush().map_err(io)?;
    writer
        .into_inner()
        .map_err(|error| io(error.into_error()))?
        .sync_all()
        .map_err(io)
}

fn io(error: std::io::Error) -> ExportError {
    ExportError::File(error.to_string())
}

#[cfg(test)]
mod tests;
