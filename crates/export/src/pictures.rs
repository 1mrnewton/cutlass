//! A movie made of pictures handed in one at a time, not of a timeline:
//! what a job that changes every frame of a clip, such as Enhance quality,
//! writes its result as.
//!
//! The movie is H.264 in MP4, pictures only, at one size, evenly at one
//! rate or at the times it is handed. It is
//! written beside its destination under a hidden part name and moved there
//! only once it is finished, as [`crate::write_movie`] does.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use cutlass_codec::{CodecId, FrameSize, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_encode::{CodecParams, CpuFrame, EncodeFactory, FrameInput, FrameTime, PixelFormat};
use cutlass_isobmff::{FileBrand, Mp4Mux};
use cutlass_model::{FrameRate, Micros};

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
    if movie.frame_rate.numerator == 0 || movie.frame_rate.denominator == 0 {
        return Err(ExportError::FrameRate);
    }
    write(
        encoders,
        movie,
        Clock::Rate(movie.frame_rate),
        destination,
        picture,
        progress,
    )
}

/// [`write_pictures`] with picture `k` shown from `times[k]` until
/// `times[k + 1]`, microseconds from the movie's start: one more time than
/// pictures, rising from 0. A movie that follows another file's own
/// frames, however unevenly they come, has a picture for each at its time.
/// `movie.frame_rate` is not used.
pub fn write_pictures_at(
    encoders: &EncodeFactory,
    movie: &PictureMovie,
    times: &[Micros],
    destination: &Path,
    picture: &mut dyn FnMut(u64) -> Result<Vec<u8>, ExportError>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), ExportError> {
    let counted = u64::try_from(times.len()).map_err(|_| ExportError::Limit)?;
    if counted != movie.frames.saturating_add(1)
        || times.first() != Some(&Micros(0))
        || times.windows(2).any(|pair| pair[1].0 <= pair[0].0)
    {
        return Err(ExportError::FrameRate);
    }
    write(
        encoders,
        movie,
        Clock::Times(times),
        destination,
        picture,
        progress,
    )
}

/// When each picture of a movie shows.
#[derive(Clone, Copy)]
enum Clock<'a> {
    /// Evenly, this many a second.
    Rate(FrameRate),
    /// From each time to the next, in microseconds.
    Times(&'a [Micros]),
}

impl Clock<'_> {
    fn timescale(self) -> u32 {
        match self {
            Self::Rate(rate) => rate.numerator,
            Self::Times(_) => 1_000_000,
        }
    }

    fn at(self, index: u64) -> Result<FrameTime, ExportError> {
        let at = i64::try_from(index).map_err(|_| ExportError::Limit)?;
        match self {
            Self::Rate(rate) => video_time(rate, at),
            Self::Times(times) => {
                let k = usize::try_from(index).map_err(|_| ExportError::Limit)?;
                let (Some(start), Some(end)) = (times.get(k), times.get(k + 1)) else {
                    return Err(ExportError::Limit);
                };
                FrameTime::new(start.0, end.0 - start.0, 1_000_000)
                    .map_err(|_| ExportError::FrameRate)
            }
        }
    }
}

fn write(
    encoders: &EncodeFactory,
    movie: &PictureMovie,
    clock: Clock<'_>,
    destination: &Path,
    picture: &mut dyn FnMut(u64) -> Result<Vec<u8>, ExportError>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), ExportError> {
    check(movie)?;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = write_part(encoders, movie, clock, &part, picture, progress)
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
        frames,
        ..
    } = *movie;
    let even = |side: u32| side >= 2 && side.is_multiple_of(2) && side <= MAX_EDGE;
    if !even(width) || !even(height) {
        return Err(ExportError::Canvas);
    }
    if frames == 0 || frames > MAX_FRAMES {
        return Err(ExportError::Limit);
    }
    Ok(())
}

fn write_part(
    encoders: &EncodeFactory,
    movie: &PictureMovie,
    clock: Clock<'_>,
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
            timescale: Timescale::new(clock.timescale()),
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
            let frame = CpuFrame::new(
                movie.width,
                movie.height,
                PixelFormat::Rgba8,
                rgba,
                clock.at(index)?,
            )
            .map_err(ExportError::Encoder)?;
            let mut packets = encoder
                .push_frame(&FrameInput::Cpu(frame))
                .map_err(ExportError::Encoder)?;
            // A picture held back to be coded between two others would be
            // stored with a neighbour's length: uneven times go out one by
            // one, in order.
            if let Clock::Times(_) = clock {
                packets.extend(encoder.drain().map_err(ExportError::Encoder)?);
            }
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
            timescale: Timescale::new(clock.timescale()),
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
