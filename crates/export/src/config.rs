//! The export request: resolution, codec, bitrate, frame rate, and audio.
//!
//! The long side is one of a fixed list. GIF uses a shorter list. A step
//! outside that list is refused. The other edge keeps the canvas aspect
//! with integer division.

use std::fmt;

use cutlass_codec::{CodecId, ContainerId};
use cutlass_container::MuxError;
use cutlass_encode::CodecError;
use cutlass_eval::AudioError;
use cutlass_model::{FrameRate, TimeRange};

/// Video long-side steps, in pixels.
pub const VIDEO_LONG_SIDES: [u32; 6] = [480, 720, 1080, 1440, 2160, 4320];

/// GIF long-side steps, in pixels.
pub const GIF_LONG_SIDES: [u32; 3] = [240, 320, 640];

/// Most frames one export will encode.
///
/// A longer range is refused before a frame is evaluated. Two million frames
/// is a little over nine hours at 60 fps.
pub const MAX_FRAMES: u64 = 2_000_000;

/// Audio settings carried on the request. Absent audio writes no audio track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioExport {
    pub codec: CodecId,
    pub bitrate: u32,
    pub sample_rate: u32,
    pub channels: u16,
    pub limiter: i32,
}

/// Which subtitle file to write. ASS is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarKind {
    /// SubRip cues.
    Srt,
    /// Cue text, one cue per line.
    Plain,
    /// Not written. Styled motion stays in the burned frame.
    Ass,
}

/// Sidecar file stem. The extension is chosen from [`SidecarKind`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SidecarRequest {
    pub kind: SidecarKind,
    pub name: String,
}

/// Resolution, codec, bitrate, frame rate, container, and audio.
///
/// `late_by` is added to each frame's request time before the export clock
/// sees it. A positive value is a late picture. Export still keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportRequest {
    pub range: TimeRange,
    pub long_side: u32,
    pub codec: CodecId,
    pub bitrate: u32,
    pub frame_rate: FrameRate,
    pub container: ContainerId,
    pub audio: Option<AudioExport>,
    pub sidecar: Option<SidecarRequest>,
    pub late_by: i64,
}

/// Why export did not write a file.
#[derive(Debug, PartialEq, Eq)]
pub enum ExportError {
    /// The long side is not a step for this codec.
    Resolution { long_side: u32 },
    /// The frame rate has a zero numerator or denominator.
    FrameRate,
    /// The range is empty, negative, or does not fit in the timeline clock.
    Range,
    /// The canvas is not a picture size.
    Canvas,
    /// A count or a product does not fit.
    Limit,
    /// The encode factory declined, or an encoder refused a frame.
    Encoder(CodecError),
    /// The mix did not return samples.
    Audio(AudioError),
    /// The channel count is not mono or stereo, or the sample rate is zero.
    AudioLayout,
    /// The mux trait refused the stream or a packet.
    Mux(MuxError),
    /// A packet timestamp could not be stored in the file timescale.
    Packet,
    /// The encoder attached side data this writer does not store.
    SideData,
    /// The mux family is not the container on the request.
    Container,
    /// The sidecar name leaves the destination directory.
    Path,
    /// The sidecar file could not be written.
    Io(String),
    /// ASS is an import. It is not written.
    Ass,
    /// Word columns do not have one end and one string per start.
    Truncated,
    /// The export clock did not accept the picture.
    Wait,
    /// The device or the compositor refused the frame.
    Picture(String),
    /// HDR is not the software RGBA comparison.
    HdrNotThisCase,
    /// The bytes are not tightly packed RGBA.
    NotSoftwareRgba,
    /// Playback and export RGBA differ.
    PixelMismatch,
}

impl fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution { long_side } => {
                write!(formatter, "long side {long_side} is not an export step")
            }
            Self::FrameRate => formatter.write_str("the frame rate is empty"),
            Self::Range => formatter.write_str("the export range is not a span"),
            Self::Canvas => formatter.write_str("the canvas is not a picture"),
            Self::Limit => formatter.write_str("the export is past its frame limit"),
            Self::Encoder(error) => write!(formatter, "{error}"),
            Self::Audio(error) => write!(formatter, "{error}"),
            Self::AudioLayout => formatter.write_str("the audio layout is not mono or stereo"),
            Self::Mux(error) => write!(formatter, "{error}"),
            Self::Packet => formatter.write_str("the packet clock does not fit the file"),
            Self::SideData => formatter.write_str("export does not store packet side data"),
            Self::Container => formatter.write_str("the mux is a different container"),
            Self::Path => formatter.write_str("the sidecar path leaves the destination"),
            Self::Io(message) => write!(formatter, "sidecar write failed: {message}"),
            Self::Ass => formatter.write_str("ASS is not written"),
            Self::Truncated => formatter.write_str("caption words are truncated"),
            Self::Wait => formatter.write_str("export did not accept the frame"),
            Self::Picture(message) => write!(formatter, "{message}"),
            Self::HdrNotThisCase => formatter.write_str("HDR is not the software pixel comparison"),
            Self::NotSoftwareRgba => formatter.write_str("the bytes are not software RGBA"),
            Self::PixelMismatch => formatter.write_str("playback and export pixels differ"),
        }
    }
}

impl std::error::Error for ExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoder(error) => Some(error),
            Self::Audio(error) => Some(error),
            Self::Mux(error) => Some(error),
            _ => None,
        }
    }
}

/// Refuses a long side, a frame rate, a range, or an audio layout that export
/// will not run.
pub fn validate(request: &ExportRequest) -> Result<(), ExportError> {
    let steps = if request.codec == CodecId::Gif {
        GIF_LONG_SIDES.as_slice()
    } else {
        VIDEO_LONG_SIDES.as_slice()
    };
    if !steps.contains(&request.long_side) {
        return Err(ExportError::Resolution {
            long_side: request.long_side,
        });
    }
    if request.frame_rate.numerator == 0 || request.frame_rate.denominator == 0 {
        return Err(ExportError::FrameRate);
    }
    if request.range.duration.0 <= 0 {
        return Err(ExportError::Range);
    }
    if request
        .range
        .start
        .0
        .checked_add(request.range.duration.0)
        .is_none()
    {
        return Err(ExportError::Range);
    }
    if let Some(audio) = &request.audio
        && (audio.sample_rate == 0 || (audio.channels != 1 && audio.channels != 2))
    {
        return Err(ExportError::AudioLayout);
    }
    Ok(())
}

/// Output size for a canvas and a long-side step.
///
/// The longer canvas edge becomes `long_side`. The other edge is
/// `short * long_side / long`, truncated toward zero, and at least 1.
pub fn output_size(
    canvas_w: i32,
    canvas_h: i32,
    long_side: u32,
) -> Result<(u32, u32), ExportError> {
    let width = u32::try_from(canvas_w).map_err(|_| ExportError::Canvas)?;
    let height = u32::try_from(canvas_h).map_err(|_| ExportError::Canvas)?;
    if width == 0 || height == 0 || long_side == 0 {
        return Err(ExportError::Canvas);
    }
    if width >= height {
        let short = scale_edge(height, width, long_side)?;
        Ok((long_side, short))
    } else {
        let short = scale_edge(width, height, long_side)?;
        Ok((short, long_side))
    }
}

fn scale_edge(short: u32, long: u32, long_side: u32) -> Result<u32, ExportError> {
    let value = u64::from(short) * u64::from(long_side) / u64::from(long);
    let value = u32::try_from(value).map_err(|_| ExportError::Limit)?;
    if value == 0 {
        return Err(ExportError::Canvas);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{
        AudioExport, ExportError, ExportRequest, GIF_LONG_SIDES, VIDEO_LONG_SIDES, output_size,
        validate,
    };
    use cutlass_codec::{CodecId, ContainerId};
    use cutlass_model::{FrameRate, TimeRange};

    fn request(codec: CodecId, long_side: u32) -> ExportRequest {
        ExportRequest {
            range: TimeRange::new(0, 1_000_000),
            long_side,
            codec,
            bitrate: 2_000_000,
            frame_rate: FrameRate::FPS_30,
            container: ContainerId::Isobmff,
            audio: None,
            sidecar: None,
            late_by: 0,
        }
    }

    #[test]
    fn video_and_gif_accept_only_their_long_side_steps() {
        for side in VIDEO_LONG_SIDES {
            assert!(validate(&request(CodecId::H264, side)).is_ok(), "{side}");
            let gif = validate(&request(CodecId::Gif, side));
            assert_eq!(gif, Err(ExportError::Resolution { long_side: side }));
        }
        for side in GIF_LONG_SIDES {
            assert!(validate(&request(CodecId::Gif, side)).is_ok(), "{side}");
            let video = validate(&request(CodecId::H264, side));
            assert_eq!(video, Err(ExportError::Resolution { long_side: side }));
        }
        assert_eq!(
            validate(&request(CodecId::H264, 1000)),
            Err(ExportError::Resolution { long_side: 1000 })
        );
    }

    #[test]
    fn a_zero_rate_an_empty_range_and_a_bad_layout_are_refused() {
        let mut empty_rate = request(CodecId::H264, 1080);
        empty_rate.frame_rate = FrameRate {
            numerator: 0,
            denominator: 1,
        };
        assert_eq!(validate(&empty_rate), Err(ExportError::FrameRate));

        let mut empty_range = request(CodecId::H264, 1080);
        empty_range.range = TimeRange::new(0, 0);
        assert_eq!(validate(&empty_range), Err(ExportError::Range));

        let mut overflow = request(CodecId::H264, 1080);
        overflow.range = TimeRange::new(1, i64::MAX);
        assert_eq!(validate(&overflow), Err(ExportError::Range));

        let mut audio = request(CodecId::H264, 1080);
        audio.audio = Some(AudioExport {
            codec: CodecId::Pcm,
            bitrate: 128_000,
            sample_rate: 0,
            channels: 1,
            limiter: 0,
        });
        assert_eq!(validate(&audio), Err(ExportError::AudioLayout));
        audio.audio.as_mut().expect("audio").sample_rate = 48_000;
        audio.audio.as_mut().expect("audio").channels = 6;
        assert_eq!(validate(&audio), Err(ExportError::AudioLayout));
    }

    #[test]
    fn the_long_side_sets_the_longer_edge_and_truncates_the_other() {
        assert_eq!(output_size(1920, 1080, 1080), Ok((1080, 607)));
        assert_eq!(output_size(1080, 1920, 1080), Ok((607, 1080)));
        assert_eq!(output_size(480, 270, 480), Ok((480, 270)));
        assert_eq!(output_size(100, 100, 480), Ok((480, 480)));
        assert_eq!(output_size(0, 10, 480), Err(ExportError::Canvas));
        assert_eq!(output_size(-1, 10, 480), Err(ExportError::Canvas));
        assert_eq!(output_size(100_000, 1, 480), Err(ExportError::Canvas));
    }
}
