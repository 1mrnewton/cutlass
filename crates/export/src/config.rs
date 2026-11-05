//! The export request: resolution, codec, bitrate, frame rate, and audio.
//!
//! The short side is one of a fixed list, so "1080" on a 16:9 canvas is
//! 1920 by 1080, as phone editors name sizes. GIF uses a smaller list. A
//! step outside that list is refused. The other edge keeps the canvas
//! aspect with integer division.

use std::fmt;

use cutlass_codec::{CodecId, ContainerId};
use cutlass_container::MuxError;
use cutlass_encode::CodecError;
use cutlass_eval::AudioError;
use cutlass_model::{FrameRate, TimeRange};

/// Video short-side steps, in pixels.
pub const VIDEO_SHORT_SIDES: [u32; 6] = [480, 720, 1080, 1440, 2160, 4320];

/// GIF short-side steps, in pixels.
pub const GIF_SHORT_SIDES: [u32; 3] = [240, 320, 640];

/// The longest edge an export may have, in pixels. A canvas thin enough
/// to stretch the long edge past this is refused.
pub const MAX_EDGE: u32 = 16_384;

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
    pub short_side: u32,
    pub codec: CodecId,
    pub bitrate: u32,
    pub frame_rate: FrameRate,
    pub container: ContainerId,
    pub audio: Option<AudioExport>,
    pub sidecar: Option<SidecarRequest>,
    pub late_by: i64,
    /// Hold `bitrate` over each second instead of on average.
    pub constant_rate: bool,
    /// The ProRes flavor when `codec` is ProRes; other codecs ignore it.
    pub prores: ProresProfile,
}

/// The ProRes flavors a movie may be written in, smallest first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProresProfile {
    Proxy,
    Light,
    /// ProRes 422.
    #[default]
    Standard,
    High,
}

impl ProresProfile {
    pub const ALL: [Self; 4] = [Self::Proxy, Self::Light, Self::Standard, Self::High];

    /// The sample entry, which the encoder and the movie file both name.
    #[must_use]
    pub fn code(self) -> [u8; 4] {
        match self {
            Self::Proxy => *b"apco",
            Self::Light => *b"apcs",
            Self::Standard => *b"apcn",
            Self::High => *b"apch",
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Proxy => "Proxy",
            Self::Light => "LT",
            Self::Standard => "422",
            Self::High => "HQ",
        }
    }
}

/// Why export did not write a file.
#[derive(Debug, PartialEq, Eq)]
pub enum ExportError {
    /// The short side is not a step for this codec.
    Resolution { short_side: u32 },
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
    /// The request asked for more than two channels. A surround source is
    /// mixed down to stereo; writing surround is not supported yet.
    SurroundExport { channels: u16 },
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
    /// The progress callback asked export to stop.
    Cancelled,
    /// The movie file could not be written or moved into place.
    File(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution { short_side } => {
                write!(formatter, "short side {short_side} is not an export step")
            }
            Self::FrameRate => formatter.write_str("the frame rate is empty"),
            Self::Range => formatter.write_str("the export range is not a span"),
            Self::Canvas => formatter.write_str("the canvas is not a picture"),
            Self::Limit => formatter.write_str("the export is past its frame limit"),
            Self::Encoder(error) => write!(formatter, "{error}"),
            Self::Audio(error) => write!(formatter, "{error}"),
            Self::AudioLayout => formatter.write_str("the audio layout is not mono or stereo"),
            Self::SurroundExport { channels } => write!(
                formatter,
                "surround export is not supported yet: export writes mono or stereo, \
                 not {channels} channels"
            ),
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
            Self::Cancelled => formatter.write_str("export was stopped"),
            Self::File(message) => write!(formatter, "the movie could not be written: {message}"),
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

/// Refuses a short side, a frame rate, a range, or an audio layout that export
/// will not run.
pub fn validate(request: &ExportRequest) -> Result<(), ExportError> {
    // A WebP is an animation at the GIF sizes or a still at the video ones.
    let allowed = match request.codec {
        CodecId::Gif => GIF_SHORT_SIDES.contains(&request.short_side),
        CodecId::Webp => {
            GIF_SHORT_SIDES.contains(&request.short_side)
                || VIDEO_SHORT_SIDES.contains(&request.short_side)
        }
        _ => VIDEO_SHORT_SIDES.contains(&request.short_side),
    };
    if !allowed {
        return Err(ExportError::Resolution {
            short_side: request.short_side,
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
    if let Some(audio) = &request.audio {
        check_audio(audio)?;
    }
    Ok(())
}

/// Export writes one or two channels at a rate above zero. More than two is
/// surround, which is named on its own so the refusal says so.
pub(crate) fn check_audio(audio: &AudioExport) -> Result<(), ExportError> {
    if audio.channels > 2 {
        return Err(ExportError::SurroundExport {
            channels: audio.channels,
        });
    }
    if audio.sample_rate == 0 || audio.channels == 0 {
        return Err(ExportError::AudioLayout);
    }
    Ok(())
}

/// Output size for a canvas and a short-side step.
///
/// The shorter canvas edge becomes `short_side`; a square canvas is
/// `short_side` both ways. The other edge is `long * short_side / short`,
/// truncated toward zero, and at most [`MAX_EDGE`].
pub fn output_size(
    canvas_w: i32,
    canvas_h: i32,
    short_side: u32,
) -> Result<(u32, u32), ExportError> {
    let width = u32::try_from(canvas_w).map_err(|_| ExportError::Canvas)?;
    let height = u32::try_from(canvas_h).map_err(|_| ExportError::Canvas)?;
    if width == 0 || height == 0 || short_side == 0 {
        return Err(ExportError::Canvas);
    }
    if width >= height {
        let long = scale_edge(width, height, short_side)?;
        Ok((long, short_side))
    } else {
        let long = scale_edge(height, width, short_side)?;
        Ok((short_side, long))
    }
}

/// `size` as `codec` can code it: a codec that halves its colour planes
/// both ways needs even edges, so an odd edge loses one pixel. Other codecs
/// keep the size.
#[must_use]
pub fn coded_size(codec: CodecId, (width, height): (u32, u32)) -> (u32, u32) {
    let halved = matches!(
        codec,
        CodecId::H264 | CodecId::Hevc | CodecId::Av1 | CodecId::Vp8 | CodecId::Vp9
    );
    if halved {
        ((width & !1).max(2), (height & !1).max(2))
    } else {
        (width, height)
    }
}

/// The long edge for a canvas `long` by `short` scaled so `short` becomes
/// `short_side`.
fn scale_edge(long: u32, short: u32, short_side: u32) -> Result<u32, ExportError> {
    let value = u64::from(long) * u64::from(short_side) / u64::from(short);
    if value > u64::from(MAX_EDGE) {
        return Err(ExportError::Limit);
    }
    u32::try_from(value).map_err(|_| ExportError::Limit)
}

#[cfg(test)]
mod tests {
    use super::{
        AudioExport, ExportError, ExportRequest, GIF_SHORT_SIDES, VIDEO_SHORT_SIDES, coded_size,
        output_size, validate,
    };
    use cutlass_codec::{CodecId, ContainerId};
    use cutlass_model::{FrameRate, TimeRange};

    fn request(codec: CodecId, short_side: u32) -> ExportRequest {
        ExportRequest {
            range: TimeRange::new(0, 1_000_000),
            short_side,
            codec,
            bitrate: 2_000_000,
            frame_rate: FrameRate::FPS_30,
            container: ContainerId::Isobmff,
            audio: None,
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: crate::ProresProfile::Standard,
        }
    }

    #[test]
    fn video_and_gif_accept_only_their_short_side_steps() {
        for side in VIDEO_SHORT_SIDES {
            assert!(validate(&request(CodecId::H264, side)).is_ok(), "{side}");
            let gif = validate(&request(CodecId::Gif, side));
            assert_eq!(gif, Err(ExportError::Resolution { short_side: side }));
        }
        for side in GIF_SHORT_SIDES {
            assert!(validate(&request(CodecId::Gif, side)).is_ok(), "{side}");
            let video = validate(&request(CodecId::H264, side));
            assert_eq!(video, Err(ExportError::Resolution { short_side: side }));
        }
        assert_eq!(
            validate(&request(CodecId::H264, 1000)),
            Err(ExportError::Resolution { short_side: 1000 })
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
        audio.audio.as_mut().expect("audio").channels = 0;
        assert_eq!(validate(&audio), Err(ExportError::AudioLayout));
        for channels in [1, 2] {
            audio.audio.as_mut().expect("audio").channels = channels;
            assert_eq!(validate(&audio), Ok(()));
        }
        for channels in [3, 6, 8, u16::MAX] {
            audio.audio.as_mut().expect("audio").channels = channels;
            let refused = validate(&audio).expect_err("surround");
            assert_eq!(refused, ExportError::SurroundExport { channels });
            let message = refused.to_string();
            assert!(message.starts_with("surround export is not supported yet"));
            assert!(
                message.contains(&format!("{channels} channels")),
                "{message}"
            );
        }
    }

    #[test]
    fn the_short_side_sets_the_shorter_edge_and_truncates_the_other() {
        assert_eq!(output_size(1920, 1080, 1080), Ok((1920, 1080)));
        assert_eq!(output_size(1080, 1920, 1080), Ok((1080, 1920)));
        assert_eq!(output_size(1920, 1080, 720), Ok((1280, 720)));
        assert_eq!(output_size(1920, 1080, 2160), Ok((3840, 2160)));
        assert_eq!(output_size(1920, 1080, 4320), Ok((7680, 4320)));
        assert_eq!(output_size(1080, 1350, 1080), Ok((1080, 1350)));
        assert_eq!(output_size(1000, 1001, 480), Ok((480, 480)));
        assert_eq!(output_size(100, 100, 480), Ok((480, 480)));
        assert_eq!(output_size(0, 10, 480), Err(ExportError::Canvas));
        assert_eq!(output_size(-1, 10, 480), Err(ExportError::Canvas));
        // A sliver canvas would stretch the long edge past any encoder.
        assert_eq!(output_size(100_000, 1, 480), Err(ExportError::Limit));
        assert_eq!(output_size(1, 100_000, 480), Err(ExportError::Limit));
        assert_eq!(output_size(i32::MAX, 1, 4320), Err(ExportError::Limit));
        assert_eq!(output_size(16_384, 4320, 4320), Ok((16_384, 4320)));
        assert_eq!(output_size(3800, 1000, 4320), Err(ExportError::Limit));
    }

    #[test]
    fn a_half_chroma_codec_codes_even_edges() {
        use cutlass_codec::CodecId;
        assert_eq!(coded_size(CodecId::H264, (720, 405)), (720, 404));
        assert_eq!(coded_size(CodecId::Hevc, (607, 1080)), (606, 1080));
        assert_eq!(coded_size(CodecId::H264, (1, 1)), (2, 2));
        assert_eq!(coded_size(CodecId::Png, (720, 405)), (720, 405));
        assert_eq!(coded_size(CodecId::Gif, (607, 1080)), (607, 1080));
    }
}
