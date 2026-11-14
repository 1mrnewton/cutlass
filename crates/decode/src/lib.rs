//! Decoder trait and the factory that picks an implementation.
//!
//! A decoder consumes one complete access unit per [`Decoder::push`].
//! Incomplete byte streams are a container concern: the container returns
//! [`CodecError::NeedMoreData`] until a unit is whole, then hands that unit
//! over. A unit that claims to be whole and runs out of bits is
//! [`CodecError::Truncated`].
//!
//! `push` returns every picture or PCM buffer that unit completed. It may
//! return nothing while overlap or a reorder buffer is still filling.
//! `push` of a unit that fails leaves the decoder as it was before that
//! call. `drain` flushes overlap that a following silent unit would have
//! released. A second `drain` is empty. `reset` drops overlap, reorder, and
//! the sample clock; the configuration stays. `push` after `drain` continues.
//! `push` of an empty packet is an invalid call, not a silent success.
//!
//! [`VideoDecoder`] is the picture path. It has the same push, drain, and
//! reset rules. One push is one access unit and may return zero pictures
//! when display order is still waiting on a later picture. `drain` releases
//! those pictures. A second `drain` is empty. `reset` drops references and
//! the reorder queue. Configuration from the open parameters stays.
//!
//! The factory walks the openers registered for that id, in order, and uses
//! the first that accepts the parameters. An opener declines with
//! [`CodecError::Unsupported`]. Any other error from one opener (a platform
//! decoder that fails to start, say) does not stop the walk either: the next
//! opener is tried, and the first such error is what the caller sees when
//! none accepts. The desktop registers a platform opener first, then the
//! Rust opener.

#![forbid(unsafe_code)]

pub mod orient;
mod surface;
pub mod video;

use std::sync::Arc;

pub use surface::{PictureSurface, Surface};

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    BitDepth, Channel, ChannelLayout, ChromaSamplePosition, CodecError, CodecParams, ColorMatrix,
    ColorPrimaries, ColorRange, ColorType, ContentLight, FrameError, ImageError, ItuT35,
    MasteringDisplay, MatrixCoefficients, Packet, Pcm, PhysicalPixel, PixelFormat, Plane, Rgb8,
    SampleFormat, SideData, StillImage, TextItem, Transfer, Transparency, VideoFrame, Yuv420High,
    Yuv420p, Yuv420p10, Yuv422High, Yuv422p, Yuv444High, Yuv444p,
};
pub use orient::Orientation;
pub use video::VideoDecodeFactory;

/// One decoded still.
///
/// `pts` and `duration` are copied from the packet. A still file has no
/// clock, so both are zero and `timescale` is zero unless the container
/// named one. The samples are the coded domain, not a display conversion.
/// A video picture is [`PictureOutput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StillOutput {
    pub image: StillImage,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
}

/// One decoded audio buffer.
///
/// `skip_leading` and `skip_trailing` count samples inside `pcm` that are
/// delay or padding when this buffer is the place those samples landed.
/// Most frames leave both at zero and report delay on the packet side data
/// instead. `concealed` is set only when a caller opted into concealment
/// and this buffer is not a faithful decode. This build does not conceal:
/// a bad unit returns an error and `concealed` stays false.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioOutput {
    pub pcm: Pcm,
    pub pts: i64,
    pub timescale: u32,
    pub skip_leading: u32,
    pub skip_trailing: u32,
    pub concealed: bool,
}

/// Coded size and color of one decoded picture.
///
/// `width` and `height` are the buffer that [`PictureOutput`] stores, tightly
/// packed, with no stride gap and no display crop. `render_width` and
/// `render_height` are the display size and do not change that buffer. Color
/// conversion stays outside this crate. [`PixelFormat::Yuv420p`] is 4:2:0: a full-size Y plane,
/// then U, then V, each `width / 2` by `height / 2`. [`PixelFormat::Yuv422p`]
/// is 4:2:2: U and V are `width / 2` by `height`. [`PixelFormat::Yuv422p10`]
/// is that layout with a little-endian `u16` per sample. [`PixelFormat::Yuv420p10`]
/// is that layout with a little-endian `u16` per sample. [`PixelFormat::Yuv444p`]
/// is 4:4:4: three full-size planes. [`PixelFormat::Yuv444p10`] is that layout
/// with a little-endian `u16` per sample. [`PixelFormat::Gray8`]
/// is one 8-bit plane, `width` by `height`. [`PixelFormat::Gray10`] is that
/// plane with a little-endian `u16` per sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PictureInfo {
    pub width: u32,
    pub height: u32,
    pub pixel: PixelFormat,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
    pub chroma_position: ChromaSamplePosition,
    /// Display size. This is the sample size when the frame does not override it.
    /// The sample buffer stays `width` by `height`.
    pub render_width: u32,
    pub render_height: u32,
    /// CICP `colour_primaries`. 2 means unspecified.
    pub cicp_primaries: u8,
    /// CICP `transfer_characteristics`. 2 means unspecified.
    pub cicp_transfer: u8,
    /// CICP `matrix_coefficients`. 2 means unspecified. 0 is identity.
    pub cicp_matrix: u8,
    /// `num_units_in_display_tick`. Zero when the sequence has no timing info.
    pub num_units_in_display_tick: u32,
    /// `time_scale`. Zero when the sequence has no timing info.
    pub time_scale: u32,
    /// `equal_picture_interval`.
    pub equal_picture_interval: bool,
    /// `num_ticks_per_picture_minus_1` when the picture interval is equal.
    pub num_ticks_per_picture_minus_1: Option<u32>,
    /// `num_units_in_decoding_tick`. Zero when the sequence has no decoder model.
    pub num_units_in_decoding_tick: u32,
    /// `seq_level_idx` of the operating point this picture was decoded under.
    /// Other codecs leave this 0.
    pub seq_level_idx: u8,
    /// Sample range the stream names. A decoder that does not read one
    /// leaves this unspecified, and the samples are then studio swing.
    pub range: ColorRange,
    /// Content light level the stream carries for this picture.
    pub content_light: Option<ContentLight>,
    /// Mastering display colour volume the stream carries for this picture.
    pub mastering_display: Option<MasteringDisplay>,
}

impl PictureInfo {
    /// A picture whose display size is the sample size.
    ///
    /// The CICP bytes follow the named primaries and transfer. The matrix stays
    /// unspecified. A decoder that read a header sets the bytes from that header.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        pixel: PixelFormat,
        primaries: ColorPrimaries,
        transfer: Transfer,
    ) -> Self {
        Self {
            width,
            height,
            pixel,
            primaries,
            transfer,
            matrix: ColorMatrix::Unspecified,
            chroma_position: ChromaSamplePosition::Unknown,
            render_width: width,
            render_height: height,
            cicp_primaries: primaries.cicp(),
            cicp_transfer: transfer.cicp(),
            cicp_matrix: 2,
            num_units_in_display_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            num_ticks_per_picture_minus_1: None,
            num_units_in_decoding_tick: 0,
            seq_level_idx: 0,
            range: ColorRange::Unspecified,
            content_light: None,
            mastering_display: None,
        }
    }
}

/// One decoded picture: CPU bytes, a platform buffer, or a GPU surface.
///
/// A platform buffer is a token. This crate does not sample it. The bytes
/// behind a CPU picture are the coded samples in [`PictureInfo::pixel`] order.
/// A [`Surface`] stands for 8-bit 4:2:0 and copies those bytes out on request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureOutput {
    info: PictureInfo,
    pts: i64,
    timescale: u32,
    storage: PictureStorage,
    /// ITU-T T.35 messages the stream carries for this picture.
    t35: Vec<ItuT35>,
}

#[derive(Clone, Debug)]
enum PictureStorage {
    Cpu(Vec<u8>),
    /// CPU bytes the decoder also holds, to fill again for a later
    /// picture once every other holder has let them go.
    Shared(Arc<Vec<u8>>),
    Platform {
        token: u64,
    },
    Surface(Surface),
}

impl PartialEq for PictureStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Platform { token: a }, Self::Platform { token: b }) => a == b,
            (Self::Surface(a), Self::Surface(b)) => a == b,
            (Self::Cpu(_) | Self::Shared(_), Self::Cpu(_) | Self::Shared(_)) => {
                self.cpu_bytes() == other.cpu_bytes()
            }
            _ => false,
        }
    }
}

impl Eq for PictureStorage {}

impl PictureStorage {
    fn cpu_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Cpu(bytes) => Some(bytes),
            Self::Shared(bytes) => Some(bytes),
            Self::Platform { .. } | Self::Surface(_) => None,
        }
    }
}

/// Borrowed view of a [`PictureOutput`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureSamples<'a> {
    Cpu(&'a [u8]),
    Platform { token: u64 },
    Surface(&'a Surface),
}

/// What one `push` or `drain` completed.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Audio(AudioOutput),
    Picture(PictureOutput),
}

/// Byte length of a tightly packed picture, or why those dimensions are not a picture.
///
/// Codecs call this before they allocate. A size that overflows is
/// [`CodecError::Limit`]. An empty edge, an odd 4:2:0 size, or
/// [`PixelFormat::None`] is [`CodecError::InvalidApi`].
pub fn picture_byte_len(width: u32, height: u32, pixel: PixelFormat) -> Result<usize, CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    let width = usize::try_from(width).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    let height = usize::try_from(height).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    match pixel {
        PixelFormat::None => Err(CodecError::InvalidApi {
            what: "picture format",
        }),
        PixelFormat::Rgba8 => width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(CodecError::Limit {
                what: "picture bytes",
            }),
        PixelFormat::Yuv420p => {
            if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                return Err(CodecError::InvalidApi { what: "4:2:0 size" });
            }
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(3))
                .and_then(|bytes| bytes.checked_div(2))
                .ok_or(CodecError::Limit {
                    what: "picture bytes",
                })
        }
        PixelFormat::Yuv420p10 | PixelFormat::Yuv420p12 => {
            if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                return Err(CodecError::InvalidApi { what: "4:2:0 size" });
            }
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or(CodecError::Limit {
                    what: "picture bytes",
                })
        }
        PixelFormat::Yuv422p10
        | PixelFormat::Yuv422p12
        | PixelFormat::Yuv440p10
        | PixelFormat::Yuv440p12
        | PixelFormat::Yuv444p10
        | PixelFormat::Yuv444p12
        | PixelFormat::Gbrp10
        | PixelFormat::Gbrp12 => wide_planar(width, height, pixel),
        PixelFormat::Yuv422p | PixelFormat::Yuv440p | PixelFormat::Yuv444p | PixelFormat::Gbrp => {
            let (chroma_w, chroma_h) =
                pixel
                    .chroma_wh(width as u32, height as u32)
                    .ok_or(CodecError::InvalidApi {
                        what: "picture format",
                    })?;
            let luma = width.checked_mul(height).ok_or(CodecError::Limit {
                what: "picture bytes",
            })?;
            let chroma = usize::try_from(chroma_w)
                .ok()
                .and_then(|w| {
                    usize::try_from(chroma_h)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .and_then(|samples| samples.checked_mul(2))
                .ok_or(CodecError::Limit {
                    what: "picture bytes",
                })?;
            luma.checked_add(chroma).ok_or(CodecError::Limit {
                what: "picture bytes",
            })
        }
        PixelFormat::Gray8 => width.checked_mul(height).ok_or(CodecError::Limit {
            what: "picture bytes",
        }),
        PixelFormat::Gray10 => width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(2))
            .ok_or(CodecError::Limit {
                what: "picture bytes",
            }),
    }
}

fn wide_planar(width: usize, height: usize, pixel: PixelFormat) -> Result<usize, CodecError> {
    let (chroma_w, chroma_h) =
        pixel
            .chroma_wh(width as u32, height as u32)
            .ok_or(CodecError::InvalidApi {
                what: "picture format",
            })?;
    let luma = width
        .checked_mul(height)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or(CodecError::Limit {
            what: "picture bytes",
        })?;
    let chroma = usize::try_from(chroma_w)
        .ok()
        .and_then(|w| {
            usize::try_from(chroma_h)
                .ok()
                .and_then(|h| w.checked_mul(h))
        })
        .and_then(|samples| samples.checked_mul(4))
        .ok_or(CodecError::Limit {
            what: "picture bytes",
        })?;
    luma.checked_add(chroma).ok_or(CodecError::Limit {
        what: "picture bytes",
    })
}

impl PictureOutput {
    /// CPU bytes whose length is [`picture_byte_len`] for `info`.
    pub fn cpu(
        info: PictureInfo,
        bytes: Vec<u8>,
        pts: i64,
        timescale: u32,
    ) -> Result<Self, CodecError> {
        let len = picture_byte_len(info.width, info.height, info.pixel)?;
        if bytes.len() != len {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        Ok(Self {
            info,
            pts,
            timescale,
            storage: PictureStorage::Cpu(bytes),
            t35: Vec::new(),
        })
    }

    /// CPU bytes as [`Self::cpu`], shared with the decoder, which may fill
    /// them again for a later picture once no one else holds them (it
    /// checks with [`Arc::get_mut`]).
    pub fn cpu_shared(
        info: PictureInfo,
        bytes: Arc<Vec<u8>>,
        pts: i64,
        timescale: u32,
    ) -> Result<Self, CodecError> {
        let len = picture_byte_len(info.width, info.height, info.pixel)?;
        if bytes.len() != len {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        Ok(Self {
            info,
            pts,
            timescale,
            storage: PictureStorage::Shared(bytes),
            t35: Vec::new(),
        })
    }

    /// A platform buffer with the same coded size rules as [`Self::cpu`].
    pub fn platform(
        info: PictureInfo,
        token: u64,
        pts: i64,
        timescale: u32,
    ) -> Result<Self, CodecError> {
        picture_byte_len(info.width, info.height, info.pixel)?;
        Ok(Self {
            info,
            pts,
            timescale,
            storage: PictureStorage::Platform { token },
            t35: Vec::new(),
        })
    }

    /// A GPU surface that stands for 8-bit 4:2:0 at `info`'s size.
    pub fn surface(
        info: PictureInfo,
        surface: Surface,
        pts: i64,
        timescale: u32,
    ) -> Result<Self, CodecError> {
        if info.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::InvalidApi {
                what: "surface pixel format",
            });
        }
        picture_byte_len(info.width, info.height, info.pixel)?;
        Ok(Self {
            info,
            pts,
            timescale,
            storage: PictureStorage::Surface(surface),
            t35: Vec::new(),
        })
    }

    /// The same picture with the ITU-T T.35 messages the stream carries
    /// for it, in stream order.
    #[must_use]
    pub fn with_t35(mut self, messages: Vec<ItuT35>) -> Self {
        self.t35 = messages;
        self
    }

    /// ITU-T T.35 messages for this picture, such as HDR10+ dynamic
    /// metadata. Empty when the stream carries none.
    #[must_use]
    pub fn t35(&self) -> &[ItuT35] {
        &self.t35
    }

    #[must_use]
    pub fn info(&self) -> PictureInfo {
        self.info
    }

    #[must_use]
    pub fn pts(&self) -> i64 {
        self.pts
    }

    #[must_use]
    pub fn timescale(&self) -> u32 {
        self.timescale
    }

    /// The CPU bytes, moved out without a copy, or `None` for a platform
    /// buffer.
    ///
    /// Bytes shared with the decoder are copied unless no one else holds
    /// them; [`Self::into_shared`] keeps them shared.
    #[must_use]
    pub fn into_cpu(self) -> Option<Vec<u8>> {
        match self.storage {
            PictureStorage::Cpu(bytes) => Some(bytes),
            PictureStorage::Shared(bytes) => {
                Some(Arc::try_unwrap(bytes).unwrap_or_else(|bytes| (*bytes).clone()))
            }
            PictureStorage::Platform { .. } | PictureStorage::Surface(_) => None,
        }
    }

    /// The CPU bytes as shared bytes, without a copy, or `None` for a
    /// platform buffer.
    #[must_use]
    pub fn into_shared(self) -> Option<Arc<Vec<u8>>> {
        match self.storage {
            PictureStorage::Cpu(bytes) => Some(Arc::new(bytes)),
            PictureStorage::Shared(bytes) => Some(bytes),
            PictureStorage::Platform { .. } | PictureStorage::Surface(_) => None,
        }
    }

    #[must_use]
    pub fn samples(&self) -> PictureSamples<'_> {
        match &self.storage {
            PictureStorage::Cpu(bytes) => PictureSamples::Cpu(bytes),
            PictureStorage::Shared(bytes) => PictureSamples::Cpu(bytes),
            PictureStorage::Platform { token } => PictureSamples::Platform { token: *token },
            PictureStorage::Surface(surface) => PictureSamples::Surface(surface),
        }
    }
}

pub trait Decoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Output>, CodecError>;

    fn reset(&mut self);

    /// Decode one still. Audio codecs refuse this.
    ///
    /// The default keeps an audio decoder on the audio methods. A picture
    /// codec overrides this and refuses [`Self::push`].
    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<StillOutput>, CodecError> {
        let _ = (self, packet);
        Err(CodecError::Unsupported {
            what: "picture output",
        })
    }
}

/// Sample aspect ratio, horizontal samples over vertical samples.
///
/// [`Self::UNSPECIFIED`] is what a bitstream writes when Annex E gives no
/// ratio. A zero edge is the same value: the ratio is not usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleAspect {
    pub width: u32,
    pub height: u32,
}

impl SampleAspect {
    pub const UNSPECIFIED: Self = Self {
        width: 0,
        height: 1,
    };

    #[must_use]
    pub const fn ratio(width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            Self::UNSPECIFIED
        } else {
            Self { width, height }
        }
    }
}

/// Sequence timing from Annex E.
///
/// Both counts are zero when the bitstream has no `timing_info`. `fixed` is
/// `fixed_frame_rate_flag`. A frame rate, when the flag is set, is
/// `time_scale / (2 * num_units)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceTiming {
    pub num_units: u32,
    pub time_scale: u32,
    pub fixed: bool,
}

impl SequenceTiming {
    pub const ABSENT: Self = Self {
        num_units: 0,
        time_scale: 0,
        fixed: false,
    };
}

/// One decoded picture, in display order once it is released.
///
/// `picture` is the cropped 8-bit 4:2:0 view. A 4:2:2 or 4:4:4 decode keeps the
/// real samples in `yuv422`, `yuv422_high`, `yuv444`, or `yuv444_high`, and
/// `picture` is those samples subsampled to 4:2:0. When `yuv420p10` is set, that
/// buffer is the decoded 10-bit 4:2:0 picture and `picture` is those samples
/// shifted right by two bits. When `yuv420_high` is set, that buffer is the
/// decoded 4:2:0 picture at 9, 11, 12, 13, or 14 bits and `picture` is those
/// samples shifted right by `bit_depth - 8`. A 10-bit 4:2:2 or 4:4:4 picture is
/// the matching high buffer with `bit_depth` 10. Color tags, the sample aspect,
/// and the sequence timing describe the decoded samples. They are not a conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    pub picture: Yuv420p,
    pub yuv420p10: Option<Yuv420p10>,
    pub yuv420_high: Option<Yuv420High>,
    pub yuv422: Option<Yuv422p>,
    pub yuv422_high: Option<Yuv422High>,
    pub yuv444: Option<Yuv444p>,
    pub yuv444_high: Option<Yuv444High>,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
    pub sample_aspect: SampleAspect,
    pub timing: SequenceTiming,
}

pub trait VideoDecoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError>;

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError>;

    fn reset(&mut self);
}

#[derive(Clone)]
pub struct DecodeFactory {
    openers: Vec<Opener>,
    video_openers: Vec<VideoOpener>,
}

#[derive(Clone)]
struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
}

#[derive(Clone)]
struct VideoOpener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
}

impl DecodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
            video_openers: Vec::new(),
        }
    }

    /// Later registrations do not hide an earlier opener that accepts the
    /// same id. An earlier opener that refuses, for any reason, does not
    /// hide a later one. The desktop registers a platform opener first, then
    /// the Rust opener.
    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    /// Later video registrations do not hide an earlier one for the same id,
    /// and an earlier one that refuses does not hide a later one.
    pub fn register_video(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
    ) {
        self.video_openers.push(VideoOpener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        first_accepting(
            self.openers
                .iter()
                .filter(|opener| opener.id == params.codec)
                .map(|opener| opener.open),
            params,
            "codec is not registered",
        )
    }

    pub fn open_video(&self, params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        first_accepting(
            self.video_openers
                .iter()
                .filter(|opener| opener.id == params.codec)
                .map(|opener| opener.open),
            params,
            "video codec is not registered",
        )
    }
}

/// The first opener that accepts `params`. When none does, the first error
/// that was not a plain decline, else a decline naming why.
fn first_accepting<T>(
    openers: impl Iterator<Item = fn(&CodecParams) -> Result<T, CodecError>>,
    params: &CodecParams,
    unregistered: &'static str,
) -> Result<T, CodecError> {
    let mut registered = false;
    let mut failed = None;
    for open in openers {
        registered = true;
        match open(params) {
            Ok(decoder) => return Ok(decoder),
            Err(CodecError::Unsupported { .. }) => {}
            Err(error) => {
                failed.get_or_insert(error);
            }
        }
    }
    if let Some(error) = failed {
        return Err(error);
    }
    Err(CodecError::Unsupported {
        what: if registered {
            "no decoder accepted the parameters"
        } else {
            unregistered
        },
    })
}

impl Default for DecodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{
        AudioOutput, DecodeFactory, Decoder, ItuT35, Output, PictureInfo, PictureOutput,
        PictureSamples, SampleAspect, SequenceTiming, VideoDecoder, VideoOutput, picture_byte_len,
    };
    use cutlass_codec::CodecId;
    use cutlass_codec::access::{
        ChannelLayout, CodecError, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, Packet,
        Pcm, PixelFormat, SampleFormat, Transfer, Yuv420p,
    };

    struct Silent;

    impl Decoder for Silent {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.0]).expect("pcm");
            assert_eq!(pcm.format, SampleFormat::F32);
            Ok(vec![Output::Audio(AudioOutput {
                pcm,
                pts: packet.pts,
                timescale: packet.timescale,
                skip_leading: 0,
                skip_trailing: 0,
                concealed: false,
            })])
        }

        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_silent(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported {
                what: "silent test",
            });
        }
        Ok(Box::new(Silent))
    }

    #[test]
    fn shared_bytes_stay_shared_and_compare_by_value() {
        let info = PictureInfo::new(
            2,
            2,
            PixelFormat::Yuv420p,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
        );
        let bytes = std::sync::Arc::new(vec![1, 2, 3, 4, 5, 6]);
        assert!(
            PictureOutput::cpu_shared(info, std::sync::Arc::new(vec![1, 2, 3]), 0, 1).is_err(),
            "the length is checked"
        );
        let shared = PictureOutput::cpu_shared(info, bytes.clone(), 0, 1).expect("yuv");
        let owned = PictureOutput::cpu(info, vec![1, 2, 3, 4, 5, 6], 0, 1).expect("yuv");
        assert_eq!(shared, owned);
        assert_eq!(shared.samples(), PictureSamples::Cpu(&[1, 2, 3, 4, 5, 6]));
        let back = shared.clone().into_shared().expect("cpu");
        assert!(std::sync::Arc::ptr_eq(&back, &bytes), "no copy");
        // Still held by `bytes`: a copy.
        assert_eq!(shared.into_cpu().expect("cpu"), vec![1, 2, 3, 4, 5, 6]);
        drop(back);
        assert_eq!(
            std::sync::Arc::strong_count(&bytes),
            1,
            "every other holder let go"
        );
    }

    fn rgba(width: u32, height: u32) -> PictureInfo {
        PictureInfo::new(
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            Transfer::Bt709,
        )
    }

    fn jpeg(width: u32, height: u32, pixel: PixelFormat) -> CodecParams {
        let mut params = CodecParams::with_format(CodecId::Jpeg, width, height, pixel);
        params.primaries = ColorPrimaries::Bt709;
        params.transfer = Transfer::Bt709;
        params
    }

    struct Raw {
        info: PictureInfo,
        mark: i64,
    }

    impl Decoder for Raw {
        fn codec(&self) -> CodecId {
            CodecId::Jpeg
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let len = picture_byte_len(self.info.width, self.info.height, self.info.pixel)?;
            if packet.data.len() != len {
                return Err(CodecError::Truncated {
                    at_bit: u64::try_from(packet.data.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(8),
                    what: "picture",
                });
            }
            let picture = PictureOutput::cpu(
                self.info,
                packet.data.clone(),
                packet.pts + self.mark,
                packet.timescale,
            )?;
            Ok(vec![Output::Picture(picture)])
        }

        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_raw(mark: i64) -> fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        if mark == 1 {
            open_raw_platform
        } else {
            open_raw_rust
        }
    }

    fn open_raw_platform(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        if params.extra.as_slice() != b"hw" {
            return Err(CodecError::Unsupported {
                what: "platform picture",
            });
        }
        Ok(Box::new(Raw {
            info: rgba(params.width, params.height),
            mark: 1,
        }))
    }

    fn open_raw_rust(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        OPEN_RUST.with(|count| count.set(count.get() + 1));
        Ok(Box::new(Raw {
            info: PictureInfo::new(
                params.width,
                params.height,
                params.pixel,
                params.primaries,
                params.transfer,
            ),
            mark: 2,
        }))
    }

    thread_local! {
        static OPEN_RUST: Cell<usize> = const { Cell::new(0) };
    }

    struct Token;

    impl Decoder for Token {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            let bytes: [u8; 8] =
                packet
                    .data
                    .as_slice()
                    .try_into()
                    .map_err(|_| CodecError::InvalidApi {
                        what: "platform token",
                    })?;
            let picture = PictureOutput::platform(
                rgba(1, 1),
                u64::from_le_bytes(bytes),
                packet.pts,
                packet.timescale,
            )?;
            Ok(vec![Output::Picture(picture)])
        }

        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_token(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "token" });
        }
        Ok(Box::new(Token))
    }

    fn open_malformed(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        Err(CodecError::Malformed {
            at_bit: 0,
            what: "configuration",
        })
    }

    fn picture_packet(data: Vec<u8>, pts: i64, timescale: u32) -> Packet {
        Packet {
            stream: 0,
            data,
            pts,
            duration: 1,
            timescale,
            keyframe: true,
            side: Vec::new(),
        }
    }

    #[test]
    fn factory_uses_the_first_registration() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Pcm, open_silent);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let mut decoder = factory.open(&params).expect("open");
        let empty = Packet::audio(Vec::new(), 0, 1, 48_000);
        let err = decoder.push(&empty).expect_err("empty");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "empty packet",
            }
        );
        let packet = Packet::audio(vec![0], 0, 1, 48_000);
        let out = decoder.push(&packet).expect("push");
        let Output::Audio(audio) = &out[0] else {
            panic!("pcm");
        };
        assert_eq!(out.len(), 1);
        assert!(!audio.concealed);
        assert!(decoder.drain().expect("drain").is_empty());
        let picture = decoder.decode_picture(&packet);
        assert_eq!(
            picture.expect_err("audio"),
            CodecError::Unsupported {
                what: "picture output",
            }
        );
    }

    #[test]
    fn unregistered_codec_is_unsupported() {
        let factory = DecodeFactory::new();
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::stereo());
        let err = match factory.open(&params) {
            Err(err) => err,
            Ok(_) => panic!("missing"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "codec is not registered",
            }
        );
    }

    struct Solid;

    impl VideoDecoder for Solid {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            let picture = Yuv420p::solid(2, 2, 16, 128, 128).expect("picture");
            Ok(vec![VideoOutput {
                picture,
                yuv420p10: None,
                yuv420_high: None,
                yuv422: None,
                yuv422_high: None,
                yuv444: None,
                yuv444_high: None,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
                keyframe: packet.keyframe,
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
                matrix: ColorMatrix::Bt709,
                sample_aspect: SampleAspect::UNSPECIFIED,
                timing: SequenceTiming::ABSENT,
            }])
        }

        fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_solid(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "solid test" });
        }
        Ok(Box::new(Solid))
    }

    fn open_broken_video(_params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        Err(CodecError::Malformed {
            at_bit: 0,
            what: "platform session",
        })
    }

    #[test]
    fn a_video_opener_that_fails_falls_through_to_the_next() {
        let mut factory = DecodeFactory::new();
        factory.register_video(CodecId::H264, open_broken_video);
        factory.register_video(CodecId::H264, open_solid);
        let params = CodecParams::video(CodecId::H264, 2, 2);
        let mut decoder = factory.open_video(&params).expect("fell through");
        let out = decoder
            .push(&Packet::video(vec![1], 0, 1, 30, true))
            .expect("push");
        assert_eq!(out.len(), 1);

        let mut broken = DecodeFactory::new();
        broken.register_video(CodecId::H264, open_broken_video);
        assert!(matches!(
            broken.open_video(&params),
            Err(CodecError::Malformed { .. })
        ));
        assert!(matches!(
            DecodeFactory::new().open_video(&params),
            Err(CodecError::Unsupported {
                what: "video codec is not registered",
            })
        ));
    }

    #[test]
    fn video_factory_uses_the_first_registration() {
        let mut factory = DecodeFactory::new();
        factory.register_video(CodecId::H264, open_solid);
        let params = CodecParams::video(CodecId::H264, 2, 2);
        let mut decoder = factory.open_video(&params).expect("open");
        let packet = Packet::video(vec![1], 0, 1, 30, true);
        let out = decoder.push(&packet).expect("push");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].picture.y(), &[16, 16, 16, 16]);
        assert!(decoder.drain().expect("drain").is_empty());
        let audio = factory.open(&CodecParams::audio(
            CodecId::H264,
            48_000,
            ChannelLayout::mono(),
        ));
        assert!(matches!(
            audio,
            Err(CodecError::Unsupported {
                what: "codec is not registered",
            })
        ));
    }

    #[test]
    fn a_refusing_opener_falls_through_to_the_picture() {
        OPEN_RUST.with(|count| count.set(0));
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(1));
        factory.register(CodecId::Jpeg, open_raw(2));
        let params = jpeg(1, 1, PixelFormat::Rgba8);
        let mut decoder = factory.open(&params).expect("rust");
        assert_eq!(OPEN_RUST.with(Cell::get), 1);
        let packet = picture_packet(vec![9, 8, 7, 6], 4, 30);
        let out = decoder.push(&packet).expect("push");
        let Output::Picture(picture) = &out[0] else {
            panic!("picture");
        };
        assert_eq!(picture.pts(), 6);
        assert_eq!(picture.timescale(), 30);
        assert_eq!(picture.info(), rgba(1, 1));
        assert_eq!(picture.samples(), PictureSamples::Cpu(&[9, 8, 7, 6]));
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn the_first_opener_that_accepts_hides_the_later_one() {
        OPEN_RUST.with(|count| count.set(0));
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(1));
        factory.register(CodecId::Jpeg, open_raw(2));
        let mut params = jpeg(1, 1, PixelFormat::Rgba8);
        params.extra = b"hw".to_vec();
        let mut decoder = factory.open(&params).expect("platform");
        assert_eq!(OPEN_RUST.with(Cell::get), 0);
        let packet = picture_packet(vec![1, 2, 3, 4], 10, 30);
        let out = decoder.push(&packet).expect("push");
        let Output::Picture(picture) = &out[0] else {
            panic!("picture");
        };
        assert_eq!(picture.pts(), 11);
    }

    #[test]
    fn a_failing_platform_opener_falls_through_to_the_rust_one() {
        OPEN_RUST.with(|count| count.set(0));
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_malformed);
        factory.register(CodecId::Jpeg, open_raw(2));
        assert!(factory.open(&jpeg(1, 1, PixelFormat::Rgba8)).is_ok());
        assert_eq!(OPEN_RUST.with(Cell::get), 1);
    }

    #[test]
    fn when_none_accepts_the_first_real_error_is_reported() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(1));
        factory.register(CodecId::Jpeg, open_malformed);
        let err = match factory.open(&jpeg(1, 1, PixelFormat::Rgba8)) {
            Err(err) => err,
            Ok(_) => panic!("malformed"),
        };
        assert_eq!(
            err,
            CodecError::Malformed {
                at_bit: 0,
                what: "configuration",
            }
        );
    }

    #[test]
    fn every_opener_refused_names_the_parameters() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(1));
        let err = match factory.open(&jpeg(1, 1, PixelFormat::Rgba8)) {
            Err(err) => err,
            Ok(_) => panic!("refused"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "no decoder accepted the parameters",
            }
        );
    }

    #[test]
    fn a_platform_token_round_trips_from_the_packet() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::H264, open_token);
        let mut params = jpeg(1, 1, PixelFormat::Rgba8);
        params.codec = CodecId::H264;
        let mut decoder = factory.open(&params).expect("open");
        let packet = picture_packet(15u64.to_le_bytes().to_vec(), 3, 0);
        let out = decoder.push(&packet).expect("push");
        let Output::Picture(picture) = &out[0] else {
            panic!("picture");
        };
        assert_eq!(picture.samples(), PictureSamples::Platform { token: 15 });
        assert_eq!(picture.timescale(), 0);
    }

    #[test]
    fn an_empty_picture_packet_is_an_invalid_call() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(2));
        let mut decoder = factory.open(&jpeg(1, 1, PixelFormat::Rgba8)).expect("open");
        let packet = picture_packet(Vec::new(), 0, 30);
        let err = decoder.push(&packet).expect_err("empty");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "empty packet",
            }
        );
    }

    #[test]
    fn a_short_picture_does_not_decode() {
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_raw(2));
        let mut decoder = factory
            .open(&jpeg(2, 2, PixelFormat::Yuv420p))
            .expect("open");
        let packet = picture_packet(vec![0, 1, 2], 0, 30);
        let err = decoder.push(&packet).expect_err("short");
        assert_eq!(
            err,
            CodecError::Truncated {
                at_bit: 24,
                what: "picture",
            }
        );
        let packet = picture_packet(vec![9, 8, 7, 6, 5, 4], 8, 30);
        let out = decoder.push(&packet).expect("after the short packet");
        let Output::Picture(picture) = &out[0] else {
            panic!("picture");
        };
        assert_eq!(picture.samples(), PictureSamples::Cpu(&[9, 8, 7, 6, 5, 4]));
        assert_eq!(picture.pts(), 10);
    }

    #[test]
    fn a_new_picture_names_no_range_and_no_hdr_metadata() {
        let info = PictureInfo::new(
            2,
            2,
            PixelFormat::Yuv420p,
            ColorPrimaries::Bt2020,
            Transfer::Pq,
        );
        assert_eq!(info.range, ColorRange::Unspecified);
        assert_eq!(info.content_light, None);
        assert_eq!(info.mastering_display, None);
        let picture = PictureOutput::cpu(info, vec![0; 6], 0, 1).expect("yuv");
        assert!(picture.t35().is_empty());
    }

    #[test]
    fn a_picture_keeps_its_t35_messages_in_order() {
        let info = PictureInfo::new(
            2,
            2,
            PixelFormat::Yuv420p,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
        );
        let messages = vec![
            ItuT35 {
                country_code: 0xb5,
                country_code_extension: None,
                payload: vec![0, 0x3c, 0, 1, 4],
            },
            ItuT35 {
                country_code: 0xff,
                country_code_extension: Some(7),
                payload: Vec::new(),
            },
        ];
        let picture = PictureOutput::cpu(info, vec![0; 6], 0, 1)
            .expect("yuv")
            .with_t35(messages.clone());
        assert_eq!(picture.t35(), messages);
        assert_eq!(picture.clone().t35(), messages);
    }

    #[test]
    fn yuv420_is_three_planes_with_even_edges() {
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv420p).expect("len"),
            6
        );
        let info = PictureInfo::new(
            2,
            2,
            PixelFormat::Yuv420p,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
        );
        let picture = PictureOutput::cpu(info, vec![1, 2, 3, 4, 5, 6], 0, 1).expect("yuv");
        assert_eq!(picture.samples(), PictureSamples::Cpu(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(
            picture_byte_len(3, 2, PixelFormat::Yuv420p).expect_err("odd"),
            CodecError::InvalidApi { what: "4:2:0 size" }
        );
        assert_eq!(picture_byte_len(3, 2, PixelFormat::Gray8).expect("gray"), 6);
        assert_eq!(
            picture_byte_len(3, 2, PixelFormat::Gray10).expect("gray10"),
            12
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv420p10).expect("10-bit"),
            12
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv444p).expect("444"),
            12
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv422p).expect("422"),
            8
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv422p10).expect("422-10"),
            16
        );
        assert_eq!(
            picture_byte_len(3, 2, PixelFormat::Yuv422p).expect("odd 4:2:2"),
            14
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv444p10).expect("444-10"),
            24
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv420p12).expect("12-bit"),
            12
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv422p12).expect("422-12"),
            16
        );
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv444p12).expect("444-12"),
            24
        );
    }

    #[test]
    fn picture_bytes_reject_an_empty_edge_a_missing_format_and_overflow() {
        assert_eq!(
            picture_byte_len(0, 1, PixelFormat::Rgba8).expect_err("empty"),
            CodecError::InvalidApi {
                what: "empty picture",
            }
        );
        assert_eq!(
            picture_byte_len(1, 1, PixelFormat::None).expect_err("format"),
            CodecError::InvalidApi {
                what: "picture format",
            }
        );
        assert_eq!(
            picture_byte_len(u32::MAX, u32::MAX, PixelFormat::Rgba8).expect_err("overflow"),
            CodecError::Limit {
                what: "picture bytes",
            }
        );
        let err = PictureOutput::cpu(rgba(1, 1), vec![1, 2, 3], 0, 1).expect_err("length");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "picture byte length",
            }
        );
        let err = PictureOutput::platform(
            PictureInfo::new(
                1,
                1,
                PixelFormat::None,
                ColorPrimaries::Unspecified,
                Transfer::Unspecified,
            ),
            1,
            0,
            1,
        )
        .expect_err("none");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "picture format",
            }
        );
    }
}
