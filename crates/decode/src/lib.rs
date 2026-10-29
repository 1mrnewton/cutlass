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
//! [`CodecError::Unsupported`]. Any other error means the parameters are
//! not openable, and the walk stops. The desktop registers a platform opener
//! first, then the Rust opener.

#![forbid(unsafe_code)]

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    BitDepth, Channel, ChannelLayout, CodecError, CodecParams, ColorMatrix, ColorPrimaries,
    ColorRange, ColorType, FrameError, ImageError, Packet, Pcm, PhysicalPixel, PixelFormat, Rgb8,
    SampleFormat, SideData, StillImage, TextItem, Transfer, Transparency, Yuv420p,
};

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
/// packed, with no stride gap and no display crop. Color conversion stays
/// outside this crate. [`PixelFormat::Yuv420p`] is 4:2:0: a full-size Y plane,
/// then U, then V, each `width / 2` by `height / 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PictureInfo {
    pub width: u32,
    pub height: u32,
    pub pixel: PixelFormat,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
}

/// One decoded picture, either CPU bytes or a platform buffer.
///
/// A platform buffer is a token. This crate does not sample it. The bytes
/// behind a CPU picture are the coded samples in [`PictureInfo::pixel`] order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureOutput {
    info: PictureInfo,
    pts: i64,
    timescale: u32,
    storage: PictureStorage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PictureStorage {
    Cpu(Vec<u8>),
    Platform { token: u64 },
}

/// Borrowed view of a [`PictureOutput`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureSamples<'a> {
    Cpu(&'a [u8]),
    Platform { token: u64 },
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
    }
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
        })
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

    #[must_use]
    pub fn samples(&self) -> PictureSamples<'_> {
        match &self.storage {
            PictureStorage::Cpu(bytes) => PictureSamples::Cpu(bytes),
            PictureStorage::Platform { token } => PictureSamples::Platform { token: *token },
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

/// One decoded picture, in display order once it is released.
///
/// `picture` is the cropped coded-domain sample buffer. Color tags describe
/// that buffer. They are not a conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOutput {
    pub picture: Yuv420p,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: ColorMatrix,
}

pub trait VideoDecoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError>;

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError>;

    fn reset(&mut self);
}

pub struct DecodeFactory {
    openers: Vec<Opener>,
    video_openers: Vec<VideoOpener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
}

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
    /// same id. An earlier [`CodecError::Unsupported`] does not hide a
    /// later opener. The desktop registers a platform opener first, then
    /// the Rust opener.
    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    /// Later video registrations do not hide an earlier one for the same id.
    pub fn register_video(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError>,
    ) {
        self.video_openers.push(VideoOpener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        let mut registered = false;
        for opener in &self.openers {
            if opener.id != params.codec {
                continue;
            }
            registered = true;
            match (opener.open)(params) {
                Ok(decoder) => return Ok(decoder),
                Err(CodecError::Unsupported { .. }) => continue,
                Err(err) => return Err(err),
            }
        }
        if registered {
            Err(CodecError::Unsupported {
                what: "no decoder accepted the parameters",
            })
        } else {
            Err(CodecError::Unsupported {
                what: "codec is not registered",
            })
        }
    }

    pub fn open_video(&self, params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
        for opener in &self.video_openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "video codec is not registered",
        })
    }
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
        AudioOutput, DecodeFactory, Decoder, Output, PictureInfo, PictureOutput, PictureSamples,
        VideoDecoder, VideoOutput, picture_byte_len,
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

    fn rgba(width: u32, height: u32) -> PictureInfo {
        PictureInfo {
            width,
            height,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
        }
    }

    fn jpeg(width: u32, height: u32, pixel: PixelFormat) -> CodecParams {
        let mut params = CodecParams::video(CodecId::Jpeg, width, height);
        params.pixel = pixel;
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
            info: PictureInfo {
                width: params.width,
                height: params.height,
                pixel: params.pixel,
                primaries: params.primaries,
                transfer: params.transfer,
            },
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
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
                keyframe: packet.keyframe,
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt709,
                transfer: Transfer::Bt709,
                matrix: ColorMatrix::Bt709,
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
    fn a_malformed_configuration_stops_the_walk() {
        OPEN_RUST.with(|count| count.set(0));
        let mut factory = DecodeFactory::new();
        factory.register(CodecId::Jpeg, open_malformed);
        factory.register(CodecId::Jpeg, open_raw(2));
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
        assert_eq!(OPEN_RUST.with(Cell::get), 0);
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
    fn yuv420_is_three_planes_with_even_edges() {
        assert_eq!(
            picture_byte_len(2, 2, PixelFormat::Yuv420p).expect("len"),
            6
        );
        let info = PictureInfo {
            width: 2,
            height: 2,
            pixel: PixelFormat::Yuv420p,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
        };
        let picture = PictureOutput::cpu(info, vec![1, 2, 3, 4, 5, 6], 0, 1).expect("yuv");
        assert_eq!(picture.samples(), PictureSamples::Cpu(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(
            picture_byte_len(3, 2, PixelFormat::Yuv420p).expect_err("odd"),
            CodecError::InvalidApi { what: "4:2:0 size" }
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
            PictureInfo {
                width: 1,
                height: 1,
                pixel: PixelFormat::None,
                primaries: ColorPrimaries::Unspecified,
                transfer: Transfer::Unspecified,
            },
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
