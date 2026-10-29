//! Encoder trait and the factory that picks an implementation.
//!
//! `push` accepts any number of PCM frames, including zero. It consumes
//! every finite, in-range sample it is given, buffering less than two
//! frames internally, and returns the access units those samples completed.
//! A sample that is not legal fails the call and adds nothing from that
//! push. `push_frame` accepts one picture, either CPU bytes or a platform
//! buffer this crate does not sample. A picture that is not legal fails
//! the call and adds nothing. An encoder that does not take that kind of
//! input returns [`CodecError::Unsupported`]. `drain` pads and flushes. A
//! second `drain` is empty. `reset` drops the buffer, the overlap, and
//! the sample clock.

#![forbid(unsafe_code)]

mod video;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use cutlass_codec::CodecId;
pub use cutlass_codec::access::{
    BitDepth, Channel, ChannelLayout, CodecError, CodecParams, ColorPrimaries, ColorRange,
    ColorType, ImageError, MatrixCoefficients, Packet, Pcm, PhysicalPixel, PixelFormat, Plane,
    Rgb8, SampleFormat, SideData, StillImage, TextItem, Transparency, VideoFrame,
};
pub use video::{VideoEncodeFactory, VideoEncoder, VideoInput, request_video_cancel};

/// PCM presented to an encoder, contiguous with the samples already accepted.
///
/// The encoder clocks packets from the count of accepted samples. `pts` on
/// the input is not a second clock: a discontinuity is a [`Encoder::reset`].
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    pub pcm: Pcm,
}

/// One still presented to an encoder.
///
/// The samples are the coded domain. The encoder does not convert them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureInput {
    pub image: StillImage,
}

/// Presentation time for one picture, in ticks of `timescale`.
///
/// `pts` may be negative. `duration` is greater than zero. `timescale` is
/// greater than zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTime {
    pts: i64,
    duration: i64,
    timescale: u32,
}

impl FrameTime {
    pub fn new(pts: i64, duration: i64, timescale: u32) -> Result<Self, CodecError> {
        if duration <= 0 {
            return Err(CodecError::InvalidApi { what: "duration" });
        }
        if timescale == 0 {
            return Err(CodecError::InvalidApi { what: "timescale" });
        }
        Ok(Self {
            pts,
            duration,
            timescale,
        })
    }

    #[must_use]
    pub fn pts(self) -> i64 {
        self.pts
    }

    #[must_use]
    pub fn duration(self) -> i64 {
        self.duration
    }

    #[must_use]
    pub fn timescale(self) -> u32 {
        self.timescale
    }
}

/// Tightly packed CPU pixels for one picture.
///
/// [`PixelFormat::Rgba8`] is four bytes per pixel, row by row.
/// [`PixelFormat::Yuv420p`] is planar Y, then U, then V. Width and height
/// are even.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuFrame {
    width: u32,
    height: u32,
    pixel: PixelFormat,
    bytes: Vec<u8>,
    time: FrameTime,
}

impl CpuFrame {
    pub fn new(
        width: u32,
        height: u32,
        pixel: PixelFormat,
        bytes: Vec<u8>,
        time: FrameTime,
    ) -> Result<Self, CodecError> {
        let len = picture_len(width, height, pixel)?;
        if bytes.len() != len {
            return Err(CodecError::InvalidApi {
                what: "picture length",
            });
        }
        Ok(Self {
            width,
            height,
            pixel,
            bytes,
            time,
        })
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn pixel(&self) -> PixelFormat {
        self.pixel
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn time(&self) -> FrameTime {
        self.time
    }
}

/// A picture owned by a platform device.
///
/// `token` is opaque. This crate does not sample the buffer it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformFrame {
    width: u32,
    height: u32,
    token: u64,
    time: FrameTime,
}

impl PlatformFrame {
    pub fn new(width: u32, height: u32, token: u64, time: FrameTime) -> Result<Self, CodecError> {
        frame_size(width, height)?;
        Ok(Self {
            width,
            height,
            token,
            time,
        })
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn token(&self) -> u64 {
        self.token
    }

    #[must_use]
    pub fn time(&self) -> FrameTime {
        self.time
    }
}

/// One picture presented to an encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameInput {
    Cpu(CpuFrame),
    Platform(PlatformFrame),
}

fn frame_size(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi { what: "frame size" });
    }
    Ok(())
}

fn picture_len(width: u32, height: u32, pixel: PixelFormat) -> Result<usize, CodecError> {
    frame_size(width, height)?;
    let width = usize::try_from(width).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let height = usize::try_from(height).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let pixels = width
        .checked_mul(height)
        .ok_or(CodecError::Limit { what: "frame size" })?;
    match pixel {
        PixelFormat::None => Err(CodecError::InvalidApi {
            what: "pixel format",
        }),
        PixelFormat::Rgba8 => pixels
            .checked_mul(4)
            .ok_or(CodecError::Limit { what: "frame size" }),
        PixelFormat::Yuv420p => {
            if width % 2 != 0 || height % 2 != 0 {
                return Err(CodecError::InvalidApi {
                    what: "chroma size",
                });
            }
            let samples = pixels / 2;
            samples
                .checked_mul(3)
                .ok_or(CodecError::Limit { what: "frame size" })
        }
    }
}

pub trait Encoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError>;

    /// Encode one still. Audio codecs refuse this.
    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        let _ = (self, input);
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError>;

    fn reset(&mut self);

    /// A handle another thread can raise. The encoder checks it between
    /// frames, never inside a lock this crate does not take.
    fn cancel_flag(&self) -> Arc<AtomicBool>;
}

pub struct EncodeFactory {
    openers: Vec<Opener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
}

impl EncodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
        }
    }

    /// Later registrations do not hide an earlier one for the same id.
    /// The desktop registers a platform opener first, then the Rust opener.
    ///
    /// An opener that returns [`CodecError::Unsupported`] does not accept
    /// those parameters, so the next opener for that id is asked. Any other
    /// error is the answer and the walk stops.
    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        let mut matched = false;
        for opener in &self.openers {
            if opener.id != params.codec {
                continue;
            }
            matched = true;
            match (opener.open)(params) {
                Ok(encoder) => return Ok(encoder),
                Err(CodecError::Unsupported { .. }) => {}
                Err(err) => return Err(err),
            }
        }
        Err(CodecError::Unsupported {
            what: if matched {
                "encoder rejected the parameters"
            } else {
                "encoder is not registered"
            },
        })
    }
}

impl Default for EncodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Raised by the owner of [`Encoder::cancel_flag`].
pub fn request_cancel(flag: &AtomicBool) {
    flag.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{
        AudioInput, CpuFrame, EncodeFactory, Encoder, FrameInput, FrameTime, PlatformFrame,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::request_cancel;
    use cutlass_codec::CodecId;
    use cutlass_codec::access::{ChannelLayout, CodecError, CodecParams, Packet, Pcm, PixelFormat};

    struct Copy;

    impl Encoder for Copy {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            let Some(samples) = input.pcm.as_f32() else {
                return Err(CodecError::InvalidApi {
                    what: "copy encoder wants f32",
                });
            };
            if samples.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![Packet::audio(
                samples
                    .iter()
                    .flat_map(|sample| sample.to_bits().to_le_bytes())
                    .collect(),
                0,
                i64::from(input.pcm.samples),
                input.pcm.rate,
            )])
        }

        fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
            Err(CodecError::Unsupported { what: "picture" })
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    fn open_copy(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported { what: "copy" });
        }
        Ok(Box::new(Copy))
    }

    #[test]
    fn empty_push_emits_nothing() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, open_copy);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let mut encoder = factory.open(&params).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), Vec::new()).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("push");
        assert!(packets.is_empty());
        assert!(encoder.drain().expect("drain").is_empty());
        let image = cutlass_codec::access::StillImage::from_samples(
            1,
            1,
            cutlass_codec::access::ColorType::Gray,
            cutlass_codec::access::BitDepth::Eight,
            vec![0],
        )
        .checked()
        .expect("still");
        let err = encoder
            .push_picture(&super::PictureInput { image })
            .expect_err("audio");
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "picture input",
            }
        );
    }

    fn refuse_params(_params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        Err(CodecError::Unsupported { what: "platform" })
    }

    fn reject_params(_params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        Err(CodecError::InvalidApi { what: "layout" })
    }

    #[test]
    fn missing_encoder_is_unsupported() {
        let err = match EncodeFactory::new().open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        )) {
            Err(err) => err,
            Ok(_) => panic!("missing"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "encoder is not registered",
            }
        );
    }

    #[test]
    fn a_declining_opener_falls_through_to_the_next() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, refuse_params);
        factory.register(CodecId::Pcm, open_copy);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let encoder = factory.open(&params).expect("open");
        assert_eq!(encoder.codec(), CodecId::Pcm);
    }

    #[test]
    fn the_first_opener_that_accepts_is_used() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, open_copy);
        factory.register(CodecId::Pcm, reject_params);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        assert!(factory.open(&params).is_ok());
    }

    #[test]
    fn a_hard_error_stops_the_walk() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, reject_params);
        factory.register(CodecId::Pcm, open_copy);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let err = match factory.open(&params) {
            Err(err) => err,
            Ok(_) => panic!("accepted"),
        };
        assert_eq!(err, CodecError::InvalidApi { what: "layout" });
    }

    #[test]
    fn every_opener_declining_is_a_named_error() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Aac, refuse_params);
        let err = match factory.open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        )) {
            Err(err) => err,
            Ok(_) => panic!("accepted"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "encoder rejected the parameters",
            }
        );
    }

    struct Stamp {
        width: u32,
        height: u32,
        pixel: PixelFormat,
        cancel: Arc<AtomicBool>,
    }

    impl Encoder for Stamp {
        fn codec(&self) -> CodecId {
            CodecId::Mjpeg
        }

        fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            Err(CodecError::Unsupported { what: "pcm" })
        }

        fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(CodecError::Cancelled);
            }
            let (data, time) = match input {
                FrameInput::Cpu(frame) => {
                    if frame.width() != self.width || frame.height() != self.height {
                        return Err(CodecError::InvalidApi { what: "frame size" });
                    }
                    if frame.pixel() != self.pixel {
                        return Err(CodecError::InvalidApi {
                            what: "pixel format",
                        });
                    }
                    (frame.bytes().to_vec(), frame.time())
                }
                FrameInput::Platform(frame) => {
                    if frame.width() != self.width || frame.height() != self.height {
                        return Err(CodecError::InvalidApi { what: "frame size" });
                    }
                    (frame.token().to_le_bytes().to_vec(), frame.time())
                }
            };
            Ok(vec![packet(data, time)])
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.cancel)
        }
    }

    fn packet(data: Vec<u8>, time: FrameTime) -> Packet {
        Packet {
            stream: 0,
            data,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
            keyframe: true,
            side: Vec::new(),
        }
    }

    fn open_stamp(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.codec != CodecId::Mjpeg {
            return Err(CodecError::Unsupported { what: "stamp" });
        }
        if params.width == 0 || params.height == 0 || params.pixel == PixelFormat::None {
            return Err(CodecError::Unsupported { what: "frame size" });
        }
        Ok(Box::new(Stamp {
            width: params.width,
            height: params.height,
            pixel: params.pixel,
            cancel: Arc::new(AtomicBool::new(false)),
        }))
    }

    fn mjpeg(width: u32, height: u32, pixel: PixelFormat) -> CodecParams {
        CodecParams {
            codec: CodecId::Mjpeg,
            width,
            height,
            pixel,
            sample_rate: 0,
            layout: None,
            sample_format: None,
            ..CodecParams::audio(CodecId::Mjpeg, 0, ChannelLayout::mono())
        }
    }

    fn time(pts: i64) -> FrameTime {
        FrameTime::new(pts, 1, 30).expect("time")
    }

    fn rgba(pts: i64) -> CpuFrame {
        CpuFrame::new(1, 1, PixelFormat::Rgba8, vec![9, 8, 7, 6], time(pts)).expect("rgba")
    }

    #[test]
    fn a_picture_reaches_the_encoder_that_accepted_the_parameters() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Mjpeg, refuse_params);
        factory.register(CodecId::Mjpeg, open_stamp);
        let mut encoder = factory
            .open(&mjpeg(1, 1, PixelFormat::Rgba8))
            .expect("open");

        let cpu = rgba(4);
        let cpu_packet = packet(cpu.bytes().to_vec(), cpu.time());
        let packets = encoder
            .push_frame(&FrameInput::Cpu(cpu))
            .expect("cpu frame");
        assert_eq!(packets, vec![cpu_packet]);

        let platform = PlatformFrame::new(1, 1, 11, time(5)).expect("platform");
        let platform_packet = packet(11u64.to_le_bytes().to_vec(), platform.time());
        let packets = encoder
            .push_frame(&FrameInput::Platform(platform))
            .expect("platform frame");
        assert_eq!(packets, vec![platform_packet]);

        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.0]).expect("pcm");
        let err = encoder.push(&AudioInput { pcm }).expect_err("pcm");
        assert_eq!(err, CodecError::Unsupported { what: "pcm" });
        assert!(encoder.drain().expect("drain").is_empty());
        assert!(encoder.drain().expect("second drain").is_empty());
    }

    #[test]
    fn a_picture_the_encoder_rejects_adds_nothing() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Mjpeg, open_stamp);
        let mut encoder = factory
            .open(&mjpeg(2, 2, PixelFormat::Rgba8))
            .expect("open");
        let yuv = CpuFrame::new(2, 2, PixelFormat::Yuv420p, vec![0; 6], time(0)).expect("yuv");
        let err = encoder
            .push_frame(&FrameInput::Cpu(yuv))
            .expect_err("format");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "pixel format"
            }
        );
        let small = PlatformFrame::new(1, 1, 3, time(0)).expect("small");
        let err = encoder
            .push_frame(&FrameInput::Platform(small))
            .expect_err("size");
        assert_eq!(err, CodecError::InvalidApi { what: "frame size" });

        let bytes = vec![4; 16];
        let cpu = CpuFrame::new(2, 2, PixelFormat::Rgba8, bytes, time(1)).expect("rgba");
        let expected = packet(cpu.bytes().to_vec(), cpu.time());
        let packets = encoder
            .push_frame(&FrameInput::Cpu(cpu))
            .expect("later frame");
        assert_eq!(packets, vec![expected]);
    }

    #[test]
    fn a_cancelled_picture_adds_nothing() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Mjpeg, open_stamp);
        let mut encoder = factory
            .open(&mjpeg(1, 1, PixelFormat::Rgba8))
            .expect("open");
        let flag = encoder.cancel_flag();
        request_cancel(&flag);
        let err = encoder
            .push_frame(&FrameInput::Cpu(rgba(0)))
            .expect_err("cancel");
        assert_eq!(err, CodecError::Cancelled);
        flag.store(false, Ordering::Relaxed);
        let cpu = rgba(2);
        let expected = packet(cpu.bytes().to_vec(), cpu.time());
        let packets = encoder
            .push_frame(&FrameInput::Cpu(cpu))
            .expect("after cancel");
        assert_eq!(packets, vec![expected]);
    }

    #[test]
    fn an_audio_encoder_refuses_a_picture_and_still_copies_pcm() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, open_copy);
        let mut encoder = factory
            .open(&CodecParams::audio(
                CodecId::Pcm,
                48_000,
                ChannelLayout::mono(),
            ))
            .expect("open");
        let err = encoder
            .push_frame(&FrameInput::Cpu(rgba(0)))
            .expect_err("picture");
        assert_eq!(err, CodecError::Unsupported { what: "picture" });
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.5]).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("pcm");
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, 0.5f32.to_bits().to_le_bytes());
    }

    #[test]
    fn a_picture_must_match_its_layout_and_clock() {
        let time = FrameTime::new(0, 1, 30).expect("time");
        assert_eq!(
            CpuFrame::new(1, 1, PixelFormat::Rgba8, vec![1, 2, 3], time),
            Err(CodecError::InvalidApi {
                what: "picture length",
            })
        );
        assert_eq!(
            CpuFrame::new(1, 1, PixelFormat::None, Vec::new(), time),
            Err(CodecError::InvalidApi {
                what: "pixel format",
            })
        );
        assert_eq!(
            CpuFrame::new(1, 1, PixelFormat::Yuv420p, vec![0; 6], time),
            Err(CodecError::InvalidApi {
                what: "chroma size"
            })
        );
        assert_eq!(
            CpuFrame::new(0, 1, PixelFormat::Rgba8, Vec::new(), time),
            Err(CodecError::InvalidApi { what: "frame size" })
        );
        let yuv =
            CpuFrame::new(2, 2, PixelFormat::Yuv420p, vec![1, 2, 3, 4, 5, 6], time).expect("yuv");
        assert_eq!(yuv.bytes().len(), 6);
        assert_eq!(
            FrameTime::new(-1, 0, 30),
            Err(CodecError::InvalidApi { what: "duration" })
        );
        assert_eq!(
            FrameTime::new(0, 1, 0),
            Err(CodecError::InvalidApi { what: "timescale" })
        );
        let early = FrameTime::new(-5, 1, 30).expect("early");
        assert_eq!(early.pts(), -5);
        assert_eq!(
            PlatformFrame::new(0, 1, 0, time),
            Err(CodecError::InvalidApi { what: "frame size" })
        );
        let platform = PlatformFrame::new(2, 2, 0, early).expect("platform");
        assert_eq!(platform.token(), 0);
        assert_eq!(platform.time().pts(), -5);
    }
}
