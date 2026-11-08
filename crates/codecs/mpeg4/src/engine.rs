//! MPEG-4 Visual on the engine's codec traits.
//!
//! Playback opens a [`Decoder`] through [`DecodeFactory::open`] and takes
//! packed 4:2:0 pictures in display order. The decoder configuration is
//! the `esds` DecoderSpecificInfo: the VOS, visual object and VOL headers.
//! Export opens an [`Encoder`](cutlass_encode::VideoEncoder) the same way.
//! Each packet is one VOP. The configuration is the video object and the
//! layer, with no picture in it.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, DecodeFactory, Decoder, Output,
    Packet, PictureInfo, PictureOutput, PixelFormat, SideData, Transfer,
};
use cutlass_encode::{EncodeFactory, VideoEncoder, VideoInput};

use crate::enc::Pictures;

use crate::dec::{self, Image};
use crate::header::Colour;

/// The largest picture playback opens: the 13-bit VOL edges allow 8191.
const MAX_PIXELS: u64 = 8192 * 8192;

/// Registers the MPEG-4 Visual decoder that playback opens.
pub fn register_engine_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mpeg4, open_engine_decoder);
}

/// Registers the MPEG-4 Visual encoder that export opens.
///
/// A platform encoder registered earlier is asked first. This one runs
/// when that opener is absent or declines the parameters.
pub fn register_engine_encoders(factory: &mut EncodeFactory) {
    factory.register_video(CodecId::Mpeg4, open_engine_encoder);
}

fn open_engine_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(EngineDecoder::open(params)?))
}

fn open_engine_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(EngineEncoder::open(params)?))
}

/// MPEG-4 Visual for playback: one packed 4:2:0 picture per displayed VOP.
///
/// A not-coded VOP gives no picture, so the previous one stays on screen.
/// Pictures leave in display order; each takes the smallest packet time
/// not yet given out, which is its presentation time when the container
/// stores them in decode order. The engine's 4:2:0 pictures have even
/// edges, so an odd-sized stream drops its last column or row.
pub struct EngineDecoder {
    inner: dec::Decoder,
    pts: BinaryHeap<Reverse<i64>>,
    timescale: u32,
}

impl EngineDecoder {
    /// Opens with the container's parameters; `params.extra` holds the
    /// stream headers when the container carries them out of band.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Self::open_limited(params, MAX_PIXELS)
    }

    /// [`Self::open`] refusing layers over `max_pixels`.
    pub(crate) fn open_limited(params: &CodecParams, max_pixels: u64) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mpeg4 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut inner = dec::Decoder::with_max_pixels(max_pixels);
        inner.configure(&params.extra)?;
        Ok(Self {
            inner,
            pts: BinaryHeap::new(),
            timescale: 0,
        })
    }

    fn output(&mut self, image: Image, fallback: i64) -> Result<Output, CodecError> {
        let pts = self.pts.pop().map_or(fallback, |Reverse(pts)| pts);
        let colour = self.inner.colour();
        Ok(Output::Picture(PictureOutput::cpu(
            info(&image, colour),
            image.bytes,
            pts,
            self.timescale,
        )?))
    }
}

/// The picture's description with the visual object's colour.
fn info(image: &Image, colour: Colour) -> PictureInfo {
    let primaries = colour
        .primaries
        .map_or(ColorPrimaries::Unspecified, primaries);
    let transfer = colour.transfer.map_or(Transfer::Unspecified, transfer);
    let mut info = PictureInfo::new(
        image.width,
        image.height,
        PixelFormat::Yuv420p,
        primaries,
        transfer,
    );
    if let Some(code) = colour.matrix {
        info.matrix = matrix(code);
        info.cicp_matrix = code;
    }
    info
}

/// ISO/IEC 14496-2 Table 6-7, the ISO/IEC 23091-2 code points.
fn primaries(code: u8) -> ColorPrimaries {
    match code {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Unspecified,
    }
}

/// ISO/IEC 14496-2 Table 6-8.
fn transfer(code: u8) -> Transfer {
    match code {
        1 => Transfer::Bt709,
        4..=6 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        _ => Transfer::Unspecified,
    }
}

/// ISO/IEC 14496-2 Table 6-9.
fn matrix(code: u8) -> ColorMatrix {
    match code {
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        _ => ColorMatrix::Unspecified,
    }
}

impl Decoder for EngineDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg4
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        for side in &packet.side {
            if let SideData::Config(config) = side {
                self.inner.configure(config)?;
            }
        }
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }
        self.timescale = packet.timescale;
        let decoded = self.inner.decode(&packet.data)?;
        if decoded.timed {
            self.pts.push(Reverse(packet.pts));
        }
        decoded
            .pictures
            .into_iter()
            .map(|image| self.output(image, packet.pts))
            .collect()
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let fallback = self.pts.peek().map_or(0, |Reverse(pts)| *pts);
        let out = self
            .inner
            .drain()
            .into_iter()
            .map(|image| self.output(image, fallback))
            .collect();
        self.pts.clear();
        out
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.pts.clear();
    }
}

/// MPEG-4 Visual for export.
///
/// Pictures are 4:2:0. RGBA in the parameters is accepted because export
/// converts each frame before [`VideoEncoder::push`]. The stream is
/// Advanced Simple: an I-VOP, then P-VOPs with the picture before each as
/// a B-VOP, quarter-sample motion, and the H.263 quantiser. A named bitrate
/// moves the quantiser. Zero keeps the starting one. GMC, sprites,
/// interlace, data partitioning, resync and RVLC are not written. A height
/// or width that is not a multiple of 16 is padded, and the layer keeps
/// the visible size.
pub struct EngineEncoder {
    inner: Pictures,
    width: u32,
    height: u32,
    timescale: u32,
    cancel: Arc<AtomicBool>,
}

impl EngineEncoder {
    /// Opens at `params.bitrate` bits a second. `nal_length_size` is not a
    /// length prefix: each packet is one VOP, beginning with its start code.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mpeg4 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if !matches!(params.pixel, PixelFormat::Yuv420p | PixelFormat::Rgba8) {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let inner = Pictures::open(
            params.width,
            params.height,
            params.bitrate,
            params.constant_rate,
        )?;
        Ok(Self {
            inner,
            width: params.width,
            height: params.height,
            timescale: 0,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl VideoEncoder for EngineEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg4
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let picture = &input.picture;
        if picture.width() != self.width || picture.height() != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        let packets = self.inner.push(crate::enc::PictureIn {
            y: picture.y(),
            u: picture.u(),
            v: picture.v(),
            pts: input.pts,
            duration: input.duration,
            timescale: input.timescale,
            keyframe: input.force_keyframe,
        })?;
        self.timescale = input.timescale;
        Ok(packets
            .into_iter()
            .map(|packet| {
                Packet::video(
                    packet.bytes,
                    packet.pts,
                    packet.duration,
                    input.timescale,
                    packet.keyframe,
                )
            })
            .collect())
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let timescale = self.timescale;
        Ok(self
            .inner
            .drain()?
            .into_iter()
            .map(|packet| {
                Packet::video(
                    packet.bytes,
                    packet.pts,
                    packet.duration,
                    timescale,
                    packet.keyframe,
                )
            })
            .collect())
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.timescale = 0;
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    fn extradata(&self) -> &[u8] {
        self.inner.headers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::Yuv420p;

    const ASP: &[u8] = include_bytes!("../tests/fixtures/asp.m4v");

    fn packet(data: &[u8]) -> Packet {
        Packet {
            stream: 0,
            data: data.to_vec(),
            pts: 0,
            duration: 1,
            timescale: 25,
            keyframe: true,
            side: Vec::new(),
        }
    }

    fn first_vop() -> usize {
        ASP.windows(4)
            .position(|w| w == [0, 0, 1, 0xB6])
            .expect("a VOP")
    }

    #[test]
    fn a_layer_over_the_pixel_limit_is_refused() {
        let params = CodecParams::video(CodecId::Mpeg4, 0, 0);
        let mut decoder = EngineDecoder::open_limited(&params, 48 * 32 - 1).expect("open");
        let refused = decoder.push(&packet(ASP));
        assert!(
            matches!(refused, Err(CodecError::Limit { .. })),
            "{refused:?}"
        );

        let mut configured = params.clone();
        configured.extra = ASP[..first_vop()].to_vec();
        let refused = EngineDecoder::open_limited(&configured, 48 * 32 - 1)
            .and_then(|mut d| d.push(&packet(&ASP[first_vop()..])));
        assert!(
            matches!(refused, Err(CodecError::Limit { .. })),
            "{refused:?}"
        );

        let mut decoder = EngineDecoder::open_limited(&params, 48 * 32).expect("open");
        assert!(!decoder.push(&packet(ASP)).expect("push").is_empty());
    }

    #[test]
    fn reserved_colour_codes_are_unspecified() {
        let colour = Colour {
            full_range: false,
            primaries: Some(3),
            transfer: Some(0),
            matrix: Some(200),
        };
        let image = Image {
            width: 2,
            height: 2,
            bytes: vec![0; 6],
        };
        let reserved = info(&image, colour);
        assert_eq!(reserved.primaries, ColorPrimaries::Unspecified);
        assert_eq!(reserved.transfer, Transfer::Unspecified);
        assert_eq!(reserved.matrix, ColorMatrix::Unspecified);
        assert_eq!(reserved.cicp_matrix, 200);

        let bt709 = Colour {
            full_range: false,
            primaries: Some(1),
            transfer: Some(1),
            matrix: Some(1),
        };
        let standard = info(&image, bt709);
        assert_eq!(standard.primaries, ColorPrimaries::Bt709);
        assert_eq!(standard.transfer, Transfer::Bt709);
        assert_eq!(standard.matrix, ColorMatrix::Bt709);
    }

    #[test]
    fn export_packets_come_back_in_display_order() {
        let mut params = CodecParams::with_format(CodecId::Mpeg4, 64, 48, PixelFormat::Rgba8);
        params.bitrate = 400_000;
        let mut factory = EncodeFactory::new();
        register_engine_encoders(&mut factory);
        let mut encoder = factory.open(&params).expect("open");
        let mut packets = Vec::new();
        for n in 0..4 {
            let (y, u, v) = moving(n);
            let picture = Yuv420p::try_new(64, 48, y, u, v).expect("picture");
            packets.extend(
                encoder
                    .push_frame(&cutlass_encode::FrameInput::Cpu(
                        cutlass_encode::CpuFrame::new(
                            64,
                            48,
                            PixelFormat::Yuv420p,
                            packed(&picture),
                            cutlass_encode::FrameTime::new(
                                i64::from(u32::try_from(n).unwrap_or(0)),
                                1,
                                30,
                            )
                            .expect("time"),
                        )
                        .expect("frame"),
                    ))
                    .expect("push"),
            );
        }
        packets.extend(encoder.drain().expect("drain"));
        let key = packets.iter().filter(|packet| packet.keyframe).count();
        assert_eq!(key, 1);
        assert!(
            packets.iter().any(|packet| {
                packet
                    .side
                    .iter()
                    .any(|side| matches!(side, SideData::Config(_)))
            }),
            "export reads the layer from the first packet"
        );
        assert!(packets.iter().any(|packet| vop_type(&packet.data) == 2));
        assert!(
            packets
                .iter()
                .all(|packet| packet.data.starts_with(&[0, 0, 1, 0xB6]))
        );

        let mut decoder =
            EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 64, 48)).expect("decoder");
        let mut shown = Vec::new();
        for packet in &packets {
            if let Some(config) = packet.side.iter().find_map(|side| match side {
                SideData::Config(config) => Some(config.clone()),
                _ => None,
            }) {
                assert!(
                    !config.windows(4).any(|w| w == [0, 0, 1, 0xB6]),
                    "the configuration holds a picture"
                );
            }
            for output in decoder.push(packet).expect("decode") {
                shown.push(output);
            }
        }
        for output in decoder.drain().expect("drain") {
            shown.push(output);
        }
        let times: Vec<i64> = shown
            .iter()
            .map(|output| match output {
                Output::Picture(picture) => picture.pts(),
                Output::Audio(_) => panic!("a picture stream returned sound"),
            })
            .collect();
        assert_eq!(times, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_keyframe_flushes_the_held_picture_first() {
        let params = CodecParams::video(CodecId::Mpeg4, 32, 32);
        let mut encoder = EngineEncoder::open(&params).expect("open");
        let mut kinds = Vec::new();
        for (n, key) in [(0, false), (1, false), (2, true)] {
            let picture = Yuv420p::solid(32, 32, 40 + u8::try_from(n).unwrap_or(0), 128, 128)
                .expect("picture");
            for packet in encoder
                .push(&VideoInput {
                    picture,
                    pts: n,
                    duration: 1,
                    timescale: 25,
                    force_keyframe: key,
                })
                .expect("push")
            {
                kinds.push((packet.pts, vop_type(&packet.data), packet.keyframe));
            }
        }
        assert_eq!(kinds, vec![(0, 0, true), (1, 1, false), (2, 0, true)]);
    }

    #[test]
    fn a_bad_picture_is_refused() {
        let params = CodecParams::video(CodecId::Mpeg4, 64, 48);
        let mut encoder = EngineEncoder::open(&params).expect("open");
        let picture = Yuv420p::solid(32, 32, 10, 128, 128).expect("picture");
        let refused = encoder.push(&VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 0,
            force_keyframe: false,
        });
        assert!(matches!(refused, Err(CodecError::InvalidApi { .. })));
        assert!(EngineEncoder::open(&CodecParams::video(CodecId::H264, 64, 48)).is_err());
        assert!(EngineEncoder::open(&CodecParams::video(CodecId::Mpeg4, 64, 15)).is_err());
        encoder.cancel_flag().store(true, Ordering::Relaxed);
        let picture = Yuv420p::solid(64, 48, 10, 128, 128).expect("picture");
        let refused = encoder.push(&VideoInput {
            picture,
            pts: 0,
            duration: 1,
            timescale: 30,
            force_keyframe: false,
        });
        assert!(matches!(refused, Err(CodecError::Cancelled)));
    }

    fn vop_type(packet: &[u8]) -> u8 {
        packet.get(4).map_or(255, |byte| byte >> 6)
    }

    fn moving(n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let shift = n * 2;
        let mut y = vec![0u8; 64 * 48];
        let mut u = vec![0u8; 32 * 24];
        let mut v = vec![0u8; 32 * 24];
        for row in 0..48 {
            for col in 0..64 {
                y[row * 64 + col] = (((row + col + shift) * 3 + (col ^ row)) % 180 + 20) as u8;
            }
        }
        for row in 0..24 {
            for col in 0..32 {
                let at = row * 32 + col;
                u[at] = ((col * 5 + row + shift) % 80 + 80) as u8;
                v[at] = ((row * 5 + col + shift / 2) % 80 + 80) as u8;
            }
        }
        (y, u, v)
    }

    fn packed(picture: &Yuv420p) -> Vec<u8> {
        let mut bytes = picture.y().to_vec();
        bytes.extend_from_slice(picture.u());
        bytes.extend_from_slice(picture.v());
        bytes
    }
}
