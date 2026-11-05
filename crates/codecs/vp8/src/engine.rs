//! The VP8 decoder and encoder on the engine's codec traits.
//!
//! Playback opens a [`Decoder`] through [`DecodeFactory::open`] and takes
//! packed 4:2:0 pictures. Export opens an encoder through
//! [`EncodeFactory::open`], which converts its RGBA frames to BT.709 4:2:0
//! and drives the [`VideoEncoder`] registered here; the decoder
//! configuration is the `vpcC` payload ([`VideoEncoder::extradata`]).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, DecodeFactory, Decoder, Output, Packet,
    PictureInfo, PictureOutput, PixelFormat, Transfer,
};
use cutlass_encode::video::VideoEncoder as _;
use cutlass_encode::{EncodeFactory, Plane, SideData, VideoEncoder, VideoFrame, VideoInput};

use crate::decode;
use crate::encode::{Vp8Encoder, Vp8Options};

/// The largest picture playback opens: 8192 by 8192.
const MAX_PIXELS: u64 = 8192 * 8192;

/// Registers the VP8 decoder that playback opens.
pub fn register_engine_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Vp8, open_engine_decoder);
}

/// Registers the VP8 encoder that export opens.
pub fn register_engine_encoders(factory: &mut EncodeFactory) {
    factory.register_video(CodecId::Vp8, open_engine_encoder);
}

fn open_engine_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(EngineDecoder::open(params)?))
}

fn open_engine_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(EngineEncoder::open(params)?))
}

/// VP8 for playback: one packed 4:2:0 picture per shown frame.
///
/// A hidden frame gives no picture. The engine's 4:2:0 pictures have even
/// edges, so an odd-sized stream drops its last column or row, as export
/// does when it codes an odd canvas.
pub struct EngineDecoder {
    inner: decode::Decoder,
    primaries: ColorPrimaries,
    transfer: Transfer,
}

impl EngineDecoder {
    /// Opens with the container's parameters. The colour comes from the
    /// `vpcC` payload in `params.extra` when there is one, and is BT.601
    /// otherwise, which is what a VP8 frame with colour space 0 means
    /// (RFC 6386 Section 9.2).
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp8 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let (primaries, transfer) = vpcc_colour(&params.extra);
        Ok(Self {
            inner: decode::Decoder::with_max_pixels(MAX_PIXELS),
            primaries,
            transfer,
        })
    }
}

/// Primaries and transfer from a `vpcC` payload (version 1): byte 7 is the
/// ISO/IEC 23091-2 colour primaries code, byte 8 the transfer code.
fn vpcc_colour(extra: &[u8]) -> (ColorPrimaries, Transfer) {
    let (Some(&primaries), Some(&transfer)) = (extra.get(7), extra.get(8)) else {
        return (ColorPrimaries::Bt601, Transfer::Bt601);
    };
    let primaries = match primaries {
        1 => ColorPrimaries::Bt709,
        5 | 6 => ColorPrimaries::Bt601,
        9 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Bt601,
    };
    let transfer = match transfer {
        1 => Transfer::Bt709,
        6 => Transfer::Bt601,
        14 | 15 => Transfer::Bt2020,
        _ => Transfer::Bt601,
    };
    (primaries, transfer)
}

impl Decoder for EngineDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }
        let Some(picture) = self.inner.decode(&packet.data)? else {
            return Ok(Vec::new());
        };
        let width = (picture.width & !1).max(2);
        let height = (picture.height & !1).max(2);
        let bytes = packed_even(&picture, width, height);
        let info = PictureInfo {
            width,
            height,
            pixel: PixelFormat::Yuv420p,
            primaries: self.primaries,
            transfer: self.transfer,
        };
        Ok(vec![Output::Picture(PictureOutput::cpu(
            info,
            bytes,
            packet.pts,
            packet.timescale,
        )?)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// The picture's top-left `width` by `height` (both even) as packed
/// 4:2:0. A 1-pixel edge repeats its one sample.
fn packed_even(picture: &decode::Picture, width: u32, height: u32) -> Vec<u8> {
    let frame = &picture.frame;
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    let ys = frame.mbw * 16;
    let cs = frame.mbw * 8;
    // The reconstruction covers whole macroblocks, so a 1-pixel picture
    // still has the second row and column to read.
    for r in 0..h {
        out.extend_from_slice(&frame.y[r * ys..r * ys + w]);
    }
    for plane in [&frame.u, &frame.v] {
        for r in 0..ch {
            out.extend_from_slice(&plane[r * cs..r * cs + cw]);
        }
    }
    out
}

/// VP8 for export: 4:2:0 BT.709 pictures in, one packet per picture.
pub struct EngineEncoder {
    inner: Vp8Encoder,
    extradata: Vec<u8>,
}

impl EngineEncoder {
    /// Opens at `params.bitrate` bits a second (0 keeps the default
    /// quantizer). The parameters may name RGBA: export converts each
    /// frame to 4:2:0 before it reaches this encoder.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp8 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if !matches!(params.pixel, PixelFormat::Yuv420p | PixelFormat::Rgba8) {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let opts = Vp8Options {
            bitrate: params.bitrate,
            ..Vp8Options::default()
        };
        let inner = Vp8Encoder::with_options(params.width, params.height, opts)?;
        // BT.709 primaries, transfer, and matrix: the colour export's
        // conversion uses.
        let extradata = inner.vpcc_with_colour([1, 1, 1]);
        Ok(Self { inner, extradata })
    }
}

impl VideoEncoder for EngineEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        let picture = &input.picture;
        let (width, height) = (picture.width(), picture.height());
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let plane = |w: u32, h: u32, bytes: &[u8]| {
            Plane::tight(w, h, bytes.to_vec()).map_err(|_| CodecError::InvalidApi {
                what: "picture size",
            })
        };
        let frame = VideoFrame::yuv420p(
            width,
            height,
            plane(width, height, picture.y())?,
            plane(cw, ch, picture.u())?,
            plane(cw, ch, picture.v())?,
        )
        .map_err(|_| CodecError::InvalidApi {
            what: "picture size",
        })?;
        if input.force_keyframe {
            self.inner.force_keyframe();
        }
        let mut packets = self.inner.push(&cutlass_encode::video::VideoInput {
            frame,
            pts: input.pts,
            duration: input.duration,
            timescale: input.timescale,
        })?;
        // The configuration is `extradata`; the export adapter puts it on
        // the first packet.
        for packet in &mut packets {
            packet
                .side
                .retain(|side| !matches!(side, SideData::Config(_)));
        }
        Ok(packets)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        self.inner.drain()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.inner.cancel_flag()
    }

    fn extradata(&self) -> &[u8] {
        &self.extradata
    }
}

#[cfg(test)]
mod tests {
    use cutlass_decode::{
        CodecId, CodecParams, ColorPrimaries, DecodeFactory, Decoder, Output, PictureSamples,
        PixelFormat, Transfer,
    };
    use cutlass_encode::video::{VideoEncoder as _, VideoInput};
    use cutlass_encode::{CpuFrame, EncodeFactory, FrameInput, FrameTime, SideData};

    use super::{register_engine_decoders, register_engine_encoders};
    use crate::prove_encode::{crop, synthetic};
    use crate::{Vp8Encoder, Vp8Options};

    fn pictures(decoder: &mut dyn Decoder, packets: &[cutlass_decode::Packet]) -> Vec<Output> {
        let mut out = Vec::new();
        for packet in packets {
            out.extend(decoder.push(packet).expect("decode"));
        }
        out.extend(decoder.drain().expect("drain"));
        out
    }

    /// Export's path: RGBA frames through the general factory, one
    /// configuration (BT.709 `vpcC`) on the first packet, a forced key
    /// frame honoured, and the pictures back through playback's decoder.
    #[test]
    fn export_and_playback_reach_vp8_through_the_engine_factories() {
        let mut encoders = EncodeFactory::new();
        register_engine_encoders(&mut encoders);
        let mut params = CodecParams::with_format(CodecId::Vp8, 64, 32, PixelFormat::Rgba8);
        params.bitrate = 400_000;
        let mut encoder = encoders.open(&params).expect("adapted encoder");
        let mut packets = Vec::new();
        for index in 0..4 {
            let time = FrameTime::new(index, 1, 30).expect("time");
            // Orange: RGB 255, 128, 0.
            let rgba = [255_u8, 128, 0, 255].repeat(64 * 32);
            let frame = CpuFrame::new(64, 32, PixelFormat::Rgba8, rgba, time).expect("frame");
            packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        assert_eq!(packets.len(), 4);
        let configs: Vec<_> = packets
            .iter()
            .flat_map(|packet| &packet.side)
            .filter_map(|side| match side {
                SideData::Config(config) => Some(config.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(configs.len(), 1, "one configuration");
        assert_eq!(&configs[0][7..10], &[1, 1, 1], "BT.709 colour");
        assert!(packets[0].keyframe && !packets[1].keyframe);

        let mut decoders = DecodeFactory::new();
        register_engine_decoders(&mut decoders);
        let mut read = CodecParams::video(CodecId::Vp8, 64, 32);
        read.extra = configs[0].clone();
        let mut decoder = decoders.open(&read).expect("decoder");
        let out = pictures(decoder.as_mut(), &packets);
        assert_eq!(out.len(), 4);
        for (index, output) in out.iter().enumerate() {
            let Output::Picture(picture) = output else {
                panic!("audio");
            };
            let info = picture.info();
            assert_eq!((info.width, info.height), (64, 32));
            assert_eq!(info.primaries, ColorPrimaries::Bt709);
            assert_eq!(info.transfer, Transfer::Bt709);
            assert_eq!(picture.pts(), index as i64);
            let PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("platform buffer");
            };
            // BT.709 limited range orange, as export converts it: Y 141, Cb 59,
            // Cr 189.
            let (y, rest) = bytes.split_at(64 * 32);
            let (u, v) = rest.split_at(rest.len() / 2);
            assert!(y.iter().all(|&s| s.abs_diff(141) <= 3), "luma");
            assert!(u.iter().all(|&s| s.abs_diff(59) <= 3), "cb");
            assert!(v.iter().all(|&s| s.abs_diff(189) <= 3), "cr");
        }
    }

    /// Odd sizes lose their last column or row, hidden frames give no
    /// picture, and a stream without `vpcC` is BT.601.
    #[test]
    fn playback_crops_odd_sizes_and_skips_hidden_frames() {
        for (width, height) in [(33, 17), (1, 1), (48, 31)] {
            let opts = Vp8Options {
                hidden_altref: true,
                golden_interval: 3,
                ..Vp8Options::default()
            };
            let mut encoder = Vp8Encoder::with_options(width, height, opts).expect("open");
            let mut packets = Vec::new();
            for t in 0..7 {
                let input = VideoInput {
                    frame: synthetic(width, height, t),
                    pts: i64::from(t),
                    duration: 1,
                    timescale: 30,
                };
                packets.extend(encoder.push(&input).expect("push"));
            }
            packets.extend(encoder.drain().expect("drain"));
            assert!(packets.len() > 7, "hidden frames");
            let mut decoders = DecodeFactory::new();
            register_engine_decoders(&mut decoders);
            let read = CodecParams::video(CodecId::Vp8, width, height);
            let mut decoder = decoders.open(&read).expect("decoder");
            let out = pictures(decoder.as_mut(), &packets);
            assert_eq!(out.len(), 7, "{width}x{height}");
            let (w, h) = ((width & !1).max(2), (height & !1).max(2));
            for (output, recon) in out.iter().zip(&encoder.shown_recon) {
                let Output::Picture(picture) = output else {
                    panic!("audio");
                };
                let info = picture.info();
                assert_eq!((info.width, info.height), (w, h));
                assert_eq!(info.primaries, ColorPrimaries::Bt601);
                let PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("platform buffer");
                };
                let mut expected = Vec::new();
                crop(recon, w, h, &mut expected);
                assert_eq!(bytes, expected.as_slice(), "{width}x{height}");
            }
        }
    }
}
