//! Profile-0 encoder.
//!
//! Pictures are error resilient, shown, and stored in every reference slot.
//! Quantizer 0 is lossless 4×4 Walsh-Hadamard. Any other quantizer uses the
//! forward DCT with transforms no larger than 8×8. A factory bitrate of zero
//! selects quantizer 32 and does not chase a rate. A positive bitrate moves
//! the quantizer around 32 so the mean stays near the target.

mod coef;
mod frame;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    CodecError, CodecId, CodecParams, Packet, PixelFormat, VideoEncoder, VideoFrame, VideoInput,
};

use self::frame::{Source, code_frame};
use crate::bits::{BitWriter, BoolWriter};
use crate::geom::{self, ALLOW_8X8};
use crate::header;
use crate::prob::FrameContext;
use crate::recon::{Picture, Reference};

const GOP: u64 = 30;
const BASE_Q: i64 = 32;
const MIN_Q: i64 = 8;
const MAX_Q: i64 = 208;

/// Profile-0 encoder on the video codec contract.
pub struct Vp9Encoder {
    width: u32,
    height: u32,
    bitrate: u32,
    fixed_q: Option<u8>,
    frame_index: u64,
    debt: i64,
    reference: Option<Reference>,
    cancel: Arc<AtomicBool>,
}

impl Vp9Encoder {
    /// Lossless when `qindex` is 0. Any other index is a fixed quantizer.
    pub fn with_quantizer(width: u32, height: u32, qindex: u8) -> Result<Self, CodecError> {
        check_encode_size(width, height)?;
        Ok(Self::new(width, height, 0, Some(qindex)))
    }

    /// Bitrate 0 keeps quantizer 32. A positive bitrate aims at that rate.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Vp9 {
            return Err(CodecError::Unsupported {
                what: "vp9 encoder",
            });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "vp9 pixel format",
            });
        }
        check_encode_size(params.width, params.height)?;
        let fixed_q = if params.bitrate == 0 { Some(32) } else { None };
        Ok(Self::new(
            params.width,
            params.height,
            params.bitrate,
            fixed_q,
        ))
    }

    fn new(width: u32, height: u32, bitrate: u32, fixed_q: Option<u8>) -> Self {
        Self {
            width,
            height,
            bitrate,
            fixed_q,
            frame_index: 0,
            debt: 0,
            reference: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    fn qindex(&self) -> u8 {
        if let Some(qindex) = self.fixed_q {
            return qindex;
        }
        let q = BASE_Q + self.debt / 2_000;
        u8::try_from(q.clamp(MIN_Q, MAX_Q)).unwrap_or(32)
    }
}

impl VideoEncoder for Vp9Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        check_input(self, input)?;
        let key = self.reference.is_none() || self.frame_index.is_multiple_of(GOP);
        let qindex = self.qindex();
        let reference = if key { None } else { self.reference.as_ref() };
        let coded = code_frame(source_of(&input.frame), reference, qindex == 0, qindex)?;
        let data = assemble(key, self.width, self.height, qindex, &coded.tile)?;
        let picture = Picture {
            y: coded.y,
            u: coded.u,
            v: coded.v,
            width: self.width,
            height: self.height,
        };
        let reference = Reference::from_picture(&picture)?;
        self.reference = Some(reference);
        self.frame_index = self.frame_index.saturating_add(1);
        account(self, data.len(), input.duration, input.timescale);
        Ok(vec![Packet::video(
            data,
            input.pts,
            input.duration,
            input.timescale,
            key,
        )])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.frame_index = 0;
        self.debt = 0;
        self.reference = None;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

fn check_input(encoder: &Vp9Encoder, input: &VideoInput) -> Result<(), CodecError> {
    let frame = &input.frame;
    if frame.format != PixelFormat::Yuv420p {
        return Err(CodecError::Unsupported {
            what: "vp9 pixel format",
        });
    }
    if frame.width != encoder.width || frame.height != encoder.height {
        return Err(CodecError::InvalidApi {
            what: "picture size",
        });
    }
    if encoder.fixed_q.is_none() && (input.duration <= 0 || input.timescale == 0) {
        return Err(CodecError::InvalidApi {
            what: "frame clock",
        });
    }
    Ok(())
}

fn account(encoder: &mut Vp9Encoder, bytes: usize, duration: i64, timescale: u32) {
    if encoder.fixed_q.is_some() {
        return;
    }
    let target = target_bytes(encoder.bitrate, duration, timescale);
    let got = i64::try_from(bytes).unwrap_or(i64::MAX);
    encoder.debt = encoder.debt.saturating_add(got.saturating_sub(target));
}

fn target_bytes(bitrate: u32, duration: i64, timescale: u32) -> i64 {
    let ticks = u128::try_from(duration).unwrap_or(0);
    let bits = u128::from(bitrate).saturating_mul(ticks) / u128::from(timescale.max(1));
    i64::try_from(bits / 8).unwrap_or(i64::MAX)
}

fn source_of(frame: &VideoFrame) -> Source<'_> {
    Source {
        y: &frame.y.samples,
        u: &frame.u.samples,
        v: &frame.v.samples,
        y_stride: frame.y.stride,
        u_stride: frame.u.stride,
        v_stride: frame.v.stride,
        width: frame.width,
        height: frame.height,
    }
}

fn check_encode_size(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > geom::MAX_EDGE || height > geom::MAX_EDGE {
        return Err(CodecError::Limit {
            what: "vp9 frame edge",
        });
    }
    if !width.is_multiple_of(8) || !height.is_multiple_of(8) {
        return Err(CodecError::Unsupported {
            what: "vp9 frame size",
        });
    }
    Ok(())
}

fn assemble(
    key: bool,
    width: u32,
    height: u32,
    qindex: u8,
    tile: &[u8],
) -> Result<Vec<u8>, CodecError> {
    let compressed = compressed_header(qindex == 0, !key)?;
    let mut bytes = uncompressed(key, width, height, qindex, compressed.len())?;
    bytes.extend_from_slice(&compressed);
    bytes.extend_from_slice(tile);
    Ok(bytes)
}

fn uncompressed(
    key: bool,
    width: u32,
    height: u32,
    qindex: u8,
    compressed_len: usize,
) -> Result<Vec<u8>, CodecError> {
    let len = u32::try_from(compressed_len).unwrap_or(u32::MAX);
    if len == 0 || len > u32::from(u16::MAX) {
        return Err(CodecError::Limit {
            what: "compressed header",
        });
    }
    let mut bits = BitWriter::new();
    bits.write(2, 2);
    bits.write(0, 2);
    bits.write_bit(false);
    bits.write_bit(!key);
    bits.write_bit(true);
    bits.write_bit(true);
    if key {
        write_key_setup(&mut bits, width, height);
    } else {
        write_inter_setup(&mut bits);
    }
    bits.write(0, 2);
    bits.write(0, 6);
    bits.write(0, 3);
    bits.write_bit(false);
    bits.write(u32::from(qindex), 8);
    bits.write(0, 3);
    bits.write_bit(false);
    write_tile_bits(&mut bits, width);
    bits.write(len, 16);
    Ok(bits.finish())
}

fn write_key_setup(bits: &mut BitWriter, width: u32, height: u32) {
    bits.write(0x49, 8);
    bits.write(0x83, 8);
    bits.write(0x42, 8);
    bits.write(1, 3);
    bits.write_bit(false);
    bits.write(width - 1, 16);
    bits.write(height - 1, 16);
    bits.write_bit(false);
}

fn write_inter_setup(bits: &mut BitWriter) {
    bits.write(0xff, 8);
    for _ in 0..3 {
        bits.write(0, 3);
        bits.write_bit(false);
    }
    bits.write_bit(true);
    bits.write_bit(false);
    bits.write_bit(true);
    bits.write_bit(false);
    bits.write(1, 2);
}

fn write_tile_bits(bits: &mut BitWriter, width: u32) {
    let (min_log2, max_log2) = header::tile_limits(geom::mi_count(width));
    if max_log2 > min_log2 {
        bits.write_bit(false);
    }
    bits.write_bit(false);
}

fn compressed_header(lossless: bool, inter: bool) -> Result<Vec<u8>, CodecError> {
    let mut writer = BoolWriter::new();
    if !lossless {
        writer.write_literal(u32::from(ALLOW_8X8), 2);
    }
    let max_tx = if lossless {
        0
    } else {
        usize::from(geom::tx_mode_biggest(ALLOW_8X8))
    };
    for _ in 0..=max_tx {
        writer.write(0, 128);
    }
    for _ in 0..3 {
        writer.write(0, 252);
    }
    if inter {
        write_inter_probs(&mut writer);
    }
    writer.finish()
}

fn write_inter_probs(writer: &mut BoolWriter) {
    let fc = FrameContext::defaults();
    for row in &fc.inter_mode {
        no_update(writer, row);
    }
    no_update(writer, &fc.intra_inter);
    for row in &fc.single_ref {
        no_update(writer, row);
    }
    for row in &fc.y_mode {
        no_update(writer, row);
    }
    for row in &fc.partition {
        no_update(writer, row);
    }
    no_update(writer, &fc.mv.joints);
    for comp in &fc.mv.comp {
        no_update(writer, &[comp.sign]);
        no_update(writer, &comp.classes);
        no_update(writer, &[comp.class0]);
        no_update(writer, &comp.bits);
    }
    for comp in &fc.mv.comp {
        for row in &comp.class0_fp {
            no_update(writer, row);
        }
        no_update(writer, &comp.fp);
    }
    for comp in &fc.mv.comp {
        no_update(writer, &[comp.class0_hp]);
        no_update(writer, &[comp.hp]);
    }
}

fn no_update(writer: &mut BoolWriter, probs: &[u8]) {
    for _ in probs {
        writer.write(0, 252);
    }
}

/// Parses one assembled frame. Used by the encoder tests and kept here so a
/// header mistake fails before reconstruction.
#[cfg(test)]
fn parsed_header(bytes: &[u8]) -> Result<header::Header, CodecError> {
    Ok(header::parse(&header::Persist::new(), bytes)?.header)
}

#[cfg(test)]
mod tests {
    use super::Vp9Encoder;
    use crate::decode::Decoder;
    use cutlass_encode::{
        CodecId, CodecParams, PixelFormat, Plane, VideoEncoder, VideoFrame, VideoInput,
    };

    fn solid(width: u32, height: u32, y: u8, u: u8, v: u8) -> VideoFrame {
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let luma = usize::try_from(width * height).expect("luma");
        let chroma = usize::try_from(chroma_w * chroma_h).expect("chroma");
        VideoFrame::yuv420p(
            width,
            height,
            Plane::tight(width, height, vec![y; luma]).expect("y"),
            Plane::tight(chroma_w, chroma_h, vec![u; chroma]).expect("u"),
            Plane::tight(chroma_w, chroma_h, vec![v; chroma]).expect("v"),
        )
        .expect("frame")
    }

    fn input(frame: VideoFrame) -> VideoInput {
        VideoInput {
            frame,
            pts: 0,
            duration: 1,
            timescale: 30,
        }
    }

    #[test]
    fn lossless_constant_matches_the_source() {
        let frame = solid(16, 16, 40, 90, 140);
        let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
        let packets = encoder.push(&input(frame.clone())).expect("encode");
        assert_eq!(packets.len(), 1);
        assert!(packets[0].keyframe);
        let header = super::parsed_header(&packets[0].data).expect("header");
        assert!(header.key_frame);
        assert!(header.lossless);
        assert!(header.error_resilient);
        assert_eq!(header.width, 16);
        assert_eq!(header.height, 16);
        let picture = Decoder::new()
            .decode(&packets[0].data)
            .expect("decode")
            .expect("shown");
        assert_eq!(picture.y, frame.y.samples);
        assert_eq!(picture.u, frame.u.samples);
        assert_eq!(picture.v, frame.v.samples);
    }

    #[test]
    fn lossless_ramp_and_odd_blocks_match_the_source() {
        let mut frames = Vec::new();
        frames.push(ramp(8, 8));
        frames.push(ramp(16, 16));
        frames.push(ramp(24, 16));
        frames.push(solid(64, 64, 200, 20, 80));
        for frame in frames {
            let mut encoder =
                Vp9Encoder::with_quantizer(frame.width, frame.height, 0).expect("encoder");
            let packets = encoder.push(&input(frame.clone())).expect("encode");
            let picture = Decoder::new()
                .decode(&packets[0].data)
                .expect("decode")
                .expect("shown");
            assert_eq!(picture.y, frame.y.samples);
            assert_eq!(picture.u, frame.u.samples);
            assert_eq!(picture.v, frame.v.samples);
        }
    }

    fn ramp(width: u32, height: u32) -> VideoFrame {
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let mut y = Vec::new();
        let mut u = Vec::new();
        let mut v = Vec::new();
        for row in 0..height {
            for col in 0..width {
                y.push(u8::try_from((row * 3 + col * 5) % 251).unwrap_or(0));
            }
        }
        for row in 0..chroma_h {
            for col in 0..chroma_w {
                u.push(u8::try_from((row * 9 + col) % 200).unwrap_or(0));
                v.push(u8::try_from((col * 7 + 30) % 180).unwrap_or(0));
            }
        }
        VideoFrame::yuv420p(
            width,
            height,
            Plane::tight(width, height, y).expect("y"),
            Plane::tight(chroma_w, chroma_h, u).expect("u"),
            Plane::tight(chroma_w, chroma_h, v).expect("v"),
        )
        .expect("frame")
    }

    #[test]
    fn factory_bitrate_zero_is_quantizer_32() {
        let params = CodecParams::video(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
        let encoder = Vp9Encoder::open(&params).expect("open");
        assert_eq!(encoder.qindex(), 32);
    }
}
