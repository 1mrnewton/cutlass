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

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{CodecError, CodecId, CodecParams, Packet, PixelFormat, VideoFrame};

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
    use crate::bits::BitWriter;
    use crate::decode::{Decoder, Vp9Decoder};
    use crate::register_decoders;
    use crate::register_encoders;
    use cutlass_decode::video::VideoDecoder;
    use cutlass_decode::{CodecError, VideoDecodeFactory};
    use cutlass_encode::video::{VideoEncoder, VideoInput};
    use cutlass_encode::{
        CodecId, CodecParams, Packet, PixelFormat, Plane, VideoEncodeFactory, VideoFrame,
        request_video_cancel,
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
    fn lossy_keyframes_meet_the_psnr_gates() {
        assert_psnr(solid(64, 64, 64, 128, 128), 32, 45.0, "flat");
        assert_psnr(ramp(64, 64), 32, 38.0, "ramp");
        assert_psnr(checker(64, 64), 32, 28.0, "checker");
    }

    fn assert_psnr(frame: VideoFrame, qindex: u8, floor: f64, name: &str) {
        let mut encoder =
            Vp9Encoder::with_quantizer(frame.width, frame.height, qindex).expect("encoder");
        let packets = encoder.push(&input(frame.clone())).expect("encode");
        let picture = Decoder::new()
            .decode(&packets[0].data)
            .expect("decode")
            .expect("shown");
        let score = psnr(&frame.y.samples, &picture.y);
        assert!(score >= floor, "{name} psnr {score} floor {floor}");
    }

    fn checker(width: u32, height: u32) -> VideoFrame {
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let mut y = Vec::new();
        let mut u = Vec::new();
        let mut v = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let on = ((row / 4) + (col / 4)).is_multiple_of(2);
                y.push(if on { 220 } else { 30 });
            }
        }
        for row in 0..chroma_h {
            for col in 0..chroma_w {
                let on = ((row / 2) + (col / 2)).is_multiple_of(2);
                u.push(if on { 160 } else { 80 });
                v.push(if on { 140 } else { 100 });
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

    fn psnr(src: &[u8], decoded: &[u8]) -> f64 {
        let mut error = 0.0f64;
        for (left, right) in src.iter().zip(decoded) {
            let delta = f64::from(i32::from(*left) - i32::from(*right));
            error += delta * delta;
        }
        if error == 0.0 {
            return 100.0;
        }
        let count = f64::from(u32::try_from(src.len()).unwrap_or(u32::MAX));
        10.0 * (255.0 * 255.0 * count / error).log10()
    }

    #[test]
    fn lossless_motion_matches_the_source() {
        let mut encoder = Vp9Encoder::with_quantizer(32, 16, 0).expect("encoder");
        let mut decoder = Decoder::new();
        for index in 0..8 {
            let frame = moving(32, 16, index);
            let packets = encoder.push(&input(frame.clone())).expect("encode");
            assert_eq!(packets[0].keyframe, index == 0);
            let picture = decoder
                .decode(&packets[0].data)
                .expect("decode")
                .expect("shown");
            assert_eq!(
                first_mismatch(&picture.y, &frame.y.samples),
                None,
                "y {index}"
            );
            assert_eq!(
                first_mismatch(&picture.u, &frame.u.samples),
                None,
                "u {index}"
            );
            assert_eq!(
                first_mismatch(&picture.v, &frame.v.samples),
                None,
                "v {index}"
            );
        }
    }

    #[test]
    fn lossy_motion_meets_the_psnr_gate() {
        let mut encoder = Vp9Encoder::with_quantizer(64, 64, 32).expect("encoder");
        let mut decoder = Decoder::new();
        let mut scores = Vec::new();
        for index in 0..16 {
            let frame = moving(64, 64, index);
            let packets = encoder.push(&input(frame.clone())).expect("encode");
            let picture = decoder
                .decode(&packets[0].data)
                .expect("decode")
                .expect("shown");
            scores.push(psnr(&frame.y.samples, &picture.y));
        }
        let mean = scores.iter().sum::<f64>() / f64::from(u32::try_from(scores.len()).unwrap_or(1));
        assert!(mean >= 32.0, "mean psnr {mean} scores {scores:?}");
    }

    fn moving(width: u32, height: u32, shift: u32) -> VideoFrame {
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let mut y = Vec::new();
        let mut u = Vec::new();
        let mut v = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let src = i32::try_from(col).unwrap_or(0) - i32::try_from(shift).unwrap_or(0);
                let sample = if src < 0 {
                    16
                } else {
                    u8::try_from((u32::try_from(src).unwrap_or(0) * 9 + row * 3) % 220 + 20)
                        .unwrap_or(20)
                };
                y.push(sample);
            }
        }
        let chroma_shift = shift / 2;
        for row in 0..chroma_h {
            for col in 0..chroma_w {
                let src =
                    i32::try_from(col).unwrap_or(0) - i32::try_from(chroma_shift).unwrap_or(0);
                let sample = if src < 0 {
                    90
                } else {
                    u8::try_from((u32::try_from(src).unwrap_or(0) * 5 + row) % 80 + 80)
                        .unwrap_or(80)
                };
                u.push(sample);
                v.push(sample.saturating_add(20));
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

    fn first_mismatch(got: &[u8], want: &[u8]) -> Option<usize> {
        got.iter().zip(want).position(|(left, right)| left != right)
    }

    #[test]
    fn thirty_frames_stay_near_500_kbps() {
        let mut params = CodecParams::with_format(CodecId::Vp9, 160, 96, PixelFormat::Yuv420p);
        params.bitrate = 500_000;
        let mut encoder = Vp9Encoder::open(&params).expect("encoder");
        let mut total = 0usize;
        for index in 0..30u32 {
            let frame = moving(160, 96, index);
            let packets = encoder
                .push(&VideoInput {
                    frame,
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("encode");
            total += packets[0].data.len();
        }
        let target = 500_000.0 / 8.0;
        let ratio = total as f64 / target;
        assert!(
            (0.70..=1.25).contains(&ratio),
            "bytes {total} ratio {ratio}"
        );
    }

    #[test]
    fn factory_bitrate_zero_is_quantizer_32() {
        let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
        let encoder = Vp9Encoder::open(&params).expect("open");
        assert_eq!(encoder.qindex(), 32);
    }

    #[test]
    fn lifecycle_keeps_references_until_reset_or_cancel() {
        let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
        assert!(encoder.drain().expect("drain").is_empty());
        let first = encoder
            .push(&input(solid(16, 16, 10, 20, 30)))
            .expect("key");
        assert!(first[0].keyframe);
        let second = encoder.push(&input(moving(16, 16, 1))).expect("inter");
        assert!(!second[0].keyframe);
        let bad = encoder.push(&input(solid(32, 16, 1, 2, 3)));
        assert!(matches!(
            bad,
            Err(CodecError::InvalidApi {
                what: "picture size"
            })
        ));
        let third = encoder
            .push(&input(moving(16, 16, 2)))
            .expect("still inter");
        assert!(!third[0].keyframe);
        encoder.reset();
        let again = encoder
            .push(&input(solid(16, 16, 10, 20, 30)))
            .expect("reset key");
        assert!(again[0].keyframe);
        assert!(encoder.drain().expect("drain again").is_empty());
        request_video_cancel(&encoder.cancel_flag());
        let cancelled = encoder.push(&input(solid(16, 16, 10, 20, 30)));
        assert!(matches!(cancelled, Err(CodecError::Cancelled)));
    }

    #[test]
    fn open_rejects_the_sizes_and_formats_outside_profile_0() {
        assert!(matches!(
            Vp9Encoder::with_quantizer(0, 8, 0),
            Err(CodecError::InvalidApi { .. })
        ));
        assert!(matches!(
            Vp9Encoder::with_quantizer(12, 8, 0),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            Vp9Encoder::with_quantizer(4097, 8, 0),
            Err(CodecError::Limit { .. })
        ));
        let rgba = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Rgba8);
        assert!(matches!(
            Vp9Encoder::open(&rgba),
            Err(CodecError::Unsupported { .. })
        ));
        let huge = CodecParams::with_format(CodecId::Vp9, 4097, 8, PixelFormat::Yuv420p);
        assert!(matches!(
            Vp9Decoder::open(&huge),
            Err(CodecError::Limit { .. })
        ));
        let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
        let mut decoder = Vp9Decoder::open(&params).expect("decoder");
        let empty = decoder.push(&Packet::video(Vec::new(), 0, 1, 30, true));
        assert!(matches!(empty, Err(CodecError::InvalidApi { .. })));
        for marker in [0xa0, 0x90, 0xb0] {
            let err = decoder
                .push(&Packet::video(vec![marker], 0, 1, 30, true))
                .expect_err("profile");
            assert!(err.to_string().contains("profile"), "{marker:#x} {err}");
        }
        let err = match Decoder::new().decode(&oversized_edge()) {
            Err(err) => err,
            Ok(_) => panic!("oversized edge was accepted"),
        };
        assert!(matches!(err, CodecError::Limit { .. }), "{err}");
        assert!(decoder.drain().expect("drain").is_empty());
        decoder.reset();
    }

    #[test]
    fn show_existing_and_a_superframe_round_trip() {
        let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
        let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
        let key = encoder
            .push(&input(solid(16, 16, 40, 90, 140)))
            .expect("key")
            .remove(0);
        let inter = encoder
            .push(&input(moving(16, 16, 1)))
            .expect("inter")
            .remove(0);
        let mut decoder = Vp9Decoder::open(&params).expect("decoder");
        let shown = decoder.push(&key).expect("key picture");
        let repeated = decoder
            .push(&Packet::video(vec![0x88], 9, 1, 30, false))
            .expect("show existing");
        assert_eq!(repeated.len(), 1);
        assert_eq!(repeated[0].pts, 9);
        assert_eq!(repeated[0].frame.y.samples, shown[0].frame.y.samples);
        assert_eq!(repeated[0].frame.u.samples, shown[0].frame.u.samples);
        assert_eq!(repeated[0].frame.v.samples, shown[0].frame.v.samples);

        let mut fresh = Vp9Decoder::open(&params).expect("fresh");
        let both = fresh
            .push(&Packet::video(
                superframe(&[&key.data, &inter.data]),
                0,
                1,
                30,
                true,
            ))
            .expect("superframe");
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].frame.y.samples, shown[0].frame.y.samples);
        let mut separate = Decoder::new();
        separate.decode(&key.data).expect("key").expect("shown");
        let second = separate.decode(&inter.data).expect("inter").expect("shown");
        assert_eq!(both[1].frame.y.samples, second.y);

        let broken = superframe(&[&key.data, &[0x87, 0x00]]);
        let err = fresh.push(&Packet::video(broken, 1, 1, 30, true));
        assert!(err.is_err());
        let still = fresh
            .push(&Packet::video(vec![0x88], 4, 1, 30, false))
            .expect("refs kept");
        assert_eq!(still[0].frame.y.samples, second.y);

        let Some(ffmpeg) = ffmpeg_bin() else {
            require_or_skip("ffmpeg");
            return;
        };
        let plane = 16 * 16 * 3 / 2;
        let repeated_ivf = wrap_ivf(16, 16, &[key.data.clone(), vec![0x88]]);
        let repeated_yuv = ffmpeg_yuv(&ffmpeg, "vp9", &repeated_ivf);
        assert_eq!(repeated_yuv.len(), plane * 2);
        assert_eq!(&repeated_yuv[..plane], &repeated_yuv[plane..]);
        let packed = superframe(&[&key.data, &inter.data]);
        let wrapped = wrap_ivf(16, 16, &[packed]);
        let native = ffmpeg_yuv(&ffmpeg, "vp9", &wrapped);
        let mut expect = Vec::new();
        expect.extend_from_slice(&shown[0].frame.y.samples);
        expect.extend_from_slice(&shown[0].frame.u.samples);
        expect.extend_from_slice(&shown[0].frame.v.samples);
        expect.extend_from_slice(&second.y);
        expect.extend_from_slice(&second.u);
        expect.extend_from_slice(&second.v);
        assert_eq!(native, expect);
        // ffmpeg's libvpx-vp9 wrapper returns the last picture of one superframe packet.
        let libvpx = ffmpeg_yuv(&ffmpeg, "libvpx-vp9", &wrapped);
        assert_eq!(libvpx, expect[plane..]);
    }

    fn superframe(frames: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for frame in frames {
            out.extend_from_slice(frame);
        }
        let count = u8::try_from(frames.len() - 1).expect("count");
        let marker = 0xc8 | count;
        out.push(marker);
        for frame in frames {
            let size = u16::try_from(frame.len()).expect("size");
            out.extend_from_slice(&size.to_le_bytes());
        }
        out.push(marker);
        out
    }

    #[test]
    fn factories_open_the_profile_0_codec() {
        let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
        let mut encoders = VideoEncodeFactory::new();
        register_encoders(&mut encoders);
        let mut encoder = encoders.open(&params).expect("encoder");
        let mut decoders = VideoDecodeFactory::new();
        register_decoders(&mut decoders);
        let mut decoder = decoders.open(&params).expect("decoder");
        let packets = encoder
            .push(&input(solid(16, 16, 40, 90, 140)))
            .expect("push");
        let shown = decoder.push(&packets[0]).expect("decode");
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].frame.y.samples, vec![40; 16 * 16]);
        assert_eq!(encoder.codec(), CodecId::Vp9);
        assert_eq!(decoder.codec(), CodecId::Vp9);
    }

    #[test]
    fn independent_decoders_match_our_yuv() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            require_or_skip("ffmpeg");
            return;
        };
        check_oracles(&ffmpeg, 0);
        check_oracles(&ffmpeg, 32);
    }

    fn check_oracles(ffmpeg: &std::path::Path, qindex: u8) {
        let mut encoder = Vp9Encoder::with_quantizer(32, 16, qindex).expect("encoder");
        let mut decoder = Decoder::new();
        let mut frames = Vec::new();
        let mut expect = Vec::new();
        for index in 0..4 {
            let frame = moving(32, 16, index);
            let packets = encoder.push(&input(frame)).expect("encode");
            let picture = decoder
                .decode(&packets[0].data)
                .expect("ours")
                .expect("shown");
            expect.extend_from_slice(&picture.y);
            expect.extend_from_slice(&picture.u);
            expect.extend_from_slice(&picture.v);
            frames.push(packets[0].data.clone());
        }
        let wrapped = wrap_ivf(32, 16, &frames);
        for name in ["vp9", "libvpx-vp9"] {
            let yuv = ffmpeg_yuv(ffmpeg, name, &wrapped);
            assert_eq!(
                first_mismatch(&yuv, &expect),
                None,
                "{name} q {qindex} mismatch"
            );
        }
        let Some(gst) = gst_bin() else {
            require_or_skip("gstreamer");
            return;
        };
        let yuv = gstreamer_yuv(&gst, &wrapped);
        assert_eq!(first_mismatch(&yuv, &expect), None, "gstreamer q {qindex}");
    }

    fn oversized_edge() -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.write(2, 2);
        bits.write(0, 2);
        bits.write_bit(false);
        bits.write_bit(false);
        bits.write_bit(true);
        bits.write_bit(true);
        bits.write(0x49, 8);
        bits.write(0x83, 8);
        bits.write(0x42, 8);
        bits.write(1, 3);
        bits.write_bit(false);
        bits.write(4096, 16);
        bits.write(7, 16);
        bits.finish()
    }

    fn wrap_ivf(width: u32, height: u32, frames: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"DKIF");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"VP90");
        out.extend_from_slice(&u16::try_from(width).expect("width").to_le_bytes());
        out.extend_from_slice(&u16::try_from(height).expect("height").to_le_bytes());
        out.extend_from_slice(&30u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&u32::try_from(frames.len()).expect("count").to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for (index, frame) in frames.iter().enumerate() {
            out.extend_from_slice(&u32::try_from(frame.len()).expect("size").to_le_bytes());
            let pts = u64::try_from(index).expect("pts");
            out.extend_from_slice(&pts.to_le_bytes());
            out.extend_from_slice(frame);
        }
        out
    }

    fn require_or_skip(tool: &str) {
        if std::env::var_os("VP9_REQUIRE_ORACLES").is_some() {
            panic!("{tool} is required");
        }
        eprintln!("skipped {tool} oracle: {tool} was not found");
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        for candidate in [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ] {
            let path = std::path::PathBuf::from(candidate);
            if std::process::Command::new(&path)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(path);
            }
        }
        None
    }

    fn gst_bin() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from("gst-launch-1.0");
        std::process::Command::new(&path)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
            .then_some(path)
    }

    fn unique_dir() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    fn ffmpeg_yuv(ffmpeg: &std::path::Path, decoder: &str, ivf: &[u8]) -> Vec<u8> {
        let dir = std::env::temp_dir().join(format!(
            "cutlass-vp9-ff-{}-{}-{}",
            std::process::id(),
            decoder,
            unique_dir()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let input = dir.join("in.ivf");
        let output = dir.join("out.yuv");
        std::fs::write(&input, ivf).expect("ivf");
        let status = std::process::Command::new(ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-c:v",
                decoder,
                "-i",
            ])
            .arg(&input)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&output)
            .output()
            .expect("ffmpeg");
        assert!(
            status.status.success(),
            "{decoder} {}",
            String::from_utf8_lossy(&status.stderr)
        );
        let bytes = std::fs::read(&output).expect("yuv");
        let _ = std::fs::remove_dir_all(&dir);
        bytes
    }

    fn gstreamer_yuv(gst: &std::path::Path, ivf: &[u8]) -> Vec<u8> {
        let dir = std::env::temp_dir().join(format!(
            "cutlass-vp9-gst-{}-{}",
            std::process::id(),
            unique_dir()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let input = dir.join("in.ivf");
        let output = dir.join("out.yuv");
        std::fs::write(&input, ivf).expect("ivf");
        let in_loc = format!("location={}", input.display());
        let out_loc = format!("location={}", output.display());
        let status = std::process::Command::new(gst)
            .args([
                "-q",
                "filesrc",
                &in_loc,
                "!",
                "ivfparse",
                "!",
                "vp9dec",
                "!",
                "video/x-raw,format=I420",
                "!",
                "filesink",
                &out_loc,
            ])
            .output()
            .expect("gst");
        assert!(
            status.status.success(),
            "gstreamer {}",
            String::from_utf8_lossy(&status.stderr)
        );
        let bytes = std::fs::read(&output).expect("yuv");
        let _ = std::fs::remove_dir_all(&dir);
        bytes
    }

    #[test]
    fn release_320x240_meets_realtime_decode() {
        if cfg!(debug_assertions) {
            eprintln!("320x240 timing is measured in release");
            return;
        }
        let frames = 30u32;
        let mut encoder = Vp9Encoder::with_quantizer(320, 240, 32).expect("encoder");
        let mut packets = Vec::new();
        let encode_started = std::time::Instant::now();
        for index in 0..frames {
            let packet = encoder
                .push(&VideoInput {
                    frame: moving(320, 240, index),
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("encode");
            packets.push(packet[0].data.clone());
        }
        let encode_elapsed = encode_started.elapsed().as_secs_f64();
        let mut decoder = Decoder::new();
        let decode_started = std::time::Instant::now();
        for packet in &packets {
            decoder.decode(packet).expect("decode").expect("shown");
        }
        let decode_elapsed = decode_started.elapsed().as_secs_f64();
        let decode_fps = f64::from(frames) / decode_elapsed;
        let encode_fps = f64::from(frames) / encode_elapsed;
        let hwm = resident_kib();
        eprintln!(
            "vp9 320x240 q32 frames {frames} decode_fps {decode_fps:.1} encode_fps {encode_fps:.1} hwm_kib {hwm}"
        );
        assert!(
            decode_fps >= 30.0,
            "decode {decode_fps:.1} fps, need 30 for 1x realtime"
        );
    }

    fn resident_kib() -> u64 {
        let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("VmHWM:") else {
                continue;
            };
            let n = rest.split_whitespace().next().unwrap_or("0");
            return n.parse().unwrap_or(0);
        }
        0
    }
}
