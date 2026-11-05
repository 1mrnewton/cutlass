//! Round trips through this crate's own encoder and decoder.

use std::sync::atomic::AtomicBool;

use cutlass_decode::{CodecId, CodecParams, PixelFormat, Plane, VideoFrame};
use cutlass_pool::Pool;

use crate::{Chroma, Interlace, Profile, ProresDecoder, ProresEncoder, decode_frame};

pub(crate) fn plane16(width: u32, height: u32, f: impl Fn(u32, u32) -> u16) -> Plane {
    let mut bytes = Vec::with_capacity((width * height * 2) as usize);
    for y in 0..height {
        for x in 0..width {
            bytes.extend_from_slice(&f(x, y).to_le_bytes());
        }
    }
    Plane::bytes(width, height, 2, bytes).expect("plane")
}

/// A busy 10-bit picture: gradients, a hash, and a hard edge.
pub(crate) fn busy(format: PixelFormat, width: u32, height: u32) -> VideoFrame {
    let (chroma_w, chroma_h) = format.chroma_wh(width, height).expect("planar");
    let (depth, scale) = match format {
        PixelFormat::Yuv422p12 | PixelFormat::Yuv444p12 => (12, 4),
        _ => (10, 1),
    };
    let max = (1u32 << depth) - 1;
    let luma = |x: u32, y: u32| {
        let hash = (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503)) >> 24;
        let edge = if x > width / 2 { 300 } else { 0 };
        ((64 + x * 3 + y * 2 + hash + edge) * scale).min(max) as u16
    };
    let cb = |x: u32, y: u32| ((300 + x * 5 + y) * scale).min(max) as u16;
    let cr = |x: u32, y: u32| ((700 - x - y * 3 % 200) * scale).min(max) as u16;
    VideoFrame::planar(
        format,
        width,
        height,
        plane16(width, height, luma),
        plane16(chroma_w, chroma_h, cb),
        plane16(chroma_w, chroma_h, cr),
    )
    .expect("frame")
}

pub(crate) fn params(profile: Profile) -> CodecParams {
    let mut params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    params.extra = profile.fourcc().to_vec();
    params
}

fn psnr(a: &Plane, b: &Plane, peak: f64) -> f64 {
    let mut sq = 0f64;
    let mut n = 0f64;
    for (pa, pb) in a.samples.chunks(2).zip(b.samples.chunks(2)) {
        let va = f64::from(u16::from_le_bytes([pa[0], pa[1]]));
        let vb = f64::from(u16::from_le_bytes([pb[0], pb[1]]));
        sq += (va - vb) * (va - vb);
        n += 1.0;
    }
    if sq == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (peak * peak / (sq / n)).log10()
}

#[test]
fn every_profile_round_trips_through_this_decoder() {
    for profile in Profile::ALL {
        let format = if profile.is_444() {
            PixelFormat::Yuv444p12
        } else {
            PixelFormat::Yuv422p10
        };
        let source = busy(format, 96, 64);
        let mut encoder = ProresEncoder::open(&params(profile)).expect("open");
        let bytes = encoder.encode(&source).expect("encode");
        let mut decoder = ProresDecoder::open(&params(profile)).expect("open");
        let decoded = decoder.decode(&bytes).expect("decode");
        assert_eq!(decoded.header.width, 96);
        assert_eq!(decoded.frame.format, format);
        let peak = if profile.is_444() { 4095.0 } else { 1023.0 };
        let y = psnr(&decoded.frame.y, &source.y, peak);
        let floor = match profile {
            Profile::Proxy => 30.0,
            Profile::Lt => 36.0,
            _ => 42.0,
        };
        assert!(y > floor, "{profile:?} Y PSNR {y}");
    }
}

#[test]
fn odd_sizes_and_alpha_survive() {
    let mut source = busy(PixelFormat::Yuv444p12, 33, 17);
    let alpha = plane16(33, 17, |x, y| ((x * 1987 + y * 7919) % 65_536) as u16);
    source = source.with_alpha(16, alpha.clone()).expect("alpha");
    let mut encoder = ProresEncoder::open(&params(Profile::P4444)).expect("open");
    let bytes = encoder.encode(&source).expect("encode");
    let decoded = ProresDecoder::open(&params(Profile::P4444))
        .expect("open")
        .decode(&bytes)
        .expect("decode");
    let got = decoded.frame.alpha.expect("alpha");
    assert_eq!(got.bit_depth, 16);
    assert_eq!(got.plane, alpha);
    assert!(psnr(&decoded.frame.y, &source.y, 4095.0) > 40.0);
}

#[test]
fn rgba_comes_back_as_its_bt709_colors() {
    let (w, h) = (48u32, 32u32);
    let mut rgba = Vec::new();
    for _ in 0..w * h {
        rgba.extend_from_slice(&[255, 0, 0, 255]);
    }
    let mut encoder = ProresEncoder::open(&params(Profile::Hq)).expect("open");
    let bytes = encoder.encode_rgba(w, h, &rgba).expect("encode");
    let decoded = ProresDecoder::open(&params(Profile::Hq))
        .expect("open")
        .decode(&bytes)
        .expect("decode");
    let at = |plane: &Plane| u16::from_le_bytes([plane.samples[0], plane.samples[1]]);
    // Red in limited-range BT.709, 10-bit: Y 250, Cb 409, Cr 960.
    assert!(
        at(&decoded.frame.y).abs_diff(250) <= 2,
        "{}",
        at(&decoded.frame.y)
    );
    assert!(
        at(&decoded.frame.u).abs_diff(409) <= 2,
        "{}",
        at(&decoded.frame.u)
    );
    assert!(
        at(&decoded.frame.v).abs_diff(960) <= 2,
        "{}",
        at(&decoded.frame.v)
    );
    assert_eq!(decoded.header.primaries, 1);
    assert_eq!(decoded.header.transfer, 1);
    assert_eq!(decoded.header.matrix, 1);
}

#[test]
fn one_worker_and_many_give_the_same_bytes() {
    let source = busy(PixelFormat::Yuv422p10, 200, 120);
    let one = ProresEncoder::open(&params(Profile::Hq))
        .expect("open")
        .with_threads(1)
        .encode(&source)
        .expect("one");
    let many = ProresEncoder::open(&params(Profile::Hq))
        .expect("open")
        .with_threads(8)
        .encode(&source)
        .expect("many");
    assert_eq!(one, many);
    let cancel = AtomicBool::new(false);
    let a = decode_frame(&one, 10, &Pool::new(1), &cancel).expect("one");
    let b = decode_frame(&one, 10, &Pool::new(6), &cancel).expect("many");
    assert_eq!(a, b);
    assert_eq!(a.header.chroma, Chroma::Yuv422);
    assert_eq!(a.header.interlace, Interlace::Progressive);
}
