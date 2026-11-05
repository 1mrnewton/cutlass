//! The encoder's packets, decoded here and by ffmpeg's `vp8` and libvpx,
//! must equal the encoder's own reconstruction.

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{Plane, VideoFrame};

use crate::decode::Frame;
use crate::prove::{Ivf, decode_shown, ffmpeg_bin, ffmpeg_yuv, first_difference, temp_dir, write_ivf};
use crate::{Vp8Encoder, Vp8Options};

/// A moving, textured picture with fine detail and smooth areas.
pub(crate) fn synthetic(width: u32, height: u32, t: u32) -> VideoFrame {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let t = t as f32;
    let hash = |x: usize, y: usize| -> f32 {
        let mut v = (x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663);
        v ^= v >> 13;
        v = v.wrapping_mul(0x5bd1_e995);
        ((v >> 24) & 15) as f32 - 7.5
    };
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let (x, yy) = (c as f32 + 2.0 * t, r as f32 - 0.75 * t);
            let mut v = 128.0 + 50.0 * (x * 0.15).sin() + 40.0 * (yy * 0.11).cos();
            if (c + r) % 64 < 32 {
                v += hash(c + 2 * t as usize, r);
            }
            if c > w / 2 && r > h / 2 {
                v = 60.0 + (c as f32 * 0.5);
            }
            y[r * w + c] = v.clamp(0.0, 255.0) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = (128.0 + 40.0 * ((c as f32 + t) * 0.05).sin()) as u8;
            v[r * cw + c] = (128.0 + 30.0 * ((r as f32 - t) * 0.07).cos()) as u8;
        }
    }
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, y).expect("y"),
        Plane::tight(width.div_ceil(2), height.div_ceil(2), u).expect("u"),
        Plane::tight(width.div_ceil(2), height.div_ceil(2), v).expect("v"),
    )
    .expect("frame")
}

/// The display-size part of a reconstruction, packed as yuv420p.
pub(crate) fn crop(frame: &Frame, width: u32, height: u32, out: &mut Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    for r in 0..h {
        out.extend_from_slice(&frame.y[r * frame.mbw * 16..r * frame.mbw * 16 + w]);
    }
    for plane in [&frame.u, &frame.v] {
        for r in 0..ch {
            out.extend_from_slice(&plane[r * frame.mbw * 8..r * frame.mbw * 8 + cw]);
        }
    }
}

/// Encodes `count` synthetic frames; returns the packets and the packed
/// reconstruction.
pub(crate) fn encode_synthetic(
    width: u32,
    height: u32,
    count: u32,
    opts: Vp8Options,
) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut encoder = Vp8Encoder::with_options(width, height, opts).expect("open");
    let mut packets = Vec::new();
    let mut recon = Vec::new();
    for t in 0..count {
        let input = VideoInput {
            frame: synthetic(width, height, t),
            pts: i64::from(t),
            duration: 1,
            timescale: 30,
        };
        let out = encoder.push(&input).expect("push");
        assert_eq!(out.len(), 1);
        packets.push(out[0].data.clone());
        crop(encoder.reconstruction().expect("recon"), width, height, &mut recon);
    }
    (packets, recon)
}

fn check(name: &str, width: u32, height: u32, count: u32, opts: Vp8Options) {
    let (packets, recon) = encode_synthetic(width, height, count, opts);
    let stream = Ivf {
        width,
        height,
        frames: packets,
    };
    let ours = decode_shown(&stream);
    if let Some(diff) = first_difference(&ours, &recon, width, height) {
        panic!("{name}: our decoder differs from the reconstruction at {diff}");
    }
    assert_eq!(ours.len(), recon.len(), "{name}: bytes");
    let bytes: usize = stream.frames.iter().map(Vec::len).sum();
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("{name}: {bytes} bytes; ffmpeg missing, oracles skipped");
        return;
    };
    let dir = temp_dir(name);
    let ivf = dir.join("ours.ivf");
    write_ivf(&ivf, width, height, &stream.frames);
    for decoder in ["vp8", "libvpx"] {
        let theirs = ffmpeg_yuv(&ffmpeg, &ivf, decoder);
        if let Some(diff) = first_difference(&theirs, &recon, width, height) {
            panic!("{name}: {decoder} differs at {diff}");
        }
        assert_eq!(theirs.len(), recon.len(), "{name}: {decoder} bytes");
    }
    eprintln!("{name}: {count} frames, {bytes} bytes, identical in all three decoders");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fixed_quantizers_decode_identically() {
    for q in [0u8, 10, 40, 80, 127] {
        let opts = Vp8Options {
            quantizer: q,
            ..Vp8Options::default()
        };
        check(&format!("q{q}"), 96, 64, 6, opts);
    }
}

#[test]
fn odd_sizes_decode_identically() {
    for (w, h) in [(1, 1), (17, 9), (33, 47), (202, 118)] {
        check(&format!("size{w}x{h}"), w, h, 4, Vp8Options::default());
    }
}

#[test]
fn golden_keys_and_partitions_decode_identically() {
    let opts = Vp8Options {
        keyframe_interval: 7,
        golden_interval: 3,
        partitions: 4,
        ..Vp8Options::default()
    };
    check("golden", 128, 96, 16, opts);
    for speed in [0, 2] {
        let opts = Vp8Options {
            speed,
            partitions: 8,
            ..Vp8Options::default()
        };
        check(&format!("speed{speed}"), 160, 128, 6, opts);
    }
}

#[test]
fn bitrate_mode_decodes_identically() {
    let opts = Vp8Options {
        bitrate: 300_000,
        ..Vp8Options::default()
    };
    check("rate", 176, 144, 20, opts);
}
