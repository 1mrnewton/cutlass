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

/// Raw yuv420p frames of a local file scaled to `width`×`height`.
pub(crate) fn local_yuv(name: &str, width: u32, height: u32, frames: u32) -> Option<Vec<u8>> {
    let source = crate::prove::local_root()?.join(name);
    if !source.is_file() {
        return None;
    }
    let ffmpeg = ffmpeg_bin()?;
    let dir = temp_dir("source");
    let out = dir.join("src.yuv");
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(&source)
        .args([
            "-an",
            "-vf",
            &format!("scale={width}:{height}:flags=bicubic"),
            "-frames:v",
            &frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .output()
        .expect("ffmpeg");
    assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stderr));
    let data = std::fs::read(&out).expect("yuv");
    let _ = std::fs::remove_dir_all(&dir);
    Some(data)
}

pub(crate) fn frame_of(yuv: &[u8], width: u32, height: u32, index: usize) -> VideoFrame {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let size = w * h + 2 * cw * ch;
    let f = &yuv[index * size..(index + 1) * size];
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, f[..w * h].to_vec()).expect("y"),
        Plane::tight(width.div_ceil(2), height.div_ceil(2), f[w * h..w * h + cw * ch].to_vec()).expect("u"),
        Plane::tight(width.div_ceil(2), height.div_ceil(2), f[w * h + cw * ch..].to_vec()).expect("v"),
    )
    .expect("frame")
}

/// PSNR of the luma planes over all frames, from the total squared error.
pub(crate) fn psnr_y(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let size = w * h + 2 * cw * ch;
    let frames = a.len().min(b.len()) / size;
    let mut sse = 0f64;
    for f in 0..frames {
        for i in 0..w * h {
            let d = f64::from(a[f * size + i]) - f64::from(b[f * size + i]);
            sse += d * d;
        }
    }
    let mse = sse / (frames * w * h) as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-10)).log10()
}

/// Our encoder and libvpx (good, cpu-used 4, one thread) at the same
/// bitrate on one source. Returns (ours psnr, ours bytes, ours seconds,
/// libvpx psnr, libvpx bytes).
pub(crate) fn compare_with_libvpx(
    name: &str,
    width: u32,
    height: u32,
    frames: u32,
    bitrate: u32,
    opts: Vp8Options,
) -> Option<(f64, usize, f64, f64, usize)> {
    let source = local_yuv(name, width, height, frames)?;
    let ffmpeg = ffmpeg_bin()?;
    let dir = temp_dir("compare");
    let mut encoder = Vp8Encoder::with_options(width, height, Vp8Options { bitrate, ..opts }).expect("open");
    let mut packets = Vec::new();
    let mut recon = Vec::new();
    let count = frames as usize;
    let inputs: Vec<VideoFrame> = (0..count).map(|i| frame_of(&source, width, height, i)).collect();
    let start = std::time::Instant::now();
    for (i, frame) in inputs.into_iter().enumerate() {
        let input = VideoInput {
            frame,
            pts: i as i64,
            duration: 1,
            timescale: 30,
        };
        let out = encoder.push(&input).expect("push");
        packets.extend(out.into_iter().map(|p| p.data));
        crop(encoder.reconstruction().expect("recon"), width, height, &mut recon);
    }
    let seconds = start.elapsed().as_secs_f64();
    let ours_bytes: usize = packets.iter().map(Vec::len).sum();
    let ivf = dir.join("ours.ivf");
    write_ivf(&ivf, width, height, &packets);
    let decoded = ffmpeg_yuv(&ffmpeg, &ivf, "libvpx");
    assert!(first_difference(&decoded, &recon, width, height).is_none(), "{name}: libvpx decodes ours differently");
    let ours_psnr = psnr_y(&source, &recon, width, height);

    let raw = dir.join("src.yuv");
    std::fs::write(&raw, &source).expect("raw");
    let theirs = dir.join("libvpx.ivf");
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s"])
        .arg(format!("{width}x{height}"))
        .args(["-r", "30", "-i"])
        .arg(&raw)
        .args([
            "-c:v", "libvpx", "-b:v", &bitrate.to_string(), "-deadline", "good", "-cpu-used", "4",
            "-threads", "1", "-auto-alt-ref", "0", "-lag-in-frames", "0", "-f", "ivf",
        ])
        .arg(&theirs)
        .output()
        .expect("libvpx");
    assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stderr));
    let theirs_stream = crate::prove::parse_ivf(&std::fs::read(&theirs).expect("ivf"));
    let theirs_bytes: usize = theirs_stream.frames.iter().map(Vec::len).sum();
    let theirs_yuv = ffmpeg_yuv(&ffmpeg, &theirs, "libvpx");
    let theirs_psnr = psnr_y(&source, &theirs_yuv, width, height);
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "{name} {width}x{height} {frames}f @{bitrate}: ours {ours_psnr:.2} dB {ours_bytes} B {:.1} fps; libvpx {theirs_psnr:.2} dB {theirs_bytes} B",
        f64::from(frames) / seconds
    );
    Some((ours_psnr, ours_bytes, seconds, theirs_psnr, theirs_bytes))
}

/// Release only: `cargo test --release -p cutlass-vp8 -- --ignored gate`.
#[test]
#[ignore = "release-mode gate on local files"]
fn gate_quality_and_rate_against_libvpx() {
    let (width, height, frames, bitrate) = (640, 360, 150, 1_000_000);
    let target = f64::from(bitrate) / 8.0 * f64::from(frames) / 30.0;
    for name in ["16929689_1920_1080_30fps.mp4", "7688387224072670472.mp4", "IMG_8144.MP4"] {
        let Some((ours, bytes, seconds, theirs, _)) =
            compare_with_libvpx(name, width, height, frames, bitrate, Vp8Options::default())
        else {
            continue;
        };
        let rate = bytes as f64 / target;
        assert!((0.85..=1.15).contains(&rate), "{name}: rate {rate:.3} of target");
        assert!(ours >= theirs - 2.0, "{name}: {ours:.2} dB against libvpx {theirs:.2} dB");
        assert!(f64::from(frames) / seconds >= 30.0, "{name}: encode {:.1} fps", f64::from(frames) / seconds);
    }
}
