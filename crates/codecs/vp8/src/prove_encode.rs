//! The encoder's packets, decoded here and by ffmpeg's `vp8` and libvpx,
//! must equal the encoder's own reconstruction.

use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{Plane, VideoFrame};

use crate::decode::Frame;
use crate::prove::{
    Ivf, decode_counted, ffmpeg_bin, ffmpeg_yuv, first_difference, temp_dir, write_ivf,
};
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
        if !opts.hidden_altref {
            assert_eq!(out.len(), 1);
        }
        packets.extend(out.into_iter().map(|p| p.data));
    }
    packets.extend(encoder.drain().expect("drain").into_iter().map(|p| p.data));
    assert_eq!(encoder.shown_recon.len(), count as usize, "shown frames");
    for frame in &encoder.shown_recon {
        crop(frame, width, height, &mut recon);
    }
    (packets, recon)
}

/// Macroblocks per reference (intra, last, golden, altref) in our decoder.
fn reference_counts(width: u32, height: u32, packets: &[Vec<u8>]) -> [u64; 4] {
    use cutlass_decode::{CodecId, CodecParams, Packet, video::VideoDecoder};
    let params = CodecParams::video(CodecId::Vp8, width, height);
    let mut decoder = crate::Vp8Decoder::open(&params).expect("open");
    for (i, data) in packets.iter().enumerate() {
        let packet = Packet::video(data.clone(), i as i64, 1, 30, i == 0);
        decoder.push(&packet).expect("decode");
    }
    decoder.ref_counts()
}

/// Encodes, checks all three decoders against the reconstruction, and
/// returns the macroblocks per luma mode.
fn check(name: &str, width: u32, height: u32, count: u32, opts: Vp8Options) -> [u64; 10] {
    let (packets, recon) = encode_synthetic(width, height, count, opts);
    verify(name, width, height, count, packets, &recon)
}

/// Checks all three decoders against a reconstruction and returns the
/// macroblocks per luma mode.
fn verify(
    name: &str,
    width: u32,
    height: u32,
    count: u32,
    packets: Vec<Vec<u8>>,
    recon: &[u8],
) -> [u64; 10] {
    let stream = Ivf {
        width,
        height,
        frames: packets,
    };
    let (ours, modes) = decode_counted(&stream);
    if let Some(diff) = first_difference(&ours, recon, width, height) {
        panic!("{name}: our decoder differs from the reconstruction at {diff}");
    }
    assert_eq!(ours.len(), recon.len(), "{name}: bytes");
    let bytes: usize = stream.frames.iter().map(Vec::len).sum();
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("{name}: {bytes} bytes; ffmpeg missing, oracles skipped");
        return modes;
    };
    let dir = temp_dir(name);
    let ivf = dir.join("ours.ivf");
    write_ivf(&ivf, width, height, &stream.frames);
    for decoder in ["vp8", "libvpx"] {
        let theirs = ffmpeg_yuv(&ffmpeg, &ivf, decoder);
        if let Some(diff) = first_difference(&theirs, recon, width, height) {
            panic!("{name}: {decoder} differs at {diff}");
        }
        assert_eq!(theirs.len(), recon.len(), "{name}: {decoder} bytes");
    }
    eprintln!(
        "{name}: {count} frames, {bytes} bytes, identical in all three decoders; modes {modes:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    modes
}

/// DC, V, H, TM, B_PRED, NEAREST, NEAR, ZERO, NEW, and SPLIT all occur.
#[test]
fn every_macroblock_mode_is_chosen() {
    let mut total = [0u64; 10];
    for (name, speed, q) in [
        ("modes_fine", 0, 12),
        ("modes_mid", 1, 40),
        ("modes_coarse", 0, 90),
    ] {
        let opts = Vp8Options {
            quantizer: q,
            speed,
            golden_interval: 4,
            ..Vp8Options::default()
        };
        for (t, m) in total.iter_mut().zip(check(name, 160, 96, 10, opts)) {
            *t += m;
        }
    }
    assert!(total.iter().all(|&m| m > 0), "modes {total:?}");
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
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
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
        Plane::tight(
            width.div_ceil(2),
            height.div_ceil(2),
            f[w * h..w * h + cw * ch].to_vec(),
        )
        .expect("u"),
        Plane::tight(
            width.div_ceil(2),
            height.div_ceil(2),
            f[w * h + cw * ch..].to_vec(),
        )
        .expect("v"),
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

/// PSNR of Y, U, and V over all frames, and the worst frame's PSNR-Y.
fn psnr_planes(a: &[u8], b: &[u8], width: u32, height: u32) -> [f64; 4] {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let size = w * h + 2 * cw * ch;
    let frames = a.len().min(b.len()) / size;
    let db = |sse: f64, n: usize| 10.0 * (255.0 * 255.0 / (sse / n as f64).max(1e-10)).log10();
    let sse = |range: std::ops::Range<usize>| -> f64 {
        range
            .map(|i| {
                let d = f64::from(a[i]) - f64::from(b[i]);
                d * d
            })
            .sum()
    };
    let (mut planes, mut worst) = ([0f64; 3], f64::INFINITY);
    for f in 0..frames {
        let base = f * size;
        let y = sse(base..base + w * h);
        planes[0] += y;
        planes[1] += sse(base + w * h..base + w * h + cw * ch);
        planes[2] += sse(base + w * h + cw * ch..base + size);
        worst = worst.min(db(y, w * h));
    }
    [
        db(planes[0], frames * w * h),
        db(planes[1], frames * cw * ch),
        db(planes[2], frames * cw * ch),
        worst,
    ]
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
    let mut encoder =
        Vp8Encoder::with_options(width, height, Vp8Options { bitrate, ..opts }).expect("open");
    let mut packets = Vec::new();
    let mut recon = Vec::new();
    let count = frames as usize;
    let inputs: Vec<VideoFrame> = (0..count)
        .map(|i| frame_of(&source, width, height, i))
        .collect();
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
    }
    packets.extend(encoder.drain().expect("drain").into_iter().map(|p| p.data));
    let seconds = start.elapsed().as_secs_f64();
    for frame in &encoder.shown_recon {
        crop(frame, width, height, &mut recon);
    }
    let ours_bytes: usize = packets.iter().map(Vec::len).sum();
    let ivf = dir.join("ours.ivf");
    write_ivf(&ivf, width, height, &packets);
    let decoded = ffmpeg_yuv(&ffmpeg, &ivf, "libvpx");
    assert!(
        first_difference(&decoded, &recon, width, height).is_none(),
        "{name}: libvpx decodes ours differently"
    );
    let ours_psnr = psnr_y(&source, &recon, width, height);

    let raw = dir.join("src.yuv");
    std::fs::write(&raw, &source).expect("raw");
    let theirs = dir.join("libvpx.ivf");
    let status = std::process::Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
        ])
        .arg(format!("{width}x{height}"))
        .args(["-r", "30", "-i"])
        .arg(&raw)
        .args([
            "-c:v",
            "libvpx",
            "-b:v",
            &bitrate.to_string(),
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-threads",
            "1",
            "-auto-alt-ref",
            "0",
            "-lag-in-frames",
            "0",
            "-f",
            "ivf",
        ])
        .arg(&theirs)
        .output()
        .expect("libvpx");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let theirs_stream = crate::prove::parse_ivf(&std::fs::read(&theirs).expect("ivf"));
    let theirs_bytes: usize = theirs_stream.frames.iter().map(Vec::len).sum();
    let theirs_yuv = ffmpeg_yuv(&ffmpeg, &theirs, "libvpx");
    let theirs_psnr = psnr_y(&source, &theirs_yuv, width, height);
    let _ = std::fs::remove_dir_all(&dir);
    let [_, ou, ov, ow] = psnr_planes(&source, &recon, width, height);
    let [_, tu, tv, tw] = psnr_planes(&source, &theirs_yuv, width, height);
    let keys = packets
        .iter()
        .filter(|p| p.first().is_some_and(|b| b & 1 == 0))
        .count();
    eprintln!(
        "{name} {width}x{height} {frames}f @{bitrate}: ours {ours_psnr:.2} dB (U {ou:.2} V {ov:.2} worst Y {ow:.2}) {ours_bytes} B {:.1} fps {keys} keys; libvpx {theirs_psnr:.2} dB (U {tu:.2} V {tv:.2} worst Y {tw:.2}) {theirs_bytes} B",
        f64::from(frames) / seconds
    );
    assert!(
        ou >= tu - 2.0 && ov >= tv - 2.0,
        "{name}: chroma U {ou:.2} V {ov:.2} dB against libvpx U {tu:.2} V {tv:.2}"
    );
    Some((ours_psnr, ours_bytes, seconds, theirs_psnr, theirs_bytes))
}

/// Release only: `cargo test --release -p cutlass-vp8 -- --ignored gate`.
#[test]
#[ignore = "release-mode gate on local files"]
fn gate_quality_and_rate_against_libvpx() {
    let (width, height, frames, bitrate) = (640, 360, 150, 1_000_000);
    let target = f64::from(bitrate) / 8.0 * f64::from(frames) / 30.0;
    for name in [
        "16929689_1920_1080_30fps.mp4",
        "7688387224072670472.mp4",
        "IMG_8144.MP4",
    ] {
        let Some((ours, bytes, seconds, theirs, _)) =
            compare_with_libvpx(name, width, height, frames, bitrate, Vp8Options::default())
        else {
            continue;
        };
        let rate = bytes as f64 / target;
        assert!(
            (0.85..=1.15).contains(&rate),
            "{name}: rate {rate:.3} of target"
        );
        assert!(
            ours >= theirs - 2.0,
            "{name}: {ours:.2} dB against libvpx {theirs:.2} dB"
        );
        assert!(
            f64::from(frames) / seconds >= 30.0,
            "{name}: encode {:.1} fps",
            f64::from(frames) / seconds
        );
    }
    // Activity segments, reported only: they trade PSNR for flat areas.
    let segmented = Vp8Options {
        segments: true,
        ..Vp8Options::default()
    };
    let _ = compare_with_libvpx(
        "16929689_1920_1080_30fps.mp4",
        width,
        height,
        frames,
        bitrate,
        segmented,
    );
    // Hidden altrefs with a 16-frame lookahead, against the same bounds.
    let hidden = Vp8Options {
        hidden_altref: true,
        ..Vp8Options::default()
    };
    for name in [
        "16929689_1920_1080_30fps.mp4",
        "7688387224072670472.mp4",
        "IMG_8144.MP4",
    ] {
        let Some((ours, bytes, _, theirs, _)) =
            compare_with_libvpx(name, width, height, frames, bitrate, hidden)
        else {
            continue;
        };
        let rate = bytes as f64 / target;
        assert!(
            (0.85..=1.15).contains(&rate),
            "{name} hidden altref: rate {rate:.3} of target"
        );
        assert!(
            ours >= theirs - 2.0,
            "{name} hidden altref: {ours:.2} dB against libvpx {theirs:.2} dB"
        );
    }
}

/// Moving content makes the encoder send new vector probabilities, and the
/// coefficient and mode probabilities follow the counts.
#[test]
fn probability_updates_are_sent() {
    let opts = Vp8Options {
        quantizer: 20,
        ..Vp8Options::default()
    };
    let (frames, _) = encode_synthetic(160, 96, 8, opts);
    let mut state = crate::header::Persist::default();
    let (mut mv, mut coeff) = (0, 0);
    for frame in &frames {
        let parsed = crate::header::parse(frame, &state).expect("parse");
        mv += usize::from(!parsed.header.tag.key && parsed.next.probs.mv != state.probs.mv);
        coeff += usize::from(parsed.next.probs.coeff != state.probs.coeff);
        state = parsed.next;
    }
    assert!(
        mv > 0 && coeff > 0,
        "mv updates {mv}, coefficient updates {coeff}"
    );
}

/// Activity segments: the map on key and golden frames, the quantizer and
/// loop-filter offsets on every frame, in both quantizer modes.
#[test]
fn segments_decode_identically() {
    for (name, bitrate, quantizer) in [
        ("seg_q", 0, 30),
        ("seg_q_edge", 0, 125),
        ("seg_rate", 400_000, 0),
    ] {
        let opts = Vp8Options {
            segments: true,
            bitrate,
            quantizer,
            golden_interval: 3,
            keyframe_interval: 8,
            ..Vp8Options::default()
        };
        let (frames, _) = encode_synthetic(176, 144, 10, opts);
        let mut state = crate::header::Persist::default();
        let (mut maps, mut filters) = (0, 0);
        for frame in &frames {
            let parsed = crate::header::parse(frame, &state).expect("parse");
            let seg = &parsed.header.segmentation;
            assert!(seg.enabled, "{name}");
            maps += usize::from(seg.update_map);
            filters += usize::from(seg.filter != [0; 4]);
            let level = i32::from(parsed.header.filter_level);
            assert!(
                seg.filter
                    .iter()
                    .all(|&f| (0..=63).contains(&(level + i32::from(f)))),
                "{name}: segment filter level outside 0..63"
            );
            state = parsed.next;
        }
        assert!(maps >= 3, "{name}: {maps} map updates");
        assert!(filters > 0, "{name}: no segment filter offsets");
        check(name, 176, 144, 10, opts);
    }
}

/// Versions 1 to 3: bilinear interpolation, the simple loop filter for
/// version 1, no loop filter for 2 and 3, whole-pixel chroma for 3.
#[test]
fn versions_decode_identically() {
    for version in 1..=3u8 {
        for (name, quantizer) in [("fine", 16u8), ("coarse", 70)] {
            let opts = Vp8Options {
                version,
                quantizer,
                golden_interval: 4,
                ..Vp8Options::default()
            };
            let (packets, _) = encode_synthetic(160, 96, 8, opts);
            let stream = Ivf {
                width: 160,
                height: 96,
                frames: packets,
            };
            let c = crate::prove::coverage(&stream);
            assert_eq!(c.versions[usize::from(version)], 8, "v{version} {c:?}");
            assert_eq!(c.simple_filter, 8, "v{version}");
            if version >= 2 {
                let mut state = crate::header::Persist::default();
                for frame in &stream.frames {
                    let parsed = crate::header::parse(frame, &state).expect("parse");
                    assert_eq!(parsed.header.filter_level, 0, "v{version} filter");
                    state = parsed.next;
                }
            }
            let modes = check(&format!("v{version}_{name}"), 160, 96, 8, opts);
            assert!(
                modes[5..].iter().sum::<u64>() > 0,
                "v{version}: inter modes"
            );
        }
    }
}

/// The old golden moves to altref (`copy_buffer_to_alternate = 2`) and
/// macroblocks predict from it.
#[test]
fn altref_copies_decode_identically() {
    let opts = Vp8Options {
        golden_interval: 3,
        quantizer: 30,
        ..Vp8Options::default()
    };
    let (packets, _) = encode_synthetic(160, 96, 12, opts);
    let mut state = crate::header::Persist::default();
    let mut copies = 0;
    for frame in &packets {
        let parsed = crate::header::parse(frame, &state).expect("parse");
        copies += usize::from(parsed.header.copy_to_altref == 2);
        state = parsed.next;
    }
    assert!(copies >= 3, "{copies} copies");
    let refs = reference_counts(160, 96, &packets);
    assert!(refs[3] > 0, "altref macroblocks {refs:?}");
    check("altref_copy", 160, 96, 12, opts);
}

/// Hidden altrefs: each golden group starts with a hidden frame of its
/// last picture, coded with `refresh_entropy_probs = 0`; the frames before
/// that picture use the altref with its sign bias set; the picture itself
/// copies the altref into golden (`copy_buffer_to_golden = 2`).
#[test]
fn hidden_altrefs_decode_identically() {
    for (name, bitrate, version) in [
        ("arf_q", 0, 0u8),
        ("arf_rate", 300_000, 0),
        ("arf_v1", 0, 1),
    ] {
        let opts = Vp8Options {
            hidden_altref: true,
            golden_interval: 5,
            keyframe_interval: 13,
            quantizer: 28,
            bitrate,
            version,
            ..Vp8Options::default()
        };
        let count = 20;
        let (packets, _) = encode_synthetic(176, 112, count, opts);
        let stream = Ivf {
            width: 176,
            height: 112,
            frames: packets,
        };
        let c = crate::prove::coverage(&stream);
        assert_eq!(c.frames, count as usize + c.hidden, "{name} {c:?}");
        assert!(c.hidden >= 3, "{name} {c:?}");
        assert!(c.no_entropy_refresh >= c.hidden, "{name} {c:?}");
        assert!(c.sign_bias > 0 && c.copies >= 3, "{name} {c:?}");
        let refs = reference_counts(176, 112, &stream.frames);
        assert!(refs[3] > 0, "{name}: altref macroblocks {refs:?}");
        check(name, 176, 112, count, opts);
    }
}

/// Error-resilient mode never keeps a frame's probability changes.
#[test]
fn error_resilient_streams_decode_identically() {
    let opts = Vp8Options {
        error_resilient: true,
        quantizer: 30,
        ..Vp8Options::default()
    };
    let (packets, _) = encode_synthetic(160, 96, 8, opts);
    let stream = Ivf {
        width: 160,
        height: 96,
        frames: packets,
    };
    let c = crate::prove::coverage(&stream);
    assert_eq!(c.no_entropy_refresh, 8, "{c:?}");
    check("resilient_enc", 160, 96, 8, opts);
}

/// Sharpness and the display-scale bits reach the header, and a frame
/// where every macroblock has coefficients drops the skip flag.
#[test]
fn header_options_decode_identically() {
    let opts = Vp8Options {
        sharpness: 5,
        scale: [1, 2],
        quantizer: 0,
        keyframe_interval: 3,
        ..Vp8Options::default()
    };
    let (packets, _) = encode_synthetic(160, 96, 6, opts);
    let mut state = crate::header::Persist::default();
    let (mut no_skip, mut scaled) = (0, 0);
    for frame in &packets {
        let parsed = crate::header::parse(frame, &state).expect("parse");
        let h = &parsed.header;
        assert_eq!(h.sharpness, 5);
        no_skip += usize::from(!h.skip_enabled);
        scaled += usize::from(h.tag.key && h.tag.hscale == 1 && h.tag.vscale == 2);
        state = parsed.next;
    }
    assert!(no_skip > 0, "no frame dropped the skip flag");
    assert_eq!(scaled, 2, "key frames carry the scale");
    check("header_options", 160, 96, 6, opts);
}

/// A hard cut to an unrelated picture is coded as a key frame, and only
/// with the option on.
#[test]
fn scene_cuts_become_key_frames() {
    let (width, height, count) = (160u32, 96u32, 10u32);
    let cut = |t: u32| {
        let mut frame = synthetic(width, height, t);
        if t >= 5 {
            // Mirror and invert luma, swap the chroma planes.
            let w = width as usize;
            for row in frame.y.samples.chunks_mut(w) {
                row.reverse();
                for v in row.iter_mut() {
                    *v = 255 - *v;
                }
            }
            std::mem::swap(&mut frame.u, &mut frame.v);
        }
        frame
    };
    for scene_cut in [true, false] {
        let opts = Vp8Options {
            scene_cut,
            quantizer: 30,
            ..Vp8Options::default()
        };
        let mut encoder = Vp8Encoder::with_options(width, height, opts).expect("open");
        let mut packets = Vec::new();
        let mut keys = Vec::new();
        for t in 0..count {
            let input = VideoInput {
                frame: cut(t),
                pts: i64::from(t),
                duration: 1,
                timescale: 30,
            };
            for packet in encoder.push(&input).expect("push") {
                if packet.keyframe {
                    keys.push(t);
                }
                packets.push(packet.data);
            }
        }
        if scene_cut {
            assert_eq!(keys, [0, 5], "the cut is a key frame");
        } else {
            assert_eq!(keys, [0], "no cut detection when off");
        }
        let mut recon = Vec::new();
        for frame in &encoder.shown_recon {
            crop(frame, width, height, &mut recon);
        }
        verify(
            if scene_cut {
                "scene_cut_on"
            } else {
                "scene_cut_off"
            },
            width,
            height,
            count,
            packets,
            &recon,
        );
    }
}
