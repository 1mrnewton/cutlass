//! Encode a real clip at every profile and measure it with ffmpeg.
//!
//! Gates, set before any tuning, for each profile on the 1080p clip:
//! - ffmpeg decodes every frame, and so does VideoToolbox on a Mac.
//! - The data rate is at most 1.05x Apple's target and at least 0.8x of
//!   it (the clip is detailed enough to fill the budget).
//! - Luma PSNR is at least prores_ks's less 0.3 dB, at no more than 1.05x
//!   prores_ks's data rate. Where prores_ks spends less than the target
//!   (it does at 4444 and XQ), this encoder is run again with its budget
//!   scaled to prores_ks's rate, so the two are compared at the same rate.
//!
//! prores_ks and VideoToolbox code the same raw frames, for the table.

use std::path::Path;

use cutlass_decode::{CodecId, CodecParams, PixelFormat, Plane, VideoFrame};
use cutlass_prores::{Profile, ProresEncoder, target_mbps};

use crate::support::{Diff, HD, RATE, Scratch, ffmpeg, local, mux_mov, run, samples16, ssim};

const FRAMES: usize = 6;

struct Measure {
    mbps: f64,
    psnr: [f64; 3],
    ssim_y: f64,
}

fn raw_frames(bytes: &[u8], width: u32, height: u32, chroma_w: u32) -> Vec<VideoFrame> {
    let luma = (width * height * 2) as usize;
    let chroma = (chroma_w * height * 2) as usize;
    let format = if chroma_w == width {
        PixelFormat::Yuv444p10
    } else {
        PixelFormat::Yuv422p10
    };
    bytes
        .chunks_exact(luma + 2 * chroma)
        .map(|frame| {
            VideoFrame::planar(
                format,
                width,
                height,
                Plane::bytes(width, height, 2, frame[..luma].to_vec()).expect("y"),
                Plane::bytes(chroma_w, height, 2, frame[luma..luma + chroma].to_vec()).expect("u"),
                Plane::bytes(chroma_w, height, 2, frame[luma + chroma..].to_vec()).expect("v"),
            )
            .expect("frame")
        })
        .collect()
}

/// The raw source's sample layout.
#[derive(Clone, Copy)]
struct Layout<'a> {
    pix_fmt: &'a str,
    width: u32,
    height: u32,
    chroma_w: u32,
}

/// Decodes `movie` with ffmpeg and measures it against the raw `source`.
fn measure(
    ffmpeg: &Path,
    movie: &Path,
    source: &Path,
    layout: Layout<'_>,
    scratch: &Scratch,
) -> Option<Measure> {
    let Layout {
        pix_fmt,
        width,
        height,
        chroma_w,
    } = layout;
    let decoded = scratch.path("decoded.raw");
    if !run(
        ffmpeg,
        &[
            "-v",
            "error",
            "-y",
            "-i",
            movie.to_str()?,
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            decoded.to_str()?,
        ],
    ) {
        return None;
    }
    let ours = samples16(&std::fs::read(&decoded).ok()?);
    let reference = samples16(&std::fs::read(source).ok()?);
    if ours.len() != reference.len() {
        eprintln!(
            "{}: {} samples, source {}",
            movie.display(),
            ours.len(),
            reference.len()
        );
        return None;
    }
    let luma = (width * height) as usize;
    let chroma = (chroma_w * height) as usize;
    let frame = luma + 2 * chroma;
    let mut planes: [Vec<u16>; 6] = Default::default();
    for (a, b) in ours.chunks_exact(frame).zip(reference.chunks_exact(frame)) {
        planes[0].extend_from_slice(&a[..luma]);
        planes[1].extend_from_slice(&a[luma..luma + chroma]);
        planes[2].extend_from_slice(&a[luma + chroma..]);
        planes[3].extend_from_slice(&b[..luma]);
        planes[4].extend_from_slice(&b[luma..luma + chroma]);
        planes[5].extend_from_slice(&b[luma + chroma..]);
    }
    let psnr = [0, 1, 2].map(|i| Diff::between(&planes[i], &planes[i + 3], 1023.0).psnr);
    let bytes = std::fs::metadata(movie).ok()?.len();
    let frames = ours.len() / frame;
    let (ssim_y, _) = ssim(ffmpeg, movie, source, pix_fmt, width, height);
    Some(Measure {
        mbps: bytes as f64 * 8.0 / frames as f64 * f64::from(RATE.0) / f64::from(RATE.1) / 1e6,
        psnr,
        ssim_y,
    })
}

fn show(name: &str, m: &Option<Measure>) -> String {
    match m {
        Some(m) => format!(
            "{name:<6} {:>8.1} Mb/s  PSNR Y {:.2} U {:.2} V {:.2}  SSIM Y {:.5}",
            m.mbps, m.psnr[0], m.psnr[1], m.psnr[2], m.ssim_y
        ),
        None => format!("{name:<6} not available"),
    }
}

#[test]
fn every_profile_meets_its_rate_and_quality_gates() {
    let Some(ffmpeg) = ffmpeg() else { return };
    let Some(clip) = local(HD) else { return };
    let (width, height) = (1920u32, 1080u32);
    let mut failures = Vec::new();
    let profiles: [(Profile, &str, &str); 6] = [
        (Profile::Proxy, "0", "proxy"),
        (Profile::Lt, "1", "lt"),
        (Profile::Standard, "2", "standard"),
        (Profile::Hq, "3", "hq"),
        (Profile::P4444, "4", "4444"),
        (Profile::Xq, "5", "xq"),
    ];
    let scratch = Scratch::new("encode");
    for (profile, ks_profile, vt_profile) in profiles {
        let (pix_fmt, chroma_w) = if profile.is_444() {
            ("yuv444p10le", width)
        } else {
            ("yuv422p10le", width / 2)
        };
        let layout = Layout {
            pix_fmt,
            width,
            height,
            chroma_w,
        };
        let source = scratch.path(&format!("source-{pix_fmt}.raw"));
        if !source.exists() {
            assert!(run(
                &ffmpeg,
                &[
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    clip.to_str().expect("path"),
                    "-frames:v",
                    &FRAMES.to_string(),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    pix_fmt,
                    source.to_str().expect("path"),
                ],
            ));
        }
        let raw = std::fs::read(&source).expect("source");
        let frames = raw_frames(&raw, width, height, chroma_w);
        let mut params =
            CodecParams::with_format(CodecId::Prores, width, height, PixelFormat::None);
        params.extra = profile.fourcc().to_vec();
        let mut encoder = ProresEncoder::open(&params).expect("open");
        let coded: Vec<Vec<u8>> = frames
            .iter()
            .map(|frame| encoder.encode(frame).expect("encode"))
            .collect();
        let ours_path = scratch.path("ours.mov");
        std::fs::write(
            &ours_path,
            mux_mov(&coded, &profile.fourcc(), width, height),
        )
        .expect("write");
        let ours = measure(&ffmpeg, &ours_path, &source, layout, &scratch);

        let size = format!("{width}x{height}");
        let rate = format!("{}/{}", RATE.0, RATE.1);
        let other = |name: &str, args: &[&str]| -> Option<Measure> {
            let out = scratch.path(&format!("{name}.mov"));
            let mut all = vec![
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                "-s",
                &size,
                "-r",
                &rate,
                "-i",
                source.to_str()?,
            ];
            all.extend_from_slice(args);
            all.push(out.to_str()?);
            if !run(&ffmpeg, &all) {
                return None;
            }
            measure(&ffmpeg, &out, &source, layout, &scratch)
        };
        let ks = other("ks", &["-c:v", "prores_ks", "-profile:v", ks_profile]);
        let vt = if cfg!(target_vendor = "apple") {
            other(
                "vt",
                &["-c:v", "prores_videotoolbox", "-profile:v", vt_profile],
            )
        } else {
            None
        };
        let target = target_mbps(profile, width, height);
        eprintln!(
            "{} target {target:.1} Mb/s, quantizer {:?}\n  {}\n  {}\n  {}",
            profile.label(),
            encoder.last_quantizer(),
            show("ours", &ours),
            show("ks", &ks),
            show("vt", &vt)
        );
        let Some(ours) = ours else {
            failures.push(format!(
                "{}: ffmpeg did not decode our stream",
                profile.label()
            ));
            continue;
        };
        if ours.mbps > 1.05 * target || ours.mbps < 0.8 * target {
            failures.push(format!(
                "{}: {:.1} Mb/s against a target of {target:.1}",
                profile.label(),
                ours.mbps
            ));
        }
        if let Some(ks) = &ks {
            let mut ours = ours;
            if ours.mbps > 1.05 * ks.mbps {
                let scale = ks.mbps / target;
                let mut matched = ProresEncoder::open(&params)
                    .expect("open")
                    .with_budget_scale(scale)
                    .expect("scale");
                let coded: Vec<Vec<u8>> = frames
                    .iter()
                    .map(|frame| matched.encode(frame).expect("encode"))
                    .collect();
                std::fs::write(
                    &ours_path,
                    mux_mov(&coded, &profile.fourcc(), width, height),
                )
                .expect("write");
                let at_ks = measure(&ffmpeg, &ours_path, &source, layout, &scratch)
                    .expect("decode at the prores_ks rate");
                eprintln!("  {}", show("ours@ks", &Some(Measure { ..at_ks })));
                ours = at_ks;
            }
            if ours.mbps > 1.05 * ks.mbps || ours.psnr[0] < ks.psnr[0] - 0.3 {
                failures.push(format!(
                    "{}: Y {:.2} dB at {:.1} Mb/s, prores_ks {:.2} dB at {:.1}",
                    profile.label(),
                    ours.psnr[0],
                    ours.mbps,
                    ks.psnr[0],
                    ks.mbps
                ));
            }
        }
        if cfg!(target_vendor = "apple") {
            let null = [
                "-v",
                "error",
                "-hwaccel",
                "videotoolbox",
                "-i",
                ours_path.to_str().expect("path"),
                "-f",
                "null",
                "-",
            ];
            if !run(&ffmpeg, &null) {
                failures.push(format!(
                    "{}: VideoToolbox did not decode our stream",
                    profile.label()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
