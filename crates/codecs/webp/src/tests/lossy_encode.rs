//! The lossy encoder against `cwebp`: the files must decode the same in
//! `dwebp`, ffmpeg and this crate, and the pictures must be about as good
//! as `cwebp`'s at the same size.
//!
//! The gates below were fixed before the encoder was tuned and are not to
//! be loosened to make a change pass. PSNR is over RGB where alpha is not
//! zero; SSIM is over BT.601 luma in 8x8 windows. `cwebp`'s quality at
//! our size is interpolated, in log size, between the two `-q` values whose
//! files bracket ours.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::lossless_encode::webpinfo_ok;
use super::lossy::lossy_planes;
use super::oracle::{
    cwebp, decode_still, dwebp_rgba, dwebp_yuv, ffmpeg_yuv, read_png, require_or_skip, scratch,
    sips_rgba, tool,
};
use super::stills::lossless_sources;
use crate::lossy::encode::{Config, encode};
use crate::riff::{Bitstream, write_still};

/// Per picture and quality: PSNR (dB) and SSIM below `cwebp` at our size.
const PSNR_SLACK: f64 = 1.0;
const SSIM_SLACK: f64 = 0.02;
/// Averaged over the pictures and qualities of a run.
const PSNR_SLACK_MEAN: f64 = 0.5;
const SSIM_SLACK_MEAN: f64 = 0.01;
/// Our size over `cwebp`'s at the same `-q` and `-m`: per picture, and the
/// geometric mean of a run.
const RATE_RANGE: (f64, f64) = (0.6, 1.6);
const RATE_RANGE_MEAN: (f64, f64) = (0.8, 1.25);

pub(crate) fn lossy_file(rgba: &[u8], width: u32, height: u32, quality: u8, method: u8) -> Vec<u8> {
    let config = Config {
        quality,
        method,
        exact: false,
    };
    let (vp8, alpha) = encode(rgba, width, height, config).expect("encode");
    let has_alpha = alpha.is_some();
    write_still(
        Bitstream::Lossy {
            vp8: &vp8,
            alpha: alpha.as_deref(),
        },
        width,
        height,
        has_alpha,
        None,
    )
}

fn luma(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let visible = p[3] > 0;
            let v = 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
            if visible { v.round() as u8 } else { 0 }
        })
        .collect()
}

pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        if p[3] == 0 {
            continue;
        }
        for c in 0..3 {
            let d = f64::from(p[c]) - f64::from(q[c]);
            sum += d * d;
            count += 1;
        }
    }
    if sum == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 * count as f64 / sum).log10()
}

pub(crate) fn ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let (a, b) = (luma(a), luma(b));
    let mut total = 0.0;
    let mut count = 0usize;
    for y in (0..height.saturating_sub(7)).step_by(4) {
        for x in (0..width.saturating_sub(7)).step_by(4) {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in y..y + 8 {
                for col in x..x + 8 {
                    let p = f64::from(a[row * width + col]);
                    let q = f64::from(b[row * width + col]);
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let (ma, mb) = (sa / 64.0, sb / 64.0);
            let (va, vb, cov) = (
                saa / 64.0 - ma * ma,
                sbb / 64.0 - mb * mb,
                sab / 64.0 - ma * mb,
            );
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

struct Point {
    size: f64,
    psnr: f64,
    ssim: f64,
}

/// One picture under comparison.
struct Case<'a> {
    tools: &'a Tools,
    source: &'a Path,
    original: &'a [u8],
    width: u32,
    height: u32,
    method: u8,
}

/// `cwebp -q quality -m method` of the picture, decoded by `dwebp`.
fn cwebp_point(case: &Case<'_>, quality: u8, out: &Path) -> Point {
    let args = ["-q", &quality.to_string(), "-m", &case.method.to_string()];
    let bytes = cwebp(&case.tools.cwebp, case.source, &args, out);
    let (_, _, decoded) = dwebp_rgba(&case.tools.dwebp, out);
    Point {
        size: bytes.len() as f64,
        psnr: psnr(case.original, &decoded),
        ssim: ssim(
            case.original,
            &decoded,
            case.width as usize,
            case.height as usize,
        ),
    }
}

/// `cwebp`'s PSNR and SSIM interpolated at `size`.
fn cwebp_at_size(case: &Case<'_>, size: f64, dir: &Path) -> (f64, f64) {
    let point = |q: u8| cwebp_point(case, q, &dir.join(format!("cwebp-q{q}.webp")));
    let (mut lo, mut hi) = (0u8, 100u8);
    let (mut lo_point, mut hi_point) = (point(lo), point(hi));
    if size <= lo_point.size {
        return (lo_point.psnr, lo_point.ssim);
    }
    if size >= hi_point.size {
        return (hi_point.psnr, hi_point.ssim);
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let p = point(mid);
        if p.size <= size {
            lo = mid;
            lo_point = p;
        } else {
            hi = mid;
            hi_point = p;
        }
    }
    let t = if hi_point.size > lo_point.size {
        (size.ln() - lo_point.size.ln()) / (hi_point.size.ln() - lo_point.size.ln())
    } else {
        0.0
    };
    (
        lo_point.psnr + t * (hi_point.psnr - lo_point.psnr),
        lo_point.ssim + t * (hi_point.ssim - lo_point.ssim),
    )
}

/// Totals of one run against the gates.
#[derive(Default)]
struct Run {
    psnr_deficit: f64,
    ssim_deficit: f64,
    log_rate: f64,
    count: usize,
    failures: Vec<String>,
}

impl Run {
    fn check(&mut self) {
        let n = self.count.max(1) as f64;
        let rate = (self.log_rate / n).exp();
        println!(
            "mean PSNR deficit {:.3} dB, SSIM deficit {:.4}, size ratio at the same -q {:.3}",
            self.psnr_deficit / n,
            self.ssim_deficit / n,
            rate
        );
        if self.psnr_deficit / n > PSNR_SLACK_MEAN {
            self.failures
                .push(format!("mean PSNR deficit {:.3}", self.psnr_deficit / n));
        }
        if self.ssim_deficit / n > SSIM_SLACK_MEAN {
            self.failures
                .push(format!("mean SSIM deficit {:.4}", self.ssim_deficit / n));
        }
        if !(RATE_RANGE_MEAN.0..=RATE_RANGE_MEAN.1).contains(&rate) {
            self.failures.push(format!("mean size ratio {rate:.3}"));
        }
        assert!(
            self.failures.is_empty(),
            "gates failed:\n{}",
            self.failures.join("\n")
        );
    }
}

/// Encode `source` at `quality`, check every decoder agrees on the file,
/// and score it against `cwebp`.
fn measure(run: &mut Run, tools: &Tools, source: &Path, quality: u8, method: u8, dir: &Path) {
    let (width, height, original) = read_png(&std::fs::read(source).expect("source"));
    let name = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("x")
        .to_string();
    let start = std::time::Instant::now();
    let bytes = lossy_file(&original, width, height, quality, method);
    let elapsed = start.elapsed().as_secs_f64();
    let path = dir.join(format!("{name}-q{quality}-m{method}.webp"));
    std::fs::write(&path, &bytes).expect("write");

    // Every decoder sees the same pixels.
    let ours = decode_still(&bytes);
    let (w, h, theirs) = dwebp_rgba(&tools.dwebp, &path);
    assert_eq!((w, h), (width, height));
    assert!(
        ours.rgba == theirs,
        "{name} q{quality}: this crate and dwebp differ"
    );
    let planes = lossy_planes(&bytes).expect("planes");
    let yuv = dwebp_yuv(&tools.dwebp, &path);
    assert!(
        planes.yuva() == yuv,
        "{name} q{quality}: planes differ from dwebp -yuv"
    );
    if let Some(ffmpeg) = &tools.ffmpeg
        && !ours.alpha
        && let Some(theirs) = ffmpeg_yuv(ffmpeg, &path)
    {
        assert!(
            planes.yuva() == theirs,
            "{name} q{quality}: planes differ from ffmpeg"
        );
    }
    if let Some(webpinfo) = &tools.webpinfo {
        assert!(
            webpinfo_ok(webpinfo, &path),
            "{name} q{quality}: webpinfo refuses it"
        );
    }
    // Alpha is lossless.
    for (a, b) in ours
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(original.as_chunks::<4>().0)
    {
        assert_eq!(a[3], b[3], "{name} q{quality}: alpha");
    }

    let our_psnr = psnr(&original, &ours.rgba);
    let our_ssim = ssim(&original, &ours.rgba, width as usize, height as usize);
    let size = bytes.len() as f64;
    let cdir = scratch("cwebp-points");
    let case = Case {
        tools,
        source,
        original: &original,
        width,
        height,
        method,
    };
    let (their_psnr, their_ssim) = cwebp_at_size(&case, size, &cdir);
    let same_q = cwebp_point(&case, quality, &cdir.join("same-q.webp"));
    let rate = size / same_q.size;
    println!(
        "{name:>22} q{quality:<3} {size:>8} B ({rate:.2}x cwebp -q) PSNR {our_psnr:.2} vs {their_psnr:.2}  SSIM {our_ssim:.4} vs {their_ssim:.4}  {:.2} MP/s",
        f64::from(width * height) / 1e6 / elapsed
    );
    let psnr_deficit = their_psnr - our_psnr;
    let ssim_deficit = their_ssim - our_ssim;
    if psnr_deficit > PSNR_SLACK {
        run.failures.push(format!(
            "{name} q{quality}: PSNR {psnr_deficit:.2} dB below cwebp"
        ));
    }
    if ssim_deficit > SSIM_SLACK {
        run.failures.push(format!(
            "{name} q{quality}: SSIM {ssim_deficit:.4} below cwebp"
        ));
    }
    if !(RATE_RANGE.0..=RATE_RANGE.1).contains(&rate) {
        run.failures.push(format!(
            "{name} q{quality}: size {rate:.2}x cwebp's at the same -q"
        ));
    }
    run.psnr_deficit += psnr_deficit;
    run.ssim_deficit += ssim_deficit;
    run.log_rate += rate.ln();
    run.count += 1;
}

struct Tools {
    cwebp: PathBuf,
    dwebp: PathBuf,
    ffmpeg: Option<PathBuf>,
    webpinfo: Option<PathBuf>,
}

fn tools() -> Option<Tools> {
    Some(Tools {
        cwebp: tool("cwebp")?,
        dwebp: tool("dwebp")?,
        ffmpeg: tool("ffmpeg"),
        webpinfo: tool("webpinfo"),
    })
}

#[test]
fn lossy_files_decode_everywhere_and_meet_the_gates() {
    let Some(tools) = tools() else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let dir = scratch("lossy-encode");
    let mut run = Run::default();
    for source in lossless_sources(&dir) {
        for quality in [30, 75, 95] {
            measure(&mut run, &tools, &source, quality, 4, &dir);
        }
    }
    run.check();
}

#[test]
fn every_method_and_odd_sizes_decode_like_dwebp() {
    let Some(tools) = tools() else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let dir = scratch("lossy-methods");
    for (width, height) in [(1u32, 1u32), (17, 33), (33, 17), (131, 97)] {
        let rgba = super::oracle::synthetic_rgba(width, height);
        for method in 0..=6 {
            for quality in [0, 50, 100] {
                let bytes = lossy_file(&rgba, width, height, quality, method);
                let path = dir.join(format!("{width}x{height}-m{method}-q{quality}.webp"));
                std::fs::write(&path, &bytes).expect("write");
                let (_, _, theirs) = dwebp_rgba(&tools.dwebp, &path);
                assert!(decode_still(&bytes).rgba == theirs, "{}", path.display());
            }
        }
    }
    if let Some(sips) = tool("sips") {
        let rgba = super::oracle::synthetic_rgba(64, 48);
        let path = dir.join("sips.webp");
        std::fs::write(&path, lossy_file(&rgba, 64, 48, 90, 4)).expect("write");
        let (w, h, imageio) = sips_rgba(&sips, &path);
        assert_eq!((w, h), (64, 48));
        let ours = decode_still(&std::fs::read(&path).expect("read")).rgba;
        // ImageIO upsamples chroma its own way: close, not exact.
        // ImageIO premultiplies: compare the opaque pixels.
        let opaque: Vec<u8> = ours
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], if p[3] == 255 { 255 } else { 0 }])
            .collect();
        assert!(psnr(&opaque, &imageio) > 30.0);
    }
}

/// Every PNG in `WEBP_SOURCE_DIR` at several qualities: the table, and the
/// gates over the whole set.
#[test]
#[ignore = "needs WEBP_SOURCE_DIR"]
fn measure_lossy_against_cwebp() {
    let Some(dir) = std::env::var_os("WEBP_SOURCE_DIR") else {
        return;
    };
    let tools = tools().expect("cwebp and dwebp");
    let out = scratch("lossy-measure");
    let method: u8 = std::env::var("WEBP_METHOD")
        .ok()
        .and_then(|m| m.parse().ok())
        .unwrap_or(4);
    let qualities: Vec<u8> = std::env::var("WEBP_QUALITIES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![30, 50, 75, 90]);
    let ffmpeg = tools.ffmpeg.clone().expect("ffmpeg");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .filter(|p| {
            std::env::var("WEBP_ONLY").map_or(true, |only| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| only.split(',').any(|o| o == s))
            })
        })
        .collect();
    sources.sort();
    let mut run = Run::default();
    for original in &sources {
        let name = original.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
        let source = out.join(format!("{name}.png"));
        let status = Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-i"])
            .arg(original)
            .args(["-pix_fmt", "rgba"])
            .arg(&source)
            .status()
            .expect("ffmpeg");
        assert!(status.success());
        for &quality in &qualities {
            measure(&mut run, &tools, &source, quality, method, &out);
        }
    }
    run.check();
}
