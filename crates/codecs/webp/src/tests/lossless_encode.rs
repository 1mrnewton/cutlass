//! The lossless encoder against `dwebp`, `webpinfo` and ImageIO, and its
//! size against `cwebp -lossless -z N`.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use super::oracle::{
    cwebp, decode_still, dwebp_rgba, first_difference, read_png, require_or_skip, scratch,
    sips_rgba, tool,
};
use super::stills::lossless_sources;
use crate::lossless::encode::{Config, encode};
use crate::riff::{Bitstream, write_still};

pub(crate) fn to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|p| u32::from_be_bytes([p[3], p[0], p[1], p[2]]))
        .collect()
}

/// A whole lossless file of straight RGBA.
pub(crate) fn lossless_file(rgba: &[u8], width: u32, height: u32, config: Config) -> Vec<u8> {
    let argb = to_argb(rgba);
    let alpha = argb.iter().any(|&p| p >> 24 != 0xff);
    let vp8l = encode(&argb, width, height, config);
    write_still(Bitstream::Lossless(&vp8l), width, height, alpha, None)
}

pub(crate) fn webpinfo_ok(webpinfo: &Path, file: &Path) -> bool {
    Command::new(webpinfo)
        .arg("-quiet")
        .arg(file)
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn encoded_files_decode_exactly_in_dwebp_webpinfo_and_imageio() {
    let Some(dwebp) = tool("dwebp") else {
        require_or_skip("dwebp");
        return;
    };
    let webpinfo = tool("webpinfo");
    let sips = tool("sips");
    let dir = scratch("lossless-encode");
    for source in lossless_sources(&dir) {
        let (width, height, rgba) = read_png(&std::fs::read(&source).expect("source"));
        for effort in [0u8, 3, 6, 9] {
            let bytes = lossless_file(
                &rgba,
                width,
                height,
                Config {
                    effort,
                    exact: true,
                },
            );
            let out = dir.join(format!(
                "{}-e{effort}.webp",
                source.file_stem().and_then(|s| s.to_str()).unwrap_or("x")
            ));
            std::fs::write(&out, &bytes).expect("write");
            assert_eq!(decode_still(&bytes).rgba, rgba, "{} ours", out.display());
            let (w, h, theirs) = dwebp_rgba(&dwebp, &out);
            assert_eq!((w, h), (width, height));
            assert_eq!(
                first_difference(&theirs, &rgba, width),
                None,
                "{} dwebp",
                out.display()
            );
            if let Some(webpinfo) = &webpinfo {
                assert!(webpinfo_ok(webpinfo, &out), "{} webpinfo", out.display());
            }
            if let Some(sips) = &sips {
                let (w, h, imageio) = sips_rgba(sips, &out);
                assert_eq!((w, h), (width, height));
                for (index, (a, b)) in rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(imageio.as_chunks::<4>().0)
                    .enumerate()
                {
                    if a[3] == 255 {
                        assert_eq!(a, b, "{} pixel {index} ImageIO", out.display());
                    }
                }
            }
        }
    }
}

#[test]
fn transparent_pixels_lose_their_color_unless_exact() {
    let rgba = [10, 20, 30, 0, 40, 50, 60, 255, 70, 80, 90, 0, 1, 2, 3, 128];
    let loose = decode_still(&lossless_file(
        &rgba,
        2,
        2,
        Config {
            effort: 5,
            exact: false,
        },
    ));
    assert_eq!(
        loose.rgba,
        [0, 0, 0, 0, 40, 50, 60, 255, 0, 0, 0, 0, 1, 2, 3, 128]
    );
    let exact = decode_still(&lossless_file(
        &rgba,
        2,
        2,
        Config {
            effort: 5,
            exact: true,
        },
    ));
    assert_eq!(exact.rgba, rgba);
}

/// Size and speed of every effort beside `cwebp -lossless -exact -z N`, on
/// the PNGs in `WEBP_SOURCE_DIR`. Prints a table; fails only on a mismatch.
#[test]
#[ignore = "needs WEBP_SOURCE_DIR"]
fn measure_lossless_against_cwebp() {
    let Some(dir) = std::env::var_os("WEBP_SOURCE_DIR") else {
        return;
    };
    let cwebp_tool = tool("cwebp").expect("cwebp");
    let out_dir = scratch("lossless-measure");
    let mut sources: Vec<_> = std::fs::read_dir(&dir)
        .expect("source dir")
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
    let levels: Vec<u8> = std::env::var("WEBP_LEVELS")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| (0..=9).collect());
    let mut totals = vec![(0u64, 0u64, 0f64, 0f64); 10];
    let ffmpeg = tool("ffmpeg").expect("ffmpeg");
    for original in &sources {
        let name = original.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
        // Eight-bit RGBA for both encoders, whatever the PNG stores.
        let source = &out_dir.join(format!("{name}.png"));
        let status = Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-i"])
            .arg(original)
            .args(["-pix_fmt", "rgba"])
            .arg(source)
            .status()
            .expect("ffmpeg");
        assert!(status.success());
        let (width, height, rgba) = read_png(&std::fs::read(source).expect("source"));
        let mut line = format!("{name:>14} {width}x{height}");
        for &level in &levels {
            let start = Instant::now();
            let ours = lossless_file(
                &rgba,
                width,
                height,
                Config {
                    effort: level,
                    exact: true,
                },
            );
            let ours_time = start.elapsed().as_secs_f64();
            assert_eq!(decode_still(&ours).rgba, rgba, "{name} z{level}");
            let out = out_dir.join(format!("{name}-z{level}.webp"));
            let start = Instant::now();
            let theirs = cwebp(
                &cwebp_tool,
                source,
                &["-lossless", "-exact", "-z", &level.to_string()],
                &out,
            );
            let theirs_time = start.elapsed().as_secs_f64();
            let total = &mut totals[usize::from(level)];
            total.0 += ours.len() as u64;
            total.1 += theirs.len() as u64;
            total.2 += ours_time;
            total.3 += theirs_time;
            line += &format!(
                " z{level} {:.3} ({}/{})",
                ours.len() as f64 / theirs.len() as f64,
                ours.len(),
                theirs.len()
            );
        }
        println!("{line}");
    }
    for &level in &levels {
        let (ours, theirs, ours_time, theirs_time) = totals[usize::from(level)];
        println!(
            "z{level}: ours {ours} cwebp {theirs} ratio {:.4}  time ours {ours_time:.2}s cwebp {theirs_time:.2}s",
            ours as f64 / theirs as f64
        );
    }
}
