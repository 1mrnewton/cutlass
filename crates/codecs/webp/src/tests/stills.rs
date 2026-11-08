//! Still decode against `dwebp`.

use super::lossy::lossy_planes;
use super::oracle::{
    cwebp, decode_still, dwebp_rgba, dwebp_yuv, ffmpeg_yuv, first_difference, local_files,
    read_png, require_or_skip, scratch, sips_rgba, synthetic_rgba, tool, try_decode_still,
    video_frame, webp_files, write_png,
};

/// Every still under `WEBP_CORPUS_DIR` decodes to `dwebp`'s exact RGBA.
///
/// The corpus is generated outside the repo (see `STATUS.md`); the test is
/// ignored unless that variable names it.
#[test]
#[ignore = "needs WEBP_CORPUS_DIR"]
fn corpus_stills_match_dwebp_exactly() {
    let Some(dir) = std::env::var_os("WEBP_CORPUS_DIR") else {
        return;
    };
    let Some(dwebp) = tool("dwebp") else {
        require_or_skip("dwebp");
        return;
    };
    let ffmpeg = tool("ffmpeg");
    let mut checked = 0;
    let mut skipped = 0;
    let mut failures = Vec::new();
    for path in webp_files(std::path::Path::new(&dir)) {
        if path.components().any(|part| part.as_os_str() == "anim") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read");
        let ours = match try_decode_still(&bytes) {
            Ok(ours) => ours,
            Err(cutlass_decode::CodecError::Unsupported { .. }) => {
                skipped += 1;
                continue;
            }
            Err(error) => {
                failures.push(format!("{}: {error:?}", path.display()));
                continue;
            }
        };
        let (width, height, theirs) = dwebp_rgba(&dwebp, &path);
        checked += 1;
        if (ours.width, ours.height) != (width, height) || ours.rgba != theirs {
            failures.push(format!(
                "{}: {:?}",
                path.display(),
                first_difference(&ours.rgba, &theirs, width)
            ));
        }
        if let Some(planes) = lossy_planes(&bytes) {
            if planes.yuva() != dwebp_yuv(&dwebp, &path) {
                failures.push(format!("{}: planes against dwebp", path.display()));
            }
            if let Some(ffmpeg) = &ffmpeg
                && let Some(theirs) = ffmpeg_yuv(ffmpeg, &path)
                && planes.yuv != theirs
            {
                failures.push(format!("{}: planes against ffmpeg", path.display()));
            }
        }
    }
    assert!(checked > 0, "empty corpus");
    assert!(
        failures.is_empty(),
        "{} of {checked} differ ({skipped} skipped):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Sources for the lossless checks: a synthetic picture with every alpha
/// level, and, when this machine has them, a frame of a real video and the
/// real PNG in `local-files`.
pub(super) fn lossless_sources(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut sources = Vec::new();
    let synthetic = dir.join("synthetic.png");
    write_png(&synthetic, 97, 61, &synthetic_rgba(97, 61));
    sources.push(synthetic);
    if let Some(local) = local_files() {
        if let Some(ffmpeg) = tool("ffmpeg")
            && let Ok(entries) = std::fs::read_dir(&local)
        {
            let video = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
                .min();
            if let Some(video) = video {
                let frame = dir.join("frame.png");
                if video_frame(&ffmpeg, &video, &frame, "scale=401:-1") {
                    sources.push(frame);
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&local) {
            let png = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
                .min();
            sources.extend(png);
        }
    }
    sources
}

#[test]
fn cwebp_lossless_files_decode_to_the_exact_pixels_of_dwebp_and_imageio() {
    let (Some(cwebp_tool), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let sips = tool("sips");
    let dir = scratch("lossless");
    for source in lossless_sources(&dir) {
        let (_, _, original) = read_png(&std::fs::read(&source).expect("source"));
        for (label, args) in [
            ("z0", &["-lossless", "-exact", "-z", "0"][..]),
            ("z9", &["-lossless", "-exact", "-z", "9"][..]),
        ] {
            let out = dir.join(format!(
                "{}-{label}.webp",
                source.file_stem().and_then(|s| s.to_str()).unwrap_or("x")
            ));
            let bytes = cwebp(&cwebp_tool, &source, args, &out);
            let ours = decode_still(&bytes);
            let (width, height, theirs) = dwebp_rgba(&dwebp, &out);
            assert_eq!((ours.width, ours.height), (width, height));
            assert_eq!(
                first_difference(&ours.rgba, &theirs, width),
                None,
                "{} against dwebp",
                out.display()
            );
            // `-exact` keeps the RGB under clear alpha, so the source is the
            // expected picture too.
            assert_eq!(first_difference(&ours.rgba, &original, width), None);
            if let Some(sips) = &sips {
                let (w, h, imageio) = sips_rgba(sips, &out);
                assert_eq!((w, h), (width, height));
                // ImageIO premultiplies: compare where alpha is opaque.
                for (index, (a, b)) in ours
                    .rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(imageio.as_chunks::<4>().0)
                    .enumerate()
                {
                    if a[3] == 255 {
                        assert_eq!(a, b, "{} pixel {index} against ImageIO", out.display());
                    }
                }
            }
        }
    }
}
