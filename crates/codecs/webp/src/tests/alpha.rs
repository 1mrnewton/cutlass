//! Lossy pictures with `ALPH`, and `VP8X` stills, against `dwebp`.

use std::path::Path;
use std::process::Command;

use cutlass_decode::{CodecId, CodecParams, ColorType, Decoder, Packet};

use super::lossy::lossy_planes;
use super::oracle::{
    cwebp, decode_still, dwebp_rgba, dwebp_yuv, first_difference, local_files, read_png,
    require_or_skip, scratch, synthetic_rgba, tool, write_png,
};
use crate::WebpDecoder;
use crate::alpha::{Filter, filter};
use crate::riff::{chunks, file, vp8x};

/// RGBA and the Y, U, V and alpha planes all equal to libwebp's.
fn check(dwebp: &Path, path: &Path) {
    let bytes = std::fs::read(path).expect("read");
    let ours = decode_still(&bytes);
    assert!(ours.alpha, "{} has alpha", path.display());
    let (width, height, theirs) = dwebp_rgba(dwebp, path);
    assert_eq!((ours.width, ours.height), (width, height));
    assert_eq!(
        first_difference(&ours.rgba, &theirs, width),
        None,
        "{} against dwebp",
        path.display()
    );
    let planes = lossy_planes(&bytes).expect("lossy");
    assert!(
        planes.yuva() == dwebp_yuv(dwebp, path),
        "{} planes against dwebp",
        path.display()
    );
}

#[test]
fn cwebp_alpha_in_every_coding_matches_dwebp() {
    let (Some(cwebp_tool), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let dir = scratch("alpha");
    let mut sources = Vec::new();
    for (width, height) in [(97, 61), (33, 17), (1, 1)] {
        let path = dir.join(format!("alpha{width}x{height}.png"));
        write_png(&path, width, height, &synthetic_rgba(width, height));
        sources.push(path);
    }
    if let Some(local) = local_files()
        && let Ok(entries) = std::fs::read_dir(&local)
    {
        // A real PNG, given alpha if it has none: a soft-edged disc.
        let png = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
            .min();
        if let Some(png) = png {
            let (width, height, mut rgba) = read_png(&std::fs::read(&png).expect("png"));
            if width * height <= 4_000_000 {
                let (cx, cy) = (f64::from(width) / 2.0, f64::from(height) / 2.0);
                for (index, pixel) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let x = f64::from(index as u32 % width) - cx;
                    let y = f64::from(index as u32 / width) - cy;
                    let r = (x * x + y * y).sqrt() / cx.min(cy);
                    pixel[3] = ((1.2 - r).clamp(0.0, 1.0) * 255.0) as u8;
                }
                let path = dir.join("real.png");
                write_png(&path, width, height, &rgba);
                sources.push(path);
            }
        }
    }
    for source in sources {
        for (label, args) in [
            ("default", &[][..]),
            ("aq50", &["-alpha_q", "50"][..]),
            ("none", &["-alpha_filter", "none"][..]),
            ("fast", &["-alpha_filter", "fast"][..]),
            ("best", &["-alpha_filter", "best", "-q", "40"][..]),
            ("raw", &["-alpha_method", "0"][..]),
            ("exact", &["-exact", "-alpha_q", "20"][..]),
        ] {
            let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
            let out = dir.join(format!("{stem}-{label}.webp"));
            cwebp(&cwebp_tool, &source, args, &out);
            check(&dwebp, &out);
        }
    }
}

/// `cwebp` picks among the filters itself and rarely writes some of them.
/// Replace its `ALPH` with a raw plane under each filter, with and without
/// the pre-processing flag, and decode both ways.
#[test]
fn every_alpha_filter_matches_dwebp() {
    let (Some(cwebp_tool), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let dir = scratch("alpha-filters");
    let (width, height) = (45u32, 29u32);
    let source = dir.join("source.png");
    let rgba = synthetic_rgba(width, height);
    write_png(&source, width, height, &rgba);
    let original = cwebp(&cwebp_tool, &source, &[], &dir.join("original.webp"));
    let vp8 = chunks(&original)
        .into_iter()
        .find(|(fourcc, _)| fourcc == b"VP8 ")
        .expect("vp8 chunk")
        .1;
    let plane: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| pixel[3])
        .collect();
    for (bits, kind) in [
        (0, Filter::None),
        (1, Filter::Horizontal),
        (2, Filter::Vertical),
        (3, Filter::Gradient),
    ] {
        for preprocessing in [0u8, 1] {
            let mut alph = vec![(bits << 2) | (preprocessing << 4)];
            alph.extend(filter(&plane, width as usize, kind));
            let data = file(&[
                (b"VP8X", &vp8x(0x10, width, height)),
                (b"ALPH", &alph),
                (b"VP8 ", &vp8),
            ]);
            let path = dir.join(format!("filter{bits}-{preprocessing}.webp"));
            std::fs::write(&path, &data).expect("write");
            check(&dwebp, &path);
            let ours = decode_still(&data);
            let alpha: Vec<u8> = ours
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| pixel[3])
                .collect();
            assert_eq!(alpha, plane, "{kind:?}");
        }
    }
}

/// An `ICCP` profile rides along on the picture; the colour type follows
/// `ALPH`.
#[test]
fn icc_profiles_pass_through_and_alpha_sets_the_colour_type() {
    let (Some(cwebp_tool), Some(webpmux)) = (tool("cwebp"), tool("webpmux")) else {
        require_or_skip("cwebp and webpmux");
        return;
    };
    let dir = scratch("icc");
    let profile_path = Path::new("/System/Library/ColorSync/Profiles/Display P3.icc");
    let profile = std::fs::read(profile_path).unwrap_or_else(|_| b"not a real profile".to_vec());
    let icc = dir.join("profile.icc");
    std::fs::write(&icc, &profile).expect("icc");
    let (width, height) = (31u32, 19u32);
    let mut opaque = synthetic_rgba(width, height);
    for pixel in opaque.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    for (label, rgba, args, color) in [
        ("lossy", &opaque, &[][..], ColorType::Rgb),
        ("lossless", &opaque, &["-lossless"][..], ColorType::Rgb),
        (
            "alpha",
            &synthetic_rgba(width, height),
            &[][..],
            ColorType::Rgba,
        ),
    ] {
        let source = dir.join(format!("{label}.png"));
        write_png(&source, width, height, rgba);
        let plain = dir.join(format!("{label}.webp"));
        cwebp(&cwebp_tool, &source, args, &plain);
        let tagged = dir.join(format!("{label}-icc.webp"));
        let status = Command::new(&webpmux)
            .arg("-set")
            .arg("icc")
            .arg(&icc)
            .arg(&plain)
            .arg("-o")
            .arg(&tagged)
            .output()
            .expect("webpmux runs");
        assert!(status.status.success());
        let data = std::fs::read(&tagged).expect("tagged");
        let mut decoder =
            WebpDecoder::open(&CodecParams::video(CodecId::Webp, width, height)).expect("open");
        let outputs = decoder
            .decode_picture(&Packet::video(data, 0, 1, 1000, true))
            .expect("decode");
        let image = &outputs[0].image;
        assert_eq!(image.icc_profile, profile, "{label}");
        assert_eq!(image.color, color, "{label}");
        assert_eq!((image.width, image.height), (width, height));
    }
}

#[test]
fn a_canvas_that_differs_from_the_bitstream_is_refused() {
    let Some(cwebp_tool) = tool("cwebp") else {
        require_or_skip("cwebp");
        return;
    };
    let dir = scratch("canvas");
    let source = dir.join("source.png");
    write_png(&source, 8, 8, &synthetic_rgba(8, 8));
    let original = cwebp(&cwebp_tool, &source, &[], &dir.join("a.webp"));
    let parts = chunks(&original);
    let alph = &parts.iter().find(|(f, _)| f == b"ALPH").expect("alph").1;
    let vp8 = &parts.iter().find(|(f, _)| f == b"VP8 ").expect("vp8").1;
    let wrong = file(&[
        (b"VP8X", &vp8x(0x10, 9, 8)),
        (b"ALPH", alph),
        (b"VP8 ", vp8),
    ]);
    let mut decoder = WebpDecoder::open(&CodecParams::video(CodecId::Webp, 8, 8)).expect("open");
    assert!(
        decoder
            .decode_picture(&Packet::video(wrong, 0, 1, 1000, true))
            .is_err()
    );
}
