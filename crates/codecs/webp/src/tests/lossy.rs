//! Lossy stills against `dwebp`, its YUV planes, and ffmpeg's decoder.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::oracle::{
    cwebp, decode_still, dwebp_rgba, dwebp_yuv, ffmpeg_yuv, first_difference, local_files,
    require_or_skip, scratch, tool, video_frame, write_png,
};
use crate::lossy;
use crate::riff::{self, Bitstream};

/// Opaque synthetic RGBA: gradients, hard edges, and a noisy patch.
fn opaque_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    let mut seed = 0x1234_5678u32;
    for y in 0..height {
        for x in 0..width {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noisy = x > width / 2 && y > height / 2;
            let red = if noisy {
                (seed >> 24) as u8
            } else {
                (x * 255 / width.max(1)) as u8
            };
            let green = (y * 255 / height.max(1)) as u8;
            let blue = if (x / 7 + y / 5) % 2 == 0 { 30 } else { 220 };
            rgba.extend_from_slice(&[red, green, blue, 255]);
        }
    }
    rgba
}

/// The cropped, packed Y, U and V planes of this crate's decode.
fn our_yuv(bytes: &[u8]) -> LossyPlanes {
    lossy_planes(bytes).expect("lossy")
}

/// Packed Y, U and V, and the alpha plane when there is one.
pub(crate) struct LossyPlanes {
    pub yuv: Vec<u8>,
    pub alpha: Vec<u8>,
}

impl LossyPlanes {
    /// The layout of `dwebp -yuv`: Y, U, V, then alpha.
    pub(crate) fn yuva(&self) -> Vec<u8> {
        [&self.yuv[..], &self.alpha[..]].concat()
    }
}

/// The planes of a lossy still, `None` for a lossless one.
pub(crate) fn lossy_planes(bytes: &[u8]) -> Option<LossyPlanes> {
    let riff::Layout::Still(still) = riff::parse(bytes).expect("riff") else {
        return None;
    };
    let Bitstream::Lossy { vp8, alpha } = still.bitstream else {
        return None;
    };
    let planes = lossy::decode(vp8).expect("decode");
    let (w, h) = (planes.width, planes.height);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for row in 0..h {
        out.extend_from_slice(&planes.y[row * planes.y_stride..][..w]);
    }
    for plane in [&planes.u, &planes.v] {
        for row in 0..ch {
            out.extend_from_slice(&plane[row * planes.uv_stride..][..cw]);
        }
    }
    let alpha = alpha.map_or_else(Vec::new, |chunk| {
        crate::alpha::decode(chunk, w as u32, h as u32).expect("alpha")
    });
    Some(LossyPlanes { yuv: out, alpha })
}

fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for (width, height) in [(1, 1), (3, 7), (17, 33), (97, 61), (16, 16)] {
        let path = dir.join(format!("opaque{width}x{height}.png"));
        write_png(&path, width, height, &opaque_rgba(width, height));
        sources.push(path);
    }
    if let (Some(local), Some(ffmpeg)) = (local_files(), tool("ffmpeg"))
        && let Ok(entries) = std::fs::read_dir(&local)
    {
        let video = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
            .min();
        if let Some(video) = video {
            let frame = dir.join("frame.png");
            if video_frame(&ffmpeg, &video, &frame, "scale=333:-1") {
                sources.push(frame);
            }
        }
    }
    sources
}

/// Every plane and every pixel equal to libwebp's, and the planes equal to
/// ffmpeg's independent decoder where this machine has one.
fn check(dwebp: &Path, ffmpeg: Option<&Path>, path: &Path) {
    let bytes = std::fs::read(path).expect("read");
    let ours = decode_still(&bytes);
    let (width, height, theirs) = dwebp_rgba(dwebp, path);
    assert_eq!((ours.width, ours.height), (width, height));
    assert_eq!(
        first_difference(&ours.rgba, &theirs, width),
        None,
        "{} against dwebp",
        path.display()
    );
    let planes = our_yuv(&bytes);
    assert!(
        planes.yuva() == dwebp_yuv(dwebp, path),
        "{} planes against dwebp",
        path.display()
    );
    if let Some(ffmpeg) = ffmpeg
        && let Some(theirs) = ffmpeg_yuv(ffmpeg, path)
    {
        assert!(
            planes.yuv == theirs,
            "{} planes against ffmpeg",
            path.display()
        );
    }
}

#[test]
fn cwebp_lossy_files_decode_to_the_exact_pixels_of_dwebp_and_ffmpeg() {
    let (Some(cwebp_tool), Some(dwebp)) = (tool("cwebp"), tool("dwebp")) else {
        require_or_skip("cwebp and dwebp");
        return;
    };
    let ffmpeg = tool("ffmpeg");
    let dir = scratch("lossy");
    for source in sources(&dir) {
        for (label, args) in [
            ("q10m0", &["-q", "10", "-m", "0"][..]),
            ("q50m2", &["-q", "50", "-m", "2"][..]),
            ("q75", &["-q", "75"][..]),
            ("q95m6", &["-q", "95", "-m", "6"][..]),
            ("q100", &["-q", "100"][..]),
            ("f0", &["-f", "0"][..]),
            ("sharp7", &["-sharpness", "7", "-f", "80"][..]),
            ("simple", &["-nostrong", "-f", "70"][..]),
            ("seg1", &["-segments", "1", "-sns", "0"][..]),
        ] {
            let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
            let out = dir.join(format!("{stem}-{label}.webp"));
            cwebp(&cwebp_tool, &source, args, &out);
            check(&dwebp, ffmpeg.as_deref(), &out);
        }
    }
}

/// libvpx writes what libwebp never does: several token partitions and
/// loop-filter deltas. Its key frames, wrapped as simple WebP files, must
/// decode like libwebp decodes them.
#[test]
fn libvpx_key_frames_with_partitions_and_filter_deltas_match_dwebp() {
    let (Some(ffmpeg), Some(dwebp)) = (tool("ffmpeg"), tool("dwebp")) else {
        require_or_skip("ffmpeg and dwebp");
        return;
    };
    let dir = scratch("libvpx");
    let mut made = 0;
    for source in sources(&dir) {
        for (label, args) in [
            ("one", &["-slices", "1", "-crf", "10", "-b:v", "2M"][..]),
            ("eight", &["-slices", "8", "-crf", "30", "-b:v", "1M"][..]),
            (
                "sharp",
                &[
                    "-slices",
                    "4",
                    "-sharpness",
                    "5",
                    "-crf",
                    "40",
                    "-b:v",
                    "300k",
                ][..],
            ),
            ("low", &["-slices", "2", "-crf", "63", "-b:v", "50k"][..]),
        ] {
            let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
            let ivf = dir.join(format!("{stem}-{label}.ivf"));
            let ok = Command::new(&ffmpeg)
                .args(["-v", "error", "-y", "-i"])
                .arg(&source)
                .args(["-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", "libvpx"])
                .args(args)
                .args(["-f", "ivf"])
                .arg(&ivf)
                .status()
                .is_ok_and(|status| status.success());
            if !ok {
                // This ffmpeg has no libvpx.
                require_or_skip("ffmpeg with libvpx");
                return;
            }
            let data = std::fs::read(&ivf).expect("ivf");
            let header = usize::from(u16::from_le_bytes([data[6], data[7]]));
            let size =
                u32::from_le_bytes(data[header..header + 4].try_into().expect("size")) as usize;
            let frame = &data[header + 12..header + 12 + size];
            let mut body = b"VP8 ".to_vec();
            body.extend_from_slice(&(frame.len() as u32).to_le_bytes());
            body.extend_from_slice(frame);
            if frame.len() % 2 == 1 {
                body.push(0);
            }
            let mut file = b"RIFF".to_vec();
            file.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
            file.extend_from_slice(b"WEBP");
            file.extend_from_slice(&body);
            let out = ivf.with_extension("webp");
            std::fs::write(&out, &file).expect("write");
            check(&dwebp, None, &out);
            made += 1;
        }
    }
    assert!(made > 0);
}
