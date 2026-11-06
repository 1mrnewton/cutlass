//! Animations, packet by packet as the `still` container splits them,
//! against libwebp's animation decoder through Pillow.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_decode::{CodecId, CodecParams, ColorType, Decoder, Packet};

use super::oracle::{cwebp, local_files, require_or_skip, scratch, tool, write_png};
use crate::WebpDecoder;

/// The `still` container's split: the file start through the first `ANMF`,
/// then each `ANMF`, the last running to the end of the file.
pub(crate) fn split(file: &[u8]) -> Vec<Vec<u8>> {
    let mut ends = Vec::new();
    let mut at = 12;
    while at + 8 <= file.len() {
        let size = u32::from_le_bytes(file[at + 4..at + 8].try_into().expect("size")) as usize;
        let next = at + 8 + size + (size & 1);
        if &file[at..at + 4] == b"ANMF" {
            ends.push(next);
        }
        at = next;
    }
    let mut packets = Vec::new();
    let mut start = 0;
    for (index, &end) in ends.iter().enumerate() {
        let end = if index + 1 == ends.len() {
            file.len()
        } else {
            end
        };
        packets.push(file[start..end].to_vec());
        start = end;
    }
    packets
}

/// Every composed canvas, one packet at a time, as straight RGBA.
pub(crate) fn decode_packets(decoder: &mut WebpDecoder, packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .enumerate()
        .map(|(index, data)| {
            let outputs = decoder
                .decode_picture(&Packet::video(
                    data.clone(),
                    index as i64 * 100,
                    100,
                    1000,
                    index == 0,
                ))
                .expect("frame");
            assert_eq!(outputs.len(), 1);
            let image = &outputs[0].image;
            assert_eq!(image.color, ColorType::Rgba);
            image.samples.iter().map(|&s| s as u8).collect()
        })
        .collect()
}

fn open() -> WebpDecoder {
    WebpDecoder::open(&CodecParams::video(CodecId::Webp, 1, 1)).expect("open")
}

const PILLOW_DUMP: &str = r#"
import sys
from PIL import Image
image = Image.open(sys.argv[1])
with open(sys.argv[2], "wb") as out:
    for index in range(image.n_frames):
        image.seek(index)
        out.write(b"A" if image.mode == "RGBA" else b"X")
        out.write(image.convert("RGBA").tobytes())
"#;

/// libwebp's composed frames through Pillow: each with whether Pillow kept
/// alpha (it drops it when `VP8X` has no alpha flag).
fn pillow_frames(dir: &Path, webp: &Path, frame_bytes: usize) -> Option<Vec<(bool, Vec<u8>)>> {
    let script = dir.join("dump.py");
    std::fs::write(&script, PILLOW_DUMP).ok()?;
    let out = webp.with_extension("frames");
    let status = Command::new("python3")
        .arg(&script)
        .arg(webp)
        .arg(&out)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    let bytes = std::fs::read(&out).ok()?;
    Some(
        bytes
            .chunks(frame_bytes + 1)
            .map(|frame| (frame[0] == b'A', frame[1..].to_vec()))
            .collect(),
    )
}

fn canvas_size(file: &[u8]) -> (usize, usize) {
    let u24 = |at: usize| {
        usize::from(file[at]) | (usize::from(file[at + 1]) << 8) | (usize::from(file[at + 2]) << 16)
    };
    (u24(24) + 1, u24(27) + 1)
}

/// Frames of a disc that moves over a gradient and fades, with holes.
fn moving_frames(dir: &Path, count: u32, width: u32, height: u32) -> Vec<PathBuf> {
    (0..count)
        .map(|index| {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height {
                for x in 0..width {
                    let cx = (index * 7 + 10) as i32;
                    let dx = x as i32 - cx;
                    let dy = y as i32 - (height / 2) as i32;
                    let inside = dx * dx + dy * dy < 150;
                    let hole = (x / 9 + y / 7 + index) % 5 == 0;
                    let alpha = if inside {
                        255
                    } else if hole {
                        0
                    } else {
                        (40 + x * 3 + index * 20).min(255) as u8 as u32
                    };
                    rgba.extend_from_slice(&[
                        (x * 4) as u8,
                        if inside { 250 } else { (y * 5) as u8 },
                        (index * 40) as u8,
                        alpha as u8,
                    ]);
                }
            }
            let path = dir.join(format!("moving{index}.png"));
            write_png(&path, width, height, &rgba);
            path
        })
        .collect()
}

fn run(tool: &Path, args: &[&std::ffi::OsStr]) -> bool {
    Command::new(tool)
        .args(args)
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Animations from libwebp's own tools: `img2webp` lossy, lossless and
/// mixed, `gif2webp`, and `webpmux` with offsets, disposal and blending.
fn animations(dir: &Path) -> Vec<PathBuf> {
    let mut made = Vec::new();
    let moving = moving_frames(dir, 6, 61, 37);
    if let Some(img2webp) = tool("img2webp") {
        for (label, options) in [
            ("lossless", &["-lossless"][..]),
            ("lossy", &["-lossy", "-q", "60"][..]),
            ("mixed", &["-mixed"][..]),
            ("keys", &["-lossy", "-q", "40"][..]),
        ] {
            let out = dir.join(format!("img2webp-{label}.webp"));
            let mut args: Vec<std::ffi::OsString> = Vec::new();
            if label == "mixed" {
                args.push("-mixed".into());
            }
            if label == "keys" {
                args.extend(["-kmin".into(), "1".into(), "-kmax".into(), "2".into()]);
            }
            args.extend(["-loop".into(), "3".into()]);
            for (index, frame) in moving.iter().enumerate() {
                if label != "mixed" {
                    args.extend(options.iter().map(Into::into));
                }
                args.extend([
                    "-d".into(),
                    (if index == 2 { 10 } else { 70 }).to_string().into(),
                ]);
                args.push(frame.into());
            }
            args.extend(["-o".into(), out.clone().into()]);
            let args: Vec<&std::ffi::OsStr> = args.iter().map(|a| a.as_os_str()).collect();
            if run(&img2webp, &args) {
                made.push(out);
            }
        }
    }
    if let (Some(ffmpeg), Some(gif2webp)) = (tool("ffmpeg"), tool("gif2webp")) {
        let mut sources = vec![dir.join("moving%d.png")];
        if let Some(local) = local_files()
            && let Ok(entries) = std::fs::read_dir(&local)
            && let Some(video) = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "mp4"))
                .min()
        {
            sources.push(video);
        }
        for (index, source) in sources.iter().enumerate() {
            let gif = dir.join(format!("source{index}.gif"));
            let ok = Command::new(&ffmpeg)
                .args(["-v", "error", "-y", "-i"])
                .arg(source)
                .args(["-frames:v", "8", "-vf", "scale=120:-2,fps=8", "-loop", "0"])
                .arg(&gif)
                .status()
                .is_ok_and(|status| status.success());
            if !ok {
                continue;
            }
            for (label, options) in [
                ("plain", &[][..]),
                ("lossy", &["-lossy", "-q", "50"][..]),
                ("mixed", &["-mixed"][..]),
            ] {
                let out = dir.join(format!("gif{index}-{label}.webp"));
                let mut args: Vec<&std::ffi::OsStr> =
                    options.iter().map(std::ffi::OsStr::new).collect();
                args.push(gif.as_os_str());
                args.push(std::ffi::OsStr::new("-o"));
                args.push(out.as_os_str());
                if run(&gif2webp, &args) {
                    made.push(out);
                }
            }
        }
    }
    if let (Some(cwebp_tool), Some(webpmux)) = (tool("cwebp"), tool("webpmux")) {
        // Frames smaller than the canvas, at even offsets, mixing every
        // disposal and blend, lossy and lossless, with and without alpha.
        let small = moving_frames(dir, 4, 24, 18);
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        for (index, png) in small.iter().enumerate() {
            let webp = dir.join(format!("small{index}.webp"));
            let options: &[&str] = if index % 2 == 0 {
                &["-lossless"]
            } else {
                &["-q", "70"]
            };
            cwebp(&cwebp_tool, png, options, &webp);
            let (x, y) = [(0, 0), (10, 6), (36, 18), (20, 2)][index];
            let dispose = index % 2;
            let blend = if index == 2 { "-b" } else { "+b" };
            args.extend([
                "-frame".into(),
                webp.into(),
                format!("+{}+{x}+{y}+{dispose}{blend}", [80, 0, 100, 110][index]).into(),
            ]);
        }
        let opaque = dir.join("opaque.png");
        let mut rgba = Vec::new();
        for i in 0..60 * 36 {
            rgba.extend_from_slice(&[(i % 251) as u8, (i % 13 * 19) as u8, 90, 255]);
        }
        write_png(&opaque, 60, 36, &rgba);
        let opaque_webp = dir.join("opaque.webp");
        cwebp(&cwebp_tool, &opaque, &["-q", "80"], &opaque_webp);
        args.extend(["-frame".into(), opaque_webp.into(), "+50+0+0+1+b".into()]);
        let small0 = dir.join("small0.webp");
        args.extend(["-frame".into(), small0.into(), "+60+30+12+0+b".into()]);
        let out = dir.join("webpmux.webp");
        args.extend([
            "-loop".into(),
            "2".into(),
            "-bgcolor".into(),
            "255,255,255,255".into(),
        ]);
        args.extend(["-o".into(), out.clone().into()]);
        let args: Vec<&std::ffi::OsStr> = args.iter().map(|a| a.as_os_str()).collect();
        if run(&webpmux, &args) {
            made.push(out);
        }
    }
    made
}

#[test]
fn animations_compose_like_libwebp_packet_by_packet() {
    let dir = scratch("animation");
    let files = animations(&dir);
    if files.len() < 8 {
        require_or_skip("img2webp, gif2webp, webpmux and ffmpeg");
    }
    check_animations(&dir, &files);
}

/// Every animation under `WEBP_CORPUS_DIR`.
#[test]
#[ignore = "needs WEBP_CORPUS_DIR"]
fn corpus_animations_match_libwebp() {
    let Some(corpus) = std::env::var_os("WEBP_CORPUS_DIR") else {
        return;
    };
    let files: Vec<PathBuf> = super::oracle::webp_files(Path::new(&corpus))
        .into_iter()
        .filter(|path| {
            std::fs::read(path).is_ok_and(|data| {
                data.len() > 30 && data[20] & 0x02 != 0 && &data[12..16] == b"VP8X"
            })
        })
        .collect();
    assert!(!files.is_empty(), "no animations");
    check_animations(&scratch("anim-corpus"), &files);
}

pub(super) fn check_animations(dir: &Path, files: &[PathBuf]) {
    let mut compared = 0;
    for path in files {
        let file = std::fs::read(path).expect("read");
        let packets = split(&file);
        let (width, height) = canvas_size(&file);
        let mut decoder = open();
        let ours = decode_packets(&mut decoder, &packets);
        assert_eq!(ours.len(), packets.len());

        // The same file as one packet gives every canvas, timed in ms.
        let whole = open()
            .decode_picture(&Packet::video(file.clone(), 0, 0, 1000, true))
            .expect("whole file");
        assert_eq!(whole.len(), ours.len());
        for (output, canvas) in whole.iter().zip(&ours) {
            let samples: Vec<u8> = output.image.samples.iter().map(|&s| s as u8).collect();
            assert!(&samples == canvas, "{} whole-file decode", path.display());
        }
        assert_eq!(whole[0].pts, 0);
        for (output, next) in whole.iter().zip(&whole[1..]) {
            assert_eq!(output.timescale, 1000);
            assert!(output.duration > 0);
            assert_eq!(output.pts + output.duration, next.pts);
        }

        // A reset and replay, as a seek does, shows the same canvases.
        decoder.reset();
        assert!(
            decode_packets(&mut decoder, &packets) == ours,
            "{} replay",
            path.display()
        );

        if let Some(theirs) = pillow_frames(dir, path, width * height * 4) {
            assert_eq!(theirs.len(), ours.len(), "{}", path.display());
            for (index, ((alpha, theirs), ours)) in theirs.iter().zip(&ours).enumerate() {
                for (pixel, (a, b)) in ours
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(theirs.as_chunks::<4>().0)
                    .enumerate()
                {
                    let same = if *alpha { a == b } else { a[..3] == b[..3] };
                    assert!(
                        same,
                        "{} frame {index} pixel ({}, {}): ours {a:?}, libwebp {b:?}",
                        path.display(),
                        pixel % width,
                        pixel / width
                    );
                }
            }
            compared += 1;
        }
    }
    if compared == 0 {
        require_or_skip("python3 with Pillow");
    }
    assert!(
        compared == 0 || compared == files.len(),
        "{compared} of {}",
        files.len()
    );
}

#[test]
fn a_frame_before_the_first_packet_or_outside_the_canvas_is_refused() {
    let Some(cwebp_tool) = tool("cwebp") else {
        require_or_skip("cwebp");
        return;
    };
    let dir = scratch("anim-bad");
    let png = dir.join("p.png");
    write_png(&png, 8, 6, &[200; 8 * 6 * 4]);
    let webp = cwebp(&cwebp_tool, &png, &["-lossless"], &dir.join("p.webp"));
    let vp8l = crate::riff::chunks(&webp).remove(0).1;
    let anmf = |x: u8| {
        let mut payload = vec![x, 0, 0, 0, 0, 0, 7, 0, 0, 5, 0, 0, 100, 0, 0, 0];
        crate::riff::put_chunk(&mut payload, b"VP8L", &vp8l);
        payload
    };
    let good = crate::riff::file(&[
        (b"VP8X", &crate::riff::vp8x(0x12, 10, 6)),
        (b"ANIM", &[0, 0, 0, 0, 0, 0]),
        (b"ANMF", &anmf(0)),
        (b"ANMF", &anmf(2)),
    ]);
    let packets = split(&good);
    assert_eq!(packets.len(), 2);
    // A later packet with no canvas before it.
    let mut decoder = open();
    assert!(
        decoder
            .decode_picture(&Packet::video(packets[1].clone(), 0, 1, 1000, false))
            .is_err()
    );
    // The first frame fits; the second starts at x = 4 and runs past 10.
    let first = decoder
        .decode_picture(&Packet::video(packets[0].clone(), 0, 1, 1000, true))
        .expect("first");
    assert!(
        decoder
            .decode_picture(&Packet::video(packets[1].clone(), 1, 1, 1000, false))
            .is_err()
    );
    // The failed packet left the canvas as it was.
    let fixed = crate::riff::file(&[
        (b"VP8X", &crate::riff::vp8x(0x12, 10, 6)),
        (b"ANIM", &[0, 0, 0, 0, 0, 0]),
        (b"ANMF", &anmf(0)),
        (b"ANMF", &anmf(0)),
    ]);
    let second = Packet::video(split(&fixed)[1].clone(), 1, 1, 1000, false);
    let after_failure = decoder.decode_picture(&second).expect("second");
    let mut clean = open();
    clean
        .decode_picture(&Packet::video(packets[0].clone(), 0, 1, 1000, true))
        .expect("first");
    let expected = clean.decode_picture(&second).expect("second");
    assert_eq!(after_failure[0].image.samples, expected[0].image.samples);
    assert_ne!(after_failure[0].image.samples, first[0].image.samples);
}
