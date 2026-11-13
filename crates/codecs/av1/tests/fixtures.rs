//! Every fixture stream through the public decoder, against libdav1d.
//!
//! `libdav1d-digests.txt` holds one line per frame libdav1d shows:
//! the stream, the frame index, its size and pixel format, and the
//! FNV-1a 64 digest of its raw samples. The ignored test
//! `write_libdav1d_digests` writes it from the local `ffmpeg` with
//! libdav1d; the README names the command. The check itself needs no
//! FFmpeg, so it runs in CI.
//!
//! A grey stream comes out of libdav1d as one plane, and out of this
//! decoder as 4:2:0 with neutral chroma: its digest covers luma only, and
//! the chroma is checked to be neutral.

// Clippy allows `expect` and `panic` in `cfg(test)` code; the helpers
// live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;

#[cfg(test)]
mod fixtures {
    use super::common;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::process::Command;

    use common::fnv1a;
    use cutlass_decode::{CodecError, PictureSamples, PixelFormat};

    const DIGESTS: &str = include_str!("libdav1d-digests.txt");

    /// Past level 6.3 (16448 wide). libdav1d refuses it too.
    const REFUSED: &str = "wide.ivf";

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Frame {
        width: u32,
        height: u32,
        pix_fmt: String,
        digest: u64,
    }

    fn references() -> BTreeMap<String, Vec<Frame>> {
        let mut out: BTreeMap<String, Vec<Frame>> = BTreeMap::new();
        for line in DIGESTS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 5, "digest line: {line}");
            let frames = out.entry(fields[0].to_string()).or_default();
            let index: usize = fields[1].parse().expect("index");
            assert_eq!(index, frames.len(), "frames of {} in order", fields[0]);
            let (width, height) = fields[2].split_once('x').expect("size");
            frames.push(Frame {
                width: width.parse().expect("width"),
                height: height.parse().expect("height"),
                pix_fmt: fields[3].to_string(),
                digest: u64::from_str_radix(fields[4], 16).expect("digest"),
            });
        }
        out
    }

    /// The pixel format this decoder publishes for libdav1d's `pix_fmt`.
    fn published(pix_fmt: &str) -> PixelFormat {
        match pix_fmt {
            "yuv420p" | "gray" => PixelFormat::Yuv420p,
            "yuv422p" => PixelFormat::Yuv422p,
            "yuv444p" => PixelFormat::Yuv444p,
            "gbrp" => PixelFormat::Gbrp,
            "yuv420p10le" | "gray10le" => PixelFormat::Yuv420p10,
            "yuv422p10le" => PixelFormat::Yuv422p10,
            "yuv444p10le" => PixelFormat::Yuv444p10,
            "gbrp10le" => PixelFormat::Gbrp10,
            "yuv420p12le" | "gray12le" => PixelFormat::Yuv420p12,
            "yuv422p12le" => PixelFormat::Yuv422p12,
            "yuv444p12le" => PixelFormat::Yuv444p12,
            "gbrp12le" => PixelFormat::Gbrp12,
            other => panic!("pix_fmt {other} has no mapping"),
        }
    }

    /// Bytes of one raw frame as FFmpeg writes it.
    fn raw_len(pix_fmt: &str, width: u32, height: u32) -> usize {
        let (w, h) = (width as usize, height as usize);
        let bytes = if pix_fmt.ends_with("le") { 2 } else { 1 };
        let chroma = if pix_fmt.starts_with("gray") {
            0
        } else if pix_fmt.starts_with("yuv420") {
            2 * w.div_ceil(2) * h.div_ceil(2)
        } else if pix_fmt.starts_with("yuv422") {
            2 * w.div_ceil(2) * h
        } else {
            2 * w * h
        };
        (w * h + chroma) * bytes
    }

    #[test]
    fn every_fixture_stream_matches_libdav1d() {
        let references = references();
        let streams = common::streams();
        let mut names: Vec<&String> = streams.keys().filter(|n| *n != REFUSED).collect();
        names.sort();
        let referenced: Vec<&String> = references.keys().collect();
        assert_eq!(
            names, referenced,
            "every stream has references, and only those"
        );
        let mut frames = 0;
        // On the default threads, then on the calling thread alone, which
        // decodes and filters a frame its own way.
        for (name, want, threads) in references
            .iter()
            .flat_map(|(n, w)| [(n, w, None), (n, w, Some(1))])
        {
            let first = &want[0];
            let pictures = common::decode_on(&streams[name], first.width, first.height, threads)
                .unwrap_or_else(|error| panic!("{name} ({threads:?} threads): {error:?}"));
            assert_eq!(pictures.len(), want.len(), "{name}: frames shown");
            for (index, (picture, want)) in pictures.iter().zip(want).enumerate() {
                let info = picture.info();
                assert_eq!(
                    (info.width, info.height),
                    (want.width, want.height),
                    "{name} frame {index}: size"
                );
                assert_eq!(info.pixel, published(&want.pix_fmt), "{name} frame {index}");
                let PictureSamples::Cpu(bytes) = picture.samples() else {
                    panic!("{name} frame {index}: not on the CPU");
                };
                let compared = if want.pix_fmt.starts_with("gray") {
                    let luma = raw_len(&want.pix_fmt, want.width, want.height);
                    let wide = want.pix_fmt.ends_with("le");
                    let neutral: u16 = match want.pix_fmt.as_str() {
                        "gray" => 128,
                        "gray10le" => 512,
                        _ => 2048,
                    };
                    let sample = if wide { 2 } else { 1 };
                    let chroma_len = 2 * want.width.div_ceil(2) * want.height.div_ceil(2);
                    assert_eq!(
                        bytes.len(),
                        luma + chroma_len as usize * sample,
                        "{name} frame {index}: bytes"
                    );
                    let chroma = &bytes[luma..];
                    if wide {
                        assert!(
                            chroma
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .all(|s| u16::from_le_bytes(*s) == neutral),
                            "{name} frame {index}: neutral chroma"
                        );
                    } else {
                        assert!(
                            chroma.iter().all(|&s| u16::from(s) == neutral),
                            "{name} frame {index}: neutral chroma"
                        );
                    }
                    &bytes[..luma]
                } else {
                    assert_eq!(
                        bytes.len(),
                        raw_len(&want.pix_fmt, want.width, want.height),
                        "{name} frame {index}: bytes"
                    );
                    bytes
                };
                assert_eq!(
                    fnv1a(compared),
                    want.digest,
                    "{name} frame {index} ({threads:?} threads): samples differ from libdav1d"
                );
                frames += 1;
            }
        }
        assert!(frames >= references.len());
    }

    #[test]
    fn a_picture_past_every_defined_level_is_refused() {
        // 16448×64 is wider than level 6.3 allows. The sequence header is
        // refused before any frame buffer is allocated.
        let packets = common::stream(REFUSED);
        let error = common::decode(&packets, 64, 64).expect_err("past every defined level");
        assert!(
            matches!(error, CodecError::Limit { what: "frame size" }),
            "{error:?}"
        );
    }

    /// Writes `libdav1d-digests.txt` from `ffmpeg -c:v libdav1d -threads 1`.
    #[test]
    #[ignore = "needs ffmpeg with libdav1d; rewrites the committed references"]
    fn write_libdav1d_digests() {
        let scratch =
            std::env::temp_dir().join(format!("cutlass-av1-digests-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).expect("scratch");
        let mut text = String::from(
            "# libdav1d's frames of every fixture stream except wide.ivf (refused).\n\
             # stream index WIDTHxHEIGHT pix_fmt fnv1a64-of-raw-samples\n\
             # Written by `cargo test --release -p cutlass-av1 --test fixtures -- --ignored write_libdav1d_digests`:\n\
             # ffmpeg -threads 1 [-f obu] -c:v libdav1d -threads 1 -i STREAM -noautoscale -fps_mode passthrough -f rawvideo -\n",
        );
        let version = Command::new("ffmpeg")
            .arg("-version")
            .output()
            .expect("ffmpeg");
        let version = String::from_utf8_lossy(&version.stdout);
        let first = version.lines().next().unwrap_or("");
        text.push_str(&format!("# {first}\n"));
        if let Ok(dav1d) = Command::new("pkg-config")
            .args(["--modversion", "dav1d"])
            .output()
            && dav1d.status.success()
        {
            let dav1d = String::from_utf8_lossy(&dav1d.stdout);
            text.push_str(&format!("# libdav1d {}\n", dav1d.trim()));
        }
        for (name, packets) in common::streams() {
            if name == REFUSED {
                continue;
            }
            let obu = name.ends_with(".obu");
            let input = if obu {
                let joined = scratch.join(&name);
                std::fs::write(&joined, packets.concat()).expect("write");
                joined
            } else {
                common::fixtures().join(&name)
            };
            for (index, frame) in libdav1d_frames(&input, obu).iter().enumerate() {
                text.push_str(&format!(
                    "{name} {index} {}x{} {} {:016x}\n",
                    frame.width, frame.height, frame.pix_fmt, frame.digest
                ));
            }
        }
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/libdav1d-digests.txt");
        std::fs::write(out, text).expect("write digests");
        std::fs::remove_dir_all(&scratch).expect("remove scratch");
    }

    fn libdav1d_frames(input: &Path, obu: bool) -> Vec<Frame> {
        let demux: &[&str] = if obu { &["-f", "obu"] } else { &[] };
        let probe = Command::new("ffprobe")
            .args(["-v", "error"])
            .args(demux)
            .args(["-c:v", "libdav1d", "-select_streams", "v:0", "-show_frames"])
            .args([
                "-show_entries",
                "frame=width,height,pix_fmt",
                "-of",
                "csv=p=0",
            ])
            .arg(input)
            .output()
            .expect("ffprobe");
        assert!(probe.status.success(), "ffprobe {input:?}");
        let sizes: Vec<(u32, u32, String)> = String::from_utf8_lossy(&probe.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let parts: Vec<&str> = line.trim().split(',').collect();
                (
                    parts[0].parse().expect("width"),
                    parts[1].parse().expect("height"),
                    parts[2].to_string(),
                )
            })
            .collect();
        let run = Command::new("ffmpeg")
            .args(["-v", "error", "-threads", "1"])
            .args(demux)
            .args(["-c:v", "libdav1d", "-threads", "1", "-i"])
            .arg(input)
            .args([
                "-noautoscale",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-",
            ])
            .output()
            .expect("ffmpeg");
        assert!(
            run.status.success(),
            "ffmpeg {input:?}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let mut at = 0;
        let frames = sizes
            .into_iter()
            .map(|(width, height, pix_fmt)| {
                let len = raw_len(&pix_fmt, width, height);
                let digest = fnv1a(&run.stdout[at..at + len]);
                at += len;
                Frame {
                    width,
                    height,
                    pix_fmt,
                    digest,
                }
            })
            .collect();
        assert_eq!(
            at,
            run.stdout.len(),
            "{input:?}: frame sizes cover the output"
        );
        frames
    }
}
