//! Bit-exact decodes against FFmpeg's reference output.
//!
//! The corpus lives beside the checkouts, at `../mpeg4-part2-corpus` from
//! the workspace root (or `MPEG4_CORPUS`). Each `gen/*.m4v` or `*.h263`
//! elementary stream has a `*.yuv` FFmpeg 8.1.1 decode beside it, made
//! with `-cpuflags 0 -flags +bitexact -idct simple -fps_mode passthrough`.
//! The test skips when the folder is missing, as it is in CI.

use std::path::{Path, PathBuf};

use cutlass_decode::{CodecId, CodecParams, Decoder, Output, Packet, PictureSamples};
use cutlass_mpeg4::EngineDecoder;

fn corpus() -> Option<PathBuf> {
    let dir = std::env::var_os("MPEG4_CORPUS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../mpeg4-part2-corpus"),
        PathBuf::from,
    );
    dir.is_dir().then_some(dir)
}

/// Splits an elementary stream into one packet per picture: each VOP with
/// the headers before it, or each short-header picture.
fn split(data: &[u8]) -> Vec<Vec<u8>> {
    let short = matches!(data, [0, 0, b, ..] if b & 0xFC == 0x80);
    let mut cuts = Vec::new();
    let mut pending: Option<usize> = None;
    let mut i = 0;
    while i + 3 < data.len() {
        let is_start = data[i] == 0 && data[i + 1] == 0;
        if short && is_start && data[i + 2] & 0xFC == 0x80 {
            cuts.push(i);
            i += 3;
            continue;
        }
        if !short && is_start && data[i + 2] == 1 {
            let code = data[i + 3];
            if code == 0xB6 {
                cuts.push(pending.take().unwrap_or(i));
            } else if pending.is_none() {
                pending = Some(i);
            }
            i += 4;
            continue;
        }
        i += 1;
    }
    let mut out = Vec::new();
    for (n, &start) in cuts.iter().enumerate() {
        let end = cuts.get(n + 1).copied().unwrap_or(data.len());
        out.push(data[start..end].to_vec());
    }
    out
}

struct Result {
    name: String,
    ours: usize,
    reference: usize,
    exact: usize,
    first_mismatch: Option<String>,
    error: Option<String>,
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 * a.len() as f64 / se).log10()
}

fn run(stream: &Path, reference: &Path) -> Result {
    let name = stream
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let data = std::fs::read(stream).expect("stream");
    let want = std::fs::read(reference).expect("reference");
    let mut decoder = EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0)).expect("open");
    let mut pictures: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let mut error = None;
    let mut keep = |out: Vec<Output>| {
        for o in out {
            if let Output::Picture(p) = o {
                let info = p.info();
                let PictureSamples::Cpu(bytes) = p.samples() else {
                    panic!("platform buffer");
                };
                pictures.push((info.width, info.height, bytes.to_vec()));
            }
        }
    };
    for (n, data) in split(&data).into_iter().enumerate() {
        let packet = Packet {
            stream: 0,
            data,
            pts: n as i64,
            duration: 1,
            timescale: 25,
            keyframe: false,
            side: Vec::new(),
        };
        match decoder.push(&packet) {
            Ok(out) => keep(out),
            Err(e) => {
                error = Some(format!("packet {n}: {e}"));
                break;
            }
        }
    }
    if error.is_none() {
        keep(decoder.drain().expect("drain"));
    }
    if let Some(dump) = std::env::var_os("MPEG4_DUMP") {
        let bytes: Vec<u8> = pictures.iter().flat_map(|p| p.2.iter().copied()).collect();
        std::fs::write(Path::new(&dump).join(format!("{name}.yuv")), bytes).expect("dump");
    }
    let mut offset = 0;
    let mut exact = 0;
    let mut reference_frames = 0;
    let mut first_mismatch = None;
    for (index, (w, h, bytes)) in pictures.iter().enumerate() {
        let size = bytes.len();
        let Some(frame) = want.get(offset..offset + size) else {
            break;
        };
        offset += size;
        reference_frames += 1;
        if frame == bytes.as_slice() {
            exact += 1;
        } else if first_mismatch.is_none() {
            let luma = (*w * *h) as usize;
            let at = frame
                .iter()
                .zip(bytes)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let (plane, local) = if at < luma {
                ("y", at)
            } else if at < luma + luma / 4 {
                ("u", at - luma)
            } else {
                ("v", at - luma - luma / 4)
            };
            let pw = if plane == "y" { *w } else { *w / 2 } as usize;
            first_mismatch = Some(format!(
                "frame {index} {w}x{h}: first {plane} at ({}, {}), psnr {:.2} dB",
                local % pw,
                local / pw,
                psnr(frame, bytes)
            ));
        }
    }
    // Reference frames past ours, counted at the size of our last picture,
    // or as one when there is no picture to size them by.
    if offset < want.len() {
        let last = pictures.last().map_or(want.len(), |p| p.2.len().max(1));
        reference_frames += (want.len() - offset).div_ceil(last);
    }
    Result {
        name,
        ours: pictures.len(),
        reference: reference_frames,
        exact,
        first_mismatch,
        error,
    }
}

#[test]
fn decodes_the_ffmpeg_corpus_bit_exactly() {
    let Some(dir) = corpus() else {
        eprintln!("skipped: no MPEG-4 Visual corpus");
        return;
    };
    let filter = std::env::var("MPEG4_ONLY").ok();
    let mut streams = Vec::new();
    for sub in ["gen", "es"] {
        let Ok(entries) = std::fs::read_dir(dir.join(sub)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "m4v" | "h263") {
                continue;
            }
            let reference = path.with_extension("yuv");
            if !reference.is_file() {
                continue;
            }
            if let Some(f) = &filter
                && !path.to_string_lossy().contains(f.as_str())
            {
                continue;
            }
            streams.push((path, reference));
        }
    }
    streams.sort();
    let mut failed = Vec::new();
    for (stream, reference) in &streams {
        let r = run(stream, reference);
        let ok = r.error.is_none() && r.exact == r.ours && r.ours == r.reference;
        eprintln!(
            "{} {}: {} of {} pictures exact, reference {}{}{}",
            if ok { "PASS" } else { "FAIL" },
            r.name,
            r.exact,
            r.ours,
            r.reference,
            r.first_mismatch
                .as_ref()
                .map_or(String::new(), |m| format!("; {m}")),
            r.error
                .as_ref()
                .map_or(String::new(), |e| format!("; error {e}")),
        );
        if !ok {
            failed.push(r.name);
        }
    }
    assert!(failed.is_empty(), "mismatches: {failed:?}");
}
