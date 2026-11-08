//! Bit-exact decodes against FFmpeg's reference output.
//!
//! The corpus lives beside the checkouts, at `../mpeg4-part2-corpus` from
//! the workspace root (or `MPEG4_CORPUS`). Each `gen/` (FFmpeg-encoded)
//! and `es/` (third-party) `*.m4v` or `*.h263` elementary stream has a
//! `*.yuv` FFmpeg 8.1.1 decode beside it, made with `-threads 1 -cpuflags 0
//! -flags +bitexact -idct simple -fps_mode passthrough -noautoscale`.
//! The test skips when the folder is missing, as it is in CI.

use std::path::{Path, PathBuf};

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder};
use cutlass_mpeg4::EngineDecoder;

use crate::common::{packet, pictures, split};

fn corpus() -> Option<PathBuf> {
    let dir = std::env::var_os("MPEG4_CORPUS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../mpeg4-part2-corpus"),
        PathBuf::from,
    );
    dir.is_dir().then_some(dir)
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
    let mut got = Vec::new();
    let mut error = None;
    for (n, data) in split(&data).into_iter().enumerate() {
        match decoder.push(&packet(data, n as i64)) {
            Ok(out) => got.extend(pictures(out)),
            Err(e) => {
                error = Some(format!("packet {n}: {e}"));
                break;
            }
        }
    }
    if error.is_none() {
        got.extend(pictures(decoder.drain().expect("drain")));
    }
    let pictures: Vec<(u32, u32, Vec<u8>)> =
        got.into_iter().map(|(w, h, _, b)| (w, h, b)).collect();
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

#[test]
fn refuses_the_studio_profile() {
    let Some(dir) = corpus() else {
        eprintln!("skipped: no MPEG-4 Visual corpus");
        return;
    };
    let data = std::fs::read(dir.join("fate/mpeg4_sstp_dpcm.m4v")).expect("studio stream");
    let mut decoder = EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0)).expect("open");
    let mut outcome = Ok(());
    for (n, data) in split(&data).into_iter().enumerate() {
        match decoder.push(&packet(data, n as i64)) {
            Ok(out) => assert!(out.is_empty(), "a studio picture decoded"),
            Err(e) => {
                outcome = Err(e);
                break;
            }
        }
    }
    assert!(
        matches!(outcome, Err(CodecError::Unsupported { .. })),
        "{outcome:?}"
    );
}
