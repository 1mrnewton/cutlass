//! Decoder tests across modules: the RFC 8251 vectors, hostile input,
//! reset, and concealment.

use super::softclip::{soft_clip, to_i16};
use super::{CeltDecoder, CeltFrame};
use crate::packet::{self, Mode};
use crate::range::RangeDecoder;

/// One packet of a `.bit` vector: the stored final range and the bytes.
struct VectorPacket {
    range: u32,
    data: Vec<u8>,
}

fn read_bit(path: &std::path::Path) -> Vec<VectorPacket> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let range =
            u32::from_be_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]]);
        at += 8;
        let end = (at + len as usize).min(bytes.len());
        out.push(VectorPacket {
            range,
            data: bytes[at..end].to_vec(),
        });
        at = end;
    }
    out
}

/// What decoding one vector found.
struct VectorRun {
    celt_packets: usize,
    range_matches: usize,
    first_mismatch: Option<usize>,
    /// 16-bit output, interleaved, as `opus_demo` writes it (CELT-only
    /// vectors only).
    pcm: Vec<i16>,
    /// Decoded audio over decode time, packet parsing included.
    real_time: f64,
}

/// Decodes every CELT-only packet of a vector with `channels` outputs at
/// 48 kHz, checking final ranges. Non-CELT packets are skipped (and make
/// the PCM meaningless, which callers know for vectors 08 and 09).
fn run_vector(path: &std::path::Path, channels: usize) -> VectorRun {
    let packets = read_bit(path);
    let mut dec = CeltDecoder::new(channels, 48_000).expect("decoder");
    let mut run = VectorRun {
        celt_packets: 0,
        range_matches: 0,
        first_mismatch: None,
        pcm: Vec::new(),
        real_time: 0.0,
    };
    let started = std::time::Instant::now();
    let mut clip_mem = vec![0f32; channels];
    let mut buf = vec![0f32; 960 * channels];
    for (index, p) in packets.iter().enumerate() {
        let Ok(frames) = packet::parse(&p.data, false) else {
            continue;
        };
        if frames.toc.mode != Mode::CeltOnly {
            continue;
        }
        run.celt_packets += 1;
        let fs = frames.toc.frame_samples as usize;
        let mut last_rng = 0u32;
        for &(off, len) in &frames.frames {
            let bytes = &p.data[off..off + len];
            let mut rd = RangeDecoder::new(bytes);
            let frame = CeltFrame {
                rd: if len > 1 { Some(&mut rd) } else { None },
                frame_size: fs,
                stream_channels: if frames.toc.stereo { 2 } else { 1 },
                start_band: 0,
                end_band: frames.toc.bandwidth.celt_end_band(),
            };
            dec.decode(frame, &mut buf).expect("decode");
            last_rng = if len > 1 { rd.rng() } else { 0 };
            let out = &mut buf[..fs * channels];
            soft_clip(out, &mut clip_mem);
            run.pcm.extend(out.iter().map(|&v| to_i16(v)));
        }
        if last_rng == p.range {
            run.range_matches += 1;
        } else if run.first_mismatch.is_none() {
            run.first_mismatch = Some(index);
        }
    }
    let seconds = run.pcm.len() as f64 / channels as f64 / 48_000.0;
    run.real_time = seconds / started.elapsed().as_secs_f64().max(1e-9);
    run
}

fn vectors_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTLASS_OPUS_VECTORS").map(std::path::PathBuf::from)
}

/// Every CELT-only packet of the RFC 8251 vectors decodes to the stored
/// final range, and with `CUTLASS_OPUS_COMPARE` pointing at
/// `opus_compare`, the output passes it in stereo and mono. Set
/// `CUTLASS_OPUS_VECTORS` to the `opus_newvectors` directory to run.
#[test]
fn rfc_vectors_match_ranges_and_pass_opus_compare() {
    let Some(dir) = vectors_dir() else {
        eprintln!("CUTLASS_OPUS_VECTORS unset; skipping the vector test");
        return;
    };
    let compare = std::env::var_os("CUTLASS_OPUS_COMPARE");
    let mut failures = Vec::new();
    for name in ["01", "07", "11", "08", "09"] {
        let bit = dir.join(format!("testvector{name}.bit"));
        let celt_only = matches!(name, "01" | "07" | "11");
        for channels in [2usize, 1] {
            let run = run_vector(&bit, channels);
            eprintln!(
                "vector {name} {}: {}/{} CELT packets match the final range{}, {:.0}x real time",
                if channels == 2 { "stereo" } else { "mono" },
                run.range_matches,
                run.celt_packets,
                run.first_mismatch
                    .map_or(String::new(), |i| format!(", first miss packet {i}")),
                run.real_time,
            );
            if run.range_matches != run.celt_packets || run.celt_packets == 0 {
                failures.push(format!("range {name} {channels}"));
            }
            if !celt_only {
                break;
            }
            let Some(compare) = &compare else {
                continue;
            };
            let ours = std::env::temp_dir().join(format!(
                "cutlass-celt-{}-{name}-{channels}.pcm",
                std::process::id()
            ));
            let bytes: Vec<u8> = run.pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ours, bytes).expect("write pcm");
            let mut cmd = std::process::Command::new(compare);
            if channels == 2 {
                cmd.arg("-s").arg(dir.join(format!("testvector{name}.dec")));
            } else {
                cmd.arg(dir.join(format!("testvector{name}m.dec")));
            }
            let output = cmd.arg(&ours).output().expect("run opus_compare");
            let _ = std::fs::remove_file(&ours);
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            eprintln!("opus_compare {name} channels {channels}: {}", text.trim());
            if !output.status.success() {
                failures.push(format!("compare {name} {channels}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

const BANDS_END: [usize; 4] = [13, 17, 19, 21];

/// Random bytes, truncations, and tiny frames at every frame size, band
/// range, and channel layout decode without panicking, to finite output.
#[test]
fn hostile_frames_decode_to_finite_output() {
    let mut seed = 0x5eed_u32;
    let mut out = vec![0f32; 960 * 2];
    for channels in [1usize, 2] {
        for rate in [48_000u32, 16_000, 8_000] {
            let mut dec = CeltDecoder::new(channels, rate).expect("decoder");
            for round in 0..1500 {
                let fs = 120usize << (round % 4);
                let stream = 1 + (lcg(&mut seed) >> 9) as usize % 2;
                let end = BANDS_END[(lcg(&mut seed) >> 7) as usize % 4];
                let start = if lcg(&mut seed).is_multiple_of(5) {
                    17.min(end - 1)
                } else {
                    0
                };
                let len = match round % 7 {
                    0 => (lcg(&mut seed) >> 8) as usize % 3,
                    1 => 2 + (lcg(&mut seed) >> 8) as usize % 8,
                    _ => (lcg(&mut seed) >> 8) as usize % 1276,
                };
                let bytes: Vec<u8> = (0..len).map(|_| (lcg(&mut seed) >> 13) as u8).collect();
                let mut rd = RangeDecoder::new(&bytes);
                let frame = CeltFrame {
                    rd: Some(&mut rd),
                    frame_size: fs,
                    stream_channels: stream,
                    start_band: start,
                    end_band: end,
                };
                dec.decode(frame, &mut out).expect("valid frame shape");
                let n = fs / (48_000 / rate as usize) * channels;
                assert!(out[..n].iter().all(|v| v.is_finite()), "round {round}");
                if round % 97 == 0 {
                    dec.reset();
                }
            }
        }
    }
}

/// Shapes the decoder cannot take come back as errors, not panics.
#[test]
fn bad_frame_shapes_are_errors() {
    let mut dec = CeltDecoder::new(2, 48_000).expect("decoder");
    let mut out = vec![0f32; 1920];
    let bytes = [0x55u8; 40];
    for (fs, stream, start, end, out_len) in [
        (100usize, 2usize, 0usize, 21usize, 1920usize),
        (960, 3, 0, 21, 1920),
        (960, 2, 21, 21, 1920),
        (960, 2, 0, 22, 1920),
        (960, 2, 0, 21, 100),
    ] {
        let mut rd = RangeDecoder::new(&bytes);
        let frame = CeltFrame {
            rd: Some(&mut rd),
            frame_size: fs,
            stream_channels: stream,
            start_band: start,
            end_band: end,
        };
        assert!(dec.decode(frame, &mut out[..out_len]).is_err());
    }
    assert!(CeltDecoder::new(3, 48_000).is_err());
    assert!(CeltDecoder::new(2, 44_100).is_err());
}

fn sample_frames() -> Vec<Vec<u8>> {
    let mut seed = 77u32;
    (0..40)
        .map(|_| (0..120).map(|_| (lcg(&mut seed) >> 11) as u8).collect())
        .collect()
}

fn decode_all(dec: &mut CeltDecoder, frames: &[Vec<u8>]) -> Vec<f32> {
    let mut all = Vec::new();
    let mut out = vec![0f32; 1920];
    for f in frames {
        let mut rd = RangeDecoder::new(f);
        let frame = CeltFrame {
            rd: Some(&mut rd),
            frame_size: 960,
            stream_channels: 2,
            start_band: 0,
            end_band: 21,
        };
        dec.decode(frame, &mut out).expect("decode");
        all.extend_from_slice(&out);
    }
    all
}

/// A reset decoder decodes exactly like a new one.
#[test]
fn reset_matches_a_fresh_decoder() {
    let frames = sample_frames();
    let mut fresh = CeltDecoder::new(2, 48_000).expect("decoder");
    let expected = decode_all(&mut fresh, &frames);
    let mut used = CeltDecoder::new(2, 48_000).expect("decoder");
    decode_all(&mut used, &frames[5..]);
    used.reset();
    assert_eq!(decode_all(&mut used, &frames), expected);
}

/// Concealment after real frames stays finite and dies away.
#[test]
fn concealment_is_finite_and_decays() {
    let frames = sample_frames();
    for channels in [1usize, 2] {
        let mut dec = CeltDecoder::new(channels, 48_000).expect("decoder");
        let mut out = vec![0f32; 960 * channels];
        for f in &frames[..10] {
            let mut rd = RangeDecoder::new(f);
            let frame = CeltFrame {
                rd: Some(&mut rd),
                frame_size: 960,
                stream_channels: 2,
                start_band: 0,
                end_band: 21,
            };
            dec.decode(frame, &mut out).expect("decode");
        }
        let mut energies = Vec::new();
        for _ in 0..30 {
            let frame = CeltFrame {
                rd: None,
                frame_size: 960,
                stream_channels: 2,
                start_band: 0,
                end_band: 21,
            };
            dec.decode(frame, &mut out).expect("conceal");
            assert!(out.iter().all(|v| v.is_finite()));
            energies.push(out.iter().map(|v| v * v).sum::<f32>());
        }
        let head: f32 = energies[..3].iter().sum();
        let tail: f32 = energies[27..].iter().sum();
        assert!(tail < head.max(1e-9), "{energies:?}");
    }
}
