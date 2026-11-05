//! Whole-decoder tests: the official RFC 8251 test vectors (when
//! `CUTLASS_OPUS_VECTORS` names their folder), lifecycle, loss, FEC, and
//! hostile input.

use super::{SilkControl, SilkDecoder, SilkLoss};
use crate::packet::{Mode, parse};
use crate::range::RangeDecoder;
use std::path::PathBuf;

/// One record of a `.bit` file: the packet and its final range.
struct Record {
    packet: Vec<u8>,
    final_range: u32,
}

fn vector_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("CUTLASS_OPUS_VECTORS")?);
    if dir.join("testvector02.bit").is_file() {
        Some(dir)
    } else {
        eprintln!("CUTLASS_OPUS_VECTORS has no testvector02.bit; skipping");
        None
    }
}

fn read_bit(path: &std::path::Path) -> Vec<Record> {
    let data = std::fs::read(path).unwrap();
    let mut records = Vec::new();
    let mut at = 0;
    while at + 8 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let final_range = u32::from_be_bytes(data[at + 4..at + 8].try_into().unwrap());
        at += 8;
        records.push(Record {
            packet: data[at..at + len].to_vec(),
            final_range,
        });
        at += len;
    }
    records
}

/// What decoding a vector's SILK packets found.
#[derive(Debug, Default)]
struct VectorRun {
    /// Interleaved 48 kHz output of the SILK-only packets (hybrid and CELT
    /// packets contribute silence).
    pcm: Vec<i16>,
    /// SILK-only packets whose final range was compared.
    checked: usize,
    /// Of those, the ones that matched.
    matched: usize,
    /// SILK-only packets carrying a CELT redundancy frame (not comparable).
    redundant: usize,
    /// Frames of one byte or less, concealed.
    concealed: usize,
    /// Hybrid packets whose SILK layer was decoded for continuity.
    hybrid: usize,
    /// CELT-only packets skipped.
    celt: usize,
    /// First mismatching packet index.
    first_mismatch: Option<usize>,
}

/// Decodes the SILK layer of every packet the way the reference
/// `opus_decode_frame` drives `silk_Decode`, at 48 kHz.
fn run_vector(records: &[Record], api_channels: usize) -> VectorRun {
    let mut run = VectorRun::default();
    let mut dec = SilkDecoder::new();
    let mut prev_mode: Option<Mode> = None;
    let mut last_ctl: Option<SilkControl> = None;
    let mut frame = vec![0i16; 2880 * 2];
    for (index, rec) in records.iter().enumerate() {
        let frames = parse(&rec.packet, false).unwrap();
        let toc = frames.toc;
        let samples = toc.frame_samples as usize;
        let mut packet_rng = None;
        let mut comparable = true;
        for &(offset, len) in &frames.frames {
            frame.fill(0);
            let bytes = &rec.packet[offset..offset + len];
            if len <= 1 {
                // DTX or a lost frame: conceal with the previous layer.
                run.concealed += 1;
                packet_rng = Some(0);
                if prev_mode.is_some_and(|m| m != Mode::CeltOnly)
                    && let Some(mut ctl) = last_ctl
                {
                    ctl.payload_ms = (samples as u32 / 48).max(10);
                    let mut rd = RangeDecoder::new(&[]);
                    let mut done = 0;
                    while done < samples {
                        let n = dec
                            .decode(
                                &ctl,
                                SilkLoss::Lost,
                                done == 0,
                                &mut rd,
                                &mut frame[done * api_channels..],
                            )
                            .unwrap();
                        done += n;
                    }
                }
                run.pcm.extend_from_slice(&frame[..samples * api_channels]);
                continue;
            }
            if toc.mode == Mode::CeltOnly {
                run.celt += 1;
                comparable = false;
                prev_mode = Some(Mode::CeltOnly);
                run.pcm.extend_from_slice(&frame[..samples * api_channels]);
                continue;
            }
            if prev_mode == Some(Mode::CeltOnly) {
                dec.reset();
            }
            let ctl = SilkControl {
                api_channels,
                internal_channels: if toc.stereo { 2 } else { 1 },
                api_rate: 48_000,
                internal_rate: toc.bandwidth.silk_rate(),
                payload_ms: (samples as u32 / 48).max(10),
            };
            last_ctl = Some(ctl);
            let mut rd = RangeDecoder::new(bytes);
            let mut done = 0;
            while done < samples {
                let n = dec
                    .decode(
                        &ctl,
                        SilkLoss::Normal,
                        done == 0,
                        &mut rd,
                        &mut frame[done * api_channels..],
                    )
                    .unwrap();
                assert!(n > 0);
                done += n;
            }
            assert_eq!(done, samples);
            if toc.mode == Mode::Hybrid {
                run.hybrid += 1;
                comparable = false;
                run.pcm.extend_from_slice(&vec![0; samples * api_channels]);
            } else {
                if rd.tell() + 17 <= 8 * len as i32 {
                    comparable = false;
                    run.redundant += 1;
                }
                run.pcm.extend_from_slice(&frame[..samples * api_channels]);
            }
            prev_mode = Some(toc.mode);
            packet_rng = Some(rd.rng());
        }
        if comparable && toc.mode == Mode::SilkOnly {
            run.checked += 1;
            if packet_rng == Some(rec.final_range) {
                run.matched += 1;
            } else if run.first_mismatch.is_none() {
                run.first_mismatch = Some(index);
            }
        }
    }
    run
}

fn read_pcm(path: &std::path::Path) -> Vec<i16> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| i16::from_le_bytes(b))
        .collect()
}

#[test]
fn official_silk_vectors_match_the_reference_decoder() {
    let Some(dir) = vector_dir() else {
        eprintln!("CUTLASS_OPUS_VECTORS unset; skipping the SILK test vectors");
        return;
    };
    let out_dir = std::env::var_os("CUTLASS_OPUS_SILK_OUT").map(PathBuf::from);
    for n in ["02", "03", "04"] {
        let records = read_bit(&dir.join(format!("testvector{n}.bit")));
        for (api_channels, suffix) in [(2usize, ""), (1, "m")] {
            let run = run_vector(&records, api_channels);
            assert_eq!(run.celt + run.hybrid, 0, "vector {n} is SILK-only");
            let reference = read_pcm(&dir.join(format!("testvector{n}{suffix}.dec")));
            let common = reference.len().min(run.pcm.len());
            let max_diff = reference[..common]
                .iter()
                .zip(&run.pcm[..common])
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .max()
                .unwrap_or(0);
            let exact = reference[..common] == run.pcm[..common];
            println!(
                "vector {n}{suffix}: final range {}/{} (redundant {}, concealed {}), samples {} vs {}, max |diff| {max_diff}, bit-exact {exact}",
                run.matched,
                run.checked,
                run.redundant,
                run.concealed,
                run.pcm.len(),
                reference.len()
            );
            if let Some(out_dir) = &out_dir {
                let bytes: Vec<u8> = run.pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                std::fs::write(out_dir.join(format!("silk{n}{suffix}.pcm")), bytes).unwrap();
            }
            assert_eq!(
                run.matched, run.checked,
                "vector {n}{suffix}: first mismatch at packet {:?}",
                run.first_mismatch
            );
            assert!(run.checked > 0);
        }
    }
}

#[test]
fn official_mixed_vector_keeps_silk_ranges_across_hybrid_packets() {
    let Some(dir) = vector_dir() else {
        return;
    };
    let records = read_bit(&dir.join("testvector12.bit"));
    let run = run_vector(&records, 2);
    println!(
        "vector 12: final range {}/{} SILK-only (redundant {}, hybrid {}, celt {})",
        run.matched, run.checked, run.redundant, run.hybrid, run.celt
    );
    assert!(run.checked > 0);
    assert_eq!(
        run.matched, run.checked,
        "first mismatch at packet {:?}",
        run.first_mismatch
    );
}
