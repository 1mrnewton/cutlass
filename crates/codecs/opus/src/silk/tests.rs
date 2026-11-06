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
            let started = std::time::Instant::now();
            let run = run_vector(&records, api_channels);
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(run.celt + run.hybrid, 0, "vector {n} is SILK-only");
            // The reference output is 48 kHz stereo for both runs;
            // opus_compare checks a mono decode against its downmix.
            let reference = read_pcm(&dir.join(format!("testvector{n}{suffix}.dec")));
            let expected: Vec<f64> = if api_channels == 2 {
                reference.iter().map(|&v| f64::from(v)).collect()
            } else {
                reference
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[l, r]| (f64::from(l) + f64::from(r)) / 2.0)
                    .collect()
            };
            assert_eq!(run.pcm.len(), expected.len(), "vector {n}{suffix} length");
            let max_diff = expected
                .iter()
                .zip(&run.pcm)
                .map(|(&a, &b)| (a - f64::from(b)).abs())
                .fold(0.0, f64::max);
            let seconds = (run.pcm.len() / api_channels) as f64 / 48_000.0;
            println!(
                "vector {n}{suffix}: final range {}/{} (redundant {}, concealed {}), max |diff| {max_diff}, {:.0}x real time",
                run.matched,
                run.checked,
                run.redundant,
                run.concealed,
                seconds / elapsed
            );
            if let Some(out_dir) = &out_dir {
                let bytes: Vec<u8> = run.pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                std::fs::write(out_dir.join(format!("silk{n}{suffix}.pcm")), bytes).unwrap();
            }
            assert!(run.checked > 0);
            assert_eq!(
                run.matched, run.checked,
                "vector {n}{suffix}: first mismatch at packet {:?}",
                run.first_mismatch
            );
            if api_channels == 2 {
                // Stereo output is the reference's, sample for sample.
                assert_eq!(max_diff, 0.0, "vector {n}");
            } else {
                // Mono is the mid channel: the downmix up to rounding.
                assert!(max_diff <= 2.0, "vector {n}m: {max_diff}");
            }
            // Coarse: SILK decoding is far faster than real time.
            assert!(seconds / elapsed > 20.0, "vector {n}{suffix}: {elapsed} s");
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

/// Pseudo-random packet bytes.
fn noise(seed: &mut u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 24) as u8
        })
        .collect()
}

fn control(channels: usize, internal_rate: u32, payload_ms: u32) -> SilkControl {
    SilkControl {
        api_channels: channels,
        internal_channels: channels,
        api_rate: 48_000,
        internal_rate,
        payload_ms,
    }
}

/// Decodes a whole packet the way the Opus layer does: one call per 10 or
/// 20 ms frame, all from one range decoder.
fn decode_packet(
    dec: &mut SilkDecoder,
    ctl: &SilkControl,
    loss: SilkLoss,
    bytes: &[u8],
) -> Vec<i16> {
    let samples = ctl.payload_ms.max(10) as usize * ctl.api_rate as usize / 1000;
    let mut out = vec![0i16; samples * ctl.api_channels];
    let mut rd = RangeDecoder::new(bytes);
    let mut done = 0;
    while done < samples {
        let n = dec
            .decode(
                ctl,
                loss,
                done == 0,
                &mut rd,
                &mut out[done * ctl.api_channels..],
            )
            .unwrap();
        assert!(n > 0);
        done += n;
    }
    assert_eq!(done, samples);
    out
}

fn energy(pcm: &[i16]) -> f64 {
    pcm.iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / pcm.len().max(1) as f64
}

#[test]
fn reset_decodes_like_a_fresh_decoder() {
    let mut seed = 7;
    let ctl = control(2, 16_000, 20);
    let packets: Vec<Vec<u8>> = (0..6).map(|_| noise(&mut seed, 60)).collect();
    let mut used = SilkDecoder::new();
    for p in &packets {
        decode_packet(&mut used, &ctl, SilkLoss::Normal, p);
    }
    // Leave it in a different configuration too.
    decode_packet(
        &mut used,
        &control(1, 8_000, 60),
        SilkLoss::Normal,
        &packets[0],
    );
    used.reset();
    let mut fresh = SilkDecoder::new();
    for p in &packets {
        assert_eq!(
            decode_packet(&mut used, &ctl, SilkLoss::Normal, p),
            decode_packet(&mut fresh, &ctl, SilkLoss::Normal, p)
        );
    }
}

#[test]
fn channel_and_rate_switches_keep_decoding() {
    let mut seed = 99;
    let mut dec = SilkDecoder::new();
    let steps = [
        // (api, internal, rate, ms)
        (2, 1, 8_000, 20),
        (2, 1, 8_000, 40),
        (2, 2, 8_000, 20),
        (2, 2, 16_000, 20),
        (2, 1, 16_000, 60),
        (1, 2, 12_000, 10),
        (1, 1, 16_000, 20),
        (2, 2, 12_000, 60),
    ];
    for &(api, internal, rate, ms) in &steps {
        let ctl = SilkControl {
            api_channels: api,
            internal_channels: internal,
            api_rate: 48_000,
            internal_rate: rate,
            payload_ms: ms,
        };
        for _ in 0..3 {
            let pcm = decode_packet(&mut dec, &ctl, SilkLoss::Normal, &noise(&mut seed, 80));
            assert_eq!(pcm.len(), ms as usize * 48 * api);
        }
        assert_eq!(dec.channels[0].fs_khz, (rate / 1000) as i32);
    }

    // Narrowband to wideband: the rate change restarts the synthesis
    // state, so once the old rate's samples have left the resampler and
    // stereo delay lines, the output is what a fresh WB decoder gives.
    let wb = control(1, 16_000, 20);
    let mut nb_first = SilkDecoder::new();
    decode_packet(
        &mut nb_first,
        &control(1, 8_000, 20),
        SilkLoss::Normal,
        &noise(&mut seed, 60),
    );
    let packet = noise(&mut seed, 60);
    let switched = decode_packet(&mut nb_first, &wb, SilkLoss::Normal, &packet);
    assert_eq!(switched.len(), 960);
    assert_eq!(nb_first.channels[0].fs_khz, 16);
    assert_eq!(nb_first.channels[0].lpc_order, 16);
    let fresh = decode_packet(&mut SilkDecoder::new(), &wb, SilkLoss::Normal, &packet);
    let last_diff = switched.iter().zip(&fresh).rposition(|(a, b)| a != b);
    assert!(last_diff.is_none_or(|i| i < 240), "{last_diff:?}");
}

/// A fresh mono WB decoder fed one pseudo-random packet that decodes as a
/// voiced frame.
fn voiced_decoder() -> (SilkDecoder, Vec<i16>) {
    let ctl = control(1, 16_000, 20);
    let mut seed = 1;
    for _ in 0..10_000 {
        let packet = noise(&mut seed, 80);
        let mut dec = SilkDecoder::new();
        let pcm = decode_packet(&mut dec, &ctl, SilkLoss::Normal, &packet);
        if dec.channels[0].prev_signal_type == super::indices::TYPE_VOICED && energy(&pcm) > 1e4 {
            return (dec, pcm);
        }
    }
    panic!("no voiced packet found");
}

#[test]
fn loss_after_a_voiced_frame_fades_to_silence() {
    let (mut dec, _) = voiced_decoder();
    let ctl = control(1, 16_000, 20);
    let mut energies = Vec::new();
    for _ in 0..40 {
        let pcm = decode_packet(&mut dec, &ctl, SilkLoss::Lost, &[]);
        energies.push(energy(&pcm));
    }
    println!("concealment energy per 20 ms: {energies:.1?}");
    assert!(energies[0] > 1.0, "concealment continues the signal");
    // Fading: each half second is quieter than the one before.
    let first: f64 = energies[..10].iter().sum();
    let second: f64 = energies[10..20].iter().sum();
    assert!(second < first);
    // After many losses only silence is left.
    let last = decode_packet(&mut dec, &ctl, SilkLoss::Lost, &[]);
    assert!(last.iter().all(|v| v.abs() <= 1), "{:?}", &last[..16]);
    // A good packet afterwards decodes normally.
    let mut seed = 5;
    decode_packet(&mut dec, &ctl, SilkLoss::Normal, &noise(&mut seed, 60));
}

#[test]
fn fec_decodes_the_lbrr_copy_when_present() {
    let ctl = control(1, 16_000, 20);
    let mut seed = 3;
    let mut with_lbrr = 0;
    for _ in 0..200 {
        let packet = noise(&mut seed, 80);
        let (mut dec, _) = voiced_decoder();
        let mut rd = RangeDecoder::new(&packet);
        let mut out = vec![0i16; 960];
        let n = dec
            .decode(&ctl, SilkLoss::Fec, true, &mut rd, &mut out)
            .unwrap();
        assert_eq!(n, 960);
        assert!(rd.tell() <= 8 * packet.len() as i32 + 32);
        if dec.channels[0].lbrr_flags[0] {
            with_lbrr += 1;
            // The LBRR frame was decoded, not concealed.
            assert_eq!(dec.channels[0].loss_cnt, 0);
        } else {
            assert_eq!(dec.channels[0].loss_cnt, 1);
        }
    }
    assert!(with_lbrr > 0);
}

/// Reference streams with in-band FEC, when `CUTLASS_OPUS_SILK_FEC` names
/// their folder (the official vectors carry no LBRR frames). Made with the
/// RFC 6716 `opus_demo` (its `-loss` check relaxed so the encoder can be
/// told to expect loss):
///
/// ```text
/// opus_demo -e voip 16000 1 20000 -bandwidth WB -framesize 20 -inbandfec -loss 20 in16m.pcm fecA.bit
/// opus_demo -e voip 48000 2 40000 -bandwidth WB -framesize 60 -inbandfec -loss 25 in48s.pcm fecB.bit
/// ```
///
/// then packets `i % 7 == 3` and `i % 23` in `{10, 11}` set to length 0
/// (`fecA_lossy.bit`), and decoded with `opus_demo -d <rate> <channels>
/// -inbandfec` into `fecA_lossy.dec`.
#[test]
fn reference_fec_streams_match_with_lost_packets() {
    let Some(dir) = std::env::var_os("CUTLASS_OPUS_SILK_FEC").map(PathBuf::from) else {
        eprintln!("CUTLASS_OPUS_SILK_FEC unset; skipping the FEC streams");
        return;
    };
    for (name, api_rate, api_channels) in
        [("fecA_lossy", 16_000u32, 1usize), ("fecB_lossy", 48_000, 2)]
    {
        let records = read_bit(&dir.join(format!("{name}.bit")));
        let reference = read_pcm(&dir.join(format!("{name}.dec")));
        let mut dec = SilkDecoder::new();
        let mut pcm = Vec::new();
        let mut last_ctl: Option<SilkControl> = None;
        let (mut fec_packets, mut lbrr_frames, mut concealed) = (0, 0, 0);
        let ctl_of = |packet: &[u8]| {
            let frames = parse(packet, false).unwrap();
            assert_eq!(frames.toc.mode, Mode::SilkOnly);
            let ctl = SilkControl {
                api_channels,
                internal_channels: if frames.toc.stereo { 2 } else { 1 },
                api_rate,
                internal_rate: frames.toc.bandwidth.silk_rate(),
                payload_ms: (frames.toc.frame_samples / 48).max(10),
            };
            (ctl, frames.frames)
        };
        // opus_demo -inbandfec decodes one packet late: a lost packet is
        // recovered from the next packet's LBRR data, or concealed if
        // that one is lost too.
        let mut lost_prev = true;
        for (count, rec) in records.iter().enumerate() {
            let lost = rec.packet.is_empty();
            if count >= 1 {
                if lost_prev && lost {
                    let ctl = last_ctl.unwrap();
                    pcm.extend(decode_packet(&mut dec, &ctl, SilkLoss::Lost, &[]));
                    concealed += 1;
                } else {
                    let (packet, loss) = if lost_prev {
                        (&rec.packet, SilkLoss::Fec)
                    } else {
                        (&records[count - 1].packet, SilkLoss::Normal)
                    };
                    let (ctl, frames) = ctl_of(packet);
                    last_ctl = Some(ctl);
                    for (offset, len) in frames {
                        pcm.extend(decode_packet(
                            &mut dec,
                            &ctl,
                            loss,
                            &packet[offset..offset + len],
                        ));
                        if loss == SilkLoss::Fec {
                            fec_packets += 1;
                            lbrr_frames +=
                                dec.channels[0].lbrr_flags.iter().filter(|&&f| f).count();
                        }
                    }
                }
            }
            lost_prev = lost;
        }
        assert_eq!(pcm.len(), reference.len(), "{name}");
        let max_diff = pcm
            .iter()
            .zip(&reference)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap_or(0);
        println!(
            "{name}: {fec_packets} FEC decodes ({lbrr_frames} LBRR frames), {concealed} concealed, max |diff| {max_diff}"
        );
        assert!(lbrr_frames > 0, "{name} carries LBRR frames");
        assert_eq!(max_diff, 0, "{name}");
    }
}

#[test]
fn hostile_packets_decode_without_panicking() {
    let mut seed = 0xdead_beef;
    let mut dec = SilkDecoder::new();
    for round in 0..400usize {
        let api = 1 + round % 2;
        let ctl = SilkControl {
            api_channels: api,
            internal_channels: 1 + (round / 2) % 2,
            api_rate: [8_000, 12_000, 16_000, 24_000, 48_000][round % 5],
            internal_rate: [8_000, 12_000, 16_000][round % 3],
            payload_ms: [10, 20, 40, 60][(round / 3) % 4],
        };
        let len = [0, 1, 2, 5, 17, 60, 300][round % 7];
        let mut bytes = noise(&mut seed, len);
        if round % 11 == 0 {
            bytes.fill(0xff);
        } else if round % 13 == 0 {
            bytes.fill(0);
        }
        let loss = [
            SilkLoss::Normal,
            SilkLoss::Normal,
            SilkLoss::Fec,
            SilkLoss::Lost,
        ][round % 4];
        let pcm = decode_packet(&mut dec, &ctl, loss, &bytes);
        assert_eq!(
            pcm.len(),
            ctl.payload_ms as usize * ctl.api_rate as usize / 1000 * api
        );
    }
    // A truncated real-looking packet: the range decoder reads zeros past
    // the end.
    let packet = noise(&mut seed, 100);
    for cut in [0, 1, 3, 10, 50] {
        decode_packet(
            &mut dec,
            &control(2, 16_000, 60),
            SilkLoss::Normal,
            &packet[..cut],
        );
    }
}

#[test]
fn api_misuse_is_an_error_and_leaves_the_decoder_alone() {
    use cutlass_decode::CodecError;
    let good = control(1, 16_000, 40);
    let mut dec = SilkDecoder::new();
    let mut out = vec![0i16; 1920];
    let mut rd = RangeDecoder::new(&[]);
    let bad = [
        SilkControl {
            api_channels: 0,
            ..good
        },
        SilkControl {
            api_channels: 3,
            ..good
        },
        SilkControl {
            internal_channels: 3,
            ..good
        },
        SilkControl {
            api_rate: 44_100,
            ..good
        },
        SilkControl {
            internal_rate: 24_000,
            ..good
        },
        SilkControl {
            payload_ms: 30,
            ..good
        },
    ];
    for ctl in &bad {
        let r = dec.decode(ctl, SilkLoss::Normal, true, &mut rd, &mut out);
        assert!(matches!(r, Err(CodecError::InvalidApi { .. })), "{ctl:?}");
    }
    // Output too small for a 20 ms frame at 48 kHz.
    let r = dec.decode(&good, SilkLoss::Normal, true, &mut rd, &mut out[..959]);
    assert!(matches!(r, Err(CodecError::InvalidApi { .. })));

    // Mid-packet: the configuration cannot change, and a packet has only
    // its own frames.
    let mut seed = 11;
    let packet = noise(&mut seed, 60);
    let mut rd = RangeDecoder::new(&packet);
    assert_eq!(
        dec.decode(&good, SilkLoss::Normal, true, &mut rd, &mut out)
            .unwrap(),
        960
    );
    let other = SilkControl {
        internal_rate: 8_000,
        ..good
    };
    let r = dec.decode(&other, SilkLoss::Normal, false, &mut rd, &mut out);
    assert!(matches!(r, Err(CodecError::InvalidApi { .. })));
    assert_eq!(
        dec.decode(&good, SilkLoss::Normal, false, &mut rd, &mut out)
            .unwrap(),
        960
    );
    let r = dec.decode(&good, SilkLoss::Normal, false, &mut rd, &mut out);
    assert!(matches!(r, Err(CodecError::InvalidApi { .. })));

    // None of the errors changed the state: the packet decodes as on a
    // fresh decoder.
    let mut fresh = SilkDecoder::new();
    let mut errored = SilkDecoder::new();
    for ctl in &bad {
        let _ = errored.decode(
            ctl,
            SilkLoss::Normal,
            true,
            &mut RangeDecoder::new(&packet),
            &mut out,
        );
    }
    assert_eq!(
        decode_packet(&mut errored, &good, SilkLoss::Normal, &packet),
        decode_packet(&mut fresh, &good, SilkLoss::Normal, &packet)
    );
}
