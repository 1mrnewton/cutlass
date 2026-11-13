//! In-process fuzz of the AV1 decoder and encoder.
//!
//! cargo-fuzz is not used: libfuzzer-sys builds C, which this workspace
//! does not allow. `mutated_fixtures_do_not_panic` and
//! `encoder_seeds_round_trip` run on every `cargo test` from fixed seeds.
//! The timed campaign is ignored:
//!
//! ```text
//! AV1_FUZZ_SECS=1800 cargo test --release -p cutlass-av1 --test fuzz -- --ignored --nocapture
//! ```
//!
//! A case is one fixture stream with a few of its packets damaged: bits
//! flipped, bytes overwritten, a packet cut short, its first bytes (the
//! headers) overwritten, OBUs dropped, repeated or swapped, or a metadata
//! OBU of random bytes put in. It is decoded on one thread and on three.
//! Any result but a panic is accepted, on any thread; after the damaged
//! stream and a reset, the stream as it was must decode to the samples it
//! decodes to in a fresh decoder.

// Clippy allows `expect` and `panic` in `cfg(test)` code; the helpers
// live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;

#[cfg(test)]
mod fuzz {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::common;
    use cutlass_av1::dec::obu::split_obus;
    use cutlass_av1::enc::bitwriter::write_obu;
    use cutlass_av1::{Av1Decoder, Av1Encoder};
    use cutlass_decode::{CodecId, CodecParams, Decoder, Output, PixelFormat};
    use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime};

    /// Panics on any thread of this test binary, the decoder's workers
    /// included.
    static PANICS: AtomicUsize = AtomicUsize::new(0);

    fn count_panics() {
        static HOOK: std::sync::Once = std::sync::Once::new();
        HOOK.call_once(|| {
            let before = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                PANICS.fetch_add(1, Ordering::Relaxed);
                before(info);
            }));
        });
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() as usize) % n.max(1)
        }
    }

    /// The OBUs of `packet` written again after `edit` changed their list.
    /// A packet that does not split is returned as it is.
    fn edit_obus(packet: &[u8], edit: impl FnOnce(&mut Vec<(u8, Vec<u8>)>)) -> Vec<u8> {
        let Ok(obus) = split_obus(packet) else {
            return packet.to_vec();
        };
        let mut list: Vec<(u8, Vec<u8>)> = obus
            .iter()
            .map(|obu| (obu.obu_type, obu.payload.to_vec()))
            .collect();
        edit(&mut list);
        let mut out = Vec::new();
        for (obu_type, payload) in list {
            write_obu(&mut out, obu_type, 0, 0, false, &payload);
        }
        out
    }

    /// Damages a few packets of `packets` in one of eight ways.
    fn damage(packets: &mut [Vec<u8>], rng: &mut Rng) {
        let mode = rng.below(8);
        for _ in 0..1 + rng.below(8) {
            let at = rng.below(packets.len());
            let len = packets[at].len();
            if len == 0 {
                continue;
            }
            match mode {
                0 => {
                    let i = rng.below(len);
                    packets[at][i] ^= 1 << rng.below(8);
                }
                1 => {
                    let i = rng.below(len);
                    packets[at][i] = rng.next() as u8;
                }
                2 => {
                    let cut = rng.below(len);
                    packets[at].truncate(cut);
                }
                3 => {
                    // The headers are in the first bytes.
                    let i = rng.below(len.min(24));
                    packets[at][i] = rng.next() as u8;
                }
                4 => {
                    let drop = rng.next();
                    packets[at] = edit_obus(&packets[at], |list| {
                        if !list.is_empty() {
                            list.remove(drop as usize % list.len());
                        }
                    });
                }
                5 => {
                    let pick = rng.next();
                    packets[at] = edit_obus(&packets[at], |list| {
                        if !list.is_empty() {
                            let i = pick as usize % list.len();
                            let again = list[i].clone();
                            list.insert(i, again);
                        }
                    });
                }
                6 => {
                    let (a, b) = (rng.next(), rng.next());
                    packets[at] = edit_obus(&packets[at], |list| {
                        if list.len() > 1 {
                            let n = list.len();
                            list.swap(a as usize % n, b as usize % n);
                        }
                    });
                }
                _ => {
                    // A metadata OBU: a type from 0 to 7 and random bytes.
                    let mut payload = vec![rng.below(8) as u8];
                    for _ in 0..rng.below(40) {
                        payload.push(rng.next() as u8);
                    }
                    let place = rng.next();
                    packets[at] = edit_obus(&packets[at], |list| {
                        let i = place as usize % (list.len() + 1);
                        list.insert(i, (5, payload));
                    });
                }
            }
        }
    }

    /// The samples of every picture `packets` decodes to; an error ends the
    /// stream there, as a player would stop.
    fn decode(decoder: &mut Av1Decoder, packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut take = |outputs: Vec<Output>| {
            for output in outputs {
                if let Output::Picture(picture) = output {
                    out.push(common::samples(&picture).to_vec());
                }
            }
        };
        for (pts, data) in packets.iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            if let Ok(outputs) = decoder.push(&common::packet(data, pts as i64)) {
                take(outputs);
            }
        }
        // A frame that failed while it decoded is reported once by drain.
        for _ in 0..packets.len() + 2 {
            match decoder.drain() {
                Ok(outputs) => {
                    take(outputs);
                    break;
                }
                Err(_) => continue,
            }
        }
        out
    }

    fn decoder(threads: usize) -> Av1Decoder {
        let mut decoder =
            Av1Decoder::open(&CodecParams::video(CodecId::Av1, 64, 64)).expect("decoder");
        decoder.set_threads(threads);
        // The largest fixture is 3840x2160. A damaged header may ask for
        // any size up to 65536 squared, which is refused past this.
        decoder.set_max_pixels(3840 * 2160);
        decoder
    }

    /// One damaged copy of `packets` through a decoder on `threads`, then
    /// the stream as it was after a reset.
    fn case(packets: &[Vec<u8>], clean: &[Vec<u8>], threads: usize, rng: &mut Rng) {
        let mut damaged = packets.to_vec();
        damage(&mut damaged, rng);
        let mut decoder = decoder(threads);
        let _ = decode(&mut decoder, &damaged);
        decoder.reset();
        let again = decode(&mut decoder, packets);
        assert!(
            again == clean,
            "the stream decodes differently after damage"
        );
    }

    /// The fixture streams that decode, with the samples they decode to.
    fn corpus() -> Vec<(String, Vec<Vec<u8>>, Vec<Vec<u8>>)> {
        common::streams()
            .into_iter()
            .filter(|(name, _)| name != "wide.ivf")
            .map(|(name, packets)| {
                let clean = decode(&mut decoder(1), &packets);
                assert!(!clean.is_empty(), "{name} decodes");
                (name, packets, clean)
            })
            .collect()
    }

    #[test]
    fn mutated_fixtures_do_not_panic() {
        count_panics();
        let corpus = corpus();
        assert!(corpus.len() >= 80, "the fixture streams are the corpus");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut cases = 0;
        for round in 0..6 {
            for (_, packets, clean) in &corpus {
                let threads = if round % 2 == 0 { 1 } else { 3 };
                case(packets, clean, threads, &mut rng);
                cases += 1;
            }
        }
        assert!(cases >= 480);
        assert_eq!(PANICS.load(Ordering::Relaxed), 0, "a thread panicked");
    }

    /// A picture of `pixel` layout from the generator, `index` pictures in.
    fn picture(
        pixel: PixelFormat,
        width: u32,
        height: u32,
        index: u32,
        rng: &mut Rng,
    ) -> FrameInput {
        let (cw, ch) = pixel.chroma_wh(width, height).unwrap_or((0, 0));
        let depth: u32 = match pixel {
            PixelFormat::Yuv420p | PixelFormat::Yuv444p | PixelFormat::Yuv422p => 8,
            PixelFormat::Yuv420p10 | PixelFormat::Yuv444p10 => 10,
            _ => 12,
        };
        let noise = rng.below(3);
        let mut bytes = Vec::new();
        for (w, h) in [(width, height), (cw, ch), (cw, ch)] {
            for y in 0..h {
                for x in 0..w {
                    let jitter = if noise == 0 { 0 } else { rng.below(40) as u32 };
                    let v = ((x * 5 + y * 3 + index * 7 + jitter) % 220 + 16) << (depth - 8);
                    if depth > 8 {
                        bytes.extend_from_slice(&(v as u16).to_le_bytes());
                    } else {
                        bytes.push(v as u8);
                    }
                }
            }
        }
        let time = FrameTime::new(i64::from(index), 1, 30).expect("time");
        FrameInput::Cpu(CpuFrame::new(width, height, pixel, bytes, time).expect("frame"))
    }

    /// One random encode: its packets decode to as many pictures as went in,
    /// in the format that went in.
    fn encode_case(rng: &mut Rng) {
        let pixel = [
            PixelFormat::Yuv420p,
            PixelFormat::Yuv420p10,
            PixelFormat::Yuv422p,
            PixelFormat::Yuv444p,
            PixelFormat::Yuv444p10,
            PixelFormat::Yuv420p12,
        ][rng.below(6)];
        // Even sizes from 8 to 136, past one 64x64 superblock both ways.
        let width = 8 + 2 * rng.below(65) as u32;
        let height = 8 + 2 * rng.below(65) as u32;
        let frames = 1 + rng.below(7) as u32;
        let mut params = CodecParams::video(CodecId::Av1, width, height);
        params.pixel = pixel;
        if rng.below(3) == 0 {
            params.bitrate = 50_000 + rng.below(400_000) as u32;
        }
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_speed(6 + rng.below(5) as u8);
        encoder.set_gop(1 + rng.below(5) as u32);
        encoder.set_quality(20 + rng.below(220) as u8);
        encoder.set_keyint(1 + rng.below(6) as u32);
        encoder.set_threads(1 + rng.below(3));
        encoder.set_loop_restoration(rng.below(4) == 0);
        encoder.set_temporal_filter(rng.below(4) == 0);
        let force = rng.below(frames as usize) as u32;
        let mut packets = Vec::new();
        for index in 0..frames {
            if index == force && rng.below(2) == 0 {
                encoder.force_keyframe();
            }
            let input = picture(pixel, width, height, index, rng);
            packets.extend(encoder.push_frame(&input).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        let what = format!("{pixel:?} {width}x{height} {frames} frames");
        assert_eq!(packets.len(), frames as usize, "{what}");
        assert!(packets[0].keyframe, "{what}");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let mut shown = 0;
        let mut take = |outputs: Vec<Output>| {
            for output in outputs {
                let Output::Picture(picture) = output else {
                    panic!("a picture");
                };
                assert_eq!(picture.info().pixel, pixel, "{what}");
                assert_eq!(
                    (picture.info().width, picture.info().height),
                    (width, height)
                );
                shown += 1;
            }
        };
        for packet in &packets {
            take(
                decoder
                    .push(packet)
                    .unwrap_or_else(|e| panic!("{what}: {e}")),
            );
        }
        take(decoder.drain().unwrap_or_else(|e| panic!("{what}: {e}")));
        assert_eq!(shown, frames, "{what}");
    }

    #[test]
    fn encoder_seeds_round_trip() {
        count_panics();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..40 {
            encode_case(&mut rng);
        }
        assert_eq!(PANICS.load(Ordering::Relaxed), 0, "a thread panicked");
    }

    #[test]
    #[ignore = "a timed campaign: AV1_FUZZ_SECS seconds (60 by default)"]
    fn campaign() {
        count_panics();
        let seconds = std::env::var("AV1_FUZZ_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60u64);
        let seed = std::env::var("AV1_FUZZ_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0x1234_5678_9abc_def1u64);
        let corpus = corpus();
        let mut rng = Rng(seed | 1);
        let until = Instant::now() + Duration::from_secs(seconds);
        let (mut decoded, mut encoded) = (0u64, 0u64);
        while Instant::now() < until {
            for (index, (_, packets, clean)) in corpus.iter().enumerate() {
                case(packets, clean, 1 + (index + decoded as usize) % 4, &mut rng);
                decoded += 1;
            }
            for _ in 0..4 {
                encode_case(&mut rng);
                encoded += 1;
            }
        }
        eprintln!(
            "seed {seed:#x}: {decoded} damaged streams, {encoded} encodes, {} panics",
            PANICS.load(Ordering::Relaxed)
        );
        assert_eq!(PANICS.load(Ordering::Relaxed), 0, "a thread panicked");
    }
}
