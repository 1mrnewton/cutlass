//! Campaigns that are not one fixture.
//!
//! The decoder fuzz and the encoder fuzz run for `MPEG4_FUZZ_SECS` and
//! `MPEG4_ENC_FUZZ_SECS` seconds. Two seconds is the default so `cargo test`
//! stays short. A recorded run sets the variable to 1800. Overflow checks
//! stay on when that run is a debug build; this package is optimised there.
//! The pixel cap is below the playback maximum so a hostile layer cannot
//! allocate an 8191-edge picture on every input.
//!
//! The release gates are ignored. Run them with `--release -- --ignored`.
//! Decode of 1280×720 must finish 30 pictures in one second, which is
//! realtime at 30 fps, and grow the resident set by at most 64 MiB. Encode
//! of 320×240 Advanced Simple must finish 30 pictures in five seconds.
//! That budget is the search: about 300 macroblocks, a hundred candidates,
//! two pictures a frame, thirty frames.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId as StreamCodecId, StreamCodec};
use cutlass_container::Demux;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder as Playback, Output, Packet, PictureSamples, Yuv420p,
};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_isobmff::IsoBmffFile;

use crate::bits::BitWriter;
use crate::dec::Decoder;
use crate::enc::{encode_advanced, encode_intra};
use crate::engine::{EngineDecoder, EngineEncoder};

const FUZZ_PIXELS: u64 = 1920 * 1088;
const RSS_BUDGET_KIB: u64 = 256 * 1024;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() as usize) % n
        }
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        for byte in bytes {
            *byte = self.byte();
        }
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn resident_kib() -> u64 {
    let pid = std::process::id().to_string();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p"])
        .arg(pid)
        .output()
        .expect("ps");
    assert!(out.status.success(), "ps did not report rss");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .and_then(|token| token.parse().ok())
        .expect("rss")
}

fn panic_text(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
        })
        .unwrap_or_else(|| "panic".to_string())
}

fn save_failure(seed: u64, inputs: u64, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mpeg4-fuzz-{seed:x}-{inputs}.bin"));
    std::fs::write(&path, bytes).expect("save the input that failed");
    path
}

/// A Simple VOL, preceded by its start code. The payload matches the
/// header tests, so a size the parser accepts reaches the pixel cap.
fn vol_packet(width: u32, height: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bytes(&[0, 0, 1, 0x20]);
    w.put(1, 0);
    w.put(8, 1);
    w.put(1, 0);
    w.put(4, 1);
    w.put(1, 0);
    w.put(2, 0);
    w.put(1, 1);
    w.put(16, 30_000);
    w.put(1, 1);
    w.put(1, 0);
    w.put(1, 1);
    w.put(13, width);
    w.put(1, 1);
    w.put(13, height);
    w.put(1, 1);
    w.put(1, 0);
    w.put(1, 1);
    w.put(1, 0);
    w.put(1, 0);
    w.put(1, 0);
    w.put(1, 1);
    w.put(1, 1);
    w.put(1, 0);
    w.put(1, 0);
    w.stuff();
    w.finish()
}

#[test]
fn a_layer_over_the_pixel_cap_is_refused() {
    let mut decoder = Decoder::with_max_pixels(320 * 240);
    let err = match decoder.decode(&vol_packet(640, 480)) {
        Err(err) => err,
        Ok(_) => panic!("a layer over the cap decoded"),
    };
    assert!(
        matches!(
            err,
            CodecError::Limit {
                what: "picture size"
            }
        ),
        "{err:?}"
    );
    decoder
        .decode(&vol_packet(160, 120))
        .expect("a layer under the cap");

    let mut params = CodecParams::video(CodecId::Mpeg4, 0, 0);
    params.extra = vol_packet(640, 480);
    let err = match EngineDecoder::open_limited(&params, 320 * 240) {
        Err(err) => err,
        Ok(_) => panic!("the engine accepted a layer over the cap"),
    };
    assert!(
        matches!(
            err,
            CodecError::Limit {
                what: "picture size"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_vop_in_the_configuration_is_refused() {
    let mut decoder = Decoder::with_max_pixels(FUZZ_PIXELS);
    let err = decoder
        .configure(&[0, 0, 1, 0xB6, 0])
        .expect_err("a picture in the configuration");
    assert!(matches!(err, CodecError::Malformed { .. }), "{err:?}");
}

fn corpus() -> Vec<Vec<u8>> {
    let y = vec![40u8; 16 * 16];
    let u = vec![128u8; 8 * 8];
    let v = vec![128u8; 8 * 8];
    let intra = encode_intra(&y, &u, &v, 16, 16, 8).expect("intra seed");
    let advanced = encode_advanced(
        &[(&y, &u, &v), (&y, &u, &v), (&y, &u, &v)],
        16,
        16,
        8,
        false,
        false,
    )
    .expect("advanced seed");
    vec![
        intra,
        advanced,
        include_bytes!("../tests/fixtures/asp.m4v").to_vec(),
        include_bytes!("../tests/fixtures/interlaced.m4v").to_vec(),
        include_bytes!("../tests/fixtures/partitioned.m4v").to_vec(),
        vec![0, 0, 1, 0xB6],
        Vec::new(),
    ]
}

fn good_intra() -> Vec<u8> {
    let y = vec![32u8; 16 * 16];
    let c = vec![120u8; 8 * 8];
    encode_intra(&y, &c, &c, 16, 16, 10).expect("good intra")
}

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut out = seed.to_vec();
    if out.is_empty() {
        out.push(rng.byte());
    }
    match rng.below(8) {
        0 => {
            let i = rng.below(out.len());
            let bit = u8::try_from(rng.below(8)).unwrap_or(0);
            out[i] ^= 1 << bit;
        }
        1 => {
            let i = rng.below(out.len());
            out[i] = rng.byte();
        }
        2 => out.truncate(rng.below(out.len())),
        3 if out.len() < 8192 => {
            let i = rng.below(out.len() + 1);
            out.insert(i, rng.byte());
        }
        4 if out.len() > 1 => {
            out.remove(rng.below(out.len()));
        }
        5 if out.len() + 8 < 8192 => {
            out.extend_from_slice(&[0, 0, 1, rng.byte()]);
            for _ in 0..rng.below(16) {
                out.push(rng.byte());
            }
        }
        6 => {
            out.clear();
            for _ in 0..rng.below(256) {
                out.push(rng.byte());
            }
        }
        _ if !out.is_empty() && out.len() < 8192 => {
            let start = rng.below(out.len());
            let len = rng.below(out.len() - start).min(64);
            let extra = out[start..start + len].to_vec();
            out.extend(extra);
        }
        _ => {}
    }
    out
}

fn core_and_engine(bytes: &[u8]) -> Result<(bool, usize), String> {
    let mut core = Decoder::with_max_pixels(FUZZ_PIXELS);
    let core_result = core.decode(bytes);
    let mut pictures = core_result
        .as_ref()
        .map_or(0, |decoded| decoded.pictures.len());
    pictures += core.drain().len();
    let mut engine =
        EngineDecoder::open_limited(&CodecParams::video(CodecId::Mpeg4, 0, 0), FUZZ_PIXELS)
            .map_err(|err| err.to_string())?;
    let packet = Packet::video(bytes.to_vec(), 0, 1, 25, false);
    let engine_result = Playback::push(&mut engine, &packet);
    let _ = Playback::drain(&mut engine);
    if core_result.is_err() != engine_result.is_err() {
        return Err(format!(
            "core {} engine {}",
            core_result.is_err(),
            engine_result.is_err()
        ));
    }
    Ok((core_result.is_err(), pictures))
}

fn configure_both(bytes: &[u8]) -> Result<(), String> {
    let mut core = Decoder::with_max_pixels(FUZZ_PIXELS);
    let core_result = core.configure(bytes);
    let mut params = CodecParams::video(CodecId::Mpeg4, 0, 0);
    params.extra = bytes.to_vec();
    let engine_result = EngineDecoder::open_limited(&params, FUZZ_PIXELS);
    if core_result.is_err() != engine_result.is_err() {
        return Err(format!(
            "configure core {} engine {}",
            core_result.is_err(),
            engine_result.is_err()
        ));
    }
    Ok(())
}

fn recovers(good: &[u8]) -> Result<(), String> {
    let mut core = Decoder::with_max_pixels(FUZZ_PIXELS);
    let _ = core.decode(&[0, 0, 1, 0xB6, 0xFF, 0x00]);
    core.reset();
    let decoded = core.decode(good).map_err(|err| err.to_string())?;
    let mut pictures = decoded.pictures.len();
    pictures += core.drain().len();
    if pictures == 0 {
        return Err("the good stream produced no picture after reset".to_string());
    }
    let mut engine =
        EngineDecoder::open_limited(&CodecParams::video(CodecId::Mpeg4, 16, 16), FUZZ_PIXELS)
            .map_err(|err| err.to_string())?;
    let _ = Playback::push(
        &mut engine,
        &Packet::video(vec![0, 0, 1, 0xB6, 0xFF], 0, 1, 25, false),
    );
    Playback::reset(&mut engine);
    let pushed = Playback::push(&mut engine, &Packet::video(good.to_vec(), 0, 1, 25, true))
        .map_err(|err| err.to_string())?;
    let drained = Playback::drain(&mut engine).map_err(|err| err.to_string())?;
    if pushed.is_empty() && drained.is_empty() {
        return Err("the engine produced no picture after reset".to_string());
    }
    Ok(())
}

#[test]
fn decoder_fuzz_does_not_panic() {
    let secs = env_u64("MPEG4_FUZZ_SECS", 2);
    let seed = env_u64("MPEG4_FUZZ_SEED", 0x4D50_4534_4432);
    let corpus = corpus();
    let good = good_intra();
    let mut rng = Rng(seed);
    let started = Instant::now();
    let limit = Duration::from_secs(secs);
    let baseline = resident_kib();
    let mut inputs = 0u64;
    let mut pictures = 0usize;
    let mut errors = 0u64;
    let mut mark = 0u64;
    while started.elapsed() < limit {
        let elapsed = started.elapsed().as_secs();
        if elapsed / 30 != mark {
            mark = elapsed / 30;
            eprintln!("mpeg4 decoder fuzz progress inputs={inputs} seconds={elapsed}");
        }
        let choice = rng.below(corpus.len());
        let bytes = if rng.below(6) == 0 {
            corpus[choice].clone()
        } else {
            mutate(&mut rng, &corpus[choice])
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if rng.below(8) == 0 {
                configure_both(&bytes).map(|()| (false, 0))
            } else {
                core_and_engine(&bytes)
            }
        }));
        let (failed, count) = match outcome {
            Ok(Ok(pair)) => pair,
            Ok(Err(reason)) => {
                let path = save_failure(seed, inputs, &bytes);
                panic!(
                    "decoder finding {reason} seed={seed:#x} path={}",
                    path.display()
                );
            }
            Err(payload) => {
                let path = save_failure(seed, inputs, &bytes);
                panic!(
                    "decoder panicked seed={seed:#x} path={} ({})",
                    path.display(),
                    panic_text(&payload)
                );
            }
        };
        if failed {
            errors += 1;
        }
        pictures += count;
        if let Err(reason) = recovers(&good) {
            panic!("decoder did not recover seed={seed:#x} input={inputs}: {reason}");
        }
        inputs += 1;
        if inputs.is_multiple_of(1_000) {
            let growth = resident_kib().saturating_sub(baseline);
            assert!(growth <= RSS_BUDGET_KIB, "decoder fuzz grew {growth} KiB");
        }
    }
    eprintln!(
        "mpeg4 decoder fuzz seed={seed:#x} seconds={secs} inputs={inputs} pictures={pictures} errors={errors} findings=0"
    );
    assert!(inputs > 100, "the decoder campaign barely ran: {inputs}");
}

fn noise(rng: &mut Rng, width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    rng.fill(&mut y);
    rng.fill(&mut u);
    rng.fill(&mut v);
    (y, u, v)
}

fn pictures_of(bytes: &[u8]) -> Result<usize, String> {
    let mut decoder = Decoder::with_max_pixels(FUZZ_PIXELS);
    let decoded = decoder.decode(bytes).map_err(|err| err.to_string())?;
    let mut pictures = decoded.pictures.len();
    pictures += decoder.drain().len();
    Ok(pictures)
}

fn batch_round_trip(rng: &mut Rng) -> Result<(), String> {
    let width = if rng.below(2) == 0 { 16 } else { 32 };
    let height = 16;
    let count = 1 + rng.below(3);
    let quant = u8::try_from(1 + rng.below(31)).unwrap_or(8);
    let mut frames = Vec::new();
    let mut owned = Vec::new();
    for _ in 0..count {
        owned.push(noise(rng, width, height));
    }
    for (y, u, v) in &owned {
        frames.push((y.as_slice(), u.as_slice(), v.as_slice()));
    }
    let intra = rng.below(2) == 0;
    let (bytes, expected) = if intra {
        let (y, u, v) = &owned[0];
        (
            encode_intra(y, u, v, width, height, quant).map_err(|err| err.to_string())?,
            1,
        )
    } else {
        (
            encode_advanced(
                &frames,
                width,
                height,
                quant,
                rng.below(3) == 0,
                rng.below(4) == 0,
            )
            .map_err(|err| err.to_string())?,
            count,
        )
    };
    let pictures = pictures_of(&bytes)?;
    if pictures != expected {
        return Err(format!(
            "encoded {expected} pictures and decoded {pictures}"
        ));
    }
    Ok(())
}

fn engine_round_trip(rng: &mut Rng) -> Result<(), String> {
    let (width, height) = [(16u32, 16u32), (32, 16)][rng.below(2)];
    let mut params = CodecParams::video(CodecId::Mpeg4, width, height);
    params.bitrate = [0, 50_000, 400_000, u32::MAX][rng.below(4)];
    params.constant_rate = rng.below(2) == 0;
    let mut encoder = EngineEncoder::open(&params).map_err(|err| err.to_string())?;
    let frames = 1 + rng.below(3);
    let mut pictures = Vec::new();
    for n in 0..frames {
        let (y, u, v) = noise(rng, width, height);
        let picture = Yuv420p::try_new(width, height, y, u, v).map_err(|err| err.to_string())?;
        let pts = i64::try_from(n).unwrap_or(0);
        pictures.extend(
            encoder
                .push(&VideoInput {
                    picture,
                    pts,
                    duration: 1,
                    timescale: 25,
                    force_keyframe: n == 0 || rng.below(5) == 0,
                })
                .map_err(|err| err.to_string())?,
        );
    }
    pictures.extend(encoder.drain().map_err(|err| err.to_string())?);
    let mut params = CodecParams::video(CodecId::Mpeg4, width, height);
    params.extra = encoder.extradata().to_vec();
    let mut decoder = EngineDecoder::open(&params).map_err(|err| err.to_string())?;
    let mut shown = 0usize;
    for packet in &pictures {
        shown += Playback::push(&mut decoder, packet)
            .map_err(|err| err.to_string())?
            .len();
    }
    shown += Playback::drain(&mut decoder)
        .map_err(|err| err.to_string())?
        .len();
    if shown != frames {
        return Err(format!("export encoded {frames} and decoded {shown}"));
    }
    Ok(())
}

fn refused_encoder_calls() {
    assert!(EngineEncoder::open(&CodecParams::video(CodecId::Mpeg4, 15, 16)).is_err());
    assert!(EngineEncoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 16)).is_err());
    assert!(EngineEncoder::open(&CodecParams::video(CodecId::H264, 16, 16)).is_err());
    let params = CodecParams::video(CodecId::Mpeg4, 16, 16);
    let mut encoder = EngineEncoder::open(&params).expect("open");
    let picture = Yuv420p::solid(16, 16, 8, 128, 128).expect("picture");
    assert!(
        encoder
            .push(&VideoInput {
                picture,
                pts: 0,
                duration: 1,
                timescale: 0,
                force_keyframe: false,
            })
            .is_err()
    );
}

#[test]
fn encoder_fuzz_does_not_panic() {
    let secs = env_u64("MPEG4_ENC_FUZZ_SECS", 2);
    let seed = env_u64("MPEG4_ENC_FUZZ_SEED", 0x4D50_4534_454E);
    let mut rng = Rng(seed);
    let started = Instant::now();
    let limit = Duration::from_secs(secs);
    let baseline = resident_kib();
    let mut inputs = 0u64;
    let mut mark = 0u64;
    refused_encoder_calls();
    while started.elapsed() < limit {
        let elapsed = started.elapsed().as_secs();
        if elapsed / 30 != mark {
            mark = elapsed / 30;
            eprintln!("mpeg4 encoder fuzz progress inputs={inputs} seconds={elapsed}");
        }
        let engine = rng.below(8) == 0;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if engine {
                engine_round_trip(&mut rng)
            } else {
                batch_round_trip(&mut rng)
            }
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => panic!("encoder finding {reason} seed={seed:#x} input={inputs}"),
            Err(payload) => panic!(
                "encoder panicked seed={seed:#x} input={inputs} ({})",
                panic_text(&payload)
            ),
        }
        inputs += 1;
        if inputs.is_multiple_of(200) {
            let growth = resident_kib().saturating_sub(baseline);
            assert!(growth <= RSS_BUDGET_KIB, "encoder fuzz grew {growth} KiB");
        }
    }
    eprintln!("mpeg4 encoder fuzz seed={seed:#x} seconds={secs} inputs={inputs} findings=0");
    assert!(inputs > 20, "the encoder campaign barely ran: {inputs}");
}

fn tool(name: &str) -> Option<String> {
    [
        format!("/opt/homebrew/bin/{name}"),
        format!("/usr/local/bin/{name}"),
        name.to_string(),
    ]
    .into_iter()
    .find(|path| Command::new(path).arg("-version").output().is_ok())
}

fn pattern(width: u32, height: u32, shift: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = (((row + col + shift) * 3 + (col ^ row)) % 180 + 20) as u8;
        }
    }
    for row in 0..h / 2 {
        for col in 0..w / 2 {
            let at = row * (w / 2) + col;
            u[at] = ((col * 5 + row + shift) % 80 + 80) as u8;
            v[at] = ((row * 5 + col + shift / 2) % 80 + 80) as u8;
        }
    }
    (y, u, v)
}

/// Release measurement. `cargo test -p cutlass-mpeg4 --release --lib --
/// --ignored release_decode_and_encode_meet_their_gates --nocapture`.
#[test]
#[ignore = "release timing and memory; run with --release -- --ignored"]
fn release_decode_and_encode_meet_their_gates() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        eprintln!("release gates not run: ffmpeg is absent");
        return;
    };
    let dir = std::env::temp_dir().join("cutlass-mpeg4-gates");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("gate dir");
    let path = dir.join("720p.m4v");
    let made = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
            "30",
            "-c:v",
            "mpeg4",
            "-q:v",
            "8",
            "-f",
            "m4v",
        ])
        .arg(&path)
        .output()
        .expect("ffmpeg");
    assert!(
        made.status.success(),
        "ffmpeg did not make the 720p stream: {}",
        String::from_utf8_lossy(&made.stderr)
    );
    let bytes = std::fs::read(&path).expect("720p stream");
    let packets = vop_packets(&bytes);
    assert!(packets.len() >= 30, "vops {}", packets.len());
    let mut decoder =
        EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 1280, 720)).expect("decoder");
    let baseline = resident_kib();
    let started = Instant::now();
    let mut pictures = 0usize;
    let mut peak = baseline;
    for (index, packet) in packets.iter().enumerate() {
        let pts = i64::try_from(index).unwrap_or(0);
        pictures += Playback::push(
            &mut decoder,
            &Packet::video(packet.clone(), pts, 1, 30, index == 0),
        )
        .expect("decode")
        .len();
        if index.is_multiple_of(10) {
            peak = peak.max(resident_kib());
        }
    }
    pictures += Playback::drain(&mut decoder).expect("drain").len();
    peak = peak.max(resident_kib());
    let elapsed = started.elapsed().as_secs_f64();
    let extra_kib = peak.saturating_sub(baseline);
    eprintln!("mpeg4 720p decode pictures={pictures} seconds={elapsed:.3} extra_kib={extra_kib}");
    assert_eq!(pictures, 30, "{pictures}");
    assert!(
        elapsed <= 1.0,
        "720p decode {pictures} pictures in {elapsed:.3}s"
    );
    assert!(extra_kib <= 64 * 1024, "720p decode grew {extra_kib} KiB");

    let width = 320u32;
    let height = 240u32;
    let made: Vec<Yuv420p> = (0..30)
        .map(|n| {
            let (y, u, v) = pattern(width, height, n * 2);
            Yuv420p::try_new(width, height, y, u, v).expect("picture")
        })
        .collect();
    let mut params = CodecParams::video(CodecId::Mpeg4, width, height);
    params.bitrate = 800_000;
    params.constant_rate = true;
    let mut encoder = EngineEncoder::open(&params).expect("encoder");
    let baseline = resident_kib();
    let started = Instant::now();
    let mut packets = 0usize;
    let mut peak = baseline;
    for (n, picture) in made.into_iter().enumerate() {
        let pts = i64::try_from(n).unwrap_or(0);
        packets += encoder
            .push(&VideoInput {
                picture,
                pts,
                duration: 1,
                timescale: 30,
                force_keyframe: n == 0,
            })
            .expect("encode")
            .len();
        if n.is_multiple_of(10) {
            peak = peak.max(resident_kib());
        }
    }
    packets += encoder.drain().expect("drain").len();
    peak = peak.max(resident_kib());
    let elapsed = started.elapsed().as_secs_f64();
    let extra_kib = peak.saturating_sub(baseline);
    eprintln!("mpeg4 320x240 encode packets={packets} seconds={elapsed:.3} extra_kib={extra_kib}");
    assert_eq!(packets, 30, "{packets}");
    assert!(
        elapsed <= 5.0,
        "320x240 encode {packets} pictures in {elapsed:.3}s"
    );
    assert!(extra_kib <= 64 * 1024, "encode grew {extra_kib} KiB");
}

fn vop_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut cuts = Vec::new();
    let mut pending: Option<usize> = None;
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let code = data[i + 3];
            if code == 0xB6 {
                cuts.push(pending.take().unwrap_or(i));
            } else if pending.is_none() {
                pending = Some(i);
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (n, &start) in cuts.iter().enumerate() {
        let end = cuts.get(n + 1).copied().unwrap_or(data.len());
        out.push(data[start..end].to_vec());
    }
    out
}

fn local_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files")
}

fn is_media(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "mp4"
            | "mov"
            | "m4v"
            | "avi"
            | "mkv"
            | "webm"
            | "mpg"
            | "mpeg"
            | "3gp"
            | "ts"
            | "m2ts"
            | "wmv"
            | "flv"
            | "m4a"
            | "png"
            | "jpg"
    )
}

fn collect_media(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_media(&path, depth + 1, out);
        } else if is_media(&path) {
            out.push(path);
        }
    }
}

fn video_codec(ffprobe: &str, path: &Path) -> Option<String> {
    let out = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let codec = text.lines().next()?.trim();
    if codec.is_empty() {
        None
    } else {
        Some(codec.to_string())
    }
}

fn push_picture(frames: &mut Vec<Vec<u8>>, output: Output) {
    let Output::Picture(picture) = output else {
        return;
    };
    let PictureSamples::Cpu(samples) = picture.samples() else {
        panic!("the software decoder returned a hardware picture");
    };
    frames.push(samples.to_vec());
}

fn decode_isobmff(path: &Path, limit: usize) -> Option<Vec<Vec<u8>>> {
    let file = std::fs::File::open(path).ok()?;
    let mut movie = IsoBmffFile::open(file).ok()?;
    let prepared = {
        let stream = movie
            .streams()
            .iter()
            .find(|stream| stream.params().codec() == &StreamCodec::Known(StreamCodecId::Mpeg4))?;
        let size = stream.params().size()?;
        let mut params = CodecParams::video(CodecId::Mpeg4, size.width(), size.height());
        params.extra = stream.params().extradata().to_vec();
        (stream.id().get(), params)
    };
    let (track, params) = prepared;
    let mut decoder = EngineDecoder::open(&params).expect("decoder");
    let mut frames = Vec::new();
    while frames.len() < limit {
        let Some(packet) = movie.read_packet().expect("read") else {
            break;
        };
        if packet.stream() != track {
            continue;
        }
        let time = packet.timestamp();
        let access = Packet::video(
            packet.bytes().to_vec(),
            time.presentation_or_decode().unwrap_or(0),
            time.duration().unwrap_or(1),
            packet.timescale().ticks_per_second(),
            packet.keyframe(),
        );
        for output in Playback::push(&mut decoder, &access).expect("decode") {
            push_picture(&mut frames, output);
        }
    }
    for output in Playback::drain(&mut decoder).expect("drain") {
        if frames.len() >= limit {
            break;
        }
        push_picture(&mut frames, output);
    }
    Some(frames)
}

fn decode_bytes(bytes: &[u8], limit: usize) -> Result<Vec<Vec<u8>>, CodecError> {
    let mut decoder = EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0))?;
    let mut frames = Vec::new();
    for (index, packet) in vop_packets(bytes).into_iter().enumerate() {
        if frames.len() >= limit {
            break;
        }
        let pts = i64::try_from(index).unwrap_or(0);
        for output in Playback::push(&mut decoder, &Packet::video(packet, pts, 1, 25, index == 0))?
        {
            push_picture(&mut frames, output);
        }
    }
    for output in Playback::drain(&mut decoder)? {
        if frames.len() >= limit {
            break;
        }
        push_picture(&mut frames, output);
    }
    Ok(frames)
}

fn ffmpeg_yuv(ffmpeg: &str, path: &Path, frames: usize) -> Vec<u8> {
    let out = std::env::temp_dir().join("cutlass-mpeg4-local.yuv");
    let count = frames.to_string();
    let made = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-cpuflags",
            "0",
            "-flags",
            "+bitexact",
            "-idct",
            "simple",
            "-i",
        ])
        .arg(path)
        .args([
            "-an",
            "-frames:v",
            &count,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .output()
        .expect("ffmpeg");
    assert!(
        made.status.success(),
        "ffmpeg did not decode {}: {}",
        path.display(),
        String::from_utf8_lossy(&made.stderr)
    );
    std::fs::read(&out).expect("yuv")
}

fn elementary_bytes(ffmpeg: &str, path: &Path) -> Vec<u8> {
    let extracted = std::env::temp_dir().join("cutlass-mpeg4-local.m4v");
    let made = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "30", "-an", "-c:v", "copy", "-f", "m4v"])
        .arg(&extracted)
        .output()
        .expect("ffmpeg");
    assert!(
        made.status.success(),
        "ffmpeg did not copy {}: {}",
        path.display(),
        String::from_utf8_lossy(&made.stderr)
    );
    std::fs::read(&extracted).expect("elementary")
}

fn assert_matches_ffmpeg(ffmpeg: &str, path: &Path) {
    let frames = decode_isobmff(path, 30).unwrap_or_else(|| {
        let bytes = elementary_bytes(ffmpeg, path);
        decode_bytes(&bytes, 30).expect("decode")
    });
    assert!(!frames.is_empty(), "{} produced no picture", path.display());
    let yuv = ffmpeg_yuv(ffmpeg, path, frames.len());
    let mut offset = 0;
    for (index, frame) in frames.iter().enumerate() {
        let end = offset + frame.len();
        assert!(
            end <= yuv.len(),
            "frame {index} of {} ran past ffmpeg",
            path.display()
        );
        assert_eq!(
            &yuv[offset..end],
            frame.as_slice(),
            "frame {index} of {}",
            path.display()
        );
        offset = end;
    }
    assert_eq!(offset, yuv.len(), "{}", path.display());
}

#[test]
fn the_checked_in_asp_file_matches_ffmpeg() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        eprintln!("asp.mp4 was not compared: ffmpeg is absent");
        return;
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/asp.mp4");
    assert_matches_ffmpeg(&ffmpeg, &path);
}

#[test]
fn local_files_that_are_mpeg4_match_ffmpeg() {
    let root = local_root();
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return;
    }
    let Some(ffprobe) = tool("ffprobe") else {
        eprintln!("local-files not identified: ffprobe is absent");
        return;
    };
    let mut files = Vec::new();
    collect_media(&root, 0, &mut files);
    let mut probed = 0usize;
    let mut mpeg4 = Vec::new();
    for path in &files {
        probed += 1;
        if video_codec(&ffprobe, path).as_deref() == Some("mpeg4") {
            mpeg4.push(path.clone());
        }
    }
    eprintln!("local-files probed={probed} mpeg4={}", mpeg4.len());
    if mpeg4.is_empty() {
        return;
    }
    let Some(ffmpeg) = tool("ffmpeg") else {
        panic!(
            "{} MPEG-4 files are in local-files and ffmpeg is absent",
            mpeg4.len()
        );
    };
    for path in &mpeg4 {
        assert_matches_ffmpeg(&ffmpeg, path);
    }
}
