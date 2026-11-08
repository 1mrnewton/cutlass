//! Streams this crate encodes, checked by `flac -t`, `flac -d`, and
//! ffmpeg, and their size against libFLAC at the same level.

use std::path::{Path, PathBuf};

use cutlass_flac::{FlacEncoder, Settings, StreamInfo};

use crate::decode_oracle::{CASES, encode_case};
use crate::support::{
    Kind, ffmpeg_samples, flac_samples, have, path_str, read_wav, run, scratch, signal, succeeds,
    wav_rate,
};

/// Encodes interleaved `samples` at `level` and writes a native file:
/// the marker, the STREAMINFO block as the last metadata block, and the
/// frames. Returns the file size.
pub fn encode_file(
    path: &Path,
    samples: &[i32],
    rate: u32,
    channels: usize,
    bits: u32,
    settings: Settings,
) -> usize {
    let mut encoder = FlacEncoder::new(
        rate,
        u8::try_from(channels).expect("channels"),
        u8::try_from(bits).expect("bits"),
        settings,
    )
    .expect("encoder");
    let mut frames = Vec::new();
    for chunk in samples.chunks(4410 * channels) {
        frames.extend(encoder.encode(chunk).expect("encode"));
    }
    frames.extend(encoder.finish().expect("finish"));
    let info: StreamInfo = encoder.stream_info();
    let mut file = b"fLaC".to_vec();
    file.extend_from_slice(&[0x80, 0, 0, 34]);
    file.extend_from_slice(&info.to_bytes());
    for frame in &frames {
        file.extend_from_slice(&frame.data);
    }
    std::fs::write(path, &file).expect("write");
    file.len()
}

fn level(level: u8) -> Settings {
    Settings::level(level).expect("level")
}

fn check(path: &Path, source: &[i32], bits: u32, channels: usize, name: &str) {
    let (ok, stderr) = succeeds("flac", &["-s", "-t", path_str(path)]);
    assert!(ok, "{name}: flac -t refused: {stderr}");
    let (reference, depth, count) = flac_samples(path);
    assert_eq!((depth, count), (bits, channels), "{name}");
    assert!(reference == source, "{name}: flac -d differs");
    assert!(
        ffmpeg_samples(path, bits) == source,
        "{name}: ffmpeg differs"
    );
}

#[test]
fn our_streams_pass_flac_and_ffmpeg_bit_exactly() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("encode-oracle");
    for case in CASES {
        let source = signal(
            case.kind,
            case.channels,
            case.bits,
            case.frames,
            0xC0DE ^ case.frames as u64,
        );
        let levels: &[u8] = if case.name == "s16-stereo-l5" {
            &[0, 1, 2, 3, 4, 5, 6, 7, 8]
        } else {
            &[0, 5, 8]
        };
        for &l in levels {
            let path = dir.join(format!("{}-ours-{l}.flac", case.name));
            encode_file(
                &path,
                &source,
                case.rate,
                case.channels,
                case.bits,
                level(l),
            );
            check(
                &path,
                &source,
                case.bits,
                case.channels,
                &format!("{} level {l}", case.name),
            );
        }
        eprintln!(
            "{}: levels {levels:?} pass flac -t, flac -d, and ffmpeg",
            case.name
        );
    }
}

#[test]
fn extreme_settings_pass_flac_and_ffmpeg() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("encode-extreme");
    let source = signal(Kind::Music, 2, 24, 80_000, 3);
    let mut big = level(8);
    big.block_size = 65_535;
    big.max_lpc_order = 32;
    big.max_partition_order = 15;
    big.exhaustive_order = true;
    let mut tiny = level(8);
    tiny.block_size = 16;
    let mut odd = level(5);
    odd.block_size = 1000;
    for (name, settings) in [("big", big), ("tiny", tiny), ("odd", odd)] {
        let path = dir.join(format!("{name}.flac"));
        encode_file(&path, &source, 48_000, 2, 24, settings);
        check(&path, &source, 24, 2, name);
    }
}

/// libFLAC's size for `wav` with `args`.
fn libflac_size(wav: &Path, args: &[&str], out: &Path) -> usize {
    let mut all = vec!["-s", "-f", "--no-padding", "--no-seektable"];
    all.extend_from_slice(args);
    all.extend_from_slice(&[path_str(wav), "-o", path_str(out)]);
    run("flac", &all);
    usize::try_from(std::fs::metadata(out).expect("size").len()).expect("size")
}

/// Real audio from `../local-files`, the first 60 seconds as 16-bit WAV.
fn local_sources(dir: &Path) -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return Vec::new();
    }
    let picks = [
        ("speech-16k-mono", "GOOGLE IS BACK! (Gemini 3.8 Flash).wav"),
        ("m4a-stereo", "videoplayback.m4a"),
        ("rain-mp3", "rain_bedroom_night_30min.mp3"),
        (
            "expo-m4a",
            "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a",
        ),
    ];
    let mut out = Vec::new();
    for (name, file) in picks {
        let source = root.join(file);
        if !source.is_file() {
            eprintln!("skipped {name}: {file} is missing");
            continue;
        }
        let wav = dir.join(format!("{name}.wav"));
        run(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-t",
                "60",
                "-i",
                path_str(&source),
                "-c:a",
                "pcm_s16le",
                path_str(&wav),
            ],
        );
        out.push((name.to_owned(), wav));
    }
    out
}

#[test]
fn compression_against_libflac_is_recorded() {
    if !have("flac") || !have("ffmpeg") {
        return;
    }
    let dir = scratch("ratio");
    let mut sources: Vec<(String, PathBuf, Vec<i32>, u32, usize, u32)> = Vec::new();
    for case in CASES.iter().filter(|case| {
        case.kind == Kind::Music
            && matches!(case.name, "s16-stereo-l5" | "s24-stereo-l8" | "s24-6ch-96k")
    }) {
        let (source, _) = encode_case(case, &dir);
        let wav = dir.join(format!("{}.wav", case.name));
        sources.push((
            case.name.to_owned(),
            wav,
            source,
            case.bits,
            case.channels,
            case.rate,
        ));
    }
    for (name, wav) in local_sources(&dir) {
        let (samples, bits, channels) = read_wav(&wav);
        let rate = wav_rate(&wav);
        sources.push((name, wav, samples, bits, channels, rate));
    }
    let mut worst5 = 0.0f64;
    for (name, wav, samples, bits, channels, rate) in &sources {
        let raw = samples.len() * (*bits as usize).div_ceil(8);
        let mut line = format!("{name}: raw {raw}");
        for l in [0u8, 5, 8] {
            let ours = dir.join(format!("{name}-ours-{l}.flac"));
            let ours_size = encode_file(&ours, samples, *rate, *channels, *bits, level(l));
            check(&ours, samples, *bits, *channels, name);
            let theirs = dir.join(format!("{name}-libflac-{l}.flac"));
            let theirs_size = libflac_size(wav, &[&format!("-{l}")], &theirs);
            let ratio = ours_size as f64 / theirs_size as f64;
            if l == 5 {
                worst5 = worst5.max(ratio);
            }
            line.push_str(&format!(
                ", -{l} ours {ours_size} libflac {theirs_size} ({:+.2}%)",
                (ratio - 1.0) * 100.0
            ));
        }
        eprintln!("{line}");
    }
    eprintln!(
        "worst level 5 size against flac -5: {:+.2}%",
        (worst5 - 1.0) * 100.0
    );
    assert!(worst5 < 1.02, "level 5 is more than 2% larger than flac -5");
}
