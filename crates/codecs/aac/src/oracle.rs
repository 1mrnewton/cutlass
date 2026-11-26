//! Decodes FFmpeg-written AAC-LC streams and compares the PCM with FFmpeg's
//! own decode of the same bytes.
//!
//! The streams and the reference PCM are committed under `tests/fixtures`,
//! so nothing here needs FFmpeg. `tests/fixtures/README.md` says how they
//! were made.

use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Packet};

use crate::decode::{AacDecoder, Census, census};

const NOPNS_44100: (&[u8], &[u8]) = (
    include_bytes!("../tests/fixtures/tools_44100_128k_nopns.aac"),
    include_bytes!("../tests/fixtures/tools_44100_128k_nopns.f32"),
);
const NOPNS_48000: (&[u8], &[u8]) = (
    include_bytes!("../tests/fixtures/tools_48000_64k_nopns.aac"),
    include_bytes!("../tests/fixtures/tools_48000_64k_nopns.f32"),
);
const PNS_44100: (&[u8], &[u8]) = (
    include_bytes!("../tests/fixtures/tools_44100_96k_pns.aac"),
    include_bytes!("../tests/fixtures/tools_44100_96k_pns.f32"),
);

/// The raw access units of an ADTS stream. Every fixture is written without
/// a CRC and with one raw data block per frame.
fn access_units(adts: &[u8]) -> Vec<&[u8]> {
    let mut units = Vec::new();
    let mut at = 0usize;
    while at + 7 <= adts.len() {
        let header = &adts[at..at + 7];
        assert_eq!(header[0], 0xFF, "sync at {at}");
        assert_eq!(header[1] & 0xF1, 0xF1, "sync, layer and no crc at {at}");
        let len = (usize::from(header[3] & 3) << 11)
            | (usize::from(header[4]) << 3)
            | usize::from(header[5] >> 5);
        assert!(len > 7 && at + len <= adts.len(), "frame length at {at}");
        units.push(&adts[at + 7..at + len]);
        at += len;
    }
    assert_eq!(at, adts.len(), "trailing bytes");
    units
}

fn reference(bytes: &[u8]) -> Vec<f32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn open(rate: u32) -> AacDecoder {
    AacDecoder::open(&CodecParams::audio(
        CodecId::Aac,
        rate,
        ChannelLayout::stereo(),
    ))
    .unwrap()
}

/// Our interleaved stereo PCM, one frame per access unit, and the tools the
/// stream used. FFmpeg writes one frame per access unit as well, so the two
/// line up sample for sample with nothing trimmed.
fn decode(adts: &[u8], rate: u32, seed: Option<u32>) -> (Vec<f32>, Census) {
    let mut decoder = open(rate);
    if let Some(seed) = seed {
        decoder.set_noise_state(seed);
    }
    let mut pcm = Vec::new();
    let mut tools = Census::default();
    for unit in access_units(adts) {
        tools.add(census(decoder.config(), unit).unwrap());
        let packet = Packet::audio(unit.to_vec(), 0, 1024, rate);
        pcm.extend(decoder.push_packet(&packet).unwrap());
    }
    (pcm, tools)
}

struct Difference {
    psnr: f64,
    largest: f32,
}

/// PSNR against a full scale of 1.0, and the largest single difference.
fn difference(ours: &[f32], theirs: &[f32]) -> Difference {
    assert_eq!(ours.len(), theirs.len());
    let mut sum = 0.0f64;
    let mut largest = 0.0f32;
    for (a, b) in ours.iter().zip(theirs) {
        let diff = a - b;
        sum += f64::from(diff) * f64::from(diff);
        largest = largest.max(diff.abs());
    }
    let mse = sum / ours.len() as f64;
    Difference {
        psnr: if mse > 0.0 {
            -10.0 * mse.log10()
        } else {
            f64::INFINITY
        },
        largest,
    }
}

fn energy(pcm: &[f32]) -> f64 {
    pcm.iter().map(|s| f64::from(*s) * f64::from(*s)).sum()
}

/// The floor for a stream that should match FFmpeg's float decoder. The two
/// differ only in the rounding of their filterbanks: about 150 dB and
/// 2.4e-7 when this was written.
const PSNR_FLOOR: f64 = 120.0;
const LARGEST: f32 = 1.0e-5;

fn assert_matches(name: &str, ours: &[f32], theirs: &[f32]) {
    let found = difference(ours, theirs);
    assert!(
        found.psnr >= PSNR_FLOOR && found.largest <= LARGEST,
        "{name}: {:.2} dB, largest difference {:e}",
        found.psnr,
        found.largest
    );
}

/// TNS on long and short windows, intensity stereo and mid/side, with noise
/// substitution off, at two sample rates.
#[test]
fn streams_with_tns_intensity_and_mid_side_match_ffmpeg() {
    for (name, (adts, pcm), rate) in [
        ("44100 Hz 128k", NOPNS_44100, 44_100),
        ("48000 Hz 64k", NOPNS_48000, 48_000),
    ] {
        let (ours, tools) = decode(adts, rate, None);
        assert!(tools.long_tns_filters > 0, "{name} {tools:?}");
        assert!(
            tools.tns_filters > tools.long_tns_filters,
            "{name} {tools:?}"
        );
        assert!(tools.intensity_bands > 0, "{name} {tools:?}");
        assert!(tools.ms_bands > 0, "{name} {tools:?}");
        assert!(tools.short_channels > 0, "{name} {tools:?}");
        assert_eq!(tools.noise_bands, 0, "{name}");
        assert_matches(name, &ours, &reference(pcm));
    }
}

/// With noise substitution on. The generator is FFmpeg's and is advanced in
/// the same order, so even the noise bands match sample for sample.
#[test]
fn a_stream_with_noise_substitution_matches_ffmpeg() {
    let (adts, pcm) = PNS_44100;
    let (ours, tools) = decode(adts, 44_100, None);
    assert!(tools.noise_bands > 0, "{tools:?}");
    assert!(tools.tns_filters > 0, "{tools:?}");
    assert!(tools.intensity_bands > 0, "{tools:?}");
    assert!(tools.ms_bands > 0, "{tools:?}");
    assert_matches("44100 Hz 96k pns", &ours, &reference(pcm));
}

/// From another generator state the noise is different noise, but each
/// band still gets the energy the stream asks for: every frame's energy
/// stays within 3% of FFmpeg's and the whole stream's within 0.1%.
#[test]
fn noise_bands_keep_ffmpegs_energy_from_another_generator_state() {
    let (adts, pcm) = PNS_44100;
    let theirs = reference(pcm);
    let (ours, _) = decode(adts, 44_100, Some(0x0BAD_5EED));
    let found = difference(&ours, &theirs);
    assert!(
        found.psnr < 60.0,
        "the noise did not change: {}",
        found.psnr
    );
    let whole = energy(&ours) / energy(&theirs);
    assert!((whole - 1.0).abs() < 1.0e-3, "{whole}");
    let mut compared = 0usize;
    for (index, (a, b)) in ours.chunks(2048).zip(theirs.chunks(2048)).enumerate() {
        if energy(b) < 1.0e-3 {
            continue;
        }
        let ratio = energy(a) / energy(b);
        assert!((ratio - 1.0).abs() < 0.03, "frame {index}: {ratio}");
        compared += 1;
    }
    assert!(compared > 10, "{compared}");
}

/// The streams without noise bands do not depend on the generator at all.
#[test]
fn a_stream_without_noise_ignores_the_generator_state() {
    let (adts, _) = NOPNS_44100;
    assert_eq!(
        decode(adts, 44_100, None).0,
        decode(adts, 44_100, Some(0x0BAD_5EED)).0
    );
}

/// Half of each access unit is pushed before the whole of it. Every half
/// fails, and none of them moves the overlap or the noise generator: the
/// PCM is the same as without them.
#[test]
fn truncated_units_between_good_ones_change_nothing() {
    let (adts, pcm) = PNS_44100;
    let mut decoder = open(44_100);
    let mut ours = Vec::new();
    for unit in access_units(adts) {
        let state = decoder.noise_state();
        let frames = decoder.frame_index();
        let half = Packet::audio(unit[..unit.len() / 2].to_vec(), 0, 1024, 44_100);
        assert!(decoder.push_packet(&half).is_err());
        assert_eq!(decoder.noise_state(), state);
        assert_eq!(decoder.frame_index(), frames);
        let packet = Packet::audio(unit.to_vec(), 0, 1024, 44_100);
        ours.extend(decoder.push_packet(&packet).unwrap());
    }
    assert_eq!(ours, decode(adts, 44_100, None).0);
    assert_matches("44100 Hz 96k pns", &ours, &reference(pcm));
}

/// FFmpeg and this decoder both give one 1024-sample stereo frame per
/// access unit, which is why the comparisons need no alignment.
#[test]
fn every_unit_decodes_to_one_frame() {
    for ((adts, pcm), rate) in [
        (NOPNS_44100, 44_100),
        (NOPNS_48000, 48_000),
        (PNS_44100, 44_100),
    ] {
        let units = access_units(adts);
        let (ours, _) = decode(adts, rate, None);
        assert_eq!(ours.len(), units.len() * 2048);
        assert_eq!(ours.len() * 4, pcm.len());
    }
}

const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Compares any mono or stereo ADTS file with FFmpeg's decode of it. Set
/// `AAC_ORACLE_ADTS` to the file and `AAC_ORACLE_PCM` to the output of
/// `ffmpeg -c:a aac -i <file> -f f32le <pcm>`; without them the test is
/// skipped. It prints the PSNR and the largest difference and holds them
/// to the same floor as the fixtures.
#[test]
fn an_outside_stream_matches_ffmpeg() {
    let (Some(adts), Some(pcm)) = (
        std::env::var_os("AAC_ORACLE_ADTS"),
        std::env::var_os("AAC_ORACLE_PCM"),
    ) else {
        eprintln!("skipped outside stream: AAC_ORACLE_ADTS and AAC_ORACLE_PCM are not set");
        return;
    };
    let adts = std::fs::read(adts).unwrap();
    let theirs = reference(&std::fs::read(pcm).unwrap());
    let rate = RATES[usize::from((adts[2] >> 2) & 15)];
    let channels = usize::from(((adts[2] & 1) << 2) | (adts[3] >> 6));
    let layout = match channels {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        other => panic!("channel configuration {other}"),
    };
    let mut decoder = AacDecoder::open(&CodecParams::audio(CodecId::Aac, rate, layout)).unwrap();
    let mut ours = Vec::with_capacity(theirs.len());
    let mut tools = Census::default();
    for unit in access_units(&adts) {
        tools.add(census(decoder.config(), unit).unwrap());
        let packet = Packet::audio(unit.to_vec(), 0, 1024, rate);
        ours.extend(decoder.push_packet(&packet).unwrap());
    }
    assert_eq!(ours.len(), theirs.len(), "sample count");
    let found = difference(&ours, &theirs);
    eprintln!(
        "outside stream: {rate} Hz, {channels} channels, {} samples, {:.2} dB PSNR, largest difference {:e}, {tools:?}",
        ours.len() / channels,
        found.psnr,
        found.largest
    );
    assert_matches("outside stream", &ours, &theirs);
}
