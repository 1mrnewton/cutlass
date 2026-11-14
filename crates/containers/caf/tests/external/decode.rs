//! PCM a CAF file holds, read by this crate and decoded by `cutlass-pcm`,
//! against AudioToolbox's decode of the same file. Bit-exact. Skips, and
//! says so, without afconvert (CI runs on Linux).

use std::fs::File;
use std::path::Path;

use cutlass_caf::CafMux;
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_decode::{Channel, ChannelLayout, CodecParams, Decoder, Output};
use cutlass_pcm::PcmDecoder;

use crate::support::{TempDir, open, path, run, source_frames, tool, wave, wave_data};

fn apple() -> bool {
    if tool("afconvert") {
        return true;
    }
    eprintln!("skipped: afconvert is not on PATH (it comes with macOS)");
    false
}

#[derive(Debug, PartialEq)]
enum Samples {
    I16(Vec<i16>),
    F32(Vec<u32>),
}

/// Every packet of the file's one stream through the Rust PCM decoder.
/// Floats are compared as bits.
fn decode(file: &Path) -> Samples {
    let mut demux = open(file);
    let stream = &demux.streams()[0];
    assert_eq!(
        stream.params().codec(),
        &StreamCodec::Known(CodecId::Pcm),
        "{}",
        file.display()
    );
    let channels = usize::from(stream.params().channels().expect("channels"));
    let layout = ChannelLayout::new(vec![Channel::FrontLeft; channels]).expect("layout");
    let rate = stream.params().sample_rate().expect("rate");
    let mut params = CodecParams::audio(CodecId::Pcm, rate, layout);
    params.extra = stream.params().extradata().to_vec();
    params.sample_format = None;
    let mut decoder = PcmDecoder::open(&params).expect("decoder");
    let mut ints = Vec::new();
    let mut floats = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        let input = cutlass_decode::Packet::audio(packet.bytes().to_vec(), 0, 1, rate);
        for output in decoder.push(&input).expect("decode") {
            let Output::Audio(audio) = output else {
                panic!("not audio");
            };
            if let Some(samples) = audio.pcm.as_i16() {
                ints.extend_from_slice(samples);
            } else {
                let samples = audio.pcm.as_f32().expect("f32");
                floats.extend(samples.iter().map(|sample| sample.to_bits()));
            }
        }
    }
    assert!(ints.is_empty() || floats.is_empty());
    if floats.is_empty() {
        Samples::I16(ints)
    } else {
        Samples::F32(floats)
    }
}

/// AudioToolbox's decode of `file` to little-endian integers of `bits`.
fn apple_ints(file: &Path, dir: &TempDir, bits: u32) -> Vec<i64> {
    let stem = file
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("stem");
    let out = dir.join(&format!("{stem}-{bits}.wav"));
    let format = format!("LEI{bits}");
    run(
        "afconvert",
        &["-f", "WAVE", "-d", &format, path(file), path(&out)],
    )
    .expect("afconvert decode");
    let width = usize::try_from(bits / 8).expect("width");
    wave_data(&std::fs::read(&out).expect("decoded"))
        .chunks_exact(width)
        .map(|word| {
            let mut value = 0u64;
            for (index, byte) in word.iter().enumerate() {
                value |= u64::from(*byte) << (8 * index);
            }
            #[allow(clippy::cast_possible_wrap)]
            let signed = (value << (64 - bits)) as i64;
            signed >> (64 - bits)
        })
        .collect()
}

/// What `cutlass-pcm` should give for AudioToolbox's integers: the same
/// value for 16 bits, and the integer over the largest positive one of its
/// width as `f32` for 24 and 32 bits.
fn expected(ints: &[i64], bits: u32) -> Samples {
    if bits == 16 {
        return Samples::I16(
            ints.iter()
                .map(|value| i16::try_from(*value).expect("i16"))
                .collect(),
        );
    }
    #[allow(clippy::cast_precision_loss)]
    let full = ((1i64 << (bits - 1)) - 1) as f64;
    Samples::F32(
        ints.iter()
            .map(|value| {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let unit = (*value as f64 / full) as f32;
                unit.to_bits()
            })
            .collect(),
    )
}

/// afconvert's 8-bit, A-law and mu-law files decode to what AudioToolbox
/// gives for them as 16-bit.
#[test]
fn one_byte_pcm_from_afconvert_decodes_like_audiotoolbox() {
    if !apple() {
        return;
    }
    let dir = TempDir::new("decode-one-byte");
    let source = dir.join("source.wav");
    std::fs::write(&source, wave(&source_frames(44_100), 44_100, 2)).expect("source");
    for format in ["I8", "alaw", "ulaw"] {
        let file = dir.join(&format!("{format}.caf"));
        run(
            "afconvert",
            &["-f", "caff", "-d", format, path(&source), path(&file)],
        )
        .expect("afconvert encode");
        let ours = decode(&file);
        let Samples::I16(ref samples) = ours else {
            panic!("{format}: not i16");
        };
        assert_eq!(samples.len(), 44_100 * 2, "{format}");
        assert_eq!(ours, expected(&apple_ints(&file, &dir, 16), 16), "{format}");
    }
}

fn write_pcm(file: &Path, extradata: Vec<u8>, payload: Vec<u8>, frames: i64) {
    let rate = Timescale::new(8_000);
    let mut mux = CafMux::new(File::create(file).expect("create"));
    mux.add_stream(NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 8_000, 1, extradata)
            .expect("params"),
        timescale: rate,
        language: None,
    })
    .expect("add");
    let time = Timestamp::try_new(Some(0), Some(0), Some(frames)).expect("time");
    let packet = Packet::try_new(0, payload, time, rate, true).expect("packet");
    mux.write_packet(&packet).expect("write");
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: rate,
        ticks: frames,
    }])
    .expect("finish");
}

/// All 256 codes of signed 8-bit, A-law and mu-law, in a file this crate
/// writes, decode to what AudioToolbox gives for them.
#[test]
fn every_one_byte_code_decodes_like_audiotoolbox() {
    if !apple() {
        return;
    }
    let dir = TempDir::new("decode-codes");
    let codes: Vec<u8> = (0..=255).collect();
    for (coding, name) in [(6u8, "s8"), (7, "alaw"), (8, "ulaw")] {
        let file = dir.join(&format!("{name}.caf"));
        write_pcm(&file, vec![coding, 1, 0], codes.clone(), 256);
        let ours = decode(&file);
        assert_eq!(ours, expected(&apple_ints(&file, &dir, 16), 16), "{name}");
    }
}

/// A CAF file by hand: `desc` with the given layout, then `data`.
fn caf_bytes(bits: u32, word: u32, little: bool, channels: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = b"caff".to_vec();
    out.extend(1u16.to_be_bytes());
    out.extend(0u16.to_be_bytes());
    out.extend(b"desc");
    out.extend(32i64.to_be_bytes());
    out.extend(44_100f64.to_be_bytes());
    out.extend(b"lpcm");
    out.extend(u32::from(little).wrapping_mul(2).to_be_bytes());
    out.extend((word * channels).to_be_bytes());
    out.extend(1u32.to_be_bytes());
    out.extend(channels.to_be_bytes());
    out.extend(bits.to_be_bytes());
    out.extend(b"data");
    out.extend(
        i64::try_from(payload.len() + 4)
            .expect("size")
            .to_be_bytes(),
    );
    out.extend(0u32.to_be_bytes());
    out.extend(payload);
    out
}

/// Unpacked integer samples, where `mBitsPerChannel` is less than the
/// word: AudioToolbox decodes every bit of the word, and so does this crate
/// through the word's coding. Random words, so the low bits are not zero.
#[test]
fn unpacked_pcm_decodes_like_audiotoolbox() {
    if !apple() {
        return;
    }
    let dir = TempDir::new("decode-unpacked");
    let mut state = 0x756E_7061_636Bu64;
    let payload: Vec<u8> = (0..7_200)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[3]
        })
        .collect();
    for (bits, word) in [(8, 2), (12, 2), (20, 3), (16, 4), (20, 4), (24, 4)] {
        for little in [true, false] {
            let name = format!("u{bits}-{word}-{}", if little { "le" } else { "be" });
            let file = dir.join(&format!("{name}.caf"));
            std::fs::write(&file, caf_bytes(bits, word, little, 2, &payload)).expect("write");
            let ours = decode(&file);
            let theirs = apple_ints(&file, &dir, word * 8);
            assert_eq!(
                theirs.len(),
                payload.len() / usize::try_from(word).expect("w")
            );
            assert_eq!(ours, expected(&theirs, word * 8), "{name}");
        }
    }
}
