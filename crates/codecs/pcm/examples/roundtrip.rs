//! Decode a tiny interleaved little-endian s16 fixture, encode those samples,
//! and decode the packet again.
//!
//! `cargo run --release -p cutlass-pcm --example roundtrip`

#![allow(clippy::print_stdout)]

use cutlass_decode::{
    ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet, Pcm, SampleFormat,
};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_pcm::{Arrangement, Coding, Endian, Layout, PcmDecoder, PcmEncoder};

/// Four stereo frames: silence, full scale, then a mid-level pair.
const FIXTURE: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x80, 0xe8, 0x03, 0x18, 0xfc, 0x2a, 0x00, 0xd6, 0xff,
];

const EXPECTED: [i16; 8] = [0, 0, 32_767, -32_768, 1_000, -1_000, 42, -42];

const RATE: u32 = 48_000;

fn main() -> Result<(), String> {
    let pcm = decode(&FIXTURE)?;
    let decoded = pcm
        .as_i16()
        .ok_or_else(|| "decoded PCM is not i16".to_string())?;
    if decoded != EXPECTED {
        return Err("decoded samples differ from the fixture".to_string());
    }
    let encoded = encode(&pcm)?;
    if encoded.as_slice() != FIXTURE.as_slice() {
        return Err("encoded bytes differ from the fixture".to_string());
    }
    let again = decode(&encoded)?;
    if again.as_i16() != Some(decoded) {
        return Err("round trip changed the samples".to_string());
    }
    let rate = pcm.rate;
    let channels = pcm.layout.len();
    let samples = pcm.samples;
    println!("rate {rate}");
    println!("channels {channels}");
    println!("samples {samples}");
    Ok(())
}

fn params() -> CodecParams {
    let spec = Layout::new(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let mut opened = CodecParams::audio(CodecId::Pcm, RATE, ChannelLayout::stereo());
    opened.extra = spec.to_extra().to_vec();
    opened.sample_format = Some(SampleFormat::I16);
    opened
}

fn decode(bytes: &[u8]) -> Result<Pcm, String> {
    let mut decoder = PcmDecoder::open(&params()).map_err(|err| err.to_string())?;
    let frames = bytes.len() / 4;
    let duration = i64::try_from(frames).map_err(|_| "frame count does not fit".to_string())?;
    let packet = Packet::audio(bytes.to_vec(), 0, duration, RATE);
    let outputs = decoder.push(&packet).map_err(|err| err.to_string())?;
    let [Output::Audio(audio)] = outputs.as_slice() else {
        return Err("decoder returned no audio".to_string());
    };
    Ok(audio.pcm.clone())
}

fn encode(pcm: &Pcm) -> Result<Vec<u8>, String> {
    let mut encoder = PcmEncoder::open(&params()).map_err(|err| err.to_string())?;
    let packets = encoder
        .push(&AudioInput { pcm: pcm.clone() })
        .map_err(|err| err.to_string())?;
    let [packet] = packets.as_slice() else {
        return Err("encoder returned no packet".to_string());
    };
    Ok(packet.data.clone())
}
