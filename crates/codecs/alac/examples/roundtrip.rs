//! Encode one second of a stereo 16-bit tone, decode the frames, and check
//! that every sample came back.
//!
//! `cargo run --release -p cutlass-alac --example roundtrip`

#![allow(clippy::print_stdout)]

use cutlass_alac::{AlacDecoder, AlacEncoder, EncoderConfig};
use cutlass_decode::ChannelLayout;
use cutlass_encode::Encoder;

const RATE: u32 = 44_100;

fn main() -> Result<(), String> {
    let samples: Vec<i32> = (0..RATE)
        .flat_map(|i| {
            let t = f64::from(i) / f64::from(RATE);
            let left = (16_000.0 * (std::f64::consts::TAU * 440.0 * t).sin()).round() as i32;
            let right = (12_000.0 * (std::f64::consts::TAU * 660.0 * t).sin()).round() as i32;
            [left, right]
        })
        .collect();

    let mut encoder = AlacEncoder::new(EncoderConfig::new(RATE, 2, 16), &ChannelLayout::stereo())
        .map_err(|err| err.to_string())?;
    let mut packets = encoder
        .push_samples(&samples)
        .map_err(|err| err.to_string())?;
    packets.extend(encoder.drain().map_err(|err| err.to_string())?);
    let cookie = encoder.cookie().to_bytes();

    let mut decoder = AlacDecoder::new(&cookie).map_err(|err| err.to_string())?;
    let mut decoded = Vec::with_capacity(samples.len());
    for packet in &packets {
        let frame = decoder
            .decode_frame(&packet.data)
            .map_err(|err| err.to_string())?;
        decoded.extend_from_slice(&frame.samples);
    }

    let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
    println!("frames {}", packets.len());
    println!("cookie {} bytes", cookie.len());
    println!("coded {bytes} bytes of {}", samples.len() * 2);
    println!("lossless {}", decoded == samples);
    if decoded != samples {
        return Err("decoded samples differ".to_string());
    }
    Ok(())
}
