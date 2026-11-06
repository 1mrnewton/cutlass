//! Encode raw interleaved little-endian `f32` to an MP3 file.
//!
//! `cargo run --release -p cutlass-mp3 --example encode -- in.f32 RATE CHANNELS out.mp3 [cbr KBPS | vbr QUALITY]`
//!
//! The default is variable quality 2. The file starts with an Info or Xing
//! frame that carries the encoder delay and padding.

use std::fs::File;
use std::io::{BufWriter, Write};

use cutlass_encode::{AudioInput, ChannelLayout, Encoder, Pcm, SideData};
use cutlass_mp3::{Mp3Encoder, Mp3EncoderConfig, Mp3Rate};
use cutlass_mp3_container::{CodecId, CodecParams, Mp3Mux, Mux, Packet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: encode in.f32 RATE CHANNELS out.mp3 [cbr KBPS | vbr QUALITY]";
    let (Some(input), Some(rate), Some(channels), Some(output)) =
        (args.get(1), args.get(2), args.get(3), args.get(4))
    else {
        return Err(usage.into());
    };
    let sample_rate: u32 = rate.parse()?;
    let channels: usize = channels.parse()?;
    let mode = match (args.get(5).map(String::as_str), args.get(6)) {
        (Some("cbr"), Some(kbps)) => Mp3Rate::Constant(kbps.parse()?),
        (Some("vbr"), Some(quality)) => Mp3Rate::Variable(quality.parse()?),
        (None, _) => Mp3Rate::Variable(2),
        _ => return Err(usage.into()),
    };
    let bytes = std::fs::read(input)?;
    let samples: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b).clamp(-1.0, 1.0))
        .collect();
    let layout = if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    };
    let mut encoder = Mp3Encoder::new(Mp3EncoderConfig {
        sample_rate,
        channels,
        rate: mode,
    })?;
    let started = std::time::Instant::now();
    let mut packets = Vec::new();
    for chunk in samples.chunks(4096 * channels) {
        let pcm = Pcm::from_f32(sample_rate, layout.clone(), chunk.to_vec())
            .map_err(|error| format!("{error:?}"))?;
        packets.extend(encoder.push(&AudioInput { pcm })?);
    }
    packets.extend(encoder.drain()?);
    let elapsed = started.elapsed().as_secs_f64();
    let mut params = CodecParams::audio(CodecId::Mp3, sample_rate, channels as u16);
    for side in packets.iter().flat_map(|packet| &packet.side) {
        if let SideData::Gapless {
            encoder_delay,
            trailing_padding,
            ..
        } = side
        {
            params.encoder_delay = *encoder_delay;
            params.trailing_padding = params.trailing_padding.max(*trailing_padding);
        }
    }
    if let Mp3Rate::Constant(kbps) = mode {
        params.bitrate = kbps * 1000;
    }
    let mut mux = Mp3Mux::new(BufWriter::new(File::create(output)?));
    mux.add_stream(params.clone())?;
    let mut bytes = 0usize;
    for packet in &packets {
        bytes += packet.data.len();
        mux.write_packet(&Packet::audio(
            packet.data.clone(),
            packet.pts,
            packet.duration,
            packet.timescale,
        ))?;
    }
    mux.finish()?;
    let seconds = samples.len() as f64 / channels as f64 / f64::from(sample_rate);
    writeln!(
        std::io::stdout(),
        "frames {} kbps {:.1} delay {} padding {} speed {:.1}x",
        packets.len(),
        bytes as f64 * 8.0 / seconds / 1000.0,
        params.encoder_delay,
        params.trailing_padding,
        seconds / elapsed
    )?;
    Ok(())
}
