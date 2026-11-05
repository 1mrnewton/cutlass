//! Decode an MP3 file to raw interleaved little-endian `f32`.
//!
//! `cargo run --release -p cutlass-mp3 --example decode -- in.mp3 out.f32 [--gapless]`
//!
//! Prints the sample rate, channel count, and frame count on stdout.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use cutlass_decode::{Decoder, Output, Packet, SideData};
use cutlass_mp3::Mp3Decoder;
use cutlass_mp3_container::{Demux, Mp3Demux};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        return Err("usage: decode in.mp3 out.f32 [--gapless]".into());
    };
    let gapless = args.iter().any(|arg| arg == "--gapless");
    let mut demux = Mp3Demux::open(BufReader::new(File::open(input)?))?;
    let stream = demux.streams()[0].clone();
    let rate = stream.params.sample_rate;
    let params = cutlass_decode::CodecParams::audio(
        cutlass_decode::CodecId::Mp3,
        rate,
        if stream.params.channels == 1 {
            cutlass_decode::ChannelLayout::mono()
        } else {
            cutlass_decode::ChannelLayout::stereo()
        },
    );
    let mut decoder = Mp3Decoder::open(&params)?;
    let mut out = BufWriter::new(File::create(output)?);
    let mut frames = 0u64;
    let write = |outputs: Vec<Output>, out: &mut BufWriter<File>| -> std::io::Result<()> {
        for output in outputs {
            if let Output::Audio(audio) = output {
                let samples = audio.pcm.as_f32().unwrap_or(&[]);
                let channels = audio.pcm.layout.len();
                let lead = audio.skip_leading as usize * channels;
                let tail = audio.skip_trailing as usize * channels;
                let keep = if gapless {
                    &samples[lead.min(samples.len())
                        ..samples
                            .len()
                            .saturating_sub(tail)
                            .max(lead.min(samples.len()))]
                } else {
                    samples
                };
                for sample in keep {
                    out.write_all(&sample.to_le_bytes())?;
                }
            }
        }
        Ok(())
    };
    while let Some(packet) = demux.next_packet()? {
        let mut access = Packet::audio(packet.data, packet.pts, packet.duration, packet.timescale);
        if gapless && let Some(trim) = stream.trim {
            access.side.push(SideData::Gapless {
                encoder_delay: trim.encoder_delay,
                trailing_padding: trim.trailing_padding,
                sample_rate: rate,
            });
        }
        let outputs = decoder.push(&access)?;
        write(outputs, &mut out)?;
        frames += 1;
    }
    write(decoder.drain()?, &mut out)?;
    out.flush()?;
    let mut stdout = std::io::stdout();
    writeln!(
        stdout,
        "rate {rate} channels {} frames {frames} trim {:?}",
        stream.params.channels, stream.trim
    )?;
    Ok(())
}
