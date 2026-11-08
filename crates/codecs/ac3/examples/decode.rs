//! Decode a raw AC-3 or E-AC-3 elementary stream to raw interleaved
//! little-endian `f32`.
//!
//! `cargo run --release -p cutlass-ac3 --example decode -- in.ac3 out.f32
//! [--drc line|heavy] [--downmix loro|ltrt|mono] [--program N] [--core]`
//!
//! Each independent substream 0 frame starts an access unit; the frames
//! after it, up to the next one, join that unit. Prints the sample rate,
//! the layout, the unit count and the last metadata on stdout.

use std::io::Write;

use cutlass_ac3::{Ac3Decoder, Downmix, DynamicRange, Extension, Options, StreamType, probe};
use cutlass_decode::{CodecId, CodecParams, Decoder, Output, Packet};

fn units(data: &[u8]) -> Result<Vec<&[u8]>, Box<dyn std::error::Error>> {
    let mut units = Vec::new();
    let mut start = 0;
    let mut at = 0;
    while at + 6 <= data.len() {
        let Ok(info) = probe(&data[at..]) else {
            at += 1;
            if units.is_empty() {
                start = at;
            }
            continue;
        };
        if at + info.frame_bytes > data.len() {
            break;
        }
        let opens = info.stream_type != StreamType::Dependent && info.substream_id == 0;
        if opens && at > start {
            units.push(&data[start..at]);
            start = at;
        }
        at += info.frame_bytes;
    }
    if at > start {
        units.push(&data[start..at]);
    }
    Ok(units)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        return Err("usage: decode in.ac3 out.f32 [--drc line|heavy] [--downmix loro|ltrt|mono] [--program N] [--core]".into());
    };
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let dynamic_range = match flag("--drc") {
        Some("line") => DynamicRange::Line {
            cut: 1.0,
            boost: 1.0,
        },
        Some("heavy") => DynamicRange::Heavy,
        _ => DynamicRange::Off,
    };
    let downmix = match flag("--downmix") {
        Some("loro") => Downmix::LoRo,
        Some("ltrt") => Downmix::LtRt,
        Some("mono") => Downmix::Mono,
        _ => Downmix::None,
    };
    let mut options = Options {
        dynamic_range,
        downmix,
        ..Options::default()
    };
    if let Some(program) = flag("--program") {
        options.program = program.parse()?;
    }
    if args.iter().any(|a| a == "--core") {
        options.extension = Extension::Core;
    }
    let data = std::fs::read(input)?;
    let codec = if data.get(5).is_some_and(|b| b >> 3 > 10) {
        CodecId::Eac3
    } else {
        CodecId::Ac3
    };
    let mut decoder = Ac3Decoder::with_options(
        &CodecParams::audio(codec, 0, cutlass_decode::ChannelLayout::stereo()),
        options,
    )?;
    let mut out = Vec::new();
    let mut count = 0u64;
    let mut write = |outputs: Vec<Output>| {
        for output in outputs {
            if let Output::Audio(audio) = output {
                for sample in audio.pcm.as_f32().unwrap_or(&[]) {
                    out.extend_from_slice(&sample.to_le_bytes());
                }
            }
        }
    };
    for unit in units(&data)? {
        let packet = Packet::audio(unit.to_vec(), 0, 0, 0);
        write(decoder.push(&packet)?);
        count += 1;
    }
    write(decoder.drain()?);
    std::fs::write(output, &out)?;
    let mut stdout = std::io::stdout();
    if let Some((rate, layout)) = decoder.stream() {
        writeln!(
            stdout,
            "rate {rate} layout {:?} units {count}",
            layout.channels()
        )?;
    }
    writeln!(stdout, "metadata {:?}", decoder.metadata())?;
    writeln!(
        stdout,
        "skipped dependents {}",
        decoder.skipped_dependents()
    )?;
    writeln!(stdout, "tools {:?}", decoder.tools())?;
    Ok(())
}
