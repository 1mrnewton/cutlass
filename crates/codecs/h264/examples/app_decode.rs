//! Decodes an Annex B file through the opener the app uses and prints how
//! long it took: `register_decoders`, then `DecodeFactory::open`, each
//! picture packed as the engine receives it and dropped.
//!
//! ```text
//! cargo run --release -p cutlass-h264 --example app_decode -- <file.h264> [runs]
//! ```
//!
//! The decoder takes its thread count from the machine, up to eight cores,
//! so `taskset -c 0-3` is how to give it four. With `ONE_THREAD` set the
//! file is decoded by `H264Decoder::open` instead, on the calling thread,
//! and packed the same way. With `DUMP=<path>` the packed pictures of the
//! first run are written there, to compare with what another decoder
//! writes; the run is then not a timing.
#![allow(clippy::print_stdout)]

use std::io::Write;

use cutlass_decode::{
    CodecId, CodecParams, DecodeFactory, Output, Packet, PictureSamples, VideoDecoder,
};
use cutlass_h264::{H264Decoder, register_decoders};

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("pass the h264 path")?;
    let runs: usize = std::env::args()
        .nth(2)
        .and_then(|runs| runs.parse().ok())
        .unwrap_or(1);
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    let packets: Vec<Packet> = access_units(&bytes)
        .into_iter()
        .enumerate()
        .map(|(index, unit)| Packet::video(unit, index as i64, 1, 30, index == 0))
        .collect();
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut dump = match std::env::var_os("DUMP") {
        Some(path) => Some(std::fs::File::create(path).map_err(|err| err.to_string())?),
        None => None,
    };
    let one_thread = std::env::var_os("ONE_THREAD").is_some();
    let mut factory = DecodeFactory::new();
    register_decoders(&mut factory);
    for _ in 0..runs {
        let started = std::time::Instant::now();
        let mut pictures = 0usize;
        let mut sum = 0xcbf2_9ce4_8422_2325u64;
        if one_thread {
            let mut decoder = H264Decoder::open(&params).map_err(|err| err.to_string())?;
            let mut packed = Vec::new();
            let mut take = |frames: Vec<cutlass_decode::VideoOutput>| -> Result<(), String> {
                for frame in frames {
                    let picture = &frame.picture;
                    packed.clear();
                    packed.extend_from_slice(picture.y());
                    packed.extend_from_slice(picture.u());
                    packed.extend_from_slice(picture.v());
                    pictures += 1;
                    note(&packed, &mut sum, dump.as_mut())?;
                }
                Ok(())
            };
            for packet in &packets {
                take(decoder.push(packet).map_err(|err| err.to_string())?)?;
            }
            take(decoder.drain().map_err(|err| err.to_string())?)?;
        } else {
            let mut decoder = factory.open(&params).map_err(|err| err.to_string())?;
            let mut take = |outputs: Vec<Output>| -> Result<(), String> {
                for output in outputs {
                    let Output::Picture(picture) = output else {
                        continue;
                    };
                    let PictureSamples::Cpu(bytes) = picture.samples() else {
                        continue;
                    };
                    pictures += 1;
                    note(bytes, &mut sum, dump.as_mut())?;
                }
                Ok(())
            };
            for packet in &packets {
                take(decoder.push(packet).map_err(|err| err.to_string())?)?;
            }
            take(decoder.drain().map_err(|err| err.to_string())?)?;
        }
        let seconds = started.elapsed().as_secs_f64();
        dump = None;
        println!("{pictures} pictures, {seconds:.3} s, sum {sum:016x}");
    }
    Ok(())
}

/// Folds the picture's first and last bytes and its length into `sum`, so
/// a run that returned other pictures shows, and writes it to `dump`.
fn note(bytes: &[u8], sum: &mut u64, dump: Option<&mut std::fs::File>) -> Result<(), String> {
    let ends = [bytes.first(), bytes.last()].map(|byte| u64::from(byte.copied().unwrap_or(0)));
    for word in [bytes.len() as u64, ends[0], ends[1]] {
        *sum = (*sum ^ word).wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(file) = dump {
        file.write_all(bytes).map_err(|err| err.to_string())?;
    }
    Ok(())
}

/// The access units of an Annex B stream: a unit ends before the first
/// parameter set, delimiter or SEI that follows a slice, and before a
/// slice that starts a picture (`first_mb_in_slice` 0).
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut index = 0usize;
    while index + 3 < data.len() {
        if data[index] == 0 && data[index + 1] == 0 && data[index + 2] == 1 {
            let header = if index > 0 && data[index - 1] == 0 {
                index - 1
            } else {
                index
            };
            starts.push((header, index + 3));
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut units = Vec::new();
    let mut begin = 0usize;
    let mut saw_slice = false;
    for &(header, body) in &starts {
        let Some(&first) = data.get(body) else {
            break;
        };
        let kind = first & 31;
        let slice = matches!(kind, 1 | 5);
        let starts_picture = slice && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        let boundary = if slice {
            saw_slice && starts_picture
        } else {
            saw_slice && matches!(kind, 6..=9)
        };
        if boundary {
            units.push(data[begin..header].to_vec());
            begin = header;
            saw_slice = false;
        }
        saw_slice |= slice;
    }
    if begin < data.len() {
        units.push(data[begin..].to_vec());
    }
    units
}
