//! Temporary: decode an annex-B file in a loop for a profiler. Prints a
//! checksum of every decoded sample so a change in output shows.
#![allow(clippy::print_stdout)]

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};
use cutlass_h264::H264Decoder;

fn main() -> Result<(), String> {
    if let Ok(file) = std::env::var("BASE_FILE") {
        let _ = std::fs::write(
            file,
            format!("{:#x}\n", {
                let f: fn() -> Result<(), String> = main;
                f as usize
            }),
        );
    }
    let path = std::env::args().nth(1).ok_or("pass the h264 path")?;
    let loops: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let units = access_units(&bytes);
    let packets: Vec<Packet> = units
        .iter()
        .enumerate()
        .map(|(i, u)| Packet::video(u.clone(), i as i64, 1, 30, i == 0))
        .collect();
    for _ in 0..loops {
        let mut params = CodecParams::video(CodecId::H264, 1280, 720);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).map_err(|e| e.to_string())?;
        if std::env::var_os("INLINE").is_some() {
            decoder = decoder.with_rows_inline();
        }
        if let Some(threads) = std::env::var("THREADS").ok().and_then(|t| t.parse().ok()) {
            decoder = decoder.with_threads(threads);
        }
        let started = std::time::Instant::now();
        let mut sum = 0xcbf2_9ce4_8422_2325u64;
        let mut pictures = 0usize;
        for packet in &packets {
            let before = std::time::Instant::now();
            let out = decoder.push(packet).map_err(|e| e.to_string())?;
            if std::env::var_os("COSTS").is_some() {
                // nal_ref_idc of the first slice in the access unit.
                let data = &packet.data;
                let mut ref_idc = 9;
                let mut i = 0;
                while i + 3 < data.len() {
                    if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                        let kind = data[i + 3] & 31;
                        if kind == 1 || kind == 5 {
                            ref_idc = (data[i + 3] >> 5) & 3;
                            break;
                        }
                        i += 3;
                    } else {
                        i += 1;
                    }
                }
                println!("cost {ref_idc} {}", before.elapsed().as_micros());
            }
            pictures += out.len();
            hash(&out, &mut sum)?;
        }
        let out = decoder.drain().map_err(|e| e.to_string())?;
        pictures += out.len();
        let seconds = started.elapsed().as_secs_f64();
        hash(&out, &mut sum)?;
        println!("{pictures} pictures, {seconds:.3} s, sum {sum:016x}");
    }
    Ok(())
}

fn hash(frames: &[VideoOutput], sum: &mut u64) -> Result<(), String> {
    if let Some(path) = std::env::var_os("DUMP") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| err.to_string())?;
        for frame in frames {
            for plane in [frame.picture.y(), frame.picture.u(), frame.picture.v()] {
                file.write_all(plane).map_err(|err| err.to_string())?;
            }
        }
    }
    if std::env::var_os("NOHASH").is_some() {
        return Ok(());
    }
    for frame in frames {
        for plane in [frame.picture.y(), frame.picture.u(), frame.picture.v()] {
            for chunk in plane.chunks(8) {
                let mut word = [0u8; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                *sum = (*sum ^ u64::from_le_bytes(word)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    Ok(())
}

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
    let mut current = Vec::new();
    let mut saw_vcl = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts
            .get(n + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if body >= data.len() || header >= end {
            continue;
        }
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}
