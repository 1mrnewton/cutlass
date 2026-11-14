//! The training run of a profile-guided build of the H.264 codec.
//!
//! Every `.h264` file in the directory given, or in `streams/` beside this
//! crate, is decoded on one thread, as the editor decodes, and on four, so
//! the threaded paths are not taken for never run. The first pictures of
//! each 8-bit stream are then encoded, as export does when no platform
//! encoder takes them. The encoder's own stream is not decoded: its few
//! tools would weigh down the branches real files take.
//!
//! The run is only as good as its streams: the compiler lays out and
//! inlines the code for the branches they take. See `streams/make.py`.

use std::io::Write;
use std::path::{Path, PathBuf};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::{H264Decoder, H264Encoder};

type Failure = Box<dyn std::error::Error>;

/// Pictures of each stream the encoder is given.
const ENCODED: usize = 30;

fn main() -> Result<(), Failure> {
    let dir = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("streams"),
        PathBuf::from,
    );
    let mut streams: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "h264"))
        .collect();
    streams.sort();
    if streams.is_empty() {
        return Err(format!("no .h264 streams in {}", dir.display()).into());
    }
    let mut out = std::io::stdout().lock();
    for path in &streams {
        let name = path
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        let packets = packets(path)?;
        let pictures = decode(&packets, 1)?;
        let threaded = decode(&packets, 4)?;
        if threaded.len() != pictures.len() {
            return Err(format!(
                "{name}: {} pictures on four threads, {} on one",
                threaded.len(),
                pictures.len()
            )
            .into());
        }
        write!(out, "{name}: {} pictures", pictures.len())?;
        if pictures
            .iter()
            .all(|picture| picture.yuv420p10.is_none() && picture.yuv420_high.is_none())
        {
            let bytes = encode(&pictures)?;
            write!(
                out,
                ", encoded {} in {} bytes",
                pictures.len().min(ENCODED),
                bytes.len()
            )?;
        }
        writeln!(out)?;
    }
    Ok(())
}

fn packets(path: &Path) -> Result<Vec<Packet>, Failure> {
    Ok(packets_of(&std::fs::read(path)?))
}

fn packets_of(bytes: &[u8]) -> Vec<Packet> {
    access_units(bytes)
        .into_iter()
        .enumerate()
        .map(|(index, unit)| Packet::video(unit, index as i64, 1, 30, index == 0))
        .collect()
}

fn decode(packets: &[Packet], threads: usize) -> Result<Vec<VideoOutput>, Failure> {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params)?.with_threads(threads);
    let mut pictures = Vec::new();
    for packet in packets {
        pictures.extend(decoder.push(packet)?);
    }
    pictures.extend(decoder.drain()?);
    Ok(pictures)
}

fn encode(pictures: &[VideoOutput]) -> Result<Vec<u8>, Failure> {
    let Some(first) = pictures.first() else {
        return Ok(Vec::new());
    };
    let mut params =
        CodecParams::video(CodecId::H264, first.picture.width(), first.picture.height());
    params.nal_length_size = 0;
    params.bitrate = 2_000_000;
    let mut encoder = H264Encoder::open(&params)?;
    let mut bytes = Vec::new();
    for (index, picture) in pictures.iter().take(ENCODED).enumerate() {
        let input = VideoInput {
            picture: picture.picture.clone(),
            pts: index as i64,
            duration: 1,
            timescale: 30,
            force_keyframe: index == 0,
        };
        for packet in encoder.push(&input)? {
            bytes.extend_from_slice(&packet.data);
        }
    }
    for packet in encoder.drain()? {
        bytes.extend_from_slice(&packet.data);
    }
    Ok(bytes)
}

/// Annex B bytes cut into access units: a new unit at each access unit
/// delimiter and at each slice that starts a picture after a picture's
/// slices.
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 < data.len() {
        if data[index..index + 3] == [0, 0, 1] {
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
    let mut saw_slice = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(data.len(), |&(next, _)| next);
        let Some(&first) = data.get(body) else {
            continue;
        };
        let kind = first & 31;
        let slice = kind == 1 || kind == 5;
        // first_mb_in_slice is 0, ue(v) "1", exactly when the next bit is set.
        let starts_picture = slice && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw_slice)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_slice = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_slice |= slice;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}
