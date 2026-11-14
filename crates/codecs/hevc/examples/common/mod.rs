//! Reading the access units of an HEVC stream for the examples: the first
//! HEVC track of an MP4 or QuickTime file (its `hvcC` record and
//! length-prefixed samples), or an Annex B byte stream.

use cutlass_codec::StreamCodec;
use cutlass_container::Demux;
use cutlass_decode::{CodecId, CodecParams};
use cutlass_isobmff::IsoBmffFile;

/// The decoder parameters and the access units of `path`.
pub fn read(path: &str) -> Result<(CodecParams, Vec<Vec<u8>>), String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if data.get(4..8) == Some(b"ftyp") {
        return read_mp4(path);
    }
    let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
    params.nal_length_size = 0;
    Ok((params, access_units(&data)))
}

fn read_mp4(path: &str) -> Result<(CodecParams, Vec<Vec<u8>>), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut demux =
        IsoBmffFile::open(std::io::BufReader::new(file)).map_err(|e| format!("{path}: {e:?}"))?;
    let stream = demux
        .streams()
        .iter()
        .find(|s| matches!(s.params().codec(), StreamCodec::Known(CodecId::Hevc)))
        .cloned()
        .ok_or_else(|| format!("{path}: no HEVC track"))?;
    let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
    params.extra = stream.params().extradata().to_vec();
    let mut units = Vec::new();
    while let Some(packet) = demux.read_packet().map_err(|e| format!("{path}: {e:?}"))? {
        if packet.stream() == stream.id().get() {
            units.push(packet.bytes().to_vec());
        }
    }
    Ok((params, units))
}

/// The NAL units of an Annex B stream, without their start codes.
fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let mut end = starts.get(k + 1).map_or(data.len(), |&n| n - 3);
            while end > s && data[end - 1] == 0 {
                end -= 1;
            }
            &data[s..end]
        })
        .collect()
}

/// Annex B access units, each with 4-byte start codes: a unit starts at a
/// parameter set, SEI or delimiter before its first slice, or at a slice
/// that starts a picture.
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut has_slice = false;
    for nal in nal_units(data) {
        if nal.len() < 3 {
            continue;
        }
        let kind = (nal[0] >> 1) & 0x3f;
        let slice = kind <= 21;
        let starts_unit = if slice {
            nal[2] & 0x80 != 0
        } else {
            matches!(kind, 32..=35 | 39 | 41..=44 | 48..=55)
        };
        if starts_unit && has_slice {
            units.push(std::mem::take(&mut current));
            has_slice = false;
        }
        current.extend_from_slice(&[0, 0, 0, 1]);
        current.extend_from_slice(nal);
        has_slice |= slice;
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}
