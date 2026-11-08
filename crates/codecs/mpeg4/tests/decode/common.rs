//! Helpers the decoder tests share.

use std::path::Path;

use cutlass_decode::{Decoder, Output, Packet, PictureSamples};

/// Splits an elementary stream into one packet per picture: each VOP with
/// the headers before it, or each short-header picture.
pub fn split(data: &[u8]) -> Vec<Vec<u8>> {
    let short = matches!(data, [0, 0, b, ..] if b & 0xFC == 0x80);
    let mut cuts = Vec::new();
    let mut pending: Option<usize> = None;
    let mut i = 0;
    while i + 3 < data.len() {
        let is_start = data[i] == 0 && data[i + 1] == 0;
        if short && is_start && data[i + 2] & 0xFC == 0x80 {
            cuts.push(i);
            i += 3;
            continue;
        }
        if !short && is_start && data[i + 2] == 1 {
            let code = data[i + 3];
            if code == 0xB6 {
                cuts.push(pending.take().unwrap_or(i));
            } else if pending.is_none() {
                pending = Some(i);
            }
            i += 4;
            continue;
        }
        i += 1;
    }
    let mut out = Vec::new();
    for (n, &start) in cuts.iter().enumerate() {
        let end = cuts.get(n + 1).copied().unwrap_or(data.len());
        out.push(data[start..end].to_vec());
    }
    out
}

/// A packet of the test streams' 25 Hz timeline.
pub fn packet(data: Vec<u8>, pts: i64) -> Packet {
    Packet {
        stream: 0,
        data,
        pts,
        duration: 1,
        timescale: 25,
        keyframe: false,
        side: Vec::new(),
    }
}

/// One output picture: width, height, presentation time, packed 4:2:0.
pub type Picture = (u32, u32, i64, Vec<u8>);

pub fn pictures(out: Vec<Output>) -> Vec<Picture> {
    out.into_iter()
        .filter_map(|o| match o {
            Output::Picture(p) => {
                let info = p.info();
                let PictureSamples::Cpu(bytes) = p.samples() else {
                    panic!("platform buffer");
                };
                Some((info.width, info.height, p.pts(), bytes.to_vec()))
            }
            _ => None,
        })
        .collect()
}

/// Pushes every packet, then drains.
pub fn decode_all(decoder: &mut dyn Decoder, packets: &[Vec<u8>]) -> Vec<Picture> {
    let mut out = Vec::new();
    for (n, data) in packets.iter().enumerate() {
        out.extend(pictures(
            decoder.push(&packet(data.clone(), n as i64)).expect("push"),
        ));
    }
    out.extend(pictures(decoder.drain().expect("drain")));
    out
}

pub fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .expect("fixture")
}
