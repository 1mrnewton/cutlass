//! Fixture streams for the integration tests.
//!
//! Each `fixtures/<name>.ivf` is one stream. The `fixtures/<name>_N.obu`
//! files form one stream named `<name>.obu`, in `N` order, one packet per
//! file.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cutlass_av1::Av1Decoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureOutput, PictureSamples,
};

/// FNV-1a 64 of raw samples, the digest the references use.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// The CPU samples of a picture.
pub fn samples(picture: &PictureOutput) -> &[u8] {
    match picture.samples() {
        PictureSamples::Cpu(bytes) => bytes,
        _ => panic!("picture is not on the CPU"),
    }
}

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// The frame payloads of an IVF file.
pub fn ivf_payloads(bytes: &[u8]) -> Vec<Vec<u8>> {
    assert_eq!(&bytes[..4], b"DKIF", "IVF signature");
    let header = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
    let mut at = header;
    let mut out = Vec::new();
    while at + 12 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("size")) as usize;
        at += 12;
        out.push(bytes[at..at + size].to_vec());
        at += size;
    }
    assert_eq!(at, bytes.len(), "IVF ends on a frame");
    out
}

/// The packets of one fixture stream: `name.ivf`, or `name.obu` for the
/// `name_N.obu` group.
pub fn stream(name: &str) -> Vec<Vec<u8>> {
    streams()
        .remove(name)
        .unwrap_or_else(|| panic!("no fixture stream {name}"))
}

/// Every fixture stream by name, with its packets.
pub fn streams() -> BTreeMap<String, Vec<Vec<u8>>> {
    let mut out = BTreeMap::new();
    let mut obu: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    for entry in std::fs::read_dir(fixtures()).expect("fixtures") {
        let path = entry.expect("entry").path();
        let name = path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .to_string();
        if name.ends_with(".ivf") {
            let packets = ivf_payloads(&std::fs::read(&path).expect("read"));
            out.insert(name, packets);
        } else if let Some(stem) = name.strip_suffix(".obu") {
            let (group, index) = stem.rsplit_once('_').expect("name_N.obu");
            obu.entry(group.to_string())
                .or_default()
                .push((index.parse().expect("N"), path));
        }
    }
    for (group, mut files) in obu {
        files.sort();
        let packets = files
            .iter()
            .map(|(_, path)| std::fs::read(path).expect("read"))
            .collect();
        out.insert(format!("{group}.obu"), packets);
    }
    out
}

pub fn packet(data: &[u8], pts: i64) -> Packet {
    Packet {
        stream: 0,
        data: data.to_vec(),
        pts,
        duration: 1,
        timescale: 30,
        keyframe: pts == 0,
        side: Vec::new(),
    }
}

/// Every picture of `packets` through a decoder already opened, drained at
/// the end.
pub fn decode_with(
    decoder: &mut Av1Decoder,
    packets: &[Vec<u8>],
) -> Result<Vec<PictureOutput>, CodecError> {
    let mut pictures = Vec::new();
    let mut take = |outputs: Vec<Output>| {
        for output in outputs {
            if let Output::Picture(picture) = output {
                pictures.push(picture);
            }
        }
    };
    for (pts, data) in packets.iter().enumerate() {
        take(decoder.push(&packet(data, pts as i64))?);
    }
    take(decoder.drain()?);
    Ok(pictures)
}

/// Every picture of `packets` through a fresh decoder.
pub fn decode(
    packets: &[Vec<u8>],
    width: u32,
    height: u32,
) -> Result<Vec<PictureOutput>, CodecError> {
    let params = CodecParams::video(CodecId::Av1, width, height);
    let mut decoder = Av1Decoder::open(&params)?;
    decode_with(&mut decoder, packets)
}
