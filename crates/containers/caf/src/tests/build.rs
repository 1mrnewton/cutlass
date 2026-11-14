//! Builds CAF files byte by byte, independently of the muxer.

use std::io::Cursor;

use crate::CafDemux;

pub(crate) fn chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = kind.to_vec();
    out.extend(i64::try_from(body.len()).expect("len").to_be_bytes());
    out.extend(body);
    out
}

pub(crate) fn sized_chunk(kind: &[u8; 4], size: i64, body: &[u8]) -> Vec<u8> {
    let mut out = kind.to_vec();
    out.extend(size.to_be_bytes());
    out.extend(body);
    out
}

pub(crate) fn desc_body(
    rate: f64,
    format: &[u8; 4],
    flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    channels: u32,
    bits: u32,
) -> Vec<u8> {
    let mut out = rate.to_bits().to_be_bytes().to_vec();
    out.extend(format);
    for field in [flags, bytes_per_packet, frames_per_packet, channels, bits] {
        out.extend(field.to_be_bytes());
    }
    out
}

pub(crate) fn varint(value: u32, out: &mut Vec<u8>) {
    // Written here from the specification's description, not with the
    // crate's encoder, so a shared bug cannot pass both ways.
    let mut groups = vec![value & 0x7F];
    let mut rest = value >> 7;
    while rest != 0 {
        groups.push(rest & 0x7F);
        rest >>= 7;
    }
    for (index, group) in groups.iter().enumerate().rev() {
        let more = if index == 0 { 0 } else { 0x80 };
        out.push(u8::try_from(*group).expect("group") | more);
    }
}

pub(crate) fn pakt_body(
    packets: i64,
    valid: i64,
    priming: i32,
    remainder: i32,
    entries: &[u32],
) -> Vec<u8> {
    let mut out = packets.to_be_bytes().to_vec();
    out.extend(valid.to_be_bytes());
    out.extend(priming.to_be_bytes());
    out.extend(remainder.to_be_bytes());
    for entry in entries {
        varint(*entry, &mut out);
    }
    out
}

pub(crate) fn data_body(payload: &[u8]) -> Vec<u8> {
    let mut out = 1_u32.to_be_bytes().to_vec();
    out.extend(payload);
    out
}

pub(crate) fn file(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut out = b"caff\x00\x01\x00\x00".to_vec();
    for part in chunks {
        out.extend(part);
    }
    out
}

/// Stereo 16-bit little-endian PCM at 48 kHz, `frames` frames counting up.
pub(crate) fn pcm_file(frames: u16) -> (Vec<u8>, Vec<u8>) {
    let mut payload = Vec::new();
    for frame in 0..frames {
        payload.extend(frame.to_le_bytes());
        payload.extend((!frame).to_le_bytes());
    }
    let bytes = file(&[
        chunk(b"desc", &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16)),
        chunk(b"data", &data_body(&payload)),
    ]);
    (bytes, payload)
}

/// Packets of distinct bytes: packet `i` is `sizes[i]` copies of `i`.
pub(crate) fn payload_of(sizes: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, size) in sizes.iter().enumerate() {
        let byte = u8::try_from(index % 251).expect("byte");
        out.extend(std::iter::repeat_n(
            byte,
            usize::try_from(*size).expect("size"),
        ));
    }
    out
}

pub(crate) fn open(
    bytes: &[u8],
) -> Result<CafDemux<Cursor<Vec<u8>>>, cutlass_container::OpenError> {
    let length = u64::try_from(bytes.len()).expect("len");
    CafDemux::open(Cursor::new(bytes.to_vec()), length)
}

pub(crate) fn open_forward(
    bytes: &[u8],
) -> Result<CafDemux<Cursor<Vec<u8>>>, cutlass_container::OpenError> {
    CafDemux::open_forward(Cursor::new(bytes.to_vec()))
}

/// An AAC cookie for AAC-LC 48 kHz stereo, as AudioToolbox lays it out.
pub(crate) fn aac_cookie() -> Vec<u8> {
    let mut out = vec![0x03, 0x80, 0x80, 0x80, 0x22, 0, 0, 0];
    out.extend([0x04, 0x80, 0x80, 0x80, 0x14, 0x40, 0x15, 0, 0x18, 0]);
    out.extend([0, 0, 0, 0, 0, 0, 0, 0]);
    out.extend([0x05, 0x80, 0x80, 0x80, 0x02, 0x11, 0x90]);
    out.extend([0x06, 0x80, 0x80, 0x80, 0x01, 0x02]);
    out
}
