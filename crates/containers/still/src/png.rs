//! PNG chunk walk.
//!
//! The signature and chunk layout are the W3C PNG specification. A chunk is
//! a 32-bit length, a type, that many data bytes, and a CRC-32 over the type
//! and the data. The image bytes stay compressed. This walk does not inflate
//! them. An `acTL` sequence is not split: the file is one picture, and the
//! support record names that limit.

use crate::bytes::{at, u32_be};
use crate::fail::{Fail, as_u64, hostile, truncated};
use crate::frame::{Frame, Sequence, StillKind};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

pub(crate) fn parse(data: &[u8]) -> Result<Sequence, Fail> {
    if data.len() < SIGNATURE.len() || data[..SIGNATURE.len()] != SIGNATURE {
        return Err(Fail::Unrecognized);
    }
    let mut index = SIGNATURE.len();
    let mut size = None;
    let mut saw_idat = false;
    let mut saw_iend = false;
    while !saw_iend {
        if data.len() - index < 8 {
            return Err(truncated(as_u64(index) + 8, as_u64(data.len())));
        }
        let declared = u32_be(data, index)?;
        let after_type = index + 8;
        let rest = as_u64(data.len().saturating_sub(after_type));
        let payload = u64::from(declared);
        if payload.saturating_add(4) > rest {
            return Err(hostile(payload, rest.saturating_sub(4)));
        }
        let payload_len = usize::try_from(payload).map_err(|_| hostile(payload, rest))?;
        let crc_at = after_type + payload_len;
        let stored = u32_be(data, crc_at)?;
        let covered = at(data, index + 4, 4 + payload_len)?;
        if png_crc(covered) != stored {
            return Err(Fail::Bad("png chunk crc"));
        }
        let kind = at(data, index + 4, 4)?;
        if size.is_none() {
            if kind != b"IHDR" {
                return Err(Fail::Bad("png ihdr"));
            }
            size = Some(ihdr(at(data, after_type, payload_len)?)?);
        } else if kind == b"IHDR" {
            return Err(Fail::Bad("png ihdr"));
        }
        if kind == b"IDAT" {
            saw_idat = true;
        }
        if kind == b"IEND" {
            if declared != 0 {
                return Err(Fail::Bad("png iend"));
            }
            saw_iend = true;
        }
        index = crc_at + 4;
    }
    if !saw_idat {
        return Err(Fail::Bad("png image"));
    }
    let Some((width, height)) = size else {
        return Err(Fail::Bad("png ihdr"));
    };
    Ok(Sequence {
        kind: StillKind::Png,
        animated: false,
        width,
        height,
        loop_count: None,
        frames: vec![Frame::still(data.len())],
    })
}

fn ihdr(data: &[u8]) -> Result<(u32, u32), Fail> {
    if data.len() != 13 {
        return Err(Fail::Bad("png ihdr"));
    }
    let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    if width == 0 || height == 0 || !color_ok(data[8], data[9], data[10], data[11], data[12]) {
        return Err(Fail::Bad("png ihdr"));
    }
    Ok((width, height))
}

fn color_ok(depth: u8, color: u8, compression: u8, filter: u8, interlace: u8) -> bool {
    if compression != 0 || filter != 0 || interlace > 1 {
        return false;
    }
    matches!(
        (color, depth),
        (0, 1 | 2 | 4 | 8 | 16) | (2, 8 | 16) | (3, 1 | 2 | 4 | 8) | (4, 8 | 16) | (6, 8 | 16)
    )
}

#[cfg(test)]
pub(crate) fn append_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
    out.extend(len.to_be_bytes());
    out.extend(kind);
    out.extend(data);
    let mut covered = Vec::with_capacity(4usize.saturating_add(data.len()));
    covered.extend(kind);
    covered.extend(data);
    out.extend(png_crc(&covered).to_be_bytes());
}

fn png_crc(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in data {
        let index = usize::from(u8::try_from((crc ^ u32::from(byte)) & 0xFF).unwrap_or(0));
        crc = CRC_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

const CRC_TABLE: [u32; 256] = crc_table();

const fn crc_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut value = 0;
    while value < 256 {
        let mut crc = value;
        let mut bit = 0;
        while bit < 8 {
            if crc & 1 == 1 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[value as usize] = crc;
        value += 1;
    }
    table
}
