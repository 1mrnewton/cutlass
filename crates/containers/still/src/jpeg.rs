//! JPEG marker walk.
//!
//! ITU-T T.81 marks a file with SOI (`FF D8`) and ends it with EOI (`FF D9`).
//! A segment between them carries a 16-bit length that includes those two
//! bytes. The entropy-coded scan after SOS is not Huffman-decoded here. It
//! is scanned for the next marker, treating `FF 00` as a stuffed byte.
//! The packet is the file. Dimensions come from the first SOF segment.

use crate::bytes::{at, u16_be};
use crate::fail::{Fail, as_u64, hostile, truncated};
use crate::frame::{Frame, Sequence, StillKind};

pub(crate) fn parse(data: &[u8]) -> Result<Sequence, Fail> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(Fail::Unrecognized);
    }
    if data.len() < 4 {
        return Err(truncated(4, as_u64(data.len())));
    }
    let mut index = 2usize;
    let mut size = None;
    let mut in_scan = false;
    let mut saw_scan = false;
    let mut saw_eoi = false;
    while index < data.len() && !saw_eoi {
        if !in_scan {
            let (marker, next) = marker_at(data, index)?;
            index = next;
            if is_standalone(marker) {
                if marker == 0xD9 {
                    saw_eoi = true;
                }
                continue;
            }
            let (payload_at, payload_len, after) = segment(data, index)?;
            if is_sof(marker) && size.is_none() {
                size = Some(sof_size(data, payload_at, payload_len)?);
            }
            if marker == 0xDA {
                saw_scan = true;
                in_scan = true;
                index = after;
                continue;
            }
            index = after;
            continue;
        }
        match scan_entropy(data, index)? {
            Entropy::Eoi(next) => {
                saw_eoi = true;
                index = next;
            }
            Entropy::Marker(at_marker) => {
                in_scan = false;
                index = at_marker;
            }
        }
    }
    if !saw_eoi {
        return Err(truncated(as_u64(data.len()) + 2, as_u64(data.len())));
    }
    let Some((width, height)) = size else {
        return Err(Fail::Bad("jpeg frame"));
    };
    if !saw_scan {
        return Err(Fail::Bad("jpeg scan"));
    }
    Ok(Sequence {
        kind: StillKind::Jpeg,
        animated: false,
        width,
        height,
        loop_count: None,
        frames: vec![Frame::still(data.len())],
    })
}

fn is_standalone(marker: u8) -> bool {
    marker == 0x01 || marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker)
}

fn is_sof(marker: u8) -> bool {
    matches!(
        marker,
        0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF
    )
}

fn marker_at(data: &[u8], index: usize) -> Result<(u8, usize), Fail> {
    if index >= data.len() {
        return Err(truncated(as_u64(index) + 1, as_u64(data.len())));
    }
    if data[index] != 0xFF {
        return Err(Fail::Bad("jpeg marker"));
    }
    let mut cursor = index;
    while cursor < data.len() && data[cursor] == 0xFF {
        cursor += 1;
    }
    if cursor >= data.len() {
        return Err(truncated(as_u64(cursor) + 1, as_u64(data.len())));
    }
    Ok((data[cursor], cursor + 1))
}

fn segment(data: &[u8], index: usize) -> Result<(usize, usize, usize), Fail> {
    let len_field = u16_be(data, index)?;
    if len_field < 2 {
        return Err(Fail::Bad("jpeg segment"));
    }
    let payload_len = usize::from(len_field - 2);
    let payload_at = index + 2;
    let declared = as_u64(payload_len);
    let available = as_u64(data.len().saturating_sub(payload_at));
    if declared > available {
        return Err(hostile(declared, available));
    }
    Ok((payload_at, payload_len, payload_at + payload_len))
}

fn sof_size(data: &[u8], payload_at: usize, payload_len: usize) -> Result<(u32, u32), Fail> {
    if payload_len < 6 {
        return Err(Fail::Bad("jpeg frame"));
    }
    let height = u32::from(u16_be(data, payload_at + 1)?);
    let width = u32::from(u16_be(data, payload_at + 3)?);
    if width == 0 || height == 0 {
        return Err(Fail::Bad("jpeg frame"));
    }
    let _ = at(data, payload_at, payload_len)?;
    Ok((width, height))
}

enum Entropy {
    Eoi(usize),
    Marker(usize),
}

fn scan_entropy(data: &[u8], start: usize) -> Result<Entropy, Fail> {
    let mut index = start;
    while index < data.len() {
        if data[index] != 0xFF {
            index += 1;
            continue;
        }
        let marker_ff = index;
        let mut cursor = index;
        while cursor < data.len() && data[cursor] == 0xFF {
            cursor += 1;
        }
        if cursor >= data.len() {
            return Err(truncated(as_u64(cursor) + 1, as_u64(data.len())));
        }
        let marker = data[cursor];
        if marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            index = cursor + 1;
            continue;
        }
        if marker == 0xD9 {
            return Ok(Entropy::Eoi(cursor + 1));
        }
        return Ok(Entropy::Marker(marker_ff));
    }
    Err(truncated(as_u64(data.len()) + 2, as_u64(data.len())))
}
