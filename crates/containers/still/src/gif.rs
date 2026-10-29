//! GIF frame split.
//!
//! GIF89a stores a graphic-control delay in hundredths of a second. Each
//! image is one packet, in file order. The first packet starts at the first
//! byte of the file, so it carries the header and the global color table.
//! Each later packet starts where the previous image ended and runs through
//! that image. The trailer stays on the last packet. Concatenating the
//! packets is the file.
//!
//! A missing graphic-control extension, and a stored delay of 0, both use
//! [`GIF_DEFAULT_DELAY`](crate::GIF_DEFAULT_DELAY). Disposal stays in the
//! graphic-control bytes. The Netscape loop count is metadata. It does not
//! repeat the packets. Plain text is skipped and is not a packet. LZW data
//! is not decoded.

use crate::bytes::{at, u16_le};
use crate::fail::{Fail, as_u64, hostile, truncated};
use crate::frame::{Frame, Sequence, StillKind};
use crate::limits::{GIF_DEFAULT_DELAY, MAX_FRAMES};

pub(crate) fn parse(data: &[u8]) -> Result<Sequence, Fail> {
    if data.len() < 6 || (&data[..6] != b"GIF87a" && &data[..6] != b"GIF89a") {
        return Err(Fail::Unrecognized);
    }
    if data.len() < 13 {
        return Err(truncated(13, as_u64(data.len())));
    }
    let width = u32::from(u16_le(data, 6)?);
    let height = u32::from(u16_le(data, 8)?);
    if width == 0 || height == 0 {
        return Err(Fail::Bad("gif screen"));
    }
    let packed = data[10];
    let mut index = 13usize;
    if packed & 0x80 != 0 {
        let colors = table_bytes(packed)?;
        if index + colors > data.len() {
            return Err(truncated(as_u64(index + colors), as_u64(data.len())));
        }
        index += colors;
    }

    let mut frames: Vec<Frame> = Vec::new();
    let mut frame_start = 0usize;
    let mut clock = 0_i64;
    let mut loop_count = None;
    let mut pending_delay = None;
    let mut pending_disposal = 0_u8;
    let mut saw_trailer = false;
    while index < data.len() && !saw_trailer {
        match data[index] {
            0x3B => {
                if frames.is_empty() {
                    return Err(Fail::Bad("gif frame"));
                }
                let last = frames.len() - 1;
                frames[last].end = data.len();
                saw_trailer = true;
            }
            0x21 => {
                let extension = extension(data, index)?;
                index = extension.next;
                if let Some(count) = extension.loop_count {
                    loop_count = Some(count);
                }
                if extension.saw_delay {
                    pending_delay = Some(extension.delay);
                    pending_disposal = extension.disposal;
                }
            }
            0x2C => {
                if frames.len() >= usize::try_from(MAX_FRAMES).unwrap_or(usize::MAX) {
                    return Err(hostile(u64::from(MAX_FRAMES) + 1, u64::from(MAX_FRAMES)));
                }
                let image_end = image(data, index)?;
                let duration = match pending_delay {
                    Some(0) | None => GIF_DEFAULT_DELAY,
                    Some(delay) => i64::from(delay),
                };
                let next_clock = clock
                    .checked_add(duration)
                    .ok_or_else(|| hostile(as_u64(frames.len()) + 1, u64::from(MAX_FRAMES)))?;
                frames.push(Frame {
                    start: frame_start,
                    end: image_end,
                    pts: clock,
                    duration,
                    timed: true,
                    keyframe: frames.is_empty(),
                    disposal: pending_disposal,
                    blend: None,
                });
                clock = next_clock;
                frame_start = image_end;
                pending_delay = None;
                pending_disposal = 0;
                index = image_end;
            }
            _ => return Err(Fail::Bad("gif block")),
        }
    }
    if !saw_trailer || frames.is_empty() {
        return Err(truncated(as_u64(data.len()) + 1, as_u64(data.len())));
    }
    Ok(Sequence {
        kind: StillKind::Gif,
        animated: frames.len() > 1,
        width,
        height,
        loop_count,
        frames,
    })
}

struct Extension {
    next: usize,
    saw_delay: bool,
    delay: u16,
    disposal: u8,
    loop_count: Option<u16>,
}

fn extension(data: &[u8], index: usize) -> Result<Extension, Fail> {
    if index + 2 > data.len() {
        return Err(truncated(as_u64(index) + 2, as_u64(data.len())));
    }
    let label = data[index + 1];
    let mut cursor = index + 2;
    let mut saw_delay = false;
    let mut delay = 0_u16;
    let mut disposal = 0_u8;
    let mut loop_count = None;
    if label == 0xF9 {
        if cursor >= data.len() {
            return Err(truncated(as_u64(cursor) + 1, as_u64(data.len())));
        }
        let size = usize::from(data[cursor]);
        cursor += 1;
        let end = fit_block(data, cursor, size)?;
        if size == 4 {
            let packed = data[cursor];
            delay = u16_le(data, cursor + 1)?;
            disposal = (packed >> 2) & 0x07;
            saw_delay = true;
        }
        cursor = skip_subblocks(data, end)?;
    } else if label == 0xFF {
        if cursor >= data.len() {
            return Err(truncated(as_u64(cursor) + 1, as_u64(data.len())));
        }
        let id_len = usize::from(data[cursor]);
        cursor += 1;
        let id_end = fit_block(data, cursor, id_len)?;
        let id = at(data, cursor, id_len)?;
        if id == b"NETSCAPE2.0" {
            loop_count = netscape_loop(data, id_end)?;
        }
        cursor = skip_subblocks(data, id_end)?;
    } else {
        cursor = skip_subblocks(data, cursor)?;
    }
    Ok(Extension {
        next: cursor,
        saw_delay,
        delay,
        disposal,
        loop_count,
    })
}

fn netscape_loop(data: &[u8], index: usize) -> Result<Option<u16>, Fail> {
    if index >= data.len() {
        return Ok(None);
    }
    let size = usize::from(data[index]);
    if size == 3 && index + 4 <= data.len() && data[index + 1] == 1 {
        return Ok(Some(u16_le(data, index + 2)?));
    }
    Ok(None)
}

fn image(data: &[u8], index: usize) -> Result<usize, Fail> {
    let header = 10usize;
    if index + header > data.len() {
        return Err(truncated(as_u64(index + header), as_u64(data.len())));
    }
    let image_width = u16_le(data, index + 5)?;
    let image_height = u16_le(data, index + 7)?;
    if image_width == 0 || image_height == 0 {
        return Err(Fail::Bad("gif frame"));
    }
    let packed = data[index + 9];
    let mut cursor = index + header;
    if packed & 0x80 != 0 {
        let colors = table_bytes(packed)?;
        if cursor + colors > data.len() {
            return Err(truncated(as_u64(cursor + colors), as_u64(data.len())));
        }
        cursor += colors;
    }
    if cursor >= data.len() {
        return Err(truncated(as_u64(cursor) + 1, as_u64(data.len())));
    }
    cursor += 1;
    skip_subblocks(data, cursor)
}

fn table_bytes(packed: u8) -> Result<usize, Fail> {
    let shift = u32::from(packed & 0x07) + 1;
    if shift > 8 {
        return Err(Fail::Bad("gif color table"));
    }
    Ok(3 * (1_usize << shift))
}

fn fit_block(data: &[u8], index: usize, size: usize) -> Result<usize, Fail> {
    let Some(end) = index.checked_add(size) else {
        return Err(hostile(
            as_u64(size),
            as_u64(data.len().saturating_sub(index)),
        ));
    };
    if end > data.len() {
        return Err(hostile(
            as_u64(size),
            as_u64(data.len().saturating_sub(index)),
        ));
    }
    Ok(end)
}

fn skip_subblocks(data: &[u8], mut index: usize) -> Result<usize, Fail> {
    loop {
        if index >= data.len() {
            return Err(truncated(as_u64(index) + 1, as_u64(data.len())));
        }
        let size = usize::from(data[index]);
        index += 1;
        if size == 0 {
            return Ok(index);
        }
        index = fit_block(data, index, size)?;
    }
}
