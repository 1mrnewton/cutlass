//! WebP RIFF form.
//!
//! A still WebP is one packet: the file bytes. An animated WebP, marked by
//! the VP8X animation flag, is one packet per `ANMF` chunk in file order.
//! The first packet starts at the first byte, so it carries `VP8X` and
//! `ANIM`. Later packets are the `ANMF` chunks. The last packet runs to the
//! end of the file. Concatenating the packets is the file.
//!
//! `ANMF` duration is milliseconds. Zero uses
//! [`WEBP_DEFAULT_DELAY_MS`](crate::WEBP_DEFAULT_DELAY_MS). The loop count
//! on `ANIM` is metadata. Frame geometry and the disposal and blend bits
//! stay in the `ANMF` header inside the packet. The bitstream is not decoded.

use crate::bytes::{u16_le, u24_le, u32_le};
use crate::fail::{Fail, as_u64, hostile, truncated};
use crate::frame::{Frame, Sequence, StillKind};
use crate::limits::{MAX_CHUNK_DEPTH, MAX_FRAMES, WEBP_DEFAULT_DELAY_MS};

pub(crate) fn parse(data: &[u8]) -> Result<Sequence, Fail> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(Fail::Unrecognized);
    }
    let declared = u64::from(u32_le(data, 4)?);
    let available = as_u64(data.len().saturating_sub(8));
    if declared > available {
        return Err(hostile(declared, available));
    }
    let form_end = 8 + usize::try_from(declared).map_err(|_| hostile(declared, available))?;
    let mut chunks = Vec::new();
    let mut cursor = 12usize;
    while cursor < form_end {
        let chunk = match next_chunk(data, cursor, form_end)? {
            Some(chunk) => chunk,
            None => break,
        };
        if chunk.fourcc == *b"ANMF" {
            validate_frame_chunks(data, &chunk)?;
        }
        cursor = chunk.next;
        chunks.push(chunk);
    }
    let Some(first) = chunks.first() else {
        return Err(Fail::Bad("webp picture"));
    };
    if first.fourcc == *b"VP8 " || first.fourcc == *b"VP8L" {
        let (width, height) = bitstream_size(data, first)?;
        return still(data, width, height);
    }
    if first.fourcc != *b"VP8X" {
        return Err(Fail::Bad("webp chunk"));
    }
    let (animated, width, height) = vp8x(data, first)?;
    if !animated {
        if chunks.iter().any(|chunk| chunk.fourcc == *b"ANMF") {
            return Err(Fail::Bad("webp anmf"));
        }
        if !chunks
            .iter()
            .any(|chunk| chunk.fourcc == *b"VP8 " || chunk.fourcc == *b"VP8L")
        {
            return Err(Fail::Bad("webp picture"));
        }
        return still(data, width, height);
    }
    animated_frames(data, &chunks, width, height)
}

fn still(data: &[u8], width: u32, height: u32) -> Result<Sequence, Fail> {
    Ok(Sequence {
        kind: StillKind::Webp,
        animated: false,
        width,
        height,
        loop_count: None,
        frames: vec![Frame::still(data.len())],
    })
}

fn animated_frames(
    data: &[u8],
    chunks: &[Chunk],
    width: u32,
    height: u32,
) -> Result<Sequence, Fail> {
    let mut saw_anim = false;
    let mut loop_count = None;
    let mut frames: Vec<Frame> = Vec::new();
    let mut clock = 0_i64;
    for chunk in chunks {
        if chunk.fourcc == *b"ANIM" {
            if saw_anim {
                return Err(Fail::Bad("webp anim"));
            }
            saw_anim = true;
            loop_count = Some(anim_loop(data, chunk)?);
            continue;
        }
        if chunk.fourcc != *b"ANMF" {
            continue;
        }
        if !saw_anim {
            return Err(Fail::Bad("webp anim"));
        }
        if frames.len() >= usize::try_from(MAX_FRAMES).unwrap_or(usize::MAX) {
            return Err(hostile(u64::from(MAX_FRAMES) + 1, u64::from(MAX_FRAMES)));
        }
        let header = anmf_header(data, chunk)?;
        let duration = if header.duration == 0 {
            WEBP_DEFAULT_DELAY_MS
        } else {
            i64::from(header.duration)
        };
        let pts = clock;
        clock = clock
            .checked_add(duration)
            .ok_or_else(|| hostile(as_u64(frames.len()) + 1, u64::from(MAX_FRAMES)))?;
        let start = frames.last().map(|frame| frame.end).unwrap_or(0);
        let end = if frames.len() + 1 == count_anmf(chunks) {
            data.len()
        } else {
            chunk.next
        };
        frames.push(Frame {
            start,
            end,
            pts,
            duration,
            timed: true,
            keyframe: frames.is_empty(),
            disposal: header.disposal,
            blend: Some(header.blend),
        });
    }
    if frames.is_empty() || !saw_anim {
        return Err(Fail::Bad("webp frame"));
    }
    Ok(Sequence {
        kind: StillKind::Webp,
        animated: true,
        width,
        height,
        loop_count,
        frames,
    })
}

fn count_anmf(chunks: &[Chunk]) -> usize {
    chunks
        .iter()
        .filter(|chunk| chunk.fourcc == *b"ANMF")
        .count()
}

struct Chunk {
    fourcc: [u8; 4],
    data_at: usize,
    data_end: usize,
    next: usize,
}

fn next_chunk(data: &[u8], at: usize, parent_end: usize) -> Result<Option<Chunk>, Fail> {
    if at >= parent_end {
        return Ok(None);
    }
    if parent_end - at < 8 {
        return Err(truncated(as_u64(at) + 8, as_u64(parent_end)));
    }
    let fourcc = [data[at], data[at + 1], data[at + 2], data[at + 3]];
    let size = u32_le(data, at + 4)?;
    let data_at = at + 8;
    let declared = u64::from(size);
    let available = as_u64(parent_end.saturating_sub(data_at));
    if declared > available {
        return Err(hostile(declared, available));
    }
    let data_end = data_at + usize::try_from(size).map_err(|_| hostile(declared, available))?;
    let mut next = data_end;
    if size % 2 == 1 {
        if next >= parent_end {
            return Err(truncated(as_u64(next) + 1, as_u64(parent_end)));
        }
        next += 1;
    }
    Ok(Some(Chunk {
        fourcc,
        data_at,
        data_end,
        next,
    }))
}

fn validate_frame_chunks(data: &[u8], chunk: &Chunk) -> Result<(), Fail> {
    if chunk.data_end - chunk.data_at < 16 {
        return Err(truncated(
            as_u64(chunk.data_at) + 16,
            as_u64(chunk.data_end),
        ));
    }
    let mut cursor = chunk.data_at + 16;
    while cursor < chunk.data_end {
        let Some(inner) = next_chunk(data, cursor, chunk.data_end)? else {
            break;
        };
        if inner.fourcc == *b"ANMF" {
            return Err(hostile(
                u64::from(MAX_CHUNK_DEPTH) + 1,
                u64::from(MAX_CHUNK_DEPTH),
            ));
        }
        cursor = inner.next;
    }
    Ok(())
}

fn vp8x(data: &[u8], chunk: &Chunk) -> Result<(bool, u32, u32), Fail> {
    if chunk.data_end - chunk.data_at < 10 {
        return Err(Fail::Bad("webp vp8x"));
    }
    let flags = data[chunk.data_at];
    let animated = flags & 0x02 != 0;
    let width = u24_le(data, chunk.data_at + 4)?.saturating_add(1);
    let height = u24_le(data, chunk.data_at + 7)?.saturating_add(1);
    if width == 0 || height == 0 {
        return Err(Fail::Bad("webp canvas"));
    }
    let product = u64::from(width).saturating_mul(u64::from(height));
    if product > u64::from(u32::MAX) {
        return Err(hostile(product, u64::from(u32::MAX)));
    }
    Ok((animated, width, height))
}

fn anim_loop(data: &[u8], chunk: &Chunk) -> Result<u16, Fail> {
    if chunk.data_end - chunk.data_at < 6 {
        return Err(Fail::Bad("webp anim"));
    }
    u16_le(data, chunk.data_at + 4)
}

struct AnmfHeader {
    duration: u32,
    disposal: u8,
    blend: u8,
}

fn anmf_header(data: &[u8], chunk: &Chunk) -> Result<AnmfHeader, Fail> {
    if chunk.data_end - chunk.data_at < 16 {
        return Err(truncated(
            as_u64(chunk.data_at) + 16,
            as_u64(chunk.data_end),
        ));
    }
    let duration = u24_le(data, chunk.data_at + 12)?;
    let flags = data[chunk.data_at + 15];
    Ok(AnmfHeader {
        duration,
        disposal: flags & 0x01,
        blend: (flags >> 1) & 0x01,
    })
}

fn bitstream_size(data: &[u8], chunk: &Chunk) -> Result<(u32, u32), Fail> {
    let payload = data
        .get(chunk.data_at..chunk.data_end)
        .ok_or(Fail::Bad("webp picture"))?;
    if chunk.fourcc == *b"VP8L" {
        return vp8l_size(payload);
    }
    vp8_size(payload)
}

fn vp8_size(payload: &[u8]) -> Result<(u32, u32), Fail> {
    if payload.len() < 10 {
        return Err(truncated(10, as_u64(payload.len())));
    }
    if payload[0] & 1 != 0 || payload[3..6] != [0x9D, 0x01, 0x2A] {
        return Err(Fail::Bad("webp vp8"));
    }
    let width = u32::from(u16::from_le_bytes([payload[6], payload[7]]) & 0x3FFF);
    let height = u32::from(u16::from_le_bytes([payload[8], payload[9]]) & 0x3FFF);
    if width == 0 || height == 0 {
        return Err(Fail::Bad("webp vp8"));
    }
    Ok((width, height))
}

fn vp8l_size(payload: &[u8]) -> Result<(u32, u32), Fail> {
    if payload.len() < 5 || payload[0] != 0x2F {
        return Err(Fail::Bad("webp vp8l"));
    }
    let bits = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
    let width = (bits & 0x3FFF) + 1;
    let height = ((bits >> 14) & 0x3FFF) + 1;
    Ok((width, height))
}
