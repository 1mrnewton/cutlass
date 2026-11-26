//! FORM walk. Chunks are big-endian. An odd payload has one pad byte.
//! The pad is inside the form and outside the chunk size. Sound bytes are
//! not read during the walk.
//!
//! A header chunk that runs past the form is an error. The sound is not:
//! a form or an `SSND` that states more bytes than the file holds, a last
//! chunk with no pad byte, and a `COMM` frame count of zero are what a
//! recording or a download that stopped leaves, and the sound that is
//! there plays, as libavformat plays it.

use std::io::{Read, Seek};

use cutlass_codec::ContainerId;
use cutlass_container::{Access, Input, OpenError};

use crate::format::{self, Sound};
use crate::limits::{MAX_CHUNKS, MAX_COMM};

struct Ssnd {
    start: u64,
    available: u64,
    /// The chunk stated more bytes than the form or the file holds.
    cut: bool,
}

pub(crate) struct Scanned {
    pub aifc: bool,
    pub sound: Sound,
    pub data_start: u64,
    /// Sample frames, or codec blocks, to return.
    pub units: u64,
    /// The file holds less sound than its header states.
    pub cut: bool,
    pub skipped: u32,
}

pub(crate) fn scan<R: Read + Seek>(input: &mut Input<R>) -> Result<Scanned, OpenError> {
    let (aifc, form_end) = read_form(input)?;
    let mut comm = None;
    let mut ssnd = None;
    let mut skipped = 0u32;
    let mut chunks = 0u32;
    while input.position() < form_end {
        if chunks >= MAX_CHUNKS {
            return Err(OpenError::HostileSize {
                declared: u64::from(chunks) + 1,
                limit: u64::from(MAX_CHUNKS),
            });
        }
        chunks += 1;
        if form_end - input.position() < 8 && comm.is_some() && ssnd.is_some() {
            // The file stops inside a chunk header after the sound.
            break;
        }
        if take_chunk(input, form_end, aifc, &mut comm, &mut ssnd, &mut skipped)? {
            break;
        }
    }
    finish(input, aifc, comm, ssnd, skipped)
}

fn read_form<R: Read>(input: &mut Input<R>) -> Result<(bool, u64), OpenError> {
    let mut header = [0u8; 12];
    input
        .read_exact(&mut header)
        .map_err(OpenError::from_header)?;
    if &header[0..4] != b"FORM" {
        return Err(unrecognized());
    }
    let aifc = if &header[8..12] == b"AIFF" {
        false
    } else if &header[8..12] == b"AIFC" {
        true
    } else {
        return Err(unrecognized());
    };
    let form_size = u64::from(u32::from_be_bytes([
        header[4], header[5], header[6], header[7],
    ]));
    if form_size < 4 {
        return Err(OpenError::Truncated {
            expected: 12,
            got: 8 + form_size,
        });
    }
    // A form that states more than the file holds ends where the file does.
    let stated = add(8, form_size)?;
    let form_end = input.length().map_or(stated, |length| stated.min(length));
    Ok((aifc, form_end))
}

fn unrecognized() -> OpenError {
    OpenError::Unrecognized {
        family: ContainerId::Aiff,
    }
}

fn take_chunk<R: Read + Seek>(
    input: &mut Input<R>,
    form_end: u64,
    aifc: bool,
    comm: &mut Option<Sound>,
    ssnd: &mut Option<Ssnd>,
    skipped: &mut u32,
) -> Result<bool, OpenError> {
    let left = form_end - input.position();
    if left < 8 {
        return Err(OpenError::Truncated {
            expected: 8,
            got: left,
        });
    }
    let mut id = [0u8; 4];
    input.read_exact(&mut id).map_err(OpenError::from_header)?;
    let size = u64::from(read_u32(input)?);
    let start = input.position();
    let stated_end = add(start, size)?;
    if &id == b"SSND" {
        return read_ssnd(input, size, start, form_end, comm.is_some(), ssnd);
    }
    if stated_end > form_end {
        if comm.is_some() && ssnd.is_some() {
            // A chunk after the sound that is not all there. The sound
            // plays; the chunk is not read.
            *skipped += 1;
            return Ok(true);
        }
        return Err(OpenError::HostileSize {
            declared: stated_end,
            limit: form_end,
        });
    }
    // The pad byte of an odd last chunk may be missing.
    let end = add(stated_end, size & 1)?.min(form_end);
    match &id {
        b"COMM" => {
            read_comm(input, size, aifc, comm)?;
            if input.position() < end {
                jump(input, end)?;
            }
            Ok(false)
        }
        _ => {
            jump(input, end)?;
            *skipped += 1;
            Ok(false)
        }
    }
}

fn read_comm<R: Read + Seek>(
    input: &mut Input<R>,
    size: u64,
    aifc: bool,
    comm: &mut Option<Sound>,
) -> Result<(), OpenError> {
    if comm.is_some() {
        return Err(format::open_source("duplicate COMM chunk"));
    }
    let limit = input.limits().max_header_bytes.min(u64::from(MAX_COMM));
    if size > limit {
        return Err(OpenError::HostileSize {
            declared: size,
            limit,
        });
    }
    let payload = input
        .read_limited(size, limit)
        .map_err(OpenError::from_header)?;
    *comm = Some(format::parse_comm(&payload, aifc)?);
    Ok(())
}

fn read_ssnd<R: Read + Seek>(
    input: &mut Input<R>,
    stated: u64,
    start: u64,
    form_end: u64,
    have_comm: bool,
    ssnd: &mut Option<Ssnd>,
) -> Result<bool, OpenError> {
    if ssnd.is_some() {
        return Err(format::open_source("duplicate SSND chunk"));
    }
    // The size is a claim. What the file holds of it is the sound, also
    // when the form size is the smaller: a form header that was not
    // rewritten does not unmake bytes that are there.
    let bound = input.length().unwrap_or(form_end);
    let room = bound.saturating_sub(start);
    let form_end = bound;
    let cut = stated > room;
    let size = stated.min(room);
    let end = add(start, add(size, size & 1)?)?.min(form_end);
    if size < 8 {
        return Err(OpenError::Truncated {
            expected: 8,
            got: size,
        });
    }
    let offset = u64::from(read_u32(input)?);
    let _block = read_u32(input)?;
    let after = size - 8;
    if offset > after {
        return Err(OpenError::Truncated {
            expected: offset,
            got: after,
        });
    }
    let sound_at = add(add(start, 8)?, offset)?;
    *ssnd = Some(Ssnd {
        start: sound_at,
        available: after - offset,
        cut,
    });
    if input.access() == Access::Random {
        jump(input, end)?;
        return Ok(cut);
    }
    if !have_comm {
        return Err(format::open_source("forward AIFF has SSND before COMM"));
    }
    let limit = input.limits().max_header_bytes;
    if offset > limit {
        return Err(OpenError::HostileSize {
            declared: offset,
            limit,
        });
    }
    discard(input, offset)?;
    Ok(true)
}

fn finish<R: Read + Seek>(
    input: &mut Input<R>,
    aifc: bool,
    comm: Option<Sound>,
    ssnd: Option<Ssnd>,
    skipped: u32,
) -> Result<Scanned, OpenError> {
    let Some(sound) = comm else {
        return Err(format::open_source("AIFF has no COMM chunk"));
    };
    let Some(ssnd) = ssnd else {
        return Err(format::open_source("AIFF has no SSND chunk"));
    };
    let frame = u64::from(sound.frame_bytes());
    if frame == 0 {
        return Err(format::open_source("frame is empty"));
    }
    // `COMM` states the length when it states one and the sound is all
    // there. Zero is a header that was not filled in, and a count past
    // the bytes is a file that was cut: either way the bytes are the
    // length, down to the last whole frame.
    let held = ssnd.available / frame;
    let stated = sound.stated_units()?;
    let (units, cut) = if stated == 0 {
        (held, ssnd.cut || !ssnd.available.is_multiple_of(frame))
    } else if stated > held {
        (held, true)
    } else {
        (stated, false)
    };
    let limit = input.limits().max_packet_bytes;
    if frame > limit {
        return Err(OpenError::HostileSize {
            declared: frame,
            limit,
        });
    }
    if input.access() == Access::Random {
        input.seek(ssnd.start).map_err(OpenError::from_header)?;
    }
    Ok(Scanned {
        aifc,
        sound,
        data_start: ssnd.start,
        units,
        cut,
        skipped,
    })
}

fn jump<R: Read + Seek>(input: &mut Input<R>, end: u64) -> Result<(), OpenError> {
    if end < input.position() {
        return Err(format::open_source("chunk ended before the read position"));
    }
    if input.access() == Access::Random {
        return input.seek(end).map_err(OpenError::from_header);
    }
    let amount = end - input.position();
    let limit = input.limits().max_header_bytes;
    if amount > limit {
        return Err(OpenError::HostileSize {
            declared: amount,
            limit,
        });
    }
    discard(input, amount)
}

fn discard<R: Read>(input: &mut Input<R>, mut left: u64) -> Result<(), OpenError> {
    let mut buf = [0u8; 256];
    while left > 0 {
        let take_u = left.min(256);
        let take = usize::try_from(take_u).map_err(|_| OpenError::HostileSize {
            declared: take_u,
            limit: 256,
        })?;
        input
            .read_exact(&mut buf[..take])
            .map_err(OpenError::from_header)?;
        left -= take_u;
    }
    Ok(())
}

fn read_u32<R: Read>(input: &mut Input<R>) -> Result<u32, OpenError> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf).map_err(OpenError::from_header)?;
    Ok(u32::from_be_bytes(buf))
}

fn add(left: u64, right: u64) -> Result<u64, OpenError> {
    left.checked_add(right)
        .ok_or_else(|| format::open_source("chunk offset overflowed"))
}
