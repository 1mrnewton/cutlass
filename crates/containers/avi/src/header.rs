//! The `hdrl` list: `avih`, one `strl` per stream (`strh`, `strf`, `indx`),
//! and the OpenDML `odml` list. The list is read whole, bounded by
//! [`cutlass_container::Limits::max_header_bytes`], and walked in memory.
//! A child whose declared size runs past its parent is an error.

use cutlass_container::{Limits, OpenError};

use crate::bytes::{Short, i32_at, id_at, u16_at, u32_at};
use crate::limits::{MAX_CHUNKS, MAX_STREAMS};

/// `AVIF_HASINDEX`: the file has an `idx1`.
pub const AVIF_HASINDEX: u32 = 0x10;
/// `AVIF_MUSTUSEINDEX`: chunk order is not presentation order.
pub const AVIF_MUSTUSEINDEX: u32 = 0x20;
/// `AVIF_ISINTERLEAVED`.
pub const AVIF_ISINTERLEAVED: u32 = 0x100;
/// `AVIF_TRUSTCKTYPE`: the chunk type, not the index, says keyframe.
pub const AVIF_TRUSTCKTYPE: u32 = 0x800;

/// `avih`, AVIMAINHEADER. Fields are the file's values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MainHeader {
    pub micro_sec_per_frame: u32,
    pub max_bytes_per_sec: u32,
    pub padding_granularity: u32,
    pub flags: u32,
    /// Frames in the first `RIFF` segment.
    pub total_frames: u32,
    pub initial_frames: u32,
    pub streams: u32,
    pub suggested_buffer_size: u32,
    pub width: u32,
    pub height: u32,
}

/// `strh`, AVISTREAMHEADER. Fields are the file's values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamHeader {
    pub kind: [u8; 4],
    pub handler: [u8; 4],
    pub flags: u32,
    pub priority: u16,
    pub language: u16,
    pub initial_frames: u32,
    pub scale: u32,
    pub rate: u32,
    pub start: u32,
    pub length: u32,
    pub suggested_buffer_size: u32,
    pub quality: u32,
    pub sample_size: u32,
    /// `rcFrame`: left, top, right, bottom. Zero when the header is 48 bytes.
    pub frame: [i16; 4],
}

pub(crate) struct RawStream {
    pub header: StreamHeader,
    pub format: Vec<u8>,
    pub super_index: Option<Vec<u8>>,
}

pub(crate) struct Hdrl {
    pub main: MainHeader,
    pub streams: Vec<RawStream>,
    /// `dmlh` total frames across every `RIFF` segment.
    pub odml_frames: Option<u32>,
}

pub(crate) fn bad(message: &str) -> OpenError {
    OpenError::Source {
        message: format!("avi: {message}"),
    }
}

pub(crate) fn short(error: Short) -> OpenError {
    OpenError::Truncated {
        expected: u64::try_from(error.need).unwrap_or(u64::MAX),
        got: u64::try_from(error.have).unwrap_or(u64::MAX),
    }
}

/// One chunk inside a list that is already in memory.
pub(crate) struct Child<'a> {
    pub id: [u8; 4],
    pub payload: &'a [u8],
}

/// The children of `data`, in order. A size past the parent is
/// [`OpenError::HostileSize`]. A trailing pad byte may be missing.
pub(crate) fn children(data: &[u8]) -> Result<Vec<Child<'_>>, OpenError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut count = 0u32;
    while data.len().saturating_sub(at) >= 8 {
        count += 1;
        if count > MAX_CHUNKS {
            return Err(OpenError::HostileSize {
                declared: u64::from(count),
                limit: u64::from(MAX_CHUNKS),
            });
        }
        let id = id_at(data, at).map_err(short)?;
        let size = u32_at(data, at + 4).map_err(short)?;
        let start = at + 8;
        let len = usize::try_from(size).map_err(|_| hostile(size, data.len() - start))?;
        let Some(end) = start.checked_add(len).filter(|end| *end <= data.len()) else {
            return Err(hostile(size, data.len() - start));
        };
        out.push(Child {
            id,
            payload: &data[start..end],
        });
        at = end + (len & 1);
    }
    Ok(out)
}

fn hostile(size: u32, room: usize) -> OpenError {
    OpenError::HostileSize {
        declared: u64::from(size),
        limit: u64::try_from(room).unwrap_or(u64::MAX),
    }
}

/// Parses the payload of `LIST hdrl`, after the `hdrl` form type.
pub(crate) fn parse_hdrl(data: &[u8], limits: &Limits) -> Result<Hdrl, OpenError> {
    let mut main = None;
    let mut streams = Vec::new();
    let mut odml_frames = None;
    let stream_limit = limits.max_tracks.min(MAX_STREAMS);
    for child in children(data)? {
        match &child.id {
            b"avih" => main = Some(parse_avih(child.payload)?),
            b"LIST" => match child.payload.get(..4) {
                Some(b"strl") => {
                    let count = u32::try_from(streams.len()).unwrap_or(u32::MAX);
                    if count >= stream_limit {
                        return Err(OpenError::HostileSize {
                            declared: u64::from(count) + 1,
                            limit: u64::from(stream_limit),
                        });
                    }
                    streams.push(parse_strl(&child.payload[4..], limits)?);
                }
                Some(b"odml") => {
                    for item in children(&child.payload[4..])? {
                        if &item.id == b"dmlh" && item.payload.len() >= 4 {
                            odml_frames = Some(u32_at(item.payload, 0).map_err(short)?);
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    let Some(main) = main else {
        return Err(bad("hdrl has no avih"));
    };
    if streams.is_empty() {
        return Err(bad("hdrl has no stream"));
    }
    Ok(Hdrl {
        main,
        streams,
        odml_frames,
    })
}

fn parse_avih(data: &[u8]) -> Result<MainHeader, OpenError> {
    if data.len() < 40 {
        return Err(OpenError::Truncated {
            expected: 40,
            got: u64::try_from(data.len()).unwrap_or(u64::MAX),
        });
    }
    let field = |index: usize| u32_at(data, index * 4).map_err(short);
    Ok(MainHeader {
        micro_sec_per_frame: field(0)?,
        max_bytes_per_sec: field(1)?,
        padding_granularity: field(2)?,
        flags: field(3)?,
        total_frames: field(4)?,
        initial_frames: field(5)?,
        streams: field(6)?,
        suggested_buffer_size: field(7)?,
        width: field(8)?,
        height: field(9)?,
    })
}

fn parse_strl(data: &[u8], limits: &Limits) -> Result<RawStream, OpenError> {
    let mut header = None;
    let mut format = None;
    let mut super_index = None;
    for child in children(data)? {
        match &child.id {
            b"strh" if header.is_none() => header = Some(parse_strh(child.payload)?),
            b"strf" if format.is_none() => {
                let len = u64::try_from(child.payload.len()).unwrap_or(u64::MAX);
                // The fixed part is at most 40 bytes (BITMAPINFOHEADER).
                let limit = limits.max_extradata_bytes.saturating_add(64);
                if len > limit {
                    return Err(OpenError::HostileSize {
                        declared: len,
                        limit,
                    });
                }
                format = Some(child.payload.to_vec());
            }
            b"indx" if super_index.is_none() => super_index = Some(child.payload.to_vec()),
            _ => {}
        }
    }
    let Some(header) = header else {
        return Err(bad("strl has no strh"));
    };
    Ok(RawStream {
        header,
        format: format.unwrap_or_default(),
        super_index,
    })
}

fn parse_strh(data: &[u8]) -> Result<StreamHeader, OpenError> {
    if data.len() < 48 {
        return Err(OpenError::Truncated {
            expected: 48,
            got: u64::try_from(data.len()).unwrap_or(u64::MAX),
        });
    }
    let word = |at: usize| u32_at(data, at).map_err(short);
    let mut frame = [0i16; 4];
    if data.len() >= 56 {
        for (index, value) in frame.iter_mut().enumerate() {
            let raw = u16_at(data, 48 + index * 2).map_err(short)?;
            *value = i16::from_le_bytes(raw.to_le_bytes());
        }
    }
    Ok(StreamHeader {
        kind: id_at(data, 0).map_err(short)?,
        handler: id_at(data, 4).map_err(short)?,
        flags: word(8)?,
        priority: u16_at(data, 12).map_err(short)?,
        language: u16_at(data, 14).map_err(short)?,
        initial_frames: word(16)?,
        scale: word(20)?,
        rate: word(24)?,
        start: word(28)?,
        length: word(32)?,
        suggested_buffer_size: word(36)?,
        quality: word(40)?,
        sample_size: word(44)?,
        frame,
    })
}

/// Signed `biHeight` and friends are read here so the sign survives.
pub(crate) fn signed_at(data: &[u8], at: usize) -> Result<i32, OpenError> {
    i32_at(data, at).map_err(short)
}

#[cfg(test)]
mod tests {
    use super::{children, parse_hdrl};
    use cutlass_container::{Limits, OpenError};

    fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    #[test]
    fn a_child_past_its_parent_is_hostile_and_a_missing_pad_is_not() {
        let mut data = chunk(b"JUNK", &[1, 2, 3]);
        data.pop();
        assert_eq!(children(&data).unwrap().len(), 1);
        let mut over = b"JUNK".to_vec();
        over.extend_from_slice(&100u32.to_le_bytes());
        over.extend_from_slice(&[0; 4]);
        assert!(matches!(
            children(&over),
            Err(OpenError::HostileSize {
                declared: 100,
                limit: 4
            })
        ));
    }

    #[test]
    fn an_hdrl_without_avih_or_streams_is_refused() {
        let avih = chunk(b"avih", &[0; 56]);
        assert!(parse_hdrl(&avih, &Limits::standard()).is_err());
        assert!(parse_hdrl(&[], &Limits::standard()).is_err());
    }

    #[test]
    fn more_strl_lists_than_the_track_limit_are_refused() {
        let mut data = chunk(b"avih", &[0; 56]);
        let mut strl = b"strl".to_vec();
        strl.extend(chunk(b"strh", &[0; 56]));
        for _ in 0..3 {
            data.extend(chunk(b"LIST", &strl));
        }
        let mut limits = Limits::standard();
        limits.max_tracks = 2;
        assert!(matches!(
            parse_hdrl(&data, &limits),
            Err(OpenError::HostileSize {
                declared: 3,
                limit: 2
            })
        ));
        assert_eq!(
            parse_hdrl(&data, &Limits::standard())
                .unwrap()
                .streams
                .len(),
            3
        );
    }
}
