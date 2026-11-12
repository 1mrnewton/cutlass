//! Where things are: the `hdrl` bytes, every `movi` list, and `idx1`.
//!
//! The first `RIFF AVI ` is walked at its top level, then each OpenDML
//! `RIFF AVIX` after it. Only chunk headers are read, plus `hdrl`, which is
//! held whole up to [`cutlass_container::Limits::max_header_bytes`].
//!
//! A `RIFF` or `movi` size that runs past the end of the file is a
//! recording that stopped early. Its span ends at the end of the file and
//! [`Layout::truncated`] says so; nothing of that size is allocated. A size
//! of zero, which some live writers leave, also runs to the end of the file.
//! `hdrl` must fit inside its parent and the file.

use std::io::{Read, Seek};

use cutlass_container::{Limits, OpenError};

use crate::header::bad;
use crate::limits::{MAX_CHUNKS, MAX_RIFFS};
use crate::source::Source;

/// One `movi` list: `list` is where its `movi` form type sits, which is the
/// base of `idx1` offsets; chunks run from `list + 4` to `end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Movi {
    pub list: u64,
    pub end: u64,
}

impl Movi {
    pub(crate) const fn data(&self) -> u64 {
        self.list + 4
    }

    /// The payload `[data, data + size)` is inside this list.
    pub(crate) fn holds(&self, data: u64, size: u64) -> bool {
        data >= self.data().saturating_add(8)
            && data.checked_add(size).is_some_and(|end| end <= self.end)
    }
}

pub(crate) struct Layout {
    pub hdrl: Vec<u8>,
    pub movi: Vec<Movi>,
    /// `idx1` payload position and the whole entries inside the file.
    pub idx1: Option<(u64, u64)>,
    pub truncated: bool,
    pub riffs: u32,
}

pub(crate) fn read_layout<R: Read + Seek>(
    source: &mut Source<R>,
    limits: &Limits,
) -> Result<Layout, OpenError> {
    let length = source.length();
    let head: [u8; 12] = source.read_array().map_err(OpenError::from_header)?;
    match crate::probe::verdict(&head) {
        cutlass_container::HeaderVerdict::Match => {}
        cutlass_container::HeaderVerdict::UnsupportedBrand => {
            return Err(OpenError::UnsupportedBrand {
                family: cutlass_codec::ContainerId::Avi,
            });
        }
        cutlass_container::HeaderVerdict::Unrecognized => {
            return Err(OpenError::Unrecognized {
                family: cutlass_codec::ContainerId::Avi,
            });
        }
    }
    let declared = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
    let mut layout = Layout {
        hdrl: Vec::new(),
        movi: Vec::new(),
        idx1: None,
        truncated: false,
        riffs: 1,
    };
    let first_end = riff_end(8, declared, length, &mut layout.truncated);
    let mut hdrl = None;
    walk_riff(source, 12, first_end, limits, &mut layout, Some(&mut hdrl))?;
    layout.hdrl = hdrl.ok_or_else(|| bad("the first RIFF has no hdrl list"))?;

    let mut at = first_end + (first_end & 1);
    while !layout.truncated && at.saturating_add(12) <= length {
        source.seek(at).map_err(OpenError::from_header)?;
        let head: [u8; 12] = source.read_array().map_err(OpenError::from_header)?;
        if &head[..4] != b"RIFF" || &head[8..] != b"AVIX" {
            break;
        }
        layout.riffs += 1;
        if layout.riffs > MAX_RIFFS {
            return Err(OpenError::HostileSize {
                declared: u64::from(layout.riffs),
                limit: u64::from(MAX_RIFFS),
            });
        }
        let declared = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
        let end = riff_end(at + 8, declared, length, &mut layout.truncated);
        walk_riff(source, at + 12, end, limits, &mut layout, None)?;
        at = end + (end & 1);
    }
    Ok(layout)
}

fn riff_end(payload: u64, declared: u64, length: u64, truncated: &mut bool) -> u64 {
    if declared == 0 {
        return length;
    }
    let end = payload.saturating_add(declared);
    if end > length {
        *truncated = true;
        return length;
    }
    end
}

fn walk_riff<R: Read + Seek>(
    source: &mut Source<R>,
    start: u64,
    end: u64,
    limits: &Limits,
    layout: &mut Layout,
    mut hdrl: Option<&mut Option<Vec<u8>>>,
) -> Result<(), OpenError> {
    let mut at = start;
    let mut count = 0u32;
    while at.saturating_add(8) <= end {
        count += 1;
        if count > MAX_CHUNKS {
            return Err(OpenError::HostileSize {
                declared: u64::from(count),
                limit: u64::from(MAX_CHUNKS),
            });
        }
        source.seek(at).map_err(OpenError::from_header)?;
        let head: [u8; 8] = source.read_array().map_err(OpenError::from_header)?;
        let size = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
        let payload = at + 8;
        let declared_end = payload.saturating_add(size);
        match &head[..4] {
            b"LIST" if size >= 4 && payload + 4 <= end => {
                let form: [u8; 4] = source.read_array().map_err(OpenError::from_header)?;
                match &form {
                    b"hdrl" => {
                        let Some(slot) = hdrl.as_deref_mut() else {
                            at = skip_to(declared_end, end, layout);
                            continue;
                        };
                        if slot.is_some() {
                            return Err(bad("a second hdrl list"));
                        }
                        if declared_end > end {
                            return Err(OpenError::HostileSize {
                                declared: size,
                                limit: end - payload,
                            });
                        }
                        let bytes = source
                            .read_vec(size - 4, limits.max_header_bytes)
                            .map_err(OpenError::from_header)?;
                        *slot = Some(bytes);
                    }
                    b"movi" => {
                        let list_end = if size == 0 || declared_end > end {
                            layout.truncated |= declared_end > end;
                            end
                        } else {
                            declared_end
                        };
                        layout.movi.push(Movi {
                            list: payload,
                            end: list_end,
                        });
                        at = list_end + (list_end & 1);
                        continue;
                    }
                    _ => {}
                }
            }
            b"idx1" if layout.idx1.is_none() && layout.riffs == 1 => {
                let room = end.saturating_sub(payload).min(size);
                if room < size {
                    layout.truncated = true;
                }
                layout.idx1 = Some((payload, room - room % 16));
            }
            _ => {}
        }
        at = skip_to(declared_end, end, layout);
    }
    Ok(())
}

fn skip_to(declared_end: u64, end: u64, layout: &mut Layout) -> u64 {
    if declared_end > end {
        layout.truncated = true;
        return end;
    }
    declared_end + (declared_end & 1)
}

#[cfg(test)]
mod tests {
    use super::Movi;

    #[test]
    fn a_payload_must_sit_after_a_chunk_header_inside_the_list() {
        let movi = Movi {
            list: 100,
            end: 200,
        };
        assert!(movi.holds(112, 88));
        assert!(!movi.holds(112, 89));
        assert!(!movi.holds(104, 4), "no room for the chunk header");
        assert!(!movi.holds(u64::MAX, 1));
    }
}
