//! Splits an MPEG-1 or MPEG-2 video elementary stream into access units.
//!
//! One access unit is one coded frame: the sequence and GOP headers in front
//! of it, its picture header, and its slices. Both fields of a field-coded
//! frame stay in one unit. Bytes may arrive in pieces of any size.

use cutlass_decode::CodecError;

use crate::headers::{
    EXT_PICTURE_CODING, EXTENSION, GROUP, PICTURE_I, PICTURE_START, SEQUENCE_END, SEQUENCE_HEADER,
    SLICE_FIRST, SLICE_LAST, STRUCTURE_FRAME,
};

/// Largest access unit the splitter holds.
pub const MAX_UNIT_BYTES: usize = 32 << 20;

/// One coded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessUnit {
    pub data: Vec<u8>,
    /// 1 I, 2 P, 3 B, 4 D, of the first picture.
    pub coding_type: u8,
    /// Display position counted in frames from the start of the stream.
    pub display_index: u64,
    /// Position in coded order.
    pub decode_index: u64,
}

impl AccessUnit {
    #[must_use]
    pub fn keyframe(&self) -> bool {
        self.coding_type == PICTURE_I
    }
}

/// Incremental splitter.
#[derive(Clone, Debug, Default)]
pub struct Splitter {
    pending: Vec<u8>,
    /// Bytes of `pending` already searched for a start code.
    scanned: usize,
    current: Vec<u8>,
    has_picture: bool,
    has_slices: bool,
    awaiting_second_field: bool,
    pictures: u8,
    coding_type: u8,
    temporal_reference: u16,
    last_reference: Option<u16>,
    wraps: u64,
    gop_base: u64,
    frames: u64,
}

impl Splitter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds bytes and returns the access units they completed.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AccessUnit>, CodecError> {
        let mut pending = std::mem::take(&mut self.pending);
        pending.extend_from_slice(bytes);
        let mut out = Vec::new();
        let mut pos = 0;
        let result = loop {
            let Some(first) = find_start(&pending, pos) else {
                // No start code yet: the last two bytes may begin one.
                pos = pending.len().saturating_sub(2).max(pos);
                self.scanned = 0;
                break Ok(());
            };
            let from = self.scanned.max(first + 4);
            let Some(next) = find_start(&pending, from) else {
                pos = first;
                self.scanned = pending.len().saturating_sub(2).max(first + 4);
                break Ok(());
            };
            self.scanned = 0;
            if let Err(error) = self.unit(&pending[first..next], &mut out) {
                break Err(error);
            }
            pos = next;
        };
        pending.drain(..pos);
        self.scanned = self.scanned.saturating_sub(pos);
        self.pending = pending;
        result?;
        if self.pending.len() + self.current.len() > MAX_UNIT_BYTES {
            return Err(CodecError::Limit {
                what: "MPEG-2 access unit size",
            });
        }
        Ok(out)
    }

    /// Ends the stream and returns the last access unit.
    pub fn finish(&mut self) -> Result<Vec<AccessUnit>, CodecError> {
        let mut out = Vec::new();
        if let Some(first) = find_start(&self.pending, 0) {
            let unit = self.pending.split_off(first);
            self.unit(&unit, &mut out)?;
        }
        self.pending.clear();
        self.scanned = 0;
        self.emit(&mut out);
        self.current.clear();
        self.has_picture = false;
        Ok(out)
    }

    fn unit(&mut self, unit: &[u8], out: &mut Vec<AccessUnit>) -> Result<(), CodecError> {
        let Some(&code) = unit.get(3) else {
            return Ok(());
        };
        let starts_frame = matches!(code, SEQUENCE_HEADER | GROUP)
            || (code == PICTURE_START && !self.awaiting_second_field);
        if starts_frame && self.has_slices {
            self.emit(out);
        }
        match code {
            GROUP => {
                self.gop_base = self.frames;
                self.wraps = 0;
                self.last_reference = None;
            }
            PICTURE_START => {
                let (Some(&a), Some(&b)) = (unit.get(4), unit.get(5)) else {
                    return Err(CodecError::Truncated {
                        at_bit: 32,
                        what: "picture header",
                    });
                };
                if self.pictures == 0 {
                    self.temporal_reference = u16::from(a) << 2 | u16::from(b >> 6);
                    self.coding_type = (b >> 3) & 7;
                    self.has_picture = true;
                } else {
                    self.awaiting_second_field = false;
                }
                self.pictures = self.pictures.saturating_add(1);
            }
            EXTENSION => {
                if unit.get(4).map(|b| b >> 4) == Some(EXT_PICTURE_CODING)
                    && self.pictures == 1
                    && let Some(&byte) = unit.get(6)
                {
                    self.awaiting_second_field = byte & 3 != STRUCTURE_FRAME;
                }
            }
            SLICE_FIRST..=SLICE_LAST => self.has_slices = true,
            _ => {}
        }
        self.current.extend_from_slice(unit);
        if code == SEQUENCE_END {
            self.emit(out);
        }
        Ok(())
    }

    fn emit(&mut self, out: &mut Vec<AccessUnit>) {
        if !self.has_picture {
            return;
        }
        let tr = self.temporal_reference;
        if let Some(last) = self.last_reference
            && tr + 512 < last
        {
            self.wraps += 1;
        }
        self.last_reference = Some(tr);
        out.push(AccessUnit {
            data: std::mem::take(&mut self.current),
            coding_type: self.coding_type,
            display_index: self.gop_base + self.wraps * 1024 + u64::from(tr),
            decode_index: self.frames,
        });
        self.frames += 1;
        self.has_picture = false;
        self.has_slices = false;
        self.awaiting_second_field = false;
        self.pictures = 0;
    }
}

/// Splits a whole elementary stream.
pub fn split(stream: &[u8]) -> Result<Vec<AccessUnit>, CodecError> {
    let mut splitter = Splitter::new();
    let mut units = splitter.push(stream)?;
    units.extend(splitter.finish()?);
    Ok(units)
}

fn find_start(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 3 <= data.len() {
        match data[i + 2] {
            1 if data[i] == 0 && data[i + 1] == 0 => return Some(i),
            0 => i += 1,
            _ => i += 3,
        }
    }
    None
}
