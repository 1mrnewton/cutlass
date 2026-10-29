//! Chunk walk. Declared sizes are checked against the parent and the file
//! before a payload is stored.

use std::io::{Read, Seek};

use cutlass_container::{Input, OpenError};

use crate::bytes::{id_at, u32_at, u64_at};
use crate::format::{FormatError, WavFormat, parse_fmt};
use crate::limits::{MAX_CHUNKS, MAX_DEPTH, MAX_DS64_ROWS, MAX_KEPT_CHUNK};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brand {
    Riff,
    Rf64,
}

pub struct KeptChunk {
    id: [u8; 4],
    payload: Vec<u8>,
}

impl KeptChunk {
    pub(crate) fn new(id: [u8; 4], payload: Vec<u8>) -> Self {
        Self { id, payload }
    }

    #[must_use]
    pub const fn id(&self) -> [u8; 4] {
        self.id
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

pub struct DroppedChunk {
    id: [u8; 4],
    declared: u64,
}

impl DroppedChunk {
    pub(crate) fn new(id: [u8; 4], declared: u64) -> Self {
        Self { id, declared }
    }

    #[must_use]
    pub const fn id(&self) -> [u8; 4] {
        self.id
    }

    #[must_use]
    pub const fn declared(&self) -> u64 {
        self.declared
    }
}

pub(crate) struct DataSpan {
    pub start: u64,
    pub len: u64,
    pub declared: u32,
}

pub(crate) struct Scanned {
    pub format: Option<WavFormat>,
    pub brand: Brand,
    pub data: Option<DataSpan>,
    pub fact: Option<u64>,
    pub ds64_samples: Option<u64>,
    pub ds64_data_size: Option<u64>,
    pub kept: Vec<KeptChunk>,
    pub dropped: Vec<DroppedChunk>,
}

struct Ds64 {
    data_size: u64,
    sample_count: u64,
    table: Vec<([u8; 4], u64)>,
}

#[derive(Clone, Copy)]
struct Scope {
    end: u64,
    depth: u32,
    resume: u64,
}

pub(crate) fn scan<R: Read + Seek>(
    input: &mut Input<R>,
    random: bool,
) -> Result<Scanned, OpenError> {
    let header = read_array::<12, R>(input)?;
    let brand = brand_of(&header)?;
    let file_len = input.length();
    let mut ds64 = None;
    let parent_end = if brand == Brand::Rf64 {
        let parsed = read_ds64_chunk(input)?;
        let end = parent_end_from(8, parsed.riff_size, file_len)?;
        ds64 = Some(parsed.ds64);
        end
    } else {
        let declared = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let payload = if declared == u32::MAX {
            file_len.unwrap_or(u64::MAX).saturating_sub(8)
        } else {
            u64::from(declared)
        };
        parent_end_from(8, payload, file_len)?
    };
    let mut walker = Walker {
        input,
        random,
        file_len,
        ds64,
        scopes: vec![Scope {
            end: parent_end,
            depth: 0,
            resume: parent_end,
        }],
        chunks: 0,
        format: None,
        data: None,
        fact: None,
        kept: Vec::new(),
        dropped: Vec::new(),
        kept_bytes: 0,
        stop: false,
    };
    walker.walk()?;
    Ok(Scanned {
        format: walker.format,
        brand,
        data: walker.data,
        fact: walker.fact,
        ds64_samples: walker.ds64.as_ref().map(|item| item.sample_count),
        ds64_data_size: walker.ds64.as_ref().map(|item| item.data_size),
        kept: walker.kept,
        dropped: walker.dropped,
    })
}

struct ParsedDs64 {
    riff_size: u64,
    ds64: Ds64,
}

struct Walker<'a, R> {
    input: &'a mut Input<R>,
    random: bool,
    file_len: Option<u64>,
    ds64: Option<Ds64>,
    scopes: Vec<Scope>,
    chunks: u32,
    format: Option<WavFormat>,
    data: Option<DataSpan>,
    fact: Option<u64>,
    kept: Vec<KeptChunk>,
    dropped: Vec<DroppedChunk>,
    kept_bytes: u64,
    stop: bool,
}

impl<R: Read + Seek> Walker<'_, R> {
    fn walk(&mut self) -> Result<(), OpenError> {
        while !self.stop {
            let Some(scope) = self.scopes.last().copied() else {
                break;
            };
            let pos = self.input.position();
            if pos >= scope.end {
                if pos < scope.resume {
                    self.skip_to(scope.resume)?;
                }
                self.scopes.pop();
                continue;
            }
            if scope.end - pos < 8 {
                if scope.depth == 0 {
                    break;
                }
                return Err(OpenError::Truncated {
                    expected: pos + 8,
                    got: scope.end,
                });
            }
            let (id, declared) = self.read_header()?;
            self.dispatch(id, declared, scope.end, scope.depth)?;
        }
        Ok(())
    }

    fn dispatch(
        &mut self,
        id: [u8; 4],
        declared: u32,
        scope_end: u64,
        depth: u32,
    ) -> Result<(), OpenError> {
        let size = self.resolve(&id, declared)?;
        let payload_at = self.input.position();
        self.ensure_fits(payload_at, size, scope_end)?;
        let next = padded_end(payload_at, size, scope_end, self.file_len)?;
        if id == *b"fmt " {
            self.take_fmt(size, next)?;
        } else if id == *b"data" {
            self.take_data(declared, size, payload_at, next)?;
        } else if id == *b"fact" {
            self.take_fact(size, next)?;
        } else if id == *b"ds64" {
            self.take_ds64(size, next)?;
        } else if id == *b"LIST" || id == *b"list" {
            self.take_list(id, size, depth, next)?;
        } else {
            self.take_unknown(id, size, next)?;
        }
        Ok(())
    }

    fn take_fmt(&mut self, size: u64, next: u64) -> Result<(), OpenError> {
        let limit = self.input.limits().max_extradata_bytes;
        if size > limit {
            return Err(OpenError::HostileSize {
                declared: size,
                limit,
            });
        }
        if self.format.is_some() {
            self.skip_to(next)?;
            return Ok(());
        }
        let bytes = self.read_limited(size, limit)?;
        let format = parse_fmt(&bytes).map_err(FormatError::into_open)?;
        self.format = Some(format);
        self.skip_to(next)?;
        Ok(())
    }

    fn take_data(
        &mut self,
        declared: u32,
        size: u64,
        start: u64,
        next: u64,
    ) -> Result<(), OpenError> {
        if self.data.is_some() {
            return Err(OpenError::Source {
                message: "duplicate data chunk".to_owned(),
            });
        }
        self.data = Some(DataSpan {
            start,
            len: size,
            declared,
        });
        if self.random {
            self.skip_to(next)?;
        } else {
            self.stop = true;
        }
        Ok(())
    }

    fn take_fact(&mut self, size: u64, next: u64) -> Result<(), OpenError> {
        if size < 4 {
            return Err(OpenError::Truncated {
                expected: 4,
                got: size,
            });
        }
        let bytes = self.read_limited(4, 4)?;
        let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if raw != u32::MAX && self.fact.is_none() {
            self.fact = Some(u64::from(raw));
        }
        self.skip_to(next)?;
        Ok(())
    }

    fn take_ds64(&mut self, size: u64, next: u64) -> Result<(), OpenError> {
        if self.ds64.is_some() {
            self.skip_to(next)?;
            return Ok(());
        }
        let limit = self.input.limits().max_header_bytes;
        if size > limit {
            return Err(OpenError::HostileSize {
                declared: size,
                limit,
            });
        }
        if size < 28 {
            return Err(OpenError::Truncated {
                expected: 28,
                got: size,
            });
        }
        let bytes = self.read_limited(size, limit)?;
        self.ds64 = Some(parse_ds64_bytes(&bytes)?);
        self.skip_to(next)?;
        Ok(())
    }

    fn take_list(
        &mut self,
        id: [u8; 4],
        size: u64,
        depth: u32,
        next: u64,
    ) -> Result<(), OpenError> {
        if !self.random && size > MAX_KEPT_CHUNK {
            return Err(OpenError::HostileSize {
                declared: size,
                limit: MAX_KEPT_CHUNK,
            });
        }
        if depth >= MAX_DEPTH {
            return Err(OpenError::HostileSize {
                declared: u64::from(depth) + 1,
                limit: u64::from(MAX_DEPTH),
            });
        }
        if size < 4 {
            return Err(OpenError::Truncated {
                expected: 4,
                got: size,
            });
        }
        if size > MAX_KEPT_CHUNK || !self.keep_budget(size) {
            self.dropped.push(DroppedChunk::new(id, size));
            let payload_at = self.input.position();
            let Some(end) = payload_at.checked_add(size) else {
                return Err(OpenError::HostileSize {
                    declared: size,
                    limit: MAX_KEPT_CHUNK,
                });
            };
            let _kind = read_array::<4, R>(self.input)?;
            self.scopes.push(Scope {
                end,
                depth: depth + 1,
                resume: next,
            });
            return Ok(());
        }
        let bytes = self.read_limited(size, MAX_KEPT_CHUNK)?;
        validate_list(&bytes, depth + 1)?;
        self.remember(id, bytes)?;
        self.skip_to(next)?;
        Ok(())
    }

    fn take_unknown(&mut self, id: [u8; 4], size: u64, next: u64) -> Result<(), OpenError> {
        if !self.random && size > MAX_KEPT_CHUNK {
            return Err(OpenError::HostileSize {
                declared: size,
                limit: MAX_KEPT_CHUNK,
            });
        }
        if size > MAX_KEPT_CHUNK || !self.keep_budget(size) {
            self.dropped.push(DroppedChunk::new(id, size));
            self.skip_to(next)?;
            return Ok(());
        }
        let bytes = self.read_limited(size, MAX_KEPT_CHUNK)?;
        self.remember(id, bytes)?;
        self.skip_to(next)?;
        Ok(())
    }

    fn remember(&mut self, id: [u8; 4], payload: Vec<u8>) -> Result<(), OpenError> {
        let len = u64::try_from(payload.len()).unwrap_or(u64::MAX);
        self.kept_bytes = self.kept_bytes.saturating_add(len);
        self.kept.push(KeptChunk::new(id, payload));
        Ok(())
    }

    fn keep_budget(&self, size: u64) -> bool {
        let limit = self.input.limits().max_header_bytes;
        self.kept_bytes.saturating_add(size) <= limit
    }

    fn resolve(&self, id: &[u8; 4], declared: u32) -> Result<u64, OpenError> {
        if declared != u32::MAX {
            return Ok(u64::from(declared));
        }
        let Some(ds64) = &self.ds64 else {
            return Err(OpenError::HostileSize {
                declared: u64::from(u32::MAX),
                limit: self.room(),
            });
        };
        if id == b"data" {
            return Ok(ds64.data_size);
        }
        if let Some((_, size)) = ds64.table.iter().find(|(chunk, _)| chunk == id) {
            return Ok(*size);
        }
        Err(OpenError::HostileSize {
            declared: u64::from(u32::MAX),
            limit: self.room(),
        })
    }

    fn room(&self) -> u64 {
        let pos = self.input.position();
        self.scopes
            .last()
            .map(|scope| scope.end.saturating_sub(pos))
            .unwrap_or(0)
    }

    fn ensure_fits(&self, at: u64, size: u64, scope_end: u64) -> Result<(), OpenError> {
        let Some(end) = at.checked_add(size) else {
            return Err(OpenError::HostileSize {
                declared: size,
                limit: scope_end.saturating_sub(at),
            });
        };
        if end > scope_end {
            return Err(OpenError::HostileSize {
                declared: size,
                limit: scope_end.saturating_sub(at),
            });
        }
        if let Some(file_len) = self.file_len
            && end > file_len
        {
            return Err(OpenError::HostileSize {
                declared: size,
                limit: file_len.saturating_sub(at),
            });
        }
        Ok(())
    }

    fn read_header(&mut self) -> Result<([u8; 4], u32), OpenError> {
        let bytes = read_array::<8, R>(self.input)?;
        self.note_chunk()?;
        let mut id = [0; 4];
        id.copy_from_slice(&bytes[..4]);
        let declared = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Ok((id, declared))
    }

    fn note_chunk(&mut self) -> Result<(), OpenError> {
        self.chunks = self.chunks.saturating_add(1);
        if self.chunks > MAX_CHUNKS {
            return Err(OpenError::HostileSize {
                declared: u64::from(self.chunks),
                limit: u64::from(MAX_CHUNKS),
            });
        }
        Ok(())
    }

    fn read_limited(&mut self, size: u64, limit: u64) -> Result<Vec<u8>, OpenError> {
        self.input
            .read_limited(size, limit)
            .map_err(OpenError::from_header)
    }

    fn skip_to(&mut self, next: u64) -> Result<(), OpenError> {
        let pos = self.input.position();
        if next < pos {
            return Err(OpenError::Source {
                message: "chunk padding moved backwards".to_owned(),
            });
        }
        if next == pos {
            return Ok(());
        }
        if self.random {
            return self.input.seek(next).map_err(OpenError::from_header);
        }
        let mut left = next - pos;
        let mut buf = [0; 256];
        while left > 0 {
            let step = usize::try_from(left.min(256)).unwrap_or(256);
            self.input
                .read_exact(&mut buf[..step])
                .map_err(OpenError::from_header)?;
            left -= u64::try_from(step).unwrap_or(0);
        }
        Ok(())
    }
}

fn brand_of(header: &[u8; 12]) -> Result<Brand, OpenError> {
    if &header[8..12] != b"WAVE" {
        return Err(OpenError::Unrecognized {
            family: cutlass_codec::ContainerId::Riff,
        });
    }
    match &header[..4] {
        b"RIFF" => Ok(Brand::Riff),
        b"RF64" => Ok(Brand::Rf64),
        b"RIFX" | b"BW64" => Err(OpenError::UnsupportedBrand {
            family: cutlass_codec::ContainerId::Riff,
        }),
        _ => Err(OpenError::Unrecognized {
            family: cutlass_codec::ContainerId::Riff,
        }),
    }
}

fn parent_end_from(origin: u64, payload: u64, file_len: Option<u64>) -> Result<u64, OpenError> {
    let Some(end) = origin.checked_add(payload) else {
        return Err(OpenError::HostileSize {
            declared: payload,
            limit: u64::MAX - origin,
        });
    };
    Ok(match file_len {
        Some(len) => end.min(len),
        None => end,
    })
}

fn padded_end(
    payload_at: u64,
    size: u64,
    scope_end: u64,
    file_len: Option<u64>,
) -> Result<u64, OpenError> {
    let Some(payload_end) = payload_at.checked_add(size) else {
        return Err(OpenError::HostileSize {
            declared: size,
            limit: scope_end.saturating_sub(payload_at),
        });
    };
    if size.is_multiple_of(2) {
        return Ok(payload_end);
    }
    let Some(padded) = payload_end.checked_add(1) else {
        return Err(OpenError::HostileSize {
            declared: size + 1,
            limit: scope_end.saturating_sub(payload_at),
        });
    };
    if padded <= scope_end {
        return Ok(padded);
    }
    let at_file_end = file_len.is_some_and(|len| payload_end == len);
    if at_file_end {
        return Ok(payload_end);
    }
    Err(OpenError::HostileSize {
        declared: size + 1,
        limit: scope_end.saturating_sub(payload_at),
    })
}

fn read_ds64_chunk<R: Read + Seek>(input: &mut Input<R>) -> Result<ParsedDs64, OpenError> {
    let header = read_array::<8, R>(input)?;
    if &header[..4] != b"ds64" {
        return Err(OpenError::Source {
            message: "rf64 file is missing ds64".to_owned(),
        });
    }
    let declared = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if declared < 28 {
        return Err(OpenError::Truncated {
            expected: 28,
            got: u64::from(declared),
        });
    }
    let limit = input.limits().max_header_bytes;
    if u64::from(declared) > limit {
        return Err(OpenError::HostileSize {
            declared: u64::from(declared),
            limit,
        });
    }
    let bytes = input
        .read_limited(u64::from(declared), limit)
        .map_err(OpenError::from_header)?;
    let riff_size = u64_at(&bytes, 0)?;
    let ds64 = parse_ds64_bytes(&bytes)?;
    if !declared.is_multiple_of(2) {
        let mut pad = [0; 1];
        input.read_exact(&mut pad).map_err(OpenError::from_header)?;
    }
    Ok(ParsedDs64 { riff_size, ds64 })
}

fn validate_list(payload: &[u8], depth: u32) -> Result<(), OpenError> {
    if payload.len() < 4 {
        return Err(OpenError::Truncated {
            expected: 4,
            got: u64::try_from(payload.len()).unwrap_or(0),
        });
    }
    if depth > MAX_DEPTH {
        return Err(OpenError::HostileSize {
            declared: u64::from(depth),
            limit: u64::from(MAX_DEPTH),
        });
    }
    let mut stack = vec![(4usize, payload.len(), depth)];
    let mut seen = 0u32;
    while let Some((mut at, end, depth)) = stack.pop() {
        while at + 8 <= end {
            seen = seen.saturating_add(1);
            if seen > MAX_CHUNKS {
                return Err(OpenError::HostileSize {
                    declared: u64::from(seen),
                    limit: u64::from(MAX_CHUNKS),
                });
            }
            let id = id_at(payload, at)?;
            let declared = u32_at(payload, at + 4)?;
            if declared == u32::MAX {
                return Err(OpenError::HostileSize {
                    declared: u64::from(u32::MAX),
                    limit: u64::try_from(end.saturating_sub(at + 8)).unwrap_or(0),
                });
            }
            let size = usize::try_from(declared).map_err(|_| OpenError::HostileSize {
                declared: u64::from(declared),
                limit: u64::try_from(end.saturating_sub(at)).unwrap_or(0),
            })?;
            let body = at + 8;
            let Some(body_end) = body.checked_add(size) else {
                return Err(OpenError::HostileSize {
                    declared: u64::from(declared),
                    limit: u64::try_from(end.saturating_sub(body)).unwrap_or(0),
                });
            };
            if body_end > end {
                return Err(OpenError::HostileSize {
                    declared: u64::from(declared),
                    limit: u64::try_from(end.saturating_sub(body)).unwrap_or(0),
                });
            }
            if (id == *b"LIST" || id == *b"list") && size >= 4 {
                let child_depth = depth.saturating_add(1);
                if child_depth > MAX_DEPTH {
                    return Err(OpenError::HostileSize {
                        declared: u64::from(child_depth),
                        limit: u64::from(MAX_DEPTH),
                    });
                }
                stack.push((body + 4, body_end, child_depth));
            }
            at = body_end;
            if !declared.is_multiple_of(2) && at < end {
                at += 1;
            }
        }
    }
    Ok(())
}

fn parse_ds64_bytes(bytes: &[u8]) -> Result<Ds64, OpenError> {
    let data_size = u64_at(bytes, 8)?;
    let sample_count = u64_at(bytes, 16)?;
    let rows = u32_at(bytes, 24)?;
    if rows > MAX_DS64_ROWS {
        return Err(OpenError::HostileSize {
            declared: u64::from(rows),
            limit: u64::from(MAX_DS64_ROWS),
        });
    }
    let table_bytes = u64::from(rows) * 12;
    let have = u64::try_from(bytes.len().saturating_sub(28)).unwrap_or(0);
    if table_bytes > have {
        return Err(OpenError::Truncated {
            expected: 28 + table_bytes,
            got: u64::try_from(bytes.len()).unwrap_or(0),
        });
    }
    let mut table = Vec::new();
    let mut at = 28usize;
    for _ in 0..rows {
        table.push((id_at(bytes, at)?, u64_at(bytes, at + 4)?));
        at += 12;
    }
    Ok(Ds64 {
        data_size,
        sample_count,
        table,
    })
}

fn read_array<const N: usize, R: Read + Seek>(input: &mut Input<R>) -> Result<[u8; N], OpenError> {
    let mut buf = [0; N];
    input.read_exact(&mut buf).map_err(OpenError::from_header)?;
    Ok(buf)
}
