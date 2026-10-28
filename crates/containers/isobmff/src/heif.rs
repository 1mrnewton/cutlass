//! HEIF items. A still image is one packet per item. Grid reconstruction stays
//! with the codec; this crate returns the grid payload and the coded tiles.

use crate::index::Sample;
use crate::limits::{MAX_PACKET_BYTES, MAX_SAMPLES, MAX_TRACKS};
use crate::movie::BuiltTrack;
use crate::parse::{self, Reader};
use crate::stsd::codec_for;
use cutlass_container::{Error, MediaKind, Stream, StreamParams, Timescale, VideoParams};

struct Item {
    id: u32,
    kind: [u8; 4],
    method: u8,
    base: u64,
    extents: Vec<(u64, u64)>,
    extradata: Vec<u8>,
    width: u32,
    height: u32,
}

pub fn parse_meta(
    payload: &[u8],
    payload_at: u64,
    budget: &mut u32,
    depth: u32,
) -> Result<Vec<BuiltTrack>, Error> {
    let depth = parse::enter(depth)?;
    let body = if payload.len() >= 12 && payload.get(8..12) == Some(b"hdlr") {
        &payload[4..]
    } else {
        payload
    };
    let body_at = payload_at
        .saturating_add(u64::try_from(payload.len().saturating_sub(body.len())).unwrap_or(0));
    let children = parse::atoms(body, body_at, budget)?;
    let mut items: Vec<Item> = Vec::new();
    let mut primary = 0u32;
    let mut idat_at = None;
    let mut iloc = None;
    let mut iprp = None;
    for atom in &children {
        match &atom.kind {
            b"pitm" => primary = parse_pitm(atom.payload)?,
            b"iinf" => items = parse_iinf(atom.payload, budget)?,
            b"iloc" => iloc = Some(atom.payload),
            b"idat" => idat_at = Some(atom.payload_at),
            b"iprp" => iprp = Some(atom.payload),
            _ => {}
        }
    }
    if let Some(payload) = iloc {
        apply_iloc(&mut items, payload)?;
    }
    if let Some(payload) = iprp {
        parse_iprp(payload, &mut items, budget, depth)?;
    }
    if items.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(index) = items.iter().position(|item| item.id == primary) {
        items.swap(0, index);
    }
    let mut groups: Vec<BuiltTrack> = Vec::new();
    for item in items {
        if item.extents.is_empty() {
            continue;
        }
        let (offset, size) = place(&item, idat_at)?;
        let sample = Sample {
            offset,
            size,
            dts: 0,
            pts: 0,
            duration: 1,
            keyframe: true,
        };
        if let Some(track) = groups
            .iter_mut()
            .find(|track| track.stream.params.sample_entry == item.kind)
        {
            let next = i64::try_from(track.samples.len()).unwrap_or(i64::MAX);
            let mut sample = sample;
            sample.dts = next;
            sample.pts = next;
            if u64::try_from(track.samples.len()).unwrap_or(u64::MAX) >= MAX_SAMPLES {
                return Err(Error::Limit {
                    what: "samples",
                    value: MAX_SAMPLES,
                });
            }
            track.samples.push(sample);
            track.stream.params.duration_ticks =
                track.stream.params.duration_ticks.saturating_add(1);
        } else {
            if groups.len() >= MAX_TRACKS {
                return Err(Error::Limit {
                    what: "tracks",
                    value: u64::try_from(groups.len()).unwrap_or(u64::MAX),
                });
            }
            let kind = if item.width > 0 || item.height > 0 {
                MediaKind::Video(VideoParams {
                    width: item.width,
                    height: item.height,
                })
            } else if matches!(&item.kind, b"hvc1" | b"hev1" | b"av01" | b"avc1") {
                MediaKind::Video(VideoParams {
                    width: 0,
                    height: 0,
                })
            } else {
                MediaKind::Data
            };
            groups.push(BuiltTrack {
                stream: Stream {
                    params: StreamParams {
                        track_id: item.id,
                        codec: codec_for(item.kind),
                        kind,
                        timescale: Timescale(1),
                        language: *b"und",
                        extradata: item.extradata,
                        duration_ticks: 1,
                        sample_entry: item.kind,
                        color: None,
                    },
                    edits: Vec::new(),
                },
                samples: vec![sample],
                no_sync: false,
                self_contained: true,
            });
        }
    }
    Ok(groups)
}

fn place(item: &Item, idat_at: Option<u64>) -> Result<(u64, u32), Error> {
    if item.extents.len() != 1 {
        return Err(Error::Unsupported {
            what: "split item extents",
        });
    }
    let Some((extent_off, extent_len)) = item.extents.first().copied() else {
        return Err(Error::Missing {
            what: "item extent",
        });
    };
    if extent_len == 0 || extent_len > MAX_PACKET_BYTES {
        return Err(Error::Limit {
            what: "packet",
            value: extent_len,
        });
    }
    let size = u32::try_from(extent_len).map_err(|_| Error::Limit {
        what: "packet",
        value: extent_len,
    })?;
    let origin = if item.method == 1 {
        idat_at.ok_or(Error::Missing { what: "idat" })?
    } else if item.method == 0 {
        0
    } else {
        return Err(Error::Unsupported {
            what: "item construction method",
        });
    };
    let offset = origin
        .checked_add(item.base)
        .and_then(|value| value.checked_add(extent_off))
        .ok_or(Error::HostileSize {
            at: origin,
            claimed: extent_off,
        })?;
    Ok((offset, size))
}

fn parse_pitm(payload: &[u8]) -> Result<u32, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    if version == 0 {
        Ok(u32::from(reader.u16()?))
    } else {
        reader.u32()
    }
}

fn parse_iinf(payload: &[u8], budget: &mut u32) -> Result<Vec<Item>, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let count = if version == 0 {
        u32::from(reader.u16()?)
    } else {
        reader.u32()?
    };
    let mut items = Vec::new();
    for atom in parse::atoms(reader.remaining(), 0, budget)? {
        if atom.kind != *b"infe" {
            continue;
        }
        if items.len() >= usize::try_from(count).unwrap_or(0) && count > 0 {
            break;
        }
        if let Some(item) = parse_infe(atom.payload)? {
            items.push(item);
        }
    }
    Ok(items)
}

fn parse_infe(payload: &[u8]) -> Result<Option<Item>, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    if version < 2 {
        return Ok(None);
    }
    let id = if version == 2 {
        u32::from(reader.u16()?)
    } else {
        reader.u32()?
    };
    let _protection = reader.u16()?;
    let mut kind = [0u8; 4];
    kind.copy_from_slice(reader.take(4)?);
    Ok(Some(Item {
        id,
        kind,
        method: 0,
        base: 0,
        extents: Vec::new(),
        extradata: Vec::new(),
        width: 0,
        height: 0,
    }))
}

fn apply_iloc(items: &mut [Item], payload: &[u8]) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let sizes = reader.u8()?;
    let sizes2 = reader.u8()?;
    let offset_size = sizes >> 4;
    let length_size = sizes & 0x0f;
    let base_size = sizes2 >> 4;
    let index_size = if version == 0 { 0 } else { sizes2 & 0x0f };
    let count = if version < 2 {
        u32::from(reader.u16()?)
    } else {
        reader.u32()?
    };
    for _ in 0..count {
        let id = if version < 2 {
            u32::from(reader.u16()?)
        } else {
            reader.u32()?
        };
        let method = if version == 0 {
            0
        } else {
            let word = reader.u16()?;
            u8::try_from(word & 0x0f).unwrap_or(0)
        };
        let _data_ref = reader.u16()?;
        let base = read_sized(&mut reader, base_size)?;
        let extent_count = reader.u16()?;
        let mut extents = Vec::new();
        for _ in 0..extent_count {
            if index_size > 0 {
                let _index = read_sized(&mut reader, index_size)?;
            }
            let off = read_sized(&mut reader, offset_size)?;
            let len = read_sized(&mut reader, length_size)?;
            extents.push((off, len));
        }
        if let Some(item) = items.iter_mut().find(|item| item.id == id) {
            item.method = method;
            item.base = base;
            item.extents = extents;
        }
    }
    Ok(())
}

struct ItemProperty {
    kind: [u8; 4],
    body: Vec<u8>,
}

fn parse_iprp(
    payload: &[u8],
    items: &mut [Item],
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let _depth = parse::enter(depth)?;
    let mut properties = Vec::new();
    let mut ipma = Vec::new();
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"ipco" => {
                for prop in parse::atoms(atom.payload, 0, budget)? {
                    properties.push(ItemProperty {
                        kind: prop.kind,
                        body: prop.payload.to_vec(),
                    });
                }
            }
            b"ipma" => ipma = atom.payload.to_vec(),
            _ => {}
        }
    }
    if !ipma.is_empty() {
        apply_ipma(&ipma, &properties, items)?;
    }
    Ok(())
}

fn apply_ipma(
    payload: &[u8],
    properties: &[ItemProperty],
    items: &mut [Item],
) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    let flags_b = reader.take(3)?;
    let wide = flags_b[2] & 1 == 1;
    let count = reader.u32()?;
    for _ in 0..count {
        let id = if version == 0 {
            u32::from(reader.u16()?)
        } else {
            reader.u32()?
        };
        let assocs = reader.u8()?;
        let mut extradata = Vec::new();
        let mut width = 0u32;
        let mut height = 0u32;
        for _ in 0..assocs {
            let index = if wide {
                usize::from(reader.u16()? & 0x7fff)
            } else {
                usize::from(reader.u8()? & 0x7f)
            };
            if index == 0 {
                continue;
            }
            let Some(property) = properties.get(index - 1) else {
                continue;
            };
            if matches!(&property.kind, b"hvcC" | b"av1C" | b"avcC" | b"vpcC") {
                extradata = property.body.clone();
            }
            if property.kind == *b"ispe" && property.body.len() >= 12 {
                let body = &property.body;
                width = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                height = u32::from_be_bytes([body[8], body[9], body[10], body[11]]);
            }
        }
        if let Some(item) = items.iter_mut().find(|item| item.id == id) {
            item.extradata = extradata;
            item.width = width;
            item.height = height;
        }
    }
    Ok(())
}

fn read_sized(reader: &mut Reader<'_>, size: u8) -> Result<u64, Error> {
    if size > 8 {
        return Err(Error::Unsupported {
            what: "item offset size",
        });
    }
    let mut value = 0u64;
    for _ in 0..size {
        value = (value << 8) | u64::from(reader.u8()?);
    }
    Ok(value)
}
