//! Audio-only ISOBMFF for one AAC track.
//!
//! A file written here is `M4A `: `ftyp`, `moov` (`mvhd`, one `soun` track,
//! `mp4a` / `esds`, sample tables), and `mdat`. Samples are raw AAC access
//! units. An edit list records encoder delay and trailing padding in the
//! media timescale. Video, fragmented movies, and a second track are
//! rejected rather than dropped.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use crate::api::{Demuxer, Muxer, StreamInfo};
use cutlass_codec::access::{Channel, ChannelLayout, CodecError, CodecParams, Packet, SideData};
use cutlass_codec::{CodecId, ContainerId};

const MAX_FILE: usize = 32 << 20;
const MAX_SAMPLES: usize = 100_000;

pub struct Mp4Demuxer {
    buf: Vec<u8>,
    ready: VecDeque<Packet>,
    streams: Vec<StreamInfo>,
    parsed: bool,
    ended: bool,
}

pub struct Mp4Muxer {
    params: Option<CodecParams>,
    packets: Vec<Packet>,
    done: bool,
}

impl Mp4Demuxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            ready: VecDeque::new(),
            streams: Vec::new(),
            parsed: false,
            ended: false,
        }
    }

    pub fn push_bytes(&mut self, data: &[u8]) -> Result<(), CodecError> {
        if self.ended {
            return Err(CodecError::InvalidApi {
                what: "push after finish",
            });
        }
        if self.buf.len().saturating_add(data.len()) > MAX_FILE {
            return Err(CodecError::Limit { what: "file" });
        }
        self.buf.extend_from_slice(data);
        self.try_parse()
    }

    pub fn finish_bytes(&mut self) -> Result<(), CodecError> {
        self.ended = true;
        self.try_parse()?;
        if self.parsed {
            Ok(())
        } else {
            Err(CodecError::Truncated {
                at_bit: 0,
                what: "moov",
            })
        }
    }

    fn try_parse(&mut self) -> Result<(), CodecError> {
        if self.parsed {
            return Ok(());
        }
        if !complete_movie(&self.buf, self.ended)? {
            return Ok(());
        }
        let movie = parse_movie(&self.buf)?;
        self.streams = movie.streams;
        self.ready = movie.packets;
        self.parsed = true;
        Ok(())
    }
}

impl Default for Mp4Demuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for Mp4Demuxer {
    fn id(&self) -> ContainerId {
        ContainerId::Isobmff
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Option<Packet>, CodecError> {
        self.try_parse()?;
        if let Some(packet) = self.ready.pop_front() {
            return Ok(Some(packet));
        }
        if self.parsed {
            Ok(None)
        } else if self.ended {
            Err(CodecError::Truncated {
                at_bit: 0,
                what: "moov",
            })
        } else {
            Err(CodecError::NeedMoreData)
        }
    }
}

impl Mp4Muxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: None,
            packets: Vec::new(),
            done: false,
        }
    }
}

impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for Mp4Muxer {
    fn id(&self) -> ContainerId {
        ContainerId::Isobmff
    }

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, CodecError> {
        if self.params.is_some() {
            return Err(CodecError::Unsupported { what: "streams" });
        }
        if params.codec != CodecId::Aac || params.extra.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "audio specific config",
            });
        }
        if params.sample_rate == 0 {
            return Err(CodecError::InvalidApi {
                what: "sample rate",
            });
        }
        self.params = Some(params);
        Ok(0)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), CodecError> {
        if self.done {
            return Err(CodecError::InvalidApi {
                what: "write after finish",
            });
        }
        if self.params.is_none() {
            return Err(CodecError::InvalidApi { what: "stream" });
        }
        if packet.data.is_empty() || looks_like_adts(&packet.data) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "adts",
            });
        }
        if packet.duration < 0 {
            return Err(CodecError::InvalidApi { what: "duration" });
        }
        let rate = self
            .params
            .as_ref()
            .map(|params| params.sample_rate)
            .unwrap_or(0);
        if packet.timescale != rate {
            return Err(CodecError::Unsupported { what: "timescale" });
        }
        if self.packets.len() >= MAX_SAMPLES {
            return Err(CodecError::Limit { what: "samples" });
        }
        self.packets.push(packet.clone());
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if self.done {
            return Err(CodecError::InvalidApi {
                what: "second finish",
            });
        }
        self.done = true;
        let Some(params) = self.params.clone() else {
            return Err(CodecError::InvalidApi { what: "stream" });
        };
        write_movie(&params, &self.packets)
    }
}

struct Movie {
    streams: Vec<StreamInfo>,
    packets: VecDeque<Packet>,
}

fn write_movie(params: &CodecParams, packets: &[Packet]) -> Result<Vec<u8>, CodecError> {
    let rate = params.sample_rate;
    let channels = params.layout.as_ref().map(ChannelLayout::len).unwrap_or(1);
    let (delay, padding) = gapless(packets, rate)?;
    let media: u64 = packets
        .iter()
        .map(|packet| u64::try_from(packet.duration).unwrap_or(0))
        .sum();
    let presentation = media
        .saturating_sub(u64::from(delay))
        .saturating_sub(u64::from(padding));
    let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
    let bitrate = if media == 0 {
        0
    } else {
        u32::try_from((u128::from(bytes as u64) * 8 * u128::from(rate)) / u128::from(media))
            .unwrap_or(u32::MAX)
    };
    let mdat_payload: Vec<u8> = packets
        .iter()
        .flat_map(|packet| packet.data.iter().copied())
        .collect();
    let ftyp = box_of(
        b"ftyp",
        &[
            b"M4A ".as_slice(),
            &0u32.to_be_bytes(),
            b"M4A ",
            b"mp42",
            b"isom",
        ]
        .concat(),
    )?;
    let plan = Built {
        rate,
        presentation,
        media,
        delay,
        channels: u16::try_from(channels).unwrap_or(u16::MAX),
        bitrate,
        asc: &params.extra,
        packets,
        sample_offset: 0,
    };
    let moov = moov(&plan)?;
    let mdat = box_of(b"mdat", &mdat_payload)?;
    let sample_offset = u32::try_from(ftyp.len() + moov.len() + 8)
        .map_err(|_| CodecError::Limit { what: "file" })?;
    let mut moov = moov;
    patch_stco(&mut moov, sample_offset)?;
    let mut file = ftyp;
    file.extend(moov);
    file.extend(mdat);
    if file.len() > MAX_FILE {
        return Err(CodecError::Limit { what: "file" });
    }
    Ok(file)
}

fn gapless(packets: &[Packet], rate: u32) -> Result<(u32, u32), CodecError> {
    let mut delay = 0u32;
    let mut padding = 0u32;
    let mut seen = false;
    for packet in packets {
        for side in &packet.side {
            if let SideData::Gapless {
                encoder_delay,
                trailing_padding,
                sample_rate,
            } = side
            {
                if *sample_rate != rate {
                    return Err(CodecError::InvalidApi {
                        what: "sample rate",
                    });
                }
                if !seen {
                    delay = *encoder_delay;
                    seen = true;
                } else if *encoder_delay != delay {
                    return Err(CodecError::Unsupported {
                        what: "encoder delay",
                    });
                }
                padding = padding.max(*trailing_padding);
            }
        }
    }
    Ok((delay, padding))
}

struct Built<'a> {
    rate: u32,
    presentation: u64,
    media: u64,
    delay: u32,
    channels: u16,
    bitrate: u32,
    asc: &'a [u8],
    packets: &'a [Packet],
    sample_offset: u32,
}

fn moov(plan: &Built<'_>) -> Result<Vec<u8>, CodecError> {
    let mvhd = full_box(b"mvhd", 0, &mvhd_body(plan.rate, plan.presentation)?)?;
    let trak = trak(plan)?;
    box_of(b"moov", &[mvhd, trak].concat())
}

fn trak(plan: &Built<'_>) -> Result<Vec<u8>, CodecError> {
    let tkhd = full_box(b"tkhd", 0, &tkhd_body(plan.presentation)?)?;
    let elst = full_box(b"elst", 0, &elst_body(plan.presentation, plan.delay)?)?;
    let edts = box_of(b"edts", &elst)?;
    let mdhd = full_box(b"mdhd", 0, &mdhd_body(plan.rate, plan.media)?)?;
    let hdlr = full_box(b"hdlr", 0, &hdlr_body())?;
    let smhd = full_box(b"smhd", 0, &[0, 0, 0, 0])?;
    let url = full_box(b"url ", 1, &[])?;
    let dref = full_box(
        b"dref",
        0,
        &[1u32.to_be_bytes().as_slice(), url.as_slice()].concat(),
    )?;
    let dinf = box_of(b"dinf", &dref)?;
    let stbl = stbl(
        plan.rate,
        plan.channels,
        plan.bitrate,
        plan.asc,
        plan.packets,
        plan.sample_offset,
    )?;
    let minf = box_of(b"minf", &[smhd, dinf, stbl].concat())?;
    let mdia = box_of(b"mdia", &[mdhd, hdlr, minf].concat())?;
    box_of(b"trak", &[tkhd, edts, mdia].concat())
}

fn stbl(
    rate: u32,
    channels: u16,
    bitrate: u32,
    asc: &[u8],
    packets: &[Packet],
    sample_offset: u32,
) -> Result<Vec<u8>, CodecError> {
    let mp4a = audio_entry(rate, channels, bitrate, asc)?;
    let stsd = full_box(
        b"stsd",
        0,
        &[1u32.to_be_bytes().as_slice(), mp4a.as_slice()].concat(),
    )?;
    let stts = full_box(b"stts", 0, &stts_body(packets)?)?;
    let count = u32::try_from(packets.len()).unwrap_or(0);
    let stsc = full_box(
        b"stsc",
        0,
        &[
            1u32.to_be_bytes().as_slice(),
            1u32.to_be_bytes().as_slice(),
            count.to_be_bytes().as_slice(),
            1u32.to_be_bytes().as_slice(),
        ]
        .concat(),
    )?;
    let stsz = full_box(b"stsz", 0, &stsz_body(packets)?)?;
    let stco = full_box(
        b"stco",
        0,
        &[
            1u32.to_be_bytes().as_slice(),
            sample_offset.to_be_bytes().as_slice(),
        ]
        .concat(),
    )?;
    box_of(b"stbl", &[stsd, stts, stsc, stsz, stco].concat())
}

fn audio_entry(rate: u32, channels: u16, bitrate: u32, asc: &[u8]) -> Result<Vec<u8>, CodecError> {
    let esds = full_box(b"esds", 0, &esds_body(bitrate, asc)?)?;
    let mut fields = vec![0u8; 28];
    fields[6] = 0;
    fields[7] = 1;
    fields[16] = u8::try_from(channels >> 8).unwrap_or(0);
    fields[17] = u8::try_from(channels & 0xFF).unwrap_or(0);
    fields[18] = 0;
    fields[19] = 16;
    let rate_fixed = rate << 16;
    fields[24..28].copy_from_slice(&rate_fixed.to_be_bytes());
    fields.extend(esds);
    box_of(b"mp4a", &fields)
}

fn esds_body(bitrate: u32, asc: &[u8]) -> Result<Vec<u8>, CodecError> {
    let specific = desc(0x05, asc)?;
    let mut decoder = Vec::new();
    decoder.push(0x40);
    decoder.push(0x15);
    decoder.extend_from_slice(&[0, 0, 0]);
    decoder.extend_from_slice(&bitrate.to_be_bytes());
    decoder.extend_from_slice(&bitrate.to_be_bytes());
    decoder.extend(specific);
    let decoder = desc(0x04, &decoder)?;
    let sl = desc(0x06, &[0x02])?;
    let mut es = Vec::new();
    es.extend_from_slice(&1u16.to_be_bytes());
    es.push(0);
    es.extend(decoder);
    es.extend(sl);
    desc(0x03, &es)
}

fn desc(tag: u8, body: &[u8]) -> Result<Vec<u8>, CodecError> {
    let mut out = Vec::with_capacity(body.len() + 5);
    out.push(tag);
    put_expandable(&mut out, body.len())?;
    out.extend_from_slice(body);
    Ok(out)
}

fn put_expandable(out: &mut Vec<u8>, size: usize) -> Result<(), CodecError> {
    if size >= 1 << 28 {
        return Err(CodecError::Limit { what: "esds" });
    }
    let mut parts = [0u8; 4];
    let mut value = size;
    let mut count = 0;
    loop {
        parts[count] = u8::try_from(value & 0x7F).unwrap_or(0);
        value >>= 7;
        count += 1;
        if value == 0 || count == 4 {
            break;
        }
    }
    for index in (0..count).rev() {
        let more = if index == 0 { 0 } else { 0x80 };
        out.push(parts[index] | more);
    }
    Ok(())
}

fn mvhd_body(rate: u32, duration: u64) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&rate.to_be_bytes());
    body.extend_from_slice(&fit_u32(duration)?.to_be_bytes());
    body.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    body.extend_from_slice(&0x0100u16.to_be_bytes());
    body.extend_from_slice(&[0u8; 10]);
    body.extend_from_slice(&identity_matrix());
    body.extend_from_slice(&[0u8; 24]);
    body.extend_from_slice(&2u32.to_be_bytes());
    Ok(body)
}

fn tkhd_body(duration: u64) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&1u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&fit_u32(duration)?.to_be_bytes());
    body.extend_from_slice(&[0u8; 8]);
    body.extend_from_slice(&0u16.to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    body.extend_from_slice(&0x0100u16.to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    body.extend_from_slice(&identity_matrix());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    Ok(body)
}

fn mdhd_body(rate: u32, duration: u64) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&rate.to_be_bytes());
    body.extend_from_slice(&fit_u32(duration)?.to_be_bytes());
    body.extend_from_slice(&0x55C4u16.to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    Ok(body)
}

fn elst_body(presentation: u64, delay: u32) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_be_bytes());
    body.extend_from_slice(&fit_u32(presentation)?.to_be_bytes());
    body.extend_from_slice(&delay.to_be_bytes());
    body.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    Ok(body)
}

fn hdlr_body() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(b"soun");
    body.extend_from_slice(&[0u8; 12]);
    body.extend_from_slice(b"SoundHandler\0");
    body
}

fn identity_matrix() -> Vec<u8> {
    let values = [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
    values
        .into_iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn stts_body(packets: &[Packet]) -> Result<Vec<u8>, CodecError> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for packet in packets {
        let duration = fit_u32(u64::try_from(packet.duration).unwrap_or(0))?;
        if let Some(last) = runs.last_mut()
            && last.1 == duration
        {
            last.0 = last.0.saturating_add(1);
        } else {
            runs.push((1, duration));
        }
    }
    let mut body = Vec::new();
    body.extend_from_slice(&u32::try_from(runs.len()).unwrap_or(0).to_be_bytes());
    for (count, duration) in runs {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&duration.to_be_bytes());
    }
    Ok(body)
}

fn stsz_body(packets: &[Packet]) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&u32::try_from(packets.len()).unwrap_or(0).to_be_bytes());
    for packet in packets {
        body.extend_from_slice(&fit_u32(packet.data.len() as u64)?.to_be_bytes());
    }
    Ok(body)
}

fn patch_stco(moov: &mut [u8], offset: u32) -> Result<(), CodecError> {
    let Some(at) = find_box(moov, b"stco") else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "stco",
        });
    };
    let payload = at + 8;
    let slot = payload + 4 + 4;
    if slot + 4 > moov.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "stco",
        });
    }
    moov[slot..slot + 4].copy_from_slice(&offset.to_be_bytes());
    Ok(())
}

fn find_box(data: &[u8], kind: &[u8; 4]) -> Option<usize> {
    let mut offset = 0;
    while offset + 8 <= data.len() {
        let size = u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        if size < 8 || offset + size > data.len() {
            return None;
        }
        let found = &data[offset + 4..offset + 8];
        if found == kind {
            return Some(offset);
        }
        if is_container(found) {
            let payload = &data[offset + 8..offset + size];
            if let Some(inner) = find_box(payload, kind) {
                return Some(offset + 8 + inner);
            }
        }
        offset += size;
    }
    None
}

fn is_container(kind: &[u8]) -> bool {
    matches!(
        kind,
        b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"dinf"
    )
}

fn box_of(kind: &[u8; 4], payload: &[u8]) -> Result<Vec<u8>, CodecError> {
    let size = 8usize
        .checked_add(payload.len())
        .ok_or(CodecError::Limit { what: "file" })?;
    let size_u32 = u32::try_from(size).map_err(|_| CodecError::Limit { what: "file" })?;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(&size_u32.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    Ok(out)
}

fn full_box(kind: &[u8; 4], flags: u32, payload: &[u8]) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::with_capacity(payload.len() + 4);
    let mut head = (flags & 0x00FF_FFFF).to_be_bytes();
    head[0] = 0;
    body.extend_from_slice(&head);
    body.extend_from_slice(payload);
    box_of(kind, &body)
}

fn fit_u32(value: u64) -> Result<u32, CodecError> {
    u32::try_from(value).map_err(|_| CodecError::Limit { what: "duration" })
}

fn complete_movie(data: &[u8], ended: bool) -> Result<bool, CodecError> {
    let mut offset = 0;
    let mut moov = false;
    let mut mdat = false;
    while offset + 8 <= data.len() {
        let header = read_box_size(data, offset)?;
        let Some((size, header_len)) = header else {
            return Ok(false);
        };
        if offset + size > data.len() {
            return Ok(false);
        }
        let kind = &data[offset + 4..offset + 8];
        if size32_is_end(data, offset) && !ended {
            return Ok(false);
        }
        if kind == b"moof" {
            return Err(CodecError::Unsupported { what: "fragment" });
        }
        if kind == b"moov" {
            moov = true;
        }
        if kind == b"mdat" {
            mdat = true;
        }
        offset += size;
        let _ = header_len;
    }
    if offset < data.len() && data.len() - offset < 8 {
        return Ok(false);
    }
    Ok(moov && mdat)
}

fn size32_is_end(data: &[u8], offset: usize) -> bool {
    data.get(offset..offset + 4)
        .is_some_and(|bytes| bytes == [0, 0, 0, 0])
}

fn read_box_size(data: &[u8], offset: usize) -> Result<Option<(usize, usize)>, CodecError> {
    if offset + 8 > data.len() {
        return Ok(None);
    }
    let size32 = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
    if size32 == 1 {
        if offset + 16 > data.len() {
            return Ok(None);
        }
        let size64 = u64::from_be_bytes(data[offset + 8..offset + 16].try_into().unwrap_or([0; 8]));
        let size = usize::try_from(size64).map_err(|_| CodecError::Limit { what: "file" })?;
        if size < 16 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "box",
            });
        }
        Ok(Some((size, 16)))
    } else if size32 == 0 {
        Ok(Some((data.len() - offset, 8)))
    } else {
        let size = usize::try_from(size32).unwrap_or(0);
        if size < 8 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "box",
            });
        }
        Ok(Some((size, 8)))
    }
}

fn parse_movie(data: &[u8]) -> Result<Movie, CodecError> {
    let mut moov = None;
    let mut offset = 0;
    while offset + 8 <= data.len() {
        let Some((size, header)) = read_box_size(data, offset)? else {
            break;
        };
        if offset + size > data.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "box",
            });
        }
        let kind = &data[offset + 4..offset + 8];
        if kind == b"moov" {
            moov = Some(&data[offset + header..offset + size]);
        }
        offset += size;
    }
    let Some(moov) = moov else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "moov",
        });
    };
    let mut movie_scale = 0u32;
    let mut audio = None;
    for child in children(moov, 0)? {
        if child.kind == *b"mvhd" {
            movie_scale = mvhd_timescale(child.payload)?;
        } else if child.kind == *b"trak" {
            if audio.is_some() {
                return Err(CodecError::Unsupported { what: "streams" });
            }
            audio = Some(parse_trak(data, child.payload, movie_scale)?);
        }
    }
    let Some(audio) = audio else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "track",
        });
    };
    if movie_scale == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "timescale",
        });
    }
    Ok(audio)
}

struct Child<'a> {
    kind: [u8; 4],
    payload: &'a [u8],
}

fn children(payload: &[u8], skip: usize) -> Result<Vec<Child<'_>>, CodecError> {
    let mut out = Vec::new();
    let mut offset = skip;
    while offset + 8 <= payload.len() {
        let Some((size, header)) = read_box_size(payload, offset)? else {
            break;
        };
        if offset + size > payload.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "box",
            });
        }
        let mut kind = [0u8; 4];
        kind.copy_from_slice(&payload[offset + 4..offset + 8]);
        out.push(Child {
            kind,
            payload: &payload[offset + header..offset + size],
        });
        offset += size;
    }
    Ok(out)
}

fn parse_trak(file: &[u8], trak: &[u8], movie_scale: u32) -> Result<Movie, CodecError> {
    let mut handler = [0u8; 4];
    let mut media_scale = 0u32;
    let mut media_duration = 0u64;
    let mut edit = Edit::none();
    let mut sample = None;
    for child in children(trak, 0)? {
        if child.kind == *b"edts" {
            edit = parse_edts(child.payload)?;
        } else if child.kind == *b"mdia" {
            for media in children(child.payload, 0)? {
                if media.kind == *b"mdhd" {
                    (media_scale, media_duration) = mdhd(media.payload)?;
                } else if media.kind == *b"hdlr" {
                    handler = hdlr_type(media.payload)?;
                } else if media.kind == *b"minf" {
                    sample = Some(parse_minf(file, media.payload)?);
                }
            }
        }
    }
    if handler == *b"vide" {
        return Err(CodecError::Unsupported {
            what: "video track",
        });
    }
    if handler != *b"soun" {
        return Err(CodecError::Unsupported { what: "handler" });
    }
    let Some(sample) = sample else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "sample table",
        });
    };
    if media_scale == 0 || movie_scale == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "timescale",
        });
    }
    let (delay, padding) = edit_gap(edit, movie_scale, media_scale, media_duration);
    let mut params = CodecParams::audio(CodecId::Aac, media_scale, sample.layout);
    params.extra = sample.asc;
    params.bitrate = sample.bitrate;
    let stream = StreamInfo {
        index: 0,
        params,
        timescale: media_scale,
    };
    let mut packets = VecDeque::new();
    let mut pts = 0i64;
    for (index, packet_data) in sample.packets.into_iter().enumerate() {
        let duration = i64::from(packet_data.duration);
        let mut packet = Packet::audio(packet_data.data, pts, duration, media_scale);
        if delay > 0 || padding > 0 {
            packet.side.push(SideData::Gapless {
                encoder_delay: delay,
                trailing_padding: padding,
                sample_rate: media_scale,
            });
        }
        if index == 0 && !stream.params.extra.is_empty() {
            packet
                .side
                .push(SideData::Config(stream.params.extra.clone()));
        }
        pts = pts.saturating_add(duration);
        packets.push_back(packet);
    }
    Ok(Movie {
        streams: vec![stream],
        packets,
    })
}

struct SampleTable {
    asc: Vec<u8>,
    layout: ChannelLayout,
    bitrate: u32,
    packets: Vec<RawSample>,
}

struct RawSample {
    data: Vec<u8>,
    duration: u32,
}

struct Edit {
    segment: u64,
    media_time: i64,
    present: bool,
}

impl Edit {
    fn none() -> Self {
        Self {
            segment: 0,
            media_time: 0,
            present: false,
        }
    }
}

fn edit_gap(edit: Edit, movie_scale: u32, media_scale: u32, media_duration: u64) -> (u32, u32) {
    if !edit.present || edit.media_time < 0 {
        return (0, 0);
    }
    let delay = u32::try_from(edit.media_time).unwrap_or(u32::MAX);
    let segment_media =
        u128::from(edit.segment) * u128::from(media_scale) / u128::from(movie_scale);
    let consumed = u128::from(delay) + segment_media;
    let trailing = u128::from(media_duration).saturating_sub(consumed);
    let padding = u32::try_from(trailing).unwrap_or(u32::MAX);
    (delay, padding)
}

fn parse_edts(payload: &[u8]) -> Result<Edit, CodecError> {
    for child in children(payload, 0)? {
        if child.kind == *b"elst" {
            return parse_elst(child.payload);
        }
    }
    Ok(Edit::none())
}

fn parse_elst(payload: &[u8]) -> Result<Edit, CodecError> {
    if payload.len() < 8 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "elst",
        });
    }
    let version = payload[0];
    let count = u32::from_be_bytes(payload[4..8].try_into().unwrap_or([0; 4]));
    if count == 0 {
        return Ok(Edit::none());
    }
    if version == 0 {
        if payload.len() < 20 {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "elst",
            });
        }
        let segment = u64::from(u32::from_be_bytes(
            payload[8..12].try_into().unwrap_or([0; 4]),
        ));
        let media_time = i32::from_be_bytes(payload[12..16].try_into().unwrap_or([0; 4]));
        return Ok(Edit {
            segment,
            media_time: i64::from(media_time),
            present: media_time >= 0,
        });
    }
    if version == 1 {
        if payload.len() < 28 {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "elst",
            });
        }
        let segment = u64::from_be_bytes(payload[8..16].try_into().unwrap_or([0; 8]));
        let media_time = i64::from_be_bytes(payload[16..24].try_into().unwrap_or([0; 8]));
        return Ok(Edit {
            segment,
            media_time,
            present: media_time >= 0,
        });
    }
    Err(CodecError::Unsupported { what: "elst" })
}

fn mvhd_timescale(payload: &[u8]) -> Result<u32, CodecError> {
    let version = *payload.first().ok_or(CodecError::Truncated {
        at_bit: 0,
        what: "mvhd",
    })?;
    if version == 0 && payload.len() >= 16 {
        return Ok(u32::from_be_bytes(
            payload[12..16].try_into().unwrap_or([0; 4]),
        ));
    }
    if version == 1 && payload.len() >= 28 {
        return Ok(u32::from_be_bytes(
            payload[20..24].try_into().unwrap_or([0; 4]),
        ));
    }
    Err(CodecError::Truncated {
        at_bit: 0,
        what: "mvhd",
    })
}

fn mdhd(payload: &[u8]) -> Result<(u32, u64), CodecError> {
    let version = *payload.first().ok_or(CodecError::Truncated {
        at_bit: 0,
        what: "mdhd",
    })?;
    if version == 0 && payload.len() >= 20 {
        let scale = u32::from_be_bytes(payload[12..16].try_into().unwrap_or([0; 4]));
        let duration = u64::from(u32::from_be_bytes(
            payload[16..20].try_into().unwrap_or([0; 4]),
        ));
        return Ok((scale, duration));
    }
    if version == 1 && payload.len() >= 32 {
        let scale = u32::from_be_bytes(payload[20..24].try_into().unwrap_or([0; 4]));
        let duration = u64::from_be_bytes(payload[24..32].try_into().unwrap_or([0; 8]));
        return Ok((scale, duration));
    }
    Err(CodecError::Truncated {
        at_bit: 0,
        what: "mdhd",
    })
}

fn hdlr_type(payload: &[u8]) -> Result<[u8; 4], CodecError> {
    if payload.len() < 12 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "hdlr",
        });
    }
    let mut kind = [0u8; 4];
    kind.copy_from_slice(&payload[8..12]);
    Ok(kind)
}

fn parse_minf(file: &[u8], minf: &[u8]) -> Result<SampleTable, CodecError> {
    for child in children(minf, 0)? {
        if child.kind == *b"stbl" {
            return parse_stbl(file, child.payload);
        }
    }
    Err(CodecError::Malformed {
        at_bit: 0,
        what: "sample table",
    })
}

fn parse_stbl(file: &[u8], stbl: &[u8]) -> Result<SampleTable, CodecError> {
    let mut entry = None;
    let mut durations = Vec::new();
    let mut sizes = Vec::new();
    let mut chunks = Vec::new();
    let mut offsets = Vec::new();
    for child in children(stbl, 0)? {
        if child.kind == *b"stsd" {
            entry = Some(parse_stsd(child.payload)?);
        } else if child.kind == *b"stts" {
            durations = parse_stts(child.payload)?;
        } else if child.kind == *b"stsz" {
            sizes = parse_stsz(child.payload)?;
        } else if child.kind == *b"stsc" {
            chunks = parse_stsc(child.payload)?;
        } else if child.kind == *b"stco" {
            offsets = parse_offsets(child.payload, false)?;
        } else if child.kind == *b"co64" {
            offsets = parse_offsets(child.payload, true)?;
        }
    }
    let Some((asc, layout, bitrate)) = entry else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "stsd",
        });
    };
    if sizes.len() > MAX_SAMPLES {
        return Err(CodecError::Limit { what: "samples" });
    }
    let packets = assemble(file, &durations, &sizes, &chunks, &offsets)?;
    Ok(SampleTable {
        asc,
        layout,
        bitrate,
        packets,
    })
}

fn assemble(
    file: &[u8],
    durations: &[(u32, u32)],
    sizes: &[u32],
    chunks: &[(u32, u32)],
    offsets: &[u64],
) -> Result<Vec<RawSample>, CodecError> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }
    if chunks.is_empty() || offsets.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "chunk",
        });
    }
    let mut duration_of = Vec::with_capacity(sizes.len());
    for (count, duration) in durations {
        for _ in 0..*count {
            duration_of.push(*duration);
            if duration_of.len() > sizes.len() {
                break;
            }
        }
    }
    if duration_of.len() < sizes.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "stts",
        });
    }
    let mut samples = Vec::with_capacity(sizes.len());
    let mut index = 0usize;
    for (chunk_index, chunk_offset) in offsets.iter().enumerate() {
        let per = samples_in_chunk(chunks, u32::try_from(chunk_index).unwrap_or(0) + 1)?;
        let mut cursor = *chunk_offset;
        for _ in 0..per {
            if index >= sizes.len() {
                break;
            }
            let size = usize::try_from(sizes[index]).unwrap_or(0);
            let start = usize::try_from(cursor).map_err(|_| CodecError::Limit { what: "file" })?;
            let end = start
                .checked_add(size)
                .ok_or(CodecError::Limit { what: "file" })?;
            if end > file.len() {
                return Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "mdat",
                });
            }
            let data = file[start..end].to_vec();
            if looks_like_adts(&data) {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "adts",
                });
            }
            samples.push(RawSample {
                data,
                duration: duration_of[index],
            });
            cursor = cursor.saturating_add(u64::try_from(size).unwrap_or(0));
            index += 1;
        }
    }
    if index != sizes.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "chunk",
        });
    }
    Ok(samples)
}

fn samples_in_chunk(chunks: &[(u32, u32)], chunk: u32) -> Result<u32, CodecError> {
    let mut per = 0u32;
    let mut found = false;
    for (first, count) in chunks {
        if *first <= chunk {
            per = *count;
            found = true;
        }
    }
    if found {
        Ok(per)
    } else {
        Err(CodecError::Malformed {
            at_bit: 0,
            what: "chunk",
        })
    }
}

fn parse_stsd(payload: &[u8]) -> Result<(Vec<u8>, ChannelLayout, u32), CodecError> {
    if payload.len() < 8 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stsd",
        });
    }
    let entries = children(payload, 8)?;
    let Some(entry) = entries.first() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "stsd",
        });
    };
    if entry.kind != *b"mp4a" {
        return Err(CodecError::Unsupported {
            what: "sample entry",
        });
    }
    let rest = audio_children(entry.payload);
    let mut asc = None;
    let mut bitrate = 0u32;
    for child in children(rest, 0)? {
        if child.kind == *b"esds" {
            let parsed = parse_esds(child.payload)?;
            asc = Some(parsed.0);
            bitrate = parsed.1;
        }
    }
    let Some(asc) = asc else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "esds",
        });
    };
    let layout = layout_from_asc(&asc).unwrap_or_else(ChannelLayout::mono);
    Ok((asc, layout, bitrate))
}

fn audio_children(payload: &[u8]) -> &[u8] {
    for start in [20usize, 28, 44] {
        let Some(rest) = payload.get(start..) else {
            continue;
        };
        if rest.len() < 8 {
            continue;
        }
        let size = u32::from_be_bytes(rest[0..4].try_into().unwrap_or([0; 4])) as usize;
        let kind = &rest[4..8];
        if size >= 8
            && size <= rest.len()
            && kind
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b' ')
        {
            return rest;
        }
    }
    &[]
}

fn parse_esds(payload: &[u8]) -> Result<(Vec<u8>, u32), CodecError> {
    if payload.len() < 4 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "esds",
        });
    }
    let (tag, body, _) = take_desc(&payload[4..], 0)?;
    if tag != 0x03 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "esds",
        });
    }
    if body.len() < 3 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "esds",
        });
    }
    let flags = body[2];
    let mut at = 3usize;
    if flags & 0x80 != 0 {
        at += 2;
    }
    if flags & 0x40 != 0 {
        if at >= body.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "esds",
            });
        }
        at += 1 + usize::from(body[at]);
    }
    if flags & 0x20 != 0 {
        at += 2;
    }
    let (decoder_tag, decoder, _) = take_desc(body, at)?;
    if decoder_tag != 0x04 || decoder.len() < 13 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "esds",
        });
    }
    if decoder[0] != 0x40 {
        return Err(CodecError::Unsupported {
            what: "audio object type",
        });
    }
    let bitrate = u32::from_be_bytes(decoder[9..13].try_into().unwrap_or([0; 4]));
    let (specific_tag, specific, _) = take_desc(decoder, 13)?;
    if specific_tag != 0x05 || specific.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "esds",
        });
    }
    Ok((specific.to_vec(), bitrate))
}

fn take_desc(data: &[u8], mut pos: usize) -> Result<(u8, &[u8], usize), CodecError> {
    if pos >= data.len() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "esds",
        });
    }
    let tag = data[pos];
    pos += 1;
    let mut size = 0usize;
    for _ in 0..4 {
        if pos >= data.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "esds",
            });
        }
        let byte = data[pos];
        pos += 1;
        size = (size << 7) | usize::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
    }
    let end = pos
        .checked_add(size)
        .ok_or(CodecError::Limit { what: "esds" })?;
    if end > data.len() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "esds",
        });
    }
    Ok((tag, &data[pos..end], end))
}

fn layout_from_asc(asc: &[u8]) -> Option<ChannelLayout> {
    let mut bits = AscBits { data: asc, pos: 0 };
    let aot = bits.aot()?;
    let _rate = bits.rate()?;
    let channels = bits.take(4)?;
    if aot == 5 || aot == 29 {
        let _ext = bits.rate()?;
        let _core = bits.aot()?;
    }
    layout_for(channels)
}

struct AscBits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl AscBits<'_> {
    fn take(&mut self, n: u32) -> Option<u32> {
        if n > 24 || self.pos + n as usize > self.data.len() * 8 {
            return None;
        }
        let mut value = 0u32;
        for _ in 0..n {
            let byte = self.data[self.pos / 8];
            let shift = 7 - (self.pos % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.pos += 1;
        }
        Some(value)
    }

    fn aot(&mut self) -> Option<u32> {
        let aot = self.take(5)?;
        if aot == 31 {
            Some(32 + self.take(6)?)
        } else {
            Some(aot)
        }
    }

    fn rate(&mut self) -> Option<u32> {
        let index = self.take(4)?;
        if index == 0xF {
            self.take(24)?;
        }
        Some(index)
    }
}

fn layout_for(config: u32) -> Option<ChannelLayout> {
    let channels = match config {
        1 => return Some(ChannelLayout::mono()),
        2 => return Some(ChannelLayout::stereo()),
        3 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
        ],
        4 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackCenter,
        ],
        5 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
        ],
        6 => return Some(ChannelLayout::five_one()),
        7 => return Some(ChannelLayout::seven_one_wide()),
        11 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::BackCenter,
            Channel::Lfe,
        ],
        12 => return Some(ChannelLayout::seven_one()),
        _ => return None,
    };
    ChannelLayout::new(channels).ok()
}

fn parse_stts(payload: &[u8]) -> Result<Vec<(u32, u32)>, CodecError> {
    if payload.len() < 8 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stts",
        });
    }
    let count = u32::from_be_bytes(payload[4..8].try_into().unwrap_or([0; 4]));
    let mut runs = Vec::new();
    let mut at = 8usize;
    for _ in 0..count {
        if at + 8 > payload.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "stts",
            });
        }
        let samples = u32::from_be_bytes(payload[at..at + 4].try_into().unwrap_or([0; 4]));
        let duration = u32::from_be_bytes(payload[at + 4..at + 8].try_into().unwrap_or([0; 4]));
        runs.push((samples, duration));
        at += 8;
    }
    Ok(runs)
}

fn parse_stsz(payload: &[u8]) -> Result<Vec<u32>, CodecError> {
    if payload.len() < 12 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stsz",
        });
    }
    let uniform = u32::from_be_bytes(payload[4..8].try_into().unwrap_or([0; 4]));
    let count = u32::from_be_bytes(payload[8..12].try_into().unwrap_or([0; 4]));
    let count = usize::try_from(count).unwrap_or(0);
    if count > MAX_SAMPLES {
        return Err(CodecError::Limit { what: "samples" });
    }
    if uniform != 0 {
        return Ok(vec![uniform; count]);
    }
    if payload.len() < 12 + count * 4 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stsz",
        });
    }
    let mut sizes = Vec::with_capacity(count);
    for index in 0..count {
        let at = 12 + index * 4;
        sizes.push(u32::from_be_bytes(
            payload[at..at + 4].try_into().unwrap_or([0; 4]),
        ));
    }
    Ok(sizes)
}

fn parse_stsc(payload: &[u8]) -> Result<Vec<(u32, u32)>, CodecError> {
    if payload.len() < 8 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stsc",
        });
    }
    let count = u32::from_be_bytes(payload[4..8].try_into().unwrap_or([0; 4]));
    let mut runs = Vec::new();
    let mut at = 8usize;
    for _ in 0..count {
        if at + 12 > payload.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "stsc",
            });
        }
        let first = u32::from_be_bytes(payload[at..at + 4].try_into().unwrap_or([0; 4]));
        let per = u32::from_be_bytes(payload[at + 4..at + 8].try_into().unwrap_or([0; 4]));
        runs.push((first, per));
        at += 12;
    }
    Ok(runs)
}

fn parse_offsets(payload: &[u8], wide: bool) -> Result<Vec<u64>, CodecError> {
    if payload.len() < 8 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stco",
        });
    }
    let count = usize::try_from(u32::from_be_bytes(
        payload[4..8].try_into().unwrap_or([0; 4]),
    ))
    .unwrap_or(0);
    let width = if wide { 8 } else { 4 };
    if payload.len() < 8 + count * width {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "stco",
        });
    }
    let mut offsets = Vec::with_capacity(count);
    for index in 0..count {
        let at = 8 + index * width;
        let offset = if wide {
            u64::from_be_bytes(payload[at..at + 8].try_into().unwrap_or([0; 8]))
        } else {
            u64::from(u32::from_be_bytes(
                payload[at..at + 4].try_into().unwrap_or([0; 4]),
            ))
        };
        offsets.push(offset);
    }
    Ok(offsets)
}

fn looks_like_adts(sample: &[u8]) -> bool {
    if sample.len() < 7 || sample[0] != 0xFF || sample[1] & 0xF6 != 0xF0 {
        return false;
    }
    let length = (usize::from(sample[3] & 3) << 11)
        | (usize::from(sample[4]) << 3)
        | (usize::from(sample[5]) >> 5);
    length == sample.len()
}

#[cfg(test)]
mod tests {
    use super::{Mp4Demuxer, Mp4Muxer};
    use crate::{Demuxer, Muxer};
    use cutlass_codec::CodecId;
    use cutlass_codec::access::{ChannelLayout, CodecError, CodecParams, Packet, SideData};

    fn params() -> CodecParams {
        let mut params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
        params.extra = vec![0x11, 0x88];
        params
    }

    fn open(bytes: &[u8]) -> Mp4Demuxer {
        let mut demux = Mp4Demuxer::new();
        demux.push_bytes(bytes).unwrap();
        demux.finish_bytes().unwrap();
        demux
    }

    #[test]
    fn priming_edit_round_trips_raw_samples() {
        let mut mux = Mp4Muxer::new();
        mux.add_stream(params()).unwrap();
        let mut first = Packet::audio(vec![1, 2, 3, 4], 0, 1024, 48_000);
        first.side.push(SideData::Gapless {
            encoder_delay: 1024,
            trailing_padding: 200,
            sample_rate: 48_000,
        });
        let mut second = Packet::audio(vec![5, 6, 7], 1024, 1024, 48_000);
        second.side.push(SideData::Gapless {
            encoder_delay: 1024,
            trailing_padding: 200,
            sample_rate: 48_000,
        });
        mux.write_packet(&first).unwrap();
        mux.write_packet(&second).unwrap();
        let file = mux.finish().unwrap();
        assert!(file.windows(4).any(|window| window == b"mdat"));
        let mut demux = open(&file);
        let a = demux.next_packet().unwrap().unwrap();
        let b = demux.next_packet().unwrap().unwrap();
        assert_eq!(a.data, vec![1, 2, 3, 4]);
        assert_eq!(b.data, vec![5, 6, 7]);
        assert!(a.data[0] != 0xFF);
        assert_eq!(demux.streams()[0].params.extra, vec![0x11, 0x88]);
        assert_eq!(demux.streams()[0].timescale, 48_000);
        match a
            .side
            .iter()
            .find(|side| matches!(side, SideData::Gapless { .. }))
        {
            Some(SideData::Gapless {
                encoder_delay,
                trailing_padding,
                sample_rate,
            }) => {
                assert_eq!(*encoder_delay, 1024);
                assert_eq!(*trailing_padding, 200);
                assert_eq!(*sample_rate, 48_000);
            }
            _ => panic!("missing edit"),
        }
        assert!(demux.next_packet().unwrap().is_none());
    }

    #[test]
    fn an_adts_sample_is_rejected() {
        let mut mux = Mp4Muxer::new();
        mux.add_stream(params()).unwrap();
        let adts = vec![0xFF, 0xF1, 0x4C, 0x40, 0x00, 0xFF, 0xFC];
        let err = mux
            .write_packet(&Packet::audio(adts, 0, 1024, 48_000))
            .unwrap_err();
        assert_eq!(
            err,
            CodecError::Malformed {
                at_bit: 0,
                what: "adts"
            }
        );
    }

    #[test]
    fn a_video_track_is_rejected() {
        let mut file = Vec::new();
        file.extend(box_bytes(b"ftyp", b"M4A "));
        let hdlr = full_bytes(b"hdlr", &{
            let mut body = vec![0u8; 4];
            body.extend_from_slice(b"vide");
            body.extend_from_slice(&[0u8; 12]);
            body
        });
        let mdia = box_bytes(b"mdia", &hdlr);
        let trak = box_bytes(b"trak", &mdia);
        let mvhd = full_bytes(b"mvhd", &{
            let mut body = vec![0u8; 8];
            body.extend_from_slice(&48_000u32.to_be_bytes());
            body
        });
        let moov = box_bytes(b"moov", &[mvhd, trak].concat());
        file.extend(moov);
        file.extend(box_bytes(b"mdat", &[1, 2, 3, 4]));
        let mut demux = Mp4Demuxer::new();
        let err = demux.push_bytes(&file).unwrap_err();
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "video track"
            }
        );
    }

    fn box_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        super::box_of(kind, payload).unwrap()
    }

    fn full_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        super::full_box(kind, 0, payload).unwrap()
    }

    #[test]
    fn ffmpeg_m4a_round_trips_through_our_boxes() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            if std::env::var_os("AAC_REQUIRE_ORACLES").is_some() {
                panic!("ffmpeg is required");
            }
            eprintln!("skipped ffmpeg m4a oracle: ffmpeg was not found");
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-m4a-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = dir.join("in.m4a");
        let remuxed = dir.join("out.m4a");
        let left = dir.join("in.pcm");
        let right = dir.join("out.pcm");
        let status = std::process::Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=0.5",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
            ])
            .arg(&original)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let mut demux = Mp4Demuxer::new();
        demux
            .push_bytes(&std::fs::read(&original).unwrap())
            .unwrap();
        demux.finish_bytes().unwrap();
        assert!(!demux.streams()[0].params.extra.is_empty());
        let mut mux = Mp4Muxer::new();
        mux.add_stream(demux.streams()[0].params.clone()).unwrap();
        let mut count = 0;
        while let Some(packet) = demux.next_packet().unwrap() {
            assert!(!packet.data.is_empty());
            assert_ne!(packet.data[0], 0xFF);
            mux.write_packet(&packet).unwrap();
            count += 1;
        }
        assert!(count > 2, "{count}");
        std::fs::write(&remuxed, mux.finish().unwrap()).unwrap();
        decode(&ffmpeg, &original, &left);
        decode(&ffmpeg, &remuxed, &right);
        assert_eq!(
            std::fs::read(&left).unwrap(),
            std::fs::read(&right).unwrap()
        );
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        for candidate in [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ] {
            let path = std::path::PathBuf::from(candidate);
            if std::process::Command::new(&path)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(path);
            }
        }
        None
    }

    fn decode(ffmpeg: &std::path::Path, input: &std::path::Path, output: &std::path::Path) {
        let status = std::process::Command::new(ffmpeg)
            .args(["-y", "-i"])
            .arg(input)
            .args(["-f", "s16le", "-ac", "1", "-ar", "48000"])
            .arg(output)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
    }

    #[test]
    fn random_bytes_fail_closed() {
        let mut state = 0xA5A5_1357u32;
        for _ in 0..128 {
            let mut demux = Mp4Demuxer::new();
            let len = usize::try_from(next_bits(&mut state) % 120).unwrap_or(1) + 1;
            let mut data = vec![0u8; len];
            for byte in &mut data {
                *byte = u8::try_from(next_bits(&mut state) & 0xFF).unwrap_or(0);
            }
            let _ = demux.push_bytes(&data);
            let _ = demux.finish_bytes();
            while demux.next_packet().ok().flatten().is_some() {}
        }
    }

    fn next_bits(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }
}
