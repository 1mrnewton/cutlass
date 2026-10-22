//! Vorbis and Opus packets rebuilt from Ogg pages.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, Packet, Timestamp};

use crate::Error;
use crate::page::{self, Page};

const CONTINUED: u8 = 0x01;
const BOS: u8 = 0x02;
const EOS: u8 = 0x04;
const FULL: u8 = 255;
const OPUS_HEAD: usize = 19;
const OPUS_HZ: u32 = 48_000;

/// A parsed Ogg file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::Inconsistent("the file has no pages"));
        }
        let mut streams = Vec::new();
        let mut at = 0;
        while at < data.len() {
            let (page, next) = page::read_page(data, at)?;
            push_page(&mut streams, &page)?;
            at = next;
        }
        let tracks = tracks_from(streams)?;
        if tracks.is_empty() {
            return Err(Error::Inconsistent("the file has no supported stream"));
        }
        Ok(Self { tracks })
    }

    /// Vorbis and Opus, in the order their streams start.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// One logical stream. The sample list is every packet after the identification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    serial: u32,
    timescale: NonZeroU32,
    config: AudioStreamConfig,
    samples: Vec<Packet>,
}

impl Track {
    /// The stream serial number.
    pub fn serial(&self) -> u32 {
        self.serial
    }

    /// Ticks per second. Vorbis uses its sample rate. Opus uses 48 kHz.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Codec, rate, and channel count. The config record is the identification packet.
    pub fn config(&self) -> &AudioStreamConfig {
        &self.config
    }

    /// Packets after the identification. `dts` is always none.
    pub fn samples(&self) -> &[Packet] {
        &self.samples
    }
}

struct Stream {
    serial: u32,
    next_seq: u32,
    partial: Vec<u8>,
    packets: Vec<Done>,
    pages: u32,
}

struct Done {
    data: Vec<u8>,
    granule: i64,
    page: u32,
}

struct Ident {
    rate: NonZeroU32,
    channels: u16,
    codec: AudioCodec,
}

impl Stream {
    fn new(serial: u32) -> Self {
        Self {
            serial,
            next_seq: 0,
            partial: Vec::new(),
            packets: Vec::new(),
            pages: 0,
        }
    }
}

fn push_page(streams: &mut Vec<Stream>, page: &Page<'_>) -> Result<(), Error> {
    let index = stream_index(streams, page)?;
    let stream = &mut streams[index];
    let page_id = stream.pages;
    stream.pages = stream.pages.checked_add(1).ok_or(Error::TooLarge)?;
    stream.next_seq = page.sequence.checked_add(1).ok_or(Error::TooLarge)?;
    let continued = has(page.flags, CONTINUED);
    if stream.partial.is_empty() {
        if continued {
            return Err(Error::Inconsistent(
                "a page continues a packet that has not started",
            ));
        }
    } else if !continued {
        return Err(Error::Inconsistent("a continued packet is not continued"));
    }
    take_segments(stream, page, page_id)?;
    if has(page.flags, EOS) && !stream.partial.is_empty() {
        return Err(Error::Inconsistent("the last packet is not finished"));
    }
    Ok(())
}

fn stream_index(streams: &mut Vec<Stream>, page: &Page<'_>) -> Result<usize, Error> {
    if let Some(index) = streams
        .iter()
        .position(|stream| stream.serial == page.serial)
    {
        if has(page.flags, BOS) {
            return Err(Error::Inconsistent("a stream starts twice"));
        }
        if page.sequence != streams[index].next_seq {
            return Err(Error::Inconsistent("the page sequence is broken"));
        }
        return Ok(index);
    }
    if !has(page.flags, BOS) {
        return Err(Error::Inconsistent("a stream has no start"));
    }
    if page.sequence != 0 {
        return Err(Error::Inconsistent("the page sequence is broken"));
    }
    streams.push(Stream::new(page.serial));
    Ok(streams.len() - 1)
}

fn take_segments(stream: &mut Stream, page: &Page<'_>, page_id: u32) -> Result<(), Error> {
    let mut cursor: usize = 0;
    for &size in page.table {
        let size = usize::from(size);
        let end = cursor.checked_add(size).ok_or(Error::TooLarge)?;
        if end > page.payload.len() {
            let at = page.payload_at.checked_add(cursor).ok_or(Error::TooLarge)?;
            return Err(page::truncated(at, size));
        }
        stream.partial.extend_from_slice(&page.payload[cursor..end]);
        cursor = end;
        if size < usize::from(FULL) {
            stream.packets.push(Done {
                data: std::mem::take(&mut stream.partial),
                granule: page.granule,
                page: page_id,
            });
        }
    }
    Ok(())
}

fn tracks_from(streams: Vec<Stream>) -> Result<Vec<Track>, Error> {
    let mut tracks = Vec::new();
    for stream in streams {
        if !stream.partial.is_empty() {
            return Err(Error::Inconsistent("the last packet is not finished"));
        }
        if let Some(track) = track_from(stream)? {
            tracks.push(track);
        }
    }
    Ok(tracks)
}

fn track_from(stream: Stream) -> Result<Option<Track>, Error> {
    let mut packets = stream.packets.into_iter();
    let Some(id_packet) = packets.next() else {
        return Ok(None);
    };
    let Some(id) = identify(&id_packet.data)? else {
        return Ok(None);
    };
    let rest = packets.collect();
    Ok(Some(Track {
        serial: stream.serial,
        timescale: id.rate,
        config: AudioStreamConfig {
            codec: id.codec,
            sample_rate: id.rate.get(),
            channels: id.channels,
            config_record: id_packet.data,
        },
        samples: time_packets(id.rate, rest)?,
    }))
}

fn identify(data: &[u8]) -> Result<Option<Ident>, Error> {
    if is_vorbis(data) {
        return Ok(Some(vorbis(data)?));
    }
    if is_opus(data) {
        return Ok(Some(opus(data)?));
    }
    Ok(None)
}

fn is_vorbis(data: &[u8]) -> bool {
    data.len() >= 7 && data[0] == 1 && &data[1..7] == b"vorbis"
}

fn is_opus(data: &[u8]) -> bool {
    data.len() >= 8 && &data[..8] == b"OpusHead"
}

fn vorbis(data: &[u8]) -> Result<Ident, Error> {
    if data.len() < 16 {
        return Err(Error::Inconsistent(
            "the identification header is too short",
        ));
    }
    if le_u32(&data[7..11]) != 0 {
        return Err(Error::Inconsistent(
            "the identification header is not supported",
        ));
    }
    let channels = data[11];
    if channels == 0 {
        return Err(Error::Inconsistent("the channel count is zero"));
    }
    Ok(Ident {
        rate: rate(le_u32(&data[12..16]))?,
        channels: u16::from(channels),
        codec: AudioCodec::Vorbis,
    })
}

fn opus(data: &[u8]) -> Result<Ident, Error> {
    if data.len() < OPUS_HEAD {
        return Err(Error::Inconsistent(
            "the identification header is too short",
        ));
    }
    if data[8] != 1 {
        return Err(Error::Inconsistent(
            "the identification header is not supported",
        ));
    }
    let channels = data[9];
    if channels == 0 {
        return Err(Error::Inconsistent("the channel count is zero"));
    }
    let family = data[18];
    if family != 0 && data.len() < OPUS_HEAD + 2 + usize::from(channels) {
        return Err(Error::Inconsistent(
            "the identification header is too short",
        ));
    }
    Ok(Ident {
        rate: rate(OPUS_HZ)?,
        channels: u16::from(channels),
        codec: AudioCodec::Opus,
    })
}

fn has(flags: u8, bit: u8) -> bool {
    (flags & bit) != 0
}

fn rate(value: u32) -> Result<NonZeroU32, Error> {
    NonZeroU32::new(value).ok_or(Error::Inconsistent("the sample rate is zero"))
}

fn time_packets(rate: NonZeroU32, mut packets: Vec<Done>) -> Result<Vec<Packet>, Error> {
    let mut samples = Vec::with_capacity(packets.len());
    let mut end = 0i64;
    let mut index = 0;
    while index < packets.len() {
        let page = packets[index].page;
        let granule = packets[index].granule;
        let mut next = index + 1;
        while next < packets.len() && packets[next].page == page {
            next += 1;
        }
        if granule < 0 {
            for packet in &mut packets[index..next] {
                samples.push(sample(rate, 0, None, std::mem::take(&mut packet.data)));
            }
        } else if granule < end {
            return Err(Error::Inconsistent("the granule moves backwards"));
        } else {
            let span = granule - end;
            let last = next - 1;
            for (offset, packet) in packets[index..next].iter_mut().enumerate() {
                let duration = if index + offset == last {
                    Some(span)
                } else {
                    None
                };
                samples.push(sample(
                    rate,
                    end,
                    duration,
                    std::mem::take(&mut packet.data),
                ));
            }
            end = granule;
        }
        index = next;
    }
    Ok(samples)
}

fn sample(rate: NonZeroU32, pts: i64, duration: Option<i64>, data: Vec<u8>) -> Packet {
    Packet {
        data,
        pts: Timestamp::new(pts, rate),
        dts: None,
        duration: duration.map(|value| Timestamp::new(value, rate)),
        keyframe: true,
    }
}

fn le_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0; 4];
    buf.copy_from_slice(bytes);
    u32::from_le_bytes(buf)
}
