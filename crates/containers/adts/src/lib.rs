//! ADTS framing for one AAC-LC stream.
//!
//! A header can carry profile, an indexed sample rate, and channel
//! configuration 1 through 7. It cannot carry a program-config element,
//! an explicit sample rate, or spectral-band-replication signaling: those
//! stay in the raw access unit, and a caller that needs them in the header
//! is rejected. `protection_absent = 0` stores the ISO/IEC 11172-3 CRC.
//! One raw block leaves the element CRC for the codec. More than one raw
//! block is accepted when the header carries each later block's byte
//! position; without those positions the frame is rejected.
//!
//! Bytes that are not yet a whole frame stay buffered. `finish` of a trailing
//! partial frame is truncation, not another packet. A decoded packet's `data`
//! is the raw access unit, without the ADTS header.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use cutlass_container::{
    Channel, ChannelLayout, CodecError, CodecId, CodecParams, ContainerId, Demuxer, Muxer, Packet,
    SideData, StreamInfo,
};

const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];
const MAX_FRAME: usize = 8191;
const MAX_BUFFER: usize = 1 << 20;
const MAX_RESYNC: usize = 1 << 20;

struct Header {
    mpeg2: bool,
    profile: u8,
    rate_index: u8,
    channels: u8,
    frame_len: usize,
    blocks: u8,
    protection_absent: bool,
    header_len: usize,
}

pub struct AdtsDemuxer {
    buf: Vec<u8>,
    ready: VecDeque<Packet>,
    streams: Vec<StreamInfo>,
    ended: bool,
    skipped: usize,
    sample_rate: Option<u32>,
    layout: Option<ChannelLayout>,
    mpeg2: Option<bool>,
    samples: u64,
}

pub struct AdtsMuxer {
    params: Option<CodecParams>,
    mpeg2: bool,
    bytes: Vec<u8>,
    done: bool,
    samples: u64,
}

impl AdtsDemuxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            ready: VecDeque::new(),
            streams: Vec::new(),
            ended: false,
            skipped: 0,
            sample_rate: None,
            layout: None,
            mpeg2: None,
            samples: 0,
        }
    }

    pub fn push_bytes(&mut self, data: &[u8]) -> Result<(), CodecError> {
        if self.ended {
            return Err(CodecError::InvalidApi {
                what: "push after finish",
            });
        }
        if self.buf.len().saturating_add(data.len()) > MAX_BUFFER {
            return Err(CodecError::Limit {
                what: "buffered bytes",
            });
        }
        self.buf.extend_from_slice(data);
        self.pull_ready()
    }

    pub fn finish_bytes(&mut self) -> Result<(), CodecError> {
        self.ended = true;
        self.pull_ready()?;
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(CodecError::Truncated {
                at_bit: 0,
                what: "adts frame",
            })
        }
    }

    fn pull_ready(&mut self) -> Result<(), CodecError> {
        loop {
            if self.buf.len() < 7 {
                return Ok(());
            }
            let header = match sniff(&self.buf) {
                Sniff::NeedMore => return Ok(()),
                Sniff::Skip => {
                    self.buf.remove(0);
                    self.skipped += 1;
                    if self.skipped > MAX_RESYNC {
                        return Err(CodecError::Limit { what: "resync" });
                    }
                    continue;
                }
                Sniff::Header(header) => header,
            };
            if self.buf.len() < header.frame_len {
                return Ok(());
            }
            self.accept(header)?;
        }
    }

    fn accept(&mut self, header: Header) -> Result<(), CodecError> {
        if header.blocks != 0 && header.protection_absent {
            self.buf.drain(..header.frame_len);
            return Err(CodecError::Unsupported {
                what: "unpositioned raw data blocks",
            });
        }
        if header.profile != 1 {
            self.buf.drain(..header.frame_len);
            return Err(named_profile(header.profile));
        }
        let rate = rate_of(header.rate_index, header.mpeg2)?;
        let layout = layout_of(header.channels)?;
        if self.conflicts(rate, &layout, header.mpeg2) {
            self.buf.drain(..header.frame_len);
            return Err(CodecError::Unsupported {
                what: "configuration change",
            });
        }
        self.sample_rate = Some(rate);
        self.mpeg2 = Some(header.mpeg2);
        if self.streams.is_empty() {
            self.streams.push(StreamInfo {
                index: 0,
                timescale: rate,
                params: CodecParams::audio(CodecId::Aac, rate, layout.clone()),
            });
        }
        self.layout = Some(layout);
        let frame: Vec<u8> = self.buf[..header.frame_len].to_vec();
        self.buf.drain(..header.frame_len);
        self.skipped = 0;
        self.emit_blocks(&frame, &header, rate)?;
        Ok(())
    }

    fn emit_blocks(&mut self, frame: &[u8], header: &Header, rate: u32) -> Result<(), CodecError> {
        if header.protection_absent {
            self.push_unit(&frame[header.header_len..], None, rate);
            return Ok(());
        }
        if header.blocks == 0 {
            let crc = u16::from_be_bytes([frame[7], frame[8]]);
            self.push_unit(
                &frame[header.header_len..],
                Some(SideData::Protection {
                    crc,
                    header: Some(frame[..7].to_vec()),
                }),
                rate,
            );
            return Ok(());
        }
        let count = usize::from(header.blocks);
        let mut positions = vec![0usize];
        for index in 0..count {
            let at = 7 + index * 2;
            positions.push(usize::from(u16::from_be_bytes([frame[at], frame[at + 1]])));
        }
        let crc_at = 7 + count * 2;
        let found = u16::from_be_bytes([frame[crc_at], frame[crc_at + 1]]);
        if mpeg_crc(&frame[..crc_at]) != found {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "crc",
            });
        }
        let base = header.header_len;
        let end = frame.len() - base;
        let blocks = usize::from(header.blocks) + 1;
        for index in 0..blocks {
            let start = positions[index];
            let next = if index + 1 < blocks {
                positions[index + 1]
            } else {
                end
            };
            if next < start.saturating_add(2) || next > end {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "raw data block position",
                });
            }
            let crc_pos = base + next - 2;
            let crc = u16::from_be_bytes([frame[crc_pos], frame[crc_pos + 1]]);
            self.push_unit(
                &frame[base + start..crc_pos],
                Some(SideData::Protection { crc, header: None }),
                rate,
            );
        }
        Ok(())
    }

    fn push_unit(&mut self, payload: &[u8], side: Option<SideData>, rate: u32) {
        let pts = i64::try_from(self.samples).unwrap_or(i64::MAX);
        self.samples = self.samples.saturating_add(1024);
        let mut packet = Packet::audio(payload.to_vec(), pts, 1024, rate);
        if let Some(side) = side {
            packet.side.push(side);
        }
        self.ready.push_back(packet);
    }

    fn conflicts(&self, rate: u32, layout: &ChannelLayout, mpeg2: bool) -> bool {
        self.sample_rate.is_some_and(|previous| previous != rate)
            || self.layout.as_ref().is_some_and(|prev| prev != layout)
            || self.mpeg2.is_some_and(|previous| previous != mpeg2)
    }
}

impl Default for AdtsDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for AdtsDemuxer {
    fn id(&self) -> ContainerId {
        ContainerId::Adts
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Option<Packet>, CodecError> {
        if let Some(packet) = self.ready.pop_front() {
            return Ok(Some(packet));
        }
        self.pull_ready()?;
        if let Some(packet) = self.ready.pop_front() {
            return Ok(Some(packet));
        }
        if self.ended {
            if self.buf.is_empty() {
                Ok(None)
            } else {
                Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "adts frame",
                })
            }
        } else {
            Err(CodecError::NeedMoreData)
        }
    }
}

impl AdtsMuxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: None,
            mpeg2: false,
            bytes: Vec::new(),
            done: false,
            samples: 0,
        }
    }

    pub fn set_mpeg2(&mut self, mpeg2: bool) {
        self.mpeg2 = mpeg2;
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Writes up to four raw blocks in one ADTS frame.
    ///
    /// Each packet carries [`SideData::Protection`] for that block alone.
    /// The header CRC covers the fixed header, the variable header, and the
    /// byte positions, which is the check ISO/IEC 13818-7 assigns to
    /// `adts_header_error_check`.
    pub fn write_blocks(&mut self, packets: &[Packet]) -> Result<(), CodecError> {
        if self.done {
            return Err(CodecError::InvalidApi {
                what: "write after finish",
            });
        }
        if packets.is_empty() || packets.len() > 4 {
            return Err(CodecError::Limit {
                what: "raw data blocks",
            });
        }
        if packets.len() == 1 {
            return self.write_packet(&packets[0]);
        }
        let Some(params) = &self.params else {
            return Err(CodecError::InvalidApi { what: "stream" });
        };
        let layout = params
            .layout
            .clone()
            .ok_or(CodecError::InvalidApi { what: "layout" })?;
        let mut positions = Vec::new();
        let mut body = Vec::new();
        let mut cursor = 0usize;
        for (index, packet) in packets.iter().enumerate() {
            if packet.data.is_empty() {
                return Err(CodecError::InvalidApi {
                    what: "empty packet",
                });
            }
            if index > 0 {
                let pos = u16::try_from(cursor).map_err(|_| CodecError::Limit {
                    what: "raw data block position",
                })?;
                positions.extend(pos.to_be_bytes());
            }
            let crc = block_crc(packet)?;
            body.extend_from_slice(&packet.data);
            body.extend_from_slice(&crc.to_be_bytes());
            cursor = cursor.saturating_add(packet.data.len() + 2);
        }
        let blocks = u8::try_from(packets.len() - 1).unwrap_or(0);
        let header = encode_header(
            positions.len() + 2 + body.len(),
            index_of_rate(params.sample_rate, self.mpeg2)?,
            config_of(&layout)?,
            self.mpeg2,
            true,
            blocks,
        )?;
        let mut covered = header.to_vec();
        covered.extend_from_slice(&positions);
        let crc = mpeg_crc(&covered);
        self.bytes.extend_from_slice(&header);
        self.bytes.extend_from_slice(&positions);
        self.bytes.extend_from_slice(&crc.to_be_bytes());
        self.bytes.extend_from_slice(&body);
        let count = u64::try_from(packets.len()).unwrap_or(0);
        self.samples = self.samples.saturating_add(1024u64.saturating_mul(count));
        Ok(())
    }
}

impl Default for AdtsMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for AdtsMuxer {
    fn id(&self) -> ContainerId {
        ContainerId::Adts
    }

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, CodecError> {
        if self.params.is_some() {
            return Err(CodecError::Unsupported { what: "streams" });
        }
        if params.codec != CodecId::Aac {
            return Err(CodecError::Unsupported {
                what: "audio object type",
            });
        }
        let layout = params
            .layout
            .clone()
            .ok_or(CodecError::InvalidApi { what: "layout" })?;
        let _ = config_of(&layout)?;
        let _ = index_of_rate(params.sample_rate, self.mpeg2)?;
        self.params = Some(params);
        Ok(0)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), CodecError> {
        if self.done {
            return Err(CodecError::InvalidApi {
                what: "write after finish",
            });
        }
        let Some(params) = &self.params else {
            return Err(CodecError::InvalidApi { what: "stream" });
        };
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let layout = params
            .layout
            .clone()
            .ok_or(CodecError::InvalidApi { what: "layout" })?;
        let rate_index = index_of_rate(params.sample_rate, self.mpeg2)?;
        let channels = config_of(&layout)?;
        if let Some(crc) = protection_crc(packet) {
            let header = encode_header(
                packet.data.len() + 2,
                rate_index,
                channels,
                self.mpeg2,
                true,
                0,
            )?;
            self.bytes.extend_from_slice(&header);
            self.bytes.extend_from_slice(&crc.to_be_bytes());
            self.bytes.extend_from_slice(&packet.data);
        } else {
            let header = encode_header(
                packet.data.len(),
                rate_index,
                channels,
                self.mpeg2,
                false,
                0,
            )?;
            self.bytes.extend_from_slice(&header);
            self.bytes.extend_from_slice(&packet.data);
        }
        self.samples = self.samples.saturating_add(1024);
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
        if self.done {
            return Err(CodecError::InvalidApi {
                what: "second finish",
            });
        }
        self.done = true;
        Ok(self.bytes.clone())
    }
}

enum Sniff {
    NeedMore,
    Skip,
    Header(Header),
}

fn sniff(data: &[u8]) -> Sniff {
    if data.len() < 7 {
        return Sniff::NeedMore;
    }
    if data[0] != 0xFF || data[1] & 0xF0 != 0xF0 {
        return Sniff::Skip;
    }
    match parse_header(data) {
        Err(CodecError::NeedMoreData) => Sniff::NeedMore,
        Ok(header) if header.frame_len <= MAX_FRAME && header.frame_len >= header.header_len => {
            if data.len() < header.header_len {
                Sniff::NeedMore
            } else {
                Sniff::Header(header)
            }
        }
        _ => Sniff::Skip,
    }
}

fn parse_header(data: &[u8]) -> Result<Header, CodecError> {
    if data.len() < 7 {
        return Err(CodecError::NeedMoreData);
    }
    let layer = (data[1] >> 1) & 0b11;
    if layer != 0 {
        return Err(CodecError::Malformed {
            at_bit: 13,
            what: "adts layer",
        });
    }
    let protection_absent = data[1] & 1 == 1;
    let blocks = data[6] & 0b11;
    let header_len = if protection_absent {
        7
    } else {
        9 + 2 * usize::from(blocks)
    };
    if data.len() < header_len {
        return Err(CodecError::NeedMoreData);
    }
    let frame_len = (usize::from(data[3] & 0b11) << 11)
        | (usize::from(data[4]) << 3)
        | (usize::from(data[5]) >> 5);
    Ok(Header {
        mpeg2: data[1] & 0b1000 != 0,
        profile: data[2] >> 6,
        rate_index: (data[2] >> 2) & 0b1111,
        channels: ((data[2] & 0b11) << 1) | (data[3] >> 6),
        frame_len,
        blocks,
        protection_absent,
        header_len,
    })
}

fn encode_header(
    extra: usize,
    rate_index: u8,
    channels: u8,
    mpeg2: bool,
    protect: bool,
    blocks: u8,
) -> Result<[u8; 7], CodecError> {
    let frame_len = extra.saturating_add(7);
    if frame_len > MAX_FRAME || blocks > 3 {
        return Err(CodecError::Limit { what: "adts frame" });
    }
    let id = u8::from(mpeg2);
    let mut bytes = [0u8; 7];
    bytes[0] = 0xFF;
    bytes[1] = 0xF0 | (id << 3) | u8::from(!protect);
    bytes[2] = (1 << 6) | (rate_index << 2) | ((channels >> 2) & 0b11);
    bytes[3] = (channels << 6) | u8::try_from((frame_len >> 11) & 0b11).unwrap_or(0);
    bytes[4] = u8::try_from((frame_len >> 3) & 0xFF).unwrap_or(0);
    bytes[5] = u8::try_from((frame_len & 0b111) << 5).unwrap_or(0) | 0x1F;
    bytes[6] = 0xFC | blocks;
    Ok(bytes)
}

fn protection_crc(packet: &Packet) -> Option<u16> {
    packet.side.iter().find_map(|side| match side {
        SideData::Protection {
            crc,
            header: Some(_),
        } => Some(*crc),
        _ => None,
    })
}

fn block_crc(packet: &Packet) -> Result<u16, CodecError> {
    packet
        .side
        .iter()
        .find_map(|side| match side {
            SideData::Protection { crc, header: None } => Some(*crc),
            _ => None,
        })
        .ok_or(CodecError::InvalidApi { what: "crc" })
}

fn mpeg_crc(data: &[u8]) -> u16 {
    let mut reg = 0xffffu16;
    for byte in data {
        for shift in (0..8).rev() {
            let bit = u16::from((byte >> shift) & 1);
            let out = (reg >> 15) & 1;
            reg <<= 1;
            if out ^ bit != 0 {
                reg ^= 0x8005;
            }
        }
    }
    reg
}

fn rate_of(index: u8, mpeg2: bool) -> Result<u32, CodecError> {
    if mpeg2 && index > 11 {
        return Err(CodecError::Unsupported {
            what: "mpeg-2 sample rate",
        });
    }
    RATES
        .get(usize::from(index))
        .copied()
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "sample rate",
        })
}

fn index_of_rate(rate: u32, mpeg2: bool) -> Result<u8, CodecError> {
    let index = RATES
        .iter()
        .position(|&candidate| candidate == rate)
        .and_then(|index| u8::try_from(index).ok())
        .ok_or(CodecError::Unsupported {
            what: "sample rate",
        })?;
    if mpeg2 && index > 11 {
        return Err(CodecError::Unsupported {
            what: "mpeg-2 sample rate",
        });
    }
    Ok(index)
}

fn layout_of(config: u8) -> Result<ChannelLayout, CodecError> {
    let channels = match config {
        1 => return Ok(ChannelLayout::mono()),
        2 => return Ok(ChannelLayout::stereo()),
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
        6 => return Ok(ChannelLayout::five_one()),
        7 => return Ok(ChannelLayout::seven_one_wide()),
        0 => {
            return Err(CodecError::Unsupported {
                what: "program config",
            });
        }
        _ => {
            return Err(CodecError::Unsupported {
                what: "channel configuration",
            });
        }
    };
    ChannelLayout::new(channels).map_err(|_| CodecError::Unsupported {
        what: "channel configuration",
    })
}

fn config_of(layout: &ChannelLayout) -> Result<u8, CodecError> {
    for config in 1u8..=7 {
        if layout_of(config).is_ok_and(|candidate| candidate.channels() == layout.channels()) {
            return Ok(config);
        }
    }
    Err(CodecError::Unsupported {
        what: "channel configuration",
    })
}

fn named_profile(profile: u8) -> CodecError {
    let what = match profile {
        0 => "main",
        2 => "ssr",
        3 => "ltp",
        _ => "profile",
    };
    CodecError::Unsupported { what }
}

#[cfg(test)]
mod tests {
    use super::{AdtsDemuxer, AdtsMuxer};
    use cutlass_container::{
        ChannelLayout, CodecError, CodecId, CodecParams, Demuxer, Muxer, Packet, SideData,
    };

    fn mono() -> CodecParams {
        CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono())
    }

    fn packets_from(bytes: &[u8]) -> Vec<Packet> {
        let mut demux = AdtsDemuxer::new();
        demux.push_bytes(bytes).unwrap();
        demux.finish_bytes().unwrap();
        let mut out = Vec::new();
        while let Some(packet) = demux.next_packet().unwrap() {
            out.push(packet);
        }
        out
    }

    #[test]
    fn a_raw_unit_round_trips_through_adts() {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_packet(&Packet::audio(vec![1, 2, 3, 4], 0, 1024, 48_000))
            .unwrap();
        mux.write_packet(&Packet::audio(vec![5, 6], 1024, 1024, 48_000))
            .unwrap();
        let bytes = mux.finish().unwrap();
        assert!(mux.finish().is_err());
        let packets = packets_from(&bytes);
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data, vec![1, 2, 3, 4]);
        assert_eq!(packets[1].data, vec![5, 6]);
        assert_eq!(packets[0].timescale, 48_000);
        assert_eq!(packets[1].pts, 1024);
    }

    #[test]
    fn one_byte_at_a_time_matches_the_whole_buffer() {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_packet(&Packet::audio(vec![9, 8, 7, 6, 5], 0, 1024, 48_000))
            .unwrap();
        let bytes = mux.finish().unwrap();
        let whole = packets_from(&bytes);
        let mut demux = AdtsDemuxer::new();
        let mut split = Vec::new();
        for byte in &bytes {
            demux.push_bytes(&[*byte]).unwrap();
            while let Ok(Some(packet)) = demux.next_packet() {
                split.push(packet);
            }
        }
        demux.finish_bytes().unwrap();
        while let Ok(Some(packet)) = demux.next_packet() {
            split.push(packet);
        }
        assert_eq!(
            split.iter().map(|p| p.data.clone()).collect::<Vec<_>>(),
            whole.iter().map(|p| p.data.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_false_sync_before_the_frame_is_skipped() {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_packet(&Packet::audio(vec![1, 2, 3], 0, 1024, 48_000))
            .unwrap();
        let mut bytes = vec![0xFF, 0x00];
        bytes.extend(mux.finish().unwrap());
        let packets = packets_from(&bytes);
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn a_truncated_tail_is_not_a_packet() {
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_packet(&Packet::audio(vec![1, 2, 3, 4], 0, 1024, 48_000))
            .unwrap();
        let bytes = mux.finish().unwrap();
        let mut demux = AdtsDemuxer::new();
        demux.push_bytes(&bytes[..bytes.len() - 1]).unwrap();
        assert!(matches!(demux.next_packet(), Err(CodecError::NeedMoreData)));
        let err = demux.finish_bytes().unwrap_err();
        assert!(matches!(err, CodecError::Truncated { .. }));
    }

    #[test]
    fn a_protected_block_keeps_its_crc_and_payload() {
        let mut packet = Packet::audio(vec![1, 2, 3, 4], 0, 1024, 48_000);
        packet.side.push(SideData::Protection {
            crc: 0xABCD,
            header: Some(vec![0; 7]),
        });
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_packet(&packet).unwrap();
        let packets = packets_from(&mux.finish().unwrap());
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data, vec![1, 2, 3, 4]);
        assert!(matches!(
            packets[0].side.as_slice(),
            [SideData::Protection {
                crc: 0xABCD,
                header: Some(header),
            }] if header.len() == 7 && header[1] & 1 == 0
        ));
    }

    #[test]
    fn two_raw_blocks_round_trip_when_positions_are_present() {
        let mut first = Packet::audio(vec![1, 2, 3], 0, 1024, 48_000);
        let mut second = Packet::audio(vec![4, 5], 1024, 1024, 48_000);
        first.side.push(SideData::Protection {
            crc: 0x1111,
            header: None,
        });
        second.side.push(SideData::Protection {
            crc: 0x2222,
            header: None,
        });
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        mux.write_blocks(&[first, second]).unwrap();
        let packets = packets_from(&mux.finish().unwrap());
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data, vec![1, 2, 3]);
        assert_eq!(packets[1].data, vec![4, 5]);
        assert_eq!(packets[1].pts, 1024);
        let mut broken = mux.bytes().to_vec();
        broken[7] ^= 0x01;
        let mut demux = AdtsDemuxer::new();
        let err = demux.push_bytes(&broken).unwrap_err();
        assert_eq!(
            err,
            CodecError::Malformed {
                at_bit: 0,
                what: "crc"
            }
        );
    }

    #[test]
    fn unpositioned_multiple_blocks_are_rejected() {
        let blocks = super::encode_header(4, 3, 1, false, false, 1).unwrap();
        let mut framed = blocks.to_vec();
        framed.extend_from_slice(&[1, 2, 3, 4]);
        let mut demux = AdtsDemuxer::new();
        let err = demux.push_bytes(&framed).unwrap_err();
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "unpositioned raw data blocks"
            }
        );
    }

    #[test]
    fn seven_point_one_does_not_fit_the_header() {
        let mut mux = AdtsMuxer::new();
        let err = mux
            .add_stream(CodecParams::audio(
                CodecId::Aac,
                48_000,
                ChannelLayout::seven_one(),
            ))
            .unwrap_err();
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "channel configuration"
            }
        );
    }

    #[test]
    fn ffmpeg_adts_survives_our_demux_and_mux() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            if std::env::var_os("AAC_REQUIRE_ORACLES").is_some() {
                panic!("ffmpeg is required");
            }
            eprintln!("skipped ffmpeg adts oracle: ffmpeg was not found");
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-adts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = dir.join("in.aac");
        let remuxed = dir.join("out.aac");
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
                "-f",
                "adts",
            ])
            .arg(&original)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let bytes = std::fs::read(&original).unwrap();
        let packets = packets_from(&bytes);
        assert!(packets.len() > 2, "{}", packets.len());
        assert!(packets.iter().all(|packet| packet.timescale == 48_000));
        let mut mux = AdtsMuxer::new();
        mux.add_stream(mono()).unwrap();
        for packet in &packets {
            mux.write_packet(packet).unwrap();
        }
        std::fs::write(&remuxed, mux.finish().unwrap()).unwrap();
        decode_pcm(&ffmpeg, &original, &left);
        decode_pcm(&ffmpeg, &remuxed, &right);
        let original_pcm = std::fs::read(&left).unwrap();
        let remuxed_pcm = std::fs::read(&right).unwrap();
        assert_eq!(original_pcm, remuxed_pcm);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        let candidates = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ];
        for candidate in candidates {
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

    fn decode_pcm(ffmpeg: &std::path::Path, input: &std::path::Path, output: &std::path::Path) {
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
    fn mpeg2_rejects_the_7350_hz_index() {
        let mut mux = AdtsMuxer::new();
        mux.set_mpeg2(true);
        let err = mux
            .add_stream(CodecParams::audio(
                CodecId::Aac,
                7_350,
                ChannelLayout::mono(),
            ))
            .unwrap_err();
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "mpeg-2 sample rate"
            }
        );
    }
}
