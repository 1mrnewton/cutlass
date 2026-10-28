//! LOAS `AudioSyncStream` around one LATM program and one layer.
//!
//! The multiplex follows ISO/IEC 14496-3 subclause 1.7 and the subset
//! ffmpeg writes: `audioMuxVersion` 0, one subframe, `frameLengthType` 0,
//! no other-data, and no LATM CRC. A second program or layer is rejected.
//! The first frame carries `StreamMuxConfig`. Later frames may reuse it.
//! `AudioSpecificConfig` has to already be in [`CodecParams::extra`]; this
//! crate does not invent one.

use std::collections::VecDeque;

use cutlass_container::{
    CodecError, CodecId, CodecParams, ContainerId, Demuxer, Muxer, Packet, StreamInfo,
};

const MAX_FRAME: usize = 0x1FFF;
const MAX_BUFFER: usize = 1 << 20;
const MAX_RESYNC: usize = 1 << 20;

pub struct LatmDemuxer {
    buf: Vec<u8>,
    ready: VecDeque<Packet>,
    streams: Vec<StreamInfo>,
    asc: Option<ParsedAsc>,
    ended: bool,
    skipped: usize,
    samples: u64,
}

pub struct LatmMuxer {
    params: Option<CodecParams>,
    bytes: Vec<u8>,
    sent_config: bool,
    done: bool,
    samples: u64,
}

impl LatmDemuxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            ready: VecDeque::new(),
            streams: Vec::new(),
            asc: None,
            ended: false,
            skipped: 0,
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
        self.pull()
    }

    pub fn finish_bytes(&mut self) -> Result<(), CodecError> {
        self.ended = true;
        self.pull()?;
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(CodecError::Truncated {
                at_bit: 0,
                what: "loas frame",
            })
        }
    }

    fn pull(&mut self) -> Result<(), CodecError> {
        loop {
            if self.buf.len() < 3 {
                return Ok(());
            }
            if self.buf[0] != 0x56 || self.buf[1] & 0xE0 != 0xE0 {
                self.buf.remove(0);
                self.skipped += 1;
                if self.skipped > MAX_RESYNC {
                    return Err(CodecError::Limit { what: "resync" });
                }
                continue;
            }
            let length = (usize::from(self.buf[1] & 0x1F) << 8) | usize::from(self.buf[2]);
            if length == 0 || length > MAX_FRAME {
                self.buf.remove(0);
                self.skipped += 1;
                continue;
            }
            if self.buf.len() < 3 + length {
                return Ok(());
            }
            let element = self.buf[3..3 + length].to_vec();
            self.buf.drain(..3 + length);
            self.skipped = 0;
            self.take_element(&element)?;
        }
    }

    fn take_element(&mut self, element: &[u8]) -> Result<(), CodecError> {
        let mut bits = Bits::new(element);
        let same = bits.read(1)? == 1;
        if !same {
            let asc = read_mux_config(&mut bits)?;
            if self
                .asc
                .as_ref()
                .is_some_and(|prev| prev.bytes != asc.bytes)
            {
                return Err(CodecError::Unsupported {
                    what: "configuration change",
                });
            }
            if self.asc.is_none() {
                let mut params = CodecParams::audio(CodecId::Aac, asc.rate, asc.layout.clone());
                params.extra = asc.bytes.clone();
                self.streams.push(StreamInfo {
                    index: 0,
                    timescale: asc.rate,
                    params,
                });
            }
            self.asc = Some(asc);
        } else if self.asc.is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "stream mux config",
            });
        }
        let Some(asc) = self.asc.as_ref() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "stream mux config",
            });
        };
        let frame = i64::from(asc.frame);
        let rate = asc.rate;
        let advance = u64::from(asc.frame);
        let size = payload_length(&mut bits)?;
        let payload = bits.read_bytes(size)?;
        let pts = i64::try_from(self.samples).unwrap_or(i64::MAX);
        self.samples = self.samples.saturating_add(advance);
        self.ready
            .push_back(Packet::audio(payload, pts, frame, rate));
        Ok(())
    }
}

impl Default for LatmDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for LatmDemuxer {
    fn id(&self) -> ContainerId {
        ContainerId::Latm
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Option<Packet>, CodecError> {
        if let Some(packet) = self.ready.pop_front() {
            return Ok(Some(packet));
        }
        self.pull()?;
        if let Some(packet) = self.ready.pop_front() {
            return Ok(Some(packet));
        }
        if self.ended {
            if self.buf.is_empty() {
                Ok(None)
            } else {
                Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "loas frame",
                })
            }
        } else {
            Err(CodecError::NeedMoreData)
        }
    }
}

impl LatmMuxer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: None,
            bytes: Vec::new(),
            sent_config: false,
            done: false,
            samples: 0,
        }
    }
}

impl Default for LatmMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for LatmMuxer {
    fn id(&self) -> ContainerId {
        ContainerId::Latm
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
        if params.extra.is_empty() || params.extra.len() > 64 {
            return Err(CodecError::InvalidApi {
                what: "audio specific config",
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
        let Some(params) = &self.params else {
            return Err(CodecError::InvalidApi { what: "stream" });
        };
        if packet.data.is_empty() || packet.data.len() > MAX_FRAME {
            return Err(CodecError::Limit { what: "loas frame" });
        }
        let mut body = BitOut::new();
        if self.sent_config {
            body.write(1, 1);
        } else {
            body.write(0, 1);
            write_mux_config(&mut body, &params.extra);
            self.sent_config = true;
        }
        write_length(&mut body, packet.data.len());
        for byte in &packet.data {
            body.write(u32::from(*byte), 8);
        }
        let element = body.finish();
        if element.len() > MAX_FRAME {
            return Err(CodecError::Limit { what: "loas frame" });
        }
        let mut header = [0x56, 0xE0, 0x00];
        header[1] |= u8::try_from((element.len() >> 8) & 0x1F).unwrap_or(0);
        header[2] |= u8::try_from(element.len() & 0xFF).unwrap_or(0);
        self.bytes.extend_from_slice(&header);
        self.bytes.extend_from_slice(&element);
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

fn write_mux_config(out: &mut BitOut, asc: &[u8]) {
    out.write(0, 1);
    out.write(1, 1);
    out.write(0, 6);
    out.write(0, 4);
    out.write(0, 3);
    for byte in asc {
        out.write(u32::from(*byte), 8);
    }
    out.write(0, 3);
    out.write(0xFF, 8);
    out.write(0, 1);
    out.write(0, 1);
}

fn write_length(out: &mut BitOut, mut size: usize) {
    while size >= 255 {
        out.write(255, 8);
        size -= 255;
    }
    let tail = u32::try_from(size).unwrap_or(0);
    out.write(tail, 8);
}

struct ParsedAsc {
    bytes: Vec<u8>,
    rate: u32,
    layout: cutlass_container::ChannelLayout,
    frame: u32,
}

fn read_mux_config(bits: &mut Bits<'_>) -> Result<ParsedAsc, CodecError> {
    if bits.read(1)? != 0 {
        return Err(CodecError::Unsupported {
            what: "audio mux version",
        });
    }
    if bits.read(1)? != 1 {
        return Err(CodecError::Unsupported {
            what: "time framing",
        });
    }
    if bits.read(6)? != 0 {
        return Err(CodecError::Unsupported { what: "subframes" });
    }
    if bits.read(4)? != 0 || bits.read(3)? != 0 {
        return Err(CodecError::Unsupported { what: "programs" });
    }
    let asc = read_asc(bits)?;
    if bits.read(3)? != 0 {
        return Err(CodecError::Unsupported {
            what: "frame length type",
        });
    }
    let _fullness = bits.read(8)?;
    if bits.read(1)? != 0 {
        return Err(CodecError::Unsupported { what: "other data" });
    }
    if bits.read(1)? != 0 {
        return Err(CodecError::Unsupported { what: "latm crc" });
    }
    Ok(asc)
}

fn read_asc(bits: &mut Bits<'_>) -> Result<ParsedAsc, CodecError> {
    let start = bits.pos;
    let aot = read_aot(bits)?;
    let rate = read_rate(bits)?;
    let channels = bits.read(4)?;
    if aot == 5 || aot == 29 {
        let _ext = read_rate(bits)?;
        let core = read_aot(bits)?;
        if core != 2 {
            return Err(CodecError::Unsupported {
                what: "audio object type",
            });
        }
    } else if aot != 2 {
        return Err(CodecError::Unsupported {
            what: "audio object type",
        });
    }
    let layout = layout_for(channels)?;
    let frame = if bits.read(1)? == 0 { 1024 } else { 960 };
    if bits.read(1)? != 0 {
        return Err(CodecError::Unsupported { what: "core coder" });
    }
    if bits.read(1)? != 0 {
        return Err(CodecError::Unsupported { what: "extension" });
    }
    Ok(ParsedAsc {
        bytes: bits.bytes_from(start)?,
        rate,
        layout,
        frame,
    })
}

fn layout_for(config: u32) -> Result<cutlass_container::ChannelLayout, CodecError> {
    use cutlass_container::{Channel, ChannelLayout};
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
        11 => vec![
            Channel::FrontCenter,
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::BackCenter,
            Channel::Lfe,
        ],
        12 => return Ok(ChannelLayout::seven_one()),
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

fn read_aot(bits: &mut Bits<'_>) -> Result<u32, CodecError> {
    let aot = bits.read(5)?;
    if aot == 31 {
        Ok(32 + bits.read(6)?)
    } else {
        Ok(aot)
    }
}

fn read_rate(bits: &mut Bits<'_>) -> Result<u32, CodecError> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    let index = bits.read(4)?;
    if index == 0xF {
        return bits.read(24);
    }
    RATES
        .get(index as usize)
        .copied()
        .ok_or(CodecError::Malformed {
            at_bit: bits.pos as u64,
            what: "sample rate",
        })
}

fn payload_length(bits: &mut Bits<'_>) -> Result<usize, CodecError> {
    let mut total = 0usize;
    loop {
        let chunk = usize::try_from(bits.read(8)?).unwrap_or(0);
        total = total.saturating_add(chunk);
        if total > MAX_FRAME {
            return Err(CodecError::Limit { what: "loas frame" });
        }
        if chunk != 255 {
            return Ok(total);
        }
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read(&mut self, n: u32) -> Result<u32, CodecError> {
        if n > 24 || self.left() < n as usize {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what: "latm",
            });
        }
        let mut value = 0u32;
        for _ in 0..n {
            let byte = self.data[self.pos / 8];
            let shift = 7 - (self.pos % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.pos += 1;
        }
        Ok(value)
    }

    fn read_bytes(&mut self, count: usize) -> Result<Vec<u8>, CodecError> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(u8::try_from(self.read(8)?).unwrap_or(0));
        }
        Ok(out)
    }

    fn bytes_from(&self, start: usize) -> Result<Vec<u8>, CodecError> {
        if self.pos < start || !(self.pos - start).is_multiple_of(8) {
            return Err(CodecError::Malformed {
                at_bit: self.pos as u64,
                what: "audio specific config",
            });
        }
        let mut out = Vec::new();
        let mut at = start;
        while at < self.pos {
            let mut byte = 0u8;
            for shift in (0..8).rev() {
                let src = self.data[at / 8];
                let bit = (src >> (7 - (at % 8))) & 1;
                byte |= bit << shift;
                at += 1;
            }
            out.push(byte);
        }
        Ok(out)
    }

    fn left(&self) -> usize {
        self.data.len().saturating_mul(8).saturating_sub(self.pos)
    }
}

struct BitOut {
    data: Vec<u8>,
    acc: u8,
    bits: u8,
}

impl BitOut {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            bits: 0,
        }
    }

    fn write(&mut self, value: u32, n: u32) {
        for shift in (0..n).rev() {
            let bit = ((value >> shift) & 1) as u8;
            self.acc = (self.acc << 1) | bit;
            self.bits += 1;
            if self.bits == 8 {
                self.data.push(self.acc);
                self.acc = 0;
                self.bits = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.acc <<= 8 - self.bits;
            self.data.push(self.acc);
        }
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::{LatmDemuxer, LatmMuxer};
    use cutlass_container::{CodecError, CodecId, CodecParams, Demuxer, Muxer, Packet};

    fn params() -> CodecParams {
        let mut params = CodecParams::audio(
            CodecId::Aac,
            48_000,
            cutlass_container::ChannelLayout::mono(),
        );
        params.extra = vec![0x11, 0x88];
        params
    }

    #[test]
    fn one_program_round_trips_and_reuses_the_config() {
        let mut mux = LatmMuxer::new();
        mux.add_stream(params()).unwrap();
        mux.write_packet(&Packet::audio(vec![1, 2, 3, 4], 0, 1024, 48_000))
            .unwrap();
        mux.write_packet(&Packet::audio(vec![5, 6], 1024, 1024, 48_000))
            .unwrap();
        let bytes = mux.finish().unwrap();
        let mut demux = LatmDemuxer::new();
        demux.push_bytes(&bytes).unwrap();
        demux.finish_bytes().unwrap();
        let first = demux.next_packet().unwrap().unwrap();
        let second = demux.next_packet().unwrap().unwrap();
        assert_eq!(first.data, vec![1, 2, 3, 4]);
        assert_eq!(first.timescale, 48_000);
        assert_eq!(first.duration, 1024);
        assert_eq!(second.data, vec![5, 6]);
        assert_eq!(second.pts, 1024);
        assert_eq!(demux.streams()[0].params.extra, vec![0x11, 0x88]);
        assert!(demux.next_packet().unwrap().is_none());
    }

    #[test]
    fn ffmpeg_decodes_a_latm_remux_of_its_own_aac() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            if std::env::var_os("AAC_REQUIRE_ORACLES").is_some() {
                panic!("ffmpeg is required");
            }
            eprintln!("skipped ffmpeg latm oracle: ffmpeg was not found");
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-latm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let adts = dir.join("in.aac");
        let loas = dir.join("out.loas");
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
            .arg(&adts)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let mut adts_demux = crate::AdtsDemuxer::new();
        adts_demux
            .push_bytes(&std::fs::read(&adts).unwrap())
            .unwrap();
        adts_demux.finish_bytes().unwrap();
        let mut mux = LatmMuxer::new();
        mux.add_stream(params()).unwrap();
        while let Some(packet) = adts_demux.next_packet().unwrap() {
            mux.write_packet(&packet).unwrap();
        }
        std::fs::write(&loas, mux.finish().unwrap()).unwrap();
        decode(&ffmpeg, &adts, &left, &[]);
        decode(&ffmpeg, &loas, &right, &["-f", "loas"]);
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

    fn decode(
        ffmpeg: &std::path::Path,
        input: &std::path::Path,
        output: &std::path::Path,
        before: &[&str],
    ) {
        let mut command = std::process::Command::new(ffmpeg);
        command.arg("-y");
        command.args(before);
        command.arg("-i").arg(input);
        command.args(["-f", "s16le", "-ac", "1", "-ar", "48000"]);
        let status = command.arg(output).output().unwrap();
        assert!(status.status.success(), "{status:?}");
    }

    #[test]
    fn a_second_program_is_rejected() {
        let mut bits = super::BitOut::new();
        bits.write(0, 1);
        bits.write(0, 1);
        bits.write(1, 1);
        bits.write(0, 6);
        bits.write(1, 4);
        let element = bits.finish();
        let mut frame = vec![0x56, 0xE0, 0x00];
        frame[2] = u8::try_from(element.len()).unwrap_or(0);
        frame.extend(element);
        let mut demux = LatmDemuxer::new();
        let err = demux.push_bytes(&frame).unwrap_err();
        assert_eq!(err, CodecError::Unsupported { what: "programs" });
    }
}
