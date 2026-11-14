//! `COMM` fields and the PCM layout bytes handed to the decoder.
//!
//! Linear, A-law and mu-law packets are one sample frame. The compressed
//! rows below are one codec block, still compressed. Anything else is
//! refused by name.

use cutlass_codec::{CodecId, StreamCodec, UnknownCodec};
use cutlass_container::{MuxError, NewStream, OpenError};

use crate::limits::MAX_CHANNELS;
use crate::rate::{self, RateError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Pcm {
        bytes_per_frame: u16,
        extra: [u8; 3],
    },
    Blocked {
        tag: [u8; 4],
        bytes_per_frame: u16,
        samples_per_packet: u16,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sound {
    channels: u16,
    rate: u32,
    sample_frames: u32,
    layout: Layout,
}

impl Sound {
    pub(crate) const fn channels(self) -> u16 {
        self.channels
    }

    pub(crate) const fn rate(self) -> u32 {
        self.rate
    }

    pub(crate) const fn sample_frames(self) -> u32 {
        self.sample_frames
    }

    pub(crate) const fn frame_bytes(self) -> u16 {
        match self.layout {
            Layout::Pcm {
                bytes_per_frame, ..
            }
            | Layout::Blocked {
                bytes_per_frame, ..
            } => bytes_per_frame,
        }
    }

    pub(crate) fn samples_per_packet(self) -> u32 {
        match self.layout {
            Layout::Pcm { .. } => 1,
            Layout::Blocked {
                samples_per_packet, ..
            } => u32::from(samples_per_packet),
        }
    }

    pub(crate) fn packet_count(self) -> Result<u32, OpenError> {
        let step = self.samples_per_packet();
        if step == 0 || !self.sample_frames.is_multiple_of(step) {
            return Err(open_source("sample count is not a whole number of packets"));
        }
        Ok(self.sample_frames / step)
    }

    pub(crate) fn data_bytes(self) -> Result<u64, OpenError> {
        let packets = u64::from(self.packet_count()?);
        let frame = u64::from(self.frame_bytes());
        packets
            .checked_mul(frame)
            .ok_or_else(|| open_source("sound data does not fit"))
    }

    pub(crate) fn stream_codec(self) -> Result<StreamCodec, OpenError> {
        match self.layout {
            Layout::Pcm { .. } => Ok(StreamCodec::Known(CodecId::Pcm)),
            Layout::Blocked { tag, .. } => {
                let codec = UnknownCodec::new(tag.to_vec())
                    .map_err(|error| open_source(error.to_string()))?;
                Ok(StreamCodec::Unknown(codec))
            }
        }
    }

    pub(crate) fn extradata(self) -> Vec<u8> {
        match self.layout {
            Layout::Pcm { extra, .. } => extra.to_vec(),
            Layout::Blocked { .. } => Vec::new(),
        }
    }
}

pub(crate) fn open_source(message: impl Into<String>) -> OpenError {
    OpenError::Source {
        message: message.into(),
    }
}

pub(crate) fn parse_comm(payload: &[u8], aifc: bool) -> Result<Sound, OpenError> {
    if payload.len() < 18 {
        return Err(OpenError::Truncated {
            expected: 18,
            got: u64::try_from(payload.len()).unwrap_or(u64::MAX),
        });
    }
    let channels = u16::from_be_bytes([payload[0], payload[1]]);
    let sample_frames = u32::from_be_bytes([payload[2], payload[3], payload[4], payload[5]]);
    let bits = u16::from_be_bytes([payload[6], payload[7]]);
    let rate_bytes = [
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
        payload[14],
        payload[15],
        payload[16],
        payload[17],
    ];
    let rate = rate::decode_rate(&rate_bytes).map_err(rate_error)?;
    if channels == 0 {
        return Err(open_source("AIFF channel count is zero"));
    }
    if channels > MAX_CHANNELS {
        return Err(OpenError::HostileSize {
            declared: u64::from(channels),
            limit: u64::from(MAX_CHANNELS),
        });
    }
    let layout = if aifc {
        if payload.len() < 22 {
            return Err(OpenError::Truncated {
                expected: 22,
                got: u64::try_from(payload.len()).unwrap_or(u64::MAX),
            });
        }
        let tag = [payload[18], payload[19], payload[20], payload[21]];
        classify(tag, bits, channels)?
    } else {
        linear(bits, true, channels)?
    };
    Ok(Sound {
        channels,
        rate,
        sample_frames,
        layout,
    })
}

fn rate_error(error: RateError) -> OpenError {
    open_source(error.to_string())
}

fn classify(tag: [u8; 4], bits: u16, channels: u16) -> Result<Layout, OpenError> {
    // A-law and mu-law are PCM codings 7 and 8. FFmpeg also takes these two
    // tags in upper case; AudioToolbox does not open such a file.
    match &tag.map(|byte| byte.to_ascii_lowercase()) {
        b"alaw" => return law(7, bits, channels),
        b"ulaw" => return law(8, bits, channels),
        _ => {}
    }
    match &tag {
        b"NONE" | b"twos" => linear(bits, true, channels),
        b"sowt" => linear(bits, false, channels),
        b"in24" => sized(bits, 24, true, channels),
        b"in32" => sized(bits, 32, true, channels),
        b"raw " => sized(bits, 8, true, channels),
        b"fl32" => float_layout(bits, 32, channels),
        b"fl64" => float_layout(bits, 64, channels),
        b"ima4" => blocks(tag, channels, 34, 64),
        b"MAC3" => blocks(tag, channels, 2, 6),
        b"MAC6" => blocks(tag, channels, 1, 6),
        b"GSM " => gsm(channels),
        _ => Err(open_source(format!(
            "AIFC compression {} is not packetized",
            fourcc(tag)
        ))),
    }
}

fn sized(bits: u16, expect: u16, big: bool, channels: u16) -> Result<Layout, OpenError> {
    if bits != expect {
        return Err(open_source(format!(
            "sample size {bits} does not match the compression type"
        )));
    }
    linear(bits, big, channels)
}

fn linear(bits: u16, big: bool, channels: u16) -> Result<Layout, OpenError> {
    let coding = match bits {
        8 => 6,
        16 => 1,
        24 => 2,
        32 => 3,
        _ => return Err(open_source(format!("sample size {bits} is not linear pcm"))),
    };
    pcm_layout(coding, bits / 8, big, channels)
}

fn float_layout(bits: u16, expect: u16, channels: u16) -> Result<Layout, OpenError> {
    if bits != expect {
        return Err(open_source(format!(
            "sample size {bits} does not match the float compression"
        )));
    }
    let coding = if expect == 32 { 4 } else { 5 };
    pcm_layout(coding, expect / 8, true, channels)
}

fn pcm_layout(coding: u8, width: u16, big: bool, channels: u16) -> Result<Layout, OpenError> {
    let Some(bytes) = width.checked_mul(channels) else {
        return Err(open_source("frame does not fit"));
    };
    if bytes == 0 {
        return Err(open_source("frame is empty"));
    }
    Ok(Layout::Pcm {
        bytes_per_frame: bytes,
        extra: [coding, u8::from(big), 0],
    })
}

fn law(coding: u8, bits: u16, channels: u16) -> Result<Layout, OpenError> {
    if bits != 0 && bits != 8 {
        return Err(open_source(format!("sample size {bits} is not 8-bit")));
    }
    pcm_layout(coding, 1, true, channels)
}

fn gsm(channels: u16) -> Result<Layout, OpenError> {
    if channels == 0 {
        return Err(open_source("AIFF channel count is zero"));
    }
    Ok(Layout::Blocked {
        tag: *b"GSM ",
        bytes_per_frame: 33,
        samples_per_packet: 160,
    })
}

fn blocks(
    tag: [u8; 4],
    channels: u16,
    per_channel: u16,
    samples: u16,
) -> Result<Layout, OpenError> {
    let Some(bytes) = per_channel.checked_mul(channels) else {
        return Err(open_source("compression block does not fit"));
    };
    if bytes == 0 || samples == 0 {
        return Err(open_source("compression block is empty"));
    }
    Ok(Layout::Blocked {
        tag,
        bytes_per_frame: bytes,
        samples_per_packet: samples,
    })
}

fn fourcc(tag: [u8; 4]) -> String {
    if tag
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
    {
        tag.iter().map(|byte| char::from(*byte)).collect()
    } else {
        format!("{:02x}{:02x}{:02x}{:02x}", tag[0], tag[1], tag[2], tag[3])
    }
}

/// PCM this muxer will write. Big-endian integer is classic AIFF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MuxPcm {
    pub channels: u16,
    pub rate: u32,
    pub bits: u16,
    pub fourcc: Option<[u8; 4]>,
}

impl MuxPcm {
    pub(crate) fn frame_bytes(self) -> Result<usize, MuxError> {
        let width = usize::from(self.bits / 8);
        usize::from(self.channels)
            .checked_mul(width)
            .ok_or_else(|| rejected("frame does not fit"))
    }
}

pub(crate) fn mux_pcm(stream: &NewStream) -> Result<MuxPcm, MuxError> {
    if let StreamCodec::Unknown(unknown) = stream.params.codec() {
        return Err(MuxError::UnknownCodec(unknown.clone()));
    }
    if stream.params.codec() != &StreamCodec::Known(CodecId::Pcm) {
        return Err(rejected("AIFF mux writes linear PCM"));
    }
    let channels = stream
        .params
        .channels()
        .ok_or_else(|| rejected("AIFF needs audio channels"))?;
    let rate = stream
        .params
        .sample_rate()
        .ok_or_else(|| rejected("AIFF needs a sample rate"))?;
    if channels == 0 || rate == 0 {
        return Err(rejected("AIFF audio parameters are empty"));
    }
    if channels > MAX_CHANNELS {
        return Err(MuxError::HostileSize {
            declared: u64::from(channels),
            limit: u64::from(MAX_CHANNELS),
        });
    }
    let extra = stream.params.extradata();
    if extra.len() != 3 {
        return Err(rejected("PCM layout is three bytes"));
    }
    if extra[2] != 0 {
        return Err(rejected("AIFF stores interleaved frames"));
    }
    let big = match extra[1] {
        0 => false,
        1 => true,
        _ => return Err(rejected("PCM endian is not 0 or 1")),
    };
    let (bits, fourcc) = match extra[0] {
        6 => (8, integer_fourcc(big)),
        1 => (16, integer_fourcc(big)),
        2 => (24, integer_fourcc(big)),
        3 => (32, integer_fourcc(big)),
        4 => float_fourcc(big, 32, *b"fl32")?,
        5 => float_fourcc(big, 64, *b"fl64")?,
        _ => return Err(rejected("PCM coding is not a layout AIFF writes")),
    };
    Ok(MuxPcm {
        channels,
        rate,
        bits,
        fourcc,
    })
}

fn integer_fourcc(big: bool) -> Option<[u8; 4]> {
    if big { None } else { Some(*b"sowt") }
}

fn float_fourcc(big: bool, bits: u16, tag: [u8; 4]) -> Result<(u16, Option<[u8; 4]>), MuxError> {
    if !big {
        return Err(rejected("float AIFF is big-endian"));
    }
    Ok((bits, Some(tag)))
}

fn rejected(message: impl Into<String>) -> MuxError {
    MuxError::Rejected {
        message: message.into(),
    }
}
