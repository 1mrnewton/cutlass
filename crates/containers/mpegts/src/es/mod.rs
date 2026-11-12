//! Elementary streams: from PES payloads to access units.
//!
//! A PES payload may hold several access units, part of one, or one and a
//! half. Each kind of stream finds its own access-unit boundaries in the
//! bitstream. The PES timestamps follow H.222.0 clause 2.4.3.7: a PTS
//! belongs to the first access unit that starts in that PES packet. An
//! audio frame that starts in a PES packet without a timestamp of its own
//! is timed from the frame before it and that frame's sample count; a video
//! access unit in that place has no timestamp.
//!
//! Only what framing needs is read from the bitstream: start codes, unit
//! types, the slice-header fields that tell one picture from the next,
//! keyframes, the picture size, and the audio rate and channels. Decoding
//! stays in the codec crates.

pub mod ac3;
pub mod adts;
pub mod audio;
pub mod bits;
pub mod h264;
pub mod hevc;
pub mod latm;
pub mod mpa;
pub mod mpeg2;
pub mod opus;
pub mod video;

use cutlass_codec::{CodecId, StreamCodec, StreamParams};

use crate::psi::Kind;

/// One PES packet's payload and its header fields, timestamps already
/// unwrapped to the program clock.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pes {
    pub payload: Vec<u8>,
    pub pts: Option<i64>,
    pub dts: Option<i64>,
    /// File offset of the transport packet that started this PES packet.
    pub position: u64,
}

/// One access unit, ready to become a [`cutlass_codec::Packet`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub bytes: Vec<u8>,
    pub pts: Option<i64>,
    pub dts: Option<i64>,
    /// Ticks of the 90 kHz clock, when the bitstream states a length.
    pub duration: Option<i64>,
    pub keyframe: bool,
    /// File offset of the transport packet that started the PES packet in
    /// which this unit starts.
    pub position: u64,
}

/// Why bytes were dropped while framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFault {
    /// An access unit grew past the packet cap and was dropped.
    Oversize { declared: u64, limit: u64 },
    /// More frames in one PES packet than [`MAX_UNITS_PER_PES`].
    TooManyUnits,
}

/// Frames an audio PES packet may split into. A real packet holds a few
/// dozen at most.
pub const MAX_UNITS_PER_PES: usize = 16_384;

/// What a stream's bitstream says about it, once enough of it was read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Facts {
    Video {
        codec: CodecId,
        width: u32,
        height: u32,
        extradata: Vec<u8>,
    },
    Audio {
        codec: StreamCodec,
        rate: u32,
        channels: u16,
        extradata: Vec<u8>,
    },
}

impl Facts {
    /// Stream parameters, falling back to an opaque stream when a size or
    /// rate is zero.
    #[must_use]
    pub fn params(&self) -> StreamParams {
        match self {
            Self::Video {
                codec,
                width,
                height,
                extradata,
            } => match cutlass_codec::FrameSize::try_new(*width, *height) {
                Ok(size) => {
                    StreamParams::video(StreamCodec::Known(*codec), size, extradata.clone())
                }
                Err(_) => StreamParams::opaque(StreamCodec::Known(*codec), extradata.clone()),
            },
            Self::Audio {
                codec,
                rate,
                channels,
                extradata,
            } => StreamParams::try_audio(codec.clone(), *rate, *channels, extradata.clone())
                .unwrap_or_else(|_| StreamParams::opaque(codec.clone(), extradata.clone())),
        }
    }
}

/// The framer of one elementary stream.
pub enum Framer {
    H264(video::VideoFramer<h264::H264>),
    Hevc(video::VideoFramer<hevc::Hevc>),
    Mpeg2(video::VideoFramer<mpeg2::Mpeg2>),
    Adts(audio::AudioFramer<adts::Adts>),
    Ac3(audio::AudioFramer<ac3::Ac3>),
    MpegAudio(audio::AudioFramer<mpa::MpegAudio>),
    Latm(audio::AudioFramer<latm::Loas>),
    Opus(opus::OpusFramer),
    /// One PES packet is one packet.
    Whole,
}

impl Framer {
    #[must_use]
    pub fn new(kind: Kind, opus_channels: Option<u8>, max_unit: usize) -> Self {
        match kind {
            Kind::H264 => Self::H264(video::VideoFramer::new(max_unit)),
            Kind::Hevc => Self::Hevc(video::VideoFramer::new(max_unit)),
            Kind::Mpeg2Video => Self::Mpeg2(video::VideoFramer::new(max_unit)),
            Kind::Adts => Self::Adts(audio::AudioFramer::new(adts::Adts::default())),
            Kind::Ac3 | Kind::Eac3 => Self::Ac3(audio::AudioFramer::new(ac3::Ac3::default())),
            Kind::MpegAudio => Self::MpegAudio(audio::AudioFramer::new(mpa::MpegAudio::default())),
            Kind::Latm => Self::Latm(audio::AudioFramer::new(latm::Loas::default())),
            Kind::Opus => Self::Opus(opus::OpusFramer::new(opus_channels)),
            Kind::Mpeg4Video | Kind::Other => Self::Whole,
        }
    }

    /// Frames one PES packet. Finished units go to `out` in decode order.
    pub fn push(&mut self, pes: Pes, out: &mut Vec<Unit>) -> Result<(), FrameFault> {
        match self {
            Self::H264(framer) => framer.push(pes, out),
            Self::Hevc(framer) => framer.push(pes, out),
            Self::Mpeg2(framer) => framer.push(pes, out),
            Self::Adts(framer) => framer.push(pes, out),
            Self::Ac3(framer) => framer.push(pes, out),
            Self::MpegAudio(framer) => framer.push(pes, out),
            Self::Latm(framer) => framer.push(pes, out),
            Self::Opus(framer) => framer.push(pes, out),
            Self::Whole => {
                out.push(Unit {
                    bytes: pes.payload,
                    pts: pes.pts,
                    dts: pes.dts,
                    duration: None,
                    keyframe: true,
                    position: pes.position,
                });
                Ok(())
            }
        }
    }

    /// The stream ended: the unit still open is finished.
    pub fn flush(&mut self, out: &mut Vec<Unit>) {
        match self {
            Self::H264(framer) => framer.flush(out),
            Self::Hevc(framer) => framer.flush(out),
            Self::Mpeg2(framer) => framer.flush(out),
            Self::Adts(framer) => framer.flush(out),
            Self::Ac3(framer) => framer.flush(out),
            Self::MpegAudio(framer) => framer.flush(out),
            Self::Latm(framer) => framer.flush(out),
            Self::Opus(_) | Self::Whole => {}
        }
    }

    /// Forgets partial units, keeping what was learned about the stream.
    pub fn reset(&mut self) {
        match self {
            Self::H264(framer) => framer.reset(),
            Self::Hevc(framer) => framer.reset(),
            Self::Mpeg2(framer) => framer.reset(),
            Self::Adts(framer) => framer.reset(),
            Self::Ac3(framer) => framer.reset(),
            Self::MpegAudio(framer) => framer.reset(),
            Self::Latm(framer) => framer.reset(),
            Self::Opus(_) | Self::Whole => {}
        }
    }

    /// Bytes held for a unit that is not finished yet.
    #[must_use]
    pub fn buffered(&self) -> usize {
        match self {
            Self::H264(framer) => framer.buffered(),
            Self::Hevc(framer) => framer.buffered(),
            Self::Mpeg2(framer) => framer.buffered(),
            Self::Adts(framer) => framer.buffered(),
            Self::Ac3(framer) => framer.buffered(),
            Self::MpegAudio(framer) => framer.buffered(),
            Self::Latm(framer) => framer.buffered(),
            Self::Opus(_) | Self::Whole => 0,
        }
    }

    /// The stream's parameters, once its bitstream gave them.
    #[must_use]
    pub fn facts(&self) -> Option<Facts> {
        match self {
            Self::H264(framer) => framer.facts(),
            Self::Hevc(framer) => framer.facts(),
            Self::Mpeg2(framer) => framer.facts(),
            Self::Adts(framer) => framer.facts(),
            Self::Ac3(framer) => framer.facts(),
            Self::MpegAudio(framer) => framer.facts(),
            Self::Latm(framer) => framer.facts(),
            Self::Opus(framer) => framer.facts(),
            Self::Whole => None,
        }
    }

    /// Whether this stream's parameters come from its bitstream at all.
    #[must_use]
    pub const fn learns_facts(&self) -> bool {
        !matches!(self, Self::Whole)
    }
}
