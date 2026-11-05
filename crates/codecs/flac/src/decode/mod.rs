//! The FLAC decoder.
//!
//! [`FlacDecoder::decode`] returns integer samples at the stream's own bit
//! depth. The [`Decoder`] implementation wraps it for the engine and writes
//! the mix format the open parameters ask for (see `pcm.rs`).

mod frame;

pub use frame::Frame;

use cutlass_decode::{
    AudioOutput, CodecError, CodecId, CodecParams, Decoder, Output, Packet, SampleFormat, SideData,
};

use crate::md5::Md5;
use crate::pcm;
use crate::streaminfo::StreamInfo;

/// Largest packet this decoder reads. A legal frame is smaller: 65535
/// samples of eight 32-bit channels, plus one 33-bit side channel, coded
/// verbatim, is under 2.2 MiB.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// What the decoded stream says about the STREAMINFO signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Md5Check {
    /// Every sample from the first was decoded in order and the digest matches.
    Match,
    /// Every sample was decoded in order and the digest differs.
    Mismatch,
    /// The STREAMINFO has no signature, or no STREAMINFO was given.
    Unknown,
    /// Not every sample was decoded in order from the first: a seek, a
    /// reset, a gap, or a stream that has not reached its last sample.
    Incomplete,
}

/// Properties fixed by the STREAMINFO or by the first frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    sample_rate: u32,
    channels: u8,
    bits: u8,
}

/// Where the next in-order frame starts, for the signature check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Expect {
    /// Next frame number in a fixed-block-size stream, or the next sample
    /// number in a variable-block-size stream.
    number: u64,
    samples: u64,
    variable: bool,
}

/// A FLAC decoder for one stream.
pub struct FlacDecoder {
    info: Option<StreamInfo>,
    shape: Option<Shape>,
    format: SampleFormat,
    planes: Vec<i64>,
    md5: Md5,
    expect: Option<Expect>,
    in_order: bool,
}

impl FlacDecoder {
    /// A decoder for frames of the stream `info` describes, or of any stream
    /// when `info` is `None`; the first frame then fixes the rate, channel
    /// count, and depth.
    #[must_use]
    pub fn new(info: Option<StreamInfo>) -> Self {
        Self {
            shape: info.map(|info| Shape {
                sample_rate: info.sample_rate,
                channels: info.channels,
                bits: info.bits,
            }),
            info,
            format: SampleFormat::F32,
            planes: Vec::new(),
            md5: Md5::new(),
            expect: None,
            in_order: true,
        }
    }

    /// Opens from engine parameters.
    ///
    /// `extra` is empty, a 34-byte STREAMINFO body, or `fLaC` and a
    /// STREAMINFO block. `sample_format` picks the output: `I16` for streams
    /// up to 16 bits, otherwise `F32` (the default). A layout whose length
    /// differs from the STREAMINFO channel count is refused.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Flac {
            return Err(CodecError::Unsupported { what: "flac" });
        }
        let info = StreamInfo::from_extra(&params.extra)?;
        let format = params.sample_format.unwrap_or(SampleFormat::F32);
        if let Some(info) = info {
            if let Some(layout) = &params.layout
                && layout.len() != usize::from(info.channels)
            {
                return Err(CodecError::InvalidApi {
                    what: "layout differs from streaminfo",
                });
            }
            if format == SampleFormat::I16 && info.bits > 16 {
                return Err(CodecError::Unsupported {
                    what: "16-bit output of a deeper flac stream",
                });
            }
        }
        let mut decoder = Self::new(info);
        decoder.format = format;
        Ok(decoder)
    }

    /// The STREAMINFO in use, if any.
    #[must_use]
    pub fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    /// Decodes one complete frame to integer samples.
    ///
    /// A failed frame leaves the decoder as it was.
    pub fn decode(&mut self, bytes: &[u8]) -> Result<Frame, CodecError> {
        let (frame, shape) = self.checked(bytes)?;
        self.commit(&frame, shape);
        Ok(frame)
    }

    /// Decodes without changing any state.
    fn checked(&mut self, bytes: &[u8]) -> Result<(Frame, Shape), CodecError> {
        if bytes.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(CodecError::Limit {
                what: "packet bytes",
            });
        }
        let frame = frame::decode_frame(bytes, self.info.as_ref(), &mut self.planes)?;
        let shape = Shape {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            bits: frame.bits,
        };
        if let Some(known) = self.shape
            && known != shape
        {
            return Err(CodecError::Unsupported {
                what: "stream property change",
            });
        }
        Ok((frame, shape))
    }

    fn commit(&mut self, frame: &Frame, shape: Shape) {
        self.shape = Some(shape);
        self.track(frame);
    }

    /// The signature check for the frames decoded so far.
    #[must_use]
    pub fn md5_check(&self) -> Md5Check {
        let Some(info) = self.info else {
            return Md5Check::Unknown;
        };
        if info.md5 == [0; 16] {
            return Md5Check::Unknown;
        }
        let complete = self.in_order
            && self
                .expect
                .is_some_and(|expect| expect.samples == info.total_samples)
            && info.total_samples > 0;
        if !complete {
            return Md5Check::Incomplete;
        }
        if self.md5.digest() == info.md5 {
            Md5Check::Match
        } else {
            Md5Check::Mismatch
        }
    }

    fn track(&mut self, frame: &Frame) {
        let header = frame.header;
        let expected = match self.expect {
            None => header.number == 0,
            Some(expect) => {
                expect.variable == header.variable_block && expect.number == header.number
            }
        };
        if !expected {
            self.in_order = false;
        }
        if self.in_order {
            self.md5.samples(&frame.samples, u32::from(frame.bits));
        }
        let samples = self
            .expect
            .map_or(0, |expect| expect.samples)
            .saturating_add(u64::from(header.block_size));
        let number = if header.variable_block {
            header.number.saturating_add(u64::from(header.block_size))
        } else {
            header.number.saturating_add(1)
        };
        self.expect = Some(Expect {
            number,
            samples,
            variable: header.variable_block,
        });
    }

    fn adopt(&mut self, config: &[u8]) -> Result<(), CodecError> {
        let Some(info) = StreamInfo::from_extra(config)? else {
            return Ok(());
        };
        let shape = Shape {
            sample_rate: info.sample_rate,
            channels: info.channels,
            bits: info.bits,
        };
        if self.shape.is_some_and(|known| known != shape) {
            return Err(CodecError::Unsupported {
                what: "configuration change",
            });
        }
        if self.format == SampleFormat::I16 && info.bits > 16 {
            return Err(CodecError::Unsupported {
                what: "16-bit output of a deeper flac stream",
            });
        }
        self.shape = Some(shape);
        self.info = Some(info);
        Ok(())
    }
}

impl Decoder for FlacDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Flac
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let saved_info = self.info;
        let saved_shape = self.shape;
        for side in &packet.side {
            if let SideData::Config(config) = side
                && let Err(error) = self.adopt(config)
            {
                self.info = saved_info;
                self.shape = saved_shape;
                return Err(error);
            }
        }
        let decoded = self.checked(&packet.data).and_then(|(frame, shape)| {
            let layout = pcm::layout(frame.channels)?;
            let pcm = pcm::to_pcm(
                &frame.samples,
                frame.bits,
                frame.sample_rate,
                layout,
                self.format,
            )?;
            Ok((frame, shape, pcm))
        });
        match decoded {
            Ok((frame, shape, pcm)) => {
                self.commit(&frame, shape);
                Ok(vec![Output::Audio(AudioOutput {
                    pcm,
                    pts: packet.pts,
                    timescale: packet.timescale,
                    skip_leading: 0,
                    skip_trailing: 0,
                    concealed: false,
                })])
            }
            Err(error) => {
                self.info = saved_info;
                self.shape = saved_shape;
                Err(error)
            }
        }
    }

    /// FLAC frames have no overlap or delay, so there is nothing to flush.
    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    /// Forgets the signature progress. The STREAMINFO and the stream shape
    /// stay; a seek is followed by a reset.
    fn reset(&mut self) {
        self.md5 = Md5::new();
        self.expect = None;
        self.in_order = true;
    }
}
