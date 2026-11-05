//! The FLAC encoder.
//!
//! [`FlacEncoder::encode`] takes interleaved integer samples at the
//! stream's depth and returns finished frames. The [`Encoder`]
//! implementation reads the mix format and returns packets: the first
//! carries the opening STREAMINFO as [`SideData::Config`], and the last
//! one from `drain` carries the final STREAMINFO with the sample count,
//! frame sizes, and MD5 signature, for a muxer to write back.

mod frame;
mod lpc;
mod rice;
mod settings;
mod subframe;

pub use settings::{Apodization, DEFAULT_LEVEL, Settings, Stereo};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    ChannelLayout, CodecError, CodecId, CodecParams, Packet, SampleFormat, SideData,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, PictureInput};

use crate::md5::Md5;
use crate::pcm;
use crate::streaminfo::StreamInfo;

use frame::FrameEncoder;

/// Highest sample rate STREAMINFO holds (20 bits).
const MAX_RATE: u32 = (1 << 20) - 1;

/// One finished frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedFrame {
    pub data: Vec<u8>,
    /// Number of the frame's first sample per channel.
    pub first_sample: u64,
    /// Samples per channel.
    pub samples: u32,
}

/// A FLAC encoder for one stream.
pub struct FlacEncoder {
    frames: FrameEncoder,
    rate: u32,
    channels: u8,
    bits: u8,
    block: usize,
    /// Input channel feeding each FLAC channel, for the trait path.
    order: Vec<usize>,
    layout: ChannelLayout,
    format: SampleFormat,
    pending: Vec<i32>,
    converted: Vec<i32>,
    number: u64,
    samples: u64,
    md5: Md5,
    min_frame: u32,
    max_frame: u32,
    config_sent: bool,
    cancel: Arc<AtomicBool>,
}

impl FlacEncoder {
    /// An encoder for `channels` channels (1 to 8) of `bits`-deep samples
    /// (4 to 32) at `rate` Hz.
    pub fn new(rate: u32, channels: u8, bits: u8, settings: Settings) -> Result<Self, CodecError> {
        settings.validate()?;
        if rate == 0 || rate > MAX_RATE {
            return Err(CodecError::Unsupported {
                what: "flac sample rate",
            });
        }
        if !(4..=32).contains(&bits) {
            return Err(CodecError::Unsupported {
                what: "flac bit depth",
            });
        }
        let layout = pcm::layout(channels)?;
        Ok(Self {
            block: usize::from(settings.block_size),
            frames: FrameEncoder::new(settings, rate, channels, bits),
            rate,
            channels,
            bits,
            order: (0..usize::from(channels)).collect(),
            layout,
            format: if bits == 16 {
                SampleFormat::I16
            } else {
                SampleFormat::F32
            },
            pending: Vec::new(),
            converted: Vec::new(),
            number: 0,
            samples: 0,
            md5: Md5::new(),
            min_frame: 0,
            max_frame: 0,
            config_sent: false,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Opens at [`DEFAULT_LEVEL`] from engine parameters.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Self::open_with(params, Settings::level(DEFAULT_LEVEL)?)
    }

    /// Opens from engine parameters with explicit settings.
    ///
    /// The depth is the STREAMINFO's when `extra` holds one, else 16 for
    /// `I16` input and 24 for `F32`. `layout` must name the channels of a
    /// FLAC channel order (RFC 9639 section 9.1.3), in any order; they are
    /// reordered on the way in.
    pub fn open_with(params: &CodecParams, settings: Settings) -> Result<Self, CodecError> {
        if params.codec != CodecId::Flac {
            return Err(CodecError::Unsupported { what: "flac" });
        }
        let input = params.layout.clone().ok_or(CodecError::InvalidApi {
            what: "channel layout",
        })?;
        let channels = u8::try_from(input.len())
            .ok()
            .filter(|count| (1..=8).contains(count))
            .ok_or(CodecError::Unsupported {
                what: "flac channel count",
            })?;
        let format = params.sample_format.unwrap_or(SampleFormat::F32);
        let bits = match StreamInfo::from_extra(&params.extra)? {
            Some(info) => {
                if info.channels != channels || info.sample_rate != params.sample_rate {
                    return Err(CodecError::InvalidApi {
                        what: "streaminfo differs from the parameters",
                    });
                }
                info.bits
            }
            None if format == SampleFormat::I16 => 16,
            None => 24,
        };
        if format == SampleFormat::I16 && bits != 16 {
            return Err(CodecError::Unsupported {
                what: "16-bit input to a flac stream of another depth",
            });
        }
        let mut encoder = Self::new(params.sample_rate, channels, bits, settings)?;
        let flac = pcm::layout(channels)?;
        let mut order = Vec::with_capacity(flac.len());
        for channel in flac.channels() {
            let at = input.channels().iter().position(|c| c == channel).ok_or(
                CodecError::Unsupported {
                    what: "channel layout without a flac order",
                },
            )?;
            order.push(at);
        }
        encoder.order = order;
        encoder.layout = input;
        encoder.format = format;
        Ok(encoder)
    }

    /// The STREAMINFO of the frames returned so far. The MD5 signature
    /// covers every sample encoded, including those still buffered only
    /// once [`Self::finish`] has run.
    #[must_use]
    pub fn stream_info(&self) -> StreamInfo {
        let block = u16::try_from(self.block).unwrap_or(u16::MAX);
        StreamInfo {
            min_block: block,
            max_block: block,
            min_frame: self.min_frame,
            max_frame: self.max_frame,
            sample_rate: self.rate,
            channels: self.channels,
            bits: self.bits,
            total_samples: if self.samples < (1 << 36) {
                self.samples
            } else {
                0
            },
            md5: if self.samples == 0 {
                [0; 16]
            } else {
                self.md5.digest()
            },
        }
    }

    /// The settings in use.
    #[must_use]
    pub fn settings(&self) -> &Settings {
        self.frames.settings()
    }

    /// Buffers interleaved samples and returns the frames they complete.
    ///
    /// A sample outside the depth is [`CodecError::OutOfRange`] and adds
    /// nothing. A frame is emitted only once a sample past it is buffered,
    /// so [`Self::finish`] always has the last frame to emit.
    pub fn encode(&mut self, interleaved: &[i32]) -> Result<Vec<EncodedFrame>, CodecError> {
        if !interleaved.len().is_multiple_of(usize::from(self.channels)) {
            return Err(CodecError::UnevenChannels);
        }
        let half = 1i64 << (self.bits - 1);
        if interleaved
            .iter()
            .any(|sample| !(-half..half).contains(&i64::from(*sample)))
        {
            return Err(CodecError::OutOfRange);
        }
        self.pending.extend_from_slice(interleaved);
        self.emit(false)
    }

    /// Emits the buffered samples as the last frame.
    pub fn finish(&mut self) -> Result<Vec<EncodedFrame>, CodecError> {
        self.emit(true)
    }

    fn emit(&mut self, all: bool) -> Result<Vec<EncodedFrame>, CodecError> {
        let channels = usize::from(self.channels);
        let width = self.block * channels;
        let mut frames = Vec::new();
        let mut start = 0;
        while self.pending.len() - start > width || (all && self.pending.len() > start) {
            if self.cancel.load(Ordering::Relaxed) {
                self.pending.drain(..start);
                return Err(CodecError::Cancelled);
            }
            let end = (start + width).min(self.pending.len());
            let block = &self.pending[start..end];
            let data = self.frames.encode(block, self.number)?;
            self.md5.samples(block, u32::from(self.bits));
            let samples = u32::try_from(block.len() / channels).unwrap_or(0);
            let size = u32::try_from(data.len()).unwrap_or(u32::MAX);
            self.min_frame = if self.min_frame == 0 {
                size
            } else {
                self.min_frame.min(size)
            };
            self.max_frame = self.max_frame.max(size);
            frames.push(EncodedFrame {
                data,
                first_sample: self.samples,
                samples,
            });
            self.number += 1;
            self.samples += u64::from(samples);
            start = end;
        }
        self.pending.drain(..start);
        Ok(frames)
    }

    fn packets(&mut self, frames: Vec<EncodedFrame>, last: bool) -> Vec<Packet> {
        let count = frames.len();
        let mut packets: Vec<Packet> = frames
            .into_iter()
            .map(|frame| {
                Packet::audio(
                    frame.data,
                    i64::try_from(frame.first_sample).unwrap_or(i64::MAX),
                    i64::from(frame.samples),
                    self.rate,
                )
            })
            .collect();
        if !self.config_sent
            && let Some(first) = packets.first_mut()
        {
            let mut opening = self.stream_info();
            opening.min_frame = 0;
            opening.max_frame = 0;
            opening.total_samples = 0;
            opening.md5 = [0; 16];
            first
                .side
                .push(SideData::Config(opening.to_bytes().to_vec()));
            self.config_sent = true;
        }
        if last && count > 0 {
            let info = self.stream_info();
            if let Some(packet) = packets.last_mut() {
                packet.side.push(SideData::Config(info.to_bytes().to_vec()));
            }
        }
        packets
    }
}

impl Encoder for FlacEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Flac
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if input.pcm.rate != self.rate || input.pcm.layout != self.layout {
            return Err(CodecError::InvalidApi {
                what: "audio format",
            });
        }
        if input.pcm.format != self.format {
            return Err(CodecError::InvalidApi {
                what: "sample format",
            });
        }
        if input.pcm.samples == 0 {
            return Ok(Vec::new());
        }
        let mut read = std::mem::take(&mut self.converted);
        read.clear();
        let result = pcm::from_pcm(&input.pcm, self.bits, &mut read).and_then(|()| {
            let channels = self.order.len();
            if self.order.iter().enumerate().all(|(i, at)| i == *at) {
                self.encode(&read)
            } else {
                let mut ordered = Vec::with_capacity(read.len());
                for frame in read.chunks_exact(channels) {
                    ordered.extend(self.order.iter().map(|at| frame[*at]));
                }
                self.encode(&ordered)
            }
        });
        self.converted = read;
        let frames = result?;
        Ok(self.packets(frames, false))
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "picture" })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let frames = self.finish()?;
        Ok(self.packets(frames, true))
    }

    /// Starts a new stream: frame numbers, the sample count, the signature,
    /// and the buffer start over.
    fn reset(&mut self) {
        self.pending.clear();
        self.number = 0;
        self.samples = 0;
        self.md5 = Md5::new();
        self.min_frame = 0;
        self.max_frame = 0;
        self.config_sent = false;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests;
