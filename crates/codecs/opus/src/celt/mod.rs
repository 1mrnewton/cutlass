//! The CELT layer (RFC 6716 section 4.3).
//!
//! The decoder runs at 48 kHz internally and writes the API rate by
//! integer decimation in the de-emphasis filter.

mod bands;
mod coder;
mod cwrs;
mod energy;
mod laplace;
mod mdct;
mod pitch;
mod postfilter;
mod rate;
pub(crate) mod softclip;
mod tables;
mod vq;

use crate::range::RangeDecoder;
use cutlass_decode::CodecError;

/// CELT decoder state for one Opus stream.
#[derive(Clone, Debug)]
pub(crate) struct CeltDecoder {
    channels: usize,
    downsample: usize,
}

/// One CELT frame for [`CeltDecoder::decode`].
pub(crate) struct CeltFrame<'d, 'a> {
    /// The frame's range decoder. In hybrid mode it is the one SILK already
    /// read from; its storage is the frame length CELT may use. `None` asks
    /// for packet-loss concealment of `frame_size` samples.
    pub rd: Option<&'d mut RangeDecoder<'a>>,
    /// Samples per channel at 48 kHz: 120, 240, 480, or 960.
    pub frame_size: usize,
    /// Coded channels in this frame, 1 or 2.
    pub stream_channels: usize,
    /// First coded band: 0, or 17 in hybrid mode.
    pub start_band: usize,
    /// Band after the last coded one: 13, 17, 19, or 21.
    pub end_band: usize,
}

impl CeltDecoder {
    /// `channels` is the output count, 1 or 2. `api_rate` is 8, 12, 16,
    /// 24, or 48 kHz.
    pub(crate) fn new(channels: usize, api_rate: u32) -> Result<Self, CodecError> {
        let downsample = match api_rate {
            48_000 => 1,
            24_000 => 2,
            16_000 => 3,
            12_000 => 4,
            8_000 => 6,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "Opus API sample rate",
                });
            }
        };
        if !(1..=2).contains(&channels) {
            return Err(CodecError::Unsupported {
                what: "CELT channel count",
            });
        }
        Ok(Self {
            channels,
            downsample,
        })
    }

    /// Drops overlap, energy history, the post-filter, and PLC state.
    pub(crate) fn reset(&mut self) {}

    /// Decodes one frame into `out`: `frame_size / downsample` interleaved
    /// samples per output channel, full scale 1.0, overwriting `out`.
    pub(crate) fn decode(
        &mut self,
        frame: CeltFrame<'_, '_>,
        out: &mut [f32],
    ) -> Result<(), CodecError> {
        let _ = (frame, out, self.channels, self.downsample);
        Err(CodecError::Unsupported { what: "CELT decode" })
    }
}
