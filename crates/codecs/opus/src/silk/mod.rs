//! The SILK layer (RFC 6716 section 4.2), with the resampler that brings
//! its 8, 12, or 16 kHz output to the API rate.

mod channel;
mod cng;
mod fixed;
mod indices;
mod nlsf;
mod plc;
mod pulses;
mod resampler;
mod stereo;
mod tables;

use crate::range::RangeDecoder;
use cutlass_decode::CodecError;

/// How [`SilkDecoder::decode`] treats this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SilkLoss {
    /// Decode the next frame of the packet.
    Normal,
    /// The packet is lost: conceal one frame.
    Lost,
    /// Decode the low-bitrate redundancy (LBRR) copy this packet carries
    /// of the previous, lost packet.
    Fec,
}

/// Stream configuration for one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SilkControl {
    /// Output channels, 1 or 2.
    pub api_channels: usize,
    /// Coded channels in this packet, 1 or 2.
    pub internal_channels: usize,
    /// Output rate: 8, 12, 16, 24, or 48 kHz.
    pub api_rate: u32,
    /// SILK rate for this packet: 8, 12, or 16 kHz.
    pub internal_rate: u32,
    /// Packet duration in milliseconds: 10, 20, 40, or 60.
    pub payload_ms: u32,
}

/// SILK decoder state for one Opus stream.
#[derive(Clone, Debug, Default)]
pub(crate) struct SilkDecoder {}

impl SilkDecoder {
    pub(crate) fn new() -> Self {
        Self {}
    }

    /// Drops every channel's history, the stereo predictor, and the resampler.
    pub(crate) fn reset(&mut self) {}

    /// Decodes the next 10 or 20 ms SILK frame and resamples it.
    ///
    /// `new_packet` is true for the first frame of a packet, when the
    /// VAD and LBRR flags are read. Returns the samples per channel written
    /// to `out`, interleaved by `ctl.api_channels`, at `ctl.api_rate`.
    pub(crate) fn decode(
        &mut self,
        ctl: &SilkControl,
        loss: SilkLoss,
        new_packet: bool,
        rd: &mut RangeDecoder<'_>,
        out: &mut [i16],
    ) -> Result<usize, CodecError> {
        let _ = (ctl, loss, new_packet, rd, out);
        Err(CodecError::Unsupported {
            what: "SILK decode",
        })
    }
}
