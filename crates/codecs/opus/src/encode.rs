//! The Opus encoder behind [`cutlass_encode::Encoder`].
//!
//! CELT-only (RFC 6716 section 4.3): mono or stereo at 48 kHz, 20 ms
//! frames, one frame per packet (TOC code 0). SILK and hybrid coding are
//! not built. The bitrate picks the coded bandwidth: below about 10 kb/s
//! per channel narrowband, then wideband, super-wideband, and fullband
//! from about 18 kb/s per channel.
//!
//! Packets are clocked at 48 kHz from 0. The first carries the `OpusHead`
//! as [`SideData::Config`]; the pre-skip is the CELT overlap, 120 samples.
//! The last packet is shortened so the durations add up to the pre-skip
//! plus the input, and carries [`SideData::Gapless`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet, SideData,
};

use crate::celt::{CeltConfig, CeltEncoder, RateMode};
use crate::header::OpusHead;
use crate::packet::{Bandwidth, Mode, Toc};

/// The only rate the encoder takes.
const RATE: u32 = 48_000;
/// Encoder delay: the CELT MDCT overlap.
const PRE_SKIP: u16 = 120;
/// How far past full scale a float sample may be before it is refused.
const FULL_SCALE_SLACK: f32 = 16.0 * f32::EPSILON;
const MIN_BITRATE: u32 = 6_000;
const MAX_BITRATE: u32 = 510_000;

/// Frame length choices for [`OpusEncoder::open_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameDuration {
    Ms2_5,
    Ms5,
    Ms10,
    Ms20,
}

impl FrameDuration {
    fn lm(self) -> usize {
        match self {
            Self::Ms2_5 => 0,
            Self::Ms5 => 1,
            Self::Ms10 => 2,
            Self::Ms20 => 3,
        }
    }
}

/// Settings the codec parameters do not carry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    pub frame: FrameDuration,
    /// Constrained VBR instead of plain VBR when the rate is not constant.
    pub constrained: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            frame: FrameDuration::Ms20,
            constrained: false,
        }
    }
}

/// An Opus encoder for one mono or stereo stream.
#[derive(Debug)]
pub struct OpusEncoder {
    cancel: Arc<AtomicBool>,
    state: State,
}

/// Everything a failed push puts back.
#[derive(Clone, Debug)]
struct State {
    channels: usize,
    celt: CeltEncoder,
    toc: u8,
    frame: usize,
    /// Accepted samples not yet coded, interleaved.
    pending: Vec<f32>,
    /// Samples per channel accepted since open or reset.
    accepted: u64,
    /// Packets emitted since open or reset.
    packets: u64,
    drained: bool,
}

impl OpusEncoder {
    /// Opens an encoder for `params`: Opus, mono or stereo, 48 kHz.
    /// Bitrate 0 asks for 64 kb/s mono or 96 kb/s stereo; others are
    /// clamped to 6..=510 kb/s. `constant_rate` asks for CBR, otherwise
    /// the rate is VBR.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Self::open_with(params, Options::default())
    }

    pub(crate) fn open_with(params: &CodecParams, options: Options) -> Result<Self, CodecError> {
        if params.codec != CodecId::Opus {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.sample_rate != RATE {
            return Err(CodecError::Unsupported {
                what: "Opus encoder sample rate other than 48 kHz",
            });
        }
        let layout = params
            .layout
            .as_ref()
            .ok_or(CodecError::InvalidApi { what: "layout" })?;
        let channels = layout.len();
        if !(1..=2).contains(&channels) {
            return Err(CodecError::Unsupported {
                what: "Opus encoder channel count",
            });
        }
        let bitrate = match params.bitrate {
            0 if channels == 1 => 64_000,
            0 => 96_000,
            rate => rate.clamp(MIN_BITRATE, MAX_BITRATE),
        };
        let mode = match (params.constant_rate, options.constrained) {
            (true, _) => RateMode::Cbr,
            (false, true) => RateMode::Constrained,
            (false, false) => RateMode::Vbr,
        };
        let bandwidth = bandwidth_for(bitrate, channels);
        let lm = options.frame.lm();
        let frame = 120usize << lm;
        let config = Toc::config(Mode::CeltOnly, bandwidth, frame as u32).ok_or(
            CodecError::Unsupported {
                what: "Opus configuration",
            },
        )?;
        let celt = CeltEncoder::new(CeltConfig {
            channels,
            lm,
            end_band: bandwidth.celt_end_band(),
            bitrate,
            mode,
        })?;
        Ok(Self {
            cancel: Arc::new(AtomicBool::new(false)),
            state: State {
                channels,
                celt,
                toc: Toc::byte(config, channels == 2, 0),
                frame,
                pending: Vec::new(),
                accepted: 0,
                packets: 0,
                drained: false,
            },
        })
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn push_samples(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancelled() {
            return Err(CodecError::Cancelled);
        }
        if self.state.drained {
            return Err(CodecError::InvalidApi {
                what: "push after drain",
            });
        }
        let samples = pcm_samples(self.state.channels, input)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let saved = self.state.clone();
        self.state.pending.extend_from_slice(&samples);
        self.state.accepted += (samples.len() / self.state.channels) as u64;
        let mut packets = Vec::new();
        while self.state.pending.len() >= self.state.frame * self.state.channels {
            if self.cancelled() {
                self.state = saved;
                return Err(CodecError::Cancelled);
            }
            match self.state.encode_pending() {
                Ok(packet) => packets.push(packet),
                Err(err) => {
                    self.state = saved;
                    return Err(err);
                }
            }
        }
        Ok(packets)
    }

    fn drain_samples(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.state.drained {
            return Ok(Vec::new());
        }
        if self.cancelled() {
            return Err(CodecError::Cancelled);
        }
        let saved = self.state.clone();
        let result = self.flush();
        match result {
            Ok(packets) => {
                self.state.drained = true;
                Ok(packets)
            }
            Err(err) => {
                self.state = saved;
                Err(err)
            }
        }
    }

    /// Pads the input with silence to the end of the delayed signal, codes
    /// it, and trims the last packet to the end of the input.
    fn flush(&mut self) -> Result<Vec<Packet>, CodecError> {
        let st = &mut self.state;
        let mut packets = Vec::new();
        if st.accepted == 0 {
            return Ok(packets);
        }
        let frame = st.frame as u64;
        let span = st.accepted + u64::from(PRE_SKIP);
        let total = span.div_ceil(frame);
        let missing = (total.saturating_sub(st.packets) * frame)
            .saturating_mul(st.channels as u64)
            .saturating_sub(st.pending.len() as u64);
        let missing = usize::try_from(missing).map_err(|_| CodecError::Limit {
            what: "Opus flush length",
        })?;
        st.pending.resize(st.pending.len() + missing, 0.0);
        while st.packets < total {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(CodecError::Cancelled);
            }
            packets.push(st.encode_pending()?);
        }
        let padding = total * frame - span;
        let padding = u32::try_from(padding).map_err(|_| CodecError::Limit {
            what: "Opus trailing padding",
        })?;
        if let Some(last) = packets.last_mut() {
            last.duration -= i64::from(padding);
            last.side.push(SideData::Gapless {
                encoder_delay: u32::from(PRE_SKIP),
                trailing_padding: padding,
                sample_rate: RATE,
            });
        }
        Ok(packets)
    }

    fn reset_state(&mut self) {
        let st = &mut self.state;
        st.celt.reset();
        st.pending.clear();
        st.accepted = 0;
        st.packets = 0;
        st.drained = false;
    }
}

impl State {
    /// Codes the first frame of `pending` into one packet.
    fn encode_pending(&mut self) -> Result<Packet, CodecError> {
        let n = self.frame * self.channels;
        let Some(frame) = self.pending.get(..n) else {
            return Err(CodecError::InvalidApi { what: "Opus frame" });
        };
        let body = self.celt.encode(frame)?;
        self.pending.drain(..n);
        let mut data = Vec::with_capacity(1 + body.len());
        data.push(self.toc);
        data.extend_from_slice(&body);
        let pts =
            i64::try_from(self.packets * self.frame as u64).map_err(|_| CodecError::Limit {
                what: "Opus timestamp",
            })?;
        let mut packet = Packet::audio(data, pts, self.frame as i64, RATE);
        if self.packets == 0 {
            let head = OpusHead::simple(self.channels as u8, PRE_SKIP, RATE);
            packet.side.push(SideData::Config(head.to_ogg()));
        }
        self.packets += 1;
        Ok(packet)
    }
}

/// The coded bandwidth for a bitrate: narrower bands leave more bits per
/// band at low rates.
fn bandwidth_for(bitrate: u32, channels: usize) -> Bandwidth {
    // A stereo stream codes the second channel at about half the cost.
    let per_channel = if channels == 2 {
        bitrate * 2 / 3
    } else {
        bitrate
    };
    match per_channel {
        0..10_000 => Bandwidth::Narrow,
        10_000..14_000 => Bandwidth::Wide,
        14_000..18_000 => Bandwidth::SuperWide,
        _ => Bandwidth::Full,
    }
}

/// The input as interleaved floats, refusing a wrong rate or layout, a
/// sample count that does not split evenly into channels, and non-finite
/// or out-of-range samples. Nothing is consumed on error.
fn pcm_samples(channels: usize, input: &AudioInput) -> Result<Vec<f32>, CodecError> {
    let pcm = &input.pcm;
    if pcm.rate != RATE {
        return Err(CodecError::InvalidApi {
            what: "sample rate",
        });
    }
    if pcm.layout.len() != channels {
        return Err(CodecError::InvalidApi { what: "layout" });
    }
    let expected = pcm.samples as usize * channels;
    if let Some(samples) = pcm.as_f32() {
        if samples.len() != expected {
            return Err(CodecError::UnevenChannels);
        }
        let mut out = Vec::with_capacity(samples.len());
        for &sample in samples {
            if !sample.is_finite() {
                return Err(CodecError::NonFinite);
            }
            if sample.abs() > 1.0 + FULL_SCALE_SLACK {
                return Err(CodecError::OutOfRange);
            }
            out.push(sample.clamp(-1.0, 1.0));
        }
        return Ok(out);
    }
    if let Some(samples) = pcm.as_i16() {
        if samples.len() != expected {
            return Err(CodecError::UnevenChannels);
        }
        return Ok(samples.iter().map(|&s| f32::from(s) / 32768.0).collect());
    }
    Err(CodecError::InvalidApi {
        what: "sample format",
    })
}

impl Encoder for OpusEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        self.push_samples(input)
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let _ = input;
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        self.drain_samples()
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests;
