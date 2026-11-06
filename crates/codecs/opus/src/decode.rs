//! The Opus decoder behind [`cutlass_decode::Decoder`].
//!
//! One [`StreamDecoder`] is the RFC 6716 decoder for one elementary Opus
//! stream: it reads the TOC, runs SILK, CELT, or both (hybrid), decodes the
//! CELT redundancy frames that smooth a SILK/CELT switch, cross-fades a mode
//! transition, and conceals a lost frame. [`OpusDecoder`] holds one stream
//! decoder per stream an `OpusHead` names (RFC 7845 channel mapping family
//! 0, 1, or 255), splits each multistream packet with the self-delimited
//! framing of RFC 6716 appendix B, maps decoded channels to output
//! channels, and applies the header's output gain.
//!
//! Output is interleaved `f32` at full scale 1.0, unclipped, at the API
//! rate (48 kHz unless the caller asks for 8, 12, 16, or 24 kHz). Pre-skip
//! is not removed here: the container states it as the stream's first
//! edit, as an MP4 edit list does. A packet whose duration is shorter than
//! what it decodes to marks the excess as `skip_trailing` (end trimming).

use crate::celt::{CeltDecoder, CeltFrame, OVERLAP_WINDOW};
use crate::header::OpusHead;
use crate::packet::{self, Bandwidth, Mode};
use crate::range::RangeDecoder;
use crate::silk::{SilkControl, SilkDecoder, SilkLoss};
use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
};

#[cfg(test)]
mod tests;

/// Rates the decoder can produce directly (RFC 6716 section 4).
const API_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

/// RFC 6716 decoder state for one elementary stream (`OpusDecoder` in the
/// reference).
#[derive(Clone, Debug)]
struct StreamDecoder {
    /// Output channels of this stream: 2 for a coupled stream, else 1.
    channels: usize,
    rate: u32,
    silk: SilkDecoder,
    celt: CeltDecoder,
    silk_ctl: SilkControl,
    /// Mode of the last decoded frame. `None` before the first packet.
    prev_mode: Option<Mode>,
    /// The last frame ended with a SILK-to-CELT redundancy frame.
    prev_redundancy: bool,
    /// The current packet's TOC values.
    mode: Mode,
    bandwidth: Bandwidth,
    /// Samples per channel of one frame, at the API rate.
    frame_size: usize,
    stream_channels: usize,
    range_final: u32,
}

impl StreamDecoder {
    fn new(channels: usize, rate: u32) -> Result<Self, CodecError> {
        Ok(Self {
            channels,
            rate,
            silk: SilkDecoder::new(),
            celt: CeltDecoder::new(channels, rate)?,
            silk_ctl: SilkControl {
                api_channels: channels,
                internal_channels: channels,
                api_rate: rate,
                internal_rate: 16_000,
                payload_ms: 20,
            },
            prev_mode: None,
            prev_redundancy: false,
            mode: Mode::CeltOnly,
            bandwidth: Bandwidth::Full,
            frame_size: (rate / 400) as usize,
            stream_channels: channels,
            range_final: 0,
        })
    }

    fn reset(&mut self) {
        self.silk.reset();
        self.celt.reset();
        self.silk_ctl.internal_channels = self.channels;
        self.silk_ctl.internal_rate = 16_000;
        self.prev_mode = None;
        self.prev_redundancy = false;
        self.mode = Mode::CeltOnly;
        self.bandwidth = Bandwidth::Full;
        self.frame_size = (self.rate / 400) as usize;
        self.stream_channels = self.channels;
        self.range_final = 0;
    }

    fn f20(&self) -> usize {
        (self.rate / 50) as usize
    }

    /// 48 kHz samples per API-rate sample.
    fn downsample(&self) -> usize {
        (48_000 / self.rate) as usize
    }

    /// Takes the TOC values of a packet as the current configuration.
    fn take_toc(&mut self, toc: &packet::Toc) {
        self.mode = toc.mode;
        self.bandwidth = toc.bandwidth;
        self.frame_size = toc.frame_samples as usize / self.downsample();
        self.stream_channels = if toc.stereo { 2 } else { 1 };
    }

    /// Decodes one packet of this stream, appending its PCM to `out`.
    ///
    /// Returns the bytes the packet took (all of `data` unless
    /// `self_delimited`) and the samples per channel decoded.
    fn decode_packet(
        &mut self,
        data: &[u8],
        self_delimited: bool,
        out: &mut Vec<f32>,
    ) -> Result<(usize, usize), CodecError> {
        let frames = packet::parse(data, self_delimited)?;
        self.take_toc(&frames.toc);
        let frame_size = self.frame_size;
        let start = out.len();
        out.resize(
            start + frames.frames.len() * frame_size * self.channels,
            0.0,
        );
        let mut at = start;
        let mut samples = 0;
        for &(offset, len) in &frames.frames {
            let bytes = data
                .get(offset..offset + len)
                .ok_or(CodecError::Truncated {
                    at_bit: (offset as u64) * 8,
                    what: "frame data",
                })?;
            let done = self.decode_frame(Some(bytes), frame_size, false, &mut out[at..])?;
            at += done * self.channels;
            samples += done;
        }
        out.truncate(at);
        Ok((frames.consumed, samples))
    }

    /// Conceals `samples` per channel of lost audio (`opus_decode` with no data).
    fn conceal(&mut self, samples: usize, out: &mut Vec<f32>) -> Result<usize, CodecError> {
        let f2_5 = self.f20() / 8;
        let mut done = 0;
        while done < samples {
            let ask = (samples - done).min(self.f20()).max(f2_5);
            let start = out.len();
            out.resize(start + ask * self.channels, 0.0);
            let got = self.decode_frame(None, ask, false, &mut out[start..])?;
            out.truncate(start + got * self.channels);
            if got == 0 {
                break;
            }
            done += got;
        }
        Ok(done)
    }

    /// Recovers the packet before `data` from the redundancy `data`
    /// carries, concealing first whatever that redundancy does not cover.
    ///
    /// `samples` is the length of the gap at the API rate. A packet without
    /// SILK (CELT-only) carries no redundancy, so the whole gap is concealed.
    fn decode_fec(
        &mut self,
        data: &[u8],
        samples: usize,
        out: &mut Vec<f32>,
    ) -> Result<usize, CodecError> {
        let frames = packet::parse(data, false)?;
        let toc = frames.toc;
        let packet_frame = toc.frame_samples as usize / self.downsample();
        if samples < packet_frame
            || toc.mode == Mode::CeltOnly
            || self.prev_mode.is_none_or(|mode| mode == Mode::CeltOnly)
        {
            return self.conceal(samples, out);
        }
        let mut done = 0;
        if samples > packet_frame {
            done = self.conceal(samples - packet_frame, out)?;
        }
        self.take_toc(&toc);
        let Some(&(offset, len)) = frames.frames.first() else {
            return Ok(done);
        };
        let bytes = data
            .get(offset..offset + len)
            .ok_or(CodecError::Truncated {
                at_bit: (offset as u64) * 8,
                what: "frame data",
            })?;
        let start = out.len();
        out.resize(start + packet_frame * self.channels, 0.0);
        let got = self.decode_frame(Some(bytes), packet_frame, true, &mut out[start..])?;
        out.truncate(start + got * self.channels);
        Ok(done + got)
    }

    /// `opus_decode_frame` of RFC 6716 appendix A: one frame of `data`, or
    /// concealment when `data` is `None` or one byte or less.
    ///
    /// `pcm` holds at least `frame_size * channels` samples. Returns the
    /// samples per channel written.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        frame_size: usize,
        decode_fec: bool,
        pcm: &mut [f32],
    ) -> Result<usize, CodecError> {
        let ch = self.channels;
        let f20 = self.f20();
        let f10 = f20 / 2;
        let f5 = f10 / 2;
        let f2_5 = f5 / 2;
        if frame_size < f2_5 {
            return Err(CodecError::InvalidApi {
                what: "frame shorter than 2.5 ms",
            });
        }
        let mut frame_size = frame_size;
        // A payload of one byte or less (two with the TOC) is a lost or DTX
        // frame: conceal it, no longer than the TOC says.
        let data = data.filter(|bytes| bytes.len() > 1);
        if data.is_none() {
            frame_size = frame_size.min(self.frame_size.max(f2_5));
        }
        let (audiosize, mode) = match data {
            Some(_) => (self.frame_size, self.mode),
            None => match self.prev_mode {
                None => {
                    // Nothing decoded yet: all concealment can give is silence.
                    let fill = pcm
                        .get_mut(..frame_size * ch)
                        .ok_or(CodecError::InvalidApi {
                            what: "output buffer",
                        })?;
                    fill.fill(0.0);
                    return Ok(frame_size);
                }
                Some(mode) => (frame_size, mode),
            },
        };
        if pcm.len() < audiosize * ch {
            return Err(CodecError::InvalidApi {
                what: "output buffer",
            });
        }
        // CELT and hybrid concealment run 20 ms at a time.
        if data.is_none() && frame_size > f20 && mode != Mode::SilkOnly {
            let mut done = 0;
            while done < frame_size {
                let ask = (frame_size - done).min(f20);
                let got = self.decode_frame(None, ask, false, &mut pcm[done * ch..])?;
                if got != ask {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "concealment length",
                    });
                }
                done += got;
            }
            return Ok(frame_size);
        }

        let mut pcm_transition = vec![0f32; f5 * ch];
        let mut transition = false;
        if data.is_some()
            && let Some(prev) = self.prev_mode
            && ((mode == Mode::CeltOnly && prev != Mode::CeltOnly && !self.prev_redundancy)
                || (mode != Mode::CeltOnly && prev == Mode::CeltOnly))
        {
            transition = true;
            if mode == Mode::CeltOnly {
                self.decode_frame(None, f5.min(audiosize), false, &mut pcm_transition)?;
            }
        }
        if audiosize > frame_size {
            return Err(CodecError::InvalidApi {
                what: "output shorter than the frame",
            });
        }
        frame_size = audiosize;

        let empty: [u8; 0] = [];
        let mut dec = RangeDecoder::new(data.unwrap_or(&empty));
        let mut len = data.map_or(0, |bytes| bytes.len() as i32);

        // SILK.
        let mut pcm_silk = vec![0i16; f10.max(frame_size) * ch];
        if mode != Mode::CeltOnly {
            if self.prev_mode == Some(Mode::CeltOnly) {
                self.silk.reset();
            }
            // SILK concealment cannot make less than 10 ms.
            self.silk_ctl.payload_ms = (1000 * audiosize as u32 / self.rate).max(10);
            if data.is_some() {
                self.silk_ctl.internal_channels = self.stream_channels;
                self.silk_ctl.internal_rate = if mode == Mode::SilkOnly {
                    self.bandwidth.silk_rate()
                } else {
                    16_000
                };
            }
            let loss = if data.is_none() {
                SilkLoss::Lost
            } else if decode_fec {
                SilkLoss::Fec
            } else {
                SilkLoss::Normal
            };
            let mut decoded = 0;
            while decoded < frame_size {
                let slot = &mut pcm_silk[decoded * ch..];
                let got = match self
                    .silk
                    .decode(&self.silk_ctl, loss, decoded == 0, &mut dec, slot)
                {
                    Ok(got) if got > 0 => got,
                    Ok(_) => {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "empty SILK frame",
                        });
                    }
                    Err(error) => {
                        if loss == SilkLoss::Normal {
                            return Err(error);
                        }
                        // Concealment failing is not fatal.
                        slot.fill(0);
                        frame_size - decoded
                    }
                };
                decoded += got;
            }
        }

        // Redundancy: a CELT frame at the end of a SILK or hybrid frame.
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut redundancy_bytes = 0i32;
        if !decode_fec
            && mode != Mode::CeltOnly
            && data.is_some()
            && dec.tell() + 17 + 20 * i32::from(self.mode == Mode::Hybrid) <= 8 * len
        {
            redundancy = if mode == Mode::Hybrid {
                dec.bit_logp(12)
            } else {
                true
            };
            if redundancy {
                celt_to_silk = dec.bit_logp(1);
                redundancy_bytes = if mode == Mode::Hybrid {
                    dec.dec_uint(256) as i32 + 2
                } else {
                    len - ((dec.tell() + 7) >> 3)
                };
                len -= redundancy_bytes;
                // Not reachable by a valid packet; RFC 6716 leaves it open.
                if len * 8 < dec.tell() {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                }
                dec.shrink_storage(redundancy_bytes.max(0) as u32);
            }
        }
        let start_band = if mode == Mode::CeltOnly { 0 } else { 17 };
        let end_band = self.bandwidth.celt_end_band();
        if redundancy {
            transition = false;
        }
        if transition && mode != Mode::CeltOnly {
            self.decode_frame(None, f5.min(audiosize), false, &mut pcm_transition)?;
        }
        let redundant_data = match data {
            Some(bytes) if redundancy => {
                let from = len.max(0) as usize;
                let to = from + redundancy_bytes.max(0) as usize;
                bytes.get(from..to)
            }
            _ => None,
        };

        // 5 ms redundant frame for a CELT-to-SILK switch.
        let mut redundant_audio = vec![0f32; f5 * ch];
        let mut redundant_rng = 0u32;
        if redundancy && celt_to_silk {
            redundant_rng =
                self.decode_redundant(redundant_data, end_band, &mut redundant_audio)?;
        }

        if mode != Mode::SilkOnly {
            let celt_frame = f20.min(frame_size);
            if Some(mode) != self.prev_mode && self.prev_mode.is_some() && !self.prev_redundancy {
                self.celt.reset();
            }
            let frame = CeltFrame {
                rd: if decode_fec || len <= 1 {
                    None
                } else {
                    Some(&mut dec)
                },
                frame_size: celt_frame * self.downsample(),
                stream_channels: self.stream_channels,
                start_band,
                end_band,
            };
            self.celt.decode(frame, pcm)?;
        } else {
            pcm[..frame_size * ch].fill(0.0);
            // A hybrid-to-SILK switch lets the CELT MDCT fade out by
            // decoding a silence frame.
            if self.prev_mode == Some(Mode::Hybrid)
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                let silence = [0xFFu8, 0xFF];
                let mut quiet = RangeDecoder::new(&silence);
                let frame = CeltFrame {
                    rd: Some(&mut quiet),
                    frame_size: 120,
                    stream_channels: self.stream_channels,
                    start_band: 0,
                    end_band,
                };
                self.celt.decode(frame, pcm)?;
            }
        }

        if mode != Mode::CeltOnly {
            for (out, &silk) in pcm[..frame_size * ch].iter_mut().zip(&pcm_silk) {
                *out += f32::from(silk) * (1.0 / 32768.0);
            }
        }

        // 5 ms redundant frame for a SILK-to-CELT switch.
        if redundancy && !celt_to_silk {
            self.celt.reset();
            redundant_rng =
                self.decode_redundant(redundant_data, end_band, &mut redundant_audio)?;
            let at = ch * (frame_size - f2_5);
            let faded = smooth_fade(
                &pcm[at..],
                &redundant_audio[ch * f2_5..],
                f2_5,
                ch,
                self.downsample(),
            );
            pcm[at..at + faded.len()].copy_from_slice(&faded);
        }
        if redundancy && celt_to_silk {
            pcm[..ch * f2_5].copy_from_slice(&redundant_audio[..ch * f2_5]);
            let at = ch * f2_5;
            let faded = smooth_fade(
                &redundant_audio[at..],
                &pcm[at..],
                f2_5,
                ch,
                self.downsample(),
            );
            pcm[at..at + faded.len()].copy_from_slice(&faded);
        }
        if transition {
            if audiosize >= f5 {
                pcm[..ch * f2_5].copy_from_slice(&pcm_transition[..ch * f2_5]);
                let at = ch * f2_5;
                let faded = smooth_fade(
                    &pcm_transition[at..],
                    &pcm[at..],
                    f2_5,
                    ch,
                    self.downsample(),
                );
                pcm[at..at + faded.len()].copy_from_slice(&faded);
            } else {
                // Too short for a clean transition; fade anyway.
                let faded = smooth_fade(&pcm_transition, pcm, f2_5, ch, self.downsample());
                pcm[..faded.len()].copy_from_slice(&faded);
            }
        }
        self.range_final = if len <= 1 {
            0
        } else {
            dec.rng() ^ redundant_rng
        };
        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(audiosize)
    }

    /// Decodes the 5 ms redundancy frame over the whole band and returns
    /// its final range.
    fn decode_redundant(
        &mut self,
        bytes: Option<&[u8]>,
        end_band: usize,
        out: &mut [f32],
    ) -> Result<u32, CodecError> {
        let bytes = bytes.unwrap_or(&[]);
        let mut rd = RangeDecoder::new(bytes);
        let frame = CeltFrame {
            rd: Some(&mut rd),
            frame_size: 240,
            stream_channels: self.stream_channels,
            start_band: 0,
            end_band,
        };
        self.celt.decode(frame, out)?;
        Ok(if bytes.len() > 1 { rd.rng() } else { 0 })
    }
}

/// Cross-fades `in1` into `in2` over `overlap` samples per channel with the
/// squared CELT window (`smooth_fade` of the reference).
fn smooth_fade(in1: &[f32], in2: &[f32], overlap: usize, channels: usize, inc: usize) -> Vec<f32> {
    let mut out = vec![0f32; overlap * channels];
    for c in 0..channels {
        for i in 0..overlap {
            let w = OVERLAP_WINDOW.get(i * inc).copied().unwrap_or(1.0);
            let w = w * w;
            let at = i * channels + c;
            let a = in1.get(at).copied().unwrap_or(0.0);
            let b = in2.get(at).copied().unwrap_or(0.0);
            out[at] = w * b + (1.0 - w) * a;
        }
    }
    out
}

/// One output channel: the stream it is taken from, and that stream's channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Silent,
    Stream { stream: usize, channel: usize },
}

/// An Opus decoder for one `OpusHead` configuration: one stream for
/// family 0, several for families 1 and 255.
#[derive(Clone, Debug)]
pub struct OpusDecoder {
    head: OpusHead,
    rate: u32,
    layout: ChannelLayout,
    streams: Vec<StreamDecoder>,
    sources: Vec<Source>,
    gain: f32,
}

impl OpusDecoder {
    /// Opens a decoder for `params`.
    ///
    /// `params.extra` is an `OpusHead` packet or a `dOps` box body. With no
    /// configuration, the layout's one or two channels make a family 0
    /// header. `params.sample_rate` picks the output rate when it is 8, 12,
    /// 16, 24, or 48 kHz; any other rate decodes at 48 kHz, the rate Opus
    /// timestamps count in.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Opus {
            return Err(CodecError::Unsupported {
                what: "codec is not Opus",
            });
        }
        let head = if params.extra.is_empty() {
            let channels = params.layout.as_ref().map_or(0, ChannelLayout::len);
            if !(1..=2).contains(&channels) {
                return Err(CodecError::Unsupported {
                    what: "Opus without OpusHead needs one or two channels",
                });
            }
            OpusHead::simple(channels as u8, 0, 48_000)
        } else {
            OpusHead::parse(&params.extra)?
        };
        let rate = if API_RATES.contains(&params.sample_rate) {
            params.sample_rate
        } else {
            48_000
        };
        Self::with_head(head, rate)
    }

    /// A decoder for `head` producing `rate` (8, 12, 16, 24, or 48 kHz).
    pub fn with_head(head: OpusHead, rate: u32) -> Result<Self, CodecError> {
        head.validate()?;
        if !API_RATES.contains(&rate) {
            return Err(CodecError::Unsupported {
                what: "Opus output rate",
            });
        }
        let layout = head.layout()?;
        let coupled = usize::from(head.coupled);
        let mut streams = Vec::with_capacity(usize::from(head.streams));
        for index in 0..usize::from(head.streams) {
            let channels = if index < coupled { 2 } else { 1 };
            streams.push(StreamDecoder::new(channels, rate)?);
        }
        let sources = head
            .mapping
            .iter()
            .map(|&index| {
                let index = usize::from(index);
                if index == 255 {
                    Source::Silent
                } else if index < 2 * coupled {
                    Source::Stream {
                        stream: index / 2,
                        channel: index % 2,
                    }
                } else {
                    Source::Stream {
                        stream: index - coupled,
                        channel: 0,
                    }
                }
            })
            .collect();
        // Q7.8 dB to a linear factor: 10^(gain / (20 * 256)).
        let gain = 10f32.powf(f32::from(head.output_gain) / (20.0 * 256.0));
        Ok(Self {
            head,
            rate,
            layout,
            streams,
            sources,
            gain,
        })
    }

    /// The configuration this decoder was opened with.
    #[must_use]
    pub fn head(&self) -> &OpusHead {
        &self.head
    }

    /// Output sample rate.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The range coder state after the last packet, XORed over the streams,
    /// as the reference decoder's `OPUS_GET_FINAL_RANGE` reports it.
    #[must_use]
    pub fn final_range(&self) -> u32 {
        self.streams
            .iter()
            .fold(0, |acc, stream| acc ^ stream.range_final)
    }

    /// Decodes one packet into interleaved PCM at the output rate.
    ///
    /// On error the decoder is left as it was before the call.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>, CodecError> {
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty Opus packet",
            });
        }
        let saved = self.streams.clone();
        let count = self.streams.len();
        let mut offset = 0usize;
        let result = self.run_streams(|index, stream, out| {
            let rest = data.get(offset..).unwrap_or(&[]);
            if rest.is_empty() {
                return Err(CodecError::Truncated {
                    at_bit: (offset as u64) * 8,
                    what: "multistream packet",
                });
            }
            let delimited = index + 1 < count;
            let (consumed, samples) = stream.decode_packet(rest, delimited, out)?;
            offset += consumed;
            Ok(samples)
        });
        if result.is_err() {
            self.streams = saved;
        }
        result
    }

    /// Conceals `samples` per channel (at the output rate) of a lost packet.
    ///
    /// This is the reference decoder's packet-loss concealment, not a
    /// faithful decode. `samples` is a multiple of 2.5 ms up to 120 ms.
    pub fn conceal(&mut self, samples: usize) -> Result<Vec<f32>, CodecError> {
        self.check_gap(samples)?;
        let saved = self.streams.clone();
        let result = self.run_streams(|_, stream, out| stream.conceal(samples, out));
        if result.is_err() {
            self.streams = saved;
        }
        result
    }

    /// Recovers `samples` per channel lost just before `data` from the
    /// low-bitrate redundancy (in-band FEC) that `data` carries, concealing
    /// whatever it does not cover. Decode `data` itself with
    /// [`Self::decode`] afterwards.
    pub fn decode_fec(&mut self, data: &[u8], samples: usize) -> Result<Vec<f32>, CodecError> {
        self.check_gap(samples)?;
        if data.is_empty() {
            return self.conceal(samples);
        }
        let saved = self.streams.clone();
        let count = self.streams.len();
        let mut offset = 0usize;
        let result = self.run_streams(|index, stream, out| {
            let rest = data.get(offset..).unwrap_or(&[]);
            if index + 1 < count {
                let frames = packet::parse(rest, true)?;
                let one = rest.get(..frames.consumed).unwrap_or(&[]);
                offset += frames.consumed;
                stream.decode_fec(&undelimit(one)?, samples, out)
            } else {
                stream.decode_fec(rest, samples, out)
            }
        });
        if result.is_err() {
            self.streams = saved;
        }
        result
    }

    /// Concealment for a lost packet as a decoder output, marked `concealed`.
    pub fn conceal_output(&mut self, samples: usize) -> Result<Output, CodecError> {
        let pcm = self.conceal(samples)?;
        self.output(pcm, None, true)
    }

    fn check_gap(&self, samples: usize) -> Result<(), CodecError> {
        let step = (self.rate / 400) as usize;
        let most = (packet::MAX_PACKET_SAMPLES as usize) * (self.rate as usize) / 48_000;
        if samples == 0 || !samples.is_multiple_of(step) || samples > most {
            return Err(CodecError::InvalidApi {
                what: "concealment length is not 2.5 ms steps up to 120 ms",
            });
        }
        Ok(())
    }

    /// Runs `step` on every stream into its own buffer, then maps channels.
    fn run_streams<F>(&mut self, mut step: F) -> Result<Vec<f32>, CodecError>
    where
        F: FnMut(usize, &mut StreamDecoder, &mut Vec<f32>) -> Result<usize, CodecError>,
    {
        let mut buffers = Vec::with_capacity(self.streams.len());
        let mut samples = None;
        for (index, stream) in self.streams.iter_mut().enumerate() {
            let mut out = Vec::new();
            let got = step(index, stream, &mut out)?;
            if samples.is_some_and(|known| known != got) {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "streams of one packet differ in duration",
                });
            }
            samples = Some(got);
            buffers.push((stream.channels, out));
        }
        let samples = samples.unwrap_or(0);
        let channels = self.sources.len();
        let mut pcm = vec![0f32; samples * channels];
        for (c, source) in self.sources.iter().enumerate() {
            let Source::Stream { stream, channel } = *source else {
                continue;
            };
            let Some((width, buffer)) = buffers.get(stream) else {
                continue;
            };
            for i in 0..samples {
                pcm[i * channels + c] =
                    buffer.get(i * width + channel).copied().unwrap_or(0.0) * self.gain;
            }
        }
        Ok(pcm)
    }

    fn output(
        &self,
        samples: Vec<f32>,
        packet: Option<&Packet>,
        concealed: bool,
    ) -> Result<Output, CodecError> {
        let channels = self.layout.len().max(1);
        let count = samples.len() / channels;
        let (pts, timescale, trailing) = match packet {
            Some(packet) => {
                // A packet stamped shorter than it decodes is an end trim.
                let trailing = if packet.timescale > 0 && packet.duration > 0 {
                    let wanted = i128::from(packet.duration) * i128::from(self.rate)
                        / i128::from(packet.timescale);
                    (count as i128 - wanted).clamp(0, count as i128) as u32
                } else {
                    0
                };
                (packet.pts, packet.timescale, trailing)
            }
            None => (0, self.rate, 0),
        };
        let pcm = Pcm::from_f32(self.rate, self.layout.clone(), samples)
            .map_err(|_| CodecError::Limit { what: "PCM buffer" })?;
        Ok(Output::Audio(AudioOutput {
            pcm,
            pts,
            timescale,
            skip_leading: 0,
            skip_trailing: trailing,
            concealed,
        }))
    }
}

/// The plain (not self-delimited) form of one self-delimited packet.
fn undelimit(delimited: &[u8]) -> Result<Vec<u8>, CodecError> {
    let frames = packet::parse(delimited, true)?;
    let mut parts = Vec::with_capacity(frames.frames.len());
    for &(offset, len) in &frames.frames {
        parts.push(delimited.get(offset..offset + len).unwrap_or(&[]));
    }
    let config = delimited.first().map_or(0, |toc| toc >> 3);
    packet::pack(config, frames.toc.stereo, &parts, None, false)
}

impl Decoder for OpusDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let samples = self.decode(&packet.data)?;
        Ok(vec![self.output(samples, Some(packet), false)?])
    }

    /// Opus output is final when its packet decodes; nothing is held back.
    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        for stream in &mut self.streams {
            stream.reset();
        }
    }
}
