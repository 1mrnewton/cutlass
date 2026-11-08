//! The SILK layer (RFC 6716 section 4.2), with the resampler that brings
//! its 8, 12, or 16 kHz output to the API rate.
//!
//! This follows the RFC 6716 reference decoder (`silk/dec_API.c` and the
//! files each submodule names) in exact fixed point, with the RFC 8251
//! fixes: the stereo state reset on init (section 3), the resampler buffer
//! (section 5), and the overflow checks in the inverse prediction gain
//! (section 6) and NLSF stabilization (section 7). A corrupt frame never
//! errors, as in the reference: the range decoder reads zeros past the
//! end, and every buffer index derived from the bitstream is bounded.

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
#[cfg(test)]
mod tests;

use crate::range::RangeDecoder;
use channel::{ChannelState, FrameMode};
use cutlass_decode::CodecError;
use indices::{CondCoding, TYPE_NO_VOICE_ACTIVITY};
use pulses::MAX_FRAME_LENGTH;
use stereo::StereoState;
use tables::LBRR_FLAGS_ICDF;

/// Most SILK frames in one Opus frame (60 ms).
const MAX_FRAMES_PER_PACKET: usize = 3;
/// Most output samples per channel of one call: 20 ms at 48 kHz.
const MAX_API_FRAME: usize = 960;

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

/// SILK decoder state for one Opus stream (`silk_decoder`).
#[derive(Clone, Debug)]
pub(crate) struct SilkDecoder {
    channels: [ChannelState; 2],
    stereo: StereoState,
    n_channels_api: usize,
    n_channels_internal: usize,
    prev_decode_only_middle: bool,
    /// The configuration of the packet being decoded, so a call in the
    /// middle of a packet cannot change it.
    packet: Option<SilkControl>,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    pub(crate) fn new() -> Self {
        Self {
            channels: [ChannelState::new(), ChannelState::new()],
            stereo: StereoState::default(),
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: false,
            packet: None,
        }
    }

    /// Drops every channel's history, the stereo predictor, and the resampler
    /// (`silk_InitDecoder`, with the RFC 8251 section 3 stereo reset).
    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

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
        let (nb_subfr, frames_per_packet) = check_control(ctl)?;
        let starting = new_packet || self.channels[0].n_frames_decoded == 0;
        let nb_subfr = if starting {
            nb_subfr
        } else {
            if self.packet != Some(*ctl) {
                return Err(CodecError::InvalidApi {
                    what: "SILK configuration changed inside a packet",
                });
            }
            if self.channels[0].n_frames_decoded >= self.channels[0].n_frames_per_packet {
                return Err(CodecError::InvalidApi {
                    what: "SILK packet has no frames left",
                });
            }
            self.channels[0].nb_subfr
        };
        let api = ctl.api_channels;
        let internal = ctl.internal_channels;
        let fs_khz = (ctl.internal_rate / 1000) as i32;
        let n_samples_out = nb_subfr * 5 * (ctl.api_rate / 1000) as usize;
        if out.len() < n_samples_out * api {
            return Err(CodecError::InvalidApi {
                what: "SILK output buffer too small",
            });
        }
        if starting {
            self.packet = Some(*ctl);
        }

        let [ch0, ch1] = &mut self.channels;
        if new_packet {
            ch0.n_frames_decoded = 0;
            if internal == 2 {
                ch1.n_frames_decoded = 0;
            }
        }
        // Mono to stereo: the side channel starts fresh.
        if internal > self.n_channels_internal {
            *ch1 = ChannelState::new();
        }
        let stereo_to_mono = internal == 1 && self.n_channels_internal == 2 && fs_khz == ch0.fs_khz;
        if ch0.n_frames_decoded == 0 {
            for ch in [&mut *ch0, &mut *ch1].into_iter().take(internal) {
                ch.n_frames_per_packet = frames_per_packet;
                ch.nb_subfr = nb_subfr;
                ch.set_fs(fs_khz, ctl.api_rate as i32);
            }
        }
        if api == 2 && internal == 2 && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0; 2];
            self.stereo.s_side = [0; 2];
            ch1.resampler = ch0.resampler.clone();
        }
        self.n_channels_api = api;
        self.n_channels_internal = internal;

        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = false;
        if loss != SilkLoss::Lost && ch0.n_frames_decoded == 0 {
            // First call for this packet: VAD and LBRR flags.
            for ch in [&mut *ch0, &mut *ch1].into_iter().take(internal) {
                for i in 0..ch.n_frames_per_packet.min(MAX_FRAMES_PER_PACKET) {
                    ch.vad_flags[i] = rd.bit_logp(1);
                }
                ch.lbrr_flag = rd.bit_logp(1);
            }
            for ch in [&mut *ch0, &mut *ch1].into_iter().take(internal) {
                ch.lbrr_flags = [false; MAX_FRAMES_PER_PACKET];
                if ch.lbrr_flag {
                    if ch.n_frames_per_packet == 1 {
                        ch.lbrr_flags[0] = true;
                    } else {
                        let table = LBRR_FLAGS_ICDF[(ch.n_frames_per_packet - 2).min(1)];
                        let symbol = rd.dec_icdf(table, 8) + 1;
                        for (i, flag) in ch.lbrr_flags.iter_mut().enumerate() {
                            *flag = (symbol >> i) & 1 != 0;
                        }
                    }
                }
            }
            if loss == SilkLoss::Normal {
                // Skip the LBRR frames this packet carries for the previous one.
                for i in 0..ch0.n_frames_per_packet.min(MAX_FRAMES_PER_PACKET) {
                    for n in 0..internal {
                        let ch = if n == 0 { &mut *ch0 } else { &mut *ch1 };
                        if !ch.lbrr_flags[i] {
                            continue;
                        }
                        if internal == 2 && n == 0 {
                            ms_pred_q13 = stereo::decode_pred(rd);
                            if !ch1.lbrr_flags[i] {
                                decode_only_middle = stereo::decode_mid_only(rd);
                            }
                        }
                        let ch = if n == 0 { &mut *ch0 } else { &mut *ch1 };
                        let cond = if i > 0 && ch.lbrr_flags[i - 1] {
                            CondCoding::Conditionally
                        } else {
                            CondCoding::Independently
                        };
                        ch.skip_frame(rd, cond);
                    }
                }
            }
        }

        // Mid/side predictors for this frame.
        let frame = ch0.n_frames_decoded;
        let lbrr0 = ch0.lbrr_flags.get(frame).copied().unwrap_or(false);
        let lbrr1 = ch1.lbrr_flags.get(frame).copied().unwrap_or(false);
        if internal == 2 {
            if loss == SilkLoss::Normal || (loss == SilkLoss::Fec && lbrr0) {
                ms_pred_q13 = stereo::decode_pred(rd);
                let vad1 = ch1.vad_flags.get(frame).copied().unwrap_or(false);
                decode_only_middle =
                    if (loss == SilkLoss::Normal && !vad1) || (loss == SilkLoss::Fec && !lbrr1) {
                        stereo::decode_mid_only(rd)
                    } else {
                        false
                    };
            } else {
                ms_pred_q13 = self.stereo.pred_prev_q13.map(i32::from);
            }
        }

        // The first frame with side coding after mid-only frames starts the
        // side channel's prediction from silence.
        if internal == 2 && !decode_only_middle && self.prev_decode_only_middle {
            ch1.out_buf.fill(0);
            ch1.s_lpc_q14_buf = [0; nlsf::MAX_LPC_ORDER];
            ch1.lag_prev = 100;
            ch1.last_gain_index = 10;
            ch1.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            ch1.first_frame_after_reset = true;
        }

        let has_side = if loss == SilkLoss::Normal {
            !decode_only_middle
        } else {
            let side_lbrr = ch1
                .lbrr_flags
                .get(ch1.n_frames_decoded)
                .copied()
                .unwrap_or(false);
            !self.prev_decode_only_middle || (internal == 2 && loss == SilkLoss::Fec && side_lbrr)
        };
        let mode = match loss {
            SilkLoss::Normal => FrameMode::Normal,
            SilkLoss::Lost => FrameMode::Lost,
            SilkLoss::Fec => FrameMode::Lbrr,
        };

        // Two samples of history ahead of each channel's frame.
        let mut tmp = [[0i16; MAX_FRAME_LENGTH + 2]; 2];
        let mut n_out_dec = 0;
        for (n, buf) in tmp.iter_mut().enumerate().take(internal) {
            if n == 0 || has_side {
                let frame_index = ch0.n_frames_decoded as isize - n as isize;
                let ch = if n == 0 { &mut *ch0 } else { &mut *ch1 };
                let cond = if frame_index <= 0 {
                    CondCoding::Independently
                } else if loss == SilkLoss::Fec {
                    let prev = ch.lbrr_flags.get(frame_index as usize - 1).copied();
                    if prev.unwrap_or(false) {
                        CondCoding::Conditionally
                    } else {
                        CondCoding::Independently
                    }
                } else if n > 0 && self.prev_decode_only_middle {
                    // A skipped side frame needs no LTP scaling: its LTP
                    // state is well defined.
                    CondCoding::IndependentlyNoLtpScaling
                } else {
                    CondCoding::Conditionally
                };
                n_out_dec = ch.decode_frame(rd, &mut buf[2..], mode, cond);
            } else {
                buf[2..2 + n_out_dec].fill(0);
            }
            let ch = if n == 0 { &mut *ch0 } else { &mut *ch1 };
            ch.n_frames_decoded += 1;
        }

        let [tmp0, tmp1] = &mut tmp;
        if api == 2 && internal == 2 {
            stereo::ms_to_lr(
                &mut self.stereo,
                tmp0,
                tmp1,
                ms_pred_q13,
                ch0.fs_khz as usize,
                n_out_dec,
            );
        } else {
            tmp0[..2].copy_from_slice(&self.stereo.s_mid);
            self.stereo
                .s_mid
                .copy_from_slice(&tmp0[n_out_dec..n_out_dec + 2]);
        }

        let n_out = ch0.resampler.output_len(n_out_dec).min(n_samples_out);
        let mut resampled = [0i16; MAX_API_FRAME];
        for n in 0..api.min(internal) {
            let ch = if n == 0 { &mut *ch0 } else { &mut *ch1 };
            let input = if n == 0 {
                &tmp0[1..=n_out_dec]
            } else {
                &tmp1[1..=n_out_dec]
            };
            if api == 2 {
                ch.resampler.process(&mut resampled[..n_out], input);
                for (i, &v) in resampled[..n_out].iter().enumerate() {
                    out[n + 2 * i] = v;
                }
            } else {
                ch.resampler.process(&mut out[..n_out], input);
            }
        }
        if api == 2 && internal == 1 {
            if stereo_to_mono {
                // Keep the right resampler running across a stereo to mono
                // switch so switching back is seamless.
                ch1.resampler
                    .process(&mut resampled[..n_out], &tmp0[1..=n_out_dec]);
                for (i, &v) in resampled[..n_out].iter().enumerate() {
                    out[1 + 2 * i] = v;
                }
            } else {
                for i in 0..n_out {
                    out[1 + 2 * i] = out[2 * i];
                }
            }
        }

        if loss == SilkLoss::Lost {
            // Without the gain clamp the level cannot bounce back after a
            // loss during a fade.
            for ch in self.channels.iter_mut().take(self.n_channels_internal) {
                ch.last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        Ok(n_out)
    }
}

/// Validates a call's configuration: subframes per SILK frame and frames
/// per packet.
fn check_control(ctl: &SilkControl) -> Result<(usize, usize), CodecError> {
    if !matches!(ctl.api_channels, 1 | 2) || !matches!(ctl.internal_channels, 1 | 2) {
        return Err(CodecError::InvalidApi {
            what: "SILK channel count",
        });
    }
    if !matches!(ctl.api_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
        return Err(CodecError::InvalidApi {
            what: "SILK API rate",
        });
    }
    if !matches!(ctl.internal_rate, 8000 | 12000 | 16000) {
        return Err(CodecError::InvalidApi {
            what: "SILK internal rate",
        });
    }
    match ctl.payload_ms {
        // A lost packet of unknown length conceals 10 ms at a time.
        0 | 10 => Ok((2, 1)),
        20 => Ok((4, 1)),
        40 => Ok((4, 2)),
        60 => Ok((4, 3)),
        _ => Err(CodecError::InvalidApi {
            what: "SILK packet duration",
        }),
    }
}
