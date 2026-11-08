//! The Vorbis encoder behind [`cutlass_encode::Encoder`].
//!
//! Block 0 is centered on input sample 0, so its left half reads as silence
//! and the first packet has duration 0. Packet `k` then returns the samples
//! between the centers of blocks `k - 1` and `k`. Output starts at input
//! sample 0. `drain` shortens the last packet to the input length and marks
//! the unused tail as [`SideData::Gapless`] trailing padding.
//!
//! A block whose window is digital silence is an unused floor and no
//! residue: that is the same time signal as encoding the zero transform.
//! Anything else is a windowed MDCT, a masking floor lifted so the residue
//! stays in the books, square-polar coupling, and a residue cascade.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet,
    SideData,
};

use crate::bits::BitWriter;
use crate::decode::vorbis_layout;
use crate::encoder::floorfit::{self, CodedFloor};
use crate::encoder::profile::Profile;
use crate::encoder::psy::{self, Bands};
use crate::encoder::quant::{self, Normalize, Pair, PairScratch};
use crate::encoder::resenc;
use crate::header::{Comments, Headers};
use crate::mdct::{Mdct, Scratch as MdctScratch};
use crate::window::Windows;

/// How far past full scale a float sample may be before it is refused.
const FULL_SCALE_SLACK: f32 = 16.0 * f32::EPSILON;
/// Default quality when the caller names neither a quality nor a bitrate.
const DEFAULT_QUALITY: f32 = 0.4;

/// Settings [`CodecParams`] does not carry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// `0.0` spends the least, `1.0` the most. Out of range is clamped.
    pub quality: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            quality: DEFAULT_QUALITY,
        }
    }
}

/// Fixed choices derived from the quality, the rate, and the channel count.
struct Tune {
    mask_shift: f32,
    ath_shift: f32,
    tolerance: i32,
    /// Largest residue magnitude aimed at for a bin, before the rate loop.
    headroom: f32,
    point_hz: f64,
    norm_hz: f64,
    norm_threshold: f32,
    bitrate: Option<u32>,
    constant: bool,
}

/// Samples and block decisions. Cloned at the start of a push so a failure
/// leaves the encoder where it was.
#[derive(Clone, Debug)]
struct State {
    pending: Vec<f32>,
    /// Input sample index of `pending[0]`.
    origin: u64,
    accepted: u64,
    /// Decided blocks not yet packetized: long flag and center, in samples.
    planned: VecDeque<(bool, i64)>,
    /// The last decided block, kept after it is packetized so the next
    /// center has a previous one.
    tail: Option<(bool, i64)>,
    previous_long: Option<bool>,
    /// First sample the next packet covers.
    covered: i64,
    emitted: u64,
    bits: u64,
    rated: u64,
    shift_extra: f32,
    drained: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            origin: 0,
            accepted: 0,
            planned: VecDeque::new(),
            tail: None,
            previous_long: None,
            covered: 0,
            emitted: 0,
            bits: 0,
            rated: 0,
            shift_extra: 0.0,
            drained: false,
        }
    }
}

/// Reused per block. Not part of [`State`]: a failed push discards it.
struct Scratch {
    time: Vec<f32>,
    spectrum: Vec<Vec<f32>>,
    mask: Vec<f32>,
    target: Vec<f32>,
    weight: Vec<f32>,
    curve: Vec<Vec<f32>>,
    floors: Vec<CodedFloor>,
    silent: Vec<bool>,
    residue: Vec<Vec<i32>>,
    ratio: Vec<f32>,
    mdct: MdctScratch,
    psy: psy::Scratch,
    pair: PairScratch,
}

impl Scratch {
    fn new(channels: usize) -> Self {
        Self {
            time: Vec::new(),
            spectrum: vec![Vec::new(); channels],
            mask: Vec::new(),
            target: Vec::new(),
            weight: Vec::new(),
            curve: vec![Vec::new(); channels],
            floors: vec![CodedFloor::default(); channels],
            silent: vec![false; channels],
            residue: Vec::new(),
            ratio: Vec::new(),
            mdct: MdctScratch::default(),
            psy: psy::Scratch::default(),
            pair: PairScratch::default(),
        }
    }
}

/// A Vorbis encoder for one stream of 1 to 8 channels.
pub struct VorbisEncoder {
    cancel: Arc<AtomicBool>,
    profile: Profile,
    windows: Windows,
    mdct: [Mdct; 2],
    bands: [Bands; 2],
    layout: ChannelLayout,
    tune: Tune,
    config: Vec<u8>,
    state: State,
    scratch: Scratch,
}

impl VorbisEncoder {
    /// Opens an encoder for `params`.
    ///
    /// Vorbis, 8 to 192 kHz, 1 to 8 channels in the Vorbis channel order.
    /// Bitrate 0 uses quality 0.4. Any other bitrate is the average-rate
    /// target, and the starting quality follows it. `constant_rate` tightens
    /// that loop.
    ///
    /// # Errors
    ///
    /// `Unsupported` for another codec, a rate outside 8 to 192 kHz, or a
    /// channel count this encoder does not write. `InvalidApi` when the
    /// layout is missing or is not the Vorbis order for its channel count.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let (channels, rate, _) = prepare(params)?;
        let quality = if params.bitrate == 0 {
            DEFAULT_QUALITY
        } else {
            quality_for_bitrate(channels, rate, params.bitrate)
        };
        Self::open_with(params, Options { quality })
    }

    /// [`Self::open`] at an explicit quality. A bitrate in `params` is still
    /// the average-rate target; the quality is where the rate loop starts.
    ///
    /// # Errors
    ///
    /// As [`Self::open`], and `InvalidApi` when `options.quality` is not finite.
    pub fn open_with(params: &CodecParams, options: Options) -> Result<Self, CodecError> {
        if !options.quality.is_finite() {
            return Err(CodecError::InvalidApi { what: "quality" });
        }
        let (channels, rate, layout) = prepare(params)?;
        let quality = options.quality.clamp(0.0, 1.0);
        let tune = Tune::new(
            channels,
            rate,
            quality,
            params.bitrate,
            params.constant_rate,
        );
        let profile = Profile::new(channels, rate, tune.lowpass(rate))?;
        let config = headers(
            &profile,
            tune.nominal(channels, rate, quality, params.bitrate),
        )?;
        let blocks = profile.blocksizes;
        Ok(Self {
            cancel: Arc::new(AtomicBool::new(false)),
            windows: Windows::new(blocks),
            mdct: [Mdct::new(blocks[0]), Mdct::new(blocks[1])],
            bands: [
                Bands::new(blocks[0] / 2, rate),
                Bands::new(blocks[1] / 2, rate),
            ],
            scratch: Scratch::new(channels),
            profile,
            layout,
            tune,
            config,
            state: State::default(),
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
        let samples = pcm_samples(&self.layout, self.profile.rate, input)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let saved = self.state.clone();
        let mut packets = Vec::new();
        for chunk in samples.chunks(4_096 * self.profile.channels) {
            self.state.pending.extend_from_slice(chunk);
            self.state.accepted += (chunk.len() / self.profile.channels) as u64;
            match self.pump(false) {
                Ok(more) => packets.extend(more),
                Err(error) => {
                    self.state = saved;
                    return Err(error);
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
        match self.pump(true) {
            Ok(packets) => {
                self.state.drained = true;
                Ok(packets)
            }
            Err(error) => {
                self.state = saved;
                Err(error)
            }
        }
    }

    /// Decides every block the buffer can support and packetizes each one
    /// whose next block size is known. `eos` finishes the last block.
    fn pump(&mut self, eos: bool) -> Result<Vec<Packet>, CodecError> {
        self.plan(eos)?;
        let mut packets = Vec::new();
        loop {
            if self.cancelled() {
                return Err(CodecError::Cancelled);
            }
            let ready = self.state.planned.len() >= 2 || (eos && !self.state.planned.is_empty());
            if !ready {
                break;
            }
            let last = eos && self.state.planned.len() == 1;
            packets.push(self.encode_front(last)?);
            self.discard();
        }
        Ok(packets)
    }

    fn plan(&mut self, eos: bool) -> Result<(), CodecError> {
        let long_n = self.profile.blocksizes[1];
        let short_n = self.profile.blocksizes[0];
        let accepted = i64::try_from(self.state.accepted).map_err(|_| CodecError::Limit {
            what: "Vorbis timestamp",
        })?;
        loop {
            let covered = self.state.tail.map_or(0, |(_, center)| center);
            if self.state.accepted == 0 || (eos && self.state.tail.is_some() && covered >= accepted)
            {
                break;
            }
            let center_if_long = self.next_center(long_n)?;
            let need = center_if_long + long_n as i64 / 2;
            if !eos && accepted < need {
                break;
            }
            let transient = self.transient(
                center_if_long - long_n as i64 / 4,
                center_if_long + long_n as i64 / 4,
            );
            let long = !transient;
            let n = if long { long_n } else { short_n };
            let center = if long {
                center_if_long
            } else {
                self.next_center(n)?
            };
            if self.state.planned.len() > 8_000_000 {
                return Err(CodecError::Limit {
                    what: "Vorbis block plan",
                });
            }
            self.state.planned.push_back((long, center));
            self.state.tail = Some((long, center));
        }
        Ok(())
    }

    /// Center of a new block of `n` samples given the blocks already decided.
    fn next_center(&self, n: usize) -> Result<i64, CodecError> {
        let Some((previous_long, previous)) = self.state.tail else {
            return Ok(0);
        };
        let previous_n = self.profile.blocksizes[usize::from(previous_long)];
        let step = (previous_n / 4 + n / 4) as i64;
        previous.checked_add(step).ok_or(CodecError::Limit {
            what: "Vorbis timestamp",
        })
    }

    /// A jump or a drop of 6 dB across a short frame, loud enough to matter.
    fn transient(&self, start: i64, end: i64) -> bool {
        let accepted = self.state.accepted as i64;
        let frame = self.profile.blocksizes[0] as i64;
        let end = end.min(accepted);
        let mut at = start.max(0);
        if at + frame * 2 > end {
            return false;
        }
        let mut previous = self.frame_energy(at, frame);
        at += frame;
        while at + frame <= end {
            let energy = self.frame_energy(at, frame);
            let rise = energy > previous * 4.0 + 1e-4 && energy > 1e-3;
            let fall = previous > energy * 4.0 + 1e-4 && previous > 1e-3;
            if rise || fall {
                return true;
            }
            previous = energy;
            at += frame;
        }
        false
    }

    fn frame_energy(&self, start: i64, len: i64) -> f32 {
        let mut loudest = 0.0_f32;
        for channel in 0..self.profile.channels {
            let mut energy = 0.0_f32;
            for i in 0..len {
                let sample = self.at(channel, start + i);
                energy += sample * sample;
            }
            loudest = loudest.max(energy / len as f32);
        }
        loudest
    }

    fn at(&self, channel: usize, sample: i64) -> f32 {
        if sample < 0 {
            return 0.0;
        }
        let Ok(sample) = u64::try_from(sample) else {
            return 0.0;
        };
        if sample >= self.state.accepted || sample < self.state.origin {
            return 0.0;
        }
        let frame = (sample - self.state.origin) as usize;
        let index = frame * self.profile.channels + channel;
        self.state.pending.get(index).copied().unwrap_or(0.0)
    }

    fn encode_front(&mut self, last: bool) -> Result<Packet, CodecError> {
        let (long, center) = self.state.planned[0];
        let next_long = self.state.planned.get(1).is_some_and(|&(long, _)| long);
        let previous_long = self.state.previous_long.unwrap_or(false);
        let n = self.profile.blocksizes[usize::from(long)];
        let half = n / 2;
        let shape = self.windows.shape(long, previous_long, next_long);
        let channels = self.profile.channels;
        self.scratch.time.resize(n, 0.0);
        self.scratch.silent.resize(channels, false);
        for channel in 0..channels {
            let start = center - n as i64 / 2;
            for i in 0..n {
                self.scratch.time[i] = self.at(channel, start + i as i64);
            }
            self.windows.apply(&shape, &mut self.scratch.time);
            let silent = self.scratch.time.iter().all(|&sample| sample == 0.0);
            self.scratch.silent[channel] = silent;
            let spectrum = &mut self.scratch.spectrum[channel];
            spectrum.resize(half, 0.0);
            if silent {
                spectrum.fill(0.0);
            } else {
                self.mdct[usize::from(long)].forward(
                    &self.scratch.time,
                    spectrum,
                    &mut self.scratch.mdct,
                );
                let scale = 4.0 / n as f32;
                for value in spectrum.iter_mut() {
                    *value *= scale;
                }
            }
        }
        let mut writer = BitWriter::new();
        writer.write_bit(false);
        writer.write(u32::from(long), 1);
        if long {
            writer.write_bit(previous_long);
            writer.write_bit(next_long);
        }
        if self.scratch.silent.iter().all(|&silent| silent) {
            for _ in 0..channels {
                writer.write_bit(false);
            }
        } else {
            self.analyze(long)?;
            for channel in 0..channels {
                if self.scratch.silent[channel] {
                    writer.write_bit(false);
                } else {
                    floorfit::write(
                        &self.profile.floors[usize::from(long)],
                        &self.profile.coders,
                        &self.scratch.floors[channel],
                        &mut writer,
                    );
                }
            }
            self.write_residues(long, &mut writer)?;
        }
        let natural = center - self.state.covered;
        let accepted = i64::try_from(self.state.accepted).map_err(|_| CodecError::Limit {
            what: "Vorbis timestamp",
        })?;
        let (duration, padding) = if last && center > accepted {
            let duration = (accepted - self.state.covered).max(0);
            let padding = u32::try_from(center - accepted).map_err(|_| CodecError::Limit {
                what: "Vorbis trailing padding",
            })?;
            (duration, padding)
        } else {
            (natural, 0)
        };
        let mut packet = Packet::audio(
            writer.finish(),
            self.state.covered,
            duration,
            self.profile.rate,
        );
        if self.state.emitted == 0 {
            packet.side.push(SideData::Config(self.config.clone()));
        }
        if last {
            packet.side.push(SideData::Gapless {
                encoder_delay: 0,
                trailing_padding: padding,
                sample_rate: self.profile.rate,
            });
        }
        self.account(packet.data.len(), duration)?;
        self.state.covered = center;
        self.state.previous_long = Some(long);
        self.state.emitted += 1;
        self.state.planned.pop_front();
        Ok(packet)
    }

    /// Floors for the block in `scratch`, refit once where the curve would
    /// push a bin past the residue range.
    fn analyze(&mut self, long: bool) -> Result<(), CodecError> {
        let size = usize::from(long);
        let half = self.profile.blocksizes[size] / 2;
        let bandwidth = self.profile.bandwidth[size].min(half);
        let shift = (self.tune.mask_shift + self.state.shift_extra).clamp(-6.0, 24.0);
        let headroom = (self.tune.headroom
            * if self.state.shift_extra < 0.0 {
                2.0_f32.powf(self.state.shift_extra / 6.0)
            } else {
                1.0
            })
        .clamp(4.0, 160.0);
        let tolerance = (self.tune.tolerance
            + if self.state.shift_extra < 0.0 {
                (-self.state.shift_extra / 3.0) as i32
            } else {
                0
            })
        .min(6);
        let (low, step) = floorfit::y_scale();
        self.scratch.mask.resize(half, 0.0);
        self.scratch.target.resize(half, 0.0);
        self.scratch.weight.resize(half, 0.0);
        for channel in 0..self.profile.channels {
            if self.scratch.silent[channel] {
                continue;
            }
            let spectrum = &self.scratch.spectrum[channel];
            self.bands[size].mask(
                spectrum,
                shift,
                self.tune.ath_shift,
                &mut self.scratch.psy,
                &mut self.scratch.mask,
            );
            fill_target(
                spectrum,
                &self.scratch.mask,
                bandwidth,
                headroom,
                &mut self.scratch.target,
                &mut self.scratch.weight,
            );
            self.scratch.floors[channel] = floorfit::fit(
                &self.profile.floors[size],
                &self.scratch.target,
                &self.scratch.weight,
                tolerance,
            );
            let curve = &mut self.scratch.curve[channel];
            curve.resize(half, 0.0);
            floorfit::curve(
                &self.profile.floors[size],
                &self.scratch.floors[channel],
                curve,
            );
            let mut lifted = false;
            for bin in 0..bandwidth {
                let need = spectrum[bin].abs() / headroom;
                if need > curve[bin] * 1.25 {
                    let db = 20.0 * need.max(1e-12).log10();
                    self.scratch.target[bin] = ((db - low) / step).clamp(0.0, 127.0);
                    self.scratch.weight[bin] = 32.0;
                    lifted = true;
                }
            }
            if lifted {
                self.scratch.floors[channel] = floorfit::fit(
                    &self.profile.floors[size],
                    &self.scratch.target,
                    &self.scratch.weight,
                    tolerance,
                );
                floorfit::curve(
                    &self.profile.floors[size],
                    &self.scratch.floors[channel],
                    curve,
                );
            }
        }
        Ok(())
    }

    fn write_residues(&mut self, long: bool, writer: &mut BitWriter) -> Result<(), CodecError> {
        let size = usize::from(long);
        let half = self.profile.blocksizes[size] / 2;
        let nyquist = f64::from(self.profile.rate) / 2.0;
        let normalize = Normalize {
            start: ((self.tune.norm_hz / nyquist) * half as f64) as usize,
            width: 16,
            threshold: self.tune.norm_threshold,
        };
        let groups = self.profile.groups.len();
        for group_at in 0..groups {
            let coupled = self.profile.groups[group_at].coupled;
            let channels = self.profile.groups[group_at].channels.clone();
            let end = self.profile.residues[size][usize::from(coupled)].end;
            if coupled {
                let bins = end / 2;
                let point = ((self.tune.point_hz / nyquist) * bins as f64) as usize;
                self.scratch.residue.resize(1, Vec::new());
                self.scratch.residue[0].resize(end, 0);
                let (left, right) = (channels[0], channels[1]);
                quantize_coupled(self, left, right, bins, point, &normalize);
            } else {
                self.scratch.residue.resize(channels.len(), Vec::new());
                for (slot, &channel) in channels.iter().enumerate() {
                    self.scratch.residue[slot].resize(end, 0);
                    self.scratch.residue[slot].fill(0);
                    if self.scratch.silent[channel] {
                        continue;
                    }
                    let coded = end.min(half);
                    let mut ratio = std::mem::take(&mut self.scratch.ratio);
                    ratio.clear();
                    for bin in 0..coded {
                        ratio.push(
                            self.scratch.spectrum[channel][bin] / self.scratch.curve[channel][bin],
                        );
                    }
                    quant::quantize(&ratio, &normalize, &mut self.scratch.residue[slot]);
                    self.scratch.ratio = ratio;
                }
            }
            let skip: Vec<bool> = if coupled {
                vec![self.scratch.silent[channels[0]] && self.scratch.silent[channels[1]]]
            } else {
                channels
                    .iter()
                    .map(|&channel| self.scratch.silent[channel])
                    .collect()
            };
            let kind = usize::from(coupled);
            resenc::write(
                &self.profile.residues[size][kind],
                &self.profile.coders,
                &self.scratch.residue,
                &skip,
                writer,
            )?;
        }
        Ok(())
    }

    fn account(&mut self, bytes: usize, duration: i64) -> Result<(), CodecError> {
        let Some(bitrate) = self.tune.bitrate else {
            return Ok(());
        };
        self.state.bits += bytes as u64 * 8;
        self.state.rated += u64::try_from(duration).map_err(|_| CodecError::Limit {
            what: "Vorbis timestamp",
        })?;
        let settle = u64::from(self.profile.rate) / 10;
        if self.state.rated < settle {
            return Ok(());
        }
        let want = f64::from(bitrate) * self.state.rated as f64 / f64::from(self.profile.rate);
        if want < 1.0 {
            return Ok(());
        }
        let error = (self.state.bits as f64 / want).log2() as f32;
        let gain = if self.tune.constant { 0.9 } else { 0.4 };
        self.state.shift_extra = (self.state.shift_extra - error * gain).clamp(-18.0, 12.0);
        Ok(())
    }

    /// Drops samples the next block's window no longer reaches.
    fn discard(&mut self) {
        let keep = self.keep_from().max(0) as u64;
        if keep <= self.state.origin {
            return;
        }
        let frames = (keep - self.state.origin) as usize;
        let drop = frames * self.profile.channels;
        if drop > self.state.pending.len() {
            return;
        }
        self.state.pending.drain(..drop);
        self.state.origin = keep;
    }

    fn keep_from(&self) -> i64 {
        if let Some(&(long, center)) = self.state.planned.front() {
            let n = self.profile.blocksizes[usize::from(long)];
            return center - n as i64 / 2;
        }
        let Some((previous_long, previous)) = self.state.tail else {
            return 0;
        };
        let previous_n = self.profile.blocksizes[usize::from(previous_long)];
        let short = self.profile.blocksizes[0];
        previous + previous_n as i64 / 4 + short as i64 / 4 - short as i64 / 2
    }

    fn reset_state(&mut self) {
        self.state = State::default();
    }
}

impl Tune {
    fn new(_channels: usize, rate: u32, quality: f32, bitrate: u32, constant: bool) -> Self {
        let nyquist = f64::from(rate) / 2.0;
        Self {
            mask_shift: quality * 14.0,
            ath_shift: (1.0 - quality) * 12.0,
            tolerance: ((1.0 - quality) * 3.0).round() as i32,
            headroom: 8.0 + quality * 72.0,
            point_hz: if quality >= 0.75 {
                nyquist
            } else {
                5_000.0 + f64::from(quality) / 0.75 * 12_000.0
            },
            norm_hz: 400.0,
            norm_threshold: if quality >= 0.9 { 1.0e9 } else { 0.75 },
            bitrate: (bitrate > 0).then_some(bitrate),
            constant,
        }
    }

    fn lowpass(&self, rate: u32) -> f64 {
        let nyquist = f64::from(rate) / 2.0;
        let quality = (self.mask_shift / 14.0).clamp(0.0, 1.0);
        let fraction = (0.50 + f64::from(quality) * 1.6).min(1.0);
        (nyquist * fraction).clamp(1_000.0, nyquist)
    }

    fn nominal(&self, channels: usize, rate: u32, quality: f32, bitrate: u32) -> u32 {
        if bitrate > 0 {
            return bitrate.min(i32::MAX as u32);
        }
        let per_sample = 0.45 + f64::from(quality) * 2.2;
        (per_sample * f64::from(rate) * channels as f64) as u32
    }
}

/// Masking floor in Y units: the mask, or high enough that `spectrum /
/// floor` stays near `headroom`. Bins past `bandwidth` hold that shelf so
/// the curve does not fall through the top of the coded band.
fn fill_target(
    spectrum: &[f32],
    mask: &[f32],
    bandwidth: usize,
    headroom: f32,
    target: &mut [f32],
    weight: &mut [f32],
) {
    let (low, step) = floorfit::y_scale();
    let bandwidth = bandwidth.min(spectrum.len()).max(1);
    for bin in 0..bandwidth {
        let signal = spectrum[bin].abs().max(1e-12);
        let noise = mask[bin].max(1e-20).sqrt();
        let amplitude = noise.max(signal / headroom);
        let db = 20.0 * amplitude.log10();
        target[bin] = ((db - low) / step).clamp(0.0, 127.0);
        weight[bin] = (signal / noise).min(32.0);
    }
    let shelf = target[bandwidth - 1];
    for bin in bandwidth..spectrum.len() {
        target[bin] = shelf;
        weight[bin] = 0.05;
    }
}

/// Square-polar residue of one pair. The inputs are copied so the residue
/// buffer can be borrowed at the same time.
fn quantize_coupled(
    encoder: &mut VorbisEncoder,
    left: usize,
    right: usize,
    bins: usize,
    point: usize,
    normalize: &Normalize,
) {
    let spectra = [
        encoder.scratch.spectrum[left].clone(),
        encoder.scratch.spectrum[right].clone(),
    ];
    let floors = [
        encoder.scratch.curve[left].clone(),
        encoder.scratch.curve[right].clone(),
    ];
    let silent = [encoder.scratch.silent[left], encoder.scratch.silent[right]];
    let pair = Pair {
        spectra: [spectra[0].as_slice(), spectra[1].as_slice()],
        floors: [floors[0].as_slice(), floors[1].as_slice()],
        silent,
    };
    quant::quantize_pair(
        &pair,
        bins,
        point,
        normalize,
        &mut encoder.scratch.pair,
        &mut encoder.scratch.residue[0],
    );
}

fn headers(profile: &Profile, nominal: u32) -> Result<Vec<u8>, CodecError> {
    let identification = profile.identification(nominal).write();
    let comment = Comments {
        vendor: String::from("Cutlass"),
        fields: Vec::new(),
    }
    .write();
    let setup = profile.setup.write(profile.channels);
    Ok(Headers {
        identification,
        comment,
        setup,
    }
    .join())
}

fn prepare(params: &CodecParams) -> Result<(usize, u32, ChannelLayout), CodecError> {
    if params.codec != CodecId::Vorbis {
        return Err(CodecError::Unsupported { what: "codec" });
    }
    let layout = params
        .layout
        .clone()
        .ok_or(CodecError::InvalidApi { what: "layout" })?;
    let channels = layout.len();
    let vorbis = vorbis_layout(channels)?;
    if layout != vorbis {
        return Err(CodecError::Unsupported {
            what: "Vorbis encoder channel order",
        });
    }
    if !(8_000..=192_000).contains(&params.sample_rate) {
        return Err(CodecError::Unsupported {
            what: "Vorbis encoder sample rate",
        });
    }
    Ok((channels, params.sample_rate, layout))
}

/// A starting quality whose untuned rate is near `bitrate`.
fn quality_for_bitrate(channels: usize, rate: u32, bitrate: u32) -> f32 {
    let per_sample = f64::from(bitrate) / (f64::from(rate) * channels as f64);
    ((per_sample - 0.5) / 2.0).clamp(0.0, 1.0) as f32
}

fn pcm_samples(
    layout: &ChannelLayout,
    rate: u32,
    input: &AudioInput,
) -> Result<Vec<f32>, CodecError> {
    let pcm = &input.pcm;
    if pcm.rate != rate {
        return Err(CodecError::InvalidApi {
            what: "sample rate",
        });
    }
    if &pcm.layout != layout {
        return Err(CodecError::InvalidApi { what: "layout" });
    }
    let channels = layout.len();
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
        return Ok(samples
            .iter()
            .map(|&sample| f32::from(sample) / 32768.0)
            .collect());
    }
    Err(CodecError::InvalidApi {
        what: "sample format",
    })
}

impl Encoder for VorbisEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
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
mod tests {
    use super::{Options, VorbisEncoder, pcm_samples};
    use crate::decode::{VorbisDecoder, vorbis_layout};
    use cutlass_decode::{Decoder, Output};
    use cutlass_encode::{
        AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet, Pcm, SideData,
    };
    use std::f32::consts::PI;

    fn layout(channels: usize) -> cutlass_encode::ChannelLayout {
        vorbis_layout(channels).expect("layout")
    }

    fn input(channels: usize, rate: u32, samples: Vec<f32>) -> AudioInput {
        AudioInput {
            pcm: Pcm::from_f32(rate, layout(channels), samples).expect("pcm"),
        }
    }

    fn open(channels: usize, rate: u32, quality: f32, bitrate: u32) -> VorbisEncoder {
        let mut params = CodecParams::audio(CodecId::Vorbis, rate, layout(channels));
        params.bitrate = bitrate;
        VorbisEncoder::open_with(&params, Options { quality }).expect("open")
    }

    fn encode(channels: usize, rate: u32, quality: f32, samples: Vec<f32>) -> Vec<Packet> {
        let mut encoder = open(channels, rate, quality, 0);
        let mut packets = encoder.push(&input(channels, rate, samples)).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        packets
    }

    fn decode(packets: &[Packet]) -> Vec<f32> {
        let SideData::Config(config) = &packets[0].side[0] else {
            panic!("first packet carries the headers");
        };
        let mut decoder = VorbisDecoder::from_config(config).expect("decoder");
        let mut out = Vec::new();
        for packet in packets {
            for output in decoder.push(packet).expect("decode") {
                let Output::Audio(audio) = output else {
                    continue;
                };
                let samples = audio.pcm.as_f32().expect("f32");
                let channels = audio.pcm.layout.len();
                let frames = audio.pcm.samples as usize;
                let lead = audio.skip_leading as usize;
                let trail = audio.skip_trailing as usize;
                for frame in lead..frames - trail {
                    out.extend_from_slice(&samples[frame * channels..(frame + 1) * channels]);
                }
            }
        }
        out
    }

    fn sine(frames: usize, rate: u32, freq: f32, amplitude: f32) -> Vec<f32> {
        (0..frames)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    fn snr(original: &[f32], decoded: &[f32]) -> f32 {
        let mut signal = 0.0_f64;
        let mut noise = 0.0_f64;
        for (&want, &got) in original.iter().zip(decoded) {
            signal += f64::from(want) * f64::from(want);
            let error = f64::from(want - got);
            noise += error * error;
        }
        if noise == 0.0 {
            return 200.0;
        }
        (10.0 * (signal / noise).log10()) as f32
    }

    #[test]
    fn durations_add_up_to_the_input_and_start_at_zero() {
        for (channels, frames) in [(1, 1), (1, 17), (1, 1000), (2, 2048), (2, 5000), (1, 3000)] {
            let packets = encode(
                channels,
                44_100,
                0.5,
                sine(frames, 44_100, 440.0, 0.4).repeat_channels(channels),
            );
            assert_eq!(packets[0].pts, 0);
            assert_eq!(packets[0].duration, 0);
            assert!(matches!(packets[0].side[0], SideData::Config(_)));
            let last = packets.last().expect("packet");
            assert!(matches!(
                last.side.last(),
                Some(SideData::Gapless {
                    encoder_delay: 0,
                    ..
                })
            ));
            let total: i64 = packets.iter().map(|packet| packet.duration).sum();
            assert_eq!(total, frames as i64, "{channels} ch, {frames} frames");
            let decoded = decode(&packets);
            assert_eq!(decoded.len(), frames * channels);
        }
    }

    trait Repeat {
        fn repeat_channels(self, channels: usize) -> Vec<f32>;
    }

    impl Repeat for Vec<f32> {
        fn repeat_channels(self, channels: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(self.len() * channels);
            for sample in self {
                for _ in 0..channels {
                    out.push(sample);
                }
            }
            out
        }
    }

    #[test]
    fn a_tone_round_trips_above_thirty_decibels() {
        let rate = 44_100;
        let frames = rate;
        let original = sine(frames as usize, rate, 440.0, 0.5).repeat_channels(2);
        let decoded = decode(&encode(2, rate, 0.7, original.clone()));
        assert_eq!(decoded.len(), original.len());
        let skip = 2048 * 2;
        let end = original.len() - skip;
        let score = snr(&original[skip..end], &decoded[skip..end]);
        assert!(score > 30.0, "snr {score}");
    }

    #[test]
    fn silence_stays_under_two_kilobits_per_second() {
        let frames = 44_100;
        let packets = encode(2, 44_100, 0.5, vec![0.0; frames * 2]);
        let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
        let bits_per_second = bytes as f64 * 8.0;
        assert!(bits_per_second <= 2_000.0, "{bits_per_second} bps");
        let decoded = decode(&packets);
        assert!(decoded.iter().all(|&sample| sample == 0.0));
        assert_eq!(decoded.len(), frames * 2);
    }

    #[test]
    fn a_higher_quality_spends_more_bits() {
        let frames = 22_050;
        let samples: Vec<f32> = (0..frames)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                0.25 * (2.0 * PI * 180.0 * t).sin()
                    + 0.2 * (2.0 * PI * 1_250.0 * t).sin()
                    + 0.15 * (2.0 * PI * 4_000.0 * t).sin()
            })
            .collect::<Vec<_>>()
            .repeat_channels(2);
        let bytes = |quality| -> usize {
            encode(2, 44_100, quality, samples.clone())
                .iter()
                .map(|packet| packet.data.len())
                .sum()
        };
        assert!(bytes(0.85) > bytes(0.15));
    }

    #[test]
    fn an_impulse_switches_to_a_short_block() {
        let mut samples = vec![0.0_f32; 16_000];
        samples[8_000] = 0.9;
        let packets = encode(1, 44_100, 0.5, samples);
        let durations: Vec<i64> = packets.iter().map(|packet| packet.duration).collect();
        assert!(
            durations
                .iter()
                .any(|&duration| duration == 128 || duration == 576),
            "{durations:?}"
        );
    }

    #[test]
    fn chunked_pushes_match_one_push() {
        let samples = sine(12_000, 44_100, 1_000.0, 0.3);
        let whole = encode(1, 44_100, 0.5, samples.clone());
        let mut encoder = open(1, 44_100, 0.5, 0);
        let mut chunked = Vec::new();
        for chunk in samples.chunks(300) {
            chunked.extend(
                encoder
                    .push(&input(1, 44_100, chunk.to_vec()))
                    .expect("chunk"),
            );
        }
        chunked.extend(encoder.drain().expect("drain"));
        let bytes = |packets: &[Packet]| {
            packets
                .iter()
                .map(|packet| packet.data.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(bytes(&whole), bytes(&chunked));
    }

    #[test]
    fn reset_repeats_the_stream() {
        let samples = sine(4_000, 44_100, 880.0, 0.2);
        let mut encoder = open(1, 44_100, 0.5, 0);
        let first = {
            let mut packets = encoder
                .push(&input(1, 44_100, samples.clone()))
                .expect("push");
            packets.extend(encoder.drain().expect("drain"));
            packets
        };
        encoder.reset();
        let second = {
            let mut packets = encoder.push(&input(1, 44_100, samples)).expect("push");
            packets.extend(encoder.drain().expect("drain"));
            packets
        };
        assert_eq!(
            first.iter().map(|packet| &packet.data).collect::<Vec<_>>(),
            second.iter().map(|packet| &packet.data).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_bad_sample_is_refused_and_leaves_the_stream() {
        let mut encoder = open(1, 44_100, 0.4, 0);
        let error = encoder
            .push(&input(1, 44_100, vec![0.2, f32::NAN, 0.2]))
            .expect_err("nan");
        assert!(matches!(error, CodecError::NonFinite));
        let error = encoder
            .push(&input(1, 44_100, vec![1.5]))
            .expect_err("range");
        assert!(matches!(error, CodecError::OutOfRange));
        let packets = {
            let mut packets = encoder
                .push(&input(1, 44_100, sine(3000, 44_100, 440.0, 0.3)))
                .expect("push");
            packets.extend(encoder.drain().expect("drain"));
            packets
        };
        let fresh = encode(1, 44_100, 0.4, sine(3000, 44_100, 440.0, 0.3));
        assert_eq!(
            packets
                .iter()
                .map(|packet| &packet.data)
                .collect::<Vec<_>>(),
            fresh.iter().map(|packet| &packet.data).collect::<Vec<_>>()
        );
    }

    #[test]
    fn drain_is_empty_the_second_time_and_push_is_refused() {
        let mut encoder = open(1, 8_000, 0.4, 0);
        let mut packets = encoder
            .push(&input(1, 8_000, sine(2000, 8_000, 200.0, 0.2)))
            .expect("push");
        packets.extend(encoder.drain().expect("drain"));
        assert!(encoder.drain().expect("second").is_empty());
        let error = encoder
            .push(&input(1, 8_000, vec![0.0; 100]))
            .expect_err("after");
        assert!(matches!(error, CodecError::InvalidApi { .. }));
        assert!(!packets.is_empty());
        let empty = open(1, 44_100, 0.4, 0);
        let mut empty = empty;
        assert!(
            empty
                .push(&input(1, 44_100, Vec::new()))
                .expect("empty")
                .is_empty()
        );
        assert!(empty.drain().expect("drain").is_empty());
    }

    #[test]
    fn cancel_stops_a_push_without_keeping_it() {
        let mut encoder = open(1, 44_100, 0.4, 0);
        encoder
            .cancel_flag()
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let error = encoder
            .push(&input(1, 44_100, sine(8000, 44_100, 440.0, 0.2)))
            .expect_err("cancel");
        assert!(matches!(error, CodecError::Cancelled));
        encoder
            .cancel_flag()
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let packets = encoder
            .push(&input(1, 44_100, sine(3000, 44_100, 440.0, 0.2)))
            .expect("push");
        assert!(!packets.is_empty());
    }

    #[test]
    fn sixteen_bit_samples_and_other_rates_round_trip_the_count() {
        let mut encoder = open(1, 48_000, 0.5, 0);
        let pcm = Pcm::from_i16(
            48_000,
            layout(1),
            sine(4000, 48_000, 1_000.0, 1.0)
                .iter()
                .map(|s| (*s * 32767.0) as i16)
                .collect(),
        )
        .expect("i16");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        assert_eq!(decode(&packets).len(), 4000);

        let high = encode(
            2,
            96_000,
            0.4,
            sine(6000, 96_000, 1_000.0, 0.2).repeat_channels(2),
        );
        assert_eq!(decode(&high).len(), 12_000);
        assert_eq!(open(1, 96_000, 0.4, 0).profile.blocksizes, [512, 4096]);
    }

    #[test]
    fn the_factory_parameters_are_checked() {
        let stereo = CodecParams::audio(CodecId::Vorbis, 44_100, layout(2));
        assert!(VorbisEncoder::open(&stereo).is_ok());
        let mut other = stereo.clone();
        other.codec = CodecId::Opus;
        assert!(matches!(
            VorbisEncoder::open(&other),
            Err(CodecError::Unsupported { .. })
        ));
        other = stereo.clone();
        other.sample_rate = 4_000;
        assert!(VorbisEncoder::open(&other).is_err());
        other = stereo.clone();
        other.layout = None;
        assert!(matches!(
            VorbisEncoder::open(&other),
            Err(CodecError::InvalidApi { .. })
        ));
        let mut odd = CodecParams::audio(CodecId::Vorbis, 44_100, layout(2));
        // Front-center twice is a layout of two channels that is not stereo.
        odd.layout = Some(
            cutlass_encode::ChannelLayout::new(vec![
                cutlass_encode::Channel::FrontCenter,
                cutlass_encode::Channel::FrontCenter,
            ])
            .expect("layout"),
        );
        assert!(VorbisEncoder::open(&odd).is_err());
        let mut encoder = VorbisEncoder::open(&stereo).expect("open");
        let picture = cutlass_encode::CpuFrame::new(
            2,
            2,
            cutlass_encode::PixelFormat::Gray8,
            vec![0, 0, 0, 0],
            cutlass_encode::FrameTime::new(0, 1, 1).expect("time"),
        )
        .expect("frame");
        assert!(encoder.push_frame(&FrameInput::Cpu(picture)).is_err());
        assert!(encoder.codec() == CodecId::Vorbis);
        let _ = pcm_samples(&layout(2), 44_100, &input(2, 44_100, vec![0.0, 0.0]));
    }
}
