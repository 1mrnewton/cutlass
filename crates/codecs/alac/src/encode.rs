//! The ALAC encoder.
//!
//! The frame encoder follows Apple's reference `ALACEncoder.cpp`: the same
//! element map per channel count, the same shift of low bytes for 24 and
//! 32-bit, a stereo mix weight searched over 0 through 4, a predictor order
//! searched over 4 and 8 on part of the frame, predictor coefficients kept
//! per channel from frame to frame, and an uncompressed element whenever
//! the compressed one would not be smaller. Fast mode is the reference's
//! fast stereo path: no mix or order search. Mono and multichannel
//! elements always search, as in the reference.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet,
    SampleFormat, SideData,
};

use crate::bits::{BitCounter, BitSink, BitWriter};
use crate::config::{
    AlacConfig, Cookie, DEFAULT_FRAME_LENGTH, DEFAULT_KB, DEFAULT_MB, DEFAULT_PB, MAX_CHANNELS,
    MAX_FRAME_LENGTH, default_layout,
};
use crate::frame::{ID_CPE, ID_END, ID_SCE};
use crate::golomb::{self, AgParams};
use crate::predictor::{DENSHIFT_DEFAULT, SEARCH_SETS, init_coefs, pc_block};

/// How hard the encoder searches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EncodeMode {
    /// The reference's full search.
    #[default]
    Default,
    /// The reference's fast stereo path. Mono and multichannel streams
    /// encode as in [`EncodeMode::Default`].
    Fast,
}

/// What one encoder writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncoderConfig {
    pub sample_rate: u32,
    /// 1 through 8.
    pub channels: u8,
    /// 16, 20, 24, or 32.
    pub bit_depth: u8,
    /// Samples per channel in a full frame, 1 through [`MAX_FRAME_LENGTH`].
    pub frame_length: u32,
    pub mode: EncodeMode,
}

impl EncoderConfig {
    /// 4096-sample frames and the default search.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Self {
        Self {
            sample_rate,
            channels,
            bit_depth,
            frame_length: DEFAULT_FRAME_LENGTH,
            mode: EncodeMode::Default,
        }
    }

    fn validate(&self) -> Result<(), CodecError> {
        if self.sample_rate == 0 {
            return Err(CodecError::InvalidApi {
                what: "alac sample rate",
            });
        }
        if self.channels == 0 || self.channels > MAX_CHANNELS {
            return Err(CodecError::Unsupported {
                what: "alac channel count",
            });
        }
        if !matches!(self.bit_depth, 16 | 20 | 24 | 32) {
            return Err(CodecError::Unsupported {
                what: "alac bit depth",
            });
        }
        if self.frame_length == 0 {
            return Err(CodecError::InvalidApi {
                what: "alac frame length",
            });
        }
        if self.frame_length > MAX_FRAME_LENGTH {
            return Err(CodecError::Limit {
                what: "alac frame length",
            });
        }
        Ok(())
    }
}

/// The reference element map: which element starts at each channel.
fn elements(channels: u8) -> &'static [u32] {
    match channels {
        1 => &[ID_SCE],
        2 => &[ID_CPE],
        3 => &[ID_SCE, ID_CPE],
        4 => &[ID_SCE, ID_CPE, ID_SCE],
        5 => &[ID_SCE, ID_CPE, ID_CPE],
        6 => &[ID_SCE, ID_CPE, ID_CPE, ID_SCE],
        7 => &[ID_SCE, ID_CPE, ID_CPE, ID_SCE, ID_SCE],
        _ => &[ID_SCE, ID_CPE, ID_CPE, ID_CPE, ID_SCE],
    }
}

/// Low bytes split off before prediction: none below 24-bit, one at 24,
/// two at 32.
fn bytes_shifted(depth: u32) -> u32 {
    match depth {
        24 => 1,
        32 => 2,
        _ => 0,
    }
}

const MIX_BITS: u32 = 2;
const MAX_MIX_RES: i32 = 4;
const PB_FACTOR: u32 = 4;
const MIN_ORDER: usize = 4;
const MAX_ORDER: usize = 8;
const START_COST: u64 = 1 << 31;

type CoefSets = [[i16; 16]; SEARCH_SETS];

fn fresh_sets() -> CoefSets {
    let mut sets = [[0i16; 16]; SEARCH_SETS];
    for set in &mut sets {
        init_coefs(set, DENSHIFT_DEFAULT);
    }
    sets
}

fn ag_params() -> AgParams {
    AgParams::new(
        u32::from(DEFAULT_MB),
        PB_FACTOR * u32::from(DEFAULT_PB) / 4,
        u32::from(DEFAULT_KB),
    )
}

fn cost(values: &[i32], bits: u32) -> u64 {
    let mut counter = BitCounter::default();
    golomb::compress(ag_params(), values, &mut counter, bits);
    counter.bits
}

/// Encodes one frame at a time from integer samples.
#[derive(Debug)]
struct FrameEncoder {
    config: EncoderConfig,
    coefs_u: Vec<CoefSets>,
    coefs_v: Vec<CoefSets>,
    last_mix_res: Vec<i32>,
    left: Vec<i32>,
    right: Vec<i32>,
    mix_u: Vec<i32>,
    mix_v: Vec<i32>,
    pred_u: Vec<i32>,
    pred_v: Vec<i32>,
    shift_u: Vec<u32>,
    shift_v: Vec<u32>,
    writer: BitWriter,
    max_frame_bytes: u32,
    total_bytes: u64,
    total_samples: u64,
}

impl FrameEncoder {
    fn new(config: EncoderConfig) -> Self {
        let channels = usize::from(config.channels);
        let len = config.frame_length as usize;
        Self {
            config,
            coefs_u: vec![fresh_sets(); channels],
            coefs_v: vec![fresh_sets(); channels],
            last_mix_res: vec![0; channels],
            left: vec![0; len],
            right: vec![0; len],
            mix_u: vec![0; len],
            mix_v: vec![0; len],
            pred_u: vec![0; len],
            pred_v: vec![0; len],
            shift_u: vec![0; len],
            shift_v: vec![0; len],
            writer: BitWriter::with_capacity(len * channels * 5 + 1),
            max_frame_bytes: 0,
            total_bytes: 0,
            total_samples: 0,
        }
    }

    fn depth(&self) -> u32 {
        u32::from(self.config.bit_depth)
    }

    /// Encode `count` samples per channel from `samples`, interleaved in
    /// frame channel order, each already in range for the bit depth.
    fn encode(&mut self, samples: &[i32], count: usize) -> Vec<u8> {
        let channels = usize::from(self.config.channels);
        self.writer.clear();
        let mut index = 0usize;
        let mut sce = 0u32;
        let mut cpe = 0u32;
        for &element in elements(self.config.channels) {
            if element == ID_CPE {
                self.writer.put(ID_CPE, 3);
                self.writer.put(cpe, 4);
                cpe += 1;
                for i in 0..count {
                    self.left[i] = samples[i * channels + index];
                    self.right[i] = samples[i * channels + index + 1];
                }
                if channels == 2 && self.config.mode == EncodeMode::Fast {
                    self.stereo_fast(index, count);
                } else {
                    self.stereo(index, count);
                }
                index += 2;
            } else {
                self.writer.put(ID_SCE, 3);
                self.writer.put(sce, 4);
                sce += 1;
                for i in 0..count {
                    self.left[i] = samples[i * channels + index];
                }
                self.mono(index, count);
                index += 1;
            }
        }
        self.writer.put(ID_END, 3);
        let bytes = self.writer.take_bytes();
        let size = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        self.max_frame_bytes = self.max_frame_bytes.max(size);
        self.total_bytes += bytes.len() as u64;
        self.total_samples += count as u64;
        bytes
    }

    fn cookie(&self) -> Cookie {
        let mut config = AlacConfig::new(
            self.config.sample_rate,
            self.config.channels,
            self.config.bit_depth,
            self.config.frame_length,
        );
        config.max_frame_bytes = self.max_frame_bytes;
        if let Some(rate) = self
            .total_bytes
            .saturating_mul(8 * u64::from(self.config.sample_rate))
            .checked_div(self.total_samples)
        {
            config.avg_bit_rate = u32::try_from(rate).unwrap_or(u32::MAX);
        }
        Cookie::new(config)
    }

    /// Split low bytes and apply the mix weight to the first `count` samples.
    fn mix(&mut self, count: usize, res: i32, shift: u32) {
        let mask = if shift == 0 { 0 } else { (1u32 << shift) - 1 };
        let weight = (1i32 << MIX_BITS) - res;
        for i in 0..count {
            let mut l = self.left[i];
            let mut r = self.right[i];
            if shift != 0 {
                self.shift_u[i] = l as u32 & mask;
                self.shift_v[i] = r as u32 & mask;
                l >>= shift;
                r >>= shift;
            }
            if res != 0 {
                self.mix_u[i] =
                    res.wrapping_mul(l).wrapping_add(weight.wrapping_mul(r)) >> MIX_BITS;
                self.mix_v[i] = l.wrapping_sub(r);
            } else {
                self.mix_u[i] = l;
                self.mix_v[i] = r;
            }
        }
    }

    fn header(&mut self, partial: bool, shifted: u32, escape: bool, count: usize) {
        self.writer.put(0, 12);
        self.writer.put(
            (u32::from(partial) << 3) | (shifted << 1) | u32::from(escape),
            4,
        );
        if partial {
            self.writer.put(count as u32, 32);
        }
    }

    fn channel_params(&mut self, coefs: [i16; 16], order: usize) {
        self.writer.put(DENSHIFT_DEFAULT, 8);
        self.writer.put((PB_FACTOR << 5) | order as u32, 8);
        for coef in coefs.iter().take(order) {
            self.writer.put(u32::from(*coef as u16), 16);
        }
    }

    fn escape_bits(&self, count: usize, width: u64, partial: bool) -> u64 {
        count as u64 * u64::from(self.depth()) * width + if partial { 32 } else { 0 } + 16
    }

    fn stereo(&mut self, ch: usize, count: usize) {
        let depth = self.depth();
        let shifted = bytes_shifted(depth);
        let shift = shifted * 8;
        let chan_bits = depth - shift + 1;
        let partial = count != self.config.frame_length as usize;

        // The mix weight, chosen on the first eighth of the frame with order 8.
        let part = count / 8;
        let mut best_cost = START_COST;
        let mut best_res = self.last_mix_res[ch];
        for res in 0..=MAX_MIX_RES {
            self.mix(part, res, shift);
            pc_block(
                &self.mix_u,
                &mut self.pred_u,
                part,
                &mut self.coefs_u[ch][MAX_ORDER - 1],
                MAX_ORDER,
                chan_bits,
                DENSHIFT_DEFAULT,
            );
            pc_block(
                &self.mix_v,
                &mut self.pred_v,
                part,
                &mut self.coefs_v[ch][MAX_ORDER - 1],
                MAX_ORDER,
                chan_bits,
                DENSHIFT_DEFAULT,
            );
            let bits =
                cost(&self.pred_u[..part], chan_bits) + cost(&self.pred_v[..part], chan_bits);
            if bits < best_cost {
                best_cost = bits;
                best_res = res;
            }
        }
        self.last_mix_res[ch] = best_res;
        let res = best_res;
        self.mix(count, res, shift);

        // The order of each channel, after the coefficients settle on the
        // first thirty-second of the frame.
        let mut order_u = MIN_ORDER;
        let mut order_v = MIN_ORDER;
        let mut cost_u = START_COST;
        let mut cost_v = START_COST;
        for order in [MIN_ORDER, MAX_ORDER] {
            for _ in 0..8 {
                pc_block(
                    &self.mix_u,
                    &mut self.pred_u,
                    count / 32,
                    &mut self.coefs_u[ch][order - 1],
                    order,
                    chan_bits,
                    DENSHIFT_DEFAULT,
                );
                pc_block(
                    &self.mix_v,
                    &mut self.pred_v,
                    count / 32,
                    &mut self.coefs_v[ch][order - 1],
                    order,
                    chan_bits,
                    DENSHIFT_DEFAULT,
                );
            }
            let bits = cost(&self.pred_u[..part], chan_bits) * 8 + 16 * order as u64;
            if bits < cost_u {
                cost_u = bits;
                order_u = order;
            }
            let bits = cost(&self.pred_v[..part], chan_bits) * 8 + 16 * order as u64;
            if bits < cost_v {
                cost_v = bits;
                order_v = order;
            }
        }

        let mut estimate = cost_u + cost_v + 64 + if partial { 32 } else { 0 };
        if shift != 0 {
            estimate += count as u64 * u64::from(shift) * 2;
        }
        let escape_bits = self.escape_bits(count, 2, partial);
        if estimate >= escape_bits {
            self.stereo_escape(count, partial);
            return;
        }
        let mark = self.writer.position();
        self.header(partial, shifted, false, count);
        self.writer.put(MIX_BITS, 8);
        self.writer.put(res as u8 as u32, 8);
        let coefs = self.coefs_u[ch][order_u - 1];
        self.channel_params(coefs, order_u);
        let coefs = self.coefs_v[ch][order_v - 1];
        self.channel_params(coefs, order_v);
        self.write_shifts(count, shift, true);
        pc_block(
            &self.mix_u,
            &mut self.pred_u,
            count,
            &mut self.coefs_u[ch][order_u - 1],
            order_u,
            chan_bits,
            DENSHIFT_DEFAULT,
        );
        golomb::compress(
            ag_params(),
            &self.pred_u[..count],
            &mut self.writer,
            chan_bits,
        );
        pc_block(
            &self.mix_v,
            &mut self.pred_v,
            count,
            &mut self.coefs_v[ch][order_v - 1],
            order_v,
            chan_bits,
            DENSHIFT_DEFAULT,
        );
        golomb::compress(
            ag_params(),
            &self.pred_v[..count],
            &mut self.writer,
            chan_bits,
        );
        if (self.writer.position() - mark) as u64 >= escape_bits {
            self.writer.rewind(mark);
            self.stereo_escape(count, partial);
        }
    }

    fn stereo_fast(&mut self, ch: usize, count: usize) {
        let depth = self.depth();
        let shifted = bytes_shifted(depth);
        let shift = shifted * 8;
        let chan_bits = depth - shift + 1;
        let partial = count != self.config.frame_length as usize;
        self.mix(count, 0, shift);
        let mark = self.writer.position();
        self.header(partial, shifted, false, count);
        self.writer.put(MIX_BITS, 8);
        self.writer.put(0, 8);
        let coefs = self.coefs_u[ch][MAX_ORDER - 1];
        self.channel_params(coefs, MAX_ORDER);
        let coefs = self.coefs_v[ch][MAX_ORDER - 1];
        self.channel_params(coefs, MAX_ORDER);
        self.write_shifts(count, shift, true);
        pc_block(
            &self.mix_u,
            &mut self.pred_u,
            count,
            &mut self.coefs_u[ch][MAX_ORDER - 1],
            MAX_ORDER,
            chan_bits,
            DENSHIFT_DEFAULT,
        );
        let before = self.writer.position();
        golomb::compress(
            ag_params(),
            &self.pred_u[..count],
            &mut self.writer,
            chan_bits,
        );
        let bits_u = (self.writer.position() - before) as u64;
        pc_block(
            &self.mix_v,
            &mut self.pred_v,
            count,
            &mut self.coefs_v[ch][MAX_ORDER - 1],
            MAX_ORDER,
            chan_bits,
            DENSHIFT_DEFAULT,
        );
        let before = self.writer.position();
        golomb::compress(
            ag_params(),
            &self.pred_v[..count],
            &mut self.writer,
            chan_bits,
        );
        let bits_v = (self.writer.position() - before) as u64;
        let order_bits = 16 * MAX_ORDER as u64;
        let mut estimate =
            bits_u + order_bits + bits_v + order_bits + 64 + if partial { 32 } else { 0 };
        if shift != 0 {
            estimate += count as u64 * u64::from(shift) * 2;
        }
        let escape_bits = self.escape_bits(count, 2, partial);
        if estimate >= escape_bits || (self.writer.position() - mark) as u64 >= escape_bits {
            self.writer.rewind(mark);
            self.stereo_escape(count, partial);
        }
    }

    fn write_shifts(&mut self, count: usize, shift: u32, pair: bool) {
        if shift == 0 {
            return;
        }
        for i in 0..count {
            if pair {
                self.writer
                    .put((self.shift_u[i] << shift) | self.shift_v[i], shift * 2);
            } else {
                self.writer.put(self.shift_u[i], shift);
            }
        }
    }

    fn stereo_escape(&mut self, count: usize, partial: bool) {
        let depth = self.depth();
        self.header(partial, 0, true, count);
        for i in 0..count {
            self.writer.put(self.left[i] as u32, depth);
            self.writer.put(self.right[i] as u32, depth);
        }
    }

    fn mono(&mut self, ch: usize, count: usize) {
        let depth = self.depth();
        let shifted = bytes_shifted(depth);
        let shift = shifted * 8;
        let chan_bits = depth - shift;
        let partial = count != self.config.frame_length as usize;
        let mask = if shift == 0 { 0 } else { (1u32 << shift) - 1 };
        for i in 0..count {
            let value = self.left[i];
            if shift != 0 {
                self.shift_u[i] = value as u32 & mask;
                self.mix_u[i] = value >> shift;
            } else {
                self.mix_u[i] = value;
            }
        }
        let part = count / 8;
        let mut best = MIN_ORDER;
        let mut best_cost = START_COST;
        for order in [MIN_ORDER, MAX_ORDER] {
            for _ in 0..7 {
                pc_block(
                    &self.mix_u,
                    &mut self.pred_u,
                    count / 32,
                    &mut self.coefs_u[ch][order - 1],
                    order,
                    chan_bits,
                    DENSHIFT_DEFAULT,
                );
            }
            pc_block(
                &self.mix_u,
                &mut self.pred_u,
                part,
                &mut self.coefs_u[ch][order - 1],
                order,
                chan_bits,
                DENSHIFT_DEFAULT,
            );
            let bits = cost(&self.pred_u[..part], chan_bits) * 8 + 16 * order as u64;
            if bits < best_cost {
                best_cost = bits;
                best = order;
            }
        }
        let mut estimate = best_cost + 32 + if partial { 32 } else { 0 };
        if shift != 0 {
            estimate += count as u64 * u64::from(shift);
        }
        let escape_bits = self.escape_bits(count, 1, partial);
        if estimate >= escape_bits {
            self.mono_escape(count, partial);
            return;
        }
        let mark = self.writer.position();
        self.header(partial, shifted, false, count);
        self.writer.put(0, 16);
        let coefs = self.coefs_u[ch][best - 1];
        self.channel_params(coefs, best);
        self.write_shifts(count, shift, false);
        pc_block(
            &self.mix_u,
            &mut self.pred_u,
            count,
            &mut self.coefs_u[ch][best - 1],
            best,
            chan_bits,
            DENSHIFT_DEFAULT,
        );
        golomb::compress(
            ag_params(),
            &self.pred_u[..count],
            &mut self.writer,
            chan_bits,
        );
        if (self.writer.position() - mark) as u64 >= escape_bits {
            self.writer.rewind(mark);
            self.mono_escape(count, partial);
        }
    }

    fn mono_escape(&mut self, count: usize, partial: bool) {
        let depth = self.depth();
        self.header(partial, 0, true, count);
        for i in 0..count {
            self.writer.put(self.left[i] as u32, depth);
        }
    }
}

/// Encodes PCM into ALAC frames, one packet per frame.
///
/// `push` buffers less than one frame. `drain` writes the remainder as a
/// shorter final frame: ALAC signals that length, so nothing is padded.
/// The first packet carries the magic cookie as [`SideData::Config`], with
/// the size and rate statistics still zero. The packet `drain` writes
/// carries it again with the largest frame and the average rate filled in.
#[derive(Debug)]
pub struct AlacEncoder {
    frame: FrameEncoder,
    /// Input channel index for each frame channel.
    order: Vec<usize>,
    pending: Vec<i32>,
    clock: i64,
    sent_cookie: bool,
    cancel: Arc<AtomicBool>,
}

impl AlacEncoder {
    /// An encoder that takes interleaved samples in `layout` order.
    ///
    /// Mono and stereo take any labels. Three or more channels must carry the
    /// labels of the reference layout for that count, in any order; samples
    /// are reordered into frame order.
    pub fn new(config: EncoderConfig, layout: &ChannelLayout) -> Result<Self, CodecError> {
        config.validate()?;
        if layout.len() != usize::from(config.channels) {
            return Err(CodecError::InvalidApi {
                what: "alac channel layout",
            });
        }
        let target = default_layout(config.channels);
        let order = if config.channels <= 2 {
            (0..layout.len()).collect()
        } else {
            let mut order = Vec::with_capacity(layout.len());
            for label in target.channels() {
                let at = layout
                    .channels()
                    .iter()
                    .position(|have| have == label)
                    .ok_or(CodecError::Unsupported {
                        what: "alac channel layout",
                    })?;
                order.push(at);
            }
            order
        };
        Ok(Self {
            frame: FrameEncoder::new(config),
            order,
            pending: Vec::new(),
            clock: 0,
            sent_cookie: false,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Open from encoder parameters.
    ///
    /// `extra` may hold a cookie: its bit depth and frame length are used,
    /// and its rate and channel count must match. Without one, `I16` input
    /// writes 16-bit and `F32` writes 24-bit, in 4096-sample frames.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Alac {
            return Err(CodecError::Unsupported { what: "alac" });
        }
        let layout = params.layout.clone().ok_or(CodecError::InvalidApi {
            what: "channel layout",
        })?;
        let channels = u8::try_from(layout.len()).map_err(|_| CodecError::Unsupported {
            what: "alac channel count",
        })?;
        let mut config = EncoderConfig::new(params.sample_rate, channels, 16);
        if params.extra.is_empty() {
            config.bit_depth = match params.sample_format {
                Some(SampleFormat::F32) | None => 24,
                Some(SampleFormat::I16) => 16,
            };
        } else {
            let cookie = Cookie::parse(&params.extra)?;
            if cookie.config.channels != channels
                || (cookie.config.sample_rate != 0
                    && cookie.config.sample_rate != params.sample_rate)
            {
                return Err(CodecError::InvalidApi {
                    what: "alac cookie differs from the stream",
                });
            }
            config.bit_depth = cookie.config.bit_depth;
            config.frame_length = cookie.config.frame_length;
        }
        Self::new(config, &layout)
    }

    #[must_use]
    pub fn config(&self) -> EncoderConfig {
        self.frame.config
    }

    /// The cookie for the frames written so far.
    #[must_use]
    pub fn cookie(&self) -> Cookie {
        self.frame.cookie()
    }

    /// Encode interleaved integer samples in the input layout's order,
    /// each a signed value of the configured bit depth.
    pub fn push_samples(&mut self, samples: &[i32]) -> Result<Vec<Packet>, CodecError> {
        let channels = usize::from(self.frame.config.channels);
        if !samples.len().is_multiple_of(channels) {
            return Err(CodecError::UnevenChannels);
        }
        let depth = u32::from(self.frame.config.bit_depth);
        let low = -(1i64 << (depth - 1));
        let high = (1i64 << (depth - 1)) - 1;
        if samples
            .iter()
            .any(|&v| !(low..=high).contains(&i64::from(v)))
        {
            return Err(CodecError::OutOfRange);
        }
        self.queue(samples.chunks_exact(channels));
        self.full_frames()
    }

    fn queue<'a>(&mut self, frames: impl Iterator<Item = &'a [i32]>) {
        for frame in frames {
            for &input in &self.order {
                self.pending.push(frame[input]);
            }
        }
    }

    fn full_frames(&mut self) -> Result<Vec<Packet>, CodecError> {
        let channels = usize::from(self.frame.config.channels);
        let frame_samples = self.frame.config.frame_length as usize * channels;
        let mut packets = Vec::new();
        let mut start = 0;
        while self.pending.len() - start >= frame_samples {
            if self.cancel.load(Ordering::Relaxed) {
                self.pending.drain(..start);
                return Err(CodecError::Cancelled);
            }
            let end = start + frame_samples;
            let block = self.pending[start..end].to_vec();
            packets.push(self.packet(&block)?);
            start = end;
        }
        self.pending.drain(..start);
        Ok(packets)
    }

    fn packet(&mut self, block: &[i32]) -> Result<Packet, CodecError> {
        let channels = usize::from(self.frame.config.channels);
        let count = block.len() / channels;
        let bytes = self.frame.encode(block, count);
        let duration = i64::try_from(count).map_err(|_| CodecError::Limit {
            what: "frame samples",
        })?;
        let mut packet = Packet::audio(bytes, self.clock, duration, self.frame.config.sample_rate);
        if !self.sent_cookie {
            packet
                .side
                .push(SideData::Config(self.frame.cookie().to_bytes()));
            self.sent_cookie = true;
        }
        self.clock = self.clock.saturating_add(duration);
        Ok(packet)
    }

    fn samples_of(&self, input: &AudioInput) -> Result<Vec<i32>, CodecError> {
        let pcm = &input.pcm;
        let config = self.frame.config;
        if pcm.rate != config.sample_rate {
            return Err(CodecError::InvalidApi {
                what: "sample rate changed",
            });
        }
        if pcm.layout.len() != usize::from(config.channels) {
            return Err(CodecError::InvalidApi {
                what: "channel count changed",
            });
        }
        let depth = u32::from(config.bit_depth);
        match pcm.format {
            SampleFormat::I16 => {
                let samples = pcm.as_i16().ok_or(CodecError::InvalidApi { what: "pcm" })?;
                Ok(samples
                    .iter()
                    .map(|&v| i32::from(v) << (depth - 16))
                    .collect())
            }
            SampleFormat::F32 => {
                let samples = pcm.as_f32().ok_or(CodecError::InvalidApi { what: "pcm" })?;
                let scale = f64::from(1u32 << (depth - 1));
                let high = scale - 1.0;
                let mut out = Vec::with_capacity(samples.len());
                for &sample in samples {
                    if !sample.is_finite() {
                        return Err(CodecError::NonFinite);
                    }
                    if !(-1.0..=1.0).contains(&sample) {
                        return Err(CodecError::OutOfRange);
                    }
                    let value = (f64::from(sample) * scale).round().clamp(-scale, high);
                    out.push(value as i32);
                }
                Ok(out)
            }
        }
    }
}

impl Encoder for AlacEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Alac
    }

    /// `I16` input is shifted up to the bit depth. `F32` input is scaled by
    /// `2^(depth - 1)` and rounded: the samples this crate's decoder writes
    /// are already on that grid and round-trip exactly.
    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        let samples = self.samples_of(input)?;
        let channels = usize::from(self.frame.config.channels);
        self.queue(samples.chunks_exact(channels));
        self.full_frames()
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let block = std::mem::take(&mut self.pending);
        let mut packet = self.packet(&block)?;
        packet
            .side
            .retain(|side| !matches!(side, SideData::Config(_)));
        packet
            .side
            .push(SideData::Config(self.frame.cookie().to_bytes()));
        Ok(vec![packet])
    }

    /// Drops buffered samples, the clock, and the predictor history, so the
    /// next frame encodes as the first frame of a new stream.
    fn reset(&mut self) {
        let config = self.frame.config;
        self.frame = FrameEncoder::new(config);
        self.pending.clear();
        self.clock = 0;
        self.sent_cookie = false;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
