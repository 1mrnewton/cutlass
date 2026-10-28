//! AAC-LC encoder.
//!
//! Analysis uses the same filterbank as the decoder. A frame is emitted once
//! the next frame is available, so a transient can open a start window before
//! the short window that carries it. That lookahead is buffering, not extra
//! delay: the overlap delay is one frame. `drain` pads a partial frame and
//! then writes one extra frame so the overlap is released.
//!
//! [`RateControl::Average`] aims at a mean payload rate. Access units are not
//! padded to equal sizes. [`RateControl::Variable`] spends more bits at higher
//! quality and does not promise a bitrate. The scale-factor search is bounded.
//! The same samples and configuration produce the same bytes in one process.
//!
//! The encoder does not emit prediction, coupling, perceptual noise
//! substitution, intensity stereo, or temporal noise shaping. Mid/side is used
//! when the two channels of a pair are strongly correlated. The decoder still
//! accepts the tools this encoder leaves out.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, SideData};
use cutlass_encode::{AudioInput, Encoder};

use crate::bits::{BitCounter, BitSink, BitWriter};
use crate::config::{
    AudioConfig, ElementKind, EncoderConfig, MAX_FRAME_BITS_PER_CHANNEL, RateControl,
    default_bitrate,
};
use crate::filterbank::{FULL_SCALE, Sequence, Windows, analyze};
use crate::huffman::{
    pair_index, quad_index, scale_code, signed_pair_index, spectral_code, write_code, write_escape,
};
use crate::quant::{MAX_QUANT, dequant, quantize};

const MAX_ACCESS_UNIT: usize = 8192;
const RATE_STEPS: i32 = 40;
const FULL_SCALE_SLACK: f32 = 16.0 * f32::EPSILON;

pub struct AacEncoder {
    config: AudioConfig,
    rate: RateControl,
    windows: Windows,
    pending: Vec<f32>,
    history: Vec<Vec<f32>>,
    prev: Sequence,
    prev_kbd: bool,
    reservoir: i32,
    frame_index: u64,
    accepted: u64,
    drained: bool,
    sent_config: bool,
    cancel: Arc<AtomicBool>,
}

struct Snapshot {
    pending: Vec<f32>,
    history: Vec<Vec<f32>>,
    prev: Sequence,
    prev_kbd: bool,
    reservoir: i32,
    frame_index: u64,
    accepted: u64,
    sent_config: bool,
}

struct PreparedChannel {
    spec: Vec<f32>,
    bases: Vec<Option<i32>>,
}

struct PreparedElement {
    kind: ElementKind,
    short: bool,
    seq: Sequence,
    channels: Vec<PreparedChannel>,
    ms: Vec<bool>,
    num_swb: usize,
}

struct CodedChannel {
    books: Vec<u8>,
    scales: Vec<i32>,
    quant: Vec<i32>,
    edges: Vec<usize>,
    max_sfb: usize,
    windows: usize,
    short: bool,
    seq: Sequence,
}

struct CodedElement {
    kind: ElementKind,
    channels: Vec<CodedChannel>,
    ms: Vec<bool>,
    num_swb: usize,
}

impl AacEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Aac {
            return Err(CodecError::Unsupported {
                what: "audio object type",
            });
        }
        let layout = params
            .layout
            .clone()
            .ok_or(CodecError::InvalidApi { what: "layout" })?;
        let bitrate = if params.bitrate == 0 {
            default_bitrate(layout.len())
        } else {
            params.bitrate
        };
        Self::from_config(EncoderConfig {
            profile: crate::config::Profile::AacLc,
            mpeg2: false,
            sample_rate: params.sample_rate,
            layout,
            frame_length: 1024,
            rate: RateControl::Average { bitrate },
        })
    }

    pub fn from_config(config: EncoderConfig) -> Result<Self, CodecError> {
        let audio = config.audio()?;
        let channels = audio.layout.len();
        let frame = audio.frame_length;
        Ok(Self {
            windows: Windows::new(frame)?,
            history: vec![vec![0.0; frame]; channels],
            pending: Vec::new(),
            prev: Sequence::OnlyLong,
            prev_kbd: false,
            reservoir: 0,
            frame_index: 0,
            accepted: 0,
            drained: false,
            sent_config: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rate: config.rate,
            config: audio,
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    fn push_samples(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if self.drained {
            return Err(CodecError::InvalidApi {
                what: "push after drain",
            });
        }
        let samples = pcm_samples(&self.config, input)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let saved = self.snapshot();
        self.pending.extend(samples);
        let mut packets = Vec::new();
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                self.restore(saved);
                return Err(CodecError::Cancelled);
            }
            if self.pending.len() < self.frame_samples() * 2 {
                break;
            }
            match self.emit_ready() {
                Ok(packet) => packets.push(packet),
                Err(err) => {
                    self.restore(saved);
                    return Err(err);
                }
            }
        }
        Ok(packets)
    }

    fn drain_samples(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        let saved = self.snapshot();
        let mut packets = Vec::new();
        while !self.pending.is_empty() {
            if self.cancel.load(Ordering::Relaxed) {
                self.restore(saved);
                return Err(CodecError::Cancelled);
            }
            match self.emit_ready() {
                Ok(packet) => packets.push(packet),
                Err(err) => {
                    self.restore(saved);
                    return Err(err);
                }
            }
        }
        if self.accepted > 0 {
            let flush = vec![vec![0.0; self.config.frame_length]; self.channel_count()];
            match self.emit_planes(flush, false) {
                Ok(packet) => packets.push(packet),
                Err(err) => {
                    self.restore(saved);
                    return Err(err);
                }
            }
        }
        let padding = self.trailing_padding()?;
        for packet in &mut packets {
            stamp_padding(packet, padding);
        }
        self.drained = true;
        Ok(packets)
    }

    fn reset_audio(&mut self) {
        let frame = self.config.frame_length;
        self.pending.clear();
        self.history = vec![vec![0.0; frame]; self.channel_count()];
        self.prev = Sequence::OnlyLong;
        self.prev_kbd = false;
        self.reservoir = 0;
        self.frame_index = 0;
        self.accepted = 0;
        self.drained = false;
        self.sent_config = false;
    }

    fn emit_ready(&mut self) -> Result<Packet, CodecError> {
        let next = self.lookahead_transient();
        let planes = self.take_planes()?;
        self.emit_planes(planes, next)
    }

    fn emit_planes(
        &mut self,
        planes: Vec<Vec<f32>>,
        next_transient: bool,
    ) -> Result<Packet, CodecError> {
        let seq = choose_sequence(self.prev, next_transient);
        let prepared = self.prepare(&planes, seq)?;
        let budget = self.budget_bits();
        let coded = select_coded(&self.config, &prepared, budget, self.offset_floor())?;
        let bytes = write_frame(&coded)?;
        if bytes.len() > MAX_ACCESS_UNIT {
            return Err(CodecError::Limit {
                what: "access unit bytes",
            });
        }
        self.commit_rate(bytes.len());
        self.history = planes;
        self.prev = seq;
        self.prev_kbd = true;
        let frame = u64::try_from(self.config.frame_length).unwrap_or(0);
        let pts = i64::try_from(self.frame_index.saturating_mul(frame)).unwrap_or(i64::MAX);
        self.frame_index = self.frame_index.saturating_add(1);
        let mut packet = Packet::audio(
            bytes,
            pts,
            i64::try_from(frame).unwrap_or(0),
            self.config.sample_rate,
        );
        packet.side.push(SideData::Gapless {
            encoder_delay: u32::try_from(self.config.frame_length).unwrap_or(u32::MAX),
            trailing_padding: 0,
            sample_rate: self.config.sample_rate,
        });
        if !self.sent_config {
            packet.side.push(SideData::Config(self.config.asc.clone()));
            self.sent_config = true;
        }
        Ok(packet)
    }

    fn prepare(
        &self,
        planes: &[Vec<f32>],
        seq: Sequence,
    ) -> Result<Vec<PreparedElement>, CodecError> {
        let mut prepared = Vec::new();
        let mut index = 0usize;
        for kind in &self.config.elements {
            let element = match kind {
                ElementKind::Sce => {
                    let short = seq == Sequence::EightShort;
                    let spec = self.analyze_channel(index, &planes[index], seq)?;
                    index += 1;
                    prepare_single(*kind, spec, seq, short, &self.config)?
                }
                ElementKind::Lfe => {
                    let mut spec =
                        self.analyze_channel(index, &planes[index], Sequence::OnlyLong)?;
                    lowpass_lfe(&mut spec, self.config.sample_rate, self.config.frame_length);
                    index += 1;
                    prepare_single(*kind, spec, Sequence::OnlyLong, false, &self.config)?
                }
                ElementKind::Cpe => {
                    let short = seq == Sequence::EightShort;
                    let mut left = self.analyze_channel(index, &planes[index], seq)?;
                    let mut right = self.analyze_channel(index + 1, &planes[index + 1], seq)?;
                    index += 2;
                    let bands = self.config.bands(short)?;
                    let ms = apply_ms(&mut left, &mut right, bands);
                    let channels = vec![
                        prepare_channel(left, bands)?,
                        prepare_channel(right, bands)?,
                    ];
                    PreparedElement {
                        kind: ElementKind::Cpe,
                        short,
                        seq,
                        num_swb: bands.len().saturating_sub(1),
                        channels,
                        ms,
                    }
                }
            };
            prepared.push(element);
        }
        Ok(prepared)
    }

    fn analyze_channel(
        &self,
        index: usize,
        current: &[f32],
        seq: Sequence,
    ) -> Result<Vec<f32>, CodecError> {
        let history = self
            .history
            .get(index)
            .ok_or(CodecError::InvalidApi { what: "channel" })?;
        let mut spec = analyze(&self.windows, history, current, seq, self.prev_kbd, true)?;
        for bin in &mut spec {
            *bin *= FULL_SCALE;
        }
        Ok(spec)
    }

    fn take_planes(&mut self) -> Result<Vec<Vec<f32>>, CodecError> {
        let channels = self.channel_count();
        let frame = self.config.frame_length;
        if channels == 0 || self.pending.len() < channels {
            return Err(CodecError::InvalidApi { what: "pcm" });
        }
        let have = self.pending.len() / channels;
        let real = have.min(frame);
        let mut planes = vec![vec![0.0; frame]; channels];
        for (sample, frame_samples) in self.pending.chunks(channels).take(real).enumerate() {
            for (channel, value) in frame_samples.iter().enumerate() {
                planes[channel][sample] = *value;
            }
        }
        let consumed = real * channels;
        self.pending.drain(..consumed);
        self.accepted = self
            .accepted
            .saturating_add(u64::try_from(real).unwrap_or(0));
        Ok(planes)
    }

    fn lookahead_transient(&self) -> bool {
        let start = self.frame_samples();
        if self.pending.len() <= start {
            return false;
        }
        let end = (start + self.frame_samples()).min(self.pending.len());
        interleaved_transient(
            &self.pending[start..end],
            self.channel_count(),
            self.config.frame_length,
        )
    }

    fn budget_bits(&self) -> u32 {
        let max_bits = MAX_FRAME_BITS_PER_CHANNEL.saturating_mul(self.channels_u32());
        let budget = match self.rate {
            RateControl::Average { bitrate } => {
                let target =
                    frame_target(bitrate, self.config.frame_length, self.config.sample_rate);
                let available = self.reservoir.saturating_add(target);
                clamp_i32(available, 0, i32::try_from(max_bits).unwrap_or(i32::MAX))
            }
            RateControl::Variable { quality, cap } => {
                let spend = max_bits.saturating_mul(u32::from(quality)) / 10;
                let spend = match cap {
                    Some(bitrate) => spend.min(target_u32(
                        bitrate,
                        self.config.frame_length,
                        self.config.sample_rate,
                    )),
                    None => spend,
                };
                i32::try_from(spend).unwrap_or(i32::MAX)
            }
        };
        u32::try_from(budget.saturating_sub(16).max(32)).unwrap_or(32)
    }

    fn offset_floor(&self) -> i32 {
        match self.rate {
            RateControl::Average { .. } => -24,
            RateControl::Variable { quality, .. } => i32::from(10 - quality) * 6 - 24,
        }
    }

    fn commit_rate(&mut self, bytes: usize) {
        let RateControl::Average { bitrate } = self.rate else {
            return;
        };
        let target = frame_target(bitrate, self.config.frame_length, self.config.sample_rate);
        let used = i32::try_from(bytes.saturating_mul(8)).unwrap_or(i32::MAX);
        let max_reservoir =
            i32::try_from(MAX_FRAME_BITS_PER_CHANNEL.saturating_mul(self.channels_u32()))
                .unwrap_or(i32::MAX);
        self.reservoir = clamp_i32(self.reservoir + target - used, 0, max_reservoir);
    }

    fn trailing_padding(&self) -> Result<u32, CodecError> {
        let frame = u64::try_from(self.config.frame_length).unwrap_or(0);
        let delay = frame;
        let produced = self.frame_index.saturating_mul(frame);
        let padding = produced.saturating_sub(delay).saturating_sub(self.accepted);
        u32::try_from(padding).map_err(|_| CodecError::Limit {
            what: "trailing padding",
        })
    }

    fn frame_samples(&self) -> usize {
        self.config
            .frame_length
            .saturating_mul(self.channel_count())
    }

    fn channel_count(&self) -> usize {
        self.config.layout.len()
    }

    fn channels_u32(&self) -> u32 {
        u32::try_from(self.channel_count()).unwrap_or(1)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            pending: self.pending.clone(),
            history: self.history.clone(),
            prev: self.prev,
            prev_kbd: self.prev_kbd,
            reservoir: self.reservoir,
            frame_index: self.frame_index,
            accepted: self.accepted,
            sent_config: self.sent_config,
        }
    }

    fn restore(&mut self, saved: Snapshot) {
        self.pending = saved.pending;
        self.history = saved.history;
        self.prev = saved.prev;
        self.prev_kbd = saved.prev_kbd;
        self.reservoir = saved.reservoir;
        self.frame_index = saved.frame_index;
        self.accepted = saved.accepted;
        self.sent_config = saved.sent_config;
    }
}

impl Encoder for AacEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Aac
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        self.push_samples(input)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        self.drain_samples()
    }

    fn reset(&mut self) {
        self.reset_audio();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

fn pcm_samples(config: &AudioConfig, input: &AudioInput) -> Result<Vec<f32>, CodecError> {
    let pcm = &input.pcm;
    if pcm.rate != config.sample_rate {
        return Err(CodecError::InvalidApi {
            what: "sample rate",
        });
    }
    if pcm.layout.channels() != config.layout.channels() {
        return Err(CodecError::InvalidApi { what: "layout" });
    }
    if let Some(samples) = pcm.as_f32() {
        return finite_f32(samples);
    }
    if let Some(samples) = pcm.as_i16() {
        return Ok(samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect());
    }
    Err(CodecError::InvalidApi {
        what: "sample format",
    })
}

fn finite_f32(samples: &[f32]) -> Result<Vec<f32>, CodecError> {
    let mut out = Vec::with_capacity(samples.len());
    for sample in samples {
        if !sample.is_finite() {
            return Err(CodecError::NonFinite);
        }
        if sample.abs() > 1.0 + FULL_SCALE_SLACK {
            return Err(CodecError::OutOfRange);
        }
        out.push(sample.clamp(-1.0, 1.0));
    }
    Ok(out)
}

fn choose_sequence(prev: Sequence, next_transient: bool) -> Sequence {
    match prev {
        Sequence::OnlyLong | Sequence::LongStop => {
            if next_transient {
                Sequence::LongStart
            } else {
                Sequence::OnlyLong
            }
        }
        Sequence::LongStart => Sequence::EightShort,
        Sequence::EightShort => {
            if next_transient {
                Sequence::EightShort
            } else {
                Sequence::LongStop
            }
        }
    }
}

fn interleaved_transient(samples: &[f32], channels: usize, frame: usize) -> bool {
    if channels == 0 || frame == 0 {
        return false;
    }
    let mut planes = vec![vec![0.0; frame]; channels];
    let have = samples.len() / channels;
    for index in 0..have.min(frame) {
        for channel in 0..channels {
            planes[channel][index] = samples[index * channels + channel];
        }
    }
    planes.iter().any(|plane| plane_transient(plane))
}

fn plane_transient(frame: &[f32]) -> bool {
    if frame.len() < 8 {
        return false;
    }
    let block = frame.len() / 8;
    let mut energy = [0.0f32; 8];
    for (index, sample) in frame.iter().enumerate() {
        energy[index / block] += sample * sample;
    }
    let max = energy.into_iter().fold(0.0f32, f32::max);
    let mean = energy.iter().sum::<f32>() / 8.0;
    max > 1.0e-4 && max > mean * 6.0
}

fn prepare_single(
    kind: ElementKind,
    spec: Vec<f32>,
    seq: Sequence,
    short: bool,
    config: &AudioConfig,
) -> Result<PreparedElement, CodecError> {
    let bands = config.bands(short)?;
    Ok(PreparedElement {
        kind,
        short,
        seq,
        num_swb: bands.len().saturating_sub(1),
        ms: Vec::new(),
        channels: vec![prepare_channel(spec, bands)?],
    })
}

fn prepare_channel(spec: Vec<f32>, bands: &[u16]) -> Result<PreparedChannel, CodecError> {
    let bases = band_bases(&spec, bands)?;
    Ok(PreparedChannel { spec, bases })
}

fn band_bases(spec: &[f32], bands: &[u16]) -> Result<Vec<Option<i32>>, CodecError> {
    let bins = usize::from(*bands.last().ok_or(CodecError::Unsupported {
        what: "sample rate",
    })?);
    if bins == 0 || !spec.len().is_multiple_of(bins) {
        return Err(CodecError::InvalidApi {
            what: "scale factor band",
        });
    }
    let windows = spec.len() / bins;
    let swb = bands.len().saturating_sub(1);
    let mut energy = vec![0.0f32; windows * swb];
    for window in 0..windows {
        for sfb in 0..swb {
            let start = usize::from(bands[sfb]);
            let end = usize::from(bands[sfb + 1]);
            let offset = window * bins;
            energy[window * swb + sfb] = spec[offset + start..offset + end]
                .iter()
                .map(|coef| coef * coef)
                .sum();
        }
    }
    let peak = energy.iter().copied().fold(0.0f32, f32::max);
    // A side channel left after mid/side can be numerical dust. Coding that
    // dust to a tight ratio spends the frame budget and coarsens the mid.
    if peak < 1.0e-5 * FULL_SCALE * FULL_SCALE {
        return Ok(vec![None; energy.len()]);
    }
    let floor = peak * 1.0e-4;
    let mut bases = Vec::with_capacity(energy.len());
    for (index, band_energy) in energy.iter().copied().enumerate() {
        if band_energy < 1.0e-8 * FULL_SCALE * FULL_SCALE || band_energy < floor {
            bases.push(None);
            continue;
        }
        let window = index / swb;
        let sfb = index % swb;
        let start = usize::from(bands[sfb]);
        let end = usize::from(bands[sfb + 1]);
        let offset = window * bins;
        bases.push(mask_sf(&spec[offset + start..offset + end]));
    }
    Ok(bases)
}

fn mask_sf(coeffs: &[f32]) -> Option<i32> {
    let energy = coeffs.iter().map(|coef| coef * coef).sum::<f32>();
    let peak = coeffs.iter().fold(0.0f32, |max, coef| max.max(coef.abs()));
    if energy < 1.0e-8 * FULL_SCALE * FULL_SCALE || peak < 1.0e-6 * FULL_SCALE {
        return None;
    }
    // 16^(4/3) is the dequant magnitude of quantizer step 16. Aiming the peak
    // there keeps the 4/3 law off both the zero step and the escape ceiling.
    // Distortion is not monotonic: too fine a scale clips, too coarse a scale
    // misses the threshold, so this walks outward from that guess.
    let ratio = (peak / 40.317_474).max(1.0e-8);
    let guess = 100.0 + 4.0 * ratio.log2();
    let mut sf = if guess.is_finite() {
        i32::try_from(guess.round() as i64)
            .unwrap_or(100)
            .clamp(0, 255)
    } else {
        100
    };
    while sf < 255 && clipped(coeffs, sf) {
        sf += 1;
    }
    let allowed = energy / 10_000.0;
    while sf < 255 {
        let next = sf + 1;
        if clipped(coeffs, next) || band_distortion(coeffs, next) > allowed {
            break;
        }
        sf = next;
    }
    Some(sf)
}

fn clipped(coeffs: &[f32], scale_factor: i32) -> bool {
    coeffs
        .iter()
        .any(|coef| quantize(*coef, scale_factor).abs() >= MAX_QUANT)
}

fn band_distortion(coeffs: &[f32], scale_factor: i32) -> f32 {
    coeffs
        .iter()
        .map(|coef| {
            let error = coef - dequant(quantize(*coef, scale_factor), scale_factor);
            error * error
        })
        .sum()
}

fn apply_ms(left: &mut [f32], right: &mut [f32], bands: &[u16]) -> Vec<bool> {
    let Some(&bins_u) = bands.last() else {
        return Vec::new();
    };
    let bins = usize::from(bins_u);
    if bins == 0 || left.len() != right.len() || !left.len().is_multiple_of(bins) {
        return Vec::new();
    }
    let windows = left.len() / bins;
    let swb = bands.len().saturating_sub(1);
    let mut mask = vec![false; windows * swb];
    for window in 0..windows {
        for sfb in 0..swb {
            let start = usize::from(bands[sfb]);
            let end = usize::from(bands[sfb + 1]);
            let offset = window * bins;
            if correlated(
                &left[offset + start..offset + end],
                &right[offset + start..offset + end],
            ) {
                for bin in start..end {
                    let l = left[offset + bin];
                    let r = right[offset + bin];
                    left[offset + bin] = 0.5 * (l + r);
                    right[offset + bin] = 0.5 * (l - r);
                }
                mask[window * swb + sfb] = true;
            }
        }
    }
    mask
}

fn correlated(left: &[f32], right: &[f32]) -> bool {
    let mut ll = 0.0f32;
    let mut rr = 0.0f32;
    let mut lr = 0.0f32;
    for (l, r) in left.iter().zip(right) {
        ll += l * l;
        rr += r * r;
        lr += l * r;
    }
    if ll < 1.0e-6 * FULL_SCALE * FULL_SCALE || rr < 1.0e-6 * FULL_SCALE * FULL_SCALE {
        return false;
    }
    let denom = ll.sqrt() * rr.sqrt();
    denom > 0.0 && (lr / denom).abs() > 0.5
}

fn lowpass_lfe(spec: &mut [f32], rate: u32, frame: usize) {
    // 500 Hz keeps the MDCT leakage of a low tone. A hard 300 Hz cut
    // removes that leakage and the overlap no longer reconstructs it.
    let width = f64::from(rate.max(1)) / f64::from(u32::try_from(frame).unwrap_or(1).max(1));
    let mut limit = 1usize;
    while limit < spec.len() && (limit as f64) * width < 500.0 {
        limit += 1;
    }
    for sample in spec.iter_mut().skip(limit) {
        *sample = 0.0;
    }
}

fn select_coded(
    config: &AudioConfig,
    prepared: &[PreparedElement],
    budget: u32,
    floor: i32,
) -> Result<Vec<CodedElement>, CodecError> {
    let mut lo = floor;
    let mut hi = floor + 160;
    let mut chosen = hi;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let bits = count_coded(config, prepared, mid, usize::MAX)?;
        if bits <= budget {
            chosen = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    for _ in 0..RATE_STEPS {
        let bits = count_coded(config, prepared, chosen, usize::MAX)?;
        if bits <= budget {
            return code_prepared(config, prepared, chosen, usize::MAX);
        }
        chosen += 1;
    }
    let limited = prepared.first().map(|el| el.num_swb / 2).unwrap_or(0);
    if count_coded(config, prepared, chosen, limited)? <= budget {
        return code_prepared(config, prepared, chosen, limited);
    }
    code_prepared(config, prepared, chosen, 0)
}

fn count_coded(
    config: &AudioConfig,
    prepared: &[PreparedElement],
    offset: i32,
    sfb_limit: usize,
) -> Result<u32, CodecError> {
    let coded = code_prepared(config, prepared, offset, sfb_limit)?;
    let mut counter = BitCounter { bits: 0 };
    write_elements(&mut counter, &coded)?;
    Ok(counter.bits)
}

fn code_prepared(
    config: &AudioConfig,
    prepared: &[PreparedElement],
    offset: i32,
    sfb_limit: usize,
) -> Result<Vec<CodedElement>, CodecError> {
    let mut out = Vec::with_capacity(prepared.len());
    for element in prepared {
        let bands = config.bands(element.short)?;
        let mut channels = Vec::with_capacity(element.channels.len());
        let channel_offset = if element.kind == ElementKind::Lfe {
            offset.saturating_sub(24)
        } else {
            offset
        };
        for channel in &element.channels {
            channels.push(code_channel(
                channel,
                bands,
                element.short,
                element.seq,
                channel_offset,
                sfb_limit,
            )?);
        }
        if channels.len() == 2 {
            let max_sfb = channels
                .iter()
                .map(|channel| channel.max_sfb)
                .max()
                .unwrap_or(0);
            for channel in &mut channels {
                pad_sfb(channel, max_sfb);
            }
        }
        out.push(CodedElement {
            kind: element.kind,
            num_swb: element.num_swb,
            ms: element.ms.clone(),
            channels,
        });
    }
    Ok(out)
}

fn code_channel(
    channel: &PreparedChannel,
    bands: &[u16],
    short: bool,
    seq: Sequence,
    offset: i32,
    sfb_limit: usize,
) -> Result<CodedChannel, CodecError> {
    let bins = usize::from(*bands.last().ok_or(CodecError::Unsupported {
        what: "sample rate",
    })?);
    let swb = bands.len().saturating_sub(1);
    let windows = channel.spec.len().checked_div(bins).unwrap_or(0);
    let mut books = vec![0u8; windows * swb];
    let mut scales = vec![0i32; windows * swb];
    let mut quant = vec![0i32; channel.spec.len()];
    let mut running: Option<i32> = None;
    for index in 0..windows * swb {
        let sfb = index % swb;
        if sfb >= sfb_limit {
            continue;
        }
        let Some(base) = channel.bases[index] else {
            continue;
        };
        let mut sf = (base + offset).clamp(0, 255);
        if let Some(prev) = running {
            sf = sf.clamp(prev - 60, prev + 60);
        }
        let window = index / swb;
        let start = usize::from(bands[sfb]);
        let end = usize::from(bands[sfb + 1]);
        let offset_bin = window * bins + start;
        let coeffs = &channel.spec[offset_bin..window * bins + end];
        let mut values = quantize_slice(coeffs, sf);
        while sf < 255 && values.iter().any(|value| value.abs() >= MAX_QUANT) {
            let next = sf + 1;
            if running.is_some_and(|prev| next > prev + 60) {
                break;
            }
            sf = next;
            values = quantize_slice(coeffs, sf);
        }
        if values.iter().all(|value| *value == 0) {
            continue;
        }
        let book = choose_book(&values)?;
        books[index] = book;
        scales[index] = sf;
        quant[offset_bin..offset_bin + values.len()].copy_from_slice(&values);
        running = Some(sf);
    }
    let edges = bands.iter().copied().map(usize::from).collect();
    compact_channel(SpreadChannel {
        books,
        scales,
        quant,
        edges,
        windows,
        swb,
        short,
        seq,
    })
}

fn quantize_slice(coeffs: &[f32], scale_factor: i32) -> Vec<i32> {
    coeffs
        .iter()
        .map(|coef| quantize(*coef, scale_factor))
        .collect()
}

struct SpreadChannel {
    books: Vec<u8>,
    scales: Vec<i32>,
    quant: Vec<i32>,
    edges: Vec<usize>,
    windows: usize,
    swb: usize,
    short: bool,
    seq: Sequence,
}

fn compact_channel(channel: SpreadChannel) -> Result<CodedChannel, CodecError> {
    let SpreadChannel {
        books,
        scales,
        quant,
        edges,
        windows,
        swb,
        short,
        seq,
    } = channel;
    let mut max_sfb = 0usize;
    for window in 0..windows {
        for sfb in 0..swb {
            if books[window * swb + sfb] != 0 {
                max_sfb = max_sfb.max(sfb + 1);
            }
        }
    }
    let limit = if short { 15 } else { 63 };
    if max_sfb > limit {
        return Err(CodecError::Limit {
            what: "scale factor band",
        });
    }
    let mut packed_books = vec![0u8; windows * max_sfb];
    let mut packed_scales = vec![0i32; windows * max_sfb];
    for window in 0..windows {
        for sfb in 0..max_sfb {
            packed_books[window * max_sfb + sfb] = books[window * swb + sfb];
            packed_scales[window * max_sfb + sfb] = scales[window * swb + sfb];
        }
    }
    Ok(CodedChannel {
        books: packed_books,
        scales: packed_scales,
        quant,
        edges,
        max_sfb,
        windows,
        short,
        seq,
    })
}

fn pad_sfb(channel: &mut CodedChannel, max_sfb: usize) {
    if channel.max_sfb == max_sfb {
        return;
    }
    let mut books = vec![0u8; channel.windows * max_sfb];
    let mut scales = vec![0i32; channel.windows * max_sfb];
    for window in 0..channel.windows {
        for sfb in 0..channel.max_sfb {
            books[window * max_sfb + sfb] = channel.books[window * channel.max_sfb + sfb];
            scales[window * max_sfb + sfb] = channel.scales[window * channel.max_sfb + sfb];
        }
    }
    channel.books = books;
    channel.scales = scales;
    channel.max_sfb = max_sfb;
}

fn choose_book(values: &[i32]) -> Result<u8, CodecError> {
    let max = values.iter().fold(0i32, |max, value| max.max(value.abs()));
    let mut candidates = candidate_books(max);
    if !values.len().is_multiple_of(4) {
        candidates.retain(|book| *book >= 5);
    }
    if candidates.is_empty() {
        candidates.push(11);
    }
    let mut best = 0u8;
    let mut best_bits = u32::MAX;
    for book in candidates {
        let mut counter = BitCounter { bits: 0 };
        write_values(&mut counter, book, values)?;
        if counter.bits < best_bits {
            best_bits = counter.bits;
            best = book;
        }
    }
    Ok(best)
}

fn candidate_books(max_abs: i32) -> Vec<u8> {
    match max_abs {
        0 => vec![0],
        1 => vec![1, 2],
        2 => vec![3, 4],
        3 | 4 => vec![5, 6],
        5..=7 => vec![7, 8],
        8..=12 => vec![9, 10],
        _ => vec![11],
    }
}

fn write_frame(elements: &[CodedElement]) -> Result<Vec<u8>, CodecError> {
    let mut writer = BitWriter::with_cap(MAX_ACCESS_UNIT);
    write_elements(&mut writer, elements)?;
    Ok(writer.finish())
}

fn write_elements<S: BitSink>(sink: &mut S, elements: &[CodedElement]) -> Result<(), CodecError> {
    let mut sce = 0u32;
    let mut cpe = 0u32;
    let mut lfe = 0u32;
    for element in elements {
        match element.kind {
            ElementKind::Sce => {
                write_single(sink, 0, sce, &element.channels[0])?;
                sce += 1;
            }
            ElementKind::Lfe => {
                write_single(sink, 3, lfe, &element.channels[0])?;
                lfe += 1;
            }
            ElementKind::Cpe => {
                write_pair(sink, cpe, element)?;
                cpe += 1;
            }
        }
    }
    sink.write_bits(7, 3)
}

fn write_single<S: BitSink>(
    sink: &mut S,
    id: u32,
    tag: u32,
    channel: &CodedChannel,
) -> Result<(), CodecError> {
    sink.write_bits(id, 3)?;
    sink.write_bits(tag, 4)?;
    write_payload(sink, channel, false)
}

fn write_pair<S: BitSink>(
    sink: &mut S,
    tag: u32,
    element: &CodedElement,
) -> Result<(), CodecError> {
    let left = &element.channels[0];
    let right = &element.channels[1];
    sink.write_bits(1, 3)?;
    sink.write_bits(tag, 4)?;
    sink.write_bits(1, 1)?;
    write_ics_info(sink, left)?;
    write_ms(
        sink,
        &element.ms,
        left.windows,
        left.max_sfb,
        element.num_swb,
    )?;
    write_payload(sink, left, true)?;
    write_payload(sink, right, true)
}

fn write_payload<S: BitSink>(
    sink: &mut S,
    channel: &CodedChannel,
    common: bool,
) -> Result<(), CodecError> {
    let gain = global_gain(channel);
    sink.write_bits(gain, 8)?;
    if !common {
        write_ics_info(sink, channel)?;
    }
    write_sections(sink, channel)?;
    write_scales(sink, channel)?;
    sink.write_bits(0, 1)?;
    sink.write_bits(0, 1)?;
    sink.write_bits(0, 1)?;
    write_spectral(sink, channel)
}

fn write_ics_info<S: BitSink>(sink: &mut S, channel: &CodedChannel) -> Result<(), CodecError> {
    sink.write_bits(0, 1)?;
    sink.write_bits(channel.seq.bits(), 2)?;
    sink.write_bits(1, 1)?;
    let width = if channel.short { 4 } else { 6 };
    sink.write_bits(u32::try_from(channel.max_sfb).unwrap_or(0), width)?;
    if channel.short {
        sink.write_bits(0, 7)
    } else {
        sink.write_bits(0, 1)
    }
}

fn write_ms<S: BitSink>(
    sink: &mut S,
    ms: &[bool],
    windows: usize,
    max_sfb: usize,
    num_swb: usize,
) -> Result<(), CodecError> {
    let mut bits = Vec::with_capacity(windows.saturating_mul(max_sfb));
    for window in 0..windows {
        for sfb in 0..max_sfb {
            let used = ms.get(window * num_swb + sfb).copied().unwrap_or(false);
            bits.push(used);
        }
    }
    if bits.iter().all(|bit| !*bit) {
        return sink.write_bits(0, 2);
    }
    if bits.iter().all(|bit| *bit) {
        return sink.write_bits(2, 2);
    }
    sink.write_bits(1, 2)?;
    for bit in bits {
        sink.write_bits(u32::from(bit), 1)?;
    }
    Ok(())
}

fn write_sections<S: BitSink>(sink: &mut S, channel: &CodedChannel) -> Result<(), CodecError> {
    if channel.max_sfb == 0 {
        return Ok(());
    }
    let sect_bits = if channel.short { 3 } else { 5 };
    let escape = (1usize << sect_bits) - 1;
    for window in 0..channel.windows {
        let row = &channel.books[window * channel.max_sfb..(window + 1) * channel.max_sfb];
        let mut sfb = 0usize;
        while sfb < channel.max_sfb {
            let book = row[sfb];
            let mut len = 1usize;
            while sfb + len < channel.max_sfb && row[sfb + len] == book {
                len += 1;
            }
            sink.write_bits(u32::from(book), 4)?;
            write_section_len(sink, len, sect_bits, escape)?;
            sfb += len;
        }
    }
    Ok(())
}

fn write_section_len<S: BitSink>(
    sink: &mut S,
    mut len: usize,
    bits: u32,
    escape: usize,
) -> Result<(), CodecError> {
    loop {
        let chunk = len.min(escape);
        sink.write_bits(u32::try_from(chunk).unwrap_or(0), bits)?;
        if chunk < escape {
            return Ok(());
        }
        len -= chunk;
        if len == 0 {
            return sink.write_bits(0, bits);
        }
    }
}

fn global_gain(channel: &CodedChannel) -> u32 {
    channel
        .books
        .iter()
        .zip(&channel.scales)
        .find(|(book, _)| (1..=11).contains(*book))
        .map(|(_, sf)| u32::try_from(*sf).unwrap_or(0))
        .unwrap_or(0)
}

fn write_scales<S: BitSink>(sink: &mut S, channel: &CodedChannel) -> Result<(), CodecError> {
    let mut running = i32::try_from(global_gain(channel)).unwrap_or(0);
    let mut started = false;
    for (book, sf) in channel.books.iter().zip(&channel.scales) {
        if !(1..=11).contains(book) {
            continue;
        }
        let delta = if started { *sf - running } else { 0 };
        let (code, len) = scale_code(delta)?;
        write_code(sink, code, len)?;
        running = *sf;
        started = true;
    }
    Ok(())
}

fn write_spectral<S: BitSink>(sink: &mut S, channel: &CodedChannel) -> Result<(), CodecError> {
    if channel.max_sfb == 0 || channel.windows == 0 {
        return Ok(());
    }
    let bins = channel.quant.len() / channel.windows;
    for window in 0..channel.windows {
        for sfb in 0..channel.max_sfb {
            let book = channel.books[window * channel.max_sfb + sfb];
            let start = channel.edges[sfb];
            let end = channel.edges[sfb + 1];
            let offset = window * bins + start;
            write_values(sink, book, &channel.quant[offset..window * bins + end])?;
        }
    }
    Ok(())
}

fn write_values<S: BitSink>(sink: &mut S, book: u8, values: &[i32]) -> Result<(), CodecError> {
    match book {
        0 => Ok(()),
        1 | 2 => write_quads(sink, book, values, true),
        3 | 4 => write_quads(sink, book, values, false),
        5 | 6 => write_signed_pairs(sink, book, values),
        7..=10 => write_unsigned_pairs(sink, book, values, false),
        11 => write_unsigned_pairs(sink, book, values, true),
        _ => Err(CodecError::Limit { what: "codebook" }),
    }
}

fn write_quads<S: BitSink>(
    sink: &mut S,
    book: u8,
    values: &[i32],
    signed: bool,
) -> Result<(), CodecError> {
    for chunk in values.chunks(4) {
        if chunk.len() != 4 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "spectral",
            });
        }
        let digits = if signed {
            [chunk[0] + 1, chunk[1] + 1, chunk[2] + 1, chunk[3] + 1]
        } else {
            [
                chunk[0].abs(),
                chunk[1].abs(),
                chunk[2].abs(),
                chunk[3].abs(),
            ]
        };
        let (code, len) = spectral_code(usize::from(book - 1), quad_index(digits))?;
        write_code(sink, code, len)?;
        if !signed {
            write_signs(sink, chunk)?;
        }
    }
    Ok(())
}

fn write_signed_pairs<S: BitSink>(
    sink: &mut S,
    book: u8,
    values: &[i32],
) -> Result<(), CodecError> {
    for chunk in values.chunks(2) {
        if chunk.len() != 2 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "spectral",
            });
        }
        let index = signed_pair_index(chunk[0], chunk[1]);
        let (code, len) = spectral_code(usize::from(book - 1), index)?;
        write_code(sink, code, len)?;
    }
    Ok(())
}

fn write_unsigned_pairs<S: BitSink>(
    sink: &mut S,
    book: u8,
    values: &[i32],
    escape: bool,
) -> Result<(), CodecError> {
    let modulus = if book <= 8 {
        8
    } else if book <= 10 {
        13
    } else {
        17
    };
    for chunk in values.chunks(2) {
        if chunk.len() != 2 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "spectral",
            });
        }
        let digits = [pair_digit(chunk[0], escape), pair_digit(chunk[1], escape)];
        let index = pair_index(digits[0], digits[1], modulus);
        let (code, len) = spectral_code(usize::from(book - 1), index)?;
        write_code(sink, code, len)?;
        write_signs(sink, chunk)?;
        if escape {
            write_escape_values(sink, chunk)?;
        }
    }
    Ok(())
}

fn pair_digit(value: i32, escape: bool) -> i32 {
    let magnitude = value.abs();
    if escape && magnitude >= 16 {
        16
    } else {
        magnitude
    }
}

fn write_signs<S: BitSink>(sink: &mut S, values: &[i32]) -> Result<(), CodecError> {
    for value in values {
        if *value != 0 {
            sink.write_bits(u32::from(*value < 0), 1)?;
        }
    }
    Ok(())
}

fn write_escape_values<S: BitSink>(sink: &mut S, values: &[i32]) -> Result<(), CodecError> {
    for value in values {
        if value.abs() >= 16 {
            write_escape(sink, value.unsigned_abs())?;
        }
    }
    Ok(())
}

fn stamp_padding(packet: &mut Packet, padding: u32) {
    for side in &mut packet.side {
        if let SideData::Gapless {
            trailing_padding, ..
        } = side
        {
            *trailing_padding = padding;
        }
    }
}

fn frame_target(bitrate: u32, frame: usize, rate: u32) -> i32 {
    i32::try_from(target_u32(bitrate, frame, rate)).unwrap_or(i32::MAX)
}

fn target_u32(bitrate: u32, frame: usize, rate: u32) -> u32 {
    let frame = u64::try_from(frame).unwrap_or(1);
    let bits = u64::from(bitrate).saturating_mul(frame) / u64::from(rate.max(1));
    u32::try_from(bits).unwrap_or(u32::MAX)
}

fn clamp_i32(value: i32, lo: i32, hi: i32) -> i32 {
    value.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::choose_sequence;
    use crate::bits::BitReader;
    use crate::config::{EncoderConfig, Profile, RateControl};
    use crate::decode::AacDecoder;
    use crate::encode::AacEncoder;
    use crate::filterbank::Sequence;
    use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, SideData};
    use cutlass_encode::{AudioInput, Encoder};

    fn params(rate: u32, layout: ChannelLayout, bitrate: u32) -> CodecParams {
        CodecParams {
            bitrate,
            ..CodecParams::audio(CodecId::Aac, rate, layout)
        }
    }

    fn sine(rate: u32, count: usize, freq: f32, amp: f32, channels: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count * channels);
        for n in 0..count {
            let sample = amp * (2.0 * std::f32::consts::PI * freq * n as f32 / rate as f32).sin();
            for _ in 0..channels {
                out.push(sample);
            }
        }
        out
    }

    fn round_trip(encoder: &mut AacEncoder, samples: Vec<f32>) -> (Vec<f32>, u32, u32) {
        let layout = encoder.config().layout.clone();
        let rate = encoder.config().sample_rate;
        let pcm = cutlass_decode::Pcm::from_f32(rate, layout.clone(), samples).unwrap();
        let mut packets = encoder.push(&AudioInput { pcm }).unwrap();
        packets.extend(encoder.drain().unwrap());
        assert!(encoder.drain().unwrap().is_empty());
        let mut params = CodecParams::audio(CodecId::Aac, rate, layout);
        params.extra = encoder.config().asc.clone();
        let mut decoder = AacDecoder::open(&params).unwrap();
        let mut decoded = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).unwrap() {
                decoded.extend_from_slice(output.pcm.as_f32().unwrap());
            }
        }
        let (delay, padding) = gapless(packets.last().unwrap());
        let channels = encoder.config().layout.len();
        let start = delay as usize * channels;
        let end = decoded.len() - padding as usize * channels;
        (decoded[start..end].to_vec(), delay, padding)
    }

    fn gapless(packet: &cutlass_decode::Packet) -> (u32, u32) {
        packet
            .side
            .iter()
            .find_map(|side| match side {
                SideData::Gapless {
                    encoder_delay,
                    trailing_padding,
                    ..
                } => Some((*encoder_delay, *trailing_padding)),
                SideData::Config(_) | SideData::Protection { .. } => None,
            })
            .unwrap()
    }

    fn snr(reference: &[f32], decoded: &[f32]) -> f32 {
        let mut signal = 0.0f64;
        let mut noise = 0.0f64;
        for (original, coded) in reference.iter().zip(decoded) {
            let original = f64::from(*original);
            let coded = f64::from(*coded);
            signal += original * original;
            let error = original - coded;
            noise += error * error;
        }
        if noise == 0.0 {
            return 120.0;
        }
        (10.0 * (signal / noise).log10()) as f32
    }

    #[test]
    fn a_transient_opens_a_short_window() {
        assert_eq!(
            choose_sequence(Sequence::OnlyLong, true),
            Sequence::LongStart
        );
        assert_eq!(
            choose_sequence(Sequence::LongStart, false),
            Sequence::EightShort
        );
        assert_eq!(
            choose_sequence(Sequence::EightShort, false),
            Sequence::LongStop
        );
        assert_eq!(
            choose_sequence(Sequence::LongStop, false),
            Sequence::OnlyLong
        );
        let mut samples = vec![0.0; 2048];
        samples[1200] = 1.0;
        let mut encoder = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 96_000)).unwrap();
        let pcm = cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), samples).unwrap();
        let packets = encoder.push(&AudioInput { pcm }).unwrap();
        assert_eq!(packets.len(), 1);
        let mut bits = BitReader::new(&packets[0].data);
        assert_eq!(bits.read(3).unwrap(), 0);
        let _tag = bits.read(4).unwrap();
        let _gain = bits.read(8).unwrap();
        assert_eq!(bits.read(1).unwrap(), 0);
        assert_eq!(bits.read(2).unwrap(), Sequence::LongStart.bits());
    }

    #[test]
    fn silence_stays_below_the_noise_floor() {
        let mut encoder =
            AacEncoder::open(&params(48_000, ChannelLayout::mono(), 128_000)).unwrap();
        let (decoded, delay, _) = round_trip(&mut encoder, vec![0.0; 2048]);
        assert_eq!(delay, 1024);
        assert_eq!(decoded.len(), 2048);
        assert!(decoded.iter().all(|sample| sample.abs() < 1.0e-4));
    }

    #[test]
    fn a_sine_clears_the_mono_snr_gate() {
        let samples = sine(48_000, 4096, 1_000.0, 0.5, 1);
        let mut encoder = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 96_000)).unwrap();
        let (decoded, _, _) = round_trip(&mut encoder, samples.clone());
        assert_eq!(decoded.len(), samples.len());
        let score = snr(&samples, &decoded);
        assert!(score >= 40.0, "snr {score}");
    }

    #[test]
    fn a_stereo_sine_clears_the_per_channel_gate() {
        let samples = sine(48_000, 4096, 1_000.0, 0.5, 2);
        let mut encoder =
            AacEncoder::open(&params(48_000, ChannelLayout::stereo(), 128_000)).unwrap();
        let (decoded, _, _) = round_trip(&mut encoder, samples.clone());
        assert_eq!(decoded.len(), samples.len());
        for channel in 0..2 {
            let reference: Vec<f32> = samples.iter().skip(channel).step_by(2).copied().collect();
            let coded: Vec<f32> = decoded.iter().skip(channel).step_by(2).copied().collect();
            let score = snr(&reference, &coded);
            assert!(score >= 35.0, "channel {channel} snr {score}");
        }
    }

    #[test]
    fn illegal_samples_leave_the_encoder_empty() {
        let mut encoder = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 64_000)).unwrap();
        let nan = cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), vec![f32::NAN; 64])
            .unwrap();
        let err = encoder.push(&AudioInput { pcm: nan }).unwrap_err();
        assert_eq!(err, cutlass_decode::CodecError::NonFinite);
        let huge =
            cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), vec![2.0; 64]).unwrap();
        let err = encoder.push(&AudioInput { pcm: huge }).unwrap_err();
        assert_eq!(err, cutlass_decode::CodecError::OutOfRange);
        let (decoded, _, _) = round_trip(&mut encoder, vec![0.0; 1024]);
        assert_eq!(decoded.len(), 1024);
    }

    #[test]
    fn the_same_samples_encode_to_the_same_bytes() {
        let samples = sine(48_000, 2048, 440.0, 0.25, 1);
        let mut first = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 64_000)).unwrap();
        let mut second = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 64_000)).unwrap();
        let pcm = |encoder: &AacEncoder| {
            cutlass_decode::Pcm::from_f32(48_000, encoder.config().layout.clone(), samples.clone())
                .unwrap()
        };
        let mut a = first.push(&AudioInput { pcm: pcm(&first) }).unwrap();
        a.extend(first.drain().unwrap());
        let mut b = second.push(&AudioInput { pcm: pcm(&second) }).unwrap();
        b.extend(second.drain().unwrap());
        let left: Vec<&[u8]> = a.iter().map(|packet| packet.data.as_slice()).collect();
        let right: Vec<&[u8]> = b.iter().map(|packet| packet.data.as_slice()).collect();
        assert_eq!(left, right);
    }

    #[test]
    fn variable_quality_rejects_zero_and_eleven() {
        let err = EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 48_000,
            layout: ChannelLayout::mono(),
            frame_length: 1024,
            rate: RateControl::Variable {
                quality: 0,
                cap: None,
            },
        }
        .audio()
        .unwrap_err();
        assert!(matches!(err, cutlass_decode::CodecError::InvalidApi { .. }));
    }

    #[test]
    fn ffmpeg_decodes_our_access_units() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            require_or_skip("ffmpeg");
            return;
        };
        let samples = sine(48_000, 8_192, 1_000.0, 0.5, 1);
        let mut encoder = AacEncoder::open(&params(48_000, ChannelLayout::mono(), 96_000)).unwrap();
        let pcm =
            cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), samples.clone()).unwrap();
        let mut packets = encoder.push(&AudioInput { pcm }).unwrap();
        packets.extend(encoder.drain().unwrap());
        let adts = packets
            .iter()
            .flat_map(|packet| adts_frame(&packet.data))
            .collect::<Vec<_>>();
        let foreign = ffmpeg_pcm(&ffmpeg, &adts);
        let mut decoder = AacDecoder::open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        ))
        .unwrap();
        let mut ours = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).unwrap() {
                ours.extend_from_slice(output.pcm.as_f32().unwrap());
            }
        }
        if let Some(tail) = decoder.drain().unwrap().pop() {
            ours.extend_from_slice(tail.pcm.as_f32().unwrap());
        }
        let (score, lag, ratio) = aligned_snr(&ours, &foreign);
        assert!(
            score >= 40.0 && lag <= 1024,
            "snr {score} lag {lag} rms ratio {ratio}"
        );
    }

    #[test]
    fn we_decode_an_ffmpeg_stream() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            require_or_skip("ffmpeg");
            return;
        };
        let dir = std::env::temp_dir().join(format!("cutlass-aac-oracle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let adts = dir.join("ffmpeg.aac");
        let status = std::process::Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=0.25",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-f",
                "adts",
            ])
            .arg(&adts)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let bytes = std::fs::read(&adts).unwrap();
        let foreign = ffmpeg_pcm(&ffmpeg, &bytes);
        let units = split_adts(&bytes);
        let mut decoder = AacDecoder::open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        ))
        .unwrap();
        let mut ours = Vec::new();
        for unit in &units {
            let packet = cutlass_decode::Packet::audio(unit.clone(), 0, 1024, 48_000);
            ours.extend(decoder.push_packet(&packet).unwrap());
        }
        if let Some(tail) = decoder.drain_pcm().unwrap() {
            ours.extend(tail);
        }
        let (score, lag, ratio) = aligned_snr(&ours, &foreign);
        assert!(
            score >= 40.0 && lag <= 1024,
            "snr {score} lag {lag} rms ratio {ratio} frames {}",
            units.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn require_or_skip(tool: &str) {
        if std::env::var_os("AAC_REQUIRE_ORACLES").is_some() {
            panic!("{tool} is required");
        }
        eprintln!("skipped {tool} oracle: {tool} was not found");
    }

    fn ffmpeg_bin() -> Option<std::path::PathBuf> {
        for candidate in [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ] {
            let path = std::path::PathBuf::from(candidate);
            if std::process::Command::new(&path)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(path);
            }
        }
        None
    }

    fn ffmpeg_pcm(ffmpeg: &std::path::Path, adts: &[u8]) -> Vec<f32> {
        let dir = std::env::temp_dir().join(format!(
            "cutlass-aac-pcm-{}-{}",
            std::process::id(),
            adts.len()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.aac");
        let output = dir.join("out.f32");
        std::fs::write(&input, adts).unwrap();
        let status = std::process::Command::new(ffmpeg)
            .args(["-y", "-i"])
            .arg(&input)
            .args(["-f", "f32le", "-ac", "1", "-ar", "48000"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(status.status.success(), "{status:?}");
        let bytes = std::fs::read(&output).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn adts_frame(payload: &[u8]) -> Vec<u8> {
        let frame_len = payload.len() + 7;
        let mut bytes = vec![0u8; frame_len];
        bytes[0] = 0xFF;
        bytes[1] = 0xF1;
        bytes[2] = (1 << 6) | (3 << 2);
        bytes[3] = (1 << 6) | u8::try_from((frame_len >> 11) & 0b11).unwrap_or(0);
        bytes[4] = u8::try_from((frame_len >> 3) & 0xFF).unwrap_or(0);
        bytes[5] = u8::try_from((frame_len & 0b111) << 5).unwrap_or(0) | 0x1F;
        bytes[6] = 0xFC;
        bytes[7..].copy_from_slice(payload);
        bytes
    }

    fn split_adts(mut data: &[u8]) -> Vec<Vec<u8>> {
        let mut units = Vec::new();
        while data.len() >= 7 {
            assert_eq!(data[0], 0xFF);
            let frame_len = (usize::from(data[3] & 0b11) << 11)
                | (usize::from(data[4]) << 3)
                | (usize::from(data[5]) >> 5);
            assert!(frame_len >= 7 && frame_len <= data.len());
            units.push(data[7..frame_len].to_vec());
            data = &data[frame_len..];
        }
        units
    }

    fn aligned_snr(ours: &[f32], foreign: &[f32]) -> (f32, usize, f32) {
        let mut best = (f32::MIN, 0usize, 0.0f32);
        let limit = 1024.min(ours.len()).min(foreign.len());
        for lag in 0..=limit {
            let (left, right) = if ours.len() >= foreign.len() {
                (&ours[lag..], foreign)
            } else {
                (ours, &foreign[lag..])
            };
            let count = left.len().min(right.len());
            if count < 1024 {
                continue;
            }
            let score = snr(&left[..count], &right[..count]);
            if score > best.0 {
                let left_rms = rms(&left[..count]);
                let right_rms = rms(&right[..count]);
                let ratio = if right_rms == 0.0 {
                    0.0
                } else {
                    left_rms / right_rms
                };
                best = (score, lag, ratio);
            }
        }
        best
    }

    fn rms(samples: &[f32]) -> f32 {
        let energy = samples.iter().map(|sample| sample * sample).sum::<f32>();
        (energy / samples.len() as f32).sqrt()
    }

    fn configured(frame: usize, layout: ChannelLayout, bitrate: u32) -> AacEncoder {
        AacEncoder::from_config(EncoderConfig {
            profile: Profile::AacLc,
            mpeg2: false,
            sample_rate: 48_000,
            layout,
            frame_length: frame,
            rate: RateControl::Average { bitrate },
        })
        .unwrap()
    }

    #[test]
    fn a_nine_sixty_frame_clears_the_mono_gate() {
        let samples = sine(48_000, 960 * 4, 1_000.0, 0.5, 1);
        let mut encoder = configured(960, ChannelLayout::mono(), 96_000);
        let (decoded, delay, _) = round_trip(&mut encoder, samples.clone());
        assert_eq!(delay, 960);
        assert_eq!(decoded.len(), samples.len());
        let score = snr(&samples, &decoded);
        assert!(score >= 40.0, "snr {score}");
    }

    #[test]
    fn dc_stays_near_the_input() {
        let samples = vec![0.25; 4096];
        let mut encoder = configured(1024, ChannelLayout::mono(), 96_000);
        let (decoded, _, _) = round_trip(&mut encoder, samples);
        let mean = decoded.iter().sum::<f32>() / decoded.len() as f32;
        assert!((mean - 0.25).abs() < 0.02, "mean {mean}");
    }

    #[test]
    fn a_near_nyquist_sine_stays_above_the_floor() {
        let samples = sine(48_000, 4096, 0.45 * 48_000.0, 0.5, 2);
        let mut encoder = configured(1024, ChannelLayout::stereo(), 128_000);
        let (decoded, _, _) = round_trip(&mut encoder, samples.clone());
        for channel in 0..2 {
            let reference: Vec<f32> = samples.iter().skip(channel).step_by(2).copied().collect();
            let coded: Vec<f32> = decoded.iter().skip(channel).step_by(2).copied().collect();
            let score = snr(&reference, &coded);
            assert!(score >= 20.0, "channel {channel} snr {score}");
        }
    }

    #[test]
    fn five_one_keeps_each_channel_and_limits_the_lfe() {
        let layout = ChannelLayout::five_one();
        let channels = layout.len();
        let mut samples = Vec::new();
        for n in 0..4096 {
            let tone = 0.4 * (2.0 * std::f32::consts::PI * 1_000.0 * n as f32 / 48_000.0).sin();
            let low = 0.4 * (2.0 * std::f32::consts::PI * 80.0 * n as f32 / 48_000.0).sin();
            for (index, label) in layout.channels().iter().enumerate() {
                let _ = index;
                samples.push(if *label == cutlass_decode::Channel::Lfe {
                    low
                } else {
                    tone
                });
            }
        }
        let mut encoder = configured(1024, layout.clone(), 384_000);
        let (decoded, _, _) = round_trip(&mut encoder, samples.clone());
        assert_eq!(decoded.len(), samples.len());
        for (index, label) in layout.channels().iter().enumerate() {
            let reference: Vec<f32> = samples
                .iter()
                .skip(index)
                .step_by(channels)
                .copied()
                .collect();
            let coded: Vec<f32> = decoded
                .iter()
                .skip(index)
                .step_by(channels)
                .copied()
                .collect();
            let score = snr(&reference, &coded);
            let floor = if *label == cutlass_decode::Channel::Lfe {
                25.0
            } else {
                30.0
            };
            assert!(score >= floor, "{label:?} snr {score}");
        }
    }

    #[test]
    fn seven_one_round_trips_a_distinct_tone_per_channel() {
        let layout = ChannelLayout::seven_one();
        let channels = layout.len();
        let mut samples = Vec::new();
        for n in 0..4096 {
            for (index, label) in layout.channels().iter().enumerate() {
                let freq = if *label == cutlass_decode::Channel::Lfe {
                    70.0
                } else {
                    300.0 + 120.0 * index as f32
                };
                samples
                    .push(0.25 * (2.0 * std::f32::consts::PI * freq * n as f32 / 48_000.0).sin());
            }
        }
        let mut encoder = configured(1024, layout.clone(), 512_000);
        let (decoded, _, _) = round_trip(&mut encoder, samples.clone());
        for (index, label) in layout.channels().iter().enumerate() {
            let reference: Vec<f32> = samples
                .iter()
                .skip(index)
                .step_by(channels)
                .copied()
                .collect();
            let coded: Vec<f32> = decoded
                .iter()
                .skip(index)
                .step_by(channels)
                .copied()
                .collect();
            let score = snr(&reference, &coded);
            assert!(score >= 25.0, "{label:?} snr {score}");
        }
    }

    #[test]
    fn noise_over_two_seconds_stays_inside_the_rate_window() {
        let mut state = 0x1234_5678u32;
        let mut samples = Vec::with_capacity(48_000 * 2);
        for _ in 0..48_000 * 2 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / ((1u32 << 24) as f32);
            samples.push(unit * 2.0 - 1.0);
        }
        let mut encoder = configured(1024, ChannelLayout::mono(), 64_000);
        let pcm = cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), samples).unwrap();
        let mut packets = encoder.push(&AudioInput { pcm }).unwrap();
        packets.extend(encoder.drain().unwrap());
        let bits: usize = packets.iter().map(|packet| packet.data.len() * 8).sum();
        let seconds = 2.0;
        let rate = bits as f32 / seconds;
        assert!(rate <= 64_000.0 * 1.10, "rate {rate}");
        assert!(rate >= 64_000.0 * 0.70, "rate {rate}");
    }

    #[test]
    fn integer_pcm_uses_the_sixteen_bit_full_scale() {
        let mut samples = Vec::new();
        for n in 0..4096 {
            let unit = (2.0 * std::f32::consts::PI * 1_000.0 * n as f32 / 48_000.0).sin();
            samples.push((unit * 0.5 * 32768.0).round() as i16);
        }
        let reference: Vec<f32> = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        let mut encoder = configured(1024, ChannelLayout::mono(), 96_000);
        let pcm = cutlass_decode::Pcm::from_i16(48_000, ChannelLayout::mono(), samples).unwrap();
        let mut packets = encoder.push(&AudioInput { pcm }).unwrap();
        packets.extend(encoder.drain().unwrap());
        let mut decoder = AacDecoder::open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        ))
        .unwrap();
        let mut decoded = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).unwrap() {
                decoded.extend_from_slice(output.pcm.as_f32().unwrap());
            }
        }
        if let Some(tail) = decoder.drain().unwrap().pop() {
            decoded.extend_from_slice(tail.pcm.as_f32().unwrap());
        }
        let (delay, padding) = gapless(packets.last().unwrap());
        let start = delay as usize;
        let end = decoded.len() - padding as usize;
        let score = snr(&reference, &decoded[start..end]);
        assert!(score >= 40.0, "snr {score}");
    }
}
