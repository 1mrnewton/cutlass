//! AC-3 and E-AC-3 encoding (A/52 section 8, Annex E): transient
//! detection, coupling, rematrixing, exponent strategies, the parametric
//! bit allocation searched for the largest SNR offset that fits, and
//! constant bit rate syncframes.
//!
//! One independent substream: mono to 5.1 with LFE at 32, 44.1 and
//! 48 kHz. E-AC-3 takes any bit rate and 1, 2, 3 or 6 blocks per
//! syncframe. Spectral extension, AHT, enhanced coupling, dependent
//! substreams, transient pre-noise processing and dynamic range words are
//! never written.

mod analysis;
mod plan;
mod write;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, Channel, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput,
    Packet, SideData,
};

use crate::bitalloc::{self, Mask, Params};
use crate::header::{Dac3, Dec3, Dec3Substream};
use crate::mdct::{COEFS, Transform};
use crate::tables::{FRAME_WORDS, NOMINAL_KBPS, REUSE};
use analysis::Analysis;
use plan::{Block, LFE, MAX_BLOCKS, Mode, SLOTS, Shape};
pub use write::MixCodes;
use write::{Frame, Layout, Meta, TAIL_BITS};

/// Samples the decoded stream lags the input: one half transform.
pub const ENCODER_DELAY: u32 = 256;

/// Rounding slack on full scale, as the other encoders allow.
const FULL_SCALE_SLACK: f32 = 16.0 * f32::EPSILON;

/// Largest E-AC-3 syncframe: `frmsiz` is 11 bits of words.
const MAX_EAC3_WORDS: usize = 2048;

/// Highest `(csnroffst << 4) | fsnroffst`.
const MAX_SNR: i32 = 1023;

/// `csnroffst` 15, plus eight coarse steps. A short frame that cannot
/// reach this gives bandwidth back until it can, or until none is left.
const NOMINAL_SNR: i32 = (15 << 4) + 128;

/// `fgaincod` every channel sends: the default fast gain.
const FGAINCOD: u32 = 4;

/// Audio bandwidth in Hz by kbit/s per full bandwidth channel.
const CUTOFF: [(u32, u32); 8] = [
    (16, 4_000),
    (24, 6_000),
    (32, 8_500),
    (48, 12_000),
    (64, 15_000),
    (80, 17_000),
    (96, 18_500),
    (112, 20_000),
];

/// Coupling begin frequency in Hz by kbit/s per full bandwidth channel.
/// At [`NO_COUPLING_KBPS`] and above, channels are coded independently.
const COUPLING: [(u32, u32); 6] = [
    (16, 3_000),
    (32, 4_500),
    (48, 7_000),
    (64, 10_000),
    (80, 12_000),
    (96, 14_000),
];
const NO_COUPLING_KBPS: u32 = 112;

/// Which syntax the encoder writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Ac3,
    Eac3,
}

/// Program metadata every syncframe carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderMetadata {
    /// `dialnorm` 1 to 31: dialogue level -1 to -31 dBFS.
    pub dialnorm: u8,
    /// `bsmod` 0 to 7 (Table 5.7).
    pub bsmod: u8,
    /// `dsurmod` of a 2/0 program: 0 not indicated, 1 not Dolby Surround
    /// encoded, 2 Dolby Surround encoded.
    pub dsurmod: u8,
    /// Downmix levels and preference. AC-3 writes them as `xbsi1` with
    /// `bsid` 6.
    pub mix: Option<MixCodes>,
    /// E-AC-3 `lfemixlevcod` 0 to 31: LFE at 10 - code dB in a downmix.
    pub lfe_mix: Option<u8>,
}

impl Default for EncoderMetadata {
    fn default() -> Self {
        Self {
            dialnorm: 31,
            bsmod: 0,
            dsurmod: 0,
            mix: None,
            lfe_mix: None,
        }
    }
}

/// Everything an encoder is opened with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    pub format: Format,
    /// 32 000, 44 100 or 48 000.
    pub sample_rate: u32,
    /// The order of interleaved input. One channel is coded as center.
    pub layout: ChannelLayout,
    /// Bits per second. AC-3 takes a rate of Table 5.18; zero picks the
    /// default for the channel count.
    pub bit_rate: u32,
    /// E-AC-3 audio blocks per syncframe: 1, 2, 3 or 6. AC-3 is 6.
    pub blocks: usize,
    pub metadata: EncoderMetadata,
}

impl EncoderConfig {
    #[must_use]
    pub fn new(format: Format, sample_rate: u32, layout: ChannelLayout) -> Self {
        Self {
            format,
            sample_rate,
            layout,
            bit_rate: 0,
            blocks: MAX_BLOCKS,
            metadata: EncoderMetadata::default(),
        }
    }
}

/// The default bit rate for `nf` full bandwidth channels, plus LFE.
#[must_use]
pub fn default_bit_rate(format: Format, nf: usize, lfe: bool) -> u32 {
    let kbps = match (format, nf, lfe) {
        (Format::Ac3, 5, true) => 448,
        (Format::Ac3, 1, _) => 96,
        (Format::Ac3, 2, _) => 192,
        (Format::Ac3, 3, _) => 256,
        (Format::Ac3, 4, _) => 320,
        (Format::Ac3, _, _) => 384,
        (Format::Eac3, 5, true) => 384,
        (Format::Eac3, 1, _) => 64,
        (Format::Eac3, 2, _) => 128,
        (Format::Eac3, 3, _) => 192,
        (Format::Eac3, 4, _) => 224,
        (Format::Eac3, _, _) => 256,
    };
    kbps * 1000
}

/// `acmod`, `lfeon`, and the input channel of each coded channel: the full
/// bandwidth channels in A/52 order, then LFE.
fn coding(layout: &ChannelLayout) -> Result<(u8, bool, Vec<usize>), CodecError> {
    let channels = layout.channels();
    let find = |wanted: Channel| channels.iter().position(|c| *c == wanted);
    let either = |a: Channel, b: Channel| match (find(a), find(b)) {
        (Some(_), Some(_)) => Err(CodecError::Unsupported {
            what: "side and back surround pairs together",
        }),
        (a, b) => Ok(a.or(b)),
    };
    let lfe = find(Channel::Lfe);
    let (acmod, order) = if channels.len() == 1 && lfe.is_none() {
        (1, vec![0])
    } else {
        let l = find(Channel::FrontLeft);
        let r = find(Channel::FrontRight);
        let c = find(Channel::FrontCenter);
        let cs = find(Channel::BackCenter);
        let ls = either(Channel::SideLeft, Channel::BackLeft)?;
        let rs = either(Channel::SideRight, Channel::BackRight)?;
        match (l, r, c, cs, ls, rs) {
            (None, None, Some(c), None, None, None) => (1, vec![c]),
            (Some(l), Some(r), None, None, None, None) => (2, vec![l, r]),
            (Some(l), Some(r), Some(c), None, None, None) => (3, vec![l, c, r]),
            (Some(l), Some(r), None, Some(s), None, None) => (4, vec![l, r, s]),
            (Some(l), Some(r), Some(c), Some(s), None, None) => (5, vec![l, c, r, s]),
            (Some(l), Some(r), None, None, Some(ls), Some(rs)) => (6, vec![l, r, ls, rs]),
            (Some(l), Some(r), Some(c), None, Some(ls), Some(rs)) => (7, vec![l, c, r, ls, rs]),
            _ => {
                return Err(CodecError::Unsupported {
                    what: "channel layout",
                });
            }
        }
    };
    let mut order = order;
    if let Some(lfe) = lfe {
        order.push(lfe);
    }
    if order.len() != channels.len() {
        return Err(CodecError::Unsupported {
            what: "channel layout",
        });
    }
    Ok((acmod, lfe.is_some(), order))
}

fn fscod_of(sample_rate: u32) -> Option<u8> {
    match sample_rate {
        48_000 => Some(0),
        44_100 => Some(1),
        32_000 => Some(2),
        _ => None,
    }
}

/// Piecewise linear in `table`, flat past either end.
fn interpolate(table: &[(u32, u32)], x: u32) -> u32 {
    let Some(&(first_x, first_y)) = table.first() else {
        return 0;
    };
    if x <= first_x {
        return first_y;
    }
    for pair in table.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if x <= x1 {
            let t = f64::from(x - x0) / f64::from(x1 - x0);
            return (f64::from(y0) + t * (f64::from(y1) - f64::from(y0))).round() as u32;
        }
    }
    table.last().map_or(0, |&(_, y)| y)
}

/// `chbwcod` for a cutoff: bin 73 + 3 * code (section 7.1.3).
fn bandwidth_code(cutoff: u32, sample_rate: u32) -> u8 {
    let bin = f64::from(cutoff) * 512.0 / f64::from(sample_rate);
    ((bin - 73.0) / 3.0).round().clamp(0.0, 60.0) as u8
}

/// Coupling sub-bands for a bandwidth: begin from the rate, end at the
/// sub-band that holds the bandwidth.
fn coupling_range(
    bandwidth: u8,
    begin_hz: Option<u32>,
    sample_rate: u32,
) -> Option<(usize, usize)> {
    let begin_hz = begin_hz?;
    let end = (usize::from(bandwidth) / 4 + 3).min(18);
    let bin = f64::from(begin_hz) * 512.0 / f64::from(sample_rate);
    let begin = ((bin - 37.0) / 12.0).round().clamp(0.0, 15.0) as usize;
    (begin < end).then_some((begin, end))
}

/// Cumulative bits against the nominal rate, for frame sizes that average
/// to it.
#[derive(Clone, Copy, Debug, Default)]
struct RateClock {
    bits: u128,
    samples: u128,
}

/// One syncframe's size.
#[derive(Clone, Copy, Debug)]
struct Size {
    bytes: usize,
    frmsizecod: u8,
}

/// The masking curve and allocation inputs of one slot that sends new
/// exponents.
type Allocation = Option<(Mask, bitalloc::Channel<'static>)>;

/// AC-3 or E-AC-3 encoder.
///
/// One push takes interleaved PCM at the configured rate and layout,
/// `f32` at full scale 1.0 or `i16`. Packets are whole syncframes, `pts`
/// and `duration` in samples at the sample rate. Each packet carries
/// [`SideData::Gapless`] with [`ENCODER_DELAY`]; the first also carries
/// the `dac3` or `dec3` payload as [`SideData::Config`]. The last packet
/// `drain` returns carries the trailing padding.
pub struct Ac3Encoder {
    config: EncoderConfig,
    shape: Shape,
    meta: Meta,
    params: Params,
    /// AC-3 `frmsizecod / 2`.
    rate_index: usize,
    /// E-AC-3 smallest syncframe in words: the most side information.
    min_words: usize,
    order: Vec<usize>,
    transform: Transform,
    analysis: Vec<Analysis>,
    /// Per coded channel, input not yet coded.
    pending: Vec<Vec<f32>>,
    blocks: Vec<Block>,
    originals: Vec<[[f32; COEFS]; SLOTS]>,
    last_blksw: [bool; plan::MAX_FBW],
    clock: RateClock,
    snr: i32,
    frame_index: u64,
    accepted: u64,
    first: bool,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

impl Ac3Encoder {
    pub fn new(config: EncoderConfig) -> Result<Self, CodecError> {
        let Some(fscod) = fscod_of(config.sample_rate) else {
            return Err(CodecError::Unsupported {
                what: "sample rate",
            });
        };
        let (acmod, lfeon, order) = coding(&config.layout)?;
        let eac3 = config.format == Format::Eac3;
        let blocks = if eac3 { config.blocks } else { MAX_BLOCKS };
        if !matches!(blocks, 1 | 2 | 3 | 6) {
            return Err(CodecError::Unsupported {
                what: "blocks per syncframe",
            });
        }
        let meta = check_metadata(&config.metadata, eac3)?;
        let nf = order.len() - usize::from(lfeon);
        let bit_rate = if config.bit_rate == 0 {
            default_bit_rate(config.format, nf, lfeon)
        } else {
            config.bit_rate
        };
        let rate_index = if eac3 {
            0
        } else {
            NOMINAL_KBPS
                .iter()
                .position(|&kbps| u32::from(kbps) * 1000 == bit_rate)
                .ok_or(CodecError::Unsupported { what: "bit rate" })?
        };
        let config = EncoderConfig {
            bit_rate,
            blocks,
            ..config
        };
        let codes = if eac3 {
            write::EAC3_PARAMS
        } else {
            write::AC3_PARAMS
        };
        let params = Params::from_codes(codes[0], codes[1], codes[2], codes[3], codes[4]);
        let per_channel = bit_rate / 1000 / nf as u32;
        let begin_hz =
            (nf > 1 && per_channel < NO_COUPLING_KBPS).then(|| interpolate(&COUPLING, per_channel));
        let mut bandwidth = bandwidth_code(interpolate(&CUTOFF, per_channel), config.sample_rate);
        let spf = blocks * COEFS;
        let nominal_bits = u64::from(bit_rate) * spf as u64 / u64::from(config.sample_rate);
        let mut found = None;
        loop {
            let shape = Shape {
                eac3,
                acmod,
                lfeon,
                nf,
                blocks,
                fscod,
                bandwidth,
                coupling: coupling_range(bandwidth, begin_hz, config.sample_rate),
            };
            // The open check is the coarse coding: one exponent set, no
            // coupling. A frame that cannot hold even that is illegal at
            // this rate; a noisier frame falls back to it at runtime.
            let coarse = coarse_side(&shape, &meta);
            let fits = if eac3 {
                let min_words = (coarse.total + TAIL_BITS + SIDE_MARGIN).div_ceil(16);
                (nominal_bits as usize / 16 >= min_words).then_some(min_words)
            } else {
                let bytes = usize::from(FRAME_WORDS[rate_index * 2][usize::from(fscod)]) * 2;
                let five_eighths = write::five_eighths_bytes(bytes) * 8;
                (coarse.total + TAIL_BITS + SIDE_MARGIN <= bytes * 8
                    && coarse.side_end[1] + SIDE_MARGIN <= five_eighths)
                    .then_some(0)
            };
            if let Some(min_words) = fits {
                found = Some((shape, min_words));
                break;
            }
            if bandwidth == 0 {
                break;
            }
            bandwidth = bandwidth.saturating_sub(4);
        }
        let Some((shape, min_words)) = found else {
            return Err(CodecError::Unsupported {
                what: "bit rate too low for the channel layout",
            });
        };
        if eac3 && nominal_bits.div_ceil(16) as usize > MAX_EAC3_WORDS {
            return Err(CodecError::Unsupported {
                what: "bit rate above the largest syncframe",
            });
        }
        let channels = order.len();
        Ok(Self {
            shape,
            meta,
            params,
            rate_index,
            min_words,
            transform: Transform::new(),
            analysis: (0..channels)
                .map(|_| Analysis::new(config.sample_rate))
                .collect(),
            pending: vec![Vec::new(); channels],
            blocks: vec![Block::default(); blocks],
            originals: vec![[[0.0; COEFS]; SLOTS]; blocks],
            last_blksw: [false; plan::MAX_FBW],
            clock: RateClock::default(),
            snr: 15 << 4,
            frame_index: 0,
            accepted: 0,
            first: true,
            drained: false,
            cancel: Arc::new(AtomicBool::new(false)),
            order,
            config,
        })
    }

    /// Open from codec parameters. A non-zero AC-3 `bitrate` snaps to the
    /// nearest rate of Table 5.18; zero picks the default.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let format = match params.codec {
            CodecId::Ac3 => Format::Ac3,
            CodecId::Eac3 => Format::Eac3,
            _ => return Err(CodecError::Unsupported { what: "codec" }),
        };
        let Some(layout) = params.layout.clone() else {
            return Err(CodecError::InvalidApi { what: "layout" });
        };
        let mut config = EncoderConfig::new(format, params.sample_rate, layout);
        config.bit_rate = match (format, params.bitrate) {
            (_, 0) => 0,
            (Format::Ac3, rate) => NOMINAL_KBPS
                .iter()
                .map(|&kbps| u32::from(kbps) * 1000)
                .min_by_key(|kbps| kbps.abs_diff(rate))
                .unwrap_or(rate),
            (Format::Eac3, rate) => rate,
        };
        Self::new(config)
    }

    /// The configuration in force, with the bit rate and blocks resolved.
    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Samples per syncframe.
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.shape.blocks * COEFS
    }

    /// The ISOBMFF `dac3` or `dec3` payload for these syncframes.
    #[must_use]
    pub fn config_record(&self) -> Vec<u8> {
        let shape = &self.shape;
        if shape.eac3 {
            Dec3 {
                data_rate: (self.config.bit_rate / 1000).min(0x1fff) as u16,
                substreams: vec![Dec3Substream {
                    fscod: shape.fscod,
                    bsid: 16,
                    asvc: false,
                    bsmod: self.meta.bsmod,
                    acmod: shape.acmod,
                    lfeon: shape.lfeon,
                    num_dep_sub: 0,
                    chan_loc: 0,
                }],
            }
            .to_bytes()
        } else {
            Dac3 {
                fscod: shape.fscod,
                bsid: if self.meta.mix.is_some() { 6 } else { 8 },
                bsmod: self.meta.bsmod,
                acmod: shape.acmod,
                lfeon: shape.lfeon,
                bit_rate_code: self.rate_index as u8,
            }
            .to_bytes()
            .to_vec()
        }
    }

    fn accept(&mut self, input: &AudioInput) -> Result<(), CodecError> {
        let pcm = &input.pcm;
        if pcm.rate != self.config.sample_rate {
            return Err(CodecError::InvalidApi {
                what: "sample rate",
            });
        }
        if pcm.layout != self.config.layout {
            return Err(CodecError::InvalidApi { what: "layout" });
        }
        let channels = self.order.len();
        let samples: Vec<f32> = if let Some(data) = pcm.as_f32() {
            for sample in data {
                if !sample.is_finite() {
                    return Err(CodecError::NonFinite);
                }
                if sample.abs() > 1.0 + FULL_SCALE_SLACK {
                    return Err(CodecError::OutOfRange);
                }
            }
            data.to_vec()
        } else if let Some(data) = pcm.as_i16() {
            data.iter().map(|s| f32::from(*s) / 32_768.0).collect()
        } else {
            return Err(CodecError::InvalidApi {
                what: "sample format",
            });
        };
        if !samples.len().is_multiple_of(channels) {
            return Err(CodecError::UnevenChannels);
        }
        for (coded, &input) in self.order.iter().enumerate() {
            let pending = &mut self.pending[coded];
            pending.extend(samples.iter().skip(input).step_by(channels));
        }
        self.accepted += (samples.len() / channels) as u64;
        Ok(())
    }

    /// Code every whole syncframe in `pending`.
    fn run(&mut self) -> Result<Vec<Packet>, CodecError> {
        let spf = self.frame_samples();
        let mut packets = Vec::new();
        let mut consumed = 0;
        let available = self.pending.first().map_or(0, Vec::len);
        while consumed + spf <= available {
            if self.cancel.load(Ordering::Relaxed) {
                self.discard(consumed);
                return Err(CodecError::Cancelled);
            }
            match self.code_frame(consumed) {
                Ok(packet) => packets.push(packet),
                Err(error) => {
                    self.discard(consumed + spf);
                    return Err(error);
                }
            }
            consumed += spf;
        }
        self.discard(consumed);
        Ok(packets)
    }

    fn discard(&mut self, samples: usize) {
        for pending in &mut self.pending {
            pending.drain(..samples.min(pending.len()));
        }
    }

    fn next_size(&mut self) -> Size {
        let spf = self.frame_samples() as u128;
        let rate = u128::from(self.config.bit_rate);
        let target = rate * (self.clock.samples + spf) / u128::from(self.config.sample_rate);
        let owed = target.saturating_sub(self.clock.bits);
        let fscod = usize::from(self.shape.fscod);
        let size = if self.shape.eac3 {
            let words = (owed.div_ceil(16) as usize).clamp(self.min_words, MAX_EAC3_WORDS);
            Size {
                bytes: words * 2,
                frmsizecod: 0,
            }
        } else {
            let mut code = self.rate_index * 2;
            let words = usize::from(FRAME_WORDS[code][fscod]);
            if fscod == 1 && ((words * 16) as u128) < owed {
                code += 1;
            }
            Size {
                bytes: usize::from(FRAME_WORDS[code][fscod]) * 2,
                frmsizecod: code as u8,
            }
        };
        self.clock.bits += (size.bytes * 8) as u128;
        self.clock.samples += spf;
        size
    }

    fn code_frame(&mut self, from: usize) -> Result<Packet, CodecError> {
        let nf = self.shape.nf;
        for blk in 0..self.shape.blocks {
            let mut block = Block::default();
            for coded in 0..self.order.len() {
                let mut samples = [0.0f32; COEFS];
                let at = from + blk * COEFS;
                samples.copy_from_slice(&self.pending[coded][at..at + COEFS]);
                let slot = if coded < nf { coded } else { LFE };
                // E-AC-3 keeps one transform length for the frame. A short
                // block would drop that channel out of coupling, and a
                // syncframe whose coupling use then changes is not what
                // the reference decoders read.
                let short = self.analysis[coded].block(
                    &self.transform,
                    &samples,
                    coded < nf && !self.shape.eac3,
                    &mut block.coefs[slot],
                );
                if coded < nf {
                    block.blksw[coded] = short;
                }
            }
            for ch in 0..nf {
                // Section 8.2.10: no dither in a short block or the next.
                block.dith[ch] = !(block.blksw[ch] || self.last_blksw[ch]);
                self.last_blksw[ch] = block.blksw[ch];
            }
            self.originals[blk] = block.coefs;
            self.blocks[blk] = block;
        }
        let size = self.next_size();
        let mut shape = self.shape;
        let mut mode = Mode::Adaptive;
        let (snr, skip) = loop {
            plan::plan(&shape, &mut self.blocks, mode);
            if let Some((offset, skip)) = self.allocate(&shape, size)? {
                // A short syncframe has the same bit rate and far fewer
                // bits, so the six-block cutoff leaves the mantissas below
                // the nominal offset. Give the bits to a narrower band.
                if shape.blocks < 6 && offset < NOMINAL_SNR && shape.bandwidth > 0 {
                    shape.bandwidth = shape.bandwidth.saturating_sub(4);
                    shape.coupling = shape.coupling.and_then(|(begin, end)| {
                        let end = end.min(usize::from(shape.bandwidth) / 4 + 3);
                        (begin < end).then_some((begin, end))
                    });
                    restore(&mut self.blocks, &self.originals);
                    continue;
                }
                break (offset, skip);
            }
            if matches!(mode, Mode::Adaptive) && shape.bandwidth > 0 {
                shape.bandwidth = shape.bandwidth.saturating_sub(8);
                shape.coupling = shape.coupling.and_then(|(begin, end)| {
                    let end = end.min(usize::from(shape.bandwidth) / 4 + 3);
                    (begin < end).then_some((begin, end))
                });
                restore(&mut self.blocks, &self.originals);
                continue;
            }
            // Full-band noise at the lowest legal rate cannot refresh
            // exponents. One coarse set still fits the frame the open
            // check accepted.
            if matches!(mode, Mode::Adaptive) {
                mode = Mode::Coarse;
                shape.coupling = None;
                restore(&mut self.blocks, &self.originals);
                continue;
            }
            return Err(CodecError::Limit {
                what: "bit rate too low for this audio",
            });
        };
        let frame = Frame {
            bytes: size.bytes,
            frmsizecod: size.frmsizecod,
            index: self.frame_index,
            snr: snr as u16,
        };
        let (writer, layout) = write::frame(&shape, &self.meta, &frame, &self.blocks, &skip, true);
        if layout.total + TAIL_BITS > size.bytes * 8 {
            return Err(CodecError::Limit {
                what: "syncframe overflow",
            });
        }
        let data = write::seal(&shape, writer, size.bytes);
        self.snr = snr;
        let spf = self.frame_samples() as i64;
        let pts = self.frame_index as i64 * spf;
        self.frame_index += 1;
        let mut packet = Packet::audio(data, pts, spf, self.config.sample_rate);
        packet.side.push(SideData::Gapless {
            encoder_delay: ENCODER_DELAY,
            trailing_padding: 0,
            sample_rate: self.config.sample_rate,
        });
        if self.first {
            self.first = false;
            packet.side.push(SideData::Config(self.config_record()));
        }
        Ok(packet)
    }

    /// Masks for every slot that sends exponents, then the largest SNR
    /// offset whose syncframe fits, and the AC-3 skip fields that keep
    /// block 5's mantissas in the last 3/8. `None` when even the lowest
    /// offset does not fit.
    fn allocate(
        &mut self,
        shape: &Shape,
        size: Size,
    ) -> Result<Option<(i32, [u16; MAX_BLOCKS])>, CodecError> {
        let probe = Frame {
            bytes: size.bytes,
            frmsizecod: size.frmsizecod,
            index: self.frame_index,
            snr: 0,
        };
        let (_, layout) = write::frame(
            shape,
            &self.meta,
            &probe,
            &self.blocks,
            &[0; MAX_BLOCKS],
            false,
        );
        let mut allocations: Vec<Allocation> = vec![None; self.blocks.len() * SLOTS];
        for (blk, block) in self.blocks.iter().enumerate() {
            for slot in block.active(shape) {
                if block.expstr[slot] == REUSE {
                    continue;
                }
                let channel = bitalloc::Channel {
                    start: block.start[slot],
                    end: block.end[slot],
                    fscod: shape.fscod,
                    params: self.params,
                    fgaincod: FGAINCOD,
                    snr: 0,
                    leak: (slot == plan::CPL).then_some((0, 0)),
                    delta: None,
                    high_efficiency: false,
                };
                let mask = bitalloc::mask(&block.exps[slot], &channel)?;
                allocations[blk * SLOTS + slot] = Some((mask, channel));
            }
        }
        let search = Search {
            shape,
            layout,
            bytes: size.bytes,
            allocations: &allocations,
        };
        let blocks = &mut self.blocks;
        let mut s = self.snr.clamp(0, MAX_SNR);
        if search.fits(blocks, s).is_none() {
            loop {
                s -= 64;
                if s <= 0 {
                    s = 0;
                    if search.fits(blocks, 0).is_none() {
                        return Ok(None);
                    }
                    break;
                }
                if search.fits(blocks, s).is_some() {
                    break;
                }
            }
        }
        for step in [64, 16, 4, 1] {
            while s + step <= MAX_SNR && search.fits(blocks, s + step).is_some() {
                s += step;
            }
        }
        Ok(search.fits(blocks, s).map(|skip| (s, skip)))
    }

    fn finish(&mut self) -> Result<Vec<Packet>, CodecError> {
        let spf = self.frame_samples() as u64;
        let frames = (self.accepted + u64::from(ENCODER_DELAY)).div_ceil(spf);
        let left = frames.saturating_sub(self.frame_index) * spf;
        for pending in &mut self.pending {
            pending.resize(left as usize, 0.0);
        }
        let mut packets = self.run()?;
        let padding = (frames * spf - self.accepted - u64::from(ENCODER_DELAY)) as u32;
        if let Some(packet) = packets.last_mut() {
            for side in &mut packet.side {
                if let SideData::Gapless {
                    trailing_padding, ..
                } = side
                {
                    *trailing_padding = padding;
                }
            }
        }
        Ok(packets)
    }
}

/// What the SNR search holds fixed for one syncframe.
struct Search<'a> {
    shape: &'a Shape,
    layout: Layout,
    bytes: usize,
    allocations: &'a [Allocation],
}

impl Search<'_> {
    /// Allocate every block at offset `s`. When the syncframe fits, the
    /// skip field lengths that place it.
    fn fits(&self, blocks: &mut [Block], s: i32) -> Option<[u16; MAX_BLOCKS]> {
        let shape = self.shape;
        let mut mantissas = [0usize; MAX_BLOCKS];
        for blk in 0..blocks.len() {
            for slot in blocks[blk].active(shape) {
                match &self.allocations[blk * SLOTS + slot] {
                    Some((mask, channel)) => {
                        bitalloc::pointers(mask, channel, s - 240, &mut blocks[blk].bap[slot]);
                    }
                    None if blk > 0 => {
                        let previous = blocks[blk - 1].bap[slot];
                        blocks[blk].bap[slot] = previous;
                    }
                    None => {}
                }
            }
            mantissas[blk] = write::mantissa_bits(shape, &blocks[blk]);
        }
        let budget = self.bytes * 8 - TAIL_BITS;
        let total = self.layout.total + mantissas.iter().sum::<usize>();
        if total > budget {
            return None;
        }
        let mut skip = [0u16; MAX_BLOCKS];
        if shape.eac3 {
            return Some(skip);
        }
        let five_eighths = write::five_eighths_bytes(self.bytes) * 8;
        let start5 = self.layout.side_end[5] + mantissas[..5].iter().sum::<usize>();
        let mut deficit = five_eighths.saturating_sub(start5);
        for blk in (0..MAX_BLOCKS).rev() {
            if deficit == 0 {
                break;
            }
            let bytes = deficit.saturating_sub(9).div_ceil(8).clamp(1, 511);
            skip[blk] = bytes as u16;
            deficit = deficit.saturating_sub(9 + 8 * bytes);
        }
        let padding = |range: std::ops::Range<usize>| -> usize {
            skip[range]
                .iter()
                .filter(|&&n| n > 0)
                .map(|&n| 9 + 8 * usize::from(n))
                .sum()
        };
        let block1_end = self.layout.side_end[1] + mantissas[0] + mantissas[1] + padding(0..2);
        (deficit == 0 && total + padding(0..MAX_BLOCKS) <= budget && block1_end <= five_eighths)
            .then_some(skip)
    }
}

/// Bits a real frame may add on top of the coarse estimate: block-switch
/// and dither flags.
const SIDE_MARGIN: usize = 64;

/// Side information of one coarse exponent set and no coupling.
fn coarse_side(shape: &Shape, meta: &Meta) -> Layout {
    let mut blocks = vec![Block::default(); shape.blocks];
    plan::plan(shape, &mut blocks, Mode::Coarse);
    let frame = Frame {
        bytes: MAX_EAC3_WORDS * 2,
        frmsizecod: 0,
        index: 0,
        snr: 0,
    };
    write::frame(shape, meta, &frame, &blocks, &[0; MAX_BLOCKS], false).1
}

fn restore(blocks: &mut [Block], originals: &[[[f32; COEFS]; SLOTS]]) {
    for (block, original) in blocks.iter_mut().zip(originals) {
        let (blksw, dith) = (block.blksw, block.dith);
        *block = Block {
            coefs: *original,
            blksw,
            dith,
            ..Block::default()
        };
    }
}

fn check_metadata(metadata: &EncoderMetadata, eac3: bool) -> Result<Meta, CodecError> {
    let EncoderMetadata {
        dialnorm,
        bsmod,
        dsurmod,
        mix,
        lfe_mix,
    } = *metadata;
    if !(1..=31).contains(&dialnorm) {
        return Err(CodecError::InvalidApi { what: "dialnorm" });
    }
    if bsmod > 7 {
        return Err(CodecError::InvalidApi { what: "bsmod" });
    }
    if dsurmod > 2 {
        return Err(CodecError::InvalidApi { what: "dsurmod" });
    }
    if let Some(mix) = mix {
        let surround = 3..=7;
        if mix.dmixmod > 2
            || mix.ltrt_center > 7
            || mix.loro_center > 7
            || !surround.contains(&mix.ltrt_surround)
            || !surround.contains(&mix.loro_surround)
        {
            return Err(CodecError::InvalidApi {
                what: "downmix levels",
            });
        }
    }
    if let Some(code) = lfe_mix {
        if !eac3 {
            return Err(CodecError::Unsupported {
                what: "LFE mix level in AC-3",
            });
        }
        if code > 31 {
            return Err(CodecError::InvalidApi {
                what: "LFE mix level",
            });
        }
    }
    Ok(Meta {
        dialnorm,
        bsmod,
        dsurmod,
        mix,
        lfe_mix,
    })
}

impl Encoder for Ac3Encoder {
    fn codec(&self) -> CodecId {
        match self.config.format {
            Format::Ac3 => CodecId::Ac3,
            Format::Eac3 => CodecId::Eac3,
        }
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.drained {
            return Err(CodecError::InvalidApi {
                what: "push after drain",
            });
        }
        self.accept(input)?;
        self.run()
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        if self.accepted == 0 {
            return Ok(Vec::new());
        }
        self.finish()
    }

    fn reset(&mut self) {
        let Ok(fresh) = Self::new(self.config.clone()) else {
            return;
        };
        let cancel = Arc::clone(&self.cancel);
        *self = fresh;
        self.cancel = cancel;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
