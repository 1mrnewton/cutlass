//! AC-3 and E-AC-3 encoding (A/52 section 8, Annex E): transient
//! detection, coupling, rematrixing, exponent strategies, the parametric
//! bit allocation searched for the largest SNR offset that fits, and
//! constant bit rate syncframes.
//!
//! One independent substream: mono to 5.1 with LFE at 32, 44.1 and
//! 48 kHz. E-AC-3 also writes a 7.1 program as that 5.1 frame plus one
//! dependent frame for the back pair. It takes any bit rate and 1, 2, 3
//! or 6 blocks per syncframe. Spectral extension and AHT are written.
//! Enhanced coupling is never written. Transient pre-noise parameters are
//! not chosen; a test may place them on channel 0.
//! A block
//! peaking above half scale sends a dynamic range word that brings it back
//! to half scale, and the frame sends a heavy-compression word for the
//! same cut. On a 7.1 program the dependent frame carries the peak of the
//! whole mix, and the independent frame keeps the peak of its own channels.
//! Delta bit
//! allocation is written only when a block carries
//! segments; the rate control leaves every segment empty. Spectral extension
//! replaces the top bins of a low-rate E-AC-3 frame. A steady six-block
//! frame with one exponent set uses the adaptive hybrid transform, and a
//! gain on the small coefficients when that gain is shorter than a fixed
//! mantissa. A mix of gains 2 and 4 is packed into 5-bit groups when that
//! is shorter than one gain for the whole channel.

mod analysis;
mod plan;
mod write;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, Channel, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput,
    Packet, Pcm, SideData,
};

use crate::bitalloc::{self, Mask, Params};
use crate::header::{Dac3, Dec3, Dec3Substream, compr_code, dynrng_code};
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
    /// E-AC-3 `pgmscl` in dB, −50 to +12. −51 writes the mute code.
    /// `None` omits the word, which is 0 dB.
    pub program_scale_db: Option<i8>,
}

impl Default for EncoderMetadata {
    fn default() -> Self {
        Self {
            dialnorm: 31,
            bsmod: 0,
            dsurmod: 0,
            mix: None,
            lfe_mix: None,
            program_scale_db: None,
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

struct SevenPointOne {
    /// Independent coded channels, in A/52 order, as public-layout indices.
    core: Vec<usize>,
    backs: [usize; 2],
}

/// 7.1 is the side pair and the back pair together. Anything else is `None`.
fn seven_point_one(layout: &ChannelLayout) -> Option<SevenPointOne> {
    let channels = layout.channels();
    if channels.len() != 8 {
        return None;
    }
    let find = |wanted: Channel| channels.iter().position(|c| *c == wanted);
    let (l, r, c, lfe, ls, rs, lb, rb) = (
        find(Channel::FrontLeft)?,
        find(Channel::FrontRight)?,
        find(Channel::FrontCenter)?,
        find(Channel::Lfe)?,
        find(Channel::SideLeft)?,
        find(Channel::SideRight)?,
        find(Channel::BackLeft)?,
        find(Channel::BackRight)?,
    );
    Some(SevenPointOne {
        core: vec![l, c, r, ls, rs, lfe],
        backs: [lb, rb],
    })
}

fn layout_of(channels: Vec<Channel>) -> Result<ChannelLayout, CodecError> {
    ChannelLayout::new(channels).map_err(|_| CodecError::Unsupported {
        what: "channel layout",
    })
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
    /// This substream's bit rate. A 7.1 program's public rate is the sum.
    rate: u32,
    /// Dependent `chanmap` when this encoder writes that substream.
    chanmap: Option<u16>,
    /// The back-pair frame of a 7.1 program.
    program: Option<Program>,
    /// Block peaks of frames this dependent substream has not coded yet.
    /// The program writes one entry per frame so the complete mix follows
    /// Annex E 3.8.5. Empty means this substream uses its own samples.
    program_peaks: VecDeque<Vec<f32>>,
    /// AC-3 delta bit allocation copied onto each frame's first block.
    /// The rate control leaves this empty.
    #[cfg(test)]
    forced_delta: Option<bitalloc::Delta>,
    /// E-AC-3 transient pre-noise parameters for channel 0. The encoder
    /// does not choose these.
    #[cfg(test)]
    forced_transient: Option<crate::frame::Transient>,
}

/// Back surrounds carried in the dependent substream of a 7.1 program.
struct Program {
    backs: Box<Ac3Encoder>,
    /// Index of each back channel in the public layout.
    back_at: [usize; 2],
}

/// Table E2.5 bit 6, `Lrs`/`Rrs`. The `dec3` `chan_loc` bit 1 is that pair.
const BACK_PAIR: u16 = 0x0200;

impl Ac3Encoder {
    pub fn new(config: EncoderConfig) -> Result<Self, CodecError> {
        Self::build(config, None)
    }

    fn build(config: EncoderConfig, chanmap: Option<u16>) -> Result<Self, CodecError> {
        if chanmap.is_none()
            && config.format == Format::Eac3
            && let Some(split) = seven_point_one(&config.layout)
        {
            return Self::with_backs(config, split);
        }
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
                spx: None,
            };
            // The open check is the coarse coding: one exponent set, no
            // coupling. A frame that cannot hold even that is illegal at
            // this rate; a noisier frame falls back to it at runtime.
            let coarse = coarse_side(&shape, &meta, chanmap);
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
        let Some((mut shape, min_words)) = found else {
            return Err(CodecError::Unsupported {
                what: "bit rate too low for the channel layout",
            });
        };
        shape.spx = plan::extension_for(eac3, per_channel, shape.end_bin());
        if shape.spx.is_some() {
            shape.coupling = None;
        }
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
            #[cfg(test)]
            forced_delta: None,
            #[cfg(test)]
            forced_transient: None,
            order,
            config,
            rate: bit_rate,
            chanmap,
            program: None,
            program_peaks: VecDeque::new(),
        })
    }

    /// An E-AC-3 7.1 program: a 5.1 independent frame and one dependent
    /// frame for the back pair. The public rate is the sum of the two.
    fn with_backs(config: EncoderConfig, split: SevenPointOne) -> Result<Self, CodecError> {
        let (core_rate, back_rate) = if config.bit_rate == 0 {
            (
                default_bit_rate(Format::Eac3, 5, true),
                default_bit_rate(Format::Eac3, 2, false),
            )
        } else {
            let core = config.bit_rate / 7 * 5;
            (core, config.bit_rate - core)
        };
        let mut core_config = config.clone();
        core_config.bit_rate = core_rate;
        core_config.layout = layout_of(vec![
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::FrontCenter,
            Channel::Lfe,
            Channel::SideLeft,
            Channel::SideRight,
        ])?;
        let mut back_config = config.clone();
        back_config.bit_rate = back_rate;
        back_config.layout = ChannelLayout::stereo();
        let mut core = Self::build(core_config, None)?;
        let backs = Self::build(back_config, Some(BACK_PAIR))?;
        core.program = Some(Program {
            backs: Box::new(backs),
            back_at: split.backs,
        });
        core.order = split.core;
        core.config.layout = config.layout;
        core.config.bit_rate = core_rate + back_rate;
        Ok(core)
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

    /// Put one delta bit allocation on each AC-3 frame's first channel.
    #[cfg(test)]
    fn force_delta(&mut self, delta: bitalloc::Delta) {
        self.forced_delta = Some(delta);
    }

    /// Write one channel's transient pre-noise parameters on every E-AC-3 frame.
    #[cfg(test)]
    pub(crate) fn force_transient(&mut self, location: u16, length: u8) {
        self.forced_transient =
            (location < 1024).then_some(crate::frame::Transient { location, length });
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
                    num_dep_sub: u8::from(self.program.is_some()),
                    chan_loc: u16::from(self.program.is_some()) * 0x002,
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
        let channels = self.config.layout.len();
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
        let frames = samples.len() / channels;
        self.accepted += frames as u64;
        if let Some(program) = &mut self.program {
            let mut stereo = Vec::with_capacity(frames * 2);
            for frame in 0..frames {
                let base = frame * channels;
                stereo.push(samples[base + program.back_at[0]]);
                stereo.push(samples[base + program.back_at[1]]);
            }
            let pcm = Pcm::from_f32(
                self.config.sample_rate,
                program.backs.config.layout.clone(),
                stereo,
            )
            .map_err(|_| CodecError::InvalidApi { what: "layout" })?;
            program.backs.accept(&AudioInput { pcm })?;
        }
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
        let rate = u128::from(self.rate);
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

    /// Cut a block that peaks above half scale. A repeated level reuses the word.
    /// `forced` is the program peak of each block, for a dependent substream.
    fn range_words(&mut self, from: usize, forced: Option<&[f32]>) {
        let mut in_force = 0u8;
        for blk in 0..self.shape.blocks {
            let peak = if let Some(peaks) = forced {
                peaks.get(blk).copied().unwrap_or(0.0)
            } else {
                let at = from + blk * COEFS;
                let mut peak = 0.0f32;
                for channel in &self.pending {
                    for &sample in &channel[at..at + COEFS] {
                        peak = peak.max(sample.abs());
                    }
                }
                peak
            };
            let code = if peak > 0.5 && peak.is_finite() {
                dynrng_code(0.5 / peak)
            } else {
                0
            };
            let block = &mut self.blocks[blk];
            block.dynrnge = code != in_force;
            block.dynrng = code;
            in_force = code;
        }
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
                // A short block leaves the coupling. The plan turns coupling
                // off for the whole E-AC-3 frame when that would change
                // from one block to the next. Spectral extension copies
                // long-transform bins, so a frame that uses it stays long.
                let short = self.analysis[coded].block(
                    &self.transform,
                    &samples,
                    coded < nf && self.shape.spx.is_none(),
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
        let forced = self.program_peaks.front().cloned();
        let heavy = heavy_word(&self.pending, from, self.shape.blocks, forced.as_deref());
        let mut shape = self.shape;
        // Spectral extension copies long-transform bins. A short block
        // does not have those bins, so the frame stays full band.
        if self.blocks[..shape.blocks]
            .iter()
            .any(|block| block.blksw[..nf].contains(&true))
        {
            shape.spx = None;
        }
        let mut mode = Mode::Adaptive;
        let (snr, skip) = loop {
            plan::plan(&shape, &mut self.blocks, mode);
            self.range_words(from, forced.as_deref());
            #[cfg(test)]
            if !shape.eac3
                && let Some(delta) = self.forced_delta
            {
                self.blocks[0].deltas[0] = delta;
            }
            if let Some((offset, skip)) = self.allocate(&shape, size, heavy)? {
                // A short syncframe has the same bit rate and far fewer
                // bits, so the six-block cutoff leaves the mantissas below
                // the nominal offset. Give the bits to a narrower band.
                if shape.blocks < 6 && offset < NOMINAL_SNR && shape.bandwidth > 0 {
                    if shape.spx.is_some() {
                        shape.spx = None;
                    } else {
                        shape.bandwidth = shape.bandwidth.saturating_sub(4);
                        shape.coupling = shape.coupling.and_then(|(begin, end)| {
                            let end = end.min(usize::from(shape.bandwidth) / 4 + 3);
                            (begin < end).then_some((begin, end))
                        });
                    }
                    restore(&mut self.blocks, &self.originals);
                    continue;
                }
                break (offset, skip);
            }
            if shape.spx.is_some() {
                shape.spx = None;
                restore(&mut self.blocks, &self.originals);
                continue;
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
                shape.spx = None;
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
            chanmap: self.chanmap,
            compr: heavy,
            #[cfg(test)]
            transient: self.forced_transient,
            #[cfg(not(test))]
            transient: None,
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
        self.note_program_peaks(from);
        if forced.is_some() {
            self.program_peaks.pop_front();
        }
        Ok(packet)
    }

    /// Queue this frame's program peaks for the dependent substream.
    fn note_program_peaks(&mut self, from: usize) {
        let blocks = self.shape.blocks;
        let peaks = self.program.as_ref().map(|program| {
            combined_block_peaks(&self.pending, &program.backs.pending, from, blocks)
        });
        if let Some(peaks) = peaks
            && let Some(program) = &mut self.program
        {
            program.backs.program_peaks.push_back(peaks);
        }
    }

    /// Masks for every slot that sends exponents, then the largest SNR
    /// offset whose syncframe fits, and the AC-3 skip fields that keep
    /// block 5's mantissas in the last 3/8. `None` when even the lowest
    /// offset does not fit.
    fn allocate(
        &mut self,
        shape: &Shape,
        size: Size,
        heavy: Option<u8>,
    ) -> Result<Option<(i32, [u16; MAX_BLOCKS])>, CodecError> {
        let probe = Frame {
            bytes: size.bytes,
            frmsizecod: size.frmsizecod,
            index: self.frame_index,
            snr: 0,
            chanmap: self.chanmap,
            compr: heavy,
            #[cfg(test)]
            transient: self.forced_transient,
            #[cfg(not(test))]
            transient: None,
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
        // A block that does not send `deltbaie` keeps the previous block's
        // segments. The mask has to follow that, or the mantissas drift.
        let mut carried = [bitalloc::Delta::default(); SLOTS];
        for (blk, block) in self.blocks.iter().enumerate() {
            for slot in block.active(shape) {
                if block.deltas[slot].count > 0 {
                    carried[slot] = block.deltas[slot];
                }
                if block.expstr[slot] == REUSE {
                    continue;
                }
                let held = carried[slot];
                let start = block.start[slot];
                let end = block.end[slot];
                let leak = (slot == plan::CPL).then_some((0, 0));
                let mask = {
                    let channel = bitalloc::Channel {
                        start,
                        end,
                        fscod: shape.fscod,
                        params: self.params,
                        fgaincod: FGAINCOD,
                        snr: 0,
                        leak,
                        delta: (held.count > 0).then_some(held.segments()),
                        high_efficiency: block.aht[slot],
                    };
                    bitalloc::mask(&block.exps[slot], &channel)?
                };
                let channel = bitalloc::Channel {
                    start,
                    end,
                    fscod: shape.fscod,
                    params: self.params,
                    fgaincod: FGAINCOD,
                    snr: 0,
                    leak,
                    delta: None,
                    high_efficiency: block.aht[slot],
                };
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
        if let Some(program) = &mut self.program {
            let backs = program.backs.finish()?;
            splice(&mut packets, backs)?;
        }
        Ok(packets)
    }

    /// Append one dependent frame to each independent frame.
    fn attach(&mut self, packets: &mut [Packet]) -> Result<(), CodecError> {
        let Some(program) = &mut self.program else {
            return Ok(());
        };
        let backs = program.backs.run()?;
        splice(packets, backs)
    }
}

fn splice(packets: &mut [Packet], backs: Vec<Packet>) -> Result<(), CodecError> {
    if backs.len() != packets.len() {
        return Err(CodecError::Limit {
            what: "dependent substream",
        });
    }
    for (packet, back) in packets.iter_mut().zip(backs) {
        packet.data.extend(back.data);
    }
    Ok(())
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
            if blk == 0 {
                mantissas[blk] += write::aht_payload_bits(shape, blocks);
            }
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

/// One heavy-compression word when the frame peaks above half scale.
/// `forced` is one peak per block for a dependent substream.
fn heavy_word(
    pending: &[Vec<f32>],
    from: usize,
    blocks: usize,
    forced: Option<&[f32]>,
) -> Option<u8> {
    let peak = if let Some(peaks) = forced {
        peaks.iter().copied().fold(0.0, f32::max)
    } else {
        let samples = blocks * COEFS;
        let mut peak = 0.0f32;
        for channel in pending {
            let span = channel.get(from..from + samples)?;
            for &sample in span {
                peak = peak.max(sample.abs());
            }
        }
        peak
    };
    if peak > 0.5 && peak.is_finite() {
        let code = compr_code(0.5 / peak);
        (code != 0).then_some(code)
    } else {
        None
    }
}

/// The louder of the independent channels and the back pair, one peak per block.
fn combined_block_peaks(
    core: &[Vec<f32>],
    backs: &[Vec<f32>],
    from: usize,
    blocks: usize,
) -> Vec<f32> {
    (0..blocks)
        .map(|blk| {
            let at = from + blk * COEFS;
            block_peak(core, at).max(block_peak(backs, at))
        })
        .collect()
}

fn block_peak(channels: &[Vec<f32>], at: usize) -> f32 {
    let mut peak = 0.0f32;
    for channel in channels {
        let Some(span) = channel.get(at..at + COEFS) else {
            continue;
        };
        for &sample in span {
            peak = peak.max(sample.abs());
        }
    }
    peak
}

/// Side information of one coarse exponent set and no coupling.
fn coarse_side(shape: &Shape, meta: &Meta, chanmap: Option<u16>) -> Layout {
    let mut blocks = vec![Block::default(); shape.blocks];
    plan::plan(shape, &mut blocks, Mode::Coarse);
    let frame = Frame {
        bytes: MAX_EAC3_WORDS * 2,
        frmsizecod: 0,
        index: 0,
        snr: 0,
        chanmap,
        compr: None,
        transient: None,
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
        program_scale_db,
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
    if let Some(db) = program_scale_db {
        if !eac3 {
            return Err(CodecError::Unsupported {
                what: "program scale in AC-3",
            });
        }
        if db != -51 && !(-50..=12).contains(&db) {
            return Err(CodecError::InvalidApi {
                what: "program scale",
            });
        }
    }
    Ok(Meta {
        dialnorm,
        bsmod,
        dsurmod,
        mix,
        lfe_mix,
        program_scale_db,
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
        let mut packets = self.run()?;
        self.attach(&mut packets)?;
        Ok(packets)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitalloc::{Delta, MAX_SEGMENTS, Segment};
    use crate::decode::Ac3Decoder;
    use cutlass_decode::{Decoder, Packet as DecodedPacket};
    use cutlass_encode::{AudioInput, ChannelLayout, Encoder, Pcm};

    #[test]
    fn a_delta_segment_matches_ffmpeg() {
        let ffmpeg = std::process::Command::new("ffmpeg")
            .arg("-version")
            .output();
        if ffmpeg.is_err() {
            eprintln!("skipped: ffmpeg is not installed");
            return;
        }
        let rate = 48_000u32;
        let layout = ChannelLayout::mono();
        let mut spec = CodecParams::audio(CodecId::Ac3, rate, layout.clone());
        spec.bitrate = 96_000;
        let mut encoder = Ac3Encoder::open(&spec).expect("encoder");
        let mut segments = [Segment::default(); MAX_SEGMENTS];
        segments[0] = Segment {
            offset: 0,
            len: 4,
            ba: 3,
        };
        encoder.force_delta(Delta { segments, count: 1 });
        let frames = 1_536 * 2;
        let samples: Vec<f32> = (0..frames)
            .map(|index| {
                let t = index as f32 / rate as f32;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.4
            })
            .collect();
        let pcm = Pcm::from_f32(rate, layout.clone(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        let bytes: Vec<u8> = packets
            .iter()
            .flat_map(|packet| packet.data.iter().copied())
            .collect();
        let dir = std::env::temp_dir().join(format!("cutlass-ac3-delta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let input = dir.join("in.ac3");
        let output = dir.join("out.f32");
        std::fs::write(&input, &bytes).expect("write");
        let ffmpeg_out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-drc_scale", "0", "-f", "ac3", "-i"])
            .arg(&input)
            .args(["-f", "f32le", "-c:a", "pcm_f32le"])
            .arg(&output)
            .output()
            .expect("ffmpeg");
        assert!(
            ffmpeg_out.status.success(),
            "ffmpeg rejected the delta segment: {}",
            String::from_utf8_lossy(&ffmpeg_out.stderr)
        );
        let mut decoder =
            Ac3Decoder::open(&CodecParams::audio(CodecId::Ac3, rate, layout)).expect("decoder");
        let mut ours = Vec::new();
        for packet in &packets {
            let decoded =
                DecodedPacket::audio(packet.data.clone(), packet.pts, packet.duration, rate);
            for output in decoder.push(&decoded).unwrap_or_else(|error| {
                panic!("decode {error:?} frame {} bytes", packet.data.len())
            }) {
                if let cutlass_decode::Output::Audio(audio) = output {
                    ours.extend_from_slice(audio.pcm.as_f32().expect("f32"));
                }
            }
        }
        assert!(
            decoder.tools().delta_blocks > 0,
            "the frame carried no delta"
        );
        let reference = std::fs::read(&output).expect("pcm");
        let theirs: Vec<f32> = reference
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let n = ours.len().min(theirs.len());
        assert!(n > 1_000, "decoded {n} samples");
        let signal: f64 = ours[..n].iter().map(|s| f64::from(*s).powi(2)).sum();
        let noise: f64 = ours[..n]
            .iter()
            .zip(&theirs[..n])
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        let quality = 10.0 * (signal / noise.max(1e-20)).log10();
        assert!(quality > 60.0, "delta allocation {quality:.1} dB");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_forced_transient_window_is_copied_into_the_frame() {
        let rate = 48_000u32;
        let layout = ChannelLayout::mono();
        let mut spec = CodecParams::audio(CodecId::Eac3, rate, layout.clone());
        spec.bitrate = 96_000;
        let mut encoder = Ac3Encoder::open(&spec).expect("encoder");
        encoder.force_transient(256, 64);
        let frames = rate as usize;
        let samples: Vec<f32> = (0..frames)
            .map(|index| {
                let t = index as f32 / rate as f32;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.25
            })
            .collect();
        let pcm = Pcm::from_f32(rate, layout.clone(), samples).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
        packets.extend(encoder.drain().expect("drain"));
        let mut decoder =
            Ac3Decoder::open(&CodecParams::audio(CodecId::Eac3, rate, layout)).expect("decoder");
        let mut ours = Vec::new();
        for packet in &packets {
            let decoded =
                DecodedPacket::audio(packet.data.clone(), packet.pts, packet.duration, rate);
            for output in decoder.push(&decoded).expect("decode") {
                if let cutlass_decode::Output::Audio(audio) = output {
                    ours.extend_from_slice(audio.pcm.as_f32().expect("f32"));
                }
            }
        }
        for output in decoder.drain().expect("drain") {
            if let cutlass_decode::Output::Audio(audio) = output {
                ours.extend_from_slice(audio.pcm.as_f32().expect("f32"));
            }
        }
        assert!(decoder.tools().transient_channels > 0);
        let base = 2 * 1536;
        let source = &ours[base + 256..base + 448];
        let copied = &ours[base + 704..base + 896];
        let err = source
            .iter()
            .zip(copied)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1.0e-6, "copied window {err}");
        let ffmpeg = std::process::Command::new("ffmpeg")
            .arg("-version")
            .output();
        if ffmpeg.is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("cutlass-ac3-tpnp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let input = dir.join("in.eac3");
        let output = dir.join("out.f32");
        let bytes: Vec<u8> = packets
            .iter()
            .flat_map(|packet| packet.data.iter().copied())
            .collect();
        std::fs::write(&input, &bytes).expect("write");
        let ffmpeg_out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-drc_scale", "0", "-f", "eac3", "-i"])
            .arg(&input)
            .args(["-f", "f32le", "-c:a", "pcm_f32le"])
            .arg(&output)
            .output()
            .expect("ffmpeg");
        assert!(
            ffmpeg_out.status.success(),
            "ffmpeg rejected the transient parameters: {}",
            String::from_utf8_lossy(&ffmpeg_out.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
