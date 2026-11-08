//! The Layer III encoder: input buffering, block switching, the filterbank,
//! the psychoacoustic model, mid/side, bit allocation with the bit
//! reservoir, constant and variable bitrate, and frame assembly.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet,
    SideData,
};

use crate::analysis::{Analysis, NORMAL, SHORT, START, STOP, mdct};
use crate::bits::BitWriter;
use crate::header::{FrameHeader, Mode, Version, bitrate_index, bitrates, classify_rate};
use crate::huffenc::{long_bounds, write_count1, write_pairs};
use crate::psy::{Attack, Masking, Params, Psy};
use crate::quantize::{self, Coded, Context, MAX_GRANULE_BITS, Slot, Spectrum, Target};
use crate::sideinfo::SideInfo;

/// Samples between the first input sample and the first sample a
/// decoder's output holds it at, less the decoder's own 529: the encoder
/// delay of the LAME tag. The filterbank and MDCT delay is 1057 samples.
pub const ENCODER_DELAY: u32 = 1057 - crate::DECODER_DELAY;

/// How the encoder spends bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mp3Rate {
    /// Every frame at this bitrate in kbit/s, one of the Layer III rates.
    Constant(u32),
    /// Each frame at the bitrate its noise target needs: quality 0 (best)
    /// to 9 (smallest), as LAME's `-V` scale.
    Variable(u8),
}

/// What [`Mp3Encoder::new`] encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mp3EncoderConfig {
    /// One of the nine Layer III rates, 8 kHz to 48 kHz.
    pub sample_rate: u32,
    /// 1 or 2. Stereo is coded as joint stereo with mid/side per frame.
    pub channels: usize,
    pub rate: Mp3Rate,
}

/// Lowpass for a constant bitrate per stereo pair (LAME's optimum
/// bandwidth table), in Hz.
fn constant_lowpass(kbps_per_pair: u32) -> f32 {
    const TABLE: [(u32, f32); 17] = [
        (8, 2_000.0),
        (16, 3_700.0),
        (24, 3_900.0),
        (32, 5_500.0),
        (40, 7_000.0),
        (48, 7_500.0),
        (56, 10_000.0),
        (64, 11_000.0),
        (80, 13_500.0),
        (96, 15_100.0),
        (112, 15_600.0),
        (128, 17_000.0),
        (160, 17_500.0),
        (192, 18_600.0),
        (224, 19_400.0),
        (256, 19_700.0),
        (320, 20_500.0),
    ];
    let mut out = TABLE[0].1;
    for (kbps, hz) in TABLE {
        if kbps_per_pair >= kbps {
            out = hz;
        }
    }
    out
}

/// The masking offset of variable quality 0, in dB over the model's
/// tone- and noise-masking offsets. Constant bitrate codes to it too
/// while the bits allow.
const VARIABLE_BEST: f32 = 10.0;

/// Lowpass and masking offset of a variable quality.
fn variable_params(quality: u8) -> (f32, f32) {
    const LOWPASS: [f32; 10] = [
        19_500.0, 19_000.0, 18_500.0, 18_000.0, 17_500.0, 16_500.0, 15_500.0, 14_500.0, 13_000.0,
        11_000.0,
    ];
    let q = usize::from(quality.min(9));
    // Each step down the scale allows 2 dB more noise.
    (LOWPASS[q], VARIABLE_BEST - 2.0 * f32::from(quality.min(9)))
}

/// The bitrate of a frame, the bytes it holds, and the padding accumulator.
#[derive(Clone, Copy, Debug)]
struct FrameSize {
    index: u8,
    kbps: u32,
    bytes: usize,
    padding: bool,
}

/// A frame waiting for main data of the frames after it.
struct Pending {
    bytes: Vec<u8>,
    /// Where the main-data slot begins.
    slot: usize,
    /// How much of the slot is written.
    filled: usize,
    pts: i64,
}

impl Pending {
    fn free(&self) -> usize {
        self.bytes.len() - self.slot - self.filled
    }
}

/// Frames whose slots are still being filled: the bit reservoir.
#[derive(Default)]
struct Assembler {
    frames: VecDeque<Pending>,
}

impl Assembler {
    fn free(&self) -> usize {
        self.frames.iter().map(Pending::free).sum()
    }

    /// Zero the oldest free bytes until at most `keep` are free.
    fn stuff(&mut self, keep: usize) {
        let mut excess = self.free().saturating_sub(keep);
        for frame in &mut self.frames {
            if excess == 0 {
                break;
            }
            let take = frame.free().min(excess);
            frame.filled += take;
            excess -= take;
        }
    }

    /// Append a frame and its main data, which starts `free()` bytes back.
    /// False when the main data does not fit.
    fn push(&mut self, frame: Pending, main: &[u8]) -> bool {
        self.frames.push_back(frame);
        let mut rest = main;
        for frame in &mut self.frames {
            if rest.is_empty() {
                break;
            }
            let take = frame.free().min(rest.len());
            if take == 0 {
                continue;
            }
            let at = frame.slot + frame.filled;
            frame.bytes[at..at + take].copy_from_slice(&rest[..take]);
            frame.filled += take;
            rest = &rest[take..];
        }
        rest.is_empty()
    }

    /// Frames no later frame can write into: full ones at the front.
    fn ready(&mut self) -> Vec<Pending> {
        let mut out = Vec::new();
        while self.frames.front().is_some_and(|frame| frame.free() == 0) {
            if let Some(frame) = self.frames.pop_front() {
                out.push(frame);
            }
        }
        out
    }

    fn flush(&mut self) -> Vec<Pending> {
        self.stuff(0);
        self.ready()
    }
}

/// One granule, analysed and waiting to be quantized.
struct Prepared {
    block_type: u8,
    /// `[ch]` spectrum in bitstream order.
    xr: Vec<[f32; 576]>,
    masking: Vec<Masking>,
}

/// Native Layer III encoder.
///
/// One push takes interleaved PCM at the configured rate and channel
/// count, `f32` at full scale 1.0 or `i16`. Packets are whole frames,
/// header included, `pts` and `duration` in samples at the sample rate.
/// Each packet carries [`SideData::Gapless`] with [`ENCODER_DELAY`]; the
/// last packet `drain` returns carries the trailing padding as well.
pub struct Mp3Encoder {
    config: Mp3EncoderConfig,
    version: Version,
    rate_index: u8,
    rate_table: usize,
    granules: usize,
    /// Per channel input from absolute sample `buffer_start`.
    buffer: Vec<Vec<f32>>,
    buffer_start: i64,
    accepted: u64,
    next_granule: i64,
    psy: Psy,
    attack: Vec<Attack>,
    analysis: Vec<Analysis>,
    previous_subbands: Vec<[[f32; 32]; 18]>,
    /// Attack flags of granules not yet coded, from `next_granule`.
    wanted: VecDeque<bool>,
    last_block: u8,
    frame_granules: Vec<Prepared>,
    assembler: Assembler,
    frame_index: i64,
    padding_rest: u64,
    lowpass_line: usize,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

/// The first absolute sample the encoder keeps: the long FFT of granule 0
/// starts 736 samples before it.
const HISTORY: i64 = 1024;
/// Samples past a granule's start that must exist before it is coded:
/// the end of the next granule's attack window.
const LOOKAHEAD: i64 = 640;
/// Rounding slack on full scale, as the AAC encoder allows.
const FULL_SCALE_SLACK: f32 = 16.0 * f32::EPSILON;

impl Mp3Encoder {
    pub fn new(config: Mp3EncoderConfig) -> Result<Self, CodecError> {
        let Some((version, rate_index)) = classify_rate(config.sample_rate) else {
            return Err(CodecError::Unsupported {
                what: "sample rate",
            });
        };
        if !matches!(config.channels, 1 | 2) {
            return Err(CodecError::Unsupported { what: "channels" });
        }
        let pair_scale = if config.channels == 1 { 2 } else { 1 };
        let (lowpass, offset) = match config.rate {
            Mp3Rate::Constant(kbps) => {
                if bitrate_index(version, kbps).is_none() {
                    return Err(CodecError::Unsupported { what: "bitrate" });
                }
                // Lower sampling rates code a narrower band at the same rate.
                let rate_scale = 44_100 / config.sample_rate.max(1);
                (
                    constant_lowpass(kbps * pair_scale * rate_scale.max(1)),
                    VARIABLE_BEST,
                )
            }
            Mp3Rate::Variable(quality) => {
                if quality > 9 {
                    return Err(CodecError::Unsupported { what: "quality" });
                }
                variable_params(quality)
            }
        };
        let nyquist = config.sample_rate as f32 / 2.0;
        let lowpass = lowpass.min(nyquist * 0.95);
        let header = Self::header_for(
            version,
            rate_index,
            config.sample_rate,
            1,
            0,
            false,
            config.channels,
            false,
        );
        let rate_table = header.rate_table();
        // ISO/IEC 11172-3 Annex D model 2 offsets: tone masking noise
        // 29 dB, noise masking tone 6 dB, both raised by the quality.
        let params = Params {
            tmn: 29.0 + offset,
            nmt: 6.0 + offset,
            ath_offset: 0.0,
            lowpass,
        };
        let channels = config.channels;
        let lowpass_line = ((lowpass / nyquist) * 576.0).ceil() as usize;
        Ok(Self {
            config,
            version,
            rate_index,
            rate_table,
            granules: header.granules(),
            buffer: vec![vec![0.0; HISTORY as usize]; channels],
            buffer_start: -HISTORY,
            accepted: 0,
            next_granule: 0,
            psy: Psy::new(config.sample_rate, rate_table, channels, params),
            attack: (0..channels).map(|_| Attack::new()).collect(),
            analysis: vec![Analysis::default(); channels],
            previous_subbands: vec![[[0.0; 32]; 18]; channels],
            wanted: VecDeque::new(),
            last_block: NORMAL,
            frame_granules: Vec::new(),
            assembler: Assembler::default(),
            frame_index: 0,
            padding_rest: 0,
            lowpass_line: lowpass_line.min(576),
            drained: false,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Open from codec parameters. A non-zero `bitrate` with
    /// `constant_rate` is constant bitrate at the nearest Layer III rate;
    /// without it, the variable quality whose typical rate is nearest.
    /// Zero is variable quality 2.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mp3 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let channels = params.layout.as_ref().map_or(0, ChannelLayout::len);
        let Some((version, _)) = classify_rate(params.sample_rate) else {
            return Err(CodecError::Unsupported {
                what: "sample rate",
            });
        };
        let kbps = params.bitrate / 1000;
        let rate = if kbps == 0 {
            Mp3Rate::Variable(2)
        } else if params.constant_rate {
            let nearest = bitrates(version)
                .into_iter()
                .min_by_key(|rate| rate.abs_diff(kbps))
                .unwrap_or(128);
            Mp3Rate::Constant(nearest)
        } else {
            // Typical stereo rates of each quality at 44.1 kHz.
            const TYPICAL: [u32; 10] = [245, 225, 190, 175, 165, 130, 115, 100, 85, 65];
            let per_pair = if channels == 1 { kbps * 2 } else { kbps };
            let quality = TYPICAL
                .iter()
                .enumerate()
                .min_by_key(|(_, rate)| rate.abs_diff(per_pair))
                .map_or(2, |(q, _)| q as u8);
            Mp3Rate::Variable(quality)
        };
        Self::new(Mp3EncoderConfig {
            sample_rate: params.sample_rate,
            channels,
            rate,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn header_for(
        version: Version,
        rate_index: u8,
        sample_rate: u32,
        bitrate_index: u8,
        kbps: u32,
        padding: bool,
        channels: usize,
        mid_side: bool,
    ) -> FrameHeader {
        FrameHeader {
            version,
            layer: 3,
            protected: false,
            bitrate_index,
            bitrate_kbps: kbps,
            rate_index,
            sample_rate,
            padding,
            private: false,
            mode: if channels == 1 {
                Mode::Mono
            } else {
                Mode::JointStereo
            },
            mode_extension: if mid_side { 2 } else { 0 },
            copyright: false,
            original: true,
            emphasis: 0,
        }
    }

    /// Bytes of a frame at `kbps` without padding, and the remainder the
    /// padding bit accumulates.
    fn frame_bytes(&self, kbps: u32) -> (usize, u64) {
        let numerator: u64 = if self.version.lsf() { 72_000 } else { 144_000 };
        let total = numerator * u64::from(kbps);
        let rate = u64::from(self.config.sample_rate);
        ((total / rate) as usize, total % rate)
    }

    fn size_for(&self, kbps: u32, commit: bool, rest: &mut u64) -> FrameSize {
        let (bytes, remainder) = self.frame_bytes(kbps);
        let rate = u64::from(self.config.sample_rate);
        let mut padding = false;
        let mut next = *rest + remainder;
        if next >= rate {
            next -= rate;
            padding = true;
        }
        if commit {
            *rest = next;
        }
        FrameSize {
            index: bitrate_index(self.version, kbps).unwrap_or(1),
            kbps,
            bytes: bytes + usize::from(padding),
            padding,
        }
    }

    fn side_bytes(&self) -> usize {
        match (self.version.lsf(), self.config.channels) {
            (false, 1) => 17,
            (false, _) => 32,
            (true, 1) => 9,
            (true, _) => 17,
        }
    }

    /// The largest `main_data_begin`: the field width, and the 7680-bit
    /// decoder buffer of ISO/IEC 11172-3 2.4.3.1 less this frame.
    fn reservoir_cap(&self, frame_bytes: usize) -> usize {
        let field = if self.version.lsf() { 255 } else { 511 };
        field.min(960usize.saturating_sub(frame_bytes))
    }

    fn sample(&self, ch: usize, at: i64) -> f32 {
        let index = at - self.buffer_start;
        if index < 0 {
            return 0.0;
        }
        self.buffer[ch].get(index as usize).copied().unwrap_or(0.0)
    }

    fn window(&self, ch: usize, from: i64, len: usize) -> Vec<f32> {
        (0..len as i64).map(|i| self.sample(ch, from + i)).collect()
    }

    /// The samples a granule's analysis needs have arrived up to `end`.
    fn available(&self) -> i64 {
        self.buffer_start + self.buffer.first().map_or(0, Vec::len) as i64
    }

    /// Code every granule whose lookahead has arrived.
    fn run(&mut self, flush_to: Option<i64>) -> Result<Vec<Packet>, CodecError> {
        let mut packets = Vec::new();
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(CodecError::Cancelled);
            }
            let g = self.next_granule;
            let start = g * 576;
            let ready = match flush_to {
                Some(end) => start < end,
                None => start + LOOKAHEAD <= self.available(),
            };
            if !ready {
                break;
            }
            self.prepare_granule()?;
            if self.frame_granules.len() == self.granules {
                packets.extend(self.code_frame()?);
            }
            self.trim_buffer();
        }
        Ok(packets)
    }

    /// Whether granule `g` starts with an attack, from its attack window.
    fn attack_at(&mut self, g: i64) -> bool {
        let from = g * 576 - 512;
        let mut any = false;
        for ch in 0..self.config.channels {
            let samples = self.window(ch, from, 576);
            any |= self.attack[ch].feed(&samples);
        }
        any
    }

    fn prepare_granule(&mut self) -> Result<(), CodecError> {
        let g = self.next_granule;
        while self.wanted.len() < 2 {
            let at = g + self.wanted.len() as i64;
            let attack = self.attack_at(at);
            self.wanted.push_back(attack);
        }
        let want_now = self.wanted.pop_front().unwrap_or(false);
        let want_next = self.wanted.front().copied().unwrap_or(false);
        // A start block always leads into a short one, and a short run
        // ends with a stop block, so the window halves meet.
        let block_type = match self.last_block {
            START => SHORT,
            SHORT if want_now || want_next => SHORT,
            SHORT => STOP,
            _ if want_now || want_next => START,
            _ => NORMAL,
        };
        self.last_block = block_type;

        let channels = self.config.channels;
        let centre = g * 576 - 224;
        let mut xr = Vec::with_capacity(channels);
        let mut masking = Vec::with_capacity(channels);
        for ch in 0..channels {
            let long = self.window(ch, centre - 512, 1024);
            let shorts: Vec<Vec<f32>> = (0..3)
                .map(|w| self.window(ch, g * 576 - 544 + 192 * w, 256))
                .collect();
            masking.push(
                self.psy
                    .analyze(ch, &long, [&shorts[0], &shorts[1], &shorts[2]]),
            );
            let mut current = [[0f32; 32]; 18];
            for (t, slot) in current.iter_mut().enumerate() {
                let input = self.window(ch, g * 576 + 32 * t as i64, 32);
                self.analysis[ch].run(&input, slot);
            }
            let mut spectrum = [0f32; 576];
            mdct(
                &self.previous_subbands[ch],
                &current,
                block_type,
                self.rate_table,
                &mut spectrum,
            );
            self.previous_subbands[ch] = current;
            self.lowpass(&mut spectrum, block_type);
            xr.push(spectrum);
        }
        self.frame_granules.push(Prepared {
            block_type,
            xr,
            masking,
        });
        self.next_granule += 1;
        Ok(())
    }

    /// Zero the lines above the lowpass. Short blocks hold a third of
    /// the lines per window, in bitstream order.
    fn lowpass(&self, xr: &mut [f32; 576], block_type: u8) {
        if block_type == SHORT {
            let starts = crate::tables::short_starts(self.rate_table);
            let widths = crate::tables::SHORT_WIDTHS[self.rate_table];
            for band in 0..13 {
                let width = usize::from(widths[band]);
                let from = usize::from(starts[band]);
                for w in 0..3 {
                    for j in 0..width {
                        if 3 * (from + j) >= self.lowpass_line {
                            xr[3 * from + w * width + j] = 0.0;
                        }
                    }
                }
            }
        } else {
            for value in xr.iter_mut().skip(self.lowpass_line) {
                *value = 0.0;
            }
        }
    }

    fn trim_buffer(&mut self) {
        // Keep what the next granule's long FFT reaches back to.
        let keep_from = self.next_granule * 576 - HISTORY;
        let drop = keep_from - self.buffer_start;
        if drop > 4096 {
            let drop = drop as usize;
            for channel in &mut self.buffer {
                channel.drain(..drop.min(channel.len()));
            }
            self.buffer_start += drop as i64;
        }
    }

    /// Allowed noise per band of `bands` for one channel: the model's
    /// ratio times the band's energy. A long band whose MDCT energy is
    /// mostly window leakage the model's spectrum does not show would get
    /// more than the model's absolute threshold; the threshold, carried
    /// over by the granule's MDCT-to-spectrum energy, caps it.
    fn xmin(&self, xr: &[f32; 576], masking: &Masking, block_type: u8) -> Vec<f32> {
        let bands = quantize::bands(block_type, self.rate_table);
        let energies: Vec<f32> = bands
            .iter()
            .map(|band| xr[band.from..band.to].iter().map(|x| x * x).sum())
            .collect();
        let spectrum: f32 = masking.long_energy.iter().sum();
        let mdct: f32 = energies.iter().sum();
        let carry = if spectrum > 0.0 { mdct / spectrum } else { 0.0 };
        bands
            .iter()
            .zip(&energies)
            .map(|(band, energy)| {
                let allowed = match band.slot {
                    Slot::Long(index) => {
                        let cap = masking.long[index] * masking.long_energy[index] * carry;
                        (masking.long[index] * energy).min(cap)
                    }
                    Slot::Fixed(None) => masking.long[21] * energy,
                    Slot::Short(index, w) => masking.short[index][w] * energy,
                    Slot::Fixed(Some(w)) => masking.short[12][w] * energy,
                };
                // A floor keeps silent bands from asking for infinite precision.
                allowed.max(1e-14)
            })
            .collect()
    }

    fn code_frame(&mut self) -> Result<Vec<Packet>, CodecError> {
        let prepared = std::mem::take(&mut self.frame_granules);
        let channels = self.config.channels;
        let lsf = self.version.lsf();

        // Allowed noise per channel, in left/right terms.
        let mut xmins: Vec<Vec<Vec<f32>>> = prepared
            .iter()
            .map(|granule| {
                (0..channels)
                    .map(|ch| self.xmin(&granule.xr[ch], &granule.masking[ch], granule.block_type))
                    .collect()
            })
            .collect();
        let mut spectra: Vec<Vec<[f32; 576]>> = prepared.iter().map(|g| g.xr.clone()).collect();
        let mid_side = channels == 2 && self.choose_mid_side(&prepared);
        if mid_side {
            for (granule, xmin) in spectra.iter_mut().zip(xmins.iter_mut()) {
                let half = std::f32::consts::FRAC_1_SQRT_2;
                let (left, right) = granule.split_at_mut(1);
                for (l, r) in left[0].iter_mut().zip(right[0].iter_mut()) {
                    let (m, s) = ((*l + *r) * half, (*l - *r) * half);
                    *l = m;
                    *r = s;
                }
                let (left, right) = xmin.split_at_mut(1);
                for (l, r) in left[0].iter_mut().zip(right[0].iter_mut()) {
                    let both = l.min(*r);
                    *l = both;
                    *r = both;
                }
            }
        }

        let free = self.assembler.free();
        let side = self.side_bytes();
        let (coded, size) = match self.config.rate {
            Mp3Rate::Constant(kbps) => {
                let mut rest = self.padding_rest;
                let size = self.size_for(kbps, true, &mut rest);
                self.padding_rest = rest;
                let reservoir = free.min(self.reservoir_cap(size.bytes));
                let slot = size.bytes - 4 - side;
                let coded = self.code_constant(&prepared, &spectra, &xmins, reservoir, slot, lsf);
                (coded, size)
            }
            Mp3Rate::Variable(_) => {
                self.code_variable(&prepared, &spectra, &xmins, free, side, lsf)
            }
        };
        let reservoir = free.min(self.reservoir_cap(size.bytes));
        self.assembler.stuff(reservoir);

        // Main data: scalefactors and Huffman code, granule by granule.
        let mut writer = BitWriter::new();
        let mut side_info = SideInfo {
            main_data_begin: reservoir as u16,
            ..SideInfo::default()
        };
        for (gr, granule) in coded.iter().enumerate() {
            for (ch, code) in granule.iter().enumerate() {
                let ctx = Context::new(lsf, self.rate_table, code.info.block_type);
                write_granule(&mut writer, &ctx, code, lsf);
                side_info.granules[gr][ch] = code.info;
            }
        }
        let main = writer.finish();
        let header = Self::header_for(
            self.version,
            self.rate_index,
            self.config.sample_rate,
            size.index,
            size.kbps,
            size.padding,
            channels,
            mid_side,
        );
        let mut head = BitWriter::new();
        for byte in header.to_bytes() {
            head.put(u32::from(byte), 8);
        }
        side_info.write(&header, &mut head);
        let mut bytes = head.finish();
        let slot = bytes.len();
        bytes.resize(size.bytes, 0);
        let pts = self.frame_index * (self.granules as i64 * 576);
        self.frame_index += 1;
        let fits = self.assembler.push(
            Pending {
                bytes,
                slot,
                filled: 0,
                pts,
            },
            &main,
        );
        if !fits {
            return Err(CodecError::Limit { what: "main data" });
        }
        let cap = self.reservoir_cap(size.bytes);
        self.assembler.stuff(cap);
        let ready = self.assembler.ready();
        Ok(self.packets(ready, 0))
    }

    /// Mid/side when the side channel is much quieter than the mid, which
    /// is when it saves bits; both channels then share block types.
    fn choose_mid_side(&self, prepared: &[Prepared]) -> bool {
        let mut mid = 0f64;
        let mut side = 0f64;
        let mut left = 0f64;
        let mut right = 0f64;
        for granule in prepared {
            for (l, r) in granule.xr[0].iter().zip(granule.xr[1].iter()) {
                let (l, r) = (f64::from(*l), f64::from(*r));
                left += l * l;
                right += r * r;
                mid += (l + r) * (l + r) / 2.0;
                side += (l - r) * (l - r) / 2.0;
            }
        }
        // The smaller channel's share of the energy, in each representation.
        let lr = left.min(right) / (left + right).max(1e-30);
        let ms = side.min(mid) / (mid + side).max(1e-30);
        ms < lr * 0.8
    }

    /// Constant bitrate: each granule takes what its noise target needs
    /// when the frame and the reservoir hold it, else a budget split by
    /// that need. Bits the reservoir cannot keep are spent, not stuffed.
    fn code_constant(
        &self,
        prepared: &[Prepared],
        spectra: &[Vec<[f32; 576]>],
        xmins: &[Vec<Vec<f32>>],
        reservoir: usize,
        slot: usize,
        lsf: bool,
    ) -> Vec<Vec<Coded>> {
        let channels = self.config.channels;
        let granules = prepared.len();
        let mean = (slot * 8 / granules) as i64;
        let mut available = ((reservoir + slot) * 8) as i64;
        let mut spare = (reservoir * 8) as i64;
        let cap = (self.reservoir_cap(slot + 4 + self.side_bytes()) * 8) as i64;
        let mut out = Vec::with_capacity(granules);
        for (gr, granule) in prepared.iter().enumerate() {
            let ctx = Context::new(lsf, self.rate_table, granule.block_type);
            let code = |ch: usize, target: Target| {
                quantize::code(
                    &ctx,
                    &Spectrum {
                        xr: &spectra[gr][ch],
                        xmin: &xmins[gr][ch],
                        block_type: granule.block_type,
                    },
                    target,
                )
            };
            let wanted: Vec<Coded> = (0..channels)
                .map(|ch| code(ch, Target::Noise(MAX_GRANULE_BITS)))
                .collect();
            let need: Vec<i64> = wanted.iter().map(|c| i64::from(c.bits())).collect();
            let total_need: i64 = need.iter().sum();
            // Leave the later granules of this frame half their share.
            let remaining = (granules - gr - 1) as i64;
            let ceiling = (available - remaining * mean / 2).max(0);
            // A hard granule borrows up to 60% of the reservoir; bits the
            // reservoir cannot hold past this frame are spent now.
            let overflow = (spare + mean - cap).max(0);
            let borrow = (spare * 3 / 5).max(overflow);
            let target = (mean + borrow).min(ceiling);
            let row: Vec<Coded> = if total_need <= target && overflow == 0 {
                wanted
            } else {
                let target = if overflow > 0 {
                    target
                } else {
                    target.min(total_need)
                };
                let mut left = target;
                (0..channels)
                    .map(|ch| {
                        let share = if ch + 1 == channels {
                            left
                        } else {
                            target * need[ch].max(1) / total_need.max(1)
                        };
                        let budget = share.clamp(0, i64::from(MAX_GRANULE_BITS)) as u32;
                        let coded = code(ch, Target::Budget(budget));
                        left -= i64::from(coded.bits());
                        coded
                    })
                    .collect()
            };
            let used: i64 = row.iter().map(|c| i64::from(c.bits())).sum();
            available -= used;
            spare += mean - used;
            out.push(row);
        }
        out
    }

    fn code_variable(
        &mut self,
        prepared: &[Prepared],
        spectra: &[Vec<[f32; 576]>],
        xmins: &[Vec<Vec<f32>>],
        free: usize,
        side: usize,
        lsf: bool,
    ) -> (Vec<Vec<Coded>>, FrameSize) {
        let channels = self.config.channels;
        let code_all = |target: &dyn Fn(usize, usize) -> Target| -> Vec<Vec<Coded>> {
            prepared
                .iter()
                .enumerate()
                .map(|(gr, granule)| {
                    (0..channels)
                        .map(|ch| {
                            let ctx = Context::new(lsf, self.rate_table, granule.block_type);
                            quantize::code(
                                &ctx,
                                &Spectrum {
                                    xr: &spectra[gr][ch],
                                    xmin: &xmins[gr][ch],
                                    block_type: granule.block_type,
                                },
                                target(gr, ch),
                            )
                        })
                        .collect()
                })
                .collect()
        };
        let coded = code_all(&|_, _| Target::Noise(MAX_GRANULE_BITS));
        let needed: usize = coded
            .iter()
            .flatten()
            .map(|c| c.bits() as usize)
            .sum::<usize>()
            .div_ceil(8);
        let rates = bitrates(self.version);
        let mut rest = self.padding_rest;
        let choice = rates.iter().copied().find(|&kbps| {
            let size = self.size_for(kbps, false, &mut rest);
            let reservoir = free.min(self.reservoir_cap(size.bytes));
            size.bytes - 4 - side + reservoir >= needed
        });
        let (kbps, coded) = match choice {
            Some(kbps) => (kbps, coded),
            None => {
                // Even the top rate is short: code to its budget instead.
                let top = rates[13];
                let size = self.size_for(top, false, &mut rest);
                let reservoir = free.min(self.reservoir_cap(size.bytes));
                let total = ((size.bytes - 4 - side + reservoir) * 8) as u32;
                let parts = (prepared.len() * channels) as u32;
                let coded = code_all(&|_, _| Target::Budget(total / parts));
                (top, coded)
            }
        };
        let mut rest = self.padding_rest;
        let size = self.size_for(kbps, true, &mut rest);
        self.padding_rest = rest;
        (coded, size)
    }

    fn packets(&self, frames: Vec<Pending>, padding: u32) -> Vec<Packet> {
        let spf = (self.granules * 576) as i64;
        let count = frames.len();
        frames
            .into_iter()
            .enumerate()
            .map(|(index, frame)| {
                let mut packet =
                    Packet::audio(frame.bytes, frame.pts, spf, self.config.sample_rate);
                packet.side.push(SideData::Gapless {
                    encoder_delay: ENCODER_DELAY,
                    trailing_padding: if index + 1 == count { padding } else { 0 },
                    sample_rate: self.config.sample_rate,
                });
                packet
            })
            .collect()
    }

    fn accept(&mut self, input: &AudioInput) -> Result<(), CodecError> {
        let pcm = &input.pcm;
        if pcm.rate != self.config.sample_rate {
            return Err(CodecError::InvalidApi {
                what: "sample rate",
            });
        }
        let channels = pcm.layout.len();
        if channels != self.config.channels {
            return Err(CodecError::InvalidApi { what: "layout" });
        }
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
        for (index, sample) in samples.iter().enumerate() {
            self.buffer[index % channels].push(*sample);
        }
        self.accepted += (samples.len() / channels) as u64;
        Ok(())
    }
}

/// Scalefactors, then the Huffman code, of one granule of one channel.
fn write_granule(writer: &mut BitWriter, ctx: &Context, coded: &Coded, lsf: bool) {
    let info = &coded.info;
    let short = info.window_switching && info.block_type == SHORT;
    if lsf {
        let counts: [usize; 4] = if short { [9, 9, 9, 9] } else { [6, 5, 5, 5] };
        let values: Vec<u8> = if short {
            (0..12).flat_map(|b| coded.sf_short[b]).collect()
        } else {
            coded.sf_long[..21].to_vec()
        };
        let mut at = 0;
        for (part, count) in counts.iter().enumerate() {
            for value in &values[at..at + count] {
                writer.put(u32::from(*value), u32::from(coded.slen[part]));
            }
            at += count;
        }
    } else if short {
        for band in 0..12 {
            let slen = if band < 6 {
                coded.slen[0]
            } else {
                coded.slen[1]
            };
            for w in 0..3 {
                writer.put(u32::from(coded.sf_short[band][w]), u32::from(slen));
            }
        }
    } else {
        for band in 0..21 {
            let slen = if band < 11 {
                coded.slen[0]
            } else {
                coded.slen[1]
            };
            writer.put(u32::from(coded.sf_long[band]), u32::from(slen));
        }
    }
    let big_end = coded.big_end;
    let bounds = if info.window_switching {
        let r0 = ctx.switch_region0.min(big_end);
        [(0, r0), (r0, big_end), (big_end, big_end)]
    } else {
        let (r0, r1) = long_bounds(
            ctx.rate_table,
            info.region0_count,
            info.region1_count,
            big_end,
        );
        [(0, r0), (r0, r1), (r1, big_end)]
    };
    for (region, (from, to)) in bounds.into_iter().enumerate() {
        write_pairs(writer, &coded.ix, from, to, info.table_select[region]);
    }
    write_count1(
        writer,
        &coded.ix,
        big_end,
        coded.count1_end,
        info.count1_table_b,
    );
}

impl Encoder for Mp3Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Mp3
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.drained {
            return Err(CodecError::InvalidApi {
                what: "push after drain",
            });
        }
        self.accept(input)?;
        self.run(None)
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
        let spf = (self.granules * 576) as u64;
        // Enough frames that the decoder's output reaches the last input
        // sample; past the input the encoder reads silence.
        let frames = (self.accepted + 1057).div_ceil(spf);
        let end = (frames * spf) as i64;
        let mut packets = self.run(Some(end))?;
        let padding = (frames * spf - self.accepted - u64::from(ENCODER_DELAY)) as u32;
        let tail = self.assembler.flush();
        let mut last = self.packets(tail, padding);
        if last.is_empty()
            && let Some(packet) = packets.last_mut()
        {
            for side in &mut packet.side {
                if let SideData::Gapless {
                    trailing_padding, ..
                } = side
                {
                    *trailing_padding = padding;
                }
            }
        }
        packets.append(&mut last);
        Ok(packets)
    }

    fn reset(&mut self) {
        let Ok(fresh) = Self::new(self.config) else {
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
