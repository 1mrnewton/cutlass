//! The AC-3 and E-AC-3 decoder: access units of syncframes, programs and
//! substreams, the channel map, downmix, and gapless trim.

use cutlass_decode::{
    AudioOutput, Channel, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet,
    Pcm, SideData,
};

use crate::frame::{self, Decoded, DynamicRange, MAX_FBW, Substream, Tools, Transient};
use crate::header::{self, Metadata, PROBE_BYTES, StreamType};
use crate::mdct::Transform;

/// Most dependent substreams one program carries (`substreamid` is 3 bits).
const MAX_DEPENDENTS: usize = 8;

/// Most syncframes one access unit may hold: six one-block frames of each
/// of eight independent programs with eight dependents each.
const MAX_FRAMES_PER_UNIT: usize = 6 * 8 * 9;

/// A stereo or mono downmix of the program's independent substream
/// (A/52 section 7.8). The LFE channel is not mixed in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Downmix {
    /// Every coded channel, dependent substreams included.
    #[default]
    None,
    /// Lo/Ro stereo with the stream's center and surround mix levels.
    LoRo,
    /// Lt/Rt matrix-surround-compatible stereo.
    LtRt,
    /// The Lo/Ro pair summed to one channel.
    Mono,
}

/// What a channel location of a dependent substream the speaker set cannot
/// carry does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Extension {
    /// Refuse the unit with [`CodecError::Unsupported`].
    #[default]
    Refuse,
    /// Play the independent substream alone (Annex E 3.8.2) and count the
    /// skipped dependent substream in [`Ac3Decoder::skipped_dependents`].
    Core,
}

/// Decoder options. The default is full range, every channel, program 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub dynamic_range: DynamicRange,
    pub downmix: Downmix,
    /// The independent substream (`substreamid`) to play.
    pub program: u8,
    pub extension: Extension,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dynamic_range: DynamicRange::Off,
            downmix: Downmix::None,
            program: 0,
            extension: Extension::Refuse,
        }
    }
}

/// Speaker locations of Table 5.8 and Table E2.5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    L,
    C,
    R,
    Ls,
    Rs,
    Lc,
    Rc,
    Lrs,
    Rrs,
    Cs,
    Ts,
    Lsd,
    Rsd,
    Lw,
    Rw,
    Vhl,
    Vhr,
    Vhc,
    Lts,
    Rts,
    Lfe2,
    Lfe,
}

impl Location {
    /// The speaker a location plays on. A/52 surrounds are side
    /// channels; the rear surround pair of a 7.1 program is the back pair.
    fn speaker(self) -> Option<Channel> {
        Some(match self {
            Self::L => Channel::FrontLeft,
            Self::R => Channel::FrontRight,
            Self::C => Channel::FrontCenter,
            Self::Lfe => Channel::Lfe,
            Self::Lrs => Channel::BackLeft,
            Self::Rrs => Channel::BackRight,
            Self::Lc => Channel::FrontLeftOfCenter,
            Self::Rc => Channel::FrontRightOfCenter,
            Self::Cs => Channel::BackCenter,
            Self::Ls => Channel::SideLeft,
            Self::Rs => Channel::SideRight,
            _ => return None,
        })
    }
}

/// Output order: WAVEFORMATEXTENSIBLE speaker order.
const SPEAKER_ORDER: [Channel; 11] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::Lfe,
    Channel::BackLeft,
    Channel::BackRight,
    Channel::FrontLeftOfCenter,
    Channel::FrontRightOfCenter,
    Channel::BackCenter,
    Channel::SideLeft,
    Channel::SideRight,
];

fn speaker_rank(channel: Channel) -> usize {
    SPEAKER_ORDER
        .iter()
        .position(|c| *c == channel)
        .unwrap_or(SPEAKER_ORDER.len())
}

/// Coded channels of an `acmod` in bit stream order (Table 5.8). 1+1 plays
/// its two programs as left and right.
fn acmod_locations(acmod: u8, lfe: bool) -> Vec<Location> {
    use Location::{C, Cs, L, Lfe, Ls, R, Rs};
    let mut out = match acmod {
        0 | 2 => vec![L, R],
        1 => vec![C],
        3 => vec![L, C, R],
        4 => vec![L, R, Cs],
        5 => vec![L, C, R, Cs],
        6 => vec![L, R, Ls, Rs],
        _ => vec![L, C, R, Ls, Rs],
    };
    if lfe {
        out.push(Lfe);
    }
    out
}

/// Coded channels of a dependent substream's `chanmap` (Table E2.5), in
/// bit stream order.
fn chanmap_locations(chanmap: u16) -> Vec<Location> {
    use Location::*;
    const TABLE: [&[Location]; 16] = [
        &[L],
        &[C],
        &[R],
        &[Ls],
        &[Rs],
        &[Lc, Rc],
        &[Lrs, Rrs],
        &[Cs],
        &[Ts],
        &[Lsd, Rsd],
        &[Lw, Rw],
        &[Vhl, Vhr],
        &[Vhc],
        &[Lts, Rts],
        &[Lfe2],
        &[Lfe],
    ];
    let mut out = Vec::new();
    for (bit, locations) in TABLE.iter().enumerate() {
        if chanmap & (0x8000 >> bit) != 0 {
            out.extend_from_slice(locations);
        }
    }
    out
}

/// The substream states one push may change, copied and committed whole.
#[derive(Clone)]
struct States {
    independent: Substream,
    dependents: Vec<Substream>,
    tools: Tools,
    skipped_dependents: u64,
}

impl States {
    fn new() -> Self {
        Self {
            independent: Substream::new(),
            dependents: vec![Substream::new(); MAX_DEPENDENTS],
            tools: Tools::default(),
            skipped_dependents: 0,
        }
    }
}

/// One frame of the selected program before channel assembly.
struct Slot {
    sample_rate: u32,
    samples: usize,
    channels: Vec<(Channel, Vec<f32>)>,
    metadata: Metadata,
    transients: [Option<Transient>; MAX_FBW],
}

/// Gapless trim from the container, in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gapless {
    encoder_delay: u32,
    padding: u32,
}

/// Native AC-3 and E-AC-3 decoder.
///
/// One push is one access unit: the syncframes that cover one span of
/// time, as an MP4 sample holds them. That is one AC-3 syncframe, or an
/// E-AC-3 independent substream frame with its dependent substream frames
/// and the frames of other programs, repeated when frames are shorter than
/// six blocks. Only [`Options::program`] and its dependent substreams are
/// decoded; other programs are skipped.
///
/// Output is interleaved `f32` at full scale 1.0 in WAVEFORMATEXTENSIBLE
/// speaker order. The layout locks on the first decoded unit; a later unit
/// with other channels is mapped onto it, and channels the locked layout
/// lacks are mixed into the front pair with the Lo/Ro levels.
///
/// The container's channel count is not checked: MP4 writes 2 for every
/// AC-3 stream. A non-zero `sample_rate` must match the frames.
pub struct Ac3Decoder {
    codec: CodecId,
    options: Options,
    transform: Box<Transform>,
    states: States,
    open_rate: u32,
    layout: Option<ChannelLayout>,
    rate: u32,
    metadata: Option<Metadata>,
    transients: [Option<Transient>; MAX_FBW],
    gapless: Option<Gapless>,
    held: Vec<f32>,
    held_start: i64,
    clock: i64,
    timescale: u32,
    drained: bool,
}

impl Ac3Decoder {
    /// Open a decoder for [`CodecId::Ac3`] or [`CodecId::Eac3`]. `extra`
    /// (a `dac3` or `dec3` payload) is not needed: every syncframe carries
    /// its own configuration. Either id decodes both syntaxes, since an
    /// E-AC-3 program may carry an AC-3 core.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Self::with_options(params, Options::default())
    }

    pub fn with_options(params: &CodecParams, options: Options) -> Result<Self, CodecError> {
        if params.codec != CodecId::Ac3 && params.codec != CodecId::Eac3 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if !matches!(params.sample_rate, 0 | 32_000 | 44_100 | 48_000) {
            return Err(CodecError::Unsupported {
                what: "sample rate",
            });
        }
        if options.program > 7 {
            return Err(CodecError::InvalidApi { what: "program" });
        }
        Ok(Self {
            codec: params.codec,
            options,
            transform: Box::new(Transform::new()),
            states: States::new(),
            open_rate: params.sample_rate,
            layout: None,
            rate: 0,
            metadata: None,
            transients: [None; MAX_FBW],
            gapless: None,
            held: Vec::new(),
            held_start: 0,
            clock: 0,
            timescale: 0,
            drained: false,
        })
    }

    /// The program metadata of the last decoded unit's independent
    /// substream.
    #[must_use]
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Transient pre-noise processing data of the last decoded unit, per
    /// full bandwidth channel of the independent substream. The decoder
    /// reports it and does not apply it.
    #[must_use]
    pub fn transients(&self) -> &[Option<Transient>; MAX_FBW] {
        &self.transients
    }

    /// Dependent substream frames skipped under [`Extension::Core`] since
    /// open or the last reset.
    #[must_use]
    pub fn skipped_dependents(&self) -> u64 {
        self.states.skipped_dependents
    }

    /// The coding tools of every syncframe decoded since open or the last
    /// reset.
    #[must_use]
    pub fn tools(&self) -> &Tools {
        &self.states.tools
    }

    /// The locked output layout and sample rate once a unit decoded.
    #[must_use]
    pub fn stream(&self) -> Option<(u32, &ChannelLayout)> {
        self.layout.as_ref().map(|layout| (self.rate, layout))
    }

    /// Split an access unit into syncframes.
    fn frames(data: &[u8]) -> Result<Vec<&[u8]>, CodecError> {
        let mut frames = Vec::new();
        let mut at = 0;
        while at < data.len() {
            let rest = &data[at..];
            if rest.len() < PROBE_BYTES {
                return Err(CodecError::Malformed {
                    at_bit: at as u64 * 8,
                    what: "bytes after the last syncframe",
                });
            }
            let info = header::probe(rest).map_err(|error| shift(error, at))?;
            let Some(frame) = rest.get(..info.frame_bytes) else {
                return Err(CodecError::Truncated {
                    at_bit: data.len() as u64 * 8,
                    what: "syncframe",
                });
            };
            if frames.len() == MAX_FRAMES_PER_UNIT {
                return Err(CodecError::Limit {
                    what: "syncframes per access unit",
                });
            }
            frames.push(frame);
            at += info.frame_bytes;
        }
        Ok(frames)
    }

    /// Decode every frame of the selected program on `states`.
    fn decode_unit(&mut self, data: &[u8], states: &mut States) -> Result<Vec<Slot>, CodecError> {
        let mut parsed = Vec::new();
        let mut offset = 0usize;
        for frame in Self::frames(data)? {
            let bsi = header::parse(frame).map_err(|error| shift(error, offset))?;
            parsed.push((offset, frame, bsi));
            offset += frame.len();
        }
        // Each independent frame of the program, with the dependent
        // frames that follow it.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut in_program = false;
        for (index, (at, _, bsi)) in parsed.iter().enumerate() {
            if bsi.stream_type != StreamType::Dependent {
                in_program = bsi.substream_id == self.options.program;
                if in_program {
                    groups.push((index, Vec::new()));
                }
            } else if in_program {
                let Some(group) = groups.last_mut() else {
                    return Err(CodecError::Malformed {
                        at_bit: *at as u64 * 8,
                        what: "dependent substream before its independent substream",
                    });
                };
                group.1.push(index);
            }
        }
        if groups.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "no frame of the selected program",
            });
        }
        let dynamics = self.options.dynamic_range;
        let mut slots = Vec::with_capacity(groups.len());
        for (head, dependents) in groups {
            let (at, frame, bsi) = &parsed[head];
            // Annex E 3.8.5: reproducing the complete mix, the last
            // dependent substream's dynamic range words apply to the whole
            // program; a downmix keeps independent substream 0's.
            let mut last = None;
            let mut words = None;
            if dynamics != DynamicRange::Off
                && self.options.downmix == Downmix::None
                && let Some(&index) = dependents.last()
            {
                let (dat, dframe, dbsi) = &parsed[index];
                let state = &mut states.dependents[usize::from(dbsi.substream_id)];
                let decoded = frame::decode(dframe, dbsi, state, &self.transform, dynamics, None)
                    .map_err(|error| shift(error, *dat))?;
                words = Some(decoded.drc);
                last = Some((index, decoded));
            }
            let decoded = frame::decode(
                frame,
                bsi,
                &mut states.independent,
                &self.transform,
                dynamics,
                words.as_ref(),
            )
            .map_err(|error| shift(error, *at))?;
            states.tools.add(&decoded.tools);
            let mut slot = Slot {
                sample_rate: bsi.sample_rate,
                samples: decoded.samples,
                channels: place(&decoded, &acmod_locations(bsi.acmod, bsi.lfeon))?,
                metadata: bsi.metadata.clone(),
                transients: decoded.transients,
            };
            for index in dependents {
                let (dat, dframe, dbsi) = &parsed[index];
                let cached = match last.take() {
                    Some((cached_index, decoded)) if cached_index == index => Some(decoded),
                    other => {
                        last = other;
                        None
                    }
                };
                if dbsi.sample_rate != slot.sample_rate || dbsi.blocks * 256 != slot.samples {
                    return Err(CodecError::Malformed {
                        at_bit: *dat as u64 * 8,
                        what: "dependent substream timing",
                    });
                }
                if let Some(decoded) = &cached {
                    states.tools.add(&decoded.tools);
                }
                if self.options.downmix != Downmix::None {
                    continue;
                }
                let locations = match dbsi.chanmap {
                    Some(map) => chanmap_locations(map),
                    None => acmod_locations(dbsi.acmod, dbsi.lfeon),
                };
                if locations.iter().any(|l| l.speaker().is_none()) {
                    if self.options.extension == Extension::Core {
                        states.skipped_dependents += 1;
                        continue;
                    }
                    return Err(CodecError::Unsupported {
                        what: "E-AC-3 channel location outside the 11 speakers",
                    });
                }
                let decoded = match cached {
                    Some(decoded) => decoded,
                    None => {
                        let state = &mut states.dependents[usize::from(dbsi.substream_id)];
                        let decoded = frame::decode(
                            dframe,
                            dbsi,
                            state,
                            &self.transform,
                            dynamics,
                            words.as_ref(),
                        )
                        .map_err(|error| shift(error, *dat))?;
                        states.tools.add(&decoded.tools);
                        decoded
                    }
                };
                for (speaker, pcm) in place(&decoded, &locations)? {
                    match slot.channels.iter_mut().find(|(c, _)| *c == speaker) {
                        Some(existing) => existing.1 = pcm,
                        None => slot.channels.push((speaker, pcm)),
                    }
                }
                if slot.channels.len() > 8 {
                    return Err(CodecError::Unsupported {
                        what: "more than 8 channels",
                    });
                }
            }
            slots.push(slot);
        }
        Ok(slots)
    }

    /// Interleave one slot into `layout`, mixing channels `layout` lacks
    /// into the front pair.
    fn render(&self, slot: &Slot, layout: &ChannelLayout, out: &mut Vec<f32>) {
        let n = layout.len();
        let start = out.len();
        out.resize(start + slot.samples * n, 0.0);
        let frame_out = &mut out[start..];
        let levels = levels(&slot.metadata);
        for (speaker, pcm) in &slot.channels {
            let targets: Vec<(usize, f32)> =
                match layout.channels().iter().position(|c| c == speaker) {
                    Some(index) => vec![(index, 1.0)],
                    None => fallback(*speaker, layout, levels),
                };
            for (index, gain) in targets {
                for (i, sample) in pcm.iter().enumerate() {
                    frame_out[i * n + index] += gain * sample;
                }
            }
        }
    }

    fn output_layout(&self, slot: &Slot) -> Result<ChannelLayout, CodecError> {
        let layout = match self.options.downmix {
            Downmix::None => {
                let mut speakers: Vec<Channel> = slot.channels.iter().map(|(c, _)| *c).collect();
                speakers.sort_by_key(|c| speaker_rank(*c));
                ChannelLayout::new(speakers)
            }
            Downmix::LoRo | Downmix::LtRt => Ok(ChannelLayout::stereo()),
            Downmix::Mono => Ok(ChannelLayout::mono()),
        };
        layout.map_err(|_| CodecError::Unsupported {
            what: "more than 8 channels",
        })
    }

    /// Apply [`Options::downmix`] to one slot's channels.
    fn downmix(&self, slot: &mut Slot) {
        let mode = self.options.downmix;
        if mode == Downmix::None {
            return;
        }
        let levels = levels(&slot.metadata);
        let get = |slot: &Slot, c: Channel| {
            slot.channels
                .iter()
                .find(|(s, _)| *s == c)
                .map(|(_, p)| p.clone())
        };
        let samples = slot.samples;
        let silent = vec![0.0f32; samples];
        let l = get(slot, Channel::FrontLeft);
        let r = get(slot, Channel::FrontRight);
        let c = get(slot, Channel::FrontCenter);
        let s = get(slot, Channel::BackCenter);
        let ls = get(slot, Channel::SideLeft);
        let rs = get(slot, Channel::SideRight);
        let lfe = get(slot, Channel::Lfe);
        let mono_source = l.is_none() && r.is_none();
        if mono_source && mode == Downmix::Mono {
            slot.channels
                .retain(|(speaker, _)| *speaker == Channel::FrontCenter);
            return;
        }
        // Annex E 3.9: with a mix level code, LFE joins a two-channel or
        // mono downmix at that level less 4.5 dB.
        let lfe_gain = slot
            .metadata
            .lfe_mix_db
            .map_or(0.0, |db| 10f32.powf((f32::from(db) - 4.5) / 20.0));
        let (clev, slev) = if mode == Downmix::LtRt {
            (levels.ltrt_center, levels.ltrt_surround)
        } else {
            (levels.loro_center, levels.loro_surround)
        };
        let s_gain = std::f32::consts::FRAC_1_SQRT_2 * slev;
        // Per-output coefficients over (L, R, C, S, Ls, Rs, LFE).
        let (left, right): ([f32; 7], [f32; 7]) = if mono_source {
            let g = std::f32::consts::FRAC_1_SQRT_2;
            (
                [0.0, 0.0, g, 0.0, 0.0, 0.0, lfe_gain],
                [0.0, 0.0, g, 0.0, 0.0, 0.0, lfe_gain],
            )
        } else if mode == Downmix::LtRt {
            (
                [1.0, 0.0, clev, -slev, -slev, -slev, lfe_gain],
                [0.0, 1.0, clev, slev, slev, slev, lfe_gain],
            )
        } else {
            (
                [1.0, 0.0, clev, s_gain, slev, 0.0, lfe_gain],
                [0.0, 1.0, clev, s_gain, 0.0, slev, lfe_gain],
            )
        };
        let sources = [&l, &r, &c, &s, &ls, &rs, &lfe];
        let present = |k: usize| sources[k].is_some();
        let norm = |coefs: &[f32; 7]| {
            let sum: f32 = (0..7).filter(|&k| present(k)).map(|k| coefs[k].abs()).sum();
            if sum > 1.0 { 1.0 / sum } else { 1.0 }
        };
        let scale = norm(&left).min(norm(&right));
        let mix = |coefs: &[f32; 7]| {
            let mut out = vec![0.0f32; samples];
            for (k, source) in sources.iter().enumerate() {
                let pcm = source.as_deref().unwrap_or(&silent);
                let g = coefs[k] * scale;
                if g != 0.0 {
                    for (o, x) in out.iter_mut().zip(pcm) {
                        *o += g * x;
                    }
                }
            }
            out
        };
        let lo = mix(&left);
        let ro = mix(&right);
        slot.channels = if mode == Downmix::Mono {
            let mono = lo.iter().zip(&ro).map(|(a, b)| 0.5 * (a + b)).collect();
            vec![(Channel::FrontCenter, mono)]
        } else {
            vec![(Channel::FrontLeft, lo), (Channel::FrontRight, ro)]
        };
    }

    fn ticks(&self, sample: i64, rate: u32) -> i64 {
        if self.timescale == 0 || rate == 0 {
            return 0;
        }
        if self.timescale == rate {
            return sample;
        }
        let scaled = i128::from(sample) * i128::from(self.timescale) / i128::from(rate);
        i64::try_from(scaled).unwrap_or(0)
    }

    fn output(
        &self,
        pcm: Vec<f32>,
        first: i64,
        skip_leading: u32,
        skip_trailing: u32,
    ) -> Result<Output, CodecError> {
        let Some(layout) = self.layout.clone() else {
            return Err(CodecError::InvalidApi {
                what: "no stream yet",
            });
        };
        let pcm =
            Pcm::from_f32(self.rate, layout, pcm).map_err(|_| CodecError::Limit { what: "pcm" })?;
        Ok(Output::Audio(AudioOutput {
            pcm,
            pts: self.ticks(first, self.rate),
            timescale: self.timescale,
            skip_leading,
            skip_trailing,
            concealed: false,
        }))
    }
}

/// Rebase an error's bit position by a frame's byte offset in the unit.
fn shift(error: CodecError, at: usize) -> CodecError {
    let base = at as u64 * 8;
    match error {
        CodecError::Truncated { at_bit, what } => CodecError::Truncated {
            at_bit: at_bit + base,
            what,
        },
        CodecError::Malformed { at_bit, what } => CodecError::Malformed {
            at_bit: at_bit + base,
            what,
        },
        other => other,
    }
}

/// Pair a decoded frame's planar channels with their speakers.
fn place(
    decoded: &Decoded,
    locations: &[Location],
) -> Result<Vec<(Channel, Vec<f32>)>, CodecError> {
    if locations.len() != decoded.channels {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "channel map and coded channels disagree",
        });
    }
    let mut out = Vec::with_capacity(locations.len());
    for (index, location) in locations.iter().enumerate() {
        let Some(speaker) = location.speaker() else {
            return Err(CodecError::Unsupported {
                what: "E-AC-3 channel location outside the 11 speakers",
            });
        };
        out.push((speaker, decoded.channel(index).to_vec()));
    }
    Ok(out)
}

/// Downmix levels from the stream, with the A/52 defaults of -3 dB.
/// The stream's downmix levels. Without extended levels, Lt/Rt uses the
/// fixed 0.707 of section 7.8.2 and Lo/Ro uses `cmixlev` and `surmixlev`;
/// a stream that carries neither (E-AC-3 without mixing metadata) gets
/// their intermediate codes, -4.5 dB and -6 dB, the values section 7.8.2
/// gives the reserved code.
fn levels(metadata: &Metadata) -> header::MixLevels {
    let d = std::f32::consts::FRAC_1_SQRT_2;
    match metadata.mix_levels {
        Some(levels) => levels,
        None => header::MixLevels {
            ltrt_center: d,
            ltrt_surround: d,
            loro_center: metadata.center_mix.unwrap_or(0.594_603_6),
            loro_surround: metadata.surround_mix.unwrap_or(0.5),
        },
    }
}

/// Where a channel the locked layout lacks goes: centers to both fronts,
/// left and right channels to their side, LFE nowhere.
fn fallback(
    speaker: Channel,
    layout: &ChannelLayout,
    levels: header::MixLevels,
) -> Vec<(usize, f32)> {
    let find = |c: Channel| layout.channels().iter().position(|x| *x == c);
    let (fl, fr, fc) = (
        find(Channel::FrontLeft),
        find(Channel::FrontRight),
        find(Channel::FrontCenter),
    );
    let pair = |gain: f32| -> Vec<(usize, f32)> {
        match (fl, fr) {
            (Some(l), Some(r)) => vec![(l, gain), (r, gain)],
            _ => fc.map(|c| vec![(c, gain)]).unwrap_or_default(),
        }
    };
    let side = |left: bool, gain: f32| -> Vec<(usize, f32)> {
        match (left, fl, fr, fc) {
            (true, Some(l), _, _) => vec![(l, gain)],
            (false, _, Some(r), _) => vec![(r, gain)],
            (_, _, _, Some(c)) => vec![(c, gain)],
            _ => Vec::new(),
        }
    };
    let d = std::f32::consts::FRAC_1_SQRT_2;
    match speaker {
        Channel::Lfe => Vec::new(),
        Channel::FrontCenter => pair(levels.loro_center),
        Channel::BackCenter => pair(d * levels.loro_surround),
        Channel::FrontLeft => side(true, 1.0),
        Channel::FrontRight => side(false, 1.0),
        Channel::FrontLeftOfCenter => side(true, d),
        Channel::FrontRightOfCenter => side(false, d),
        Channel::SideLeft | Channel::BackLeft => side(true, levels.loro_surround),
        Channel::SideRight | Channel::BackRight => side(false, levels.loro_surround),
    }
}

fn gapless_of(packet: &Packet) -> Option<(Gapless, u32)> {
    packet.side.iter().find_map(|side| match side {
        SideData::Gapless {
            encoder_delay,
            trailing_padding,
            sample_rate,
        } => Some((
            Gapless {
                encoder_delay: *encoder_delay,
                padding: *trailing_padding,
            },
            *sample_rate,
        )),
        _ => None,
    })
}

impl Decoder for Ac3Decoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut states = self.states.clone();
        let mut slots = self.decode_unit(&packet.data, &mut states)?;
        let rate = slots[0].sample_rate;
        if slots.iter().any(|s| s.sample_rate != rate)
            || (self.rate != 0 && self.rate != rate)
            || (self.open_rate != 0 && self.open_rate != rate)
        {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "sample rate change",
            });
        }
        for slot in &mut slots {
            self.downmix(slot);
        }
        let layout = match &self.layout {
            Some(layout) => layout.clone(),
            None => self.output_layout(&slots[0])?,
        };

        // Everything after this point is infallible: commit.
        let mut pcm = Vec::new();
        for slot in &slots {
            self.render(slot, &layout, &mut pcm);
        }
        let samples: i64 = slots.iter().map(|s| s.samples as i64).sum();
        let last = slots.len() - 1;
        self.metadata = Some(slots[last].metadata.clone());
        self.transients = slots[last].transients;
        self.states = states;
        self.layout = Some(layout.clone());
        self.rate = rate;
        self.drained = false;
        if let Some((gapless, gapless_rate)) = gapless_of(packet)
            && (gapless_rate == rate || gapless_rate == 0)
        {
            self.gapless = Some(gapless);
        }
        self.timescale = packet.timescale;
        let first = if packet.timescale > 0 {
            let scaled = i128::from(packet.pts) * i128::from(rate) / i128::from(packet.timescale);
            i64::try_from(scaled).unwrap_or(self.clock)
        } else {
            self.clock
        };
        self.clock = first.saturating_add(samples);
        let Some(gapless) = self.gapless else {
            let pcm =
                Pcm::from_f32(rate, layout, pcm).map_err(|_| CodecError::Limit { what: "pcm" })?;
            return Ok(vec![Output::Audio(AudioOutput {
                pcm,
                pts: packet.pts,
                timescale: packet.timescale,
                skip_leading: 0,
                skip_trailing: 0,
                concealed: false,
            })]);
        };
        let channels = layout.len();
        let lead = i64::from(gapless.encoder_delay);
        if self.held.is_empty() {
            self.held_start = first;
        }
        let start = self.held_start;
        let mut combined = std::mem::take(&mut self.held);
        combined.extend_from_slice(&pcm);
        let total = combined.len() / channels;
        let emit = total.saturating_sub(gapless.padding as usize);
        self.held = combined.split_off(emit * channels);
        self.held_start = start + emit as i64;
        if emit == 0 {
            return Ok(Vec::new());
        }
        let skip = (lead - start).clamp(0, emit as i64) as u32;
        Ok(vec![self.output(combined, start - lead, skip, 0)?])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        let (Some(gapless), Some(layout)) = (self.gapless, self.layout.as_ref()) else {
            return Ok(Vec::new());
        };
        let channels = layout.len();
        let pcm = std::mem::take(&mut self.held);
        let total = pcm.len() / channels;
        if total == 0 {
            return Ok(Vec::new());
        }
        let lead = i64::from(gapless.encoder_delay);
        let start = self.held_start;
        let skip = (lead - start).clamp(0, total as i64) as u32;
        let trailing = (total as u32).saturating_sub(skip);
        Ok(vec![self.output(pcm, start - lead, skip, trailing)?])
    }

    fn reset(&mut self) {
        self.states = States::new();
        self.held.clear();
        self.held_start = 0;
        self.clock = 0;
        self.drained = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chanmap_reads_msb_first_and_expands_pairs() {
        let map = 0x8000 | 0x0200 | 0x0001; // Left, Lrs/Rrs, LFE
        assert_eq!(
            chanmap_locations(map),
            vec![Location::L, Location::Lrs, Location::Rrs, Location::Lfe]
        );
    }

    #[test]
    fn acmod_orders_follow_table_5_8() {
        assert_eq!(
            acmod_locations(7, true),
            vec![
                Location::L,
                Location::C,
                Location::R,
                Location::Ls,
                Location::Rs,
                Location::Lfe
            ]
        );
        assert_eq!(acmod_locations(4, false)[2], Location::Cs);
        assert_eq!(acmod_locations(1, false), vec![Location::C]);
    }

    #[test]
    fn locations_beyond_the_speaker_set_have_no_speaker() {
        for location in [Location::Ts, Location::Lw, Location::Vhc, Location::Lfe2] {
            assert_eq!(location.speaker(), None);
        }
        assert_eq!(Location::Lrs.speaker(), Some(Channel::BackLeft));
        assert_eq!(Location::Ls.speaker(), Some(Channel::SideLeft));
    }
}
