//! Sync information and bit stream information for AC-3 (A/52 section 5.3,
//! with the Annex D alternate syntax) and E-AC-3 (Annex E section 2.2).

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::{CMIXLEV, EXT_MIXLEV, FRAME_WORDS, NOMINAL_KBPS, SURMIXLEV};

/// The syncword that opens every syncframe.
pub const SYNCWORD: u16 = 0x0B77;

/// Largest syncframe in bytes: E-AC-3 `frmsiz` is 11 bits of 16-bit words.
pub const MAX_FRAME_BYTES: usize = 4096;

/// Smallest syncframe either syntax can describe: AC-3 `frmsizecod` 0 at
/// 48 kHz, 64 words.
const MIN_AC3_FRAME_BYTES: usize = 128;

/// Bytes [`probe`] reads.
pub const PROBE_BYTES: usize = 6;

/// Which syntax a syncframe uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Syntax {
    /// `bsid` 0 to 8: A/52 section 5, with Annex D when `bsid` is 6.
    Ac3,
    /// `bsid` 11 to 16: Annex E.
    Eac3,
}

/// E-AC-3 `strmtyp` (Annex E Table E2.1). AC-3 frames are `Independent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Type 0: an independent substream.
    Independent,
    /// Type 1: a dependent substream that adds or replaces channels.
    Dependent,
    /// Type 2: an independent substream converted from AC-3.
    Converted,
}

/// What the first six bytes of a syncframe say about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncInfo {
    pub syntax: Syntax,
    pub stream_type: StreamType,
    pub substream_id: u8,
    pub frame_bytes: usize,
    pub sample_rate: u32,
    /// Audio blocks of 256 samples.
    pub blocks: usize,
}

/// Read the syncword, the syntax and the frame size from the start of a
/// syncframe. `bytes` needs at least [`PROBE_BYTES`].
///
/// A container that splits a raw elementary stream calls this to find the
/// next frame. Reserved sample rates and frame size codes are
/// [`CodecError::Malformed`]. AC-3 `bsid` 9 and 10 and the E-AC-3 reduced
/// sample rates are [`CodecError::Unsupported`]: A/52 defines neither their
/// sample rates for AC-3 nor their bit allocation for E-AC-3.
pub fn probe(bytes: &[u8]) -> Result<SyncInfo, CodecError> {
    let Some(head) = bytes.get(..PROBE_BYTES) else {
        return Err(CodecError::NeedMoreData);
    };
    if u16::from_be_bytes([head[0], head[1]]) != SYNCWORD {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "syncword",
        });
    }
    let bsid = head[5] >> 3;
    match bsid {
        0..=8 => {
            let fscod = head[4] >> 6;
            let code = usize::from(head[4] & 0x3f);
            if fscod == 3 {
                return Err(CodecError::Malformed {
                    at_bit: 32,
                    what: "fscod",
                });
            }
            let Some(words) = FRAME_WORDS.get(code) else {
                return Err(CodecError::Malformed {
                    at_bit: 34,
                    what: "frmsizecod",
                });
            };
            Ok(SyncInfo {
                syntax: Syntax::Ac3,
                stream_type: StreamType::Independent,
                substream_id: 0,
                frame_bytes: usize::from(words[usize::from(fscod)]) * 2,
                sample_rate: rate(fscod),
                blocks: 6,
            })
        }
        9 | 10 => Err(CodecError::Unsupported {
            what: "AC-3 bsid 9 or 10",
        }),
        11..=16 => {
            let stream_type = match head[2] >> 6 {
                0 => StreamType::Independent,
                1 => StreamType::Dependent,
                2 => StreamType::Converted,
                _ => {
                    return Err(CodecError::Malformed {
                        at_bit: 16,
                        what: "strmtyp",
                    });
                }
            };
            let substream_id = (head[2] >> 3) & 7;
            let words = (usize::from(head[2] & 7) << 8 | usize::from(head[3])) + 1;
            let fscod = head[4] >> 6;
            if fscod == 3 {
                return Err(CodecError::Unsupported {
                    what: "E-AC-3 reduced sample rate",
                });
            }
            let blocks = [1, 2, 3, 6][usize::from((head[4] >> 4) & 3)];
            Ok(SyncInfo {
                syntax: Syntax::Eac3,
                stream_type,
                substream_id,
                frame_bytes: words * 2,
                sample_rate: rate(fscod),
                blocks,
            })
        }
        _ => Err(CodecError::Unsupported {
            what: "bsid above 16",
        }),
    }
}

fn rate(fscod: u8) -> u32 {
    match fscod {
        0 => 48_000,
        1 => 44_100,
        _ => 32_000,
    }
}

/// Nominal AC-3 bit rate in bits per second for `frmsizecod`.
#[must_use]
pub fn ac3_bit_rate(frmsizecod: u8) -> Option<u32> {
    NOMINAL_KBPS
        .get(usize::from(frmsizecod / 2))
        .map(|kbps| u32::from(*kbps) * 1000)
}

/// Mixing levels a downmix may use, as linear gains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixLevels {
    /// Lt/Rt center mix level (Annex D Table D2.3, Annex E `ltrtcmixlev`).
    pub ltrt_center: f32,
    /// Lt/Rt surround mix level.
    pub ltrt_surround: f32,
    /// Lo/Ro center mix level.
    pub loro_center: f32,
    /// Lo/Ro surround mix level.
    pub loro_surround: f32,
}

/// Program metadata from the bit stream information.
///
/// Levels are what the stream carries. The decoder applies none of them
/// unless its options ask for it: `dialnorm` and `compr` describe the
/// program for a downstream renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub syntax: Syntax,
    pub bsid: u8,
    /// Bit stream mode (Table 5.7): complete main, music and effects, and
    /// the rest. Zero when an E-AC-3 frame carries no informational metadata.
    pub bsmod: u8,
    pub acmod: u8,
    pub lfe: bool,
    /// Dialogue level in dB below full scale, 1 to 31. `dialnorm` 0 reads as 31.
    pub dialnorm: u8,
    /// The second program's dialogue level in 1+1 mode.
    pub dialnorm2: Option<u8>,
    /// Heavy compression word (`compr`), when present, in dB.
    pub compr_db: Option<f32>,
    /// The second program's `compr2` in 1+1 mode.
    pub compr2_db: Option<f32>,
    /// Dolby Surround mode in 2/0 (Table 5.11): 0 not indicated, 1 not
    /// encoded, 2 encoded.
    pub dsurmod: u8,
    /// AC-3 `cmixlev`, the center level a Lo/Ro downmix uses.
    pub center_mix: Option<f32>,
    /// AC-3 `surmixlev`, the surround level a Lo/Ro downmix uses.
    pub surround_mix: Option<f32>,
    /// Preferred stereo downmix (`dmixmod`): 1 Lt/Rt, 2 Lo/Ro, 0 or 3 none.
    pub dmixmod: Option<u8>,
    /// Annex D or Annex E extended mix levels.
    pub mix_levels: Option<MixLevels>,
    /// LFE level for a downmix in dB (`lfemixlevcod`, 10 minus the code).
    pub lfe_mix_db: Option<i8>,
    /// Surround EX mode (`dsurexmod`).
    pub dsurexmod: Option<u8>,
    /// Headphone mode (`dheadphonmod`).
    pub dheadphonmod: Option<u8>,
    /// Mixing room peak level in dB SPL (`mixlevel` + 80) and room type.
    pub mixing_room: Option<(u8, u8)>,
    /// A/D converter type (Annex D `adconvtyp`).
    pub adconvtyp: Option<u8>,
    pub copyright: bool,
    pub original: bool,
    /// Program scale factor in dB (`pgmscl` - 51), E-AC-3 independent only.
    pub program_scale_db: Option<i8>,
    /// External program scale factor in dB (`extpgmscl` - 51).
    pub ext_program_scale_db: Option<i8>,
    /// Dependent substream custom channel map (Table E2.5), MSB is Left.
    pub chanmap: Option<u16>,
}

impl Metadata {
    fn new(syntax: Syntax, bsid: u8, acmod: u8, lfe: bool) -> Self {
        Self {
            syntax,
            bsid,
            bsmod: 0,
            acmod,
            lfe,
            dialnorm: 31,
            dialnorm2: None,
            compr_db: None,
            compr2_db: None,
            dsurmod: 0,
            center_mix: None,
            surround_mix: None,
            dmixmod: None,
            mix_levels: None,
            lfe_mix_db: None,
            dsurexmod: None,
            dheadphonmod: None,
            mixing_room: None,
            adconvtyp: None,
            copyright: false,
            original: false,
            program_scale_db: None,
            ext_program_scale_db: None,
            chanmap: None,
        }
    }
}

/// `compr` in dB.
#[must_use]
pub fn compr_db(code: u8) -> f32 {
    20.0 * compr_gain(code).log10()
}

/// `dynrng` as a linear gain (section 7.7.1.1): a signed 3-bit exponent of
/// 6.02 dB steps and the mantissa 0.5 + m/64 above it.
#[must_use]
pub fn dynrng_gain(code: u8) -> f32 {
    let exponent = i32::from(code >> 5) - if code & 0x80 != 0 { 8 } else { 0 };
    let mantissa = 32.0 + f32::from(code & 0x1f);
    mantissa / 64.0 * ((exponent + 1) as f32).exp2()
}

/// `compr` as a linear gain (section 7.7.2.1): a signed 4-bit exponent of
/// 6.02 dB steps and the mantissa 0.5 + m/32 above it.
#[must_use]
pub fn compr_gain(code: u8) -> f32 {
    let exponent = i32::from(code >> 4) - if code & 0x80 != 0 { 16 } else { 0 };
    let mantissa = 16.0 + f32::from(code & 0x0f);
    mantissa / 32.0 * ((exponent + 1) as f32).exp2()
}

/// One syncframe's bit stream information.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bsi {
    pub syntax: Syntax,
    pub stream_type: StreamType,
    pub substream_id: u8,
    pub frame_bytes: usize,
    pub fscod: u8,
    pub sample_rate: u32,
    pub blocks: usize,
    pub acmod: u8,
    pub lfeon: bool,
    /// Full bandwidth channels.
    pub nfchans: usize,
    /// `compre`/`compr` for each program of 1+1 mode.
    pub compr: [Option<u8>; 2],
    pub dialnorm: [u8; 2],
    pub chanmap: Option<u16>,
    pub metadata: Metadata,
    /// Bits from the start of the frame to the end of `bsi`.
    pub end_bit: usize,
}

/// Channels of each `acmod` (Table 5.8).
pub(crate) const NFCHANS: [usize; 8] = [2, 1, 2, 3, 3, 4, 4, 5];

fn mix_level(code: u32) -> f32 {
    EXT_MIXLEV[(code & 7) as usize]
}

/// Surround mix level codes 0 to 2 are reserved and read as code 3
/// (Annex D Tables D2.4 and D2.6).
fn surround_level(code: u32) -> f32 {
    mix_level(code.max(3))
}

/// Parse `syncinfo` and `bsi` from a whole syncframe.
pub(crate) fn parse(frame: &[u8]) -> Result<Bsi, CodecError> {
    let info = probe(frame)?;
    if frame.len() < info.frame_bytes {
        return Err(CodecError::Truncated {
            at_bit: frame.len() as u64 * 8,
            what: "syncframe",
        });
    }
    let frame = &frame[..info.frame_bytes];
    match info.syntax {
        Syntax::Ac3 => parse_ac3(frame, info),
        Syntax::Eac3 => parse_eac3(frame, info),
    }
}

fn parse_ac3(frame: &[u8], info: SyncInfo) -> Result<Bsi, CodecError> {
    let mut r = BitReader::new(frame);
    r.skip(16 + 16, "syncinfo")?;
    let fscod = r.read(2, "fscod")? as u8;
    r.skip(6, "frmsizecod")?;
    let bsid = r.read(5, "bsid")? as u8;
    let bsmod = r.read(3, "bsmod")? as u8;
    let acmod = r.read(3, "acmod")? as u8;
    let mut metadata = Metadata::new(Syntax::Ac3, bsid, acmod, false);
    metadata.bsmod = bsmod;
    if acmod & 1 != 0 && acmod != 1 {
        metadata.center_mix = Some(CMIXLEV[r.read(2, "cmixlev")? as usize]);
    }
    if acmod & 4 != 0 {
        metadata.surround_mix = Some(SURMIXLEV[r.read(2, "surmixlev")? as usize]);
    }
    if acmod == 2 {
        metadata.dsurmod = r.read(2, "dsurmod")? as u8;
    }
    let lfeon = r.bit("lfeon")?;
    metadata.lfe = lfeon;
    let mut dialnorm = [31u8; 2];
    let mut compr = [None; 2];
    let programs = if acmod == 0 { 2 } else { 1 };
    for program in 0..programs {
        dialnorm[program] = dialnorm_of(r.read(5, "dialnorm")?);
        if r.bit("compre")? {
            compr[program] = Some(r.read(8, "compr")? as u8);
        }
        if r.bit("langcode")? {
            r.skip(8, "langcod")?;
        }
        if r.bit("audprodie")? {
            let level = r.read(5, "mixlevel")? as u8;
            let room = r.read(2, "roomtyp")? as u8;
            if program == 0 {
                metadata.mixing_room = Some((level + 80, room));
            }
        }
    }
    metadata.copyright = r.bit("copyrightb")?;
    metadata.original = r.bit("origbs")?;
    if bsid == 6 {
        if r.bit("xbsi1e")? {
            metadata.dmixmod = Some(r.read(2, "dmixmod")? as u8);
            let ltrt_center = mix_level(r.read(3, "ltrtcmixlev")?);
            let ltrt_surround = surround_level(r.read(3, "ltrtsurmixlev")?);
            let loro_center = mix_level(r.read(3, "lorocmixlev")?);
            let loro_surround = surround_level(r.read(3, "lorosurmixlev")?);
            metadata.mix_levels = Some(MixLevels {
                ltrt_center,
                ltrt_surround,
                loro_center,
                loro_surround,
            });
        }
        if r.bit("xbsi2e")? {
            metadata.dsurexmod = Some(r.read(2, "dsurexmod")? as u8);
            metadata.dheadphonmod = Some(r.read(2, "dheadphonmod")? as u8);
            metadata.adconvtyp = Some(r.read(1, "adconvtyp")? as u8);
            r.skip(8 + 1, "xbsi2 and encinfo")?;
        }
    } else {
        if r.bit("timecod1e")? {
            r.skip(14, "timecod1")?;
        }
        if r.bit("timecod2e")? {
            r.skip(14, "timecod2")?;
        }
    }
    if r.bit("addbsie")? {
        let len = r.read(6, "addbsil")? as usize + 1;
        r.skip(len * 8, "addbsi")?;
    }
    metadata.dialnorm = dialnorm[0];
    metadata.dialnorm2 = (acmod == 0).then_some(dialnorm[1]);
    metadata.compr_db = compr[0].map(compr_db);
    metadata.compr2_db = compr[1].map(compr_db);
    Ok(Bsi {
        syntax: Syntax::Ac3,
        stream_type: StreamType::Independent,
        substream_id: 0,
        frame_bytes: info.frame_bytes,
        fscod,
        sample_rate: info.sample_rate,
        blocks: 6,
        acmod,
        lfeon,
        nfchans: NFCHANS[usize::from(acmod)],
        compr,
        dialnorm,
        chanmap: None,
        metadata,
        end_bit: r.pos(),
    })
}

fn dialnorm_of(code: u32) -> u8 {
    if code == 0 { 31 } else { code as u8 }
}

fn parse_eac3(frame: &[u8], info: SyncInfo) -> Result<Bsi, CodecError> {
    let mut r = BitReader::new(frame);
    r.skip(16, "syncword")?;
    r.skip(2 + 3 + 11, "strmtyp, substreamid, frmsiz")?;
    let fscod = r.read(2, "fscod")? as u8;
    let numblkscod = r.read(2, "numblkscod")?;
    let acmod = r.read(3, "acmod")? as u8;
    let lfeon = r.bit("lfeon")?;
    let bsid = r.read(5, "bsid")? as u8;
    let mut metadata = Metadata::new(Syntax::Eac3, bsid, acmod, lfeon);
    let mut dialnorm = [31u8; 2];
    let mut compr = [None; 2];
    let programs = if acmod == 0 { 2 } else { 1 };
    for program in 0..programs {
        dialnorm[program] = dialnorm_of(r.read(5, "dialnorm")?);
        if r.bit("compre")? {
            compr[program] = Some(r.read(8, "compr")? as u8);
        }
    }
    let mut chanmap = None;
    if info.stream_type == StreamType::Dependent && r.bit("chanmape")? {
        chanmap = Some(r.read(16, "chanmap")? as u16);
    }
    if r.bit("mixmdate")? {
        if acmod > 2 {
            metadata.dmixmod = Some(r.read(2, "dmixmod")? as u8);
        }
        let mut levels = MixLevels {
            ltrt_center: mix_level(4),
            ltrt_surround: mix_level(4),
            loro_center: mix_level(4),
            loro_surround: mix_level(4),
        };
        let mut any = false;
        if acmod & 1 != 0 && acmod > 2 {
            levels.ltrt_center = mix_level(r.read(3, "ltrtcmixlev")?);
            levels.loro_center = mix_level(r.read(3, "lorocmixlev")?);
            any = true;
        }
        if acmod & 4 != 0 {
            levels.ltrt_surround = surround_level(r.read(3, "ltrtsurmixlev")?);
            levels.loro_surround = surround_level(r.read(3, "lorosurmixlev")?);
            any = true;
        }
        if any {
            metadata.mix_levels = Some(levels);
        }
        if lfeon && r.bit("lfemixlevcode")? {
            metadata.lfe_mix_db = Some(10 - r.read(5, "lfemixlevcod")? as i8);
        }
        if info.stream_type == StreamType::Independent {
            if r.bit("pgmscle")? {
                metadata.program_scale_db = Some(r.read(6, "pgmscl")? as i8 - 51);
            }
            if acmod == 0 && r.bit("pgmscl2e")? {
                r.skip(6, "pgmscl2")?;
            }
            if r.bit("extpgmscle")? {
                metadata.ext_program_scale_db = Some(r.read(6, "extpgmscl")? as i8 - 51);
            }
            match r.read(2, "mixdef")? {
                1 => r.skip(1 + 1 + 3, "mixdef 1")?,
                2 => r.skip(12, "mixdata")?,
                3 => {
                    let len = r.read(5, "mixdeflen")? as usize;
                    r.skip(8 * (len + 2), "mixdata")?;
                }
                _ => {}
            }
            if acmod < 2 {
                if r.bit("paninfoe")? {
                    r.skip(8 + 6, "paninfo")?;
                }
                if acmod == 0 && r.bit("paninfo2e")? {
                    r.skip(8 + 6, "paninfo2")?;
                }
            }
            if r.bit("frmmixcfginfoe")? {
                if numblkscod == 0 {
                    r.skip(5, "blkmixcfginfo")?;
                } else {
                    for _ in 0..info.blocks {
                        if r.bit("blkmixcfginfoe")? {
                            r.skip(5, "blkmixcfginfo")?;
                        }
                    }
                }
            }
        }
    }
    if r.bit("infomdate")? {
        metadata.bsmod = r.read(3, "bsmod")? as u8;
        metadata.copyright = r.bit("copyrightb")?;
        metadata.original = r.bit("origbs")?;
        if acmod == 2 {
            metadata.dsurmod = r.read(2, "dsurmod")? as u8;
            metadata.dheadphonmod = Some(r.read(2, "dheadphonmod")? as u8);
        }
        if acmod >= 6 {
            metadata.dsurexmod = Some(r.read(2, "dsurexmod")? as u8);
        }
        for program in 0..programs {
            if r.bit("audprodie")? {
                let level = r.read(5, "mixlevel")? as u8;
                let room = r.read(2, "roomtyp")? as u8;
                let converter = r.read(1, "adconvtyp")? as u8;
                if program == 0 {
                    metadata.mixing_room = Some((level + 80, room));
                    metadata.adconvtyp = Some(converter);
                }
            }
        }
        if fscod < 3 {
            r.skip(1, "sourcefscod")?;
        }
    }
    if info.stream_type == StreamType::Independent && numblkscod != 3 {
        r.skip(1, "convsync")?;
    }
    if info.stream_type == StreamType::Converted {
        let blkid = numblkscod == 3 || r.bit("blkid")?;
        if blkid {
            r.skip(6, "frmsizecod")?;
        }
    }
    if r.bit("addbsie")? {
        let len = r.read(6, "addbsil")? as usize + 1;
        r.skip(len * 8, "addbsi")?;
    }
    metadata.dialnorm = dialnorm[0];
    metadata.dialnorm2 = (acmod == 0).then_some(dialnorm[1]);
    metadata.compr_db = compr[0].map(compr_db);
    metadata.compr2_db = compr[1].map(compr_db);
    metadata.chanmap = chanmap;
    let nfchans = NFCHANS[usize::from(acmod)];
    if let Some(map) = chanmap {
        let mut locations = 0usize;
        for bit in 0..16 {
            if map & (0x8000 >> bit) != 0 {
                locations += if matches!(bit, 5 | 6 | 9 | 10 | 11 | 13) {
                    2
                } else {
                    1
                };
            }
        }
        if locations != nfchans + usize::from(lfeon) {
            return Err(CodecError::Malformed {
                at_bit: 40,
                what: "chanmap channel count",
            });
        }
    }
    Ok(Bsi {
        syntax: Syntax::Eac3,
        stream_type: info.stream_type,
        substream_id: info.substream_id,
        frame_bytes: info.frame_bytes,
        fscod,
        sample_rate: info.sample_rate,
        blocks: info.blocks,
        acmod,
        lfeon,
        nfchans,
        compr,
        dialnorm,
        chanmap,
        metadata,
        end_bit: r.pos(),
    })
}

/// The `dac3` payload ETSI TS 102 366 Annex F.4 stores in an MP4 sample
/// entry: `fscod`, `bsid`, `bsmod`, `acmod`, `lfeon`, `bit_rate_code`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dac3 {
    pub fscod: u8,
    pub bsid: u8,
    pub bsmod: u8,
    pub acmod: u8,
    pub lfeon: bool,
    pub bit_rate_code: u8,
}

impl Dac3 {
    /// The three payload bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 3] {
        let bits: u32 = u32::from(self.fscod & 3) << 22
            | u32::from(self.bsid & 0x1f) << 17
            | u32::from(self.bsmod & 7) << 14
            | u32::from(self.acmod & 7) << 11
            | u32::from(self.lfeon) << 10
            | u32::from(self.bit_rate_code & 0x1f) << 5;
        [(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }

    /// Read the payload. Trailing bytes are ignored.
    pub fn parse(bytes: &[u8]) -> Result<Self, CodecError> {
        let Some(head) = bytes.get(..3) else {
            return Err(CodecError::Truncated {
                at_bit: bytes.len() as u64 * 8,
                what: "dac3",
            });
        };
        let bits = u32::from(head[0]) << 16 | u32::from(head[1]) << 8 | u32::from(head[2]);
        let dac3 = Self {
            fscod: (bits >> 22) as u8 & 3,
            bsid: (bits >> 17) as u8 & 0x1f,
            bsmod: (bits >> 14) as u8 & 7,
            acmod: (bits >> 11) as u8 & 7,
            lfeon: bits >> 10 & 1 != 0,
            bit_rate_code: (bits >> 5) as u8 & 0x1f,
        };
        if dac3.fscod == 3 || dac3.bit_rate_code > 18 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "dac3",
            });
        }
        Ok(dac3)
    }
}

/// One independent substream entry of a `dec3` box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dec3Substream {
    pub fscod: u8,
    pub bsid: u8,
    pub asvc: bool,
    pub bsmod: u8,
    pub acmod: u8,
    pub lfeon: bool,
    pub num_dep_sub: u8,
    /// `chan_loc` when `num_dep_sub` is not zero: the channel locations
    /// the dependent substreams add, bit 8 Lc/Rc down to bit 0 LFE2.
    pub chan_loc: u16,
}

/// The `dec3` payload ETSI TS 102 366 Annex F.6 stores in an MP4 sample
/// entry. The Atmos JOC extension after the substreams is not written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dec3 {
    /// Kilobits per second.
    pub data_rate: u16,
    pub substreams: Vec<Dec3Substream>,
}

impl Dec3 {
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = crate::bits::BitWriter::with_capacity(2 + 3 * self.substreams.len());
        w.put(13, u32::from(self.data_rate.min(0x1fff)));
        w.put(3, self.substreams.len().saturating_sub(1).min(7) as u32);
        for sub in self.substreams.iter().take(8) {
            w.put(2, u32::from(sub.fscod));
            w.put(5, u32::from(sub.bsid));
            w.put(1, 0);
            w.flag(sub.asvc);
            w.put(3, u32::from(sub.bsmod));
            w.put(3, u32::from(sub.acmod));
            w.flag(sub.lfeon);
            w.put(3, 0);
            w.put(4, u32::from(sub.num_dep_sub));
            if sub.num_dep_sub > 0 {
                w.put(9, u32::from(sub.chan_loc & 0x1ff));
            } else {
                w.put(1, 0);
            }
        }
        w.finish()
    }

    /// Read the payload. Bytes after the last substream are ignored.
    pub fn parse(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut r = BitReader::new(bytes);
        let data_rate = r.read(13, "data_rate")? as u16;
        let count = r.read(3, "num_ind_sub")? as usize + 1;
        let mut substreams = Vec::with_capacity(count);
        for _ in 0..count {
            let fscod = r.read(2, "fscod")? as u8;
            let bsid = r.read(5, "bsid")? as u8;
            r.skip(1, "reserved")?;
            let asvc = r.bit("asvc")?;
            let bsmod = r.read(3, "bsmod")? as u8;
            let acmod = r.read(3, "acmod")? as u8;
            let lfeon = r.bit("lfeon")?;
            r.skip(3, "reserved")?;
            let num_dep_sub = r.read(4, "num_dep_sub")? as u8;
            let chan_loc = if num_dep_sub > 0 {
                r.read(9, "chan_loc")? as u16
            } else {
                r.skip(1, "reserved")?;
                0
            };
            substreams.push(Dec3Substream {
                fscod,
                bsid,
                asvc,
                bsmod,
                acmod,
                lfeon,
                num_dep_sub,
                chan_loc,
            });
        }
        Ok(Self {
            data_rate,
            substreams,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    fn ac3_head(fscod: u8, frmsizecod: u8, bsid: u8) -> Vec<u8> {
        vec![0x0B, 0x77, 0, 0, fscod << 6 | frmsizecod, bsid << 3]
    }

    #[test]
    fn an_ac3_head_names_its_frame_size() {
        let info = probe(&ac3_head(0, 30, 8)).unwrap();
        assert_eq!(info.syntax, Syntax::Ac3);
        assert_eq!(info.frame_bytes, 1792);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.blocks, 6);
        let odd = probe(&ac3_head(1, 31, 8)).unwrap();
        assert_eq!(odd.frame_bytes, 976 * 2);
        assert_eq!(odd.sample_rate, 44_100);
    }

    #[test]
    fn reserved_codes_and_short_input_are_refused() {
        assert_eq!(probe(&[0x0B, 0x77, 0]), Err(CodecError::NeedMoreData));
        assert!(matches!(
            probe(&ac3_head(3, 0, 8)),
            Err(CodecError::Malformed { what: "fscod", .. })
        ));
        assert!(matches!(
            probe(&ac3_head(0, 38, 8)),
            Err(CodecError::Malformed {
                what: "frmsizecod",
                ..
            })
        ));
        assert!(matches!(
            probe(&ac3_head(0, 0, 9)),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            probe(&ac3_head(0, 0, 17)),
            Err(CodecError::Unsupported { .. })
        ));
        let mut bad = ac3_head(0, 0, 8);
        bad[1] = 0x78;
        assert!(matches!(
            probe(&bad),
            Err(CodecError::Malformed {
                what: "syncword",
                ..
            })
        ));
    }

    #[test]
    fn an_eac3_head_names_its_substream_and_blocks() {
        let mut w = BitWriter::with_capacity(6);
        w.put(16, 0x0B77);
        w.put(2, 1);
        w.put(3, 2);
        w.put(11, 383);
        w.put(2, 1);
        w.put(2, 2);
        w.put(3, 7);
        w.put(1, 1);
        w.put(5, 16);
        w.put(3, 0);
        let info = probe(&w.finish()).unwrap();
        assert_eq!(info.syntax, Syntax::Eac3);
        assert_eq!(info.stream_type, StreamType::Dependent);
        assert_eq!(info.substream_id, 2);
        assert_eq!(info.frame_bytes, 768);
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.blocks, 3);
    }

    #[test]
    fn the_eac3_reduced_rates_are_unsupported() {
        let head = [0x0B, 0x77, 0x00, 0x10, 0xC0 | 0x0E, 16 << 3];
        assert!(matches!(probe(&head), Err(CodecError::Unsupported { .. })));
    }

    #[test]
    fn dac3_round_trips_its_fields() {
        let dac3 = Dac3 {
            fscod: 1,
            bsid: 8,
            bsmod: 2,
            acmod: 7,
            lfeon: true,
            bit_rate_code: 15,
        };
        let bytes = dac3.to_bytes();
        assert_eq!(Dac3::parse(&bytes).unwrap(), dac3);
        assert!(Dac3::parse(&bytes[..2]).is_err());
        assert!(Dac3::parse(&[0xC0, 0, 0]).is_err());
    }

    #[test]
    fn dec3_round_trips_with_and_without_dependents() {
        let dec3 = Dec3 {
            data_rate: 1024,
            substreams: vec![
                Dec3Substream {
                    fscod: 0,
                    bsid: 16,
                    asvc: false,
                    bsmod: 0,
                    acmod: 7,
                    lfeon: true,
                    num_dep_sub: 1,
                    chan_loc: 0x002,
                },
                Dec3Substream {
                    fscod: 0,
                    bsid: 16,
                    asvc: true,
                    bsmod: 2,
                    acmod: 2,
                    lfeon: false,
                    num_dep_sub: 0,
                    chan_loc: 0,
                },
            ],
        };
        let bytes = dec3.to_bytes();
        assert_eq!(bytes.len(), (16 + 32 + 24) / 8);
        assert_eq!(Dec3::parse(&bytes).unwrap(), dec3);
        assert!(Dec3::parse(&bytes[..4]).is_err());
    }

    #[test]
    fn compression_words_read_as_the_spec_tables() {
        assert!((dynrng_gain(0) - 1.0).abs() < 1e-6);
        assert!((dynrng_gain(0x1f) - 63.0 / 32.0).abs() < 1e-6);
        assert!((dynrng_gain(0x80) - 1.0 / 16.0).abs() < 1e-6);
        assert!((compr_gain(0) - 1.0).abs() < 1e-6);
        assert!((compr_gain(0xF0) - 0.5).abs() < 1e-9);
        assert!((compr_gain(0x80) - 1.0 / 256.0).abs() < 1e-9);
        assert!(compr_db(0).abs() < 1e-4);
        assert!((compr_db(0xF0) + 6.0206).abs() < 1e-3);
    }
}
