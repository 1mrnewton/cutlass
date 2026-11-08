//! One syncframe's audio: the AC-3 audio blocks (A/52 sections 5.4.3 and 7)
//! and the E-AC-3 audio frame and blocks (Annex E clauses 2 and 3), from
//! the bits after `bsi` to PCM.
//!
//! Decoding runs on copies of the substream's overlap and dither state and
//! commits them only when the whole frame decodes, so a refused frame
//! leaves the substream as it was.

use cutlass_decode::CodecError;

use crate::aht::{self, Premantissas};
use crate::bitalloc::{self, Delta, MAX_SEGMENTS, Params, Segment};
use crate::bits::BitReader;
use crate::crc;
use crate::exponents;
use crate::header::{Bsi, StreamType, Syntax, compr_gain, dynrng_gain};
use crate::mdct::{COEFS, SPAN, Transform, overlap_add};
use crate::spx::{self, Bands};
use crate::tables::{
    BAP_BITS, BAP_LEVELS, D15, DEFAULT_CPL_BANDS, DEFAULT_SPX_BANDS, FRAME_EXPSTR, REUSE,
};

/// Full bandwidth channels in one substream.
pub(crate) const MAX_FBW: usize = 5;
/// Coded channels in one substream: the full bandwidth channels and LFE.
pub(crate) const MAX_CODED: usize = MAX_FBW + 1;

const CPL: usize = MAX_FBW;
const LFE: usize = MAX_FBW + 1;
const SLOTS: usize = MAX_FBW + 2;
const MAX_BLOCKS: usize = 6;

/// How the decoder applies the stream's dynamic range words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DynamicRange {
    /// Full range: no `dynrng` or `compr` gain.
    Off,
    /// Line mode: `dynrng`, its cuts raised to `cut` and its boosts to
    /// `boost` (1.0 applies the word as sent, 0.0 ignores it).
    Line { cut: f32, boost: f32 },
    /// RF mode: `compr` where the frame carries it, `dynrng` otherwise.
    Heavy,
}

/// Transient pre-noise processing data for one channel (Annex E 3.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transient {
    /// `transprocloc`: the transient, in units of four samples from the
    /// frame's first sample.
    pub location: u16,
    /// `transproclen`: the time scaling length in samples.
    pub length: u8,
}

/// How often each coding tool appeared in the decoded syncframes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tools {
    /// Syncframes decoded, every substream counted.
    pub frames: u64,
    /// Of those, frames of a dependent substream.
    pub dependent_frames: u64,
    /// E-AC-3 frames with one, two or three blocks.
    pub short_frames: u64,
    /// Audio blocks decoded.
    pub blocks: u64,
    /// Channel blocks that used the two 256-sample transforms.
    pub short_blocks: u64,
    /// Blocks with coupling in use.
    pub coupling_blocks: u64,
    /// Of those, blocks with phase flags in use.
    pub phase_blocks: u64,
    /// Blocks that rematrixed at least one band.
    pub rematrix_blocks: u64,
    /// Blocks with spectral extension in use.
    pub spx_blocks: u64,
    /// Channel frames coded with the adaptive hybrid transform, coupling
    /// and LFE included.
    pub aht_channels: u64,
    /// Channel frames that carried transient pre-noise data.
    pub transient_channels: u64,
    /// Blocks with delta bit allocation in use on any channel.
    pub delta_blocks: u64,
}

impl Tools {
    pub(crate) fn add(&mut self, other: &Self) {
        self.frames += other.frames;
        self.dependent_frames += other.dependent_frames;
        self.short_frames += other.short_frames;
        self.blocks += other.blocks;
        self.short_blocks += other.short_blocks;
        self.coupling_blocks += other.coupling_blocks;
        self.phase_blocks += other.phase_blocks;
        self.rematrix_blocks += other.rematrix_blocks;
        self.spx_blocks += other.spx_blocks;
        self.aht_channels += other.aht_channels;
        self.transient_channels += other.transient_channels;
        self.delta_blocks += other.delta_blocks;
    }
}

/// The dynamic range words of one syncframe: `compr` per program and
/// `dynrng` per block and program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Drc {
    pub compr: [Option<u8>; 2],
    pub dynrng: [[Option<u8>; 2]; MAX_BLOCKS],
}

/// The substream state that outlives one syncframe.
#[derive(Clone, Debug)]
pub(crate) struct Substream {
    delay: [[f32; COEFS]; MAX_CODED],
    seed: u32,
}

impl Substream {
    pub(crate) fn new() -> Self {
        Self {
            delay: [[0.0; COEFS]; MAX_CODED],
            seed: 0x1234_5678,
        }
    }
}

/// One decoded syncframe.
#[derive(Clone, Debug)]
pub(crate) struct Decoded {
    /// Coded channels in bit stream order: full bandwidth, then LFE.
    pub channels: usize,
    /// Samples per channel: blocks times 256.
    pub samples: usize,
    /// Planar PCM, `channels` runs of `samples`.
    pub pcm: Vec<f32>,
    pub transients: [Option<Transient>; MAX_FBW],
    pub tools: Tools,
    /// The words this frame carried.
    pub drc: Drc,
}

impl Decoded {
    pub(crate) fn channel(&self, index: usize) -> &[f32] {
        let start = index * self.samples;
        self.pcm.get(start..start + self.samples).unwrap_or(&[])
    }
}

fn malformed(r: &BitReader<'_>, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: r.pos() as u64,
        what,
    }
}

/// Check the frame's CRC words: `crc1` over the first 5/8 and `crc2` over
/// the rest for AC-3, the one CRC over the whole frame for E-AC-3.
pub(crate) fn check_crc(frame: &[u8], syntax: Syntax) -> Result<(), CodecError> {
    let body = frame.get(2..).unwrap_or(&[]);
    if syntax == Syntax::Ac3 {
        let words = frame.len() / 2;
        let five_eighths = ((words >> 1) + (words >> 3)) * 2;
        let first = frame.get(2..five_eighths).unwrap_or(&[]);
        if crc::update(0, first) != 0 {
            return Err(CodecError::Malformed {
                at_bit: 16,
                what: "crc1",
            });
        }
    }
    if crc::update(0, body) != 0 {
        return Err(CodecError::Malformed {
            at_bit: 16,
            what: if syntax == Syntax::Ac3 { "crc2" } else { "crc" },
        });
    }
    Ok(())
}

/// Linear-congruential noise for dither and spectral extension.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in about +-0.707 (section 7.3.4).
    fn dither(&mut self) -> f32 {
        (self.next() - 0.5) * std::f32::consts::SQRT_2
    }

    /// Zero mean, unit variance.
    fn unit(&mut self) -> f32 {
        (self.next() * 2.0 - 1.0) * 1.732_050_8
    }
}

/// `2^-exp` for exponents 0 to 24.
fn scale(exp: u8) -> f32 {
    (-f32::from(exp)).exp2()
}

/// Grouped mantissas of `bap` 1, 2 and 4 share their groups across all
/// channels of one block (section 7.3.5).
#[derive(Default)]
struct Groups {
    one: [f32; 3],
    one_left: usize,
    two: [f32; 3],
    two_left: usize,
    four: [f32; 2],
    four_left: usize,
}

fn level(code: u32, levels: u32) -> f32 {
    (2.0 * code as f32 - (levels - 1) as f32) / levels as f32
}

impl Groups {
    fn mantissa(&mut self, r: &mut BitReader<'_>, bap: u8) -> Result<f32, CodecError> {
        Ok(match bap {
            1 => {
                if self.one_left == 0 {
                    let code = r.read(5, "bap 1 group")?;
                    if code > 26 {
                        return Err(malformed(r, "bap 1 group"));
                    }
                    self.one = [code / 9, (code % 9) / 3, code % 3].map(|m| level(m, 3));
                    self.one_left = 3;
                }
                self.one_left -= 1;
                self.one[2 - self.one_left]
            }
            2 => {
                if self.two_left == 0 {
                    let code = r.read(7, "bap 2 group")?;
                    if code > 124 {
                        return Err(malformed(r, "bap 2 group"));
                    }
                    self.two = [code / 25, (code % 25) / 5, code % 5].map(|m| level(m, 5));
                    self.two_left = 3;
                }
                self.two_left -= 1;
                self.two[2 - self.two_left]
            }
            4 => {
                if self.four_left == 0 {
                    let code = r.read(7, "bap 4 group")?;
                    if code > 120 {
                        return Err(malformed(r, "bap 4 group"));
                    }
                    self.four = [code / 11, code % 11].map(|m| level(m, 11));
                    self.four_left = 2;
                }
                self.four_left -= 1;
                self.four[1 - self.four_left]
            }
            3 | 5 => {
                let levels = BAP_LEVELS[usize::from(bap)];
                let code = r.read(BAP_BITS[usize::from(bap)], "mantissa")?;
                if code >= levels {
                    return Err(malformed(r, "mantissa"));
                }
                level(code, levels)
            }
            6..=15 => {
                let bits = BAP_BITS[usize::from(bap)];
                r.signed(bits, "mantissa")? as f32 / (1u32 << (bits - 1)) as f32
            }
            _ => return Err(malformed(r, "bit allocation pointer")),
        })
    }
}

/// One coded channel's exponents and allocation, kept across blocks.
#[derive(Clone)]
struct Slot {
    exps: [u8; 256],
    bap: [u8; 256],
    start: usize,
    end: usize,
    have_exps: bool,
    fsnr: u32,
    fgaincod: u32,
    delta: Delta,
    aht: bool,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            exps: [0; 256],
            bap: [0; 256],
            start: 0,
            end: 0,
            have_exps: false,
            fsnr: 0,
            fgaincod: 4,
            delta: Delta::default(),
            aht: false,
        }
    }
}

/// The E-AC-3 `audfrm` fields, or their AC-3 equivalents per block.
struct FrameInfo {
    snroffststr: u32,
    blkswe: bool,
    dithflage: bool,
    bamode: bool,
    frmfgaincode: bool,
    dbaflde: bool,
    skipflde: bool,
    cplstre: [bool; MAX_BLOCKS],
    cplinu: [bool; MAX_BLOCKS],
    cplexpstr: [u8; MAX_BLOCKS],
    chexpstr: [[u8; MAX_FBW]; MAX_BLOCKS],
    lfeexpstr: [u8; MAX_BLOCKS],
    aht: [bool; SLOTS],
    frm_csnr: u32,
    frm_fsnr: u32,
    transients: [Option<Transient>; MAX_FBW],
    spxatten: [Option<u8>; MAX_FBW],
}

impl FrameInfo {
    fn ac3() -> Self {
        Self {
            snroffststr: 0,
            blkswe: true,
            dithflage: true,
            bamode: true,
            frmfgaincode: true,
            dbaflde: true,
            skipflde: true,
            cplstre: [false; MAX_BLOCKS],
            cplinu: [false; MAX_BLOCKS],
            cplexpstr: [REUSE; MAX_BLOCKS],
            chexpstr: [[REUSE; MAX_FBW]; MAX_BLOCKS],
            lfeexpstr: [REUSE; MAX_BLOCKS],
            aht: [false; SLOTS],
            frm_csnr: 0,
            frm_fsnr: 0,
            transients: [None; MAX_FBW],
            spxatten: [None; MAX_FBW],
        }
    }
}

/// Annex E Table E1.3.
fn read_audfrm(r: &mut BitReader<'_>, bsi: &Bsi) -> Result<FrameInfo, CodecError> {
    let blocks = bsi.blocks;
    let nf = bsi.nfchans;
    let mut fi = FrameInfo::ac3();
    let (expstre, ahte) = if blocks == 6 {
        (r.bit("expstre")?, r.bit("ahte")?)
    } else {
        (true, false)
    };
    fi.snroffststr = r.read(2, "snroffststr")?;
    let transproce = r.bit("transproce")?;
    fi.blkswe = r.bit("blkswe")?;
    fi.dithflage = r.bit("dithflage")?;
    fi.bamode = r.bit("bamode")?;
    fi.frmfgaincode = r.bit("frmfgaincode")?;
    fi.dbaflde = r.bit("dbaflde")?;
    fi.skipflde = r.bit("skipflde")?;
    let spxattene = r.bit("spxattene")?;
    if bsi.acmod > 1 {
        fi.cplstre[0] = true;
        fi.cplinu[0] = r.bit("cplinu")?;
        for blk in 1..blocks {
            fi.cplstre[blk] = r.bit("cplstre")?;
            fi.cplinu[blk] = if fi.cplstre[blk] {
                r.bit("cplinu")?
            } else {
                fi.cplinu[blk - 1]
            };
        }
    }
    let ncplblks = fi.cplinu[..blocks].iter().filter(|&&u| u).count();
    if expstre {
        for blk in 0..blocks {
            if fi.cplinu[blk] {
                fi.cplexpstr[blk] = r.read(2, "cplexpstr")? as u8;
            }
            for ch in 0..nf {
                fi.chexpstr[blk][ch] = r.read(2, "chexpstr")? as u8;
            }
        }
    } else {
        if bsi.acmod > 1 && ncplblks > 0 {
            let code = r.read(5, "frmcplexpstr")? as usize;
            for (slot, &value) in fi
                .cplexpstr
                .iter_mut()
                .zip(&FRAME_EXPSTR[code])
                .take(blocks)
            {
                *slot = value;
            }
        }
        for ch in 0..nf {
            let code = r.read(5, "frmchexpstr")? as usize;
            for (row, &value) in fi.chexpstr.iter_mut().zip(&FRAME_EXPSTR[code]).take(blocks) {
                row[ch] = value;
            }
        }
    }
    if bsi.lfeon {
        for blk in 0..blocks {
            fi.lfeexpstr[blk] = r.read(1, "lfeexpstr")? as u8;
        }
    }
    if bsi.stream_type == StreamType::Independent {
        let convexpstre = blocks == 6 || r.bit("convexpstre")?;
        if convexpstre {
            r.skip(5 * nf, "convexpstr")?;
        }
    }
    if ahte {
        let ncplregs = (0..blocks)
            .filter(|&blk| fi.cplstre[blk] || fi.cplexpstr[blk] != REUSE)
            .count();
        if ncplblks == 6 && ncplregs == 1 {
            fi.aht[CPL] = r.bit("cplahtinu")?;
        }
        for ch in 0..nf {
            let regs = (0..blocks).filter(|&b| fi.chexpstr[b][ch] != REUSE).count();
            if regs == 1 {
                fi.aht[ch] = r.bit("chahtinu")?;
            }
        }
        if bsi.lfeon {
            let regs = (0..blocks).filter(|&b| fi.lfeexpstr[b] != REUSE).count();
            if regs == 1 {
                fi.aht[LFE] = r.bit("lfeahtinu")?;
            }
        }
    }
    if fi.snroffststr == 0 {
        fi.frm_csnr = r.read(6, "frmcsnroffst")?;
        fi.frm_fsnr = r.read(4, "frmfsnroffst")?;
    }
    if transproce {
        for ch in 0..nf {
            if r.bit("chintransproc")? {
                fi.transients[ch] = Some(Transient {
                    location: r.read(10, "transprocloc")? as u16,
                    length: r.read(8, "transproclen")? as u8,
                });
            }
        }
    }
    if spxattene {
        for ch in 0..nf {
            if r.bit("chinspxatten")? {
                fi.spxatten[ch] = Some(r.read(5, "spxattencod")? as u8);
            }
        }
    }
    if blocks > 1 && r.bit("blkstrtinfoe")? {
        let words = bsi.frame_bytes / 2;
        let bits = (blocks - 1) * (4 + words.next_power_of_two().trailing_zeros() as usize);
        r.skip(bits, "blkstrtinfo")?;
    }
    Ok(fi)
}

#[derive(Clone, Copy)]
struct Active {
    slots: [usize; SLOTS],
    count: usize,
}

impl Active {
    fn push(&mut self, slot: usize) {
        if self.count < SLOTS {
            self.slots[self.count] = slot;
            self.count += 1;
        }
    }

    fn list(&self) -> &[usize] {
        &self.slots[..self.count]
    }
}

/// Block state that persists from one block to the next within a frame.
struct State<'a> {
    bsi: &'a Bsi,
    eac3: bool,
    nf: usize,
    slots: [Slot; SLOTS],
    blksw: [bool; MAX_FBW],
    dith: [bool; MAX_FBW],
    dynrng: [Option<u8>; 2],
    cplinu: bool,
    chincpl: [bool; MAX_FBW],
    phsflginu: bool,
    cpl_begin: usize,
    cpl_end: usize,
    cpl_struct: [bool; 18],
    cpl_bands: Bands,
    cplco: [[f32; spx::MAX_BANDS]; MAX_FBW],
    have_cplco: [bool; MAX_FBW],
    phsflg: [bool; spx::MAX_BANDS],
    first_cplcos: [bool; MAX_FBW],
    first_cplleak: bool,
    leak: Option<(u32, u32)>,
    nrematbd: usize,
    rematflg: [bool; 4],
    spxinu: bool,
    spxbegf: usize,
    chinspx: [bool; MAX_FBW],
    spx: spx::Range,
    spx_struct: [bool; 17],
    spx_coords: [spx::Coords; MAX_FBW],
    first_spxcos: [bool; MAX_FBW],
    params: Option<Params>,
    csnr: Option<u32>,
    pre: [Option<Box<Premantissas>>; SLOTS],
    pre_zero: [[bool; 256]; SLOTS],
    noise: Noise,
}

impl<'a> State<'a> {
    fn new(bsi: &'a Bsi, seed: u32) -> Self {
        Self {
            bsi,
            eac3: bsi.syntax == Syntax::Eac3,
            nf: bsi.nfchans,
            slots: Default::default(),
            blksw: [false; MAX_FBW],
            dith: [true; MAX_FBW],
            dynrng: [None; 2],
            cplinu: false,
            chincpl: [false; MAX_FBW],
            phsflginu: false,
            cpl_begin: 0,
            cpl_end: 0,
            cpl_struct: DEFAULT_CPL_BANDS,
            cpl_bands: Bands::default(),
            cplco: [[0.0; spx::MAX_BANDS]; MAX_FBW],
            have_cplco: [false; MAX_FBW],
            phsflg: [false; spx::MAX_BANDS],
            first_cplcos: [true; MAX_FBW],
            first_cplleak: true,
            leak: None,
            nrematbd: 0,
            rematflg: [false; 4],
            spxinu: false,
            spxbegf: 0,
            chinspx: [false; MAX_FBW],
            spx: spx::Range::default(),
            spx_struct: DEFAULT_SPX_BANDS,
            spx_coords: [spx::Coords::default(); MAX_FBW],
            first_spxcos: [true; MAX_FBW],
            params: None,
            csnr: None,
            pre: Default::default(),
            pre_zero: [[false; 256]; SLOTS],
            noise: Noise(seed),
        }
    }

    fn lfeon(&self) -> bool {
        self.bsi.lfeon
    }

    /// Coded slots active in this block, in bit stream order: coupling,
    /// the channels, LFE.
    fn active(&self) -> Active {
        let mut list = Active {
            slots: [0; SLOTS],
            count: 0,
        };
        if self.cplinu {
            list.push(CPL);
        }
        for ch in 0..self.nf {
            list.push(ch);
        }
        if self.lfeon() {
            list.push(LFE);
        }
        list
    }

    fn block(
        &mut self,
        r: &mut BitReader<'_>,
        blk: usize,
        fi: &FrameInfo,
        coefs: &mut [[f32; COEFS]; MAX_CODED],
    ) -> Result<(), CodecError> {
        let eac3 = self.eac3;
        let nf = self.nf;
        let acmod = self.bsi.acmod;

        if !eac3 || fi.blkswe {
            for ch in 0..nf {
                self.blksw[ch] = r.bit("blksw")?;
            }
        } else {
            self.blksw = [false; MAX_FBW];
        }
        if !eac3 || fi.dithflage {
            for ch in 0..nf {
                self.dith[ch] = r.bit("dithflag")?;
            }
        } else {
            self.dith = [true; MAX_FBW];
        }
        let programs = if acmod == 0 { 2 } else { 1 };
        for program in 0..programs {
            if r.bit("dynrnge")? {
                self.dynrng[program] = Some(r.read(8, "dynrng")? as u8);
            } else if blk == 0 {
                self.dynrng[program] = None;
            }
        }

        if eac3 {
            self.spx_strategy(r, blk)?;
        }
        self.coupling_strategy(r, blk, fi)?;
        if self.cplinu {
            self.coupling_coordinates(r)?;
        } else {
            self.have_cplco = [false; MAX_FBW];
        }
        if acmod == 2 {
            let rematstr = (eac3 && blk == 0) || r.bit("rematstr")?;
            if rematstr {
                self.nrematbd = self.rematrix_bands();
                for flag in &mut self.rematflg[..self.nrematbd] {
                    *flag = r.bit("rematflg")?;
                }
            } else if blk == 0 {
                self.nrematbd = 0;
            }
        }

        // Exponent strategies, bandwidths and exponents.
        let (cplexpstr, chexpstr, lfeexpstr) = if eac3 {
            (fi.cplexpstr[blk], fi.chexpstr[blk], fi.lfeexpstr[blk])
        } else {
            let cpl = if self.cplinu {
                r.read(2, "cplexpstr")? as u8
            } else {
                REUSE
            };
            let mut ch = [REUSE; MAX_FBW];
            for value in &mut ch[..nf] {
                *value = r.read(2, "chexpstr")? as u8;
            }
            let lfe = if self.lfeon() {
                r.read(1, "lfeexpstr")? as u8
            } else {
                REUSE
            };
            (cpl, ch, lfe)
        };
        for (ch, &strategy) in chexpstr.iter().enumerate().take(nf) {
            if strategy == REUSE {
                continue;
            }
            let end = if self.chincpl[ch] {
                self.cpl_begin_bin()
            } else if self.spxinu && self.chinspx[ch] {
                self.spx.begin_bin()
            } else {
                let code = r.read(6, "chbwcod")? as usize;
                if code > 60 {
                    return Err(malformed(r, "chbwcod"));
                }
                3 * code + 73
            };
            self.slots[ch].start = 0;
            self.slots[ch].end = end;
        }
        if self.cplinu {
            self.slots[CPL].start = self.cpl_begin_bin();
            self.slots[CPL].end = self.cpl_end_bin();
            if cplexpstr != REUSE {
                let absexp = r.read(4, "cplabsexp")? as u8;
                let slot = &mut self.slots[CPL];
                let groups = exponents::cpl_groups(cplexpstr, slot.start, slot.end);
                exponents::read_groups(
                    r,
                    cplexpstr,
                    absexp << 1,
                    groups,
                    slot.start,
                    &mut slot.exps,
                )?;
                slot.have_exps = true;
            } else if !self.slots[CPL].have_exps {
                return Err(malformed(r, "coupling exponents reused before sent"));
            }
        }
        for (slot, &strategy) in self.slots.iter_mut().zip(&chexpstr).take(nf) {
            if strategy != REUSE {
                let first = r.read(4, "exps")? as u8;
                slot.exps[0] = first;
                let groups = exponents::fbw_groups(strategy, slot.end);
                exponents::read_groups(r, strategy, first, groups, 1, &mut slot.exps)?;
                r.skip(2, "gainrng")?;
                slot.have_exps = true;
            } else if !slot.have_exps {
                return Err(malformed(r, "exponents reused before sent"));
            }
        }
        if self.lfeon() {
            let slot = &mut self.slots[LFE];
            slot.start = 0;
            slot.end = 7;
            if lfeexpstr != REUSE {
                let first = r.read(4, "lfeexps")? as u8;
                slot.exps[0] = first;
                exponents::read_groups(r, D15, first, 2, 1, &mut slot.exps)?;
                slot.have_exps = true;
            } else if !slot.have_exps {
                return Err(malformed(r, "LFE exponents reused before sent"));
            }
        }

        self.allocation_info(r, blk, fi)?;

        if (!eac3 || fi.skipflde) && r.bit("skiple")? {
            let bytes = r.read(9, "skipl")? as usize;
            r.skip(8 * bytes, "skipfld")?;
        }

        // Bit allocation.
        let fscod = self.bsi.fscod;
        let params = self
            .params
            .ok_or_else(|| malformed(r, "bit allocation parameters"))?;
        let csnr = self.csnr.ok_or_else(|| malformed(r, "SNR offsets"))? as i32;
        let active = self.active();
        for &slot_index in active.list() {
            let slot = &mut self.slots[slot_index];
            if slot.aht && blk > 0 {
                continue;
            }
            let delta = (slot.delta.count > 0 && slot_index != LFE).then_some(slot.delta.segments);
            let channel = bitalloc::Channel {
                start: slot.start,
                end: slot.end,
                fscod,
                params,
                fgaincod: slot.fgaincod,
                snr: ((csnr - 15) << 4) + slot.fsnr as i32,
                leak: if slot_index == CPL { self.leak } else { None },
                delta: delta
                    .as_ref()
                    .map(|s| &s[..slot.delta.count.min(MAX_SEGMENTS)]),
                high_efficiency: slot.aht,
            };
            let exps = slot.exps;
            bitalloc::allocate(&exps, &channel, &mut slot.bap).map_err(|error| match error {
                CodecError::Malformed { what, .. } => malformed(r, what),
                other => other,
            })?;
        }

        self.mantissas(r, blk, coefs)
    }

    fn cpl_begin_bin(&self) -> usize {
        37 + 12 * self.cpl_begin
    }

    fn cpl_end_bin(&self) -> usize {
        37 + 12 * self.cpl_end
    }

    /// Annex E 3.3.2, which reduces to Tables 7.25 to 7.28 for AC-3.
    fn rematrix_bands(&self) -> usize {
        if self.cplinu {
            match self.cpl_begin {
                0 => 2,
                1 | 2 => 3,
                _ => 4,
            }
        } else if self.spxinu {
            if self.spxbegf < 2 { 3 } else { 4 }
        } else {
            4
        }
    }

    fn spx_strategy(&mut self, r: &mut BitReader<'_>, blk: usize) -> Result<(), CodecError> {
        let nf = self.nf;
        let spxstre = blk == 0 || r.bit("spxstre")?;
        if spxstre {
            self.spxinu = r.bit("spxinu")?;
            if self.spxinu {
                if self.bsi.acmod == 1 {
                    self.chinspx[0] = true;
                } else {
                    for ch in 0..nf {
                        self.chinspx[ch] = r.bit("chinspx")?;
                    }
                }
                let start = r.read(2, "spxstrtf")? as usize;
                self.spxbegf = r.read(3, "spxbegf")? as usize;
                let end = spx::end_subband(r.read(3, "spxendf")? as usize);
                let begin = spx::begin_subband(self.spxbegf);
                if begin >= end || start >= begin {
                    return Err(malformed(r, "spectral extension range"));
                }
                if r.bit("spxbndstrce")? {
                    for sbnd in begin + 1..end {
                        self.spx_struct[sbnd] = r.bit("spxbndstrc")?;
                    }
                }
                self.spx = spx::Range {
                    start,
                    begin,
                    end,
                    bands: Bands::new(&self.spx_struct, begin, end),
                };
            } else {
                self.chinspx = [false; MAX_FBW];
                self.first_spxcos = [true; MAX_FBW];
            }
        }
        if self.spxinu {
            for ch in 0..nf {
                if !self.chinspx[ch] {
                    self.first_spxcos[ch] = true;
                    continue;
                }
                let spxcoe = if self.first_spxcos[ch] {
                    self.first_spxcos[ch] = false;
                    true
                } else {
                    r.bit("spxcoe")?
                };
                if spxcoe {
                    let blend = r.read(5, "spxblnd")?;
                    let master = r.read(2, "mstrspxco")?;
                    let coords = &mut self.spx_coords[ch];
                    for band in 0..self.spx.bands.count {
                        let exp = r.read(4, "spxcoexp")?;
                        let mant = r.read(2, "spxcomant")?;
                        coords.set(band, exp, mant, master);
                    }
                    coords.blend(&self.spx, blend);
                }
            }
        }
        Ok(())
    }

    fn coupling_strategy(
        &mut self,
        r: &mut BitReader<'_>,
        blk: usize,
        fi: &FrameInfo,
    ) -> Result<(), CodecError> {
        let nf = self.nf;
        let acmod = self.bsi.acmod;
        let (cplstre, cplinu) = if self.eac3 {
            (fi.cplstre[blk], fi.cplinu[blk])
        } else {
            let cplstre = r.bit("cplstre")?;
            if !cplstre && blk == 0 {
                return Err(malformed(r, "coupling strategy missing in block 0"));
            }
            let cplinu = if cplstre {
                r.bit("cplinu")?
            } else {
                self.cplinu
            };
            (cplstre, cplinu)
        };
        if !cplstre {
            return Ok(());
        }
        self.cplinu = cplinu;
        if !cplinu {
            self.chincpl = [false; MAX_FBW];
            self.first_cplcos = [true; MAX_FBW];
            self.first_cplleak = true;
            self.phsflginu = false;
            return Ok(());
        }
        if acmod < 2 {
            return Err(malformed(r, "coupling in a mono or 1+1 stream"));
        }
        if self.eac3 && r.bit("ecplinu")? {
            return Err(CodecError::Unsupported {
                what: "E-AC-3 enhanced coupling",
            });
        }
        if self.eac3 && acmod == 2 {
            self.chincpl = [true, true, false, false, false];
        } else {
            for ch in 0..nf {
                self.chincpl[ch] = r.bit("chincpl")?;
            }
        }
        self.phsflginu = acmod == 2 && r.bit("phsflginu")?;
        self.cpl_begin = r.read(4, "cplbegf")? as usize;
        self.cpl_end = if self.eac3 && self.spxinu {
            self.spx.begin - 1
        } else {
            r.read(4, "cplendf")? as usize + 3
        };
        if self.cpl_begin >= self.cpl_end {
            return Err(malformed(r, "coupling range"));
        }
        if !self.eac3 || r.bit("cplbndstrce")? {
            for sbnd in self.cpl_begin + 1..self.cpl_end {
                self.cpl_struct[sbnd] = r.bit("cplbndstrc")?;
            }
        }
        self.cpl_bands = Bands::new(&self.cpl_struct, self.cpl_begin, self.cpl_end);
        Ok(())
    }

    fn coupling_coordinates(&mut self, r: &mut BitReader<'_>) -> Result<(), CodecError> {
        let mut any = false;
        for ch in 0..self.nf {
            if !self.chincpl[ch] {
                self.first_cplcos[ch] = true;
                self.have_cplco[ch] = false;
                continue;
            }
            let cplcoe = if self.eac3 && self.first_cplcos[ch] {
                self.first_cplcos[ch] = false;
                true
            } else {
                r.bit("cplcoe")?
            };
            if cplcoe {
                any = true;
                let master = r.read(2, "mstrcplco")?;
                for band in 0..self.cpl_bands.count {
                    let exp = r.read(4, "cplcoexp")?;
                    let mant = r.read(4, "cplcomant")?;
                    let temp = if exp == 15 {
                        mant as f32 / 16.0
                    } else {
                        (mant + 16) as f32 / 32.0
                    };
                    self.cplco[ch][band] = 8.0 * temp * (-((exp + 3 * master) as f32)).exp2();
                }
                self.have_cplco[ch] = true;
            } else if !self.have_cplco[ch] {
                return Err(malformed(r, "coupling coordinates missing"));
            }
        }
        if self.bsi.acmod == 2 {
            if !self.phsflginu {
                self.phsflg = [false; spx::MAX_BANDS];
            } else if any {
                for band in 0..self.cpl_bands.count {
                    self.phsflg[band] = r.bit("phsflg")?;
                }
            }
        }
        Ok(())
    }

    /// `baie` to `deltbaie`: parameters, SNR offsets, fast gains, leak and
    /// delta bit allocation.
    fn allocation_info(
        &mut self,
        r: &mut BitReader<'_>,
        blk: usize,
        fi: &FrameInfo,
    ) -> Result<(), CodecError> {
        let eac3 = self.eac3;
        let cplinu = self.cplinu;
        if !eac3 || fi.bamode {
            if r.bit("baie")? {
                let sdcycod = r.read(2, "sdcycod")?;
                let fdcycod = r.read(2, "fdcycod")?;
                let sgaincod = r.read(2, "sgaincod")?;
                let dbpbcod = r.read(2, "dbpbcod")?;
                let floorcod = r.read(3, "floorcod")?;
                self.params = Some(Params::from_codes(
                    sdcycod, fdcycod, sgaincod, dbpbcod, floorcod,
                ));
            }
        } else {
            self.params = Some(Params::from_codes(2, 1, 1, 2, 7));
        }

        if !eac3 {
            if r.bit("snroffste")? {
                self.csnr = Some(r.read(6, "csnroffst")?);
                for &slot in self.active().list() {
                    self.slots[slot].fsnr = r.read(4, "fsnroffst")?;
                    self.slots[slot].fgaincod = r.read(3, "fgaincod")?;
                }
            }
        } else {
            if fi.snroffststr == 0 {
                self.csnr = Some(fi.frm_csnr);
                for slot in &mut self.slots {
                    slot.fsnr = fi.frm_fsnr;
                }
            } else if blk == 0 || r.bit("snroffste")? {
                self.csnr = Some(r.read(6, "csnroffst")?);
                if fi.snroffststr == 1 {
                    let fsnr = r.read(4, "blkfsnroffst")?;
                    for slot in &mut self.slots {
                        slot.fsnr = fsnr;
                    }
                } else {
                    for &slot in self.active().list() {
                        self.slots[slot].fsnr = r.read(4, "fsnroffst")?;
                    }
                }
            }
            let fgaincode = fi.frmfgaincode && r.bit("fgaincode")?;
            for &slot in self.active().list() {
                self.slots[slot].fgaincod = if fgaincode { r.read(3, "fgaincod")? } else { 4 };
            }
            if self.bsi.stream_type == StreamType::Independent && r.bit("convsnroffste")? {
                r.skip(10, "convsnroffst")?;
            }
        }

        if cplinu {
            let cplleake = if eac3 && self.first_cplleak {
                self.first_cplleak = false;
                true
            } else {
                r.bit("cplleake")?
            };
            if cplleake {
                let fast = r.read(3, "cplfleak")?;
                let slow = r.read(3, "cplsleak")?;
                self.leak = Some((fast, slow));
            } else if self.leak.is_none() {
                return Err(malformed(r, "coupling leak missing"));
            }
        }

        if (!eac3 || fi.dbaflde) && r.bit("deltbaie")? {
            let mut modes = [0u32; SLOTS];
            let mut with_delta = self.active();
            if self.lfeon() {
                with_delta.count -= 1;
            }
            for &slot in with_delta.list() {
                modes[slot] = r.read(2, "deltbae")?;
                if modes[slot] == 3 {
                    return Err(malformed(r, "deltbae reserved"));
                }
            }
            for &slot in with_delta.list() {
                match modes[slot] {
                    1 => {
                        let count = r.read(3, "deltnseg")? as usize + 1;
                        let mut delta = Delta {
                            segments: [Segment::default(); MAX_SEGMENTS],
                            count,
                        };
                        for segment in &mut delta.segments[..count] {
                            segment.offset = r.read(5, "deltoffst")? as u8;
                            segment.len = r.read(4, "deltlen")? as u8;
                            segment.ba = r.read(3, "deltba")? as u8;
                        }
                        self.slots[slot].delta = delta;
                    }
                    2 => self.slots[slot].delta = Delta::default(),
                    _ => {}
                }
            }
        } else if blk == 0 {
            for slot in &mut self.slots {
                slot.delta = Delta::default();
            }
        }
        Ok(())
    }

    /// Mantissas in bit stream order, dither, decoupling and rematrixing
    /// for one block. `coefs` holds the coded channels, LFE last.
    fn mantissas(
        &mut self,
        r: &mut BitReader<'_>,
        blk: usize,
        coefs: &mut [[f32; COEFS]; MAX_CODED],
    ) -> Result<(), CodecError> {
        let nf = self.nf;
        for row in coefs.iter_mut() {
            *row = [0.0; COEFS];
        }
        let mut cpl = [0.0f32; COEFS];
        let mut cpl_zero = [false; COEFS];
        let mut groups = Groups::default();
        let mut got_cpl = false;
        for (ch, row) in coefs.iter_mut().enumerate().take(nf) {
            self.read_slot(r, blk, ch, &mut groups, row, None)?;
            if self.cplinu && self.chincpl[ch] && !got_cpl {
                self.read_slot(r, blk, CPL, &mut groups, &mut cpl, Some(&mut cpl_zero))?;
                got_cpl = true;
            }
        }
        if self.lfeon() {
            self.read_slot(r, blk, LFE, &mut groups, &mut coefs[nf], None)?;
        }

        if self.cplinu {
            let start = self.cpl_begin_bin();
            let exps = &self.slots[CPL].exps;
            for (ch, row) in coefs.iter_mut().enumerate().take(nf) {
                if !self.chincpl[ch] {
                    continue;
                }
                let mut bin = start;
                for (band, &size) in self.cpl_bands.sizes().iter().enumerate() {
                    let mut gain = self.cplco[ch][band];
                    if self.bsi.acmod == 2 && ch == 1 && self.phsflg[band] {
                        gain = -gain;
                    }
                    for b in bin..bin + size {
                        let m = if cpl_zero[b] {
                            if self.dith[ch] {
                                self.noise.dither()
                            } else {
                                0.0
                            }
                        } else {
                            cpl[b]
                        };
                        row[b] = m * scale(exps[b]) * gain;
                    }
                    bin += size;
                }
            }
        }

        if self.bsi.acmod == 2 {
            const BANDS: [usize; 5] = [13, 25, 37, 61, 253];
            let end = self.slots[0].end.min(self.slots[1].end);
            for band in 0..self.nrematbd {
                if !self.rematflg[band] {
                    continue;
                }
                let (left, right) = coefs.split_at_mut(1);
                for bin in BANDS[band]..BANDS[band + 1].min(end) {
                    let (l, r) = (left[0][bin], right[0][bin]);
                    left[0][bin] = l + r;
                    right[0][bin] = l - r;
                }
            }
        }
        Ok(())
    }

    /// One slot's mantissas, as coefficients with their exponents, or for
    /// the coupling channel as bare mantissas with `zero` marking `bap` 0.
    fn read_slot(
        &mut self,
        r: &mut BitReader<'_>,
        blk: usize,
        slot_index: usize,
        groups: &mut Groups,
        out: &mut [f32; COEFS],
        zero: Option<&mut [bool; COEFS]>,
    ) -> Result<(), CodecError> {
        let (start, end) = (self.slots[slot_index].start, self.slots[slot_index].end);
        let dither = slot_index < MAX_FBW && self.dith[slot_index];
        let is_cpl = slot_index == CPL;
        if self.slots[slot_index].aht {
            if blk == 0 {
                let pre = self.pre[slot_index].get_or_insert_with(|| Box::new([[0.0; 6]; 256]));
                aht::read(
                    r,
                    &self.slots[slot_index].bap,
                    start,
                    end,
                    pre,
                    &mut self.pre_zero[slot_index],
                )?;
            }
            let Some(pre) = self.pre[slot_index].as_ref() else {
                return Err(malformed(r, "AHT mantissas missing"));
            };
            let exps = &self.slots[slot_index].exps;
            let zeros = &self.pre_zero[slot_index];
            match zero {
                Some(flags) => {
                    for bin in start..end {
                        out[bin] = pre[bin][blk];
                        flags[bin] = zeros[bin];
                    }
                }
                None => {
                    for bin in start..end {
                        out[bin] = if zeros[bin] {
                            if dither {
                                self.noise.dither() * scale(exps[bin])
                            } else {
                                0.0
                            }
                        } else {
                            pre[bin][blk] * scale(exps[bin])
                        };
                    }
                }
            }
            return Ok(());
        }
        let slot = &self.slots[slot_index];
        match zero {
            Some(flags) => {
                for bin in start..end {
                    let bap = slot.bap[bin];
                    flags[bin] = bap == 0;
                    out[bin] = if bap == 0 {
                        0.0
                    } else {
                        groups.mantissa(r, bap)?
                    };
                }
            }
            None => {
                for (bin, value) in out.iter_mut().enumerate().take(end).skip(start) {
                    let bap = slot.bap[bin];
                    let m = if bap == 0 {
                        if dither && !is_cpl {
                            self.noise.dither()
                        } else {
                            0.0
                        }
                    } else {
                        groups.mantissa(r, bap)?
                    };
                    *value = m * scale(slot.exps[bin]);
                }
            }
        }
        Ok(())
    }
}

fn gain(code: Option<u8>, compr: Option<u8>, mode: DynamicRange) -> f32 {
    match mode {
        DynamicRange::Off => 1.0,
        DynamicRange::Line { cut, boost } => {
            let g = code.map_or(1.0, dynrng_gain);
            if g < 1.0 { g.powf(cut) } else { g.powf(boost) }
        }
        DynamicRange::Heavy => match compr {
            Some(c) => compr_gain(c),
            None => code.map_or(1.0, dynrng_gain),
        },
    }
}

/// Decode the audio of one whole syncframe whose `bsi` is `bsi`.
pub(crate) fn decode(
    frame: &[u8],
    bsi: &Bsi,
    substream: &mut Substream,
    transform: &Transform,
    dynamics: DynamicRange,
    words: Option<&Drc>,
) -> Result<Decoded, CodecError> {
    let frame = frame.get(..bsi.frame_bytes).ok_or(CodecError::Truncated {
        at_bit: frame.len() as u64 * 8,
        what: "syncframe",
    })?;
    if bsi.nfchans > MAX_FBW || bsi.blocks > MAX_BLOCKS {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "channel or block count",
        });
    }
    check_crc(frame, bsi.syntax)?;
    let payload = &frame[..frame.len().saturating_sub(2)];
    let mut r = BitReader::new(payload);
    r.skip(bsi.end_bit, "bsi")?;
    let fi = if bsi.syntax == Syntax::Eac3 {
        read_audfrm(&mut r, bsi)?
    } else {
        FrameInfo::ac3()
    };

    let mut state = State::new(bsi, substream.seed);
    for slot in [CPL, LFE].into_iter().chain(0..bsi.nfchans) {
        state.slots[slot].aht = fi.aht[slot];
    }
    let coded = bsi.nfchans + usize::from(bsi.lfeon);
    let samples = bsi.blocks * COEFS;
    let mut pcm = vec![0.0f32; coded * samples];
    let mut delay = substream.delay;
    let mut coefs = [[0.0f32; COEFS]; MAX_CODED];
    let mut block = [0.0f32; SPAN];
    let mut tools = Tools {
        frames: 1,
        dependent_frames: u64::from(bsi.stream_type == StreamType::Dependent),
        short_frames: u64::from(bsi.syntax == Syntax::Eac3 && bsi.blocks < MAX_BLOCKS),
        aht_channels: fi.aht.iter().filter(|&&on| on).count() as u64,
        transient_channels: fi.transients.iter().flatten().count() as u64,
        ..Tools::default()
    };
    let mut drc = Drc {
        compr: bsi.compr,
        ..Drc::default()
    };
    for blk in 0..bsi.blocks {
        state.block(&mut r, blk, &fi, &mut coefs)?;
        drc.dynrng[blk] = state.dynrng;
        let applied = words.unwrap_or(&drc);
        tools.blocks += 1;
        tools.short_blocks += state.blksw[..bsi.nfchans].iter().filter(|&&on| on).count() as u64;
        tools.coupling_blocks += u64::from(state.cplinu);
        tools.phase_blocks += u64::from(state.cplinu && state.phsflginu);
        tools.rematrix_blocks +=
            u64::from(bsi.acmod == 2 && state.rematflg[..state.nrematbd].contains(&true));
        tools.spx_blocks += u64::from(state.spxinu);
        tools.delta_blocks += u64::from(
            state
                .active()
                .list()
                .iter()
                .any(|&slot| state.slots[slot].delta.count > 0),
        );
        for (ch, row) in coefs.iter_mut().enumerate().take(coded) {
            let program = usize::from(bsi.acmod == 0 && ch == 1);
            let g = gain(
                applied.dynrng[blk][program],
                applied.compr[program],
                dynamics,
            );
            if g != 1.0 {
                for value in row.iter_mut() {
                    *value *= g;
                }
            }
        }
        if state.spxinu {
            for (ch, row) in coefs.iter_mut().enumerate().take(bsi.nfchans) {
                if state.chinspx[ch] {
                    let noise = &mut state.noise;
                    spx::synthesize(
                        row,
                        &state.spx,
                        &state.spx_coords[ch],
                        fi.spxatten[ch],
                        || noise.unit(),
                    );
                }
            }
        }
        for ch in 0..coded {
            let short = ch < bsi.nfchans && state.blksw[ch];
            if short {
                transform.inverse_short(&coefs[ch], &mut block);
            } else {
                transform.inverse_long(&coefs[ch], &mut block);
            }
            let start = ch * samples + blk * COEFS;
            overlap_add(&block, &mut delay[ch], &mut pcm[start..start + COEFS]);
        }
    }
    substream.delay = delay;
    substream.seed = state.noise.0;
    Ok(Decoded {
        channels: coded,
        samples,
        pcm,
        transients: fi.transients,
        tools,
        drc,
    })
}
