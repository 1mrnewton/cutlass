//! Syncframe syntax: AC-3 sections 5.3 and 5.4 (with the Annex D `xbsi`)
//! and E-AC-3 Annex E section 2, field for field as `header` and `frame`
//! read them, with the mantissa quantizers of section 7.3 in reverse.

use crate::bits::BitWriter;
use crate::crc;
use crate::exponents::{cpl_groups, fbw_groups, group_size};
use crate::mdct::COEFS;
use crate::tables::{BAP_BITS, BAP_LEVELS, FRAME_EXPSTR, REUSE};

use super::plan::{Block, CPL, LFE, MAX_BLOCKS, MAX_FBW, Shape};

/// Bit allocation parameter codes: `sdcycod`, `fdcycod`, `sgaincod`,
/// `dbpbcod`, `floorcod`. E-AC-3 without `bamode` implies its own.
pub(crate) const AC3_PARAMS: [u32; 5] = [2, 1, 1, 3, 7];
pub(crate) const EAC3_PARAMS: [u32; 5] = [2, 1, 1, 2, 7];

/// `auxdatae`, `crcrsv` (or `encinfo`) and the final CRC word.
pub(crate) const TAIL_BITS: usize = 18;

/// Downmix codes of Annex D Tables D2.2 to D2.6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixCodes {
    /// Preferred stereo downmix: 0 not indicated, 1 Lt/Rt, 2 Lo/Ro.
    pub dmixmod: u8,
    /// Center level codes 0 to 7: +3, +1.5, 0, -1.5, -3, -4.5, -6 dB, off.
    pub ltrt_center: u8,
    /// Surround level codes 3 to 7: -1.5, -3, -4.5, -6 dB, off.
    pub ltrt_surround: u8,
    pub loro_center: u8,
    pub loro_surround: u8,
}

impl MixCodes {
    /// The levels a decoder assumes when a stream carries none: Lt/Rt
    /// -3 dB, Lo/Ro -4.5 dB center and -6 dB surround.
    pub const DEFAULT: Self = Self {
        dmixmod: 0,
        ltrt_center: 4,
        ltrt_surround: 4,
        loro_center: 5,
        loro_surround: 6,
    };
}

/// Stream-level metadata the header carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Meta {
    pub dialnorm: u8,
    pub bsmod: u8,
    pub dsurmod: u8,
    pub mix: Option<MixCodes>,
    /// E-AC-3 `lfemixlevcod`.
    pub lfe_mix: Option<u8>,
}

/// What changes per syncframe besides the blocks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub bytes: usize,
    /// AC-3 `frmsizecod`.
    pub frmsizecod: u8,
    pub index: u64,
    /// `(csnroffst << 4) | fsnroffst`.
    pub snr: u16,
}

/// Bit positions a syncframe written without mantissas reaches: the end
/// of each block's side information, and the total.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Layout {
    pub side_end: [usize; MAX_BLOCKS],
    pub total: usize,
}

/// Write one syncframe up to its last mantissa. Without `mantissas` only
/// the side information is written, for the allocation search.
pub(crate) fn frame(
    shape: &Shape,
    meta: &Meta,
    frame: &Frame,
    blocks: &[Block],
    skip: &[u16; MAX_BLOCKS],
    mantissas: bool,
) -> (BitWriter, Layout) {
    let mut w = BitWriter::with_capacity(frame.bytes);
    if shape.eac3 {
        eac3_header(&mut w, shape, meta, frame);
        audfrm(&mut w, shape, blocks, frame.snr);
    } else {
        ac3_header(&mut w, shape, meta, frame);
    }
    let mut layout = Layout::default();
    let mut coupling = CouplingState::new();
    let context = Context {
        shape,
        flags: FrameFlags::of(shape, blocks),
        snr: frame.snr,
    };
    for (blk, block) in blocks.iter().enumerate() {
        audblk(&mut w, &context, blk, block, &mut coupling, skip[blk]);
        layout.side_end[blk] = w.bits();
        if mantissas {
            write_mantissas(&mut w, shape, block);
        }
    }
    layout.total = w.bits();
    (w, layout)
}

/// Pad to the frame size and seal it with its CRC words.
pub(crate) fn seal(shape: &Shape, w: BitWriter, bytes: usize) -> Vec<u8> {
    let mut data = w.finish();
    data.resize(bytes, 0);
    if shape.eac3 {
        let word = crc::update(0, &data[2..bytes - 2]);
        data[bytes - 2..].copy_from_slice(&word.to_be_bytes());
        return data;
    }
    let five_eighths = five_eighths_bytes(bytes);
    let word = crc::crc1_for(&data[4..five_eighths]);
    data[2..4].copy_from_slice(&word.to_be_bytes());
    let mut word = crc::update(0, &data[five_eighths..bytes - 2]);
    if word == crate::header::SYNCWORD {
        // Section 5.4.5.1: flip `crcrsv` so crc2 cannot read as a sync word.
        data[bytes - 3] ^= 1;
        word = crc::update(0, &data[five_eighths..bytes - 2]);
    }
    data[bytes - 2..].copy_from_slice(&word.to_be_bytes());
    data
}

/// Bytes in the first 5/8 of an AC-3 syncframe (Table 7.34).
pub(crate) fn five_eighths_bytes(bytes: usize) -> usize {
    let words = bytes / 2;
    ((words >> 1) + (words >> 3)) * 2
}

fn ac3_header(w: &mut BitWriter, shape: &Shape, meta: &Meta, frame: &Frame) {
    let acmod = shape.acmod;
    w.put(16, u32::from(crate::header::SYNCWORD));
    w.put(16, 0);
    w.put(2, u32::from(shape.fscod));
    w.put(6, u32::from(frame.frmsizecod));
    w.put(5, if meta.mix.is_some() { 6 } else { 8 });
    w.put(3, u32::from(meta.bsmod));
    w.put(3, u32::from(acmod));
    let (cmixlev, surmixlev) = meta.mix.map_or((1, 1), |mix| {
        (
            match mix.loro_center {
                4 => 0,
                6 => 2,
                _ => 1,
            },
            match mix.loro_surround {
                4 => 0,
                7 => 2,
                _ => 1,
            },
        )
    });
    if acmod & 1 != 0 && acmod != 1 {
        w.put(2, cmixlev);
    }
    if acmod & 4 != 0 {
        w.put(2, surmixlev);
    }
    if acmod == 2 {
        w.put(2, u32::from(meta.dsurmod));
    }
    w.flag(shape.lfeon);
    w.put(5, u32::from(meta.dialnorm));
    w.flag(false); // compre
    w.flag(false); // langcode
    w.flag(false); // audprodie
    w.flag(false); // copyrightb
    w.flag(true); // origbs
    if let Some(mix) = meta.mix {
        w.flag(true); // xbsi1e
        w.put(2, u32::from(mix.dmixmod));
        w.put(3, u32::from(mix.ltrt_center));
        w.put(3, u32::from(mix.ltrt_surround));
        w.put(3, u32::from(mix.loro_center));
        w.put(3, u32::from(mix.loro_surround));
        w.flag(false); // xbsi2e
    } else {
        w.flag(false); // timecod1e
        w.flag(false); // timecod2e
    }
    w.flag(false); // addbsie
}

fn eac3_header(w: &mut BitWriter, shape: &Shape, meta: &Meta, frame: &Frame) {
    let acmod = shape.acmod;
    w.put(16, u32::from(crate::header::SYNCWORD));
    w.put(2, 0); // strmtyp: independent
    w.put(3, 0); // substreamid
    w.put(11, (frame.bytes / 2 - 1) as u32);
    w.put(2, u32::from(shape.fscod));
    w.put(2, numblkscod(shape.blocks));
    w.put(3, u32::from(acmod));
    w.flag(shape.lfeon);
    w.put(5, 16);
    w.put(5, u32::from(meta.dialnorm));
    w.flag(false); // compre
    let mixing = meta.mix.is_some() || (shape.lfeon && meta.lfe_mix.is_some());
    w.flag(mixing);
    if mixing {
        let mix = meta.mix.unwrap_or(MixCodes::DEFAULT);
        if acmod > 2 {
            w.put(2, u32::from(mix.dmixmod));
        }
        if acmod & 1 != 0 && acmod > 2 {
            w.put(3, u32::from(mix.ltrt_center));
            w.put(3, u32::from(mix.loro_center));
        }
        if acmod & 4 != 0 {
            w.put(3, u32::from(mix.ltrt_surround));
            w.put(3, u32::from(mix.loro_surround));
        }
        if shape.lfeon {
            w.flag(meta.lfe_mix.is_some());
            if let Some(code) = meta.lfe_mix {
                w.put(5, u32::from(code));
            }
        }
        w.flag(false); // pgmscle
        w.flag(false); // extpgmscle
        w.put(2, 0); // mixdef
        if acmod < 2 {
            w.flag(false); // paninfoe
        }
        w.flag(false); // frmmixcfginfoe
    }
    w.flag(true); // infomdate
    w.put(3, u32::from(meta.bsmod));
    w.flag(false); // copyrightb
    w.flag(true); // origbs
    if acmod == 2 {
        w.put(2, u32::from(meta.dsurmod));
        w.put(2, 0); // dheadphonmod
    }
    if acmod >= 6 {
        w.put(2, 0); // dsurexmod
    }
    w.flag(false); // audprodie
    w.flag(false); // sourcefscod
    if shape.blocks != 6 {
        w.flag((frame.index * shape.blocks as u64).is_multiple_of(6)); // convsync
    }
    w.flag(false); // addbsie
}

fn numblkscod(blocks: usize) -> u32 {
    match blocks {
        1 => 0,
        2 => 1,
        3 => 2,
        _ => 3,
    }
}

/// Annex E Table E1.3.
fn audfrm(w: &mut BitWriter, shape: &Shape, blocks: &[Block], snr: u16) {
    let nf = shape.nf;
    if shape.blocks == 6 {
        w.flag(true); // expstre
        w.flag(false); // ahte
    }
    w.put(2, 0); // snroffststr
    w.flag(false); // transproce
    let flags = FrameFlags::of(shape, blocks);
    w.flag(flags.blkswe);
    w.flag(flags.dithflage);
    w.flag(false); // bamode
    w.flag(false); // frmfgaincode
    w.flag(false); // dbaflde
    w.flag(false); // skipflde
    w.flag(false); // spxattene
    if shape.acmod > 1 {
        w.flag(blocks[0].cplinu);
        for block in &blocks[1..] {
            w.flag(block.cplstre);
            if block.cplstre {
                w.flag(block.cplinu);
            }
        }
    }
    for block in blocks {
        if block.cplinu {
            w.put(2, u32::from(block.expstr[CPL]));
        }
        for ch in 0..nf {
            w.put(2, u32::from(block.expstr[ch]));
        }
    }
    if shape.lfeon {
        for block in blocks {
            w.flag(block.expstr[LFE] != REUSE);
        }
    }
    if shape.blocks == 6 {
        for ch in 0..nf {
            let strategies: Vec<u8> = blocks.iter().map(|b| b.expstr[ch]).collect();
            let code = FRAME_EXPSTR
                .iter()
                .position(|row| row[..] == strategies[..])
                .unwrap_or(0);
            w.put(5, code as u32); // convexpstr
        }
    } else {
        w.flag(false); // convexpstre
    }
    w.put(6, u32::from(snr >> 4));
    w.put(4, u32::from(snr & 15));
    if shape.blocks > 1 {
        w.flag(false); // blkstrtinfoe
    }
}

/// E-AC-3 `blkswe` and `dithflage`: per-block flags only when a block
/// needs them. AC-3 always sends them.
#[derive(Clone, Copy, Debug)]
struct FrameFlags {
    blkswe: bool,
    dithflage: bool,
}

impl FrameFlags {
    fn of(shape: &Shape, blocks: &[Block]) -> Self {
        let nf = shape.nf;
        Self {
            blkswe: !shape.eac3 || blocks.iter().any(|b| b.blksw[..nf].contains(&true)),
            dithflage: !shape.eac3 || blocks.iter().any(|b| b.dith[..nf].contains(&false)),
        }
    }
}

/// The decoder's E-AC-3 implicit-first flags (Annex E 2.3.3).
struct CouplingState {
    first_coords: [bool; MAX_FBW],
    first_leak: bool,
}

impl CouplingState {
    fn new() -> Self {
        Self {
            first_coords: [true; MAX_FBW],
            first_leak: true,
        }
    }
}

/// What every block of one frame shares.
struct Context<'a> {
    shape: &'a Shape,
    flags: FrameFlags,
    snr: u16,
}

fn audblk(
    w: &mut BitWriter,
    context: &Context<'_>,
    blk: usize,
    block: &Block,
    coupling: &mut CouplingState,
    skip: u16,
) {
    let Context { shape, flags, snr } = *context;
    let eac3 = shape.eac3;
    let nf = shape.nf;
    let acmod = shape.acmod;
    if flags.blkswe {
        for ch in 0..nf {
            w.flag(block.blksw[ch]);
        }
    }
    if flags.dithflage {
        for ch in 0..nf {
            w.flag(block.dith[ch]);
        }
    }
    w.flag(false); // dynrnge
    if eac3 {
        w.flag(false); // spxinu in block 0, spxstre after
    }

    // Coupling strategy.
    if !eac3 {
        w.flag(block.cplstre);
        if block.cplstre {
            w.flag(block.cplinu);
        }
    }
    if block.cplstre {
        if block.cplinu {
            if eac3 {
                w.flag(false); // ecplinu
            }
            if !(eac3 && acmod == 2) {
                for ch in 0..nf {
                    w.flag(block.chincpl[ch]);
                }
            }
            if acmod == 2 {
                w.flag(block.phsflginu);
            }
            let (begin, end) = shape.coupling.unwrap_or((0, 3));
            w.put(4, begin as u32);
            w.put(4, (end - 3) as u32);
            // Send the structure on every coupling strategy. A decoder that
            // loads the default structure only in block 0 mis-counts the
            // bands when coupling starts in a later block, and the rest of
            // the frame shifts.
            if eac3 {
                w.flag(true); // cplbndstrce
            }
            for sbnd in begin + 1..end {
                w.flag(crate::tables::DEFAULT_CPL_BANDS[sbnd]);
            }
        } else {
            coupling.first_coords = [true; MAX_FBW];
            coupling.first_leak = true;
        }
    }

    // Coupling coordinates.
    if block.cplinu {
        let bands = shape.cpl_bands().count;
        let mut sent = false;
        for ch in 0..nf {
            if !block.chincpl[ch] {
                coupling.first_coords[ch] = true;
                continue;
            }
            if eac3 && coupling.first_coords[ch] {
                coupling.first_coords[ch] = false;
            } else {
                w.flag(block.cplcoe[ch]);
            }
            if block.cplcoe[ch] {
                let coords = &block.coords[ch];
                w.put(2, u32::from(coords.master));
                for band in 0..bands {
                    w.put(4, u32::from(coords.exp[band]));
                    w.put(4, u32::from(coords.mant[band]));
                }
                sent = true;
            }
        }
        if acmod == 2 && block.phsflginu && sent {
            for &flag in &block.phsflg[..bands] {
                w.flag(flag);
            }
        }
    }

    // Rematrixing.
    if acmod == 2 {
        if !(eac3 && blk == 0) {
            w.flag(block.rematstr);
        }
        if block.rematstr {
            for &flag in &block.rematflg[..block.nrematbd] {
                w.flag(flag);
            }
        }
    }

    // Exponent strategies, bandwidths and exponents.
    if !eac3 {
        if block.cplinu {
            w.put(2, u32::from(block.expstr[CPL]));
        }
        for ch in 0..nf {
            w.put(2, u32::from(block.expstr[ch]));
        }
        if shape.lfeon {
            w.flag(block.expstr[LFE] != REUSE);
        }
    }
    for ch in 0..nf {
        if block.expstr[ch] != REUSE && !block.chincpl[ch] {
            w.put(6, u32::from(shape.bandwidth));
        }
    }
    if block.cplinu && block.expstr[CPL] != REUSE {
        let start = block.start[CPL];
        let exps = &block.exps[CPL];
        let reference = exps[start] & !1;
        w.put(4, u32::from(reference >> 1));
        let strategy = block.expstr[CPL];
        let size = group_size(strategy);
        let positions = 3 * cpl_groups(strategy, start, block.end[CPL]);
        groups(w, reference, (0..positions).map(|i| exps[start + i * size]));
    }
    for ch in 0..nf {
        let strategy = block.expstr[ch];
        if strategy == REUSE {
            continue;
        }
        let exps = &block.exps[ch];
        w.put(4, u32::from(exps[0]));
        let size = group_size(strategy);
        let positions = 3 * fbw_groups(strategy, block.end[ch]);
        groups(w, exps[0], (0..positions).map(|i| exps[1 + i * size]));
        w.put(2, 0); // gainrng
    }
    if shape.lfeon && block.expstr[LFE] != REUSE {
        let exps = &block.exps[LFE];
        w.put(4, u32::from(exps[0]));
        groups(w, exps[0], (0..6).map(|i| exps[1 + i]));
    }

    // Bit allocation information.
    if !eac3 {
        w.flag(blk == 0); // baie
        if blk == 0 {
            for (code, bits) in AC3_PARAMS.iter().zip([2, 2, 2, 2, 3]) {
                w.put(bits, *code);
            }
        }
        w.flag(block.snroffste);
        if block.snroffste {
            w.put(6, u32::from(snr >> 4));
            for _ in block.active(shape) {
                w.put(4, u32::from(snr & 15));
                w.put(3, 4); // fgaincod
            }
        }
    } else {
        w.flag(false); // convsnroffste
    }
    if block.cplinu {
        let leak = if eac3 && coupling.first_leak {
            coupling.first_leak = false;
            true
        } else {
            w.flag(block.cplleake);
            block.cplleake
        };
        if leak {
            w.put(3, 0); // cplfleak
            w.put(3, 0); // cplsleak
        }
    }
    if !eac3 {
        w.flag(false); // deltbaie
        w.flag(skip > 0); // skiple
        if skip > 0 {
            w.put(9, u32::from(skip));
            for _ in 0..skip {
                w.put(8, 0);
            }
        }
    }
}

/// Grouped differential exponents: three steps of -2 to +2 per 7 bits.
fn groups(w: &mut BitWriter, reference: u8, values: impl Iterator<Item = u8>) {
    let mut previous = i32::from(reference);
    let mut code = 0u32;
    let mut filled = 0;
    for value in values {
        let delta = (i32::from(value) - previous + 2).clamp(0, 4) as u32;
        // The next group continues from the exponent a decoder reconstructs.
        previous += delta as i32 - 2;
        code = code * 5 + delta;
        filled += 1;
        if filled == 3 {
            w.put(7, code);
            code = 0;
            filled = 0;
        }
    }
}

/// Mantissa bits of one block for its bit allocation pointers.
pub(crate) fn mantissa_bits(shape: &Shape, block: &Block) -> usize {
    let (mut ones, mut twos, mut fours, mut bits) = (0usize, 0usize, 0usize, 0usize);
    for slot in block.active(shape) {
        for &bap in &block.bap[slot][block.start[slot]..block.end[slot]] {
            match bap {
                0 => {}
                1 => ones += 1,
                2 => twos += 1,
                4 => fours += 1,
                _ => bits += BAP_BITS[usize::from(bap)] as usize,
            }
        }
    }
    bits + ones.div_ceil(3) * 5 + twos.div_ceil(3) * 7 + fours.div_ceil(2) * 7
}

/// Symmetric quantizer of `levels` levels for a mantissa in [-1, 1).
fn symmetric(m: f32, levels: u32) -> u32 {
    let code = ((m * levels as f32 + (levels - 1) as f32) / 2.0).round();
    code.clamp(0.0, (levels - 1) as f32) as u32
}

/// Two's complement quantizer of `bits` bits.
fn asymmetric(m: f32, bits: u32) -> i32 {
    let half = (1i32 << (bits - 1)) as f32;
    (m * half).round().clamp(-half, half - 1.0) as i32
}

/// One block's mantissas in bit stream order, grouped as section 7.3.5
/// reads them.
fn write_mantissas(w: &mut BitWriter, shape: &Shape, block: &Block) {
    let mut order: Vec<(u8, u32)> = Vec::with_capacity(COEFS * 2);
    let mut push = |slot: usize| {
        let (start, end) = (block.start[slot], block.end[slot]);
        for bin in start..end {
            let bap = block.bap[slot][bin];
            if bap == 0 {
                continue;
            }
            let m = block.coefs[slot][bin] * f32::from(block.exps[slot][bin]).exp2();
            let code = match bap {
                1..=5 => symmetric(m, BAP_LEVELS[usize::from(bap)]),
                _ => asymmetric(m, BAP_BITS[usize::from(bap)]) as u32,
            };
            order.push((bap, code));
        }
    };
    let mut wrote_cpl = false;
    for ch in 0..shape.nf {
        push(ch);
        if block.cplinu && block.chincpl[ch] && !wrote_cpl {
            push(CPL);
            wrote_cpl = true;
        }
    }
    if shape.lfeon {
        push(LFE);
    }
    // Group codes, each written where its first member falls.
    let mut grouped: [Vec<u32>; 3] = Default::default();
    for &(bap, code) in &order {
        if let Some(kind) = group_kind(bap) {
            grouped[kind].push(code);
        }
    }
    let mut seen = [0usize; 3];
    for &(bap, code) in &order {
        match group_kind(bap) {
            Some(kind) => {
                let (per, bits, radix, pad) = GROUPS[kind];
                if seen[kind] % per == 0 {
                    let members = &grouped[kind][seen[kind]..];
                    let mut value = 0;
                    for i in 0..per {
                        value = value * radix + members.get(i).copied().unwrap_or(pad);
                    }
                    w.put(bits, value);
                }
                seen[kind] += 1;
            }
            None => w.put(BAP_BITS[usize::from(bap)], code),
        }
    }
}

/// Members per group, bits per group, levels, and the code for zero, of
/// `bap` 1, 2 and 4.
const GROUPS: [(usize, u32, u32, u32); 3] = [(3, 5, 3, 1), (3, 7, 5, 2), (2, 7, 11, 5)];

fn group_kind(bap: u8) -> Option<usize> {
    match bap {
        1 => Some(0),
        2 => Some(1),
        4 => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(code: u32, levels: u32) -> f32 {
        (2.0 * code as f32 - (levels - 1) as f32) / levels as f32
    }

    #[test]
    fn symmetric_codes_pick_the_nearest_level() {
        for levels in [3u32, 5, 7, 11, 15] {
            for step in -100..100 {
                let m = step as f32 / 100.0;
                let code = symmetric(m, levels);
                let best = (0..levels)
                    .min_by(|&a, &b| {
                        (level(a, levels) - m)
                            .abs()
                            .total_cmp(&(level(b, levels) - m).abs())
                    })
                    .unwrap_or(0);
                assert!(
                    (level(code, levels) - m).abs() <= (level(best, levels) - m).abs() + 1e-6,
                    "{levels} {m}"
                );
            }
        }
    }

    #[test]
    fn asymmetric_codes_saturate_at_full_scale() {
        assert_eq!(asymmetric(0.999_99, 5), 15);
        assert_eq!(asymmetric(-1.0, 5), -16);
        assert_eq!(asymmetric(0.0, 16), 0);
        assert_eq!(asymmetric(0.5, 6), 16);
    }

    #[test]
    fn the_five_eighths_point_follows_table_7_34() {
        assert_eq!(five_eighths_bytes(128 * 2), 80 * 2);
        assert_eq!(five_eighths_bytes(69 * 2), (34 + 8) * 2);
        assert_eq!(five_eighths_bytes(1920 * 2), 1200 * 2);
    }
}
