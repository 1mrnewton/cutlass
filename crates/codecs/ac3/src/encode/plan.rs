//! One syncframe's coding decisions (A/52 sections 8.2.4 to 8.2.10):
//! coupling, rematrixing, exponent strategies, and the exponents the
//! decoder will rebuild.

use crate::exponents::{cpl_groups, fbw_groups, group_size};
use crate::mdct::COEFS;
use crate::spx::Bands;
use crate::tables::{D15, D25, D45, DEFAULT_CPL_BANDS, DEFAULT_SPX_BANDS, REUSE};

pub(crate) const MAX_FBW: usize = 5;
pub(crate) const CPL: usize = MAX_FBW;
pub(crate) const LFE: usize = MAX_FBW + 1;
pub(crate) const SLOTS: usize = MAX_FBW + 2;
pub(crate) const MAX_BLOCKS: usize = 6;
pub(crate) const MAX_CPL_BANDS: usize = 18;

/// Rematrixing band edges (Table 7.25).
const REMAT_BANDS: [usize; 5] = [13, 25, 37, 61, 253];

/// Mean absolute exponent change per bin above which a block sends new
/// exponents.
const EXP_CHANGE: u32 = 2;

/// Mean coupling coordinate change above which a block sends new ones.
const COORD_CHANGE: f32 = 0.24;

/// What does not change from frame to frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub eac3: bool,
    pub acmod: u8,
    pub lfeon: bool,
    pub nf: usize,
    pub blocks: usize,
    pub fscod: u8,
    /// `chbwcod` of channels outside coupling.
    pub bandwidth: u8,
    /// `cplbegf` and the coupling end sub-band, `cplendf + 3`.
    pub coupling: Option<(usize, usize)>,
    /// Spectral extension. Channels are coded up to its begin bin.
    pub spx: Option<Spx>,
}

/// The low-rate spectral extension: copy sub-band 2, code through
/// sub-band 5, synthesize through sub-band 13. `spxblnd` 31 leaves the
/// blend on the copied signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Spx {
    pub start: usize,
    pub begf: usize,
    pub endf: usize,
}

impl Spx {
    pub(crate) const LOW_RATE: Self = Self {
        start: 2,
        begf: 3,
        endf: 5,
    };

    pub(crate) fn range(self) -> crate::spx::Range {
        let begin = crate::spx::begin_subband(self.begf);
        let end = crate::spx::end_subband(self.endf);
        crate::spx::Range {
            start: self.start,
            begin,
            end,
            bands: Bands::new(&DEFAULT_SPX_BANDS, begin, end),
        }
    }

    pub(crate) fn begin_bin(self) -> usize {
        self.range().begin_bin()
    }
}

/// Spectral extension below 48 kbit/s per channel, when it replaces bins
/// that rate would still code. 48 kbit/s per channel still couples.
pub(crate) fn extension_for(eac3: bool, per_channel_kbps: u32, end_bin: usize) -> Option<Spx> {
    let spx = Spx::LOW_RATE;
    (eac3 && per_channel_kbps < 48 && end_bin > spx.begin_bin() + 12).then_some(spx)
}

impl Shape {
    pub(crate) fn end_bin(&self) -> usize {
        73 + 3 * usize::from(self.bandwidth)
    }

    pub(crate) fn cpl_bands(&self) -> Bands {
        match self.coupling {
            Some((begin, end)) => Bands::new(&DEFAULT_CPL_BANDS, begin, end),
            None => Bands::default(),
        }
    }
}

/// One channel's quantized coupling coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Coords {
    pub master: u8,
    pub exp: [u8; MAX_CPL_BANDS],
    pub mant: [u8; MAX_CPL_BANDS],
}

/// One audio block: the coefficients and every decision about them.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    /// Full bandwidth channels, then the coupling channel and LFE.
    pub coefs: [[f32; COEFS]; SLOTS],
    pub blksw: [bool; MAX_FBW],
    pub dith: [bool; MAX_FBW],
    pub cplstre: bool,
    pub cplinu: bool,
    pub chincpl: [bool; MAX_FBW],
    pub cplcoe: [bool; MAX_FBW],
    pub coords: [Coords; MAX_FBW],
    /// Stereo coupling: negate the right channel's coordinate in a band.
    pub phsflginu: bool,
    pub phsflg: [bool; MAX_CPL_BANDS],
    pub cplleake: bool,
    pub rematstr: bool,
    pub nrematbd: usize,
    pub rematflg: [bool; 4],
    pub expstr: [u8; SLOTS],
    pub start: [usize; SLOTS],
    pub end: [usize; SLOTS],
    /// Exponents as the decoder rebuilds them.
    pub exps: [[u8; COEFS]; SLOTS],
    pub bap: [[u8; COEFS]; SLOTS],
    /// Delta bit allocation. Empty unless a test asks a block to carry it.
    pub deltas: [crate::bitalloc::Delta; SLOTS],
    /// Send new spectral-extension coordinates for this channel.
    pub spx_send: [bool; MAX_FBW],
    pub spx_master: [u8; MAX_FBW],
    pub spx_exp: [[u8; crate::spx::MAX_BANDS]; MAX_FBW],
    pub spx_mant: [[u8; crate::spx::MAX_BANDS]; MAX_FBW],
    /// Adaptive hybrid transform. One exponent set, mantissas once per frame.
    pub aht: [bool; SLOTS],
    /// `dynrng` in force for this block, and whether this block sends it.
    pub dynrng: u8,
    pub dynrnge: bool,
    /// AC-3 `snroffste`.
    pub snroffste: bool,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            coefs: [[0.0; COEFS]; SLOTS],
            blksw: [false; MAX_FBW],
            dith: [true; MAX_FBW],
            cplstre: false,
            cplinu: false,
            chincpl: [false; MAX_FBW],
            cplcoe: [false; MAX_FBW],
            coords: [Coords::default(); MAX_FBW],
            phsflginu: false,
            phsflg: [false; MAX_CPL_BANDS],
            cplleake: false,
            rematstr: false,
            nrematbd: 0,
            rematflg: [false; 4],
            expstr: [REUSE; SLOTS],
            start: [0; SLOTS],
            end: [0; SLOTS],
            exps: [[0; COEFS]; SLOTS],
            bap: [[0; COEFS]; SLOTS],
            deltas: [crate::bitalloc::Delta::default(); SLOTS],
            spx_send: [false; MAX_FBW],
            spx_master: [0; MAX_FBW],
            spx_exp: [[0; crate::spx::MAX_BANDS]; MAX_FBW],
            spx_mant: [[0; crate::spx::MAX_BANDS]; MAX_FBW],
            aht: [false; SLOTS],
            dynrng: 0,
            dynrnge: false,
            snroffste: false,
        }
    }
}

impl Block {
    /// Coded slots of this block in bit stream order: coupling, the
    /// channels, LFE.
    pub(crate) fn active(&self, shape: &Shape) -> impl Iterator<Item = usize> + use<> {
        let cpl = self.cplinu.then_some(CPL);
        let lfe = shape.lfeon.then_some(LFE);
        cpl.into_iter().chain(0..shape.nf).chain(lfe)
    }
}

/// How much side information a frame is allowed to spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Adaptive,
    /// One exponent set per channel, no coupling: the smallest side
    /// information a syncframe can send.
    Coarse,
}

/// Fill in every decision for `blocks`, whose `coefs` (full bandwidth
/// and LFE), `blksw`, and `dith` the analysis set.
pub(crate) fn plan(shape: &Shape, blocks: &mut [Block], mode: Mode) {
    coupling(shape, blocks, mode);
    spectral(shape, blocks);
    rematrix(shape, blocks);
    exponents(shape, blocks, mode);
    snr_flags(shape, blocks);
    hybrid(shape, blocks);
}

/// One hybrid transform when every channel keeps a single exponent set.
fn hybrid(shape: &Shape, blocks: &mut [Block]) {
    if !shape.eac3 || shape.blocks != 6 || shape.spx.is_some() {
        return;
    }
    if blocks.iter().any(|block| block.cplinu) {
        return;
    }
    // The hybrid transform is one long spectrum. A short block is not.
    if blocks
        .iter()
        .any(|block| block.blksw[..shape.nf].contains(&true))
    {
        return;
    }
    let mut slots = Vec::with_capacity(shape.nf + 1);
    slots.extend(0..shape.nf);
    if shape.lfeon {
        slots.push(LFE);
    }
    let steady = slots.iter().all(|&slot| {
        blocks
            .iter()
            .filter(|block| block.expstr[slot] != REUSE)
            .count()
            == 1
    });
    if !steady {
        return;
    }
    for block in blocks.iter_mut() {
        for &slot in &slots {
            block.aht[slot] = true;
        }
    }
}

fn uncoupled(blocks: &mut [Block], nf: usize, full: usize) {
    for block in blocks.iter_mut() {
        block.cplstre = true;
        block.cplinu = false;
        block.chincpl = [false; MAX_FBW];
        block.cplcoe = [false; MAX_FBW];
        block.cplleake = false;
        for ch in 0..nf {
            block.start[ch] = 0;
            block.end[ch] = full;
        }
    }
}

/// Channels that would be coupled in this block. A short block leaves
/// the coupling, and coupling needs two channels.
fn coupled_channels(nf: usize, block: &Block) -> [bool; MAX_FBW] {
    let mut member = [false; MAX_FBW];
    for (on, &short) in member.iter_mut().zip(&block.blksw).take(nf) {
        *on = !short;
    }
    if member[..nf].iter().filter(|&&on| on).count() < 2 {
        member = [false; MAX_FBW];
    }
    member
}

/// E-AC-3 reference decoders reject a syncframe whose coupling use
/// changes between blocks. AC-3 carries the strategy in each block and
/// can change it.
fn eac3_coupling_stable(nf: usize, blocks: &[Block], mode: Mode) -> bool {
    if !matches!(mode, Mode::Adaptive) {
        return true;
    }
    let Some(first) = blocks.first() else {
        return true;
    };
    let member = coupled_channels(nf, first);
    blocks
        .iter()
        .all(|block| coupled_channels(nf, block) == member)
}

fn coupling(shape: &Shape, blocks: &mut [Block], mode: Mode) {
    let nf = shape.nf;
    let full = shape.end_bin();
    if let Some(spx) = shape.spx {
        uncoupled(blocks, nf, spx.begin_bin());
        return;
    }
    let Some((begf, endf)) = shape.coupling else {
        uncoupled(blocks, nf, full);
        return;
    };
    if shape.eac3 && !eac3_coupling_stable(nf, blocks, mode) {
        uncoupled(blocks, nf, full);
        return;
    }
    let begin = 37 + 12 * begf;
    let end = 37 + 12 * endf;
    let bands = Bands::new(&DEFAULT_CPL_BANDS, begf, endf);
    let mut previous: Option<([bool; MAX_FBW], bool)> = None;
    let mut previous_phase = false;
    // Band energies per block: [channel][band], then the coupling channel.
    let mut energy = vec![[[0.0f64; MAX_CPL_BANDS]; MAX_FBW + 1]; blocks.len()];
    let mut raw = vec![[[0.0f32; MAX_CPL_BANDS]; MAX_FBW]; blocks.len()];
    for (blk, block) in blocks.iter_mut().enumerate() {
        let mut chincpl = [false; MAX_FBW];
        if matches!(mode, Mode::Adaptive) {
            for (flag, &short) in chincpl[..nf].iter_mut().zip(&block.blksw[..nf]) {
                *flag = !short;
            }
        }
        let count = chincpl.iter().filter(|&&c| c).count();
        let cplinu = count >= 2;
        if !cplinu {
            chincpl = [false; MAX_FBW];
        }
        block.cplinu = cplinu;
        block.chincpl = chincpl;
        block.cplstre = match previous {
            None => true,
            Some((before, used)) => used != cplinu || before != chincpl,
        };
        previous = Some((chincpl, cplinu));
        block.cplleake = cplinu && block.cplstre;
        for (ch, &coupled) in chincpl[..nf].iter().enumerate() {
            block.start[ch] = 0;
            block.end[ch] = if coupled { begin } else { full };
        }
        block.start[CPL] = begin;
        block.end[CPL] = end;
        if !cplinu {
            block.cplcoe = [false; MAX_FBW];
            block.phsflginu = false;
            if blk > 0 && previous_phase {
                block.cplstre = true;
            }
            previous_phase = false;
            continue;
        }
        let gain = 1.0 / count as f32;
        let mut cpl = [0.0f32; COEFS];
        for ch in (0..nf).filter(|&ch| chincpl[ch]) {
            for (sum, &value) in cpl[begin..end].iter_mut().zip(&block.coefs[ch][begin..end]) {
                *sum += value * gain;
            }
        }
        // Section 7.4: a negative dot product in a stereo band is a phase
        // flag, and the coupling channel is summed with that channel negated.
        block.phsflg = [false; MAX_CPL_BANDS];
        block.phsflginu = false;
        if shape.acmod == 2 && chincpl[0] && chincpl[1] {
            let mut bin = begin;
            for (band, &size) in bands.sizes().iter().enumerate() {
                let dot: f64 = block.coefs[0][bin..bin + size]
                    .iter()
                    .zip(&block.coefs[1][bin..bin + size])
                    .map(|(&left, &right)| f64::from(left) * f64::from(right))
                    .sum();
                if dot < 0.0 {
                    block.phsflg[band] = true;
                    block.phsflginu = true;
                    for (sum, &value) in cpl[bin..bin + size]
                        .iter_mut()
                        .zip(&block.coefs[1][bin..bin + size])
                    {
                        *sum -= 2.0 * value * gain;
                    }
                }
                bin += size;
            }
        }
        if blk > 0 && block.phsflginu != previous_phase {
            block.cplstre = true;
        }
        previous_phase = block.phsflginu;
        block.coefs[CPL] = cpl;
        let mut bin = begin;
        for (band, &size) in bands.sizes().iter().enumerate() {
            for ch in (0..nf).filter(|&ch| chincpl[ch]) {
                energy[blk][ch][band] = sum_squares(&block.coefs[ch][bin..bin + size]);
            }
            energy[blk][MAX_FBW][band] = sum_squares(&cpl[bin..bin + size]);
            bin += size;
        }
        for ch in (0..nf).filter(|&ch| chincpl[ch]) {
            for band in 0..bands.count {
                raw[blk][ch][band] = coordinate(energy[blk][ch][band], energy[blk][MAX_FBW][band]);
            }
        }
    }
    // New coordinates where coupling starts or changes, or where they
    // drift from the last ones sent.
    for blk in 0..blocks.len() {
        let before = blk.checked_sub(1).map(|b| (&raw[b], blocks[b].chincpl));
        for (ch, current) in raw[blk].iter().enumerate().take(nf) {
            let block = &blocks[blk];
            let fresh = block.chincpl[ch]
                && match before {
                    None => true,
                    Some((previous, chincpl)) => {
                        let phase_changed = blocks[blk].phsflg[..bands.count]
                            != blocks[blk - 1].phsflg[..bands.count];
                        block.cplstre || !chincpl[ch] || phase_changed || {
                            let drift: f32 = current[..bands.count]
                                .iter()
                                .zip(&previous[ch][..bands.count])
                                .map(|(now, then)| (now - then).abs())
                                .sum();
                            drift / bands.count.max(1) as f32 > COORD_CHANGE
                        }
                    }
                };
            blocks[blk].cplcoe[ch] = fresh;
        }
    }
    // Each sent set codes the energy of the blocks that reuse it.
    for ch in 0..nf {
        let mut blk = 0;
        while blk < blocks.len() {
            if !blocks[blk].cplcoe[ch] {
                blk += 1;
                continue;
            }
            let mut ech = [0.0f64; MAX_CPL_BANDS];
            let mut ecpl = [0.0f64; MAX_CPL_BANDS];
            let mut last = blk;
            while last < blocks.len()
                && blocks[last].chincpl[ch]
                && (last == blk || !blocks[last].cplcoe[ch])
            {
                for band in 0..bands.count {
                    ech[band] += energy[last][ch][band];
                    ecpl[band] += energy[last][MAX_FBW][band];
                }
                last += 1;
            }
            let mut values = [0.0f32; MAX_CPL_BANDS];
            for band in 0..bands.count {
                values[band] = coordinate(ech[band], ecpl[band]);
            }
            blocks[blk].coords[ch] = quantize_coords(&values[..bands.count]);
            blk = last;
        }
    }
}

fn sum_squares(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

/// The coupling coordinate that restores a channel's band energy.
fn coordinate(channel: f64, coupled: f64) -> f32 {
    if coupled <= 0.0 {
        return 1.0;
    }
    (channel / coupled).sqrt() as f32
}

/// Section 7.4.3 in reverse: `cplco = 8 * mant * 2^-(exp + 3 master)`.
pub(crate) fn quantize_coords(values: &[f32]) -> Coords {
    let mut coords = Coords::default();
    let mut total = [24i32; MAX_CPL_BANDS];
    for (slot, &value) in total.iter_mut().zip(values) {
        let x = value / 8.0;
        if x > 0.0 {
            *slot = (-x.log2()).floor().clamp(0.0, 24.0) as i32;
        }
    }
    let used = &total[..values.len()];
    let (Some(&low), Some(&high)) = (used.iter().min(), used.iter().max()) else {
        return coords;
    };
    let mut master = ((high - 15 + 2) / 3).clamp(0, 3);
    while master > 0 && low < master * 3 {
        master -= 1;
    }
    coords.master = master as u8;
    for (band, &value) in values.iter().enumerate() {
        let x = f64::from(value / 8.0).min(31.0 / 32.0);
        let exp = (total[band] - 3 * master).clamp(0, 15);
        let scale = f64::from(exp + 3 * master).exp2();
        let mant = if exp == 15 {
            (x * scale * 16.0).round()
        } else {
            (x * scale * 32.0 - 16.0).round()
        };
        coords.exp[band] = exp as u8;
        coords.mant[band] = mant.clamp(0.0, 15.0) as u8;
    }
    coords
}

/// The decoder's value of one quantized coordinate.
#[cfg(test)]
pub(crate) fn coord_value(coords: &Coords, band: usize) -> f32 {
    let exp = u32::from(coords.exp[band]);
    let mant = f32::from(coords.mant[band]);
    let temp = if exp == 15 {
        mant / 16.0
    } else {
        (mant + 16.0) / 32.0
    };
    8.0 * temp * (-((exp + 3 * u32::from(coords.master)) as f32)).exp2()
}

/// Coordinates that scale the copied bins up to the original band energy.
fn spectral(shape: &Shape, blocks: &mut [Block]) {
    let Some(spx) = shape.spx else {
        return;
    };
    let range = spx.range();
    let count = range.bands.count;
    let mut previous = [[(0u8, 0u8); crate::spx::MAX_BANDS]; MAX_FBW];
    let mut previous_master = [0u8; MAX_FBW];
    let mut have_previous = false;
    for block in blocks.iter_mut() {
        for ch in 0..shape.nf {
            let gains = band_gains(&block.coefs[ch], &range);
            let (master, coded) = crate::spx::quantize(&gains[..count]);
            let mut same = have_previous && master == previous_master[ch];
            block.spx_master[ch] = master;
            previous_master[ch] = master;
            for band in 0..count {
                block.spx_exp[ch][band] = coded[band].0;
                block.spx_mant[ch][band] = coded[band].1;
                same = same && previous[ch][band] == coded[band];
                previous[ch][band] = coded[band];
            }
            block.spx_send[ch] = !same;
        }
        have_previous = true;
    }
}

/// Energy of the original band over the energy of the bins that will be copied into it.
fn band_gains(coefs: &[f32; COEFS], range: &crate::spx::Range) -> [f32; crate::spx::MAX_BANDS] {
    let copy_start = range.copy_start();
    let copy_end = range.begin_bin();
    let mut copied = [0.0f32; COEFS];
    let mut copy = copy_start;
    let mut insert = copy_end;
    for &size in range.bands.sizes() {
        for _ in 0..size {
            if copy == copy_end {
                copy = copy_start;
            }
            if let (Some(&value), Some(slot)) = (coefs.get(copy), copied.get_mut(insert)) {
                *slot = value;
            }
            insert += 1;
            copy += 1;
        }
    }
    let mut gains = [0.0f32; crate::spx::MAX_BANDS];
    let mut bin = copy_end;
    for (band, &size) in range.bands.sizes().iter().enumerate() {
        let energy = |samples: &[f32]| -> f32 {
            let total: f32 = samples.iter().map(|value| value * value).sum();
            (total / size as f32).sqrt()
        };
        let from = copied.get(bin..bin + size).unwrap_or(&[]);
        let to = coefs.get(bin..bin + size).unwrap_or(&[]);
        let source = energy(from);
        gains[band] = if source > 1.0e-8 {
            energy(to) / source
        } else {
            0.0
        };
        bin += size;
    }
    gains
}

/// Annex E 3.3.2, which reduces to Tables 7.25 to 7.28 for AC-3.
pub(crate) fn rematrix_bands(block: &Block, shape: &Shape) -> usize {
    match (block.cplinu, shape.coupling) {
        (true, Some((0, _))) => 2,
        (true, Some((1 | 2, _))) => 3,
        _ => 4,
    }
}

fn rematrix(shape: &Shape, blocks: &mut [Block]) {
    if shape.acmod != 2 {
        return;
    }
    let mut previous: Option<(usize, [bool; 4])> = None;
    for block in blocks.iter_mut() {
        let count = rematrix_bands(block, shape);
        let end = block.end[0].min(block.end[1]);
        let mut flags = [false; 4];
        for (band, flag) in flags.iter_mut().enumerate().take(count) {
            let (lo, hi) = (REMAT_BANDS[band], REMAT_BANDS[band + 1].min(end));
            if lo >= hi {
                continue;
            }
            let mut sums = [0.0f64; 4];
            for bin in lo..hi {
                let (l, r) = (
                    f64::from(block.coefs[0][bin]),
                    f64::from(block.coefs[1][bin]),
                );
                sums[0] += l * l;
                sums[1] += r * r;
                sums[2] += (l + r) * (l + r) / 4.0;
                sums[3] += (l - r) * (l - r) / 4.0;
            }
            *flag = sums[2].min(sums[3]) < sums[0].min(sums[1]);
        }
        block.rematstr = match previous {
            None => true,
            Some((n, before)) => n != count || before != flags,
        };
        previous = Some((count, flags));
        block.nrematbd = count;
        block.rematflg = flags;
        for band in 0..count {
            if !flags[band] {
                continue;
            }
            let (lo, hi) = (REMAT_BANDS[band], REMAT_BANDS[band + 1].min(end));
            for bin in lo..hi {
                let (l, r) = (block.coefs[0][bin], block.coefs[1][bin]);
                block.coefs[0][bin] = (l + r) * 0.5;
                block.coefs[1][bin] = (l - r) * 0.5;
            }
        }
    }
}

/// The exponent of one coefficient: the left shift that brings it into
/// [0.5, 1), at most 24.
pub(crate) fn exponent_of(value: f32) -> u8 {
    let biased = ((value.abs().to_bits() >> 23) & 0xff) as i32;
    if biased == 0 {
        return 24;
    }
    (126 - biased).clamp(0, 24) as u8
}

/// Strategy for a run of `len` blocks sharing one set (as FFmpeg's and
/// the section 8.2.8 basic encoder): coarser sets for shorter runs.
fn strategy_for(blocks: usize, len: usize) -> u8 {
    match blocks {
        6 => match len {
            1 => D45,
            2 | 3 => D25,
            _ => D15,
        },
        3 => {
            if len < 3 {
                D25
            } else {
                D15
            }
        }
        // A one- or two-block syncframe cannot amortize a fine set.
        2 => {
            if len < 2 {
                D45
            } else {
                D25
            }
        }
        _ => D45,
    }
}

fn exponents(shape: &Shape, blocks: &mut [Block], mode: Mode) {
    let count = blocks.len();
    let mut raw = vec![[0u8; COEFS]; count];
    let slots: Vec<usize> = (0..shape.nf)
        .chain(shape.coupling.map(|_| CPL))
        .chain(shape.lfeon.then_some(LFE))
        .collect();
    for slot in slots {
        let mut fresh = [false; MAX_BLOCKS];
        for (blk, block) in blocks.iter().enumerate() {
            let (start, end) = if slot == LFE {
                (0, 7)
            } else {
                (block.start[slot], block.end[slot])
            };
            for (value, &coef) in raw[blk][start..end]
                .iter_mut()
                .zip(&block.coefs[slot][start..end])
            {
                *value = exponent_of(coef);
            }
        }
        for blk in 0..count {
            let block = &blocks[blk];
            if slot == CPL && !block.cplinu {
                continue;
            }
            fresh[blk] = if matches!(mode, Mode::Coarse) {
                blk == 0
            } else {
                blk == 0
                    || match slot {
                        LFE => false,
                        CPL => !blocks[blk - 1].cplinu || block.cplstre,
                        ch => {
                            block.end[ch] != blocks[blk - 1].end[ch]
                                || block.blksw[ch]
                                || blocks[blk - 1].blksw[ch]
                        }
                    }
                    || (slot != LFE && {
                        let (start, end) = (block.start[slot], block.end[slot]);
                        let change: u32 = (start..end)
                            .map(|bin| u32::from(raw[blk][bin].abs_diff(raw[blk - 1][bin])))
                            .sum();
                        change > EXP_CHANGE * (end - start) as u32
                    })
            };
        }
        let mut blk = 0;
        while blk < count {
            if !fresh[blk] {
                blocks[blk].expstr[slot] = REUSE;
                blk += 1;
                continue;
            }
            let mut last = blk + 1;
            while last < count && !fresh[last] && (slot != CPL || blocks[last].cplinu) {
                last += 1;
            }
            let strategy = if slot == LFE {
                D15
            } else {
                match mode {
                    Mode::Coarse => D45,
                    Mode::Adaptive => strategy_for(shape.blocks, last - blk),
                }
            };
            let (start, end) = if slot == LFE {
                (0, 7)
            } else {
                (blocks[blk].start[slot], blocks[blk].end[slot])
            };
            let mut least = [24u8; COEFS];
            for row in &raw[blk..last] {
                for bin in start..end {
                    least[bin] = least[bin].min(row[bin]);
                }
            }
            let mut decoded = [0u8; COEFS];
            if slot == CPL {
                shape_cpl(&least, start, end, strategy, &mut decoded);
            } else {
                shape_fbw(&least, end, strategy, &mut decoded);
            }
            blocks[blk].expstr[slot] = strategy;
            for block in &mut blocks[blk..last] {
                block.exps[slot] = decoded;
            }
            for block in &mut blocks[blk + 1..last] {
                block.expstr[slot] = REUSE;
            }
            blk = last;
        }
        if slot == LFE {
            for block in blocks.iter_mut() {
                block.start[LFE] = 0;
                block.end[LFE] = 7;
            }
        }
    }
}

/// Section 8.2.10 for a full bandwidth or LFE channel: share one
/// exponent per group, keep each step within +-2, and expand as the
/// decoder will. `out[1 + i * size]` is the value of group position `i`.
pub(crate) fn shape_fbw(raw: &[u8; COEFS], end: usize, strategy: u8, out: &mut [u8; COEFS]) {
    let size = group_size(strategy);
    let positions = 3 * fbw_groups(strategy, end);
    let mut values = [24u8; 1 + 3 * 84];
    values[0] = raw[0].min(15);
    for (i, value) in values[1..=positions].iter_mut().enumerate() {
        let lo = 1 + i * size;
        let hi = (lo + size).min(end);
        if lo < hi {
            *value = raw[lo..hi].iter().copied().min().unwrap_or(24);
        }
    }
    limit_steps(&mut values[..=positions]);
    out[0] = values[0];
    for (i, &value) in values[1..=positions].iter().enumerate() {
        let lo = (1 + i * size).min(COEFS);
        out[lo..(lo + size).min(COEFS)].fill(value);
    }
}

/// The coupling channel's version: no bin carries the reference, which is
/// the first value rounded down to even.
pub(crate) fn shape_cpl(
    raw: &[u8; COEFS],
    start: usize,
    end: usize,
    strategy: u8,
    out: &mut [u8; COEFS],
) {
    let size = group_size(strategy);
    let positions = 3 * cpl_groups(strategy, start, end);
    let mut values = [24u8; 1 + 3 * 72];
    for (i, value) in values[1..=positions].iter_mut().enumerate() {
        let lo = start + i * size;
        let hi = (lo + size).min(end);
        if lo < hi {
            *value = raw[lo..hi].iter().copied().min().unwrap_or(24);
        }
    }
    values[0] = values[1] & !1;
    limit_steps(&mut values[..=positions]);
    values[0] = values[1] & !1;
    for (i, &value) in values[1..=positions].iter().enumerate() {
        let lo = (start + i * size).min(COEFS);
        out[lo..(lo + size).min(COEFS)].fill(value);
    }
}

/// Lower values until neighbours differ by at most 2.
fn limit_steps(values: &mut [u8]) {
    for i in 1..values.len() {
        values[i] = values[i].min(values[i - 1].saturating_add(2));
    }
    for i in (0..values.len().saturating_sub(1)).rev() {
        values[i] = values[i].min(values[i + 1].saturating_add(2));
    }
}

/// AC-3 sends SNR offsets in block 0 and wherever coupling starts, so the
/// coupling channel always has its own.
fn snr_flags(shape: &Shape, blocks: &mut [Block]) {
    if shape.eac3 {
        return;
    }
    let mut had_cpl = false;
    for (blk, block) in blocks.iter_mut().enumerate() {
        block.snroffste = blk == 0 || (block.cplinu && !had_cpl);
        had_cpl = block.cplinu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(acmod: u8, nf: usize, coupling: Option<(usize, usize)>) -> Shape {
        Shape {
            eac3: false,
            acmod,
            lfeon: false,
            nf,
            blocks: 6,
            fscod: 0,
            bandwidth: 48,
            coupling,
            spx: None,
        }
    }

    #[test]
    fn exponents_bring_each_value_into_half_to_one() {
        for value in [0.9f32, 0.5, 0.49, 0.25, 1e-3, -0.7, 3e-8] {
            let e = exponent_of(value);
            let m = value.abs() * (e as f32).exp2();
            assert!((0.5..1.0).contains(&m), "{value} {e} {m}");
        }
        assert_eq!(exponent_of(0.0), 24);
        assert_eq!(exponent_of(1.5), 0);
        assert_eq!(exponent_of(1e-12), 24);
    }

    #[test]
    fn shaped_exponents_never_exceed_the_raw_and_step_by_two() {
        let mut raw = [0u8; COEFS];
        for (bin, value) in raw.iter_mut().enumerate() {
            *value = [3, 20, 7, 24, 0, 18, 9, 12][bin % 8];
        }
        for strategy in [D15, D25, D45] {
            let mut out = [0u8; COEFS];
            shape_fbw(&raw, 229, strategy, &mut out);
            assert!(out[0] <= 15);
            for bin in 0..229 {
                assert!(out[bin] <= raw[bin], "{strategy} {bin}");
            }
            let size = group_size(strategy);
            let mut previous = out[0];
            for i in 0..3 * fbw_groups(strategy, 229) {
                let value = out[1 + i * size];
                assert!(value.abs_diff(previous) <= 2);
                previous = value;
            }
        }
    }

    #[test]
    fn coupling_coordinates_round_trip_within_a_step() {
        for value in [0.01f32, 0.2, 0.5, 1.0, 1.9, 4.0, 7.5] {
            let coords = quantize_coords(&[value, value * 0.5, value * 0.03]);
            for (band, expected) in [value, value * 0.5, value * 0.03].into_iter().enumerate() {
                let got = coord_value(&coords, band);
                let db = 20.0 * (got / expected).log10();
                assert!(db.abs() < 0.6, "{value} band {band}: {got} {db}");
            }
        }
    }

    #[test]
    fn transients_take_a_channel_out_of_coupling() {
        let shape = shape(7, 5, Some((6, 15)));
        let mut blocks: Vec<Block> = (0..6).map(|_| Block::default()).collect();
        for block in &mut blocks {
            for (ch, row) in block.coefs[..5].iter_mut().enumerate() {
                for (bin, coef) in row[..253].iter_mut().enumerate() {
                    *coef = 0.01 * ((bin + ch) % 7) as f32;
                }
            }
        }
        blocks[3].blksw[2] = true;
        plan(&shape, &mut blocks, Mode::Adaptive);
        assert!(blocks.iter().all(|b| b.cplinu));
        assert!(!blocks[3].chincpl[2] && blocks[2].chincpl[2] && blocks[4].chincpl[2]);
        assert!(blocks[3].cplstre && blocks[4].cplstre && !blocks[2].cplstre);
        assert_eq!(blocks[3].end[2], shape.end_bin());
        assert_eq!(blocks[2].end[2], 37 + 12 * 6);
        assert!(blocks[3].expstr[2] != REUSE && blocks[4].expstr[2] != REUSE);
        assert!(blocks[0].cplcoe[..5].iter().all(|&c| c));
        assert!(blocks[4].cplcoe[0]);
    }

    #[test]
    fn eac3_coupling_stays_off_when_it_would_start_late() {
        let mut shape = shape(2, 2, Some((6, 15)));
        shape.eac3 = true;
        let mut blocks: Vec<Block> = (0..6).map(|_| Block::default()).collect();
        for block in &mut blocks {
            for row in &mut block.coefs[..2] {
                for coef in &mut row[..253] {
                    *coef = 0.01;
                }
            }
        }
        blocks[0].blksw[0] = true;
        plan(&shape, &mut blocks, Mode::Adaptive);
        assert!(blocks.iter().all(|block| !block.cplinu));
        assert!(blocks.iter().all(|block| block.end[0] == shape.end_bin()));
    }

    #[test]
    fn eac3_couples_every_block_when_none_are_short() {
        let mut shape = shape(2, 2, Some((6, 15)));
        shape.eac3 = true;
        let mut blocks: Vec<Block> = (0..6).map(|_| Block::default()).collect();
        for block in &mut blocks {
            for row in &mut block.coefs[..2] {
                for coef in &mut row[..253] {
                    *coef = 0.01;
                }
            }
        }
        plan(&shape, &mut blocks, Mode::Adaptive);
        assert!(blocks.iter().all(|block| block.cplinu && block.chincpl[0]));
    }

    #[test]
    fn rematrixing_codes_sum_and_difference_of_a_centred_image() {
        let shape = shape(2, 2, None);
        let mut blocks: Vec<Block> = (0..6).map(|_| Block::default()).collect();
        for block in &mut blocks {
            for bin in 0..253 {
                let v = 0.1 * ((bin % 5) as f32 - 2.0);
                block.coefs[0][bin] = v;
                block.coefs[1][bin] = v * 0.99;
            }
        }
        plan(&shape, &mut blocks, Mode::Adaptive);
        assert_eq!(blocks[0].nrematbd, 4);
        assert!(blocks[0].rematflg.iter().all(|&f| f));
        assert!(blocks[0].rematstr && !blocks[1].rematstr);
        let bin = 20;
        assert!(blocks[0].coefs[1][bin].abs() < 0.01 * blocks[0].coefs[0][bin].abs() + 1e-6);
    }

    #[test]
    fn steady_exponents_are_sent_once_in_d15() {
        let shape = shape(1, 1, None);
        let mut blocks: Vec<Block> = (0..6).map(|_| Block::default()).collect();
        for block in &mut blocks {
            for (bin, coef) in block.coefs[0][..253].iter_mut().enumerate() {
                *coef = 0.3 / (1 + bin) as f32;
            }
        }
        plan(&shape, &mut blocks, Mode::Adaptive);
        assert_eq!(blocks[0].expstr[0], D15);
        assert!(blocks[1..].iter().all(|b| b.expstr[0] == REUSE));
        assert!(blocks.iter().all(|b| b.exps[0] == blocks[0].exps[0]));
    }
}
