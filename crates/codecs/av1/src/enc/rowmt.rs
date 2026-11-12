//! Row-based multithreading of the tile coding.
//!
//! The superblock rows of each tile are decided in a wavefront: the row
//! below starts a superblock once the row above has finished the one above
//! and to the right of it, which is all of the row above that the
//! superblock's syntax, prediction and mode search look at. Each worker
//! thread codes with its own copy of the frame state and picture and copies
//! in what the row above published after each superblock (its mode info,
//! samples and above contexts).
//!
//! A row cannot know the CDFs the rows above leave, so its rate estimates
//! adapt a copy of the CDFs the row above had after its second superblock
//! (the first row of a tile starts from the frame's CDFs), and its symbols
//! are logged by value. The tile's bitstream is then coded from the logs
//! with the real CDFs, row after row. Every row is decided the same way
//! whatever the number of threads, so the bitstream does not depend on it.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use cutlass_decode::CodecError;

use crate::dec::cdf::{CdfContext, pack4, unpack4};
use crate::dec::coder::{Coder, masked_prob};
use crate::dec::frame::{FrameState, Picture};
use crate::dec::obu::{FrameHeader, SequenceHeader};
use crate::dec::tile::{EncPlan, RefView, TileDec};
use crate::dec::types::MiInfo;
use crate::dsp::Pixel;

use super::msac_enc::{
    COST_SHIFT, CoderMark, MsacEnc, adapt_bool, adapt_symbol, bool_cost, symbol_cost,
};
use super::quant::QuantParams;
use super::rdo::ModeParams;

type R<T> = Result<T, CodecError>;

fn err(what: &'static str) -> CodecError {
    CodecError::InvalidApi { what }
}

// ---------------------------------------------------------------------------
// CDF layout

/// Views of nested CDF arrays as one flat slice.
trait Flat<E> {
    fn flat(&mut self) -> &mut [E];
}

macro_rules! flat_impls {
    ($e:ty) => {
        impl<const A: usize> Flat<$e> for [$e; A] {
            fn flat(&mut self) -> &mut [$e] {
                self
            }
        }
        impl<const A: usize, const B: usize> Flat<$e> for [[$e; A]; B] {
            fn flat(&mut self) -> &mut [$e] {
                self.as_flattened_mut()
            }
        }
        impl<const A: usize, const B: usize, const C: usize> Flat<$e> for [[[$e; A]; B]; C] {
            fn flat(&mut self) -> &mut [$e] {
                self.as_flattened_mut().as_flattened_mut()
            }
        }
        impl<const A: usize, const B: usize, const C: usize, const D: usize> Flat<$e>
            for [[[[$e; A]; B]; C]; D]
        {
            fn flat(&mut self) -> &mut [$e] {
                self.as_flattened_mut()
                    .as_flattened_mut()
                    .as_flattened_mut()
            }
        }
    };
}
flat_impls!(u16);
flat_impls!(u64);

macro_rules! cdf_arrays {
    ($c:expr; $($f:ident),* ; $($p:ident),*) => {
        (vec![$(Flat::<u16>::flat(&mut $c.$f)),*], vec![$(Flat::<u64>::flat(&mut $c.$p)),*])
    };
}

/// Every CDF array of `c`: the 16-bit ones, then the packed ones (see
/// [`pack4`]).
#[allow(clippy::type_complexity)]
fn cdf_arrays(c: &mut CdfContext) -> (Vec<&mut [u16]>, Vec<&mut [u64]>) {
    cdf_arrays!(c;
        intra_frame_y_mode, y_mode, uv_mode_cfl_not_allowed, uv_mode_cfl_allowed,
        angle_delta, intrabc, partition_w8, partition_w16, partition_w32, partition_w64,
        partition_w128, tx_8x8, tx_16x16, tx_32x32, tx_64x64, txfm_split, filter_intra_mode,
        filter_intra, segment_id, segment_id_predicted, mv_class0_hp, mv_hp, mv_sign, mv_bit,
        mv_class0_bit, new_mv, zero_mv, ref_mv, drl_mode, is_inter, comp_mode, skip_mode, skip,
        comp_ref, comp_bwd_ref, single_ref, compound_mode, interp_filter, motion_mode, mv_joint,
        mv_class, mv_class0_fr, mv_fr, palette_y_size, palette_uv_size, palette_size_2_y_color,
        palette_size_3_y_color, palette_size_4_y_color, palette_size_5_y_color,
        palette_size_6_y_color, palette_size_7_y_color, palette_size_8_y_color,
        palette_size_2_uv_color, palette_size_3_uv_color, palette_size_4_uv_color,
        palette_size_5_uv_color, palette_size_6_uv_color, palette_size_7_uv_color,
        palette_size_8_uv_color, palette_y_mode, palette_uv_mode, delta_q, delta_lf,
        intra_tx_type_set1, intra_tx_type_set2, inter_tx_type_set1, inter_tx_type_set2,
        inter_tx_type_set3, compound_idx, comp_group_idx, compound_type, inter_intra,
        inter_intra_mode, wedge_index, wedge_inter_intra, use_obmc, comp_ref_type, uni_comp_ref,
        cfl_sign, cfl_alpha, use_wiener, use_sgrproj, restoration_type, txb_skip, eob_pt_16,
        eob_pt_32, eob_pt_64, eob_pt_128, eob_pt_256, eob_pt_512, eob_pt_1024, eob_extra,
        dc_sign, coeff_base_eob, delta_lf_multi;
        coeff_base, coeff_br)
}

/// Where each CDF array lies inside a [`CdfContext`]: for every 16-bit
/// unit of the context, the index of the array holding it in
/// [`cdf_arrays`] (the 16-bit arrays, then the packed ones; `u8::MAX` for
/// padding), and each array's byte offset and length.
#[derive(Clone, Debug, Default)]
pub struct CdfLayout {
    unit: Vec<u8>,
    arrays: Vec<(usize, usize)>,
    n_short: usize,
}

impl CdfLayout {
    pub fn new(c: &mut CdfContext) -> Self {
        let base = std::ptr::from_mut(c) as usize;
        let (s, p) = cdf_arrays(c);
        let n_short = s.len();
        let arrays: Vec<(usize, usize)> = s
            .iter()
            .map(|a| (a.as_ptr() as usize - base, a.len() * 2))
            .chain(p.iter().map(|a| (a.as_ptr() as usize - base, a.len() * 8)))
            .collect();
        let mut unit = vec![u8::MAX; size_of::<CdfContext>() / 2];
        for (i, &(o, bytes)) in arrays.iter().enumerate() {
            for u in &mut unit[o / 2..(o + bytes) / 2] {
                *u = i as u8;
            }
        }
        Self {
            unit,
            arrays,
            n_short,
        }
    }

    /// The array (index among its kind) and element at byte offset `off`
    /// of `n` elements of `size` bytes.
    #[inline(always)]
    fn find(&self, off: usize, n: usize, size: usize, packed: bool) -> Option<(u8, u16)> {
        let a = usize::from(*self.unit.get(off / 2)?);
        let &(o, bytes) = self.arrays.get(a)?;
        let d = off - o;
        let ok = (a >= self.n_short) == packed && d.is_multiple_of(size) && d + n * size <= bytes;
        let a = if packed { a - self.n_short } else { a };
        ok.then_some((a as u8, (d / size) as u16))
    }
}

// ---------------------------------------------------------------------------
// Symbol log

/// One coded syntax element, by value. CDFs are named by array index and
/// element offset in [`cdf_arrays`].
#[derive(Clone, Copy, Debug)]
pub enum SymOp {
    Symbol {
        a: u8,
        at: u16,
        n: u8,
        v: u8,
    },
    Packed {
        a: u8,
        at: u16,
        v: u8,
    },
    BoolAdapt {
        a: u8,
        at: u16,
        v: bool,
    },
    Masked {
        a: u8,
        at: u16,
        n: u8,
        mask: u16,
        v: bool,
    },
    Prob {
        f: u32,
        v: bool,
    },
    Literal {
        n: u8,
        v: u32,
    },
    Golomb {
        v: u32,
    },
    Ns {
        n: u32,
        v: u32,
    },
    Mark(u32),
}

/// A [`Coder`] that adapts the CDFs it is given (for rate estimates) and
/// logs every element, to be coded later with other CDFs.
pub struct SymLog<'l> {
    /// Address of the CDF context the tile code uses.
    base: usize,
    layout: &'l CdfLayout,
    allow_update: bool,
    ops: Vec<SymOp>,
    /// Set when a CDF outside the context was used.
    bad: bool,
    /// The cost of the elements so far, in bits (for closed-loop trials).
    bits: f64,
}

/// A saved [`SymLog`] state: the elements logged after `from` and the cost.
#[derive(Clone)]
pub struct LogMark {
    from: usize,
    tail: Vec<SymOp>,
    bits: f64,
    bad: bool,
}

impl<'l> SymLog<'l> {
    fn new(cdf: &CdfContext, layout: &'l CdfLayout, allow_update: bool) -> Self {
        Self {
            base: std::ptr::from_ref(cdf) as usize,
            layout,
            allow_update,
            ops: Vec::new(),
            bad: false,
            bits: 0.0,
        }
    }

    #[inline(always)]
    fn locate<E>(&mut self, p: *const E, n: usize, packed: bool) -> (u8, u16) {
        let off = (p as usize).wrapping_sub(self.base);
        match self.layout.find(off, n, size_of::<E>(), packed) {
            Some(x) => x,
            None => {
                self.bad = true;
                (0, 0)
            }
        }
    }
}

impl Coder for SymLog<'_> {
    const WRITE: bool = true;
    #[inline(always)]
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], v: bool) -> bool {
        let (a, at) = self.locate(cdf.as_ptr(), 2, false);
        self.ops.push(SymOp::BoolAdapt { a, at, v });
        self.cost(bool_cost(cdf[0], v));
        if self.allow_update {
            adapt_bool(cdf, v);
        }
        v
    }
    #[inline(always)]
    fn bool_prob(&mut self, f: u32, v: bool) -> bool {
        self.ops.push(SymOp::Prob { f, v });
        self.cost(bool_cost(f as u16, v));
        v
    }
    #[inline(always)]
    fn bool_masked<const N: usize>(&mut self, cdf: &[u16; N], mask: u16, v: bool) -> bool {
        let (a, at) = self.locate(cdf.as_ptr(), N, false);
        self.ops.push(SymOp::Masked {
            a,
            at,
            n: N as u8,
            mask,
            v,
        });
        self.cost(bool_cost(masked_prob(cdf, mask) as u16, v));
        v
    }
    #[inline(always)]
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], v: usize) -> usize {
        let v = v.min(N - 1);
        let (a, at) = self.locate(cdf.as_ptr(), N, false);
        self.ops.push(SymOp::Symbol {
            a,
            at,
            n: N as u8,
            v: v as u8,
        });
        self.cost(symbol_cost(cdf, v));
        if self.allow_update {
            adapt_symbol(cdf, v);
        }
        v
    }
    #[inline(always)]
    fn symbol_packed(&mut self, cdf: &mut u64, v: usize) -> usize {
        let v = v.min(3);
        let (a, at) = self.locate(std::ptr::from_mut(cdf).cast_const(), 1, true);
        self.ops.push(SymOp::Packed { a, at, v: v as u8 });
        let mut c = unpack4(*cdf);
        self.cost(symbol_cost(&c, v));
        if self.allow_update {
            adapt_symbol(&mut c, v);
            *cdf = pack4(c);
        }
        v
    }
    #[inline(always)]
    fn literal(&mut self, n: u32, v: u32) -> u32 {
        let v = if n >= 32 { v } else { v & ((1u32 << n) - 1) };
        self.ops.push(SymOp::Literal { n: n as u8, v });
        self.bits += f64::from(n);
        v
    }
    #[inline(always)]
    fn golomb(&mut self, v: u32) -> u32 {
        self.ops.push(SymOp::Golomb { v });
        self.bits += f64::from(2 * (32 - (v + 1).leading_zeros()) - 1);
        v
    }
    #[inline(always)]
    fn ns(&mut self, n: u32, v: u32) -> u32 {
        let v = if n == 0 { 0 } else { v.min(n - 1) };
        self.ops.push(SymOp::Ns { n, v });
        self.bits += f64::from(32 - n.leading_zeros());
        v
    }
    #[inline(always)]
    fn mark(&mut self, tag: u32) {
        self.ops.push(SymOp::Mark(tag));
    }
    fn checkpoint(&self, base: Option<&CoderMark>) -> Option<CoderMark> {
        let from = match base {
            Some(CoderMark::Log(b)) => b.from,
            Some(_) => return None,
            None => self.ops.len(),
        };
        Some(CoderMark::Log(LogMark {
            from,
            tail: self.ops[from.min(self.ops.len())..].to_vec(),
            bits: self.bits,
            bad: self.bad,
        }))
    }
    fn restore(&mut self, s: &CoderMark) {
        if let CoderMark::Log(s) = s {
            self.ops.truncate(s.from);
            self.ops.extend_from_slice(&s.tail);
            self.bits = s.bits;
            self.bad = s.bad;
        }
    }
    fn bits_written(&self) -> f64 {
        self.bits
    }
}

impl SymLog<'_> {
    #[inline(always)]
    fn cost(&mut self, c: u32) {
        self.bits += f64::from(c) / f64::from(1u32 << COST_SHIFT);
    }
}

/// Codes `ops` into `e` with the CDF arrays `short` / `packed` of
/// [`cdf_arrays`] (which adapt as they are used). False when an operation
/// names a CDF that does not exist.
fn replay(
    ops: &[SymOp],
    short: &mut [&mut [u16]],
    packed: &mut [&mut [u64]],
    e: &mut MsacEnc,
) -> bool {
    macro_rules! sized {
        ($s:expr, $n:expr, $k:ident => $body:expr) => {
            match $n {
                2 => sized!(@ $s, 2, $k => $body),
                3 => sized!(@ $s, 3, $k => $body),
                4 => sized!(@ $s, 4, $k => $body),
                5 => sized!(@ $s, 5, $k => $body),
                6 => sized!(@ $s, 6, $k => $body),
                7 => sized!(@ $s, 7, $k => $body),
                8 => sized!(@ $s, 8, $k => $body),
                9 => sized!(@ $s, 9, $k => $body),
                10 => sized!(@ $s, 10, $k => $body),
                11 => sized!(@ $s, 11, $k => $body),
                12 => sized!(@ $s, 12, $k => $body),
                13 => sized!(@ $s, 13, $k => $body),
                14 => sized!(@ $s, 14, $k => $body),
                15 => sized!(@ $s, 15, $k => $body),
                16 => sized!(@ $s, 16, $k => $body),
                _ => return false,
            }
        };
        (@ $s:expr, $n:literal, $k:ident => $body:expr) => {{
            let Ok($k) = <&mut [u16; $n]>::try_from($s) else {
                return false;
            };
            $body
        }};
    }
    for &op in ops {
        match op {
            SymOp::Symbol { a, at, n, v } => {
                let at = usize::from(at);
                let Some(s) = short
                    .get_mut(usize::from(a))
                    .and_then(|s| s.get_mut(at..at + usize::from(n)))
                else {
                    return false;
                };
                sized!(s, n, c => e.symbol(c, usize::from(v)));
            }
            SymOp::BoolAdapt { a, at, v } => {
                let at = usize::from(at);
                let Some(s) = short
                    .get_mut(usize::from(a))
                    .and_then(|s| s.get_mut(at..at + 2))
                else {
                    return false;
                };
                let Ok(c) = <&mut [u16; 2]>::try_from(s) else {
                    return false;
                };
                e.bool_adapt(c, v);
            }
            SymOp::Masked { a, at, n, mask, v } => {
                let at = usize::from(at);
                let Some(s) = short
                    .get_mut(usize::from(a))
                    .and_then(|s| s.get_mut(at..at + usize::from(n)))
                else {
                    return false;
                };
                sized!(s, n, c => e.bool_prob(masked_prob(c, mask), v));
            }
            SymOp::Packed { a, at, v } => {
                let Some(p) = packed
                    .get_mut(usize::from(a))
                    .and_then(|s| s.get_mut(usize::from(at)))
                else {
                    return false;
                };
                let mut c = unpack4(*p);
                e.symbol(&mut c, usize::from(v));
                *p = pack4(c);
            }
            SymOp::Prob { f, v } => e.bool_prob(f, v),
            SymOp::Literal { n, v } => e.literal(u32::from(n), v),
            SymOp::Golomb { v } => e.golomb(v),
            SymOp::Ns { n, v } => e.ns(n, v),
            SymOp::Mark(tag) => e.mark(tag),
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Superblock snapshots

/// What one superblock passes on: the CDFs and coder state it left and,
/// when other workers need them, its share of the frame state.
struct SbSnap<P> {
    /// The worker that coded it.
    by: usize,
    /// Its share of the frame state, when other workers need it.
    data: Option<SbData<P>>,
    /// The above contexts it left (a worker's own may have been cleared
    /// since by another tile's first row in the same columns).
    above: SbAbove,
    /// The coder's q index and loop filter deltas after the superblock.
    q_state: (i32, [i8; 4]),
    /// The CDFs after this superblock, for the row below.
    cdf: Option<Box<CdfContext>>,
}

/// What coding one superblock left in the frame state and the picture (the
/// superblock clipped to its tile).
struct SbData<P> {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    mi: Vec<MiInfo>,
    /// uv_modes, tx_types, inter_tx_sizes, segment_ids, row by row.
    bytes: [Vec<u8>; 4],
    lf_tx: [Vec<u8>; 3],
    px: Vec<Vec<P>>,
    cdef: Vec<i8>,
}

/// The above contexts in a superblock's columns.
struct SbAbove {
    c0: usize,
    /// above_level and above_dc per plane, above_seg_pred.
    above: [Vec<u8>; 7],
    above_palette: [Vec<[u16; 8]>; 2],
}

impl SbAbove {
    fn capture<P: Pixel, C: Coder>(td: &TileDec<'_, P, C>, c0: usize) -> Self {
        let mut above: [Vec<u8>; 7] = Default::default();
        for p in 0..3 {
            let cols = above_cols(c0, td.sb_size4, p > 0 && td.sub_x, td.above_level[p].len());
            above[p] = td.above_level[p][cols.clone()].to_vec();
            above[3 + p] = td.above_dc[p][cols].to_vec();
        }
        let cols = above_cols(c0, td.sb_size4, false, td.above_seg_pred.len());
        above[6] = td.above_seg_pred[cols].to_vec();
        let above_palette = std::array::from_fn(|i| {
            let cols = above_cols(c0, td.sb_size4, false, td.above_palette[i].len());
            td.above_palette[i][cols].to_vec()
        });
        Self {
            c0,
            above,
            above_palette,
        }
    }

    fn apply_to<P: Pixel, C: Coder>(&self, td: &mut TileDec<'_, P, C>) {
        for p in 0..3 {
            let cols = above_cols(
                self.c0,
                td.sb_size4,
                p > 0 && td.sub_x,
                td.above_level[p].len(),
            );
            td.above_level[p][cols.clone()].copy_from_slice(&self.above[p]);
            td.above_dc[p][cols].copy_from_slice(&self.above[3 + p]);
        }
        let cols = above_cols(self.c0, td.sb_size4, false, td.above_seg_pred.len());
        td.above_seg_pred[cols].copy_from_slice(&self.above[6]);
        for i in 0..2 {
            let cols = above_cols(self.c0, td.sb_size4, false, td.above_palette[i].len());
            td.above_palette[i][cols].copy_from_slice(&self.above_palette[i]);
        }
    }
}

/// Plane sample area of the 4x4 area `r0..r1` x `c0..c1` (as `merge_tile`):
/// subsampled 4x4 rows and columns, sample rows and columns.
fn plane_area<P: Pixel>(
    pl: &crate::dec::frame::Plane<P>,
    luma_w: usize,
    luma_h: usize,
    p: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) -> [std::ops::Range<usize>; 4] {
    let sx = usize::from(pl.w < luma_w && p > 0);
    let sy = usize::from(pl.h < luma_h && p > 0);
    let (pr0, pr1) = (r0 >> sy, r1.div_ceil(1 << sy));
    let (pc0, pc1) = (c0 >> sx, c1.div_ceil(1 << sx));
    let (y0, y1) = ((pr0 * 4).min(pl.alloc_h), (pr1 * 4).min(pl.alloc_h));
    let (x0, x1) = ((pc0 * 4).min(pl.stride), (pc1 * 4).min(pl.stride));
    [pr0..pr1, pc0..pc1, y0..y1, x0..x1]
}

/// Columns of the above context of plane `p` that the superblock at mi
/// column `c0` writes.
fn above_cols(c0: usize, sb4: usize, sx: bool, len: usize) -> std::ops::Range<usize> {
    let s = usize::from(sx);
    (c0 >> s).min(len)..((c0 + sb4) >> s).min(len)
}

impl<P: Pixel> SbData<P> {
    fn capture<C: Coder>(td: &TileDec<'_, P, C>, r0: usize, c0: usize) -> Self {
        let r1 = (r0 + td.sb_size4).min(td.mi_row_end);
        let c1 = (c0 + td.sb_size4).min(td.mi_col_end);
        let fs = &*td.fs;
        let st = fs.mi_stride;
        let mut mi = Vec::with_capacity((r1 - r0) * (c1 - c0));
        let mut bytes: [Vec<u8>; 4] = Default::default();
        for r in r0..r1 {
            let (a, b) = (r * st + c0, r * st + c1);
            mi.extend_from_slice(&fs.mi[a..b]);
            bytes[0].extend_from_slice(&fs.uv_modes[a..b]);
            bytes[1].extend_from_slice(&fs.tx_types[a..b]);
            bytes[2].extend_from_slice(&fs.inter_tx_sizes[a..b]);
            bytes[3].extend_from_slice(&fs.segment_ids[a..b]);
        }
        let mut lf_tx: [Vec<u8>; 3] = Default::default();
        let mut px = Vec::with_capacity(td.cur.planes.len());
        let (lw, lh) = (td.cur.planes[0].w, td.cur.planes[0].h);
        for (p, pl) in td.cur.planes.iter().enumerate() {
            let [prs, pcs, ys, xs] = plane_area(pl, lw, lh, p, r0, r1, c0, c1);
            for r in prs {
                lf_tx[p].extend_from_slice(&fs.lf_tx[p][r * st + pcs.start..r * st + pcs.end]);
            }
            let mut v = Vec::with_capacity(ys.len() * xs.len());
            for y in ys {
                v.extend_from_slice(&pl.data[y * pl.stride + xs.start..y * pl.stride + xs.end]);
            }
            px.push(v);
        }
        let cs = fs.cdef_stride;
        let mut cdef = Vec::new();
        for r in r0 / 16..r1.div_ceil(16) {
            let (a, b) = (r * cs + c0 / 16, r * cs + c1.div_ceil(16));
            if b <= fs.cdef_idx.len() {
                cdef.extend_from_slice(&fs.cdef_idx[a..b]);
            }
        }
        Self {
            r0,
            r1,
            c0,
            c1,
            mi,
            bytes,
            lf_tx,
            px,
            cdef,
        }
    }

    /// Copies the snapshot into a frame state and picture.
    fn apply(&self, fs: &mut FrameState, cur: &mut Picture<P>) {
        let (r0, r1, c0, c1) = (self.r0, self.r1, self.c0, self.c1);
        let st = fs.mi_stride;
        let w = c1 - c0;
        for (i, r) in (r0..r1).enumerate() {
            let (a, b) = (r * st + c0, r * st + c1);
            let s = i * w..(i + 1) * w;
            fs.mi[a..b].copy_from_slice(&self.mi[s.clone()]);
            fs.uv_modes[a..b].copy_from_slice(&self.bytes[0][s.clone()]);
            fs.tx_types[a..b].copy_from_slice(&self.bytes[1][s.clone()]);
            fs.inter_tx_sizes[a..b].copy_from_slice(&self.bytes[2][s.clone()]);
            fs.segment_ids[a..b].copy_from_slice(&self.bytes[3][s]);
        }
        let (lw, lh) = (cur.planes[0].w, cur.planes[0].h);
        for (p, pl) in cur.planes.iter_mut().enumerate() {
            let [prs, pcs, ys, xs] = plane_area(pl, lw, lh, p, r0, r1, c0, c1);
            let n = pcs.len();
            for (i, r) in prs.enumerate() {
                fs.lf_tx[p][r * st + pcs.start..r * st + pcs.end]
                    .copy_from_slice(&self.lf_tx[p][i * n..(i + 1) * n]);
            }
            let n = xs.len();
            for (i, y) in ys.enumerate() {
                pl.data[y * pl.stride + xs.start..y * pl.stride + xs.end]
                    .copy_from_slice(&self.px[p][i * n..(i + 1) * n]);
            }
        }
        let cs = fs.cdef_stride;
        let mut k = 0;
        for r in r0 / 16..r1.div_ceil(16) {
            let (a, b) = (r * cs + c0 / 16, r * cs + c1.div_ceil(16));
            if b <= fs.cdef_idx.len() {
                fs.cdef_idx[a..b].copy_from_slice(&self.cdef[k..k + (b - a)]);
                k += b - a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Driver

/// A reference picture as the tiles see it.
#[derive(Clone, Copy)]
pub struct RefInfo<'a, P> {
    pub pic: &'a Picture<P>,
    pub upscaled_width: usize,
    pub frame_height: usize,
    pub frame_width: usize,
}

/// Everything the tiles of a frame are coded from.
pub struct TileJob<'a, P: Pixel> {
    pub seq: &'a SequenceHeader,
    pub hdr: &'a FrameHeader,
    pub refs: [Option<RefInfo<'a, P>>; 8],
    pub src: &'a Picture<P>,
    pub init_cdf: &'a CdfContext,
    pub qp: QuantParams,
    pub mode: ModeParams,
    pub sb_q: &'a [(i32, f64)],
    pub sb_cols: usize,
    /// The tile streams log their intervals (see [`MsacEnc::record`]).
    pub record: bool,
    pub threads: usize,
}

/// One superblock row of a tile.
struct Row {
    tile: usize,
    /// First mi row of the superblock row.
    r: usize,
    /// First row of its tile.
    first: bool,
    /// Index in the snapshot list of its first superblock.
    snap0: usize,
    /// Position of the tile's row above in the row list.
    above: usize,
    /// Row index inside its tile (the order rows are taken in).
    k: usize,
}

struct Shared<P> {
    /// Number of worker threads.
    workers: usize,
    done: Mutex<Vec<usize>>,
    cv: Condvar,
    snaps: Vec<OnceLock<SbSnap<P>>>,
    logs: Vec<OnceLock<Vec<SymOp>>>,
    next: AtomicUsize,
    failed: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<P> Shared<P> {
    /// Waits until row `i` has `n` superblocks done; false when a worker failed.
    fn wait(&self, i: usize, n: usize) -> bool {
        if self.workers <= 1 {
            // One worker takes the rows in order.
            return true;
        }
        let mut g = lock(&self.done);
        loop {
            if self.failed.load(Ordering::Acquire) {
                return false;
            }
            if g[i] >= n {
                return true;
            }
            g = self
                .cv
                .wait(g)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        let _g = lock(&self.done);
        self.cv.notify_all();
    }
}

/// Coded tiles: each tile's stream and, for the context tile, its final
/// CDFs.
pub type TileCoded = (MsacEnc, Option<Box<CdfContext>>);

/// Codes every tile of the frame `job` describes into `fs` and `cur`;
/// returns the tile streams in tile order.
pub fn code_tiles<P: Pixel>(
    job: &TileJob<'_, P>,
    fs: &mut FrameState,
    cur: &mut Picture<P>,
) -> R<Vec<TileCoded>> {
    let ti = &job.hdr.tile_info;
    let sb4 = if job.seq.use_128x128_superblock {
        32
    } else {
        16
    };
    let n_tiles = ti.cols * ti.rows;
    let mut rows = Vec::new();
    let mut tile_cols = Vec::with_capacity(n_tiles);
    let mut n_snaps = 0;
    for tile in 0..n_tiles {
        let (tr, tc) = (tile / ti.cols, tile % ti.cols);
        let (r0, r1) = (
            ti.mi_row_starts[tr] as usize,
            ti.mi_row_starts[tr + 1] as usize,
        );
        let (c0, c1) = (
            ti.mi_col_starts[tc] as usize,
            ti.mi_col_starts[tc + 1] as usize,
        );
        let ncols = (c1 - c0).div_ceil(sb4);
        tile_cols.push(ncols);
        for (k, r) in (r0..r1).step_by(sb4).enumerate() {
            rows.push(Row {
                tile,
                r,
                first: k == 0,
                snap0: n_snaps,
                above: 0,
                k,
            });
            n_snaps += ncols;
        }
    }
    // The tiles' rows are taken in turn (first rows of every tile, then
    // second rows, ...), so threads move to the tiles with rows left.
    rows.sort_by_key(|r| (r.k, r.tile));
    for i in 0..rows.len() {
        if !rows[i].first {
            let (t, k) = (rows[i].tile, rows[i].k);
            rows[i].above = rows[..i]
                .iter()
                .rposition(|r| r.tile == t && r.k + 1 == k)
                .unwrap_or(0);
        }
    }
    let threads = job.threads.clamp(1, rows.len().max(1));
    let shared = Shared::<P> {
        workers: threads,
        done: Mutex::new(vec![0; rows.len()]),
        cv: Condvar::new(),
        snaps: (0..n_snaps).map(|_| OnceLock::new()).collect(),
        logs: (0..rows.len()).map(|_| OnceLock::new()).collect(),
        next: AtomicUsize::new(0),
        failed: AtomicBool::new(false),
    };
    let mut layout_cdf = Box::new(job.init_cdf.clone());
    let layout = CdfLayout::new(&mut layout_cdf);
    // Worker 0 codes into the frame's own state and picture, the others
    // into copies; the rows they code are copied in afterwards.
    let work = |wid: usize, fs: &mut FrameState, cur: &mut Picture<P>| -> R<()> {
        let r = worker(job, &rows, &tile_cols, &shared, &layout, sb4, wid, fs, cur);
        if r.is_err() {
            shared.fail();
        }
        r
    };
    if threads <= 1 {
        work(0, fs, cur)?;
    } else {
        let motion_field = &fs.motion_field.clone();
        let results: Vec<R<()>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (1..threads)
                .map(|wid| {
                    let work = &work;
                    sc.spawn(move || {
                        let mut f = FrameState::new(job.seq, job.hdr);
                        f.motion_field.clone_from(motion_field);
                        let mut c = Picture::<P>::new(
                            job.hdr.frame_width as usize,
                            job.hdr.frame_height as usize,
                            job.seq.num_planes(),
                            job.seq.subsampling_x,
                            job.seq.subsampling_y,
                            job.seq.bit_depth,
                        );
                        work(wid, &mut f, &mut c)
                    })
                })
                .collect();
            let mut out = vec![work(0, fs, cur)];
            for h in hs {
                out.push(h.join().unwrap_or(Err(err("tile row thread"))));
            }
            out
        });
        for r in results {
            r?;
        }
    }
    // The tiles' bitstreams, coded from the logs with the real CDFs.
    let ctx_tile = ti.context_update_tile_id as usize;
    let code_tile = |tile: usize| -> R<TileCoded> {
        let mut e = MsacEnc::new(!job.hdr.disable_cdf_update);
        if job.record {
            e.record();
        }
        let mut cdf = Box::new(job.init_cdf.clone());
        {
            let (mut short, mut packed) = cdf_arrays(&mut cdf);
            for (i, row) in rows.iter().enumerate() {
                if row.tile != tile {
                    continue;
                }
                let ops = shared.logs[i].get().ok_or_else(|| err("tile row log"))?;
                if !replay(ops, &mut short, &mut packed, &mut e) {
                    return Err(err("tile row replay"));
                }
            }
        }
        Ok((e, (tile == ctx_tile).then_some(cdf)))
    };
    let coded: Vec<R<TileCoded>> = if n_tiles > 1 && job.threads > 1 {
        let next = AtomicUsize::new(0);
        let t = job.threads.min(n_tiles);
        let mut parts: Vec<(usize, R<TileCoded>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..t)
                .map(|_| {
                    let (code_tile, next) = (&code_tile, &next);
                    sc.spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n_tiles {
                                break;
                            }
                            out.push((i, code_tile(i)));
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        parts.sort_by_key(|p| p.0);
        if parts.len() != n_tiles {
            return Err(err("tile thread"));
        }
        parts.into_iter().map(|p| p.1).collect()
    } else {
        (0..n_tiles).map(code_tile).collect()
    };
    // The rows the other workers coded.
    for s in &shared.snaps {
        let s = s.get().ok_or_else(|| err("superblock snapshot"))?;
        if s.by != 0
            && let Some(d) = &s.data
        {
            d.apply(fs, cur);
        }
    }
    coded.into_iter().collect()
}

fn tile_dec<'a, P: Pixel>(
    job: &'a TileJob<'_, P>,
    fs: &'a mut FrameState,
    cur: &'a mut Picture<P>,
    cdf: &'a mut CdfContext,
    msac: SymLog<'a>,
) -> TileDec<'a, P, SymLog<'a>> {
    let seq = job.seq;
    let n_above = fs.mi_stride + 64;
    let qp = job.qp;
    let mode = job.mode;
    TileDec {
        seq,
        hdr: job.hdr,
        fs,
        cur,
        refs: std::array::from_fn(|i| {
            job.refs[i].map(|r| RefView {
                pic: r.pic,
                mirror: None,
                upscaled_width: r.upscaled_width,
                frame_height: r.frame_height,
                frame_width: r.frame_width,
            })
        }),
        mirrors: Vec::new(),
        msac,
        pb: Default::default(),
        plan: EncPlan {
            sb_q: job.sb_q.to_vec(),
            sb_cols: job.sb_cols,
            base_lambda: (qp.lambda, mode.lambda_satd, mode.skip_sse_per_px),
            ..Default::default()
        },
        plan_q: 0,
        cfl_choice: (0, 0),
        intra_choice: (0, None),
        est_hint: (0, 0),
        cur_filters: [0, 0],
        plan_delta_lf: [0; 4],
        src: Some(job.src),
        enc: qp,
        mode,
        me_scratch: Default::default(),
        cdf,
        mi_row_start: 0,
        mi_row_end: 0,
        mi_col_start: 0,
        mi_col_end: 0,
        sb_size: if seq.use_128x128_superblock { 15 } else { 12 },
        sb_size4: if seq.use_128x128_superblock { 32 } else { 16 },
        bd: seq.bit_depth,
        sub_x: seq.subsampling_x,
        sub_y: seq.subsampling_y,
        num_planes: seq.num_planes(),
        above_level: std::array::from_fn(|_| vec![0u8; n_above]),
        above_dc: std::array::from_fn(|_| vec![0u8; n_above]),
        above_seg_pred: vec![0u8; n_above],
        left_level: [[0; 32]; 3],
        left_dc: [[0; 32]; 3],
        left_seg_pred: [0; 32],
        above_palette: std::array::from_fn(|_| vec![[0u16; 8]; n_above]),
        left_palette: [[[0; 8]; 32]; 2],
        delta_lf: [0; 4],
        current_q_index: i32::from(job.hdr.quant.base_q_idx),
        read_deltas: false,
        ref_lr_wiener: [[[0; 3]; 2]; 3],
        ref_sgr_xqd: [[0; 2]; 3],
        block_decoded: [[[false; 35]; 35]; 3],
        b: Default::default(),
        color_map_y: vec![0u8; 128 * 128],
        color_map_uv: vec![0u8; 128 * 128],
        scratch: Default::default(),
        rec: None,
        y0: 0,
        mi_r0: 0,
        mi_edges: false,
    }
}

/// Takes superblock rows in order until none is left.
fn worker<P: Pixel>(
    job: &TileJob<'_, P>,
    rows: &[Row],
    tile_cols: &[usize],
    shared: &Shared<P>,
    layout: &CdfLayout,
    sb4: usize,
    wid: usize,
    fs: &mut FrameState,
    cur: &mut Picture<P>,
) -> R<()> {
    let hdr = job.hdr;
    let ti = &hdr.tile_info;
    let mut cdf = Box::new(job.init_cdf.clone());
    let msac = SymLog::new(&cdf, layout, !hdr.disable_cdf_update);
    let mut td = tile_dec(job, fs, cur, &mut cdf, msac);
    // Delta q and loop filter deltas carry from row to row: the rows of a
    // tile then follow each other.
    let serial = hdr.delta_q_present || hdr.delta_lf_present;
    loop {
        let i = shared.next.fetch_add(1, Ordering::Relaxed);
        let Some(row) = rows.get(i) else {
            return Ok(());
        };
        let (tr, tc) = (row.tile / ti.cols, row.tile % ti.cols);
        td.mi_row_start = ti.mi_row_starts[tr] as usize;
        td.mi_row_end = ti.mi_row_starts[tr + 1] as usize;
        td.mi_col_start = ti.mi_col_starts[tc] as usize;
        td.mi_col_end = ti.mi_col_starts[tc + 1] as usize;
        let ncols = tile_cols[row.tile];
        // The CDFs the row starts from.
        let cdf_sb = 1.min(ncols - 1);
        if row.first {
            td.cdf.clone_from(job.init_cdf);
        } else {
            if !shared.wait(row.above, if serial { ncols } else { cdf_sb + 1 }) {
                return Err(err("tile row"));
            }
            let above = rows[row.above].snap0;
            let s = shared.snaps[above + cdf_sb]
                .get()
                .and_then(|s| s.cdf.as_deref())
                .ok_or_else(|| err("tile row CDFs"))?;
            td.cdf.clone_from(s);
        }
        // begin_tile(). A decoder clears the above contexts of every plane
        // in luma 4x4 units from the tile's first column to 32 past its
        // last, which with the tiles before it (in tile order) leaves all of
        // the tile's own contexts zero; this worker's state may hold other
        // tiles' rows, so it clears exactly the tile's columns of each
        // plane (and the superblock overhang at the frame's right edge).
        // Below the first row the row above left the contexts (in this
        // worker's state, or copied in below).
        if row.first {
            let s = td.mi_col_start;
            let e = if tc + 1 == ti.cols {
                td.mi_col_end + 32
            } else {
                td.mi_col_end
            };
            for p in 0..3 {
                let sh = usize::from(p > 0 && td.sub_x);
                let len = td.above_level[p].len();
                let cols = (s >> sh).min(len)..(e >> sh).min(len);
                td.above_level[p][cols.clone()].fill(0);
                td.above_dc[p][cols].fill(0);
            }
            let len = td.above_seg_pred.len();
            td.above_seg_pred[s.min(len)..e.min(len)].fill(0);
        }
        td.delta_lf = [0; 4];
        td.current_q_index = i32::from(hdr.quant.base_q_idx);
        for p in 0..td.num_planes {
            td.ref_sgr_xqd[p] = [-32, 31];
            td.ref_lr_wiener[p] = [[3, -7, 15]; 2];
        }
        if !row.first && serial {
            let s = shared.snaps[rows[row.above].snap0 + ncols - 1]
                .get()
                .ok_or_else(|| err("tile row state"))?;
            (td.current_q_index, td.delta_lf) = s.q_state;
        }
        td.begin_sb_row();
        td.b = Default::default();
        td.est_hint = (0, 0);
        td.cur_filters = [0, 0];
        td.msac.ops.clear();
        let mut applied = 0;
        for k in 0..ncols {
            if !row.first {
                // The row above up to the superblock above and to the right.
                let need = (k + 2).min(ncols);
                if !shared.wait(row.above, need) {
                    return Err(err("tile row"));
                }
                while applied < need {
                    let s = shared.snaps[rows[row.above].snap0 + applied]
                        .get()
                        .ok_or_else(|| err("tile row snapshot"))?;
                    if s.by != wid {
                        let d = s.data.as_ref().ok_or_else(|| err("tile row snapshot"))?;
                        d.apply(td.fs, td.cur);
                    }
                    s.above.apply_to(&mut td);
                    applied += 1;
                }
            }
            let c = td.mi_col_start + k * sb4;
            td.code_superblock(row.r, c)?;
            if td.msac.bad {
                return Err(err("CDF outside the context"));
            }
            let snap = SbSnap {
                by: wid,
                data: (shared.workers > 1).then(|| SbData::capture(&td, row.r, c)),
                above: SbAbove::capture(&td, c),
                q_state: (td.current_q_index, td.delta_lf),
                cdf: (k == cdf_sb).then(|| Box::new(td.cdf.clone())),
            };
            if shared.snaps[row.snap0 + k].set(snap).is_err() {
                return Err(err("superblock snapshot"));
            }
            if shared.workers > 1 {
                {
                    let mut g = lock(&shared.done);
                    g[i] = k + 1;
                }
                shared.cv.notify_all();
            }
        }
        let ops = std::mem::take(&mut td.msac.ops);
        if shared.logs[i].set(ops).is_err() {
            return Err(err("tile row log"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CdfLayout, cdf_arrays};
    use crate::dec::cdf::CdfContext;

    #[test]
    fn layout_covers_the_context() {
        let mut c = CdfContext::new(100);
        let (short, packed) = cdf_arrays(&mut c);
        let bytes: usize = short.iter().map(|s| s.len() * 2).sum::<usize>()
            + packed.iter().map(|s| s.len() * 8).sum::<usize>();
        let size = size_of::<CdfContext>();
        // Every field is listed (at most alignment padding is left over).
        assert!(bytes <= size && size - bytes < 8, "{bytes} of {size}");
        drop((short, packed));
        let l = CdfLayout::new(&mut c);
        assert_eq!(l.arrays.len(), 97);
    }
}
