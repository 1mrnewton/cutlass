//! Inverse transforms and reconstruction (spec 7.13.2, 7.13.3 and step 3 of
//! 7.12.3).
//!
//! The row pass runs on groups of `L` rows at once and the column pass on
//! groups of `L` columns at once: every 1D transform works on an array of
//! `[E; L]` lanes, so each butterfly is one short loop across lanes that LLVM
//! vectorizes. `E` is `i16` for 8-bit streams (which vectorizes on plain
//! SSE2) and `i32` otherwise. The DCT and ADST are tables of butterfly steps
//! generated at compile time from the spec's step lists, with the steps whose
//! inputs are known to be zero pruned away; the input permutations are folded
//! into the loads and the identity transforms into the outputs. All-zero rows
//! are skipped, and DC-only blocks and DC-only rows or columns of a DCT take
//! shortcuts.
//!
//! The arithmetic inside the 1D transforms wraps. For a conformant stream the
//! spec guarantees that every butterfly result fits its clamping range, which
//! makes `x * c0 - y * c1 + 2048` fit an `i32` exactly, so the results are
//! bit exact. A non-conformant stream gives meaningless samples, never a
//! panic.

use super::Pixel;
// The DCT and ADST as tables of steps, shared with the vector kernel.
use crate::spec::tables::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, SINPI_1_9, SINPI_2_9, SINPI_3_9,
    SINPI_4_9, TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2, V_ADST, V_DCT, V_FLIPADST,
};
use cutlass_simd::av1::itx::steps::{
    Prog, Step as Op, adst_in_perm, adst_out_perm, adst8_prog, adst16_prog, dct_perm, dct_prog,
};

/// Returns `(flip_ud, flip_lr)` for a transform type (spec 7.12.3).
pub fn tx_type_flips(tx_type: usize) -> (bool, bool) {
    let t = tx_type as i32;
    let ud = matches!(
        t,
        FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
    );
    let lr = matches!(
        t,
        DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
    );
    (ud, lr)
}

/// A 1D inverse transform kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Dct,
    Adst,
    Idt,
}

/// The 1D transform applied to each row (horizontal).
fn row_kind(tx_type: usize) -> Kind {
    match tx_type as i32 {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => Kind::Dct,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | H_ADST | H_FLIPADST => Kind::Adst,
        _ => Kind::Idt,
    }
}

/// The 1D transform applied to each column (vertical).
fn col_kind(tx_type: usize) -> Kind {
    match tx_type as i32 {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => Kind::Dct,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | V_ADST | V_FLIPADST => Kind::Adst,
        _ => Kind::Idt,
    }
}

// ---------------------------------------------------------------------------
// Permutation tables, computed at compile time.

const fn ident_perm() -> [u8; 64] {
    let mut p = [0u8; 64];
    let mut k = 0;
    while k < 64 {
        p[k] = k as u8;
        k += 1;
    }
    p
}

static DCT4_PERM: [u8; 4] = dct_perm::<4>();
static DCT8_PERM: [u8; 8] = dct_perm::<8>();
static DCT16_PERM: [u8; 16] = dct_perm::<16>();
static DCT32_PERM: [u8; 32] = dct_perm::<32>();
static DCT64_PERM: [u8; 64] = dct_perm::<64>();
static ADST8_IN: [u8; 8] = adst_in_perm::<8>();
static ADST16_IN: [u8; 16] = adst_in_perm::<16>();
static ADST8_OUT: [u8; 8] = adst_out_perm::<8>();
static ADST16_OUT: [u8; 16] = adst_out_perm::<16>();
static IDENT: [u8; 64] = ident_perm();

/// Where input `k` of a transform of length `n` lands in T.
fn in_perm(kind: Kind, n: usize) -> &'static [u8] {
    match (kind, n) {
        (Kind::Dct, 4) => &DCT4_PERM,
        (Kind::Dct, 8) => &DCT8_PERM,
        (Kind::Dct, 16) => &DCT16_PERM,
        (Kind::Dct, 32) => &DCT32_PERM,
        (Kind::Dct, _) => &DCT64_PERM,
        (Kind::Adst, 8) => &ADST8_IN,
        (Kind::Adst, 16) => &ADST16_IN,
        _ => &IDENT,
    }
}

// ---------------------------------------------------------------------------
// Butterflies over lanes.

/// `Round2(x, 12)` for a sum that the spec guarantees fits after rounding.
#[inline(always)]
fn rnd12(x: i32) -> i32 {
    x.wrapping_add(2048) >> 12
}

/// The element type of the transform lanes.
///
/// `i32` serves every bit depth. `i16` serves 8-bit streams only: there both
/// clamping ranges are 16 bits, so the Hadamard clamp is a saturating add and
/// every butterfly output of a conformant stream fits. The only value that
/// can leave the range is `32768` (a negated `-32768`, or an ADST4 output),
/// and saturating it to `32767` is invisible after the following round and
/// clamp. The 16-bit lanes vectorize on plain SSE2, which has no 32-bit
/// multiply or min/max.
trait Lane: Copy + Default + Send + Sync + 'static {
    /// True when every value of the type is inside the 8-bit clamping range.
    const NARROW: bool;
    /// The rotation `(Round2(a*c - b*s, 12), Round2(a*s + b*c, 12))`.
    fn rot(a: Self, b: Self, c: i32, s: i32) -> (Self, Self);
    /// `(Clip(x + y), Clip(x - y))` to `mn..=mx`.
    fn had(x: Self, y: Self, mn: i32, mx: i32) -> (Self, Self);
    fn neg(self) -> Self;
    /// `self * c` for `|c| < 32768`.
    fn mulc(self, c: i32) -> i32;
    fn get(self) -> i32;
    /// Converts a value inside the clamping range.
    fn put(v: i32) -> Self;
    /// Converts a value inside the clamping range or equal to its maximum
    /// plus one (saturating).
    fn put_sat1(v: i32) -> Self;
    /// `Round2(self, 4)` as 16 bits (used for 16-bit lanes only).
    fn round4(self) -> i16;
}

impl Lane for i32 {
    const NARROW: bool = false;
    #[inline(always)]
    fn rot(a: i32, b: i32, c: i32, s: i32) -> (i32, i32) {
        (
            rnd12(a.wrapping_mul(c).wrapping_sub(b.wrapping_mul(s))),
            rnd12(a.wrapping_mul(s).wrapping_add(b.wrapping_mul(c))),
        )
    }
    #[inline(always)]
    fn had(x: i32, y: i32, mn: i32, mx: i32) -> (i32, i32) {
        (
            x.wrapping_add(y).max(mn).min(mx),
            x.wrapping_sub(y).max(mn).min(mx),
        )
    }
    #[inline(always)]
    fn neg(self) -> i32 {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn mulc(self, c: i32) -> i32 {
        self.wrapping_mul(c)
    }
    #[inline(always)]
    fn get(self) -> i32 {
        self
    }
    #[inline(always)]
    fn put(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn put_sat1(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn round4(self) -> i16 {
        (self.wrapping_add(8) >> 4) as i16
    }
}

/// The exact product of two 16-bit values, written as the low and high
/// halves (`pmullw` and `pmulhw`) so that SSE2 vectorizes it.
#[inline(always)]
fn prod16(a: i16, c: i16) -> i32 {
    let lo = a.wrapping_mul(c);
    let hi = ((a as i32 * c as i32) >> 16) as i16;
    ((hi as i32) << 16) | (lo as u16 as i32)
}

impl Lane for i16 {
    const NARROW: bool = true;
    #[inline(always)]
    fn rot(a: i16, b: i16, c: i32, s: i32) -> (i16, i16) {
        let (a, b) = (a as i32, b as i32);
        let (c, s) = (c as i16 as i32, s as i16 as i32);
        (rnd12(a * c - b * s) as i16, rnd12(a * s + b * c) as i16)
    }
    #[inline(always)]
    fn had(x: i16, y: i16, _mn: i32, _mx: i32) -> (i16, i16) {
        (x.saturating_add(y), x.saturating_sub(y))
    }
    #[inline(always)]
    fn neg(self) -> i16 {
        self.saturating_neg()
    }
    #[inline(always)]
    fn mulc(self, c: i32) -> i32 {
        prod16(self, c as i16)
    }
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn put(v: i32) -> i16 {
        v as i16
    }
    #[inline(always)]
    fn put_sat1(v: i32) -> i16 {
        (v - (v == 32768) as i32) as i16
    }
    #[inline(always)]
    fn round4(self) -> i16 {
        (self >> 4) + ((self >> 3) & 1)
    }
}

/// The spec's butterfly rotation `B(a, b, angle, flip)` with `c = cos128(angle)`
/// and `s = sin128(angle)`.
#[inline(always)]
fn bf<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    a: usize,
    b: usize,
    c: i32,
    s: i32,
    flip: bool,
) {
    let ta = t[a];
    let tb = t[b];
    let mut x = [E::default(); L];
    let mut y = [E::default(); L];
    for l in 0..L {
        (x[l], y[l]) = E::rot(ta[l], tb[l], c, s);
    }
    if flip {
        t[a] = y;
        t[b] = x;
    } else {
        t[a] = x;
        t[b] = y;
    }
}

/// The spec's Hadamard rotation `H(a, b, flip)` with clamping to `mn..=mx`.
#[inline(always)]
fn hd<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    a: usize,
    b: usize,
    flip: bool,
    mn: i32,
    mx: i32,
) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    let mut s = [E::default(); L];
    let mut d = [E::default(); L];
    for l in 0..L {
        (s[l], d[l]) = E::had(x[l], y[l], mn, mx);
    }
    t[a] = s;
    t[b] = d;
}

// ---------------------------------------------------------------------------
// 1D transforms on permuted input.

/// Drops the steps of `full` whose two inputs are known to be zero, given
/// that only the inputs `0..nz` can be non-zero and input `k` sits at
/// `perm[k]`. Both kinds of step map a pair of zeros to zeros, so the result
/// is unchanged.
const fn prune<const M: usize>(full: Prog<1024>, perm: &[u8], nz: usize) -> Prog<M> {
    let mut live = [false; 64];
    let mut k = 0;
    while k < nz && k < perm.len() {
        live[perm[k] as usize] = true;
        k += 1;
    }
    let mut p = Prog::<M>::new();
    let mut i = 0;
    while i < full.len {
        let op = full.ops[i];
        let (a, b) = (op.a as usize, op.b as usize);
        if live[a] || live[b] {
            live[a] = true;
            live[b] = true;
            if p.len < M {
                p.ops[p.len] = op;
            }
            p.len += 1;
        }
        i += 1;
    }
    p
}

const fn dct_nz<const M: usize>(n: u32, nz: usize) -> Prog<M> {
    let perm: &[u8] = match n {
        2 => &dct_perm::<4>(),
        3 => &dct_perm::<8>(),
        4 => &dct_perm::<16>(),
        5 => &dct_perm::<32>(),
        _ => &dct_perm::<64>(),
    };
    prune::<M>(dct_prog::<1024>(n), perm, nz)
}

const fn adst_nz<const M: usize>(n: u32, nz: usize) -> Prog<M> {
    if n == 3 {
        prune::<M>(adst8_prog::<1024>(), &adst_in_perm::<8>(), nz)
    } else {
        prune::<M>(adst16_prog::<1024>(), &adst_in_perm::<16>(), nz)
    }
}

/// A static step table built by `$f::<M>($args)`.
macro_rules! prog_table {
    ($name:ident, $f:ident, $($arg:expr),*) => {
        static $name: [Op; $f::<1024>($($arg),*).len] =
            $f::<{ $f::<1024>($($arg),*).len }>($($arg),*).ops;
    };
}

prog_table!(DCT4_1, dct_nz, 2, 1);
prog_table!(DCT4_2, dct_nz, 2, 2);
prog_table!(DCT4_4, dct_nz, 2, 4);
prog_table!(DCT8_2, dct_nz, 3, 2);
prog_table!(DCT8_4, dct_nz, 3, 4);
prog_table!(DCT8_8, dct_nz, 3, 8);
prog_table!(DCT16_4, dct_nz, 4, 4);
prog_table!(DCT16_8, dct_nz, 4, 8);
prog_table!(DCT16_16, dct_nz, 4, 16);
prog_table!(DCT32_8, dct_nz, 5, 8);
prog_table!(DCT32_16, dct_nz, 5, 16);
prog_table!(DCT32_32, dct_nz, 5, 32);
prog_table!(DCT64_8, dct_nz, 6, 8);
prog_table!(DCT64_16, dct_nz, 6, 16);
prog_table!(DCT64_32, dct_nz, 6, 32);
prog_table!(ADST8_2, adst_nz, 3, 2);
prog_table!(ADST8_4, adst_nz, 3, 4);
prog_table!(ADST8_8, adst_nz, 3, 8);
prog_table!(ADST16_4, adst_nz, 4, 4);
prog_table!(ADST16_8, adst_nz, 4, 8);
prog_table!(ADST16_16, adst_nz, 4, 16);

/// The step table for a transform of length `n` whose inputs `nz..` are zero:
/// the full table pruned for `n / 4`, `n / 2` or `n` live inputs (at most 32
/// inputs of a 64-point transform are non-zero).
fn prog_for(kind: Kind, n: usize, nz: usize) -> &'static [Op] {
    let q = if nz <= n / 4 {
        0
    } else if nz <= n / 2 {
        1
    } else {
        2
    };
    match (kind, n, q) {
        (Kind::Dct, 4, 0) => &DCT4_1,
        (Kind::Dct, 4, 1) => &DCT4_2,
        (Kind::Dct, 4, _) => &DCT4_4,
        (Kind::Dct, 8, 0) => &DCT8_2,
        (Kind::Dct, 8, 1) => &DCT8_4,
        (Kind::Dct, 8, _) => &DCT8_8,
        (Kind::Dct, 16, 0) => &DCT16_4,
        (Kind::Dct, 16, 1) => &DCT16_8,
        (Kind::Dct, 16, _) => &DCT16_16,
        (Kind::Dct, 32, 0) => &DCT32_8,
        (Kind::Dct, 32, 1) => &DCT32_16,
        (Kind::Dct, 32, _) => &DCT32_32,
        (Kind::Dct, _, 0) if nz <= 8 => &DCT64_8,
        (Kind::Dct, _, 0) => &DCT64_16,
        (Kind::Dct, _, _) => &DCT64_32,
        (_, 8, 0) => &ADST8_2,
        (_, 8, 1) => &ADST8_4,
        (_, 8, _) => &ADST8_8,
        (_, _, 0) => &ADST16_4,
        (_, _, 1) => &ADST16_8,
        (_, _, _) => &ADST16_16,
    }
}

/// Runs a step table on `t`.
#[inline(always)]
fn run_prog<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    ops: &[Op],
    mn: i32,
    mx: i32,
) {
    // Keep the step loop a loop: unrolled, LLVM vectorizes across steps.
    let ops = if L > 1 {
        core::hint::black_box(ops)
    } else {
        ops
    };
    for op in ops {
        let a = op.a as usize & (N - 1);
        let b = op.b as usize & (N - 1);
        if op.had {
            hd(t, a, b, op.flip, mn, mx);
        } else {
            bf(t, a, b, op.cos as i32, op.sin as i32, op.flip);
        }
    }
}

/// Inverse ADST4 (spec 7.13.2.6).
#[inline(always)]
fn adst4<E: Lane, const L: usize, const N: usize>(t: &mut [[E; L]; N]) {
    let (i0, i1, i2, i3) = (t[0], t[1], t[2], t[3]);
    let mut o = [[E::default(); L]; 4];
    // Exact Round2(x, 12) even when x + 2048 would overflow.
    let r = |x: i32| E::put_sat1((x >> 12) + ((x >> 11) & 1));
    for l in 0..L {
        let (t0, t1, t2, t3) = (i0[l], i1[l], i2[l], i3[l]);
        let mut s0 = t0.mulc(SINPI_1_9);
        let mut s1 = t0.mulc(SINPI_2_9);
        let mut s2 = t1.mulc(SINPI_3_9);
        let mut s3 = t2.mulc(SINPI_4_9);
        let s4 = t2.mulc(SINPI_1_9);
        let s5 = t3.mulc(SINPI_2_9);
        let s6 = t3.mulc(SINPI_4_9);
        let a7 = t0.get().wrapping_sub(t2.get());
        let b7 = E::put(a7.wrapping_add(t3.get()));
        s0 = s0.wrapping_add(s3);
        s1 = s1.wrapping_sub(s4);
        s3 = s2;
        s2 = b7.mulc(SINPI_3_9);
        s0 = s0.wrapping_add(s5);
        s1 = s1.wrapping_sub(s6);
        let x0 = s0.wrapping_add(s3);
        let x1 = s1.wrapping_add(s3);
        let x2 = s2;
        let x3 = s0.wrapping_add(s1).wrapping_sub(s3);
        o[0][l] = r(x0);
        o[1][l] = r(x1);
        o[2][l] = r(x2);
        o[3][l] = r(x3);
    }
    t[..4].copy_from_slice(&o);
}

/// Inverse ADST8 (spec 7.13.2.7, input permutation already applied).
#[inline(always)]
fn adst8<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    nz: usize,
    mn: i32,
    mx: i32,
) {
    run_prog(t, prog_for(Kind::Adst, 8, nz), mn, mx);
    adst_out(t, &ADST8_OUT);
}

/// Inverse ADST16 (spec 7.13.2.8, input permutation already applied).
#[inline(always)]
fn adst16<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    nz: usize,
    mn: i32,
    mx: i32,
) {
    run_prog(t, prog_for(Kind::Adst, 16, nz), mn, mx);
    adst_out(t, &ADST16_OUT);
}

/// ADST output permutation (spec 7.13.2.5).
#[inline(always)]
fn adst_out<E: Lane, const L: usize, const N: usize>(t: &mut [[E; L]; N], perm: &[u8]) {
    let copy = *t;
    for (i, &p) in perm.iter().enumerate() {
        let src = copy[p as usize & (N - 1)];
        if i & 1 == 1 {
            let mut v = [E::default(); L];
            for l in 0..L {
                v[l] = src[l].neg();
            }
            t[i & (N - 1)] = v;
        } else {
            t[i & (N - 1)] = src;
        }
    }
}

/// Inverse identity transform of length `N` (spec 7.13.2.11 to 7.13.2.15) on
/// one value, written so that no product can overflow:
/// `Round2(x * 5793, 12) == x + Round2(x * 1697, 12)` and
/// `Round2(x * 11586, 12) == 2 * x + Round2(x * 3394, 12)` exactly.
/// The identity is elementwise, so the passes apply it at their output.
#[inline(always)]
fn idt<const N: usize>(a: i32) -> i32 {
    match N {
        4 => a.wrapping_add(rnd12(a.wrapping_mul(1697))),
        8 => a.wrapping_mul(2),
        16 => a.wrapping_mul(2).wrapping_add(rnd12(a.wrapping_mul(3394))),
        _ => a.wrapping_mul(4),
    }
}

const K_DCT: u8 = 0;
const K_ADST: u8 = 1;
const K_IDT: u8 = 2;

/// The non-identity 1D transforms; the identity is applied at output.
#[inline(always)]
fn tx1d<E: Lane, const L: usize, const N: usize, const K: u8>(
    t: &mut [[E; L]; N],
    nz: usize,
    mn: i32,
    mx: i32,
) {
    match K {
        K_DCT => run_prog(t, prog_for(Kind::Dct, N, nz), mn, mx),
        K_ADST => match N {
            4 => adst4(t),
            8 => adst8(t, nz, mn, mx),
            _ => adst16(t, nz, mn, mx),
        },
        _ => {}
    }
}

/// Where input `k` lands in T, as a fixed-size table.
#[inline(always)]
fn perm_for<const N: usize, const K: u8>() -> Option<&'static [u8; N]> {
    let kind = match K {
        K_DCT => Kind::Dct,
        K_ADST => Kind::Adst,
        _ => Kind::Idt,
    };
    <&[u8; N]>::try_from(in_perm(kind, N).get(..N)?).ok()
}

/// Per-block constants shared by both passes.
struct Params {
    /// Width of the transform.
    w: usize,
    /// Column stride of the intermediate: `min(h, 32)`.
    ts: usize,
    nz_rows: usize,
    nz_cols: usize,
    rect: bool,
    row_shift: u32,
    rmin: i32,
    rmax: i32,
    cmin: i32,
    cmax: i32,
    flip_ud: bool,
    flip_lr: bool,
    pmax: i32,
}

// ---------------------------------------------------------------------------
// Passes.

/// Row transforms of rows `0..nz_rows` (rounded up to `L`). Writes the
/// clamped intermediate into `tmp` column-major: `tmp[j * ts + i]` is
/// Residual[i][j].
#[inline(never)]
fn row_pass<E: Lane, const L: usize, const N: usize, const K: u8>(
    p: &Params,
    coefs: &[i32],
    tmp: &mut [i32],
) {
    // The row stride of `coefs` is known from the row length.
    let tw = if N > 32 { 32 } else { N };
    let ts = p.ts;
    let shift = p.row_shift;
    let rnd = (1i32 << shift) >> 1;
    let (cmin, cmax) = (p.cmin, p.cmax);
    if K == K_DCT && p.nz_cols == 1 {
        // Only T[0] is non-zero: every output of the DCT is Round2(T[0] * 2896,
        // 12), and the Hadamard clamps cannot bite since |output| <= |T[0]|.
        for i in 0..p.nz_rows {
            let mut c = coefs[i * tw];
            if p.rect {
                c = rnd12(c.wrapping_mul(2896));
            }
            let v = rnd12(c.wrapping_mul(2896));
            let v = (v.wrapping_add(rnd) >> shift).max(cmin).min(cmax);
            for j in 0..N {
                tmp[j * ts + i] = v;
            }
        }
        return;
    }
    let Some(perm) = perm_for::<N, K>() else {
        return;
    };
    let ncols = p.nz_cols.min(tw).min(N);
    let mut t = [[E::default(); L]; N];
    let mut r0 = 0;
    while r0 < p.nz_rows {
        if ncols < N {
            t = [[E::default(); L]; N];
        }
        // A constant count when all columns are loaded.
        if ncols == tw {
            load_rows(&mut t, coefs, perm, r0 * tw, tw, tw, p.rect);
        } else {
            load_rows(&mut t, coefs, perm, r0 * tw, tw, ncols, p.rect);
        }
        tx1d::<E, L, N, K>(&mut t, ncols, p.rmin, p.rmax);
        for (j, v) in t.iter().enumerate() {
            let out = &mut tmp[j * ts + r0..][..L];
            for l in 0..L {
                let x = if K == K_IDT {
                    idt::<N>(v[l].get())
                } else {
                    v[l].get()
                };
                let x = x.wrapping_add(rnd) >> shift;
                // With 16-bit lanes the range is the 16-bit range, which a
                // non-identity output is already inside.
                out[l] = if !E::NARROW {
                    x.max(cmin).min(cmax)
                } else if K == K_IDT {
                    x.clamp(-32768, 32767)
                } else {
                    x
                };
            }
        }
        r0 += L;
    }
}

/// Transposing load: `t[perm[k]][l] = src[start + l * stride + k]` for
/// `k < n`, optionally with the rectangular scaling.
#[inline(always)]
fn load_rows<E: Lane, const L: usize, const N: usize>(
    t: &mut [[E; L]; N],
    src: &[i32],
    perm: &[u8; N],
    start: usize,
    stride: usize,
    n: usize,
    rect: bool,
) {
    for l in 0..L {
        let row = &src[start + l * stride..][..n];
        if rect {
            for (&k, &c) in perm.iter().zip(row) {
                t[k as usize & (N - 1)][l] = E::put(rnd12(c.wrapping_mul(2896)));
            }
        } else {
            for (&k, &c) in perm.iter().zip(row) {
                t[k as usize & (N - 1)][l] = E::put(c);
            }
        }
    }
}

/// Adds `res` (one row of `L` residuals) to `row`.
#[inline(always)]
fn add_row<P: Pixel, const L: usize>(row: &mut [P], res: &[i32; L], pmax: i32) {
    let row = &mut row[..L];
    for l in 0..L {
        row[l] = P::from_i32((row[l].to_i32() + res[l]).max(0).min(pmax));
    }
}

/// Column transforms of `L` columns at a time, then the flipped add to `dst`.
#[inline(never)]
fn col_pass<P: Pixel, E: Lane, const L: usize, const N: usize, const K: u8>(
    p: &Params,
    tmp: &[i32],
    dst: &mut [P],
    stride: usize,
) {
    let (w, pmax) = (p.w, p.pmax);
    // The column stride of `tmp` is known from the column length.
    let ts = if N > 32 { 32 } else { N };
    if K == K_DCT && p.nz_rows == 1 {
        // Only T[0] of each column is non-zero: the column is constant.
        let mut vals = [0i32; 64];
        for j in 0..w {
            let v = rnd12(tmp[j * ts].wrapping_mul(2896));
            let x = if p.flip_lr { w - 1 - j } else { j };
            vals[x & 63] = v.wrapping_add(8) >> 4;
        }
        let vals = &vals[..w];
        for y in 0..N {
            for (px, &v) in dst[y * stride..][..w].iter_mut().zip(vals) {
                *px = P::from_i32((px.to_i32() + v).max(0).min(pmax));
            }
        }
        return;
    }
    let Some(perm) = perm_for::<N, K>() else {
        return;
    };
    let nrows = p.nz_rows.min(N);
    let mut t = [[E::default(); L]; N];
    let mut j0 = 0;
    while j0 < w {
        if nrows < N {
            t = [[E::default(); L]; N];
        }
        // A constant count when all rows are loaded.
        if nrows == ts {
            load_rows(&mut t, tmp, perm, j0 * ts, ts, ts, false);
        } else {
            load_rows(&mut t, tmp, perm, j0 * ts, ts, nrows, false);
        }
        tx1d::<E, L, N, K>(&mut t, nrows, p.cmin, p.cmax);
        let x0 = if p.flip_lr { w - j0 - L } else { j0 };
        for (i, v) in t.iter().enumerate() {
            let y = if p.flip_ud { N - 1 - i } else { i };
            let row = &mut dst[y * stride + x0..][..L];
            if E::NARROW && !P::HBD {
                // 8-bit: residuals and sums stay in 16 bits.
                let mut res = [0i16; L];
                for l in 0..L {
                    res[l] = if K == K_IDT {
                        (idt::<N>(v[l].get()).wrapping_add(8) >> 4) as i16
                    } else {
                        v[l].round4()
                    };
                }
                if p.flip_lr {
                    res.reverse();
                }
                for l in 0..L {
                    let px = row[l].to_i32() as i16 + res[l];
                    row[l] = P::from_i32(px.clamp(0, 255) as i32);
                }
            } else {
                let mut res = [0i32; L];
                for l in 0..L {
                    let x = if K == K_IDT {
                        idt::<N>(v[l].get())
                    } else {
                        v[l].get()
                    };
                    res[l] = x.wrapping_add(8) >> 4;
                }
                if p.flip_lr {
                    res.reverse();
                }
                add_row::<P, L>(row, &res, pmax);
            }
        }
        j0 += L;
    }
}

fn row_dispatch<E: Lane>(log2w: u32, kind: Kind, p: &Params, coefs: &[i32], tmp: &mut [i32]) {
    macro_rules! go {
        ($l:literal, $n:literal) => {
            match kind {
                Kind::Dct => row_pass::<E, $l, $n, K_DCT>(p, coefs, tmp),
                Kind::Adst => row_pass::<E, $l, $n, K_ADST>(p, coefs, tmp),
                Kind::Idt => row_pass::<E, $l, $n, K_IDT>(p, coefs, tmp),
            }
        };
        ($l:literal, $n:literal, no_adst) => {
            match kind {
                Kind::Idt => row_pass::<E, $l, $n, K_IDT>(p, coefs, tmp),
                _ => row_pass::<E, $l, $n, K_DCT>(p, coefs, tmp),
            }
        };
    }
    // Four rows at a time only when the block has four: with more, a few
    // non-zero rows still go eight at a time, which is the faster code.
    match (log2w, p.ts == 4) {
        (2, true) if p.ts == 4 => go!(1, 4),
        (2, true) => go!(4, 4),
        (2, false) => go!(8, 4),
        (3, true) => go!(4, 8),
        (3, false) => go!(8, 8),
        (4, true) => go!(4, 16),
        (4, false) => go!(8, 16),
        (5, true) => go!(4, 32, no_adst),
        (5, false) => go!(8, 32, no_adst),
        (_, true) => go!(4, 64, no_adst),
        (_, false) => go!(8, 64, no_adst),
    }
}

fn col_dispatch<P: Pixel, E: Lane>(
    log2h: u32,
    kind: Kind,
    p: &Params,
    tmp: &[i32],
    dst: &mut [P],
    stride: usize,
) {
    macro_rules! go {
        ($l:literal, $n:literal) => {
            match kind {
                Kind::Dct => col_pass::<P, E, $l, $n, K_DCT>(p, tmp, dst, stride),
                Kind::Adst => col_pass::<P, E, $l, $n, K_ADST>(p, tmp, dst, stride),
                Kind::Idt => col_pass::<P, E, $l, $n, K_IDT>(p, tmp, dst, stride),
            }
        };
        ($l:literal, $n:literal, no_adst) => {
            match kind {
                Kind::Idt => col_pass::<P, E, $l, $n, K_IDT>(p, tmp, dst, stride),
                _ => col_pass::<P, E, $l, $n, K_DCT>(p, tmp, dst, stride),
            }
        };
    }
    match (log2h, p.w == 4) {
        (2, true) => go!(1, 4),
        (2, false) => go!(8, 4),
        (3, true) => go!(4, 8),
        (3, false) => go!(8, 8),
        (4, true) => go!(4, 16),
        (4, false) => go!(8, 16),
        (5, true) => go!(4, 32, no_adst),
        (5, false) => go!(8, 32, no_adst),
        (_, true) => go!(4, 64, no_adst),
        (_, false) => go!(8, 64, no_adst),
    }
}

/// Both passes with an intermediate buffer of `TMP` entries (>= w * th).
fn run<P: Pixel, E: Lane, const TMP: usize>(
    dst: &mut [P],
    stride: usize,
    coefs: &[i32],
    tx_size: usize,
    tx_type: usize,
    nz_rows: usize,
    nz_cols: usize,
    bd: u32,
) {
    let log2w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
    let w = 1usize << log2w;
    let rr = bd + 8;
    let cr = (bd + 6).max(16);
    let (flip_ud, flip_lr) = tx_type_flips(tx_type);
    let p = Params {
        w,
        ts: (1usize << log2h).min(32),
        nz_rows,
        nz_cols,
        rect: log2w.abs_diff(log2h) == 1,
        row_shift: TRANSFORM_ROW_SHIFT[tx_size] as u32,
        rmin: -(1i32 << (rr - 1)),
        rmax: (1i32 << (rr - 1)) - 1,
        cmin: -(1i32 << (cr - 1)),
        cmax: (1i32 << (cr - 1)) - 1,
        flip_ud,
        flip_lr,
        pmax: (1i32 << bd) - 1,
    };
    let mut tmp = [0i32; TMP];
    row_dispatch::<E>(log2w, row_kind(tx_type), &p, coefs, &mut tmp);
    col_dispatch::<P, E>(log2h, col_kind(tx_type), &p, &tmp, dst, stride);
}

/// The block of an 8-bit stream on the vector kernel, where there is one
/// and it has the block's transforms. `false` when the block is still to
/// be done.
fn on_kernel<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    coefs: &[i32],
    tx_size: usize,
    tx_type: usize,
    nz_rows: usize,
    nz_cols: usize,
) -> bool {
    use cutlass_simd::av1::itx as kernel;
    if !kernel::ACCELERATED {
        return false;
    }
    let of = |k: Kind| match k {
        Kind::Dct => kernel::Kind::Dct,
        Kind::Adst => kernel::Kind::Adst,
        Kind::Idt => kernel::Kind::Identity,
    };
    let (w, h) = (
        1usize << TX_WIDTH_LOG2[tx_size],
        1usize << TX_HEIGHT_LOG2[tx_size],
    );
    let (row, col) = (of(row_kind(tx_type)), of(col_kind(tx_type)));
    if !kernel::supports(w, h, row, col) {
        return false;
    }
    let Some(dst) = P::bytes_mut(dst) else {
        return false;
    };
    let (flip_ud, flip_lr) = tx_type_flips(tx_type);
    kernel::add(&mut kernel::Block {
        dst,
        stride,
        coefs,
        w,
        h,
        row,
        col,
        nz_rows,
        nz_cols,
        shift: u32::from(TRANSFORM_ROW_SHIFT[tx_size]),
        flip_ud,
        flip_lr,
    })
}

/// `run` with the lane type chosen: 16-bit lanes for 8-bit streams.
fn run_sized<P: Pixel, const TMP: usize>(
    dst: &mut [P],
    stride: usize,
    coefs: &[i32],
    tx_size: usize,
    tx_type: usize,
    nz_rows: usize,
    nz_cols: usize,
    bd: u32,
) {
    if !P::HBD && bd == 8 {
        if on_kernel(dst, stride, coefs, tx_size, tx_type, nz_rows, nz_cols) {
            return;
        }
        run::<P, i16, TMP>(dst, stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd);
    } else {
        run::<P, i32, TMP>(dst, stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd);
    }
}

/// Lossless 4x4 Walsh-Hadamard reconstruction (spec 7.13.2.10).
fn wht_add<P: Pixel>(dst: &mut [P], stride: usize, coefs: &[i32], tx_type: usize, bd: u32) {
    fn wht(t: &mut [i32; 4], shift: u32) {
        let mut a = t[0] >> shift;
        let mut c = t[1] >> shift;
        let mut d = t[2] >> shift;
        let mut b = t[3] >> shift;
        a = a.wrapping_add(c);
        d = d.wrapping_sub(b);
        let e = a.wrapping_sub(d) >> 1;
        b = e.wrapping_sub(b);
        c = e.wrapping_sub(c);
        a = a.wrapping_sub(b);
        d = d.wrapping_add(c);
        *t = [a, b, c, d];
    }
    let cr = (bd + 6).max(16);
    let (cmin, cmax) = (-(1i32 << (cr - 1)), (1i32 << (cr - 1)) - 1);
    let mut res = [[0i32; 4]; 4];
    for i in 0..4 {
        let mut t = [
            coefs[i * 4],
            coefs[i * 4 + 1],
            coefs[i * 4 + 2],
            coefs[i * 4 + 3],
        ];
        wht(&mut t, 2);
        for j in 0..4 {
            res[i][j] = t[j].max(cmin).min(cmax);
        }
    }
    for j in 0..4 {
        let mut t = [res[0][j], res[1][j], res[2][j], res[3][j]];
        wht(&mut t, 0);
        for i in 0..4 {
            res[i][j] = t[i];
        }
    }
    let (flip_ud, flip_lr) = tx_type_flips(tx_type);
    let pmax = (1i32 << bd) - 1;
    for i in 0..4 {
        let y = if flip_ud { 3 - i } else { i };
        let mut r = res[i];
        if flip_lr {
            r.reverse();
        }
        add_row::<P, 4>(&mut dst[y * stride..], &r, pmax);
    }
}

/// Adds the inverse transform of `coefs` to the prediction already in `dst`.
///
/// `coefs` is the spec's Dequant array, row-major with row stride
/// `tw = min(32, w)` and `th = min(32, h)` rows (w, h = transform size),
/// already dequantized and clamped by the caller to the spec's
/// `-(1 << (7 + bd))..(1 << (7 + bd))`. Only the first `nz_rows` rows and
/// `nz_cols` columns may be non-zero (1 <= each; the caller passes the
/// bounding box of non-zero coefficients). On return every entry of
/// `coefs[..tw*th]` is zero, so the caller can reuse the buffer.
///
/// `dst[0]` is the top-left sample of the transform block; `dst` must hold
/// `h` rows of `w` samples at `dst_stride`. `tx_size` and `tx_type` use spec
/// numbering and must be a combination the spec allows (64-point transforms
/// are DCT only; 32-point transforms are DCT or identity). When `lossless` is
/// set the block must be `TX_4X4` and the Walsh-Hadamard transform is used.
pub fn inv_txfm_add<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    coefs: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    nz_rows: usize,
    nz_cols: usize,
    lossless: bool,
    bd: u32,
) {
    if lossless {
        let coefs = &mut coefs[..16];
        wht_add(dst, dst_stride, coefs, tx_type, bd);
        coefs.fill(0);
        return;
    }
    let log2w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let tw = w.min(32);
    let th = h.min(32);
    let nz_rows = nz_rows.clamp(1, th);
    let nz_cols = nz_cols.clamp(1, tw);
    let coefs = &mut coefs[..tw * th];

    if tx_type as i32 == DCT_DCT && nz_rows == 1 && nz_cols == 1 {
        let shift = TRANSFORM_ROW_SHIFT[tx_size] as u32;
        let cr = (bd + 6).max(16);
        let (cmin, cmax) = (-(1i32 << (cr - 1)), (1i32 << (cr - 1)) - 1);
        let mut c = coefs[0];
        coefs[0] = 0;
        if log2w.abs_diff(log2h) == 1 {
            c = rnd12(c.wrapping_mul(2896));
        }
        let v = rnd12(c.wrapping_mul(2896));
        let v = (v.wrapping_add((1i32 << shift) >> 1) >> shift)
            .max(cmin)
            .min(cmax);
        let v = rnd12(v.wrapping_mul(2896)).wrapping_add(8) >> 4;
        // 8-bit samples take the flat residual on the vector kernel.
        if bd == 8
            && cutlass_simd::av1::itx::ACCELERATED
            && let Some(bytes) = P::bytes_mut(dst)
            && cutlass_simd::av1::itx::add_flat(bytes, dst_stride, w, h, v)
        {
            return;
        }
        let pmax = (1i32 << bd) - 1;
        for y in 0..h {
            let row = &mut dst[y * dst_stride..y * dst_stride + w];
            for px in row.iter_mut() {
                *px = P::from_i32((px.to_i32() + v).max(0).min(pmax));
            }
        }
        return;
    }

    let need = w * th;
    if need <= 16 {
        run_sized::<P, 16>(
            dst, dst_stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd,
        );
    } else if need <= 64 {
        run_sized::<P, 64>(
            dst, dst_stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd,
        );
    } else if need <= 256 {
        run_sized::<P, 256>(
            dst, dst_stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd,
        );
    } else if need <= 1024 {
        run_sized::<P, 1024>(
            dst, dst_stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd,
        );
    } else {
        run_sized::<P, 2048>(
            dst, dst_stride, coefs, tx_size, tx_type, nz_rows, nz_cols, bd,
        );
    }
    for i in 0..nz_rows {
        coefs[i * tw..][..nz_cols].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::{inv_txfm_add, tx_type_flips};
    use crate::spec::tables::{
        COS128_LOOKUP, DCT_DCT, IDTX, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9,
        TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2,
    };

    // ---- Literal transliteration of spec 7.13.2 / 7.13.3 in i64. ----

    struct Spec {
        t: Vec<i64>,
        /// Cleared when a bitstream conformance requirement is violated.
        ok: bool,
    }

    fn fits(v: i64, bits: u32) -> bool {
        v >= -(1i64 << (bits - 1)) && v < (1i64 << (bits - 1))
    }

    fn round2(x: i64, n: u32) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    fn brev(num_bits: u32, x: usize) -> usize {
        let mut t = 0;
        for i in 0..num_bits {
            let bit = (x >> i) & 1;
            t += bit << (num_bits - 1 - i);
        }
        t
    }

    fn cos128(angle: i64) -> i64 {
        let a2 = (angle & 255) as usize;
        if a2 <= 64 {
            COS128_LOOKUP[a2] as i64
        } else if a2 <= 128 {
            -(COS128_LOOKUP[128 - a2] as i64)
        } else if a2 <= 192 {
            -(COS128_LOOKUP[a2 - 128] as i64)
        } else {
            COS128_LOOKUP[256 - a2] as i64
        }
    }

    fn sin128(angle: i64) -> i64 {
        cos128(angle - 64)
    }

    impl Spec {
        fn b(&mut self, a: usize, b: usize, angle: i64, flip: bool, r: u32) {
            let x = self.t[a] * cos128(angle) - self.t[b] * sin128(angle);
            let y = self.t[a] * sin128(angle) + self.t[b] * cos128(angle);
            self.t[a] = round2(x, 12);
            self.t[b] = round2(y, 12);
            if !fits(self.t[a], r) || !fits(self.t[b], r) {
                self.ok = false;
            }
            if flip {
                self.t.swap(a, b);
            }
        }

        fn h(&mut self, a: usize, b: usize, flip: bool, r: u32) {
            if flip {
                return self.h(b, a, false, r);
            }
            let x = self.t[a];
            let y = self.t[b];
            let lo = -(1i64 << (r - 1));
            let hi = (1i64 << (r - 1)) - 1;
            self.t[a] = (x + y).clamp(lo, hi);
            self.t[b] = (x - y).clamp(lo, hi);
        }

        fn dct_perm(&mut self, n: u32) {
            let copy = self.t.clone();
            for i in 0..(1usize << n) {
                self.t[i] = copy[brev(n, i)];
            }
        }

        fn dct(&mut self, n: u32, r: u32) {
            self.dct_perm(n);
            if n == 6 {
                for i in 0..16 {
                    self.b(32 + i, 63 - i, 63 - 4 * brev(4, i) as i64, false, r);
                }
            }
            if n >= 5 {
                for i in 0..8 {
                    self.b(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i64) << 3), false, r);
                }
            }
            if n == 6 {
                for i in 0..16 {
                    self.h(32 + i * 2, 33 + i * 2, i & 1 == 1, r);
                }
            }
            if n >= 4 {
                for i in 0..4 {
                    self.b(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i64) << 4), false, r);
                }
            }
            if n >= 5 {
                for i in 0..8 {
                    self.h(16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
                }
            }
            if n == 6 {
                for i in 0..4 {
                    for j in 0..2 {
                        let ang = 60 - 16 * brev(2, i) as i64 + 64 * j as i64;
                        self.b(62 - i * 4 - j, 33 + i * 4 + j, ang, true, r);
                    }
                }
            }
            if n >= 3 {
                for i in 0..2 {
                    self.b(4 + i, 7 - i, 56 - 32 * i as i64, false, r);
                }
            }
            if n >= 4 {
                for i in 0..4 {
                    self.h(8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
                }
            }
            if n >= 5 {
                for i in 0..2 {
                    for j in 0..2 {
                        let ang = 24 + ((j as i64) << 6) + ((1 - i as i64) << 5);
                        self.b(30 - 4 * i - j, 17 + 4 * i + j, ang, true, r);
                    }
                }
            }
            if n == 6 {
                for i in 0..8 {
                    for j in 0..2 {
                        self.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
                    }
                }
            }
            for i in 0..2 {
                self.b(2 * i, 2 * i + 1, 32 + 16 * i as i64, 1 - i == 1, r);
            }
            if n >= 3 {
                for i in 0..2 {
                    self.h(4 + 2 * i, 5 + 2 * i, i == 1, r);
                }
            }
            if n >= 4 {
                for i in 0..2 {
                    self.b(14 - i, 9 + i, 48 + 64 * i as i64, true, r);
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    for j in 0..2 {
                        self.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
                    }
                }
            }
            if n == 6 {
                for i in 0..2 {
                    for j in 0..4 {
                        let ang = 56 - i as i64 * 32 + (j as i64 >> 1) * 64;
                        self.b(61 - i * 8 - j, 34 + i * 8 + j, ang, true, r);
                    }
                }
            }
            for i in 0..2 {
                self.h(i, 3 - i, false, r);
            }
            if n >= 3 {
                self.b(6, 5, 32, true, r);
            }
            if n >= 4 {
                for i in 0..2 {
                    for j in 0..2 {
                        self.h(8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
                    }
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    self.b(29 - i, 18 + i, 48 + (i as i64 >> 1) * 64, true, r);
                }
            }
            if n == 6 {
                for i in 0..4 {
                    for j in 0..4 {
                        self.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
                    }
                }
            }
            if n >= 3 {
                for i in 0..4 {
                    self.h(i, 7 - i, false, r);
                }
            }
            if n >= 4 {
                for i in 0..2 {
                    self.b(13 - i, 10 + i, 32, true, r);
                }
            }
            if n >= 5 {
                for i in 0..2 {
                    for j in 0..4 {
                        self.h(16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
                    }
                }
            }
            if n == 6 {
                for i in 0..8 {
                    self.b(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true, r);
                }
            }
            if n >= 4 {
                for i in 0..8 {
                    self.h(i, 15 - i, false, r);
                }
            }
            if n >= 5 {
                for i in 0..4 {
                    self.b(27 - i, 20 + i, 32, true, r);
                }
            }
            if n == 6 {
                for i in 0..8 {
                    self.h(32 + i, 47 - i, false, r);
                    self.h(48 + i, 63 - i, true, r);
                }
            }
            if n >= 5 {
                for i in 0..16 {
                    self.h(i, 31 - i, false, r);
                }
            }
            if n == 6 {
                for i in 0..8 {
                    self.b(55 - i, 40 + i, 32, true, r);
                }
            }
            if n == 6 {
                for i in 0..32 {
                    self.h(i, 63 - i, false, r);
                }
            }
        }

        fn adst_in_perm(&mut self, n: u32) {
            let n0 = 1usize << n;
            let copy = self.t.clone();
            for i in 0..n0 {
                let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
                self.t[i] = copy[idx];
            }
        }

        fn adst_out_perm(&mut self, n: u32) {
            let n0 = 1usize << n;
            let copy = self.t.clone();
            for i in 0..n0 {
                let a = (i >> 3) & 1;
                let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
                let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
                let d = (i & 1) ^ ((i >> 1) & 1);
                let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
                self.t[i] = if i & 1 == 1 { -copy[idx] } else { copy[idx] };
            }
        }

        fn adst4(&mut self, r: u32) {
            let t = &self.t;
            let (s1_9, s2_9, s3_9, s4_9) = (
                SINPI_1_9 as i64,
                SINPI_2_9 as i64,
                SINPI_3_9 as i64,
                SINPI_4_9 as i64,
            );
            let mut s = [0i64; 7];
            let mut x = [0i64; 4];
            let mut ok = true;
            let mut chk = |v: i64, bits: u32| ok &= fits(v, bits);
            s[0] = s1_9 * t[0];
            s[1] = s2_9 * t[0];
            s[2] = s3_9 * t[1];
            s[3] = s4_9 * t[2];
            s[4] = s1_9 * t[2];
            s[5] = s2_9 * t[3];
            s[6] = s4_9 * t[3];
            let a7 = t[0] - t[2];
            chk(a7, r + 1);
            let b7 = a7 + t[3];
            chk(b7, r);
            for v in s {
                chk(v, r + 12);
            }
            s[0] += s[3];
            s[1] -= s[4];
            s[3] = s[2];
            s[2] = s3_9 * b7;
            s[0] += s[5];
            s[1] -= s[6];
            for v in s {
                chk(v, r + 12);
            }
            x[0] = s[0] + s[3];
            x[1] = s[1] + s[3];
            x[2] = s[2];
            x[3] = s[0] + s[1];
            chk(x[3], r + 12);
            x[3] -= s[3];
            for v in x {
                chk(v, r + 12);
            }
            for i in 0..4 {
                self.t[i] = round2(x[i], 12);
            }
            self.ok &= ok;
        }

        fn adst8(&mut self, r: u32) {
            self.adst_in_perm(3);
            for i in 0..4 {
                self.b(2 * i, 2 * i + 1, 60 - 16 * i as i64, true, r);
            }
            for i in 0..4 {
                self.h(i, 4 + i, false, r);
            }
            for i in 0..2 {
                self.b(4 + 3 * i, 5 + i, 48 - 32 * i as i64, true, r);
            }
            for i in 0..2 {
                for j in 0..2 {
                    self.h(4 * j + i, 2 + 4 * j + i, false, r);
                }
            }
            for i in 0..2 {
                self.b(2 + 4 * i, 3 + 4 * i, 32, true, r);
            }
            self.adst_out_perm(3);
        }

        fn adst16(&mut self, r: u32) {
            self.adst_in_perm(4);
            for i in 0..8 {
                self.b(2 * i, 2 * i + 1, 62 - 8 * i as i64, true, r);
            }
            for i in 0..8 {
                self.h(i, 8 + i, false, r);
            }
            for i in 0..2 {
                self.b(8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i64, true, r);
                self.b(13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i64, true, r);
            }
            for i in 0..4 {
                for j in 0..2 {
                    self.h(8 * j + i, 4 + 8 * j + i, false, r);
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    self.b(
                        4 + 8 * j + 3 * i,
                        5 + 8 * j + i,
                        48 - 32 * i as i64,
                        true,
                        r,
                    );
                }
            }
            for i in 0..2 {
                for j in 0..4 {
                    self.h(4 * j + i, 2 + 4 * j + i, false, r);
                }
            }
            for i in 0..4 {
                self.b(2 + 4 * i, 3 + 4 * i, 32, true, r);
            }
            self.adst_out_perm(4);
        }

        fn adst(&mut self, n: u32, r: u32) {
            match n {
                2 => self.adst4(r),
                3 => self.adst8(r),
                _ => self.adst16(r),
            }
        }

        fn wht(&mut self, shift: u32) {
            let t = &mut self.t;
            let mut a = t[0] >> shift;
            let mut c = t[1] >> shift;
            let mut d = t[2] >> shift;
            let mut b = t[3] >> shift;
            a += c;
            d -= b;
            let e = (a - d) >> 1;
            b = e - b;
            c = e - c;
            a -= b;
            d += c;
            t[0] = a;
            t[1] = b;
            t[2] = c;
            t[3] = d;
        }

        fn identity(&mut self, n: u32) {
            for i in 0..(1usize << n) {
                self.t[i] = match n {
                    2 => round2(self.t[i] * 5793, 12),
                    3 => self.t[i] * 2,
                    4 => round2(self.t[i] * 11586, 12),
                    _ => self.t[i] * 4,
                };
            }
        }
    }

    /// Spec 7.13.3 plus step 3 of 7.12.3. Returns None when the input breaks a
    /// conformance requirement.
    fn spec_recon(
        pred: &[i32],
        stride: usize,
        dq: &[i32],
        tx_size: usize,
        tx_type: usize,
        lossless: bool,
        bd: u32,
    ) -> Option<Vec<i32>> {
        let log2w = TX_WIDTH_LOG2[tx_size] as u32;
        let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let tw = w.min(32);
        let row_shift = if lossless {
            0
        } else {
            TRANSFORM_ROW_SHIFT[tx_size] as u32
        };
        let col_shift = if lossless { 0 } else { 4 };
        let rr = bd + 8;
        let cr = (bd + 6).max(16);
        let t = tx_type as i32;
        let row_dct = matches!(t, 0 | 1 | 4 | 11);
        let row_adst = matches!(t, 2 | 3 | 5 | 6 | 7 | 8 | 13 | 15);
        let col_dct = matches!(t, 0 | 2 | 5 | 10);
        let col_adst = matches!(t, 1 | 3 | 4 | 6 | 7 | 8 | 12 | 14);
        let mut s = Spec {
            t: vec![0; 64],
            ok: true,
        };
        let mut res = vec![0i64; w * h];
        for i in 0..h {
            for j in 0..w {
                s.t[j] = if i < 32 && j < 32 {
                    dq[i * tw + j] as i64
                } else {
                    0
                };
            }
            if log2w.abs_diff(log2h) == 1 {
                for j in 0..w {
                    s.t[j] = round2(s.t[j] * 2896, 12);
                }
            }
            if lossless {
                s.wht(2);
            } else if row_dct {
                s.dct(log2w, rr);
            } else if row_adst {
                s.adst(log2w, rr);
            } else {
                s.identity(log2w);
            }
            for j in 0..w {
                res[i * w + j] = round2(s.t[j], row_shift);
            }
        }
        let lo = -(1i64 << (cr - 1));
        let hi = (1i64 << (cr - 1)) - 1;
        for v in res.iter_mut() {
            *v = (*v).clamp(lo, hi);
        }
        for j in 0..w {
            for i in 0..h {
                s.t[i] = res[i * w + j];
            }
            if lossless {
                s.wht(0);
            } else if col_dct {
                s.dct(log2h, cr);
            } else if col_adst {
                s.adst(log2h, cr);
            } else {
                s.identity(log2h);
            }
            for i in 0..h {
                res[i * w + j] = round2(s.t[i], col_shift);
            }
        }
        if !s.ok {
            return None;
        }
        let (flip_ud, flip_lr) = tx_type_flips(tx_type);
        let mut out = pred.to_vec();
        for i in 0..h {
            for j in 0..w {
                let xx = if flip_lr { w - j - 1 } else { j };
                let yy = if flip_ud { h - i - 1 } else { i };
                let p = &mut out[yy * stride + xx];
                *p = (*p as i64 + res[i * w + j]).clamp(0, (1 << bd) - 1) as i32;
            }
        }
        Some(out)
    }

    // ---- Random testing. ----

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + self.below((hi - lo + 1) as u64) as i64
        }
    }

    fn allowed_types(tx_size: usize) -> Vec<usize> {
        let w = 1 << TX_WIDTH_LOG2[tx_size];
        let h = 1 << TX_HEIGHT_LOG2[tx_size];
        let m = w.max(h);
        if m == 64 {
            vec![DCT_DCT as usize]
        } else if m == 32 {
            vec![DCT_DCT as usize, IDTX as usize]
        } else {
            (0..16).collect()
        }
    }

    fn check<P: super::Pixel>(
        rng: &mut Rng,
        tx_size: usize,
        tx_type: usize,
        lossless: bool,
        bd: u32,
        mode: u32,
    ) {
        let w = 1usize << TX_WIDTH_LOG2[tx_size];
        let h = 1usize << TX_HEIGHT_LOG2[tx_size];
        let tw = w.min(32);
        let th = h.min(32);
        let stride = w + rng.below(9) as usize;
        let pred: Vec<i32> = (0..stride * h).map(|_| rng.below(1 << bd) as i32).collect();
        let (nz_rows, nz_cols) = match mode {
            0 => (1, 1),
            1 => (1, 1 + rng.below(tw as u64) as usize),
            2 => (1 + rng.below(th as u64) as usize, 1),
            _ => (
                1 + rng.below(th as u64) as usize,
                1 + rng.below(tw as u64) as usize,
            ),
        };
        let lim = 1i64 << (7 + bd);
        let mut mag = match rng.below(4) {
            0 => lim,
            1 => lim >> 4,
            2 => lim >> 8,
            _ => 64,
        };
        let sparse = rng.below(3) == 0;
        let mut dq = vec![0i32; tw * th];
        for i in 0..nz_rows {
            for j in 0..nz_cols {
                if !sparse || rng.below(8) == 0 || (i == 0 && j == 0) {
                    dq[i * tw + j] = rng.range(-lim, lim - 1) as i32;
                }
            }
        }
        // Scale down until the input is conformant.
        let scale =
            |mag: i64| -> Vec<i32> { dq.iter().map(|&d| (d as i64 * mag / lim) as i32).collect() };
        let mut base = scale(mag);
        let expect = loop {
            if let Some(e) = spec_recon(&pred, stride, &base, tx_size, tx_type, lossless, bd) {
                break e;
            }
            mag /= 2;
            base = scale(mag);
        };
        let mut dst: Vec<P> = pred.iter().map(|&v| P::from_i32(v)).collect();
        let mut coefs = base.clone();
        inv_txfm_add(
            &mut dst, stride, &mut coefs, tx_size, tx_type, nz_rows, nz_cols, lossless, bd,
        );
        let got: Vec<i32> = dst.iter().map(|p| p.to_i32()).collect();
        assert!(
            coefs.iter().all(|&c| c == 0),
            "coefs not cleared: size {tx_size} type {tx_type}"
        );
        assert_eq!(
            got, expect,
            "mismatch: bd {bd} size {tx_size} type {tx_type} lossless {lossless} \
             nz {nz_rows}x{nz_cols} coefs {base:?}"
        );
    }

    fn run_bd(bd: u32, iters: usize) {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ bd as u64);
        for tx_size in 0..19 {
            for tx_type in allowed_types(tx_size) {
                for it in 0..iters {
                    let mode = (it % 5) as u32;
                    if bd == 8 {
                        check::<u8>(&mut rng, tx_size, tx_type, false, bd, mode);
                    } else {
                        check::<u16>(&mut rng, tx_size, tx_type, false, bd, mode);
                    }
                }
            }
        }
        for tx_type in 0..16 {
            for it in 0..iters {
                let mode = (it % 5) as u32;
                if bd == 8 {
                    check::<u8>(&mut rng, 0, tx_type, true, bd, mode);
                } else {
                    check::<u16>(&mut rng, 0, tx_type, true, bd, mode);
                }
            }
        }
    }

    #[test]
    fn the_kernel_crate_has_the_cosine_table_of_the_spec() {
        assert_eq!(cutlass_simd::av1::itx::steps::COS128, COS128_LOOKUP);
    }

    #[test]
    fn matches_spec_8bit() {
        run_bd(8, 200);
    }

    /// 8-bit samples held in `u16` take the 32-bit lane path.
    #[test]
    fn matches_spec_8bit_in_u16() {
        let mut rng = Rng(77);
        for tx_size in 0..19 {
            for tx_type in allowed_types(tx_size) {
                for it in 0..40 {
                    check::<u16>(&mut rng, tx_size, tx_type, false, 8, (it % 5) as u32);
                }
            }
        }
    }

    #[test]
    fn matches_spec_10bit() {
        run_bd(10, 200);
    }

    #[test]
    fn matches_spec_12bit() {
        run_bd(12, 200);
    }

    /// Non-conformant extreme input must not panic (also in debug builds).
    #[test]
    fn extreme_inputs_do_not_panic() {
        let mut rng = Rng(12345);
        for bd in [8u32, 10, 12] {
            let lim = 1i64 << (7 + bd);
            for tx_size in 0..19 {
                let w = 1usize << TX_WIDTH_LOG2[tx_size];
                let h = 1usize << TX_HEIGHT_LOG2[tx_size];
                let (tw, th) = (w.min(32), h.min(32));
                for tx_type in 0..16 {
                    for lossless in [false, true] {
                        let mut coefs: Vec<i32> = (0..tw * th)
                            .map(|_| {
                                if rng.below(2) == 0 {
                                    (lim - 1) as i32
                                } else {
                                    -lim as i32
                                }
                            })
                            .collect();
                        let mut dst = vec![0u16; w * h];
                        inv_txfm_add(
                            &mut dst, w, &mut coefs, tx_size, tx_type, th, tw, lossless, bd,
                        );
                        assert!(dst.iter().all(|&p| (p as i32) < (1 << bd)));
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench() {
        // (name, tx_size, tx_type, nz_rows, nz_cols, bd)
        let cases = [
            ("4x4 dct", 0, 0, 4, 4, 8),
            ("8x8 dct", 1, 0, 8, 8, 8),
            ("16x16 dct", 2, 0, 16, 16, 8),
            ("32x32 dct", 3, 0, 32, 32, 8),
            ("64x64 dct", 4, 0, 32, 32, 8),
            ("8x8 adst", 1, 3, 8, 8, 8),
            ("16x16 dct 4x4nz", 2, 0, 4, 4, 8),
            ("32x32 dct 8x8nz", 3, 0, 8, 8, 8),
            ("8x8 dct 10b", 1, 0, 8, 8, 10),
            ("16x16 dct 10b", 2, 0, 16, 16, 10),
            ("8x8 idtx", 1, 9, 8, 8, 8),
            ("4x4 dc", 0, 0, 1, 1, 8),
        ];
        let only: Option<usize> = std::env::var("ITX_CASE").ok().and_then(|v| v.parse().ok());
        let px: usize = std::env::var("ITX_PX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_000_000);
        let mut rng = Rng(7);
        for (idx, &(name, tx_size, tx_type, nzr, nzc, bd)) in cases.iter().enumerate() {
            if only.is_some_and(|o| o != idx) {
                continue;
            }
            let w = 1usize << TX_WIDTH_LOG2[tx_size];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size];
            let (tw, th) = (w.min(32), h.min(32));
            let n = px / (w * h);
            let mut src = vec![0i32; tw * th];
            for i in 0..nzr {
                for j in 0..nzc {
                    src[i * tw + j] = rng.range(-200, 200) as i32;
                }
            }
            let mut coefs = src.clone();
            let t0 = std::time::Instant::now();
            if bd == 8 {
                let mut dst = vec![128u8; w * h];
                for _ in 0..n {
                    coefs.copy_from_slice(&src);
                    inv_txfm_add(
                        &mut dst, w, &mut coefs, tx_size, tx_type, nzr, nzc, false, bd,
                    );
                }
            } else {
                let mut dst = vec![512u16; w * h];
                for _ in 0..n {
                    coefs.copy_from_slice(&src);
                    inv_txfm_add(
                        &mut dst, w, &mut coefs, tx_size, tx_type, nzr, nzc, false, bd,
                    );
                }
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            println!(
                "{name}: {ns:.0} ns/block, {:.2} ns/px, {n} blocks",
                ns / (w * h) as f64
            );
        }
    }

    #[test]
    fn flips() {
        assert_eq!(tx_type_flips(0), (false, false));
        assert_eq!(tx_type_flips(4), (true, false));
        assert_eq!(tx_type_flips(5), (false, true));
        assert_eq!(tx_type_flips(6), (true, true));
        assert_eq!(tx_type_flips(14), (true, false));
        assert_eq!(tx_type_flips(15), (false, true));
    }
}
