//! Forward transforms: the counterpart of `dsp::itx`.
//!
//! `fwd_txfm` turns a residual block into coefficients in the spec's
//! "Dequant" domain: passed unchanged to `inv_txfm_add` over a zero
//! prediction they give the residual back, up to rounding. The design is
//! libaom's `av1_fwd_txfm2d`: a column pass, a rounding shift, a row pass,
//! a final rounding shift with the `1 / sqrt(2)` factor of 2:1 rectangles.
//!
//! Each 1D DCT or ADST is the transpose of the decoder's inverse: the spec's
//! butterfly network (7.13.2) run backwards with every step transposed. A
//! Hadamard step is symmetric and a rotation transposes to the rotation by the
//! opposite angle, so the forward transform has exactly the scale of libaom's
//! `fdct`/`fadst` (`sqrt(n / 2)` times orthonormal). The ADST4 is libaom's
//! `fadst4` and the identities scale like the inverse ones. The networks are
//! built at compile time; steps that only feed discarded outputs (the high
//! half of a 64-point DCT) are pruned.
//!
//! Every pass works on `L` columns (or rows) at once as `[i32; L]` lanes, so
//! each step is a short loop across lanes that LLVM vectorizes. All-zero
//! groups of columns or rows are skipped.
//!
//! Scaling: the inverse has gain `2^-(rowShift + 4) * sqrt(w h) / 2 * rect`,
//! a power of two. The forward shifts the input up by 2 and down by
//! `d1 + d2 = log2(w) + log2(h) - rect - rowShift - 4` bits in total, `d1`
//! after the column pass and `d2` (with the rectangular factor) at the end.
//!
//! Two arithmetic paths: 8-bit blocks of up to 32 points multiply in `i32`
//! with the spec's 12-bit constants (the input is clamped to the residual
//! range, which bounds every intermediate, and `d1` is just large enough to
//! keep the row pass in range). Deeper blocks and 64-point blocks multiply in
//! `i64` with 16-bit constants and keep full precision until the final shift;
//! with 12-bit constants full-range 12-bit content would miss the round-trip
//! error budget.

use crate::dsp::itx::tx_type_flips;
use crate::spec::tables::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, COS128_LOOKUP, DCT_ADST, DCT_DCT, DCT_FLIPADST,
    FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, SINPI_1_9,
    SINPI_2_9, SINPI_3_9, SINPI_4_9, TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2, V_ADST,
    V_DCT, V_FLIPADST,
};

// ---------------------------------------------------------------------------
// Transform kinds.

const K_DCT: u8 = 0;
const K_ADST: u8 = 1;
const K_IDT: u8 = 2;

/// The 1D transform applied to each row (horizontal).
fn row_kind(tx_type: usize) -> u8 {
    match tx_type as i32 {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => K_DCT,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | H_ADST | H_FLIPADST => K_ADST,
        _ => K_IDT,
    }
}

/// The 1D transform applied to each column (vertical).
fn col_kind(tx_type: usize) -> u8 {
    match tx_type as i32 {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => K_DCT,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | V_ADST | V_FLIPADST => K_ADST,
        _ => K_IDT,
    }
}

// ---------------------------------------------------------------------------
// Permutations (spec 7.13.2.2, 7.13.2.4, 7.13.2.5), computed at compile time.

const fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    let mut i = 0;
    while i < num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
        i += 1;
    }
    t
}

/// Position in T of DCT coefficient `k`.
const fn dct_perm<const N: usize>() -> [u8; N] {
    let n = N.trailing_zeros();
    let mut p = [0u8; N];
    let mut k = 0;
    while k < N {
        p[k] = brev(n, k) as u8;
        k += 1;
    }
    p
}

/// Position in T of ADST8/16 coefficient `k`.
const fn adst_in_perm<const N: usize>() -> [u8; N] {
    let mut p = [0u8; N];
    let mut k = 0;
    while k < N {
        p[k] = if k & 1 == 0 { k + 1 } else { N - 1 - k } as u8;
        k += 1;
    }
    p
}

/// Position in T of ADST8/16 sample `i`; odd samples are negated.
const fn adst_out_perm<const N: usize>() -> [u8; N] {
    let n = N.trailing_zeros();
    let mut p = [0u8; N];
    let mut i = 0;
    while i < N {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        let d = (i & 1) ^ ((i >> 1) & 1);
        p[i] = (((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n)) as u8;
        i += 1;
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

// ---------------------------------------------------------------------------
// Butterfly networks.

#[inline(always)]
const fn cos128(angle: i32) -> i32 {
    let a2 = (angle & 255) as usize;
    if a2 <= 64 {
        COS128_LOOKUP[a2] as i32
    } else if a2 <= 128 {
        -(COS128_LOOKUP[128 - a2] as i32)
    } else if a2 <= 192 {
        -(COS128_LOOKUP[a2 - 128] as i32)
    } else {
        COS128_LOOKUP[256 - a2] as i32
    }
}

#[inline(always)]
const fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

/// `cos(i * pi / 128)` with 16 fractional bits, for the 64-bit path.
const COS128_HI: [i32; 65] = [
    65536, 65516, 65457, 65358, 65220, 65043, 64827, 64571, 64277, 63944, 63572, 63162, 62714,
    62228, 61705, 61145, 60547, 59914, 59244, 58538, 57798, 57022, 56212, 55368, 54491, 53581,
    52639, 51665, 50660, 49624, 48559, 47464, 46341, 45190, 44011, 42806, 41576, 40320, 39040,
    37736, 36410, 35062, 33692, 32303, 30893, 29466, 28020, 26558, 25080, 23586, 22078, 20557,
    19024, 17479, 15924, 14359, 12785, 11204, 9616, 8022, 6424, 4821, 3216, 1608, 0,
];

/// `cos128` with 16 fractional bits.
const fn cos128_hi(angle: i32) -> i32 {
    let a2 = (angle & 255) as usize;
    if a2 <= 64 {
        COS128_HI[a2]
    } else if a2 <= 128 {
        -COS128_HI[128 - a2]
    } else if a2 <= 192 {
        -COS128_HI[a2 - 128]
    } else {
        COS128_HI[256 - a2]
    }
}

/// The ADST4 constants `SINPI_k_9` with 16 fractional bits.
const SINPI_HI: [i32; 4] = [21133, 39717, 53510, 60849];

/// One step of the inverse network: the butterfly `B(a, b, angle, flip)` or
/// the Hadamard rotation `H(a, b, flip)` (spec 7.13.2.1).
#[derive(Clone, Copy)]
struct Op {
    had: bool,
    flip: bool,
    a: u8,
    b: u8,
    cos: i16,
    sin: i16,
    /// The same with 16 fractional bits.
    cos_hi: i32,
    sin_hi: i32,
}

struct Prog<const M: usize> {
    ops: [Op; M],
    len: usize,
}

impl<const M: usize> Prog<M> {
    const fn new() -> Self {
        let nop = Op {
            had: false,
            flip: false,
            a: 0,
            b: 0,
            cos: 0,
            sin: 0,
            cos_hi: 0,
            sin_hi: 0,
        };
        Prog {
            ops: [nop; M],
            len: 0,
        }
    }

    const fn push(mut self, op: Op) -> Self {
        if self.len < M {
            self.ops[self.len] = op;
        }
        self.len += 1;
        self
    }

    const fn bf(self, a: usize, b: usize, angle: usize, flip: bool) -> Self {
        let angle = (angle & 255) as i32;
        self.push(Op {
            had: false,
            flip,
            a: a as u8,
            b: b as u8,
            cos: cos128(angle) as i16,
            sin: sin128(angle) as i16,
            cos_hi: cos128_hi(angle),
            sin_hi: cos128_hi(angle - 64),
        })
    }

    const fn h(self, a: usize, b: usize, flip: bool) -> Self {
        self.push(Op {
            had: true,
            flip,
            a: a as u8,
            b: b as u8,
            cos: 0,
            sin: 0,
            cos_hi: 0,
            sin_hi: 0,
        })
    }
}

/// The inverse DCT of length `1 << n` (spec 7.13.2.3, steps 2 to 31).
const fn dct_prog<const M: usize>(n: u32) -> Prog<M> {
    let mut p = Prog::<M>::new();
    let mut i;
    let mut j;
    if n == 6 {
        i = 0;
        while i < 16 {
            p = p.bf(32 + i, 63 - i, 63 - 4 * brev(4, i), false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 8 {
            p = p.bf(16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), false);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 16 {
            p = p.h(32 + i * 2, 33 + i * 2, i & 1 != 0);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 4 {
            p = p.bf(8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 8 {
            p = p.h(16 + 2 * i, 17 + 2 * i, i & 1 != 0);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 2 {
                p = p.bf(
                    62 - i * 4 - j,
                    33 + i * 4 + j,
                    60 - 16 * brev(2, i) + 64 * j,
                    true,
                );
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 3 {
        i = 0;
        while i < 2 {
            p = p.bf(4 + i, 7 - i, 56 - 32 * i, false);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 4 {
            p = p.h(8 + 2 * i, 9 + 2 * i, i & 1 != 0);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 2 {
                p = p.bf(
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + (j << 6) + ((1 - i) << 5),
                    true,
                );
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            j = 0;
            while j < 2 {
                p = p.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1 != 0);
                j += 1;
            }
            i += 1;
        }
    }
    i = 0;
    while i < 2 {
        p = p.bf(2 * i, 2 * i + 1, 32 + 16 * i, i == 0);
        i += 1;
    }
    if n >= 3 {
        i = 0;
        while i < 2 {
            p = p.h(4 + 2 * i, 5 + 2 * i, i != 0);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            p = p.bf(14 - i, 9 + i, 48 + 64 * i, true);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 2 {
                p = p.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1 != 0);
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 4 {
                p = p.bf(
                    61 - i * 8 - j,
                    34 + i * 8 + j,
                    56 + 256 - i * 32 + (j >> 1) * 64,
                    true,
                );
                j += 1;
            }
            i += 1;
        }
    }
    i = 0;
    while i < 2 {
        p = p.h(i, 3 - i, false);
        i += 1;
    }
    if n >= 3 {
        p = p.bf(6, 5, 32, true);
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 2 {
                p = p.h(8 + 4 * i + j, 11 + 4 * i - j, i != 0);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            p = p.bf(29 - i, 18 + i, 48 + (i >> 1) * 64, true);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 4 {
            j = 0;
            while j < 4 {
                p = p.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1 != 0);
                j += 1;
            }
            i += 1;
        }
    }
    if n >= 3 {
        i = 0;
        while i < 4 {
            p = p.h(i, 7 - i, false);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 2 {
            p = p.bf(13 - i, 10 + i, 32, true);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 2 {
            j = 0;
            while j < 4 {
                p = p.h(16 + i * 8 + j, 23 + i * 8 - j, i != 0);
                j += 1;
            }
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            p = p.bf(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            i += 1;
        }
    }
    if n >= 4 {
        i = 0;
        while i < 8 {
            p = p.h(i, 15 - i, false);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 4 {
            p = p.bf(27 - i, 20 + i, 32, true);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            p = p.h(32 + i, 47 - i, false);
            p = p.h(48 + i, 63 - i, true);
            i += 1;
        }
    }
    if n >= 5 {
        i = 0;
        while i < 16 {
            p = p.h(i, 31 - i, false);
            i += 1;
        }
    }
    if n == 6 {
        i = 0;
        while i < 8 {
            p = p.bf(55 - i, 40 + i, 32, true);
            i += 1;
        }
        i = 0;
        while i < 32 {
            p = p.h(i, 63 - i, false);
            i += 1;
        }
    }
    p
}

/// The inverse ADST8 (spec 7.13.2.7, steps 2 to 6).
const fn adst8_prog<const M: usize>() -> Prog<M> {
    let mut p = Prog::<M>::new();
    let mut i = 0;
    while i < 4 {
        p = p.bf(2 * i, 2 * i + 1, 60 - 16 * i, true);
        i += 1;
    }
    i = 0;
    while i < 4 {
        p = p.h(i, 4 + i, false);
        i += 1;
    }
    i = 0;
    while i < 2 {
        p = p.bf(4 + 3 * i, 5 + i, 48 - 32 * i, true);
        i += 1;
    }
    i = 0;
    while i < 2 {
        let mut j = 0;
        while j < 2 {
            p = p.h(4 * j + i, 2 + 4 * j + i, false);
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < 2 {
        p = p.bf(2 + 4 * i, 3 + 4 * i, 32, true);
        i += 1;
    }
    p
}

/// The inverse ADST16 (spec 7.13.2.8, steps 2 to 8).
const fn adst16_prog<const M: usize>() -> Prog<M> {
    let mut p = Prog::<M>::new();
    let mut i = 0;
    let mut j;
    while i < 8 {
        p = p.bf(2 * i, 2 * i + 1, 62 - 8 * i, true);
        i += 1;
    }
    i = 0;
    while i < 8 {
        p = p.h(i, 8 + i, false);
        i += 1;
    }
    i = 0;
    while i < 2 {
        p = p.bf(8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true);
        p = p.bf(13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true);
        i += 1;
    }
    i = 0;
    while i < 4 {
        j = 0;
        while j < 2 {
            p = p.h(8 * j + i, 4 + 8 * j + i, false);
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < 2 {
        j = 0;
        while j < 2 {
            p = p.bf(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true);
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < 2 {
        j = 0;
        while j < 4 {
            p = p.h(4 * j + i, 2 + 4 * j + i, false);
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < 4 {
        p = p.bf(2 + 4 * i, 3 + 4 * i, 32, true);
        i += 1;
    }
    p
}

/// The forward network for the inverse network `inv` whose coefficient `k`
/// sits at `perm[k]`, computing only coefficients `0..nout`: the inverse
/// steps that touch a live coefficient, in reverse order. (The forward steps
/// are the transposes; `run_fwd` applies them.)
const fn fwd_prog<const M: usize>(inv: Prog<1024>, perm: &[u8], nout: usize) -> Prog<M> {
    let mut live = [false; 64];
    let mut k = 0;
    while k < nout && k < perm.len() {
        live[perm[k] as usize] = true;
        k += 1;
    }
    let mut kept = Prog::<1024>::new();
    let mut i = 0;
    while i < inv.len {
        let op = inv.ops[i];
        let (a, b) = (op.a as usize, op.b as usize);
        if live[a] || live[b] {
            live[a] = true;
            live[b] = true;
            kept = kept.push(op);
        }
        i += 1;
    }
    let mut p = Prog::<M>::new();
    let mut i = kept.len;
    while i > 0 {
        i -= 1;
        p = p.push(kept.ops[i]);
    }
    p
}

const fn fdct_prog<const M: usize>(n: u32) -> Prog<M> {
    let perm: &[u8] = match n {
        2 => &dct_perm::<4>(),
        3 => &dct_perm::<8>(),
        4 => &dct_perm::<16>(),
        5 => &dct_perm::<32>(),
        _ => &dct_perm::<64>(),
    };
    // At most 32 coefficients of a 64-point transform are kept.
    fwd_prog::<M>(dct_prog::<1024>(n), perm, 32)
}

const fn fadst_prog<const M: usize>(n: u32) -> Prog<M> {
    if n == 3 {
        fwd_prog::<M>(adst8_prog::<1024>(), &adst_in_perm::<8>(), 8)
    } else {
        fwd_prog::<M>(adst16_prog::<1024>(), &adst_in_perm::<16>(), 16)
    }
}

macro_rules! prog_table {
    ($name:ident, $f:ident, $arg:expr) => {
        static $name: [Op; $f::<1024>($arg).len] = $f::<{ $f::<1024>($arg).len }>($arg).ops;
    };
}

prog_table!(FDCT4, fdct_prog, 2);
prog_table!(FDCT8, fdct_prog, 3);
prog_table!(FDCT16, fdct_prog, 4);
prog_table!(FDCT32, fdct_prog, 5);
prog_table!(FDCT64, fdct_prog, 6);
prog_table!(FADST8, fadst_prog, 3);
prog_table!(FADST16, fadst_prog, 4);

// ---------------------------------------------------------------------------
// Arithmetic.

// The constants have 12 fractional bits (the spec's) in 32-bit arithmetic
// and 16 in 64-bit arithmetic, where the extra precision is free and keeps
// full-range 12-bit content within the round-trip error budget.

/// `Round2(x * c + y * s, 16 or 12)`, in `i64` when `WIDE`.
#[inline(always)]
fn mul2<const WIDE: bool>(x: i32, c: i32, y: i32, s: i32) -> i32 {
    if WIDE {
        ((x as i64 * c as i64 + y as i64 * s as i64 + (1 << 15)) >> 16) as i32
    } else {
        x.wrapping_mul(c)
            .wrapping_add(y.wrapping_mul(s))
            .wrapping_add(2048)
            >> 12
    }
}

/// `Round2(x * c, 16 or 12)`, in `i64` when `WIDE`.
#[inline(always)]
fn mul1<const WIDE: bool>(x: i32, c: i32) -> i32 {
    if WIDE {
        ((x as i64 * c as i64 + (1 << 15)) >> 16) as i32
    } else {
        x.wrapping_mul(c).wrapping_add(2048) >> 12
    }
}

/// Runs a forward network: each inverse step transposed, in table order.
#[inline(always)]
fn run_fwd<const L: usize, const N: usize, const WIDE: bool>(t: &mut [[i32; L]; N], ops: &[Op]) {
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
            // [[1, 1], [1, -1]] is its own transpose.
            let (a, b) = if op.flip { (b, a) } else { (a, b) };
            let x = t[a];
            let y = t[b];
            let mut s = [0i32; L];
            let mut d = [0i32; L];
            for l in 0..L {
                s[l] = x[l].wrapping_add(y[l]);
                d[l] = x[l].wrapping_sub(y[l]);
            }
            t[a] = s;
            t[b] = d;
        } else {
            // B is a rotation followed by an optional exchange; its transpose
            // is the exchange followed by the rotation by the opposite angle.
            let (c, s) = if WIDE {
                (op.cos_hi, op.sin_hi)
            } else {
                (op.cos as i32, op.sin as i32)
            };
            let (x, y) = if op.flip { (t[b], t[a]) } else { (t[a], t[b]) };
            let mut u = [0i32; L];
            let mut v = [0i32; L];
            for l in 0..L {
                u[l] = mul2::<WIDE>(x[l], c, y[l], s);
                v[l] = mul2::<WIDE>(y[l], c, x[l], -s);
            }
            t[a] = u;
            t[b] = v;
        }
    }
}

/// Forward ADST4: the transpose of spec 7.13.2.6 (libaom's `fadst4`).
#[inline(always)]
fn fadst4<const L: usize, const N: usize, const WIDE: bool>(t: &mut [[i32; L]; N]) {
    let (i0, i1, i2, i3) = (t[0], t[1], t[2], t[3]);
    let mut o = [[0i32; L]; 4];
    let [s1, s2, s3, s4] = if WIDE {
        SINPI_HI
    } else {
        [SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9]
    };
    let m = |x: i32, c: i32| -> i64 {
        if WIDE {
            x as i64 * c as i64
        } else {
            x.wrapping_mul(c) as i64
        }
    };
    let r = |x: i64| -> i32 {
        if WIDE {
            ((x + (1 << 15)) >> 16) as i32
        } else {
            (x as i32).wrapping_add(2048) >> 12
        }
    };
    for l in 0..L {
        let (x0, x1, x2, x3) = (i0[l], i1[l], i2[l], i3[l]);
        let s7 = x0.wrapping_add(x1).wrapping_sub(x3);
        let a = m(x0, s1) + m(x1, s2) + m(x3, s4);
        let c = m(x0, s4) - m(x1, s1) + m(x3, s2);
        let d = m(x2, s3);
        o[0][l] = r(a + d);
        o[1][l] = r(m(s7, s3));
        o[2][l] = r(c - d);
        o[3][l] = r(c - a + d);
    }
    t[..4].copy_from_slice(&o);
}

/// Forward identity of length `N` (same scale as the inverse).
#[inline(always)]
fn fidt<const N: usize, const WIDE: bool>(x: i32) -> i32 {
    match N {
        4 => mul1::<WIDE>(x, if WIDE { 92682 } else { 5793 }),
        8 => x.wrapping_mul(2),
        16 => mul1::<WIDE>(x, if WIDE { 185364 } else { 11586 }),
        _ => x.wrapping_mul(4),
    }
}

/// The 1D forward transform of kind `K` on `t` (samples in natural order);
/// leaves coefficients `0..min(N, 32)` in natural order in `t`.
#[inline(always)]
fn tx1d<const L: usize, const N: usize, const K: u8, const WIDE: bool>(t: &mut [[i32; L]; N]) {
    match K {
        K_DCT => {
            let (ops, perm): (&[Op], &[u8]) = match N {
                4 => (&FDCT4, &DCT4_PERM),
                8 => (&FDCT8, &DCT8_PERM),
                16 => (&FDCT16, &DCT16_PERM),
                32 => (&FDCT32, &DCT32_PERM),
                _ => (&FDCT64, &DCT64_PERM),
            };
            run_fwd::<L, N, WIDE>(t, ops);
            let c = *t;
            for (k, &p) in perm.iter().enumerate().take(N.min(32)) {
                t[k & (N - 1)] = c[p as usize & (N - 1)];
            }
        }
        K_ADST if N == 4 => fadst4::<L, N, WIDE>(t),
        K_ADST => {
            let (ops, pin, pout): (&[Op], &[u8], &[u8]) = if N == 8 {
                (&FADST8, &ADST8_IN, &ADST8_OUT)
            } else {
                (&FADST16, &ADST16_IN, &ADST16_OUT)
            };
            let c = *t;
            for (i, &p) in pout.iter().enumerate() {
                let src = c[i & (N - 1)];
                let dst = &mut t[p as usize & (N - 1)];
                if i & 1 == 1 {
                    for l in 0..L {
                        dst[l] = src[l].wrapping_neg();
                    }
                } else {
                    *dst = src;
                }
            }
            run_fwd::<L, N, WIDE>(t, ops);
            let c = *t;
            for (k, &p) in pin.iter().enumerate() {
                t[k & (N - 1)] = c[p as usize & (N - 1)];
            }
        }
        _ => {
            for row in t.iter_mut() {
                for v in row.iter_mut() {
                    *v = fidt::<N, WIDE>(*v);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Passes.

/// Per-block constants shared by both passes.
struct Params {
    w: usize,
    /// Rows of coefficients kept: `min(h, 32)`.
    th: usize,
    flip_ud: bool,
    flip_lr: bool,
    /// Residual clamp: `(1 << bd) - 1`.
    imax: i32,
    /// Down shift after the column pass.
    d1: u32,
    /// Final multiplier (4096, or 2896 for 2:1 rectangles) and shift.
    fmul: i32,
    fshift: u32,
    /// Coefficient clamp (spec's Dequant range).
    omin: i32,
    omax: i32,
}

/// Up shift of the input.
const U0: u32 = 2;

/// Column transforms of length `N`, `L` columns at a time. Writes the first
/// `th` outputs of every column into `tmp` column-major (`tmp[c * th + k]`);
/// the row pass reads the columns flipped for `flip_lr`, so the column pass
/// of a block serves the types that differ only in their rows.
#[inline(never)]
fn col_pass<const L: usize, const N: usize, const K: u8, const WIDE: bool>(
    p: &Params,
    src: &[i32],
    stride: usize,
    tmp: &mut [i32],
) {
    let th = p.th;
    let w = p.w;
    let rnd = (1i32 << p.d1) >> 1;
    let mut c0 = 0;
    while c0 < w {
        let mut t = [[0i32; L]; N];
        let mut any = 0;
        for (r, tr) in t.iter_mut().enumerate() {
            let rr = if p.flip_ud { N - 1 - r } else { r };
            let row = &src[rr * stride + c0..][..L];
            for l in 0..L {
                let v = row[l].clamp(-p.imax, p.imax);
                any |= v;
                tr[l] = v << U0;
            }
        }
        if any == 0 {
            tmp[c0 * th..][..L * th].fill(0);
        } else {
            tx1d::<L, N, K, WIDE>(&mut t);
            for l in 0..L {
                let out = &mut tmp[(c0 + l) * th..][..th];
                for (k, o) in out.iter_mut().enumerate() {
                    *o = t[k & (N - 1)][l].wrapping_add(rnd) >> p.d1;
                }
            }
        }
        c0 += L;
    }
}

/// Row transforms of length `N`, `L` rows at a time, from the column-major
/// `tmp`; writes the coefficients row-major with stride `min(N, 32)`.
#[inline(never)]
fn row_pass<const L: usize, const N: usize, const K: u8, const WIDE: bool>(
    p: &Params,
    tmp: &[i32],
    coefs: &mut [i32],
) {
    let th = p.th;
    let tw = N.min(32);
    let rnd = (1i64 << p.fshift) >> 1;
    let fm = p.fmul;
    let mut r0 = 0;
    while r0 < th {
        let mut t = [[0i32; L]; N];
        let mut any = 0;
        for (j, tj) in t.iter_mut().enumerate() {
            let jj = if p.flip_lr { N - 1 - j } else { j };
            let s = &tmp[jj * th + r0..][..L];
            for l in 0..L {
                any |= s[l];
                tj[l] = s[l];
            }
        }
        if any == 0 {
            coefs[r0 * tw..][..L * tw].fill(0);
        } else {
            tx1d::<L, N, K, WIDE>(&mut t);
            for l in 0..L {
                let out = &mut coefs[(r0 + l) * tw..][..tw];
                for (k, o) in out.iter_mut().enumerate() {
                    let v = t[k & (N - 1)][l];
                    let v = if WIDE {
                        ((v as i64 * fm as i64 + rnd) >> p.fshift) as i32
                    } else {
                        v.wrapping_mul(fm).wrapping_add(rnd as i32) >> p.fshift
                    };
                    *o = v.clamp(p.omin, p.omax);
                }
            }
        }
        r0 += L;
    }
}

fn col_dispatch<const WIDE: bool>(
    log2h: u32,
    kind: u8,
    p: &Params,
    src: &[i32],
    stride: usize,
    tmp: &mut [i32],
) {
    macro_rules! go {
        ($l:literal, $n:literal) => {
            match kind {
                K_ADST => col_pass::<$l, $n, K_ADST, WIDE>(p, src, stride, tmp),
                K_IDT => col_pass::<$l, $n, K_IDT, WIDE>(p, src, stride, tmp),
                _ => col_pass::<$l, $n, K_DCT, WIDE>(p, src, stride, tmp),
            }
        };
        ($l:literal, $n:literal, no_adst) => {
            match kind {
                K_IDT => col_pass::<$l, $n, K_IDT, WIDE>(p, src, stride, tmp),
                _ => col_pass::<$l, $n, K_DCT, WIDE>(p, src, stride, tmp),
            }
        };
        ($l:literal, $n:literal, dct) => {
            col_pass::<$l, $n, K_DCT, WIDE>(p, src, stride, tmp)
        };
    }
    match (log2h, p.w == 4) {
        (2, true) => go!(4, 4),
        (2, false) => go!(8, 4),
        (3, true) => go!(4, 8),
        (3, false) => go!(8, 8),
        (4, true) => go!(4, 16),
        (4, false) => go!(8, 16),
        (5, true) => go!(4, 32, no_adst),
        (5, false) => go!(8, 32, no_adst),
        (_, true) => go!(4, 64, dct),
        (_, false) => go!(8, 64, dct),
    }
}

fn row_dispatch<const WIDE: bool>(
    log2w: u32,
    kind: u8,
    p: &Params,
    tmp: &[i32],
    coefs: &mut [i32],
) {
    macro_rules! go {
        ($l:literal, $n:literal) => {
            match kind {
                K_ADST => row_pass::<$l, $n, K_ADST, WIDE>(p, tmp, coefs),
                K_IDT => row_pass::<$l, $n, K_IDT, WIDE>(p, tmp, coefs),
                _ => row_pass::<$l, $n, K_DCT, WIDE>(p, tmp, coefs),
            }
        };
        ($l:literal, $n:literal, no_adst) => {
            match kind {
                K_IDT => row_pass::<$l, $n, K_IDT, WIDE>(p, tmp, coefs),
                _ => row_pass::<$l, $n, K_DCT, WIDE>(p, tmp, coefs),
            }
        };
        ($l:literal, $n:literal, dct) => {
            row_pass::<$l, $n, K_DCT, WIDE>(p, tmp, coefs)
        };
    }
    match (log2w, p.th == 4) {
        (2, true) => go!(4, 4),
        (2, false) => go!(8, 4),
        (3, true) => go!(4, 8),
        (3, false) => go!(8, 8),
        (4, true) => go!(4, 16),
        (4, false) => go!(8, 16),
        (5, true) => go!(4, 32, no_adst),
        (5, false) => go!(8, 32, no_adst),
        (_, true) => go!(4, 64, dct),
        (_, false) => go!(8, 64, dct),
    }
}

/// Both passes with an intermediate of `TMP` (>= w * min(h, 32)) entries.
/// `WIDE` selects 64-bit products; the shifts depend on `bd` only.
fn run<const TMP: usize, const WIDE: bool>(
    src: &[i32],
    stride: usize,
    coefs: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    bd: u32,
    split: bool,
) {
    let p = params(tx_size, tx_type, bd, split);
    let log2w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
    let mut tmp = [0i32; TMP];
    col_dispatch::<WIDE>(log2h, col_kind(tx_type), &p, src, stride, &mut tmp);
    row_dispatch::<WIDE>(log2w, row_kind(tx_type), &p, &tmp, coefs);
}

/// The pass constants of a block.
fn params(tx_size: usize, tx_type: usize, bd: u32, split: bool) -> Params {
    let log2w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
    let rect = log2w.abs_diff(log2h) == 1;
    let total = log2w + log2h - rect as u32 - TRANSFORM_ROW_SHIFT[tx_size] as u32 - 4;
    // In the 32-bit path (8 bits, at most 32 points) a coefficient is at
    // most about `2^(10 + log2w + log2h - d1) / 2` inside the row pass, so the
    // column pass sheds `log2w + log2h - 8` bits to keep every product of a
    // 12-bit constant below 2^30. The 64-bit path keeps full precision to
    // the end.
    let d1 = if split {
        (log2w + log2h).saturating_sub(8).min(total)
    } else {
        0
    };
    let (flip_ud, flip_lr) = tx_type_flips(tx_type);
    let olim = 1i32 << (7 + bd);
    Params {
        w: 1 << log2w,
        th: (1usize << log2h).min(32),
        flip_ud,
        flip_lr,
        imax: (1i32 << bd) - 1,
        d1,
        fmul: if rect { 2896 } else { 4096 },
        fshift: 12 + total - d1,
        omin: -olim,
        omax: olim - 1,
    }
}

fn run_sized<const WIDE: bool>(
    src: &[i32],
    stride: usize,
    coefs: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    bd: u32,
    split: bool,
) {
    let w = 1usize << TX_WIDTH_LOG2[tx_size];
    let th = (1usize << TX_HEIGHT_LOG2[tx_size]).min(32);
    let need = w * th;
    if need <= 16 {
        run::<16, WIDE>(src, stride, coefs, tx_size, tx_type, bd, split);
    } else if need <= 64 {
        run::<64, WIDE>(src, stride, coefs, tx_size, tx_type, bd, split);
    } else if need <= 256 {
        run::<256, WIDE>(src, stride, coefs, tx_size, tx_type, bd, split);
    } else if need <= 1024 {
        run::<1024, WIDE>(src, stride, coefs, tx_size, tx_type, bd, split);
    } else {
        run::<2048, WIDE>(src, stride, coefs, tx_size, tx_type, bd, split);
    }
}

/// Forward 2D transform of a `w x h` residual block.
///
/// `src` is row-major at `src_stride` and holds values in the residual range
/// for `bd` (`-(2^bd - 1)..=2^bd - 1`; values outside are clamped).
/// `tx_size` and `tx_type` use spec numbering and must be a combination the
/// spec allows (64-point transforms are DCT only, 32-point ones DCT or
/// identity; other types fall back to the DCT there).
///
/// Writes `coefs[..min(w, 32) * min(h, 32)]`, row-major with stride
/// `min(w, 32)`, in exactly the layout and scale `inv_txfm_add` takes (the
/// spec's Dequant domain, clamped to its range); for 64-point sizes only the
/// low-frequency 32x32 part is kept. With a zero prediction `inv_txfm_add` of
/// the result gives back the residual up to rounding.
pub fn fwd_txfm(
    src: &[i32],
    src_stride: usize,
    coefs: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    bd: u32,
) {
    // 8-bit blocks multiply in 32 bits, except 64-point ones: there the
    // column pass would have to shed too many bits to keep the row pass in
    // range.
    if bd <= 8 && TX_WIDTH_LOG2[tx_size].max(TX_HEIGHT_LOG2[tx_size]) < 6 {
        run_sized::<false>(src, src_stride, coefs, tx_size, tx_type, bd, true);
    } else {
        run_sized::<true>(src, src_stride, coefs, tx_size, tx_type, bd, false);
    }
}

/// The column passes of one residual block, kept so that the transform
/// types sharing a vertical transform share them ([`fwd_txfm_cols`]).
#[derive(Clone, Debug, Default)]
pub struct FwdCols {
    tmp: [Vec<i32>; 4],
    done: [bool; 4],
}

impl FwdCols {
    /// Forgets the passes (for a new residual block or transform size).
    pub fn clear(&mut self) {
        self.done = [false; 4];
    }
}

/// [`fwd_txfm`] of the residual block `cols` was last used with (since its
/// [`FwdCols::clear`]): equal output, but each vertical transform (DCT,
/// ADST, flipped ADST, identity) runs over the block only once.
pub fn fwd_txfm_cols(
    cols: &mut FwdCols,
    src: &[i32],
    src_stride: usize,
    coefs: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    bd: u32,
) {
    let log2w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_size] as u32;
    let wide = !(bd <= 8 && log2w.max(log2h) < 6);
    let p = params(tx_size, tx_type, bd, !wide);
    let kind = col_kind(tx_type);
    let k = match kind {
        K_DCT => 0,
        K_ADST if p.flip_ud => 2,
        K_ADST => 1,
        _ => 3,
    };
    let tmp = &mut cols.tmp[k];
    if !cols.done[k] {
        if tmp.len() < 2048 {
            tmp.resize(2048, 0);
        }
        if wide {
            col_dispatch::<true>(log2h, kind, &p, src, src_stride, tmp);
        } else {
            col_dispatch::<false>(log2h, kind, &p, src, src_stride, tmp);
        }
        cols.done[k] = true;
    }
    if wide {
        row_dispatch::<true>(log2w, row_kind(tx_type), &p, tmp, coefs);
    } else {
        row_dispatch::<false>(log2w, row_kind(tx_type), &p, tmp, coefs);
    }
}

// ---------------------------------------------------------------------------
// Estimate path: an approximate 8-bit DCT in 16-bit lanes.

/// Largest magnitude anywhere in the forward DCT network `ops` of length
/// `len` (12-bit constants, rounding included) when every input is at most
/// `b` in magnitude.
const fn dct_bound(ops: &[Op], len: usize, b: i64) -> i64 {
    let mut v = [0i64; 64];
    let mut i = 0;
    while i < len {
        v[i] = b;
        i += 1;
    }
    let mut m = b;
    let mut k = 0;
    while k < ops.len() {
        let op = ops[k];
        let (a, bb) = (op.a as usize & (len - 1), op.b as usize & (len - 1));
        let (x, y) = if op.flip {
            (v[bb], v[a])
        } else {
            (v[a], v[bb])
        };
        let (u, w) = if op.had {
            (x + y, x + y)
        } else {
            let c = (op.cos as i64).abs();
            let s = (op.sin as i64).abs();
            ((x * c + y * s) / 4096 + 2, (y * c + x * s) / 4096 + 2)
        };
        v[a] = u;
        v[bb] = w;
        if u > m {
            m = u;
        }
        if w > m {
            m = w;
        }
        k += 1;
    }
    m
}

/// The forward DCT network and output permutation of length `1 << n`.
const fn dct_net(n: usize) -> (&'static [Op], &'static [u8]) {
    match n {
        2 => (&FDCT4, &DCT4_PERM),
        3 => (&FDCT8, &DCT8_PERM),
        4 => (&FDCT16, &DCT16_PERM),
        5 => (&FDCT32, &DCT32_PERM),
        _ => (&FDCT64, &DCT64_PERM),
    }
}

/// Largest 16-bit value the estimate networks may reach.
const EST_MAX: i32 = i16::MAX as i32;

/// Gain bound of the DCT network of length `1 << n` with 16 fractional bits
/// and the rounding it may add: no value inside it exceeds
/// `(b * gain >> 16) + slack` for inputs of at most `b`.
const fn dct_gain(n: usize) -> (i32, i32) {
    let (ops, _) = dct_net(n);
    let len = 1 << n;
    let gain = (dct_bound(ops, len, 1 << 20) + 15) >> 4;
    (gain as i32, dct_bound(ops, len, 0) as i32 + 1)
}

static EST_GAIN: [(i32, i32); 7] = [
    (0, 0),
    (0, 0),
    dct_gain(2),
    dct_gain(3),
    dct_gain(4),
    dct_gain(5),
    dct_gain(6),
];

/// Whether inputs of at most `b` keep the network of length `1 << n` in range.
#[inline(always)]
fn est_fits(n: usize, b: i32) -> bool {
    let (g, s) = EST_GAIN[n];
    ((i64::from(b) * i64::from(g)) >> 16) + i64::from(s) <= i64::from(EST_MAX)
}

/// A 12-bit constant `k` as a whole part and a 16-bit fraction:
/// `x * k / 4096 = x * a + x * m / 65536`.
#[inline(always)]
fn split12(k: i32) -> (i16, i16) {
    let a = (k + 2048) >> 12;
    (a as i16, ((k - (a << 12)) << 4) as i16)
}

/// The high half of a 16-bit product.
#[inline(always)]
fn mulhi(x: i16, m: i16) -> i16 {
    ((i32::from(x) * i32::from(m)) >> 16) as i16
}

/// The forward DCT network on 16-bit lanes (12-bit constants as a whole
/// part and a 16-bit fraction). The callers' shifts keep every value in
/// range.
#[inline(always)]
fn run_est<const L: usize, const N: usize>(t: &mut [[i16; L]; N], ops: &[Op]) {
    let ops = core::hint::black_box(ops);
    for op in ops {
        let a = op.a as usize & (N - 1);
        let b = op.b as usize & (N - 1);
        if op.had {
            let (a, b) = if op.flip { (b, a) } else { (a, b) };
            let x = t[a];
            let y = t[b];
            let mut s = [0i16; L];
            let mut d = [0i16; L];
            for l in 0..L {
                s[l] = x[l].wrapping_add(y[l]);
                d[l] = x[l].wrapping_sub(y[l]);
            }
            t[a] = s;
            t[b] = d;
        } else {
            let (c, s) = (i32::from(op.cos), i32::from(op.sin));
            let ((ca, cm), (sa, sm), (na, nm)) = (split12(c), split12(s), split12(-s));
            let (x, y) = if op.flip { (t[b], t[a]) } else { (t[a], t[b]) };
            let mut u = [0i16; L];
            let mut v = [0i16; L];
            for l in 0..L {
                let (xl, yl) = (x[l], y[l]);
                // Each high half is floored: the 1 makes up the average.
                u[l] = xl
                    .wrapping_mul(ca)
                    .wrapping_add(mulhi(xl, cm))
                    .wrapping_add(yl.wrapping_mul(sa))
                    .wrapping_add(mulhi(yl, sm))
                    .wrapping_add(1);
                v[l] = yl
                    .wrapping_mul(ca)
                    .wrapping_add(mulhi(yl, cm))
                    .wrapping_add(xl.wrapping_mul(na))
                    .wrapping_add(mulhi(xl, nm))
                    .wrapping_add(1);
            }
            t[a] = u;
            t[b] = v;
        }
    }
}

/// Column DCTs of length `N`, `L` columns at a time, of the input shifted up
/// by `up`, into the column-major `tmp` (`tmp[c * th + k]`). Returns the
/// largest magnitude written.
#[inline(never)]
fn col_est<const L: usize, const N: usize>(
    src: &[i32],
    stride: usize,
    w: usize,
    up: u32,
    tmp: &mut [i16],
) -> i32 {
    let th = N.min(32);
    let (ops, perm) = dct_net(N.trailing_zeros() as usize);
    let mut m = [0i16; L];
    let mut c0 = 0;
    while c0 < w {
        let mut t = [[0i16; L]; N];
        let mut any = 0;
        for (r, tr) in t.iter_mut().enumerate() {
            let row = &src[r * stride + c0..][..L];
            for l in 0..L {
                let v = row[l].clamp(-255, 255);
                any |= v;
                tr[l] = (v << up) as i16;
            }
        }
        if any == 0 {
            tmp[c0 * th..][..L * th].fill(0);
        } else {
            run_est::<L, N>(&mut t, ops);
            for k in 0..th {
                let r = &t[perm[k] as usize & (N - 1)];
                for l in 0..L {
                    m[l] = m[l].max(r[l].wrapping_abs());
                }
            }
            for l in 0..L {
                let out = &mut tmp[(c0 + l) * th..][..th];
                for (k, o) in out.iter_mut().enumerate() {
                    *o = t[perm[k] as usize & (N - 1)][l];
                }
            }
        }
        c0 += L;
    }
    m.iter().fold(0, |a, &v| a.max(i32::from(v)))
}

/// Row DCTs of length `N`, `L` rows at a time, from the column-major `tmp`
/// shifted down by `mid`; writes the coefficients row-major with stride
/// `min(N, 32)`, times `fmul / 4096` and shifted down by `shift` (up by
/// `-shift`).
#[inline(never)]
fn row_est<const L: usize, const N: usize>(
    tmp: &[i16],
    th: usize,
    mid: u32,
    fmul: i32,
    shift: i32,
    coefs: &mut [i32],
) {
    let tw = N.min(32);
    let (ops, perm) = dct_net(N.trailing_zeros() as usize);
    let (mul, sh) = if fmul == 4096 {
        (1, shift)
    } else {
        (fmul, shift + 12)
    };
    let rnd = if sh > 0 { (1i32 << sh) >> 1 } else { 0 };
    let mut r0 = 0;
    while r0 < th {
        let mut t = [[0i16; L]; N];
        let mut any = 0;
        for (j, tj) in t.iter_mut().enumerate() {
            let s = &tmp[j * th + r0..][..L];
            for l in 0..L {
                let v = if mid == 0 {
                    s[l]
                } else {
                    (s[l] >> mid) + ((s[l] >> (mid - 1)) & 1)
                };
                any |= v;
                tj[l] = v;
            }
        }
        if any == 0 {
            coefs[r0 * tw..][..L * tw].fill(0);
        } else {
            run_est::<L, N>(&mut t, ops);
            for l in 0..L {
                let out = &mut coefs[(r0 + l) * tw..][..tw];
                for (k, o) in out.iter_mut().enumerate() {
                    let v = i32::from(t[perm[k] as usize & (N - 1)][l]) * mul;
                    *o = if sh >= 0 { (v + rnd) >> sh } else { v << -sh };
                }
            }
        }
        r0 += L;
    }
}

/// An approximation of `fwd_txfm(src, src_stride, coefs, tx_size, DCT_DCT,
/// bd)` for rate-distortion estimates, in the same layout and scale: 8-bit
/// content is transformed in 16-bit lanes, scaled to the block's range
/// (deeper content takes the exact path). `tmp` holds at least 2048 values.
pub fn fwd_dct_est(
    src: &[i32],
    src_stride: usize,
    coefs: &mut [i32],
    tmp: &mut [i16],
    tx_size: usize,
    bd: u32,
) {
    if bd > 8 {
        fwd_txfm(src, src_stride, coefs, tx_size, DCT_DCT as usize, bd);
        return;
    }
    let nw = TX_WIDTH_LOG2[tx_size] as usize;
    let nh = TX_HEIGHT_LOG2[tx_size] as usize;
    let w = 1usize << nw;
    let h = 1usize << nh;
    let th = h.min(32);
    let rect = nw.abs_diff(nh) == 1;
    let total = (nw + nh) as i32 - rect as i32 - TRANSFORM_ROW_SHIFT[tx_size] as i32 - 4;
    // The largest input magnitude sets the up shift.
    let mut amax = 0;
    for r in 0..h {
        for &v in &src[r * src_stride..][..w] {
            amax = amax.max(v.unsigned_abs());
        }
    }
    let amax = amax.min(255) as i32;
    if amax == 0 {
        coefs[..w.min(32) * th].fill(0);
        return;
    }
    let mut up = 0;
    while up < 8 && est_fits(nh, amax << (up + 1)) {
        up += 1;
    }
    let tmp = &mut tmp[..w * th];
    macro_rules! col {
        ($n:literal) => {
            if w == 4 {
                col_est::<4, $n>(src, src_stride, w, up, tmp)
            } else {
                col_est::<8, $n>(src, src_stride, w, up, tmp)
            }
        };
    }
    let cmax = match nh {
        2 => col!(4),
        3 => col!(8),
        4 => col!(16),
        5 => col!(32),
        _ => col!(64),
    };
    let mut mid = 0;
    while !est_fits(nw, (cmax >> mid) + 1) {
        mid += 1;
    }
    let shift = total + up as i32 - mid as i32 - U0 as i32;
    let fmul = if rect { 2896 } else { 4096 };
    macro_rules! row {
        ($n:literal) => {
            if th == 4 {
                row_est::<4, $n>(tmp, th, mid, fmul, shift, coefs)
            } else {
                row_est::<8, $n>(tmp, th, mid, fmul, shift, coefs)
            }
        };
    }
    match nw {
        2 => row!(4),
        3 => row!(8),
        4 => row!(16),
        5 => row!(32),
        _ => row!(64),
    }
}

/// Lossless forward 4x4 Walsh-Hadamard transform (libaom's `av1_fwht4x4`):
/// the exact inverse of the decoder's lossless path (spec 7.13.2.10), in the
/// Dequant domain (the lossless quantizer step is 4). `coefs[..16]` is
/// row-major with stride 4.
pub fn fwd_wht4x4(src: &[i32], src_stride: usize, coefs: &mut [i32]) {
    fn wht(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
        let a = a.wrapping_add(b);
        let d = d.wrapping_sub(c);
        let e = a.wrapping_sub(d) >> 1;
        let b = e.wrapping_sub(b);
        let c = e.wrapping_sub(c);
        let a = a.wrapping_sub(c);
        let d = d.wrapping_add(b);
        [a, c, d, b]
    }
    let mut tmp = [[0i32; 4]; 4];
    for j in 0..4 {
        let o = wht(
            src[j],
            src[src_stride + j],
            src[2 * src_stride + j],
            src[3 * src_stride + j],
        );
        for i in 0..4 {
            tmp[i][j] = o[i];
        }
    }
    for i in 0..4 {
        let o = wht(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3]);
        for j in 0..4 {
            coefs[i * 4 + j] = o[j].wrapping_mul(4);
        }
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout)]
mod tests {
    use super::{fwd_dct_est, fwd_txfm, fwd_wht4x4, run_sized};
    use crate::dsp::Pixel;
    use crate::dsp::itx::inv_txfm_add;
    use crate::spec::tables::{DCT_DCT, IDTX, TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    fn dims(tx_size: usize) -> (usize, usize) {
        (1 << TX_WIDTH_LOG2[tx_size], 1 << TX_HEIGHT_LOG2[tx_size])
    }

    fn allowed_types(tx_size: usize) -> Vec<usize> {
        let (w, h) = dims(tx_size);
        let m = w.max(h);
        if m == 64 {
            vec![DCT_DCT as usize]
        } else if m == 32 {
            vec![DCT_DCT as usize, IDTX as usize]
        } else {
            (0..16).collect()
        }
    }

    /// Reconstructs `coefs` over a constant prediction `pred`.
    fn recon<P: Pixel>(
        coefs: &[i32],
        tx_size: usize,
        tx_type: usize,
        bd: u32,
        pred: i32,
    ) -> Vec<i32> {
        let (w, h) = dims(tx_size);
        let (tw, th) = (w.min(32), h.min(32));
        let mut nr = 0;
        let mut nc = 0;
        for i in 0..th {
            for j in 0..tw {
                if coefs[i * tw + j] != 0 {
                    nr = nr.max(i + 1);
                    nc = nc.max(j + 1);
                }
            }
        }
        let mut c = coefs[..tw * th].to_vec();
        let mut dst = vec![P::from_i32(pred); w * h];
        inv_txfm_add(
            &mut dst,
            w,
            &mut c,
            tx_size,
            tx_type,
            nr.max(1),
            nc.max(1),
            false,
            bd,
        );
        dst.iter().map(|p| p.to_i32()).collect()
    }

    /// A residual: 0 full-range noise, 1 small noise, 2 smooth, 3 smooth with
    /// noise, 4 sparse spikes.
    fn residual(rng: &mut Rng, w: usize, h: usize, bd: u32, mode: u32) -> Vec<i32> {
        let lim = (1i32 << (bd - 1)) - 1;
        let mut r = vec![0i32; w * h];
        match mode {
            0 => r.iter_mut().for_each(|v| *v = rng.range(-lim, lim)),
            1 => {
                let s = 1 << (bd - 6);
                r.iter_mut().for_each(|v| *v = rng.range(-s, s));
            }
            4 => {
                for _ in 0..3 {
                    let i = rng.next() as usize % (w * h);
                    r[i] = rng.range(-lim, lim);
                }
            }
            _ => {
                // A few low-frequency cosines.
                let mut f = vec![0f64; w * h];
                for _ in 0..4 {
                    let kx = (rng.next() % 6) as f64;
                    let ky = (rng.next() % 6) as f64;
                    let a = rng.range(-1000, 1000) as f64 / 1000.0;
                    for y in 0..h {
                        for x in 0..w {
                            let cx = (std::f64::consts::PI * (2 * x + 1) as f64 * kx
                                / (2 * w) as f64)
                                .cos();
                            let cy = (std::f64::consts::PI * (2 * y + 1) as f64 * ky
                                / (2 * h) as f64)
                                .cos();
                            f[y * w + x] += a * cx * cy;
                        }
                    }
                }
                let m = f.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
                let amp = lim as f64 * 0.9;
                let n = if mode == 3 { 1 << (bd - 7) } else { 0 };
                for (o, v) in r.iter_mut().zip(f) {
                    *o = ((v / m * amp).round() as i32 + rng.range(-n, n)).clamp(-lim, lim);
                }
            }
        }
        r
    }

    fn round_trip(rng: &mut Rng, tx_size: usize, tx_type: usize, bd: u32, mode: u32) -> i32 {
        let (w, h) = dims(tx_size);
        let res = residual(rng, w, h, bd, mode);
        let mut coefs = vec![0i32; 32 * 32];
        fwd_txfm(&res, w, &mut coefs, tx_size, tx_type, bd);
        let pred = 1 << (bd - 1);
        let got = if bd == 8 {
            recon::<u8>(&coefs, tx_size, tx_type, bd, pred)
        } else {
            recon::<u16>(&coefs, tx_size, tx_type, bd, pred)
        };
        got.iter()
            .zip(&res)
            .map(|(&g, &r)| (g - pred - r).abs())
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn round_trip_all() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for bd in [8u32, 10, 12] {
            let mut worst = [0i32; 3]; // <= 16, 32, 64 point
            for tx_size in 0..19 {
                let (w, h) = dims(tx_size);
                let m = w.max(h);
                for tx_type in allowed_types(tx_size) {
                    let modes: &[u32] = if m == 64 { &[2] } else { &[0, 1, 2, 3, 4] };
                    for &mode in modes {
                        for _ in 0..6 {
                            let e = round_trip(&mut rng, tx_size, tx_type, bd, mode);
                            let class = if m == 64 {
                                2
                            } else if m == 32 {
                                1
                            } else {
                                0
                            };
                            worst[class] = worst[class].max(e);
                            let bound = if m == 64 { 3 } else { 2 };
                            assert!(
                                e <= bound,
                                "bd {bd} size {tx_size} type {tx_type} mode {mode}: max err {e}"
                            );
                        }
                    }
                }
            }
            println!(
                "bd {bd}: max abs err <=16pt {} 32pt {} 64pt(smooth) {}",
                worst[0], worst[1], worst[2]
            );
        }
    }

    /// 64-point sizes with noisy content: the error is the dropped
    /// high-frequency part of the noise, so the mean squared error stays
    /// near the noise variance.
    #[test]
    fn round_trip_64_noisy() {
        let mut rng = Rng(99);
        for bd in [8u32, 10, 12] {
            for tx_size in [4usize, 11, 12, 17, 18] {
                let (w, h) = dims(tx_size);
                let res = residual(&mut rng, w, h, bd, 3);
                let mut coefs = vec![0i32; 32 * 32];
                fwd_txfm(&res, w, &mut coefs, tx_size, 0, bd);
                let pred = 1 << (bd - 1);
                let got = if bd == 8 {
                    recon::<u8>(&coefs, tx_size, 0, bd, pred)
                } else {
                    recon::<u16>(&coefs, tx_size, 0, bd, pred)
                };
                let err: f64 = got
                    .iter()
                    .zip(&res)
                    .map(|(&g, &r)| ((g - pred - r) as f64).powi(2))
                    .sum();
                let mse = err / (w * h) as f64;
                // Uniform noise in -n..=n has variance n (n + 1) / 3.
                let n = (1i32 << (bd - 7)) as f64;
                let var = n * (n + 1.0) / 3.0;
                println!("bd {bd} size {tx_size}: mse {mse:.2} noise variance {var:.2}");
                assert!(mse <= var + 1.0, "bd {bd} size {tx_size}: mse {mse}");
            }
        }
    }

    #[test]
    fn dc_block() {
        for bd in [8u32, 10, 12] {
            for tx_size in 0..19 {
                let (w, h) = dims(tx_size);
                let (tw, th) = (w.min(32), h.min(32));
                for v in [1i32, -3, 37, (1 << bd) - 1, -((1 << bd) - 1)] {
                    let src = vec![v; w * h];
                    let mut coefs = vec![7i32; tw * th];
                    fwd_txfm(&src, w, &mut coefs, tx_size, 0, bd);
                    assert!(coefs[1..].iter().all(|&c| c == 0), "size {tx_size}");
                    // The inverse maps a lone DC coefficient c to
                    // c * rect / 2 * 2^-(rowShift + 4) everywhere.
                    let lw = TX_WIDTH_LOG2[tx_size] as i32;
                    let lh = TX_HEIGHT_LOG2[tx_size] as i32;
                    let rs = TRANSFORM_ROW_SHIFT[tx_size] as i32;
                    let mut expect = v as f64 * 2f64.powi(rs + 5);
                    if (lw - lh).abs() == 1 {
                        expect *= std::f64::consts::SQRT_2;
                    }
                    assert!(
                        (coefs[0] as f64 - expect).abs() <= 2.0 + expect.abs() * 0.03,
                        "size {tx_size} v {v}: dc {} expect {expect}",
                        coefs[0]
                    );
                    let pred = if v < 0 { (1 << bd) - 1 } else { 0 };
                    let got = if bd == 8 {
                        recon::<u8>(&coefs, tx_size, 0, bd, pred)
                    } else {
                        recon::<u16>(&coefs, tx_size, 0, bd, pred)
                    };
                    // Exact, except that the decoder's 12-bit 1/sqrt(2)
                    // (applied two or three times) loses a level or two on
                    // full-range 12-bit values.
                    let tol = (v.abs() as f64 * 5e-4) as i32;
                    assert!(
                        got.iter().all(|&g| (g - pred - v).abs() <= tol),
                        "size {tx_size} v {v}"
                    );
                }
            }
        }
    }

    /// The estimate transform stays within rounding of the exact one, on
    /// extreme and on small residuals.
    #[test]
    fn estimate_close_to_exact() {
        let mut rng = Rng(777);
        for tx_size in 0..19 {
            let (w, h) = dims(tx_size);
            let (cw, ch) = (w.min(32), h.min(32));
            for it in 0..60 {
                let amp = [255, 255, 64, 8][it % 4];
                let src: Vec<i32> = (0..w * h)
                    .map(|i| match it % 4 {
                        0 => {
                            if rng.next() & 1 == 0 {
                                255
                            } else {
                                -255
                            }
                        }
                        1 => 255 - ((i % w) as i32 & 1) * 510 * ((it / 4) as i32 & 1),
                        _ => rng.range(-amp, amp),
                    })
                    .collect();
                let mut a = vec![0i32; 32 * 32];
                let mut b = vec![0i32; 32 * 32];
                fwd_txfm(&src, w, &mut a, tx_size, DCT_DCT as usize, 8);
                fwd_dct_est(&src, w, &mut b, &mut [0; 2048], tx_size, 8);
                let e: f64 = a[..cw * ch]
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum();
                let d: f64 = a[..cw * ch]
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
                    .sum();
                // Rounding noise only: an overflow would be far larger.
                let rel = d / e.max(1.0);
                let mx = a[..cw * ch]
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| (x - y).abs())
                    .max()
                    .unwrap_or(0);
                let big = a[..cw * ch].iter().map(|x| x.abs()).max().unwrap_or(0);
                assert!(
                    rel < 2e-3 && mx <= 4 + big / 64,
                    "size {tx_size} it {it}: relative error {rel}, largest {mx}"
                );
            }
        }
    }

    /// The 8-bit path multiplies in 32 bits; on extreme inputs it must agree
    /// with the 64-bit path up to precision (an overflow would be far off).
    #[test]
    fn narrow_matches_wide() {
        let mut rng = Rng(4242);
        for tx_size in 0..19 {
            let (w, h) = dims(tx_size);
            if w.max(h) == 64 {
                continue;
            }
            for tx_type in allowed_types(tx_size) {
                for it in 0..40 {
                    let src: Vec<i32> = (0..w * h)
                        .map(|i| {
                            let (x, y) = (i % w, i / w);
                            match it % 4 {
                                0 => {
                                    if rng.next() & 1 == 0 {
                                        255
                                    } else {
                                        -255
                                    }
                                }
                                1 => 255,
                                // Sign patterns of a DCT basis function.
                                _ => {
                                    let kx = (it * 7) % w;
                                    let ky = (it * 3) % h;
                                    let cx =
                                        (std::f64::consts::PI * (2 * x + 1) as f64 * kx as f64
                                            / (2 * w) as f64)
                                            .cos();
                                    let cy =
                                        (std::f64::consts::PI * (2 * y + 1) as f64 * ky as f64
                                            / (2 * h) as f64)
                                            .cos();
                                    if cx * cy >= 0.0 { 255 } else { -255 }
                                }
                            }
                        })
                        .collect();
                    let mut a = vec![0i32; 32 * 32];
                    let mut b = vec![0i32; 32 * 32];
                    run_sized::<false>(&src, w, &mut a, tx_size, tx_type, 8, true);
                    run_sized::<true>(&src, w, &mut b, tx_size, tx_type, 8, true);
                    // Precision differs slightly; an overflow would not.
                    for (&x, &y) in a.iter().zip(&b) {
                        assert!(
                            (x - y).abs() <= 8 + y.abs() / 256,
                            "size {tx_size} type {tx_type} it {it}: {x} vs {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wht_lossless_exact() {
        let mut rng = Rng(7);
        for bd in [8u32, 10, 12] {
            let pmax = (1i32 << bd) - 1;
            for tx_type in [0usize, 9] {
                for _ in 0..2000 {
                    let pred: Vec<i32> = (0..16).map(|_| rng.range(0, pmax)).collect();
                    let orig: Vec<i32> = (0..16).map(|_| rng.range(0, pmax)).collect();
                    let res: Vec<i32> = orig.iter().zip(&pred).map(|(o, p)| o - p).collect();
                    let mut coefs = [0i32; 16];
                    fwd_wht4x4(&res, 4, &mut coefs);
                    let got: Vec<i32> = if bd == 8 {
                        let mut d: Vec<u8> = pred.iter().map(|&p| p as u8).collect();
                        inv_txfm_add(&mut d, 4, &mut coefs, 0, tx_type, 4, 4, true, bd);
                        d.iter().map(|&p| p as i32).collect()
                    } else {
                        let mut d: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
                        inv_txfm_add(&mut d, 4, &mut coefs, 0, tx_type, 4, 4, true, bd);
                        d.iter().map(|&p| p as i32).collect()
                    };
                    assert_eq!(got, orig, "bd {bd}");
                }
            }
        }
    }
}
