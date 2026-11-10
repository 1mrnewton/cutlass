//! The inverse DCT and ADST as tables of steps, built at compile time
//! from the step lists of the AV1 specification (7.13.2.3, 7.13.2.7 and
//! 7.13.2.8), and where each input lands in the array `T` they work on.

/// One step on entries `a` and `b` of `T` (7.13.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// The Hadamard rotation `H(a, b, flip)`. Otherwise the butterfly
    /// rotation `B(a, b, angle, flip)`.
    pub had: bool,
    pub flip: bool,
    pub a: u8,
    pub b: u8,
    /// `cos128(angle)` and `sin128(angle)` of a butterfly rotation.
    pub cos: i16,
    pub sin: i16,
}

/// `Cos128_Lookup` of the specification: `cos(i * pi / 128)` in 12 bits.
pub const COS128: [u16; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

/// `cos128(angle)` (7.13.2.1); angles count modulo 256.
#[must_use]
pub const fn cos128(angle: i32) -> i32 {
    let a2 = (angle & 255) as usize;
    if a2 <= 64 {
        COS128[a2] as i32
    } else if a2 <= 128 {
        -(COS128[128 - a2] as i32)
    } else if a2 <= 192 {
        -(COS128[a2 - 128] as i32)
    } else {
        COS128[256 - a2] as i32
    }
}

/// `sin128(angle)` (7.13.2.1).
#[must_use]
pub const fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

/// `x` with its low `num_bits` bits in reverse order (7.13.2.2).
#[must_use]
pub const fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    let mut i = 0;
    while i < num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
        i += 1;
    }
    t
}

/// Position in `T` of input `k` of the DCT (7.13.2.2). `brev` is its own
/// inverse.
#[must_use]
pub const fn dct_perm<const N: usize>() -> [u8; N] {
    let n = N.trailing_zeros();
    let mut p = [0u8; N];
    let mut k = 0;
    while k < N {
        p[k] = brev(n, k) as u8;
        k += 1;
    }
    p
}

/// Position in `T` of input `k` of the ADST8 and ADST16 (the inverse of
/// 7.13.2.4).
#[must_use]
pub const fn adst_in_perm<const N: usize>() -> [u8; N] {
    let mut p = [0u8; N];
    let mut k = 0;
    while k < N {
        p[k] = if k & 1 == 0 { k + 1 } else { N - 1 - k } as u8;
        k += 1;
    }
    p
}

/// Entry of `T` that output `i` of the ADST8 and ADST16 is taken from
/// (7.13.2.5); odd outputs are negated.
#[must_use]
pub const fn adst_out_perm<const N: usize>() -> [u8; N] {
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

/// A transform written out as its sequence of steps: the first `len` of
/// `ops`. `len` counts every step pushed, so a table too short for a
/// transform shows as `len > M`.
#[derive(Clone, Copy)]
pub struct Prog<const M: usize> {
    pub ops: [Step; M],
    pub len: usize,
}

impl<const M: usize> Default for Prog<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> Prog<M> {
    /// No steps.
    #[must_use]
    pub const fn new() -> Self {
        let nop = Step {
            had: false,
            flip: false,
            a: 0,
            b: 0,
            cos: 0,
            sin: 0,
        };
        Prog {
            ops: [nop; M],
            len: 0,
        }
    }

    const fn push(mut self, had: bool, a: usize, b: usize, angle: usize, flip: bool) -> Self {
        if self.len < M {
            self.ops[self.len] = Step {
                had,
                flip,
                a: a as u8,
                b: b as u8,
                cos: cos128(angle as i32) as i16,
                sin: sin128(angle as i32) as i16,
            };
        }
        self.len += 1;
        self
    }

    /// `B(a, b, angle, flip)`; angles are given modulo 256.
    #[must_use]
    pub const fn bf(self, a: usize, b: usize, angle: usize, flip: bool) -> Self {
        self.push(false, a, b, angle & 255, flip)
    }

    /// `H(a, b, flip)`.
    #[must_use]
    pub const fn h(self, a: usize, b: usize, flip: bool) -> Self {
        self.push(true, a, b, 0, flip)
    }
}

/// The inverse DCT of length `1 << n` (7.13.2.3, steps 2 to 31; the
/// permutation of step 1 is the caller's, see [`dct_perm`]).
#[must_use]
pub const fn dct_prog<const M: usize>(n: u32) -> Prog<M> {
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

/// The inverse ADST8 (7.13.2.7, steps 2 to 6).
#[must_use]
pub const fn adst8_prog<const M: usize>() -> Prog<M> {
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

/// The inverse ADST16 (7.13.2.8, steps 2 to 8).
#[must_use]
pub const fn adst16_prog<const M: usize>() -> Prog<M> {
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

/// What a step comes to when some entries of `T` are known to be zero.
pub mod kind {
    /// Both entries are zero, and stay zero.
    pub const SKIP: u8 = 0;
    /// A butterfly rotation of two entries.
    pub const ROT: u8 = 1;
    /// A butterfly rotation where only entry `a` can be non-zero: two
    /// products of it.
    pub const ROT_A: u8 = 2;
    /// A butterfly rotation where only entry `b` can be non-zero.
    pub const ROT_B: u8 = 3;
    /// A Hadamard rotation of two entries.
    pub const HAD: u8 = 4;
    /// A Hadamard rotation where only the entry the sum goes to can be
    /// non-zero: the other becomes a copy of it.
    pub const HAD_FIRST: u8 = 5;
    /// A Hadamard rotation where only the entry the difference goes to
    /// can be non-zero: the sum is a copy of it, and it is negated.
    pub const HAD_SECOND: u8 = 6;
}

/// Whether a product by `c` and a rounding shift by 12 can be one doubling
/// multiply that keeps the high half: `8 * c` has to fit 16 bits.
const fn one_multiply(c: i16) -> bool {
    c >= -4095 && c <= 4095
}

/// The [`kind`] of every step of `prog` when only the inputs `0..nz` can
/// be non-zero and input `k` sits at `perm[k]` of `T`. An entry no step
/// writes stays zero.
#[must_use]
pub const fn plan<const M: usize>(prog: &Prog<M>, perm: &[u8; 64], nz: usize) -> [u8; M] {
    let mut live = [false; 64];
    let mut k = 0;
    while k < nz && k < 64 {
        live[perm[k] as usize & 63] = true;
        k += 1;
    }
    let mut kinds = [kind::SKIP; M];
    let mut i = 0;
    while i < prog.len && i < M {
        let op = prog.ops[i];
        let (a, b) = (op.a as usize & 63, op.b as usize & 63);
        let (la, lb) = (live[a], live[b]);
        if la || lb {
            kinds[i] = if op.had {
                // The sum goes to `b` when flipped.
                let (first, second) = if op.flip { (lb, la) } else { (la, lb) };
                if first && second {
                    kind::HAD
                } else if first {
                    kind::HAD_FIRST
                } else {
                    kind::HAD_SECOND
                }
            } else if la && lb || !one_multiply(op.cos) || !one_multiply(op.sin) {
                kind::ROT
            } else if la {
                kind::ROT_A
            } else {
                kind::ROT_B
            };
            live[a] = true;
            live[b] = true;
        }
        i += 1;
    }
    kinds
}

/// A permutation of fewer than 64 entries in a table of 64.
#[must_use]
pub const fn padded<const N: usize>(p: [u8; N]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut i = 0;
    while i < N && i < 64 {
        out[i] = p[i];
        i += 1;
    }
    out
}

/// The step tables of the transforms, and where their inputs and outputs
/// sit in `T`.
pub mod table {
    use super::{
        Prog, adst_in_perm, adst_out_perm, adst8_prog, adst16_prog, dct_perm, dct_prog, padded,
    };

    pub const DCT4: Prog<256> = dct_prog(2);
    pub const DCT8: Prog<256> = dct_prog(3);
    pub const DCT16: Prog<256> = dct_prog(4);
    pub const DCT32: Prog<256> = dct_prog(5);
    pub const DCT64: Prog<256> = dct_prog(6);
    pub const ADST8: Prog<256> = adst8_prog();
    pub const ADST16: Prog<256> = adst16_prog();

    pub const DCT4_IN: [u8; 64] = padded(dct_perm::<4>());
    pub const DCT8_IN: [u8; 64] = padded(dct_perm::<8>());
    pub const DCT16_IN: [u8; 64] = padded(dct_perm::<16>());
    pub const DCT32_IN: [u8; 64] = padded(dct_perm::<32>());
    pub const DCT64_IN: [u8; 64] = dct_perm::<64>();
    pub const ADST8_IN: [u8; 64] = padded(adst_in_perm::<8>());
    pub const ADST16_IN: [u8; 64] = padded(adst_in_perm::<16>());
    /// Output `i` of a DCT is entry `i`.
    pub const SAME: [u8; 64] = dct_perm_identity();
    pub const ADST8_OUT: [u8; 64] = padded(adst_out_perm::<8>());
    pub const ADST16_OUT: [u8; 64] = padded(adst_out_perm::<16>());

    const fn dct_perm_identity() -> [u8; 64] {
        let mut p = [0u8; 64];
        let mut k = 0;
        while k < 64 {
            p[k] = k as u8;
            k += 1;
        }
        p
    }
}
