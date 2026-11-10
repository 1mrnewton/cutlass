//! The one-dimensional transforms as straight-line code.
//!
//! A transform is a table of steps ([`super::steps`]). Here each step of a
//! table becomes its own piece of code, with the entries it works on, its
//! multipliers and what it comes to for the inputs that can be non-zero
//! all constants. So the entries of `T` are plain values the compiler
//! keeps in registers, and a step whose inputs are zero is no code at
//! all. Both the vector kernel and its scalar twin are made from these
//! macros, each with its own `ops` module for the arithmetic of a step.

/// `$m!(args, i)` for four values of `i` from `$b`.
macro_rules! rep4 {
    ($m:ident ! ( $($a:tt)* ), $b:expr) => {
        $m!($($a)*, $b);
        $m!($($a)*, $b + 1);
        $m!($($a)*, $b + 2);
        $m!($($a)*, $b + 3);
    };
}

macro_rules! rep16 {
    ($m:ident ! ( $($a:tt)* ), $b:expr) => {
        rep4!($m!($($a)*), $b);
        rep4!($m!($($a)*), $b + 4);
        rep4!($m!($($a)*), $b + 8);
        rep4!($m!($($a)*), $b + 12);
    };
}

macro_rules! rep64 {
    ($m:ident ! ( $($a:tt)* ), $b:expr) => {
        rep16!($m!($($a)*), $b);
        rep16!($m!($($a)*), $b + 16);
        rep16!($m!($($a)*), $b + 32);
        rep16!($m!($($a)*), $b + 48);
    };
}

macro_rules! rep128 {
    ($m:ident ! ( $($a:tt)* ), $b:expr) => {
        rep64!($m!($($a)*), $b);
        rep64!($m!($($a)*), $b + 64);
    };
}

macro_rules! rep256 {
    ($m:ident ! ( $($a:tt)* ), $b:expr) => {
        rep128!($m!($($a)*), $b);
        rep128!($m!($($a)*), $b + 128);
    };
}

/// Input `$k` of the transform to its entry of `T`, when it can be
/// non-zero.
macro_rules! place {
    ($v:ident, $t:ident, $k:expr) => {{
        const K: usize = $k;
        const AT: usize = if K < N { IN[K] as usize } else { 0 };
        if K < NZ && K < N {
            $v[AT] = $t[K];
        }
    }};
}

/// Step `$i` of the transform's table.
macro_rules! step {
    ($v:ident, $i:expr) => {{
        const I: usize = $i;
        const ST: Step = PROG.ops[I];
        const A: usize = ST.a as usize;
        const B: usize = ST.b as usize;
        // Where the first and the second result of the step go.
        const FIRST: usize = if ST.flip { B } else { A };
        const SECOND: usize = if ST.flip { A } else { B };
        const KIND: u8 = KINDS[I];
        match KIND {
            kind::ROT => {
                let (p, q) = ops::rot($v[A], $v[B], ST.cos, ST.sin);
                $v[FIRST] = p;
                $v[SECOND] = q;
            }
            kind::ROT_A => {
                let x = $v[A];
                $v[FIRST] = ops::scale(x, ST.cos);
                $v[SECOND] = ops::scale(x, ST.sin);
            }
            kind::ROT_B => {
                let y = $v[B];
                $v[FIRST] = ops::scale(y, -ST.sin);
                $v[SECOND] = ops::scale(y, ST.cos);
            }
            kind::HAD => {
                let (x, y) = ($v[FIRST], $v[SECOND]);
                $v[FIRST] = ops::add(x, y);
                $v[SECOND] = ops::sub(x, y);
            }
            kind::HAD_FIRST => $v[SECOND] = $v[FIRST],
            kind::HAD_SECOND => {
                let y = $v[SECOND];
                $v[FIRST] = y;
                $v[SECOND] = ops::neg(y);
            }
            _ => {}
        }
    }};
}

/// Output `$i` of the transform from its entry of `T`.
macro_rules! take {
    ($v:ident, $t:ident, $i:expr) => {{
        const I: usize = $i;
        const AT: usize = if I < N { OUT[I] as usize } else { 0 };
        if I < N {
            $t[I] = if NEGATE_ODD && I & 1 == 1 {
                ops::neg($v[AT])
            } else {
                $v[AT]
            };
        }
    }};
}

/// A transform of `$n` values whose inputs from `$nz` on are zero: reads
/// `t[..$nz]`, writes `t[..$n]`. `$rep` expands at least as many steps as
/// the table has.
macro_rules! tx1d {
    ($(#[$attr:meta])* $name:ident, $rep:ident, $n:literal, $prog:ident, $in:ident, $out:ident, $neg:literal, $nz:literal) => {
        $(#[$attr])*
        fn $name(t: &mut [ops::V; 64]) {
            const N: usize = $n;
            const NZ: usize = $nz;
            const PROG: Prog<256> = table::$prog;
            const IN: [u8; 64] = table::$in;
            const OUT: [u8; 64] = table::$out;
            const NEGATE_ODD: bool = $neg;
            const KINDS: [u8; 256] = plan(&PROG, &IN, NZ);
            let mut v = [ops::zero(); N];
            rep64!(place!(v, t), 0);
            $rep!(step!(v), 0);
            rep64!(take!(v, t), 0);
        }
    };
}

/// Every transform, and [`run_1d`] to pick one: the transform `kind` of
/// `n` values for inputs that are zero from `nz` on. Returns `false` for
/// a transform there is none for.
macro_rules! transforms {
    ($(#[$attr:meta])*) => {
        tx1d!($(#[$attr])* dct4_1, rep16, 4, DCT4, DCT4_IN, SAME, false, 1);
        tx1d!($(#[$attr])* dct4_4, rep16, 4, DCT4, DCT4_IN, SAME, false, 4);
        tx1d!($(#[$attr])* dct8_1, rep16, 8, DCT8, DCT8_IN, SAME, false, 1);
        tx1d!($(#[$attr])* dct8_4, rep16, 8, DCT8, DCT8_IN, SAME, false, 4);
        tx1d!($(#[$attr])* dct8_8, rep16, 8, DCT8, DCT8_IN, SAME, false, 8);
        tx1d!($(#[$attr])* dct16_1, rep64, 16, DCT16, DCT16_IN, SAME, false, 1);
        tx1d!($(#[$attr])* dct16_4, rep64, 16, DCT16, DCT16_IN, SAME, false, 4);
        tx1d!($(#[$attr])* dct16_8, rep64, 16, DCT16, DCT16_IN, SAME, false, 8);
        tx1d!($(#[$attr])* dct16_16, rep64, 16, DCT16, DCT16_IN, SAME, false, 16);
        tx1d!($(#[$attr])* dct32_1, rep128, 32, DCT32, DCT32_IN, SAME, false, 1);
        tx1d!($(#[$attr])* dct32_4, rep128, 32, DCT32, DCT32_IN, SAME, false, 4);
        tx1d!($(#[$attr])* dct32_8, rep128, 32, DCT32, DCT32_IN, SAME, false, 8);
        tx1d!($(#[$attr])* dct32_16, rep128, 32, DCT32, DCT32_IN, SAME, false, 16);
        tx1d!($(#[$attr])* dct32_32, rep128, 32, DCT32, DCT32_IN, SAME, false, 32);
        tx1d!($(#[$attr])* dct64_1, rep256, 64, DCT64, DCT64_IN, SAME, false, 1);
        tx1d!($(#[$attr])* dct64_4, rep256, 64, DCT64, DCT64_IN, SAME, false, 4);
        tx1d!($(#[$attr])* dct64_8, rep256, 64, DCT64, DCT64_IN, SAME, false, 8);
        tx1d!($(#[$attr])* dct64_16, rep256, 64, DCT64, DCT64_IN, SAME, false, 16);
        tx1d!($(#[$attr])* dct64_32, rep256, 64, DCT64, DCT64_IN, SAME, false, 32);

        $(#[$attr])*
        pub(super) fn run_1d(kind: Kind, n: usize, nz: usize, t: &mut [ops::V; 64]) -> bool {
            match (kind, n, nz) {
                (Kind::Dct, 4, 0 | 1) => dct4_1(t),
                (Kind::Dct, 4, _) => dct4_4(t),
                (Kind::Dct, 8, 0 | 1) => dct8_1(t),
                (Kind::Dct, 8, 2..=4) => dct8_4(t),
                (Kind::Dct, 8, _) => dct8_8(t),
                (Kind::Dct, 16, 0 | 1) => dct16_1(t),
                (Kind::Dct, 16, 2..=4) => dct16_4(t),
                (Kind::Dct, 16, 5..=8) => dct16_8(t),
                (Kind::Dct, 16, _) => dct16_16(t),
                (Kind::Dct, 32, 0 | 1) => dct32_1(t),
                (Kind::Dct, 32, 2..=4) => dct32_4(t),
                (Kind::Dct, 32, 5..=8) => dct32_8(t),
                (Kind::Dct, 32, 9..=16) => dct32_16(t),
                (Kind::Dct, 32, _) => dct32_32(t),
                (Kind::Dct, 64, 0 | 1) => dct64_1(t),
                (Kind::Dct, 64, 2..=4) => dct64_4(t),
                (Kind::Dct, 64, 5..=8) => dct64_8(t),
                (Kind::Dct, 64, 9..=16) => dct64_16(t),
                (Kind::Dct, 64, _) => dct64_32(t),
                _ => return false,
            }
            true
        }
    };
}
