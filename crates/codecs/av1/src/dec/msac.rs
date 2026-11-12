//! Symbol decoder (spec 8.2).
//!
//! The window keeps the complement of the coded bits, as dav1d does, and the
//! CDFs are stored inverted: `icdf[i] = 32768 - cdf[i]` for the first N-1
//! entries, with the adaptation counter in the last slot. A CDF for N symbols
//! is therefore an array of N `u16`.

const WIN: u32 = 64;
const PROB_SHIFT: u32 = 6;
const MIN_PROB: u32 = 4;

#[derive(Clone, Copy)]
pub struct Msac<'a> {
    /// The bytes not yet read.
    buf: &'a [u8],
    dif: u64,
    rng: u32,
    cnt: i32,
    pub allow_update: bool,
}

impl<'a> Msac<'a> {
    pub fn new(data: &'a [u8], disable_cdf_update: bool) -> Self {
        let mut s = Self {
            buf: data,
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update: !disable_cdf_update,
        };
        s.refill();
        s
    }

    #[inline(always)]
    fn refill(&mut self) {
        let c = WIN as i32 - self.cnt - 24;
        // Fast path: eight bytes at once. Bits of the byte after the last
        // whole one consumed may land below `c`; the next refill ORs the
        // same bits in again.
        if let Some(b) = self.buf.first_chunk::<8>() {
            let v = !u64::from_be_bytes(*b);
            self.dif |= v >> (56 - c);
            let n = (c >> 3) + 1;
            self.buf = self.buf.get(n as usize..).unwrap_or_default();
            self.cnt = WIN as i32 - (c - 8 * n) - 24;
            return;
        }
        let (buf, dif, cnt) = refill_slow(self.buf, self.dif, self.cnt);
        self.buf = buf;
        self.dif = dif;
        self.cnt = cnt;
    }

    /// True once the decoder has consumed more bits than the data holds,
    /// beyond the look-ahead its window allows: the ones it pads the end
    /// with are being read as symbols. libdav1d refuses a tile in this
    /// state at the end of each superblock row.
    pub fn overread(&self) -> bool {
        self.cnt <= -15
    }

    /// Runs `f` on a copy of the decoder state held in a local, then stores
    /// it back: inside a loop the state then stays in registers instead of
    /// going through memory for every symbol.
    #[inline(always)]
    pub fn with_local<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let mut m = *self;
        let r = f(&mut m);
        *self = m;
        r
    }

    #[inline(always)]
    fn norm(&mut self, dif: u64, rng: u32) {
        let d = 15 ^ (31 ^ rng.leading_zeros());
        let cnt = self.cnt;
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;
        if (cnt as u32) < d {
            self.refill();
        }
    }

    /// read_bool() with probability one half (also read_literal bits).
    #[inline(always)]
    pub fn bool_equi(&mut self) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = ((r >> 8) << 7) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = dif >= vw;
        // A select, not a branch: the bit is as good as random.
        (dif, v) = core::hint::select_unpredictable(
            ret,
            (dif.wrapping_sub(vw), r.wrapping_sub(v)),
            (dif, v),
        );
        self.norm(dif, v);
        !ret
    }

    /// Boolean with a fixed inverted probability `f` (no adaptation).
    #[inline(always)]
    pub fn bool_prob(&mut self, f: u32) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = (((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = dif >= vw;
        // A select, not a branch: the bit is as good as random.
        (dif, v) = core::hint::select_unpredictable(
            ret,
            (dif.wrapping_sub(vw), r.wrapping_sub(v)),
            (dif, v),
        );
        self.norm(dif, v);
        !ret
    }

    /// Adaptive boolean on a 2-entry inverted CDF.
    #[inline(always)]
    pub fn bool_adapt(&mut self, cdf: &mut [u16; 2]) -> bool {
        // `bool_prob` with the adaptation before the renormalization.
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = (((r >> 8) * (u32::from(cdf[0]) >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = dif >= vw;
        // A select, not a branch: the bit is as good as random.
        (dif, v) = core::hint::select_unpredictable(
            ret,
            (dif.wrapping_sub(vw), r.wrapping_sub(v)),
            (dif, v),
        );
        let bit = !ret;
        if self.allow_update {
            let count = cdf[1];
            let rate = 4 + (count >> 4);
            let c = cdf[0];
            cdf[0] =
                core::hint::select_unpredictable(bit, c + ((32768 - c) >> rate), c - (c >> rate));
            cdf[1] = count + u16::from(count < 32);
        }
        self.norm(dif, v);
        bit
    }

    /// Adaptive symbol over an N-entry inverted CDF (N symbols).
    ///
    /// Written on 16-bit lanes over the whole array (the counter lane gives
    /// `v = 0` and never counts), so the compiler can keep it in vector
    /// registers: `v[i] = ((r * (cdf[i] >> 6)) >> 1) + 4 * (n - i)` is the
    /// high half of `(rng & 0xff00) * ((cdf[i] >> 6) << 7)`, and the lanes
    /// with `c < v[i]` are exactly the symbols below the decoded one, which
    /// adapt upwards.
    #[inline(always)]
    pub fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        if N <= 4 {
            return self.symbol_small(cdf);
        }
        let n = N - 1; // number of probability entries
        let c = (self.dif >> (WIN - 16)) as u16;
        let r = (self.rng & 0xff00) as u16;
        let mut v = [0u16; N];
        let mut below = [0u16; N];
        for i in 0..N {
            let p = (cdf[i] >> PROB_SHIFT) << (7 + PROB_SHIFT - 6);
            let m = ((u32::from(r) * u32::from(p)) >> 16) as u16;
            v[i] = m.wrapping_add((MIN_PROB as usize * n.saturating_sub(i)) as u16);
            below[i] = if c < v[i] { 0xFFFF } else { 0 };
        }
        let val = below.iter().map(|&b| usize::from(b & 1)).sum::<usize>();
        let u = if val == 0 {
            self.rng
        } else {
            u32::from(v[val.saturating_sub(1)])
        };
        let vv = u32::from(v[val.min(n)]);
        if self.allow_update {
            let count = cdf[n];
            let rate = 4 + (count >> 4) + u16::from(n > 2);
            for i in 0..N {
                let x = cdf[i];
                let up = x.wrapping_add(32768u16.wrapping_sub(x) >> rate);
                let down = x.wrapping_sub(x >> rate);
                cdf[i] = (up & below[i]) | (down & !below[i]);
            }
            cdf[n] = count + u16::from(count < 32);
        }
        self.norm(self.dif - (u64::from(vv) << (WIN - 16)), u - vv);
        val
    }

    /// [`Msac::symbol`] for at most 4 symbols: the thresholds in scalar
    /// registers and the adaptation on the four 16-bit lanes of one `u64`.
    #[inline(always)]
    fn symbol_small<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        let n = N - 1;
        let c = (self.dif >> (WIN - 16)) as u32;
        let r8 = self.rng >> 8;
        // The thresholds decrease with the index, so the symbols below the
        // decoded one are those whose threshold exceeds `c`.
        let mut u = self.rng;
        let mut vv = 0;
        let mut val = 0;
        let mut prev = 0;
        for i in 0..n {
            let v = ((r8 * (u32::from(cdf[i]) >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * (n - i) as u32;
            if i == 0 {
                vv = v;
            } else if c < prev {
                u = prev;
                vv = v;
            }
            val += usize::from(c < v);
            prev = v;
        }
        if c < prev {
            u = prev;
            vv = 0;
        }
        // The adaptation goes first (see `Hot::sym4`).
        if self.allow_update {
            // All N lanes, the counter included, in one word.
            let mut w = 0u64;
            for (i, &v) in cdf.iter().enumerate() {
                w |= u64::from(v) << (16 * i);
            }
            let count = (w >> (16 * n)) as u32;
            let rate = 4 + (count >> 4) + u32::from(n > 2);
            let lanes = (1u64 << (16 * n)) - 1;
            let low = 0x0001_0001_0001_0001 & lanes;
            let keep = low * (0xFFFF >> rate);
            // The counter counts first; `keep` is zero on its lane, so it
            // passes through both sums unchanged. Every stored lane then
            // comes from the one word `y`, and the stores merge into one
            // that a later load of this CDF is forwarded from.
            let x = w + (u64::from(count < 32) << (16 * n));
            let down = x - ((x >> rate) & keep);
            let up = x + ((((low << 15).wrapping_sub(x)) >> rate) & keep);
            let m = (1u64 << (16 * val)) - 1;
            let y = down ^ ((up ^ down) & m);
            for (i, v) in cdf.iter_mut().enumerate() {
                *v = (y >> (16 * i)) as u16;
            }
        }
        self.norm(self.dif - (u64::from(vv) << (WIN - 16)), u - vv);
        val
    }

    /// L(n): n raw bits, MSB first.
    #[inline(always)]
    pub fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | u32::from(self.bool_equi());
        }
        x
    }

    /// read_golomb()
    #[inline(always)]
    pub fn golomb(&mut self) -> u32 {
        let (m, v) = Self::golomb_by_value(*self);
        *self = m;
        v
    }

    /// [`Msac::golomb`] on a copy of the state, which keeps the caller's
    /// state out of memory.
    #[inline(never)]
    fn golomb_by_value(mut m: Self) -> (Self, u32) {
        let v = m.golomb_inner();
        (m, v)
    }

    fn golomb_inner(&mut self) -> u32 {
        let mut len = 0u32;
        loop {
            if self.bool_equi() {
                break;
            }
            len += 1;
            if len == 32 {
                return u32::MAX;
            }
        }
        let mut x = 1u32;
        for _ in 0..len {
            x = (x << 1) | u32::from(self.bool_equi());
        }
        x - 1
    }

    /// NS(n) as coded with the arithmetic decoder (spec 4.10.10 via read_literal).
    pub fn ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let v = self.literal(w - 1);
        if v < m {
            return v;
        }
        let extra = self.literal(1);
        (v << 1) - m + extra
    }
}

/// A [`Msac`] whose window, range and bit count are locals for the length of
/// a symbol loop, with the byte buffer left behind in the decoder: only a
/// refill, once in tens of symbols, goes there. A loop then has the
/// registers for its own state. [`Hot::done`] writes the state back.
pub struct Hot<'m, 'a> {
    dif: u64,
    rng: u32,
    cnt: i32,
    m: &'m mut Msac<'a>,
}

impl<'m, 'a> Hot<'m, 'a> {
    #[inline(always)]
    pub fn new(m: &'m mut Msac<'a>) -> Self {
        Self {
            dif: m.dif,
            rng: m.rng,
            cnt: m.cnt,
            m,
        }
    }

    #[inline(always)]
    pub fn done(self) {
        self.m.dif = self.dif;
        self.m.rng = self.rng;
        self.m.cnt = self.cnt;
    }

    #[inline(always)]
    fn norm(&mut self, dif: u64, rng: u32) {
        let d = 15 ^ (31 ^ rng.leading_zeros());
        let cnt = self.cnt;
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;
        if (cnt as u32) < d {
            (self.m.buf, self.dif, self.cnt) = hot_refill(self.m.buf, self.dif, self.cnt);
        }
    }

    /// [`Msac::bool_equi`].
    #[inline(always)]
    pub fn bool_equi(&mut self) -> bool {
        let r = self.rng;
        let v = ((r >> 8) << 7) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = self.dif >= vw;
        let (dif, rng) = if ret {
            (self.dif - vw, r - v)
        } else {
            (self.dif, v)
        };
        self.norm(dif, rng);
        !ret
    }

    /// [`Msac::golomb`].
    #[inline(always)]
    pub fn golomb(&mut self) -> u32 {
        self.m.dif = self.dif;
        self.m.rng = self.rng;
        self.m.cnt = self.cnt;
        let v = self.m.golomb();
        (self.dif, self.rng, self.cnt) = (self.m.dif, self.m.rng, self.m.cnt);
        v
    }

    /// [`Msac::symbol`] for 4 symbols, adapting the CDF whatever
    /// `allow_update` says: a caller checks that once.
    #[inline(always)]
    pub fn sym4(&mut self, cdf: &mut u64) -> u32 {
        // The whole CDF, counter included, in one load (see
        // [`super::cdf::pack4`]).
        let w = *cdf;
        let c = (self.dif >> (WIN - 16)) as u32;
        let r8 = self.rng >> 8;
        let v = |i: usize| {
            ((r8 * ((w >> (16 * i + PROB_SHIFT as usize)) as u32 & 0x3FF)) >> (7 - PROB_SHIFT))
                + MIN_PROB * (3 - i as u32)
        };
        let (v0, v1, v2) = (v(0), v(1), v(2));
        // The thresholds decrease: `b[i]` holds exactly for the symbols
        // below the decoded one.
        let (b0, b1, b2) = (c < v0, c < v1, c < v2);
        let val = u32::from(b0) + u32::from(b1) + u32::from(b2);
        let (mut u, mut vv) = (self.rng, v0);
        if b0 {
            (u, vv) = (v0, v1);
        }
        if b1 {
            (u, vv) = (v1, v2);
        }
        if b2 {
            (u, vv) = (v2, 0);
        }
        // As in `Msac::symbol_small`, on one word stored at once. The
        // counter (at most 32) picks the rate: `k = count >> 4`. Most CDFs
        // soon reach 32, the last rate, with constant shifts.
        let below = SYM4_BELOW[val as usize & 3];
        *cdf = if w >> 53 != 0 {
            let keep = 0x0001_0001_0001 * (0xFFFF >> 7);
            let down = w - ((w >> 7) & keep);
            let up = w + (((0x8000_8000_8000u64.wrapping_sub(w)) >> 7) & keep);
            down ^ ((up ^ down) & below)
        } else {
            sym4_adapt_young(w, below)
        };
        self.norm(self.dif - (u64::from(vv) << (WIN - 16)), u - vv);
        val
    }
}

/// [`Hot::sym4`]'s adaptation of a CDF whose counter is below 32; `below`
/// masks the lanes of the symbols below the decoded one.
#[cold]
#[inline(never)]
fn sym4_adapt_young(w: u64, below: u64) -> u64 {
    let k = (w >> 52) as usize & 3;
    let rate = 5 + k as u32;
    let x = w + SYM4_INC[k];
    let down = x - ((x >> rate) & SYM4_KEEP[k]);
    let up = x + (((0x8000_8000_8000u64.wrapping_sub(x)) >> rate) & SYM4_KEEP[k]);
    down ^ ((up ^ down) & below)
}

/// [`Hot::sym4`]'s lane masks of the adaptation per `count >> 4`: the bits
/// of `x >> rate` that come from the lane itself, on the probability lanes.
const SYM4_KEEP: [u64; 4] = [
    0x0001_0001_0001 * (0xFFFF >> 5),
    0x0001_0001_0001 * (0xFFFF >> 6),
    0x0001_0001_0001 * (0xFFFF >> 7),
    0x0001_0001_0001 * (0xFFFF >> 7),
];
/// The counter increment per `count >> 4` (it stops at 32).
const SYM4_INC: [u64; 4] = [1 << 48, 1 << 48, 0, 0];
/// The probability lanes of the symbols below the decoded one.
const SYM4_BELOW: [u64; 4] = [0, 0xFFFF, 0xFFFF_FFFF, 0xFFFF_FFFF_FFFF];

#[cold]
#[inline(never)]
fn hot_refill(buf: &[u8], dif: u64, cnt: i32) -> (&[u8], u64, i32) {
    let mut m = Msac {
        buf,
        dif,
        rng: 0,
        cnt,
        allow_update: false,
    };
    m.refill();
    (m.buf, m.dif, m.cnt)
}

/// The refill near the end of the data. Takes and returns plain values, so
/// a caller keeping the state in a local (see [`Msac::with_local`]) can
/// keep it in registers.
#[cold]
#[inline(never)]
fn refill_slow(mut buf: &[u8], mut dif: u64, cnt: i32) -> (&[u8], u64, i32) {
    let mut c = WIN as i32 - cnt - 24;
    loop {
        let Some((&b, rest)) = buf.split_first() else {
            dif |= !(!0xffu64 << c);
            break;
        };
        dif |= u64::from(b ^ 0xff) << c;
        buf = rest;
        c -= 8;
        if c < 0 {
            break;
        }
    }
    (buf, dif, WIN as i32 - c - 24)
}

/// Converts a spec CDF row (N values ending in 32768, then the counter) to
/// the inverted N-entry form.
pub const fn invert<const N: usize>(spec: &[u16]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i + 1 < N {
        out[i] = 32768 - spec[i];
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Literal transliteration of spec 8.2 for comparison.
    struct SpecDec<'a> {
        data: &'a [u8],
        bitpos: usize,
        range: u32,
        value: u32,
        max_bits: i64,
    }
    impl<'a> SpecDec<'a> {
        fn f(&mut self, n: u32) -> u32 {
            let mut x = 0;
            for _ in 0..n {
                let byte = self.data.get(self.bitpos >> 3).copied().unwrap_or(0);
                x = (x << 1) | u32::from((byte >> (7 - (self.bitpos & 7))) & 1);
                self.bitpos += 1;
            }
            x
        }
        fn new(data: &'a [u8]) -> Self {
            let mut s = Self {
                data,
                bitpos: 0,
                range: 0,
                value: 0,
                max_bits: 0,
            };
            let sz = data.len();
            let num_bits = 15.min(8 * sz as u32);
            let buf = s.f(num_bits);
            let padded = buf << (15 - num_bits);
            s.value = ((1 << 15) - 1) ^ padded;
            s.range = 1 << 15;
            s.max_bits = 8 * sz as i64 - 15;
            s
        }
        fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
            let n = cdf.len() - 1;
            let mut cur = self.range;
            let mut symbol: i32 = -1;
            let mut prev;
            loop {
                symbol += 1;
                prev = cur;
                let f = (1u32 << 15) - u32::from(cdf[symbol as usize]);
                cur = (((self.range >> 8) * (f >> 6)) >> 1) + 4 * (n as u32 - symbol as u32 - 1);
                if self.value >= cur {
                    break;
                }
            }
            self.range = prev - cur;
            self.value -= cur;
            let bits = 15 - (31 - self.range.leading_zeros());
            self.range <<= bits;
            let num_bits = (bits as i64).min(self.max_bits.max(0)) as u32;
            let newdata = self.f(num_bits);
            let paddedata = newdata << (bits - num_bits);
            self.value = paddedata ^ (((self.value + 1) << bits) - 1);
            self.max_bits -= i64::from(bits);
            // adaptation
            let rate = 3
                + u32::from(cdf[n] > 15)
                + u32::from(cdf[n] > 31)
                + (31 - (n as u32).leading_zeros()).min(2);
            let mut tmp = 0u32;
            for i in 0..n - 1 {
                if i as i32 == symbol {
                    tmp = 1 << 15;
                }
                let c = u32::from(cdf[i]);
                if tmp < c {
                    cdf[i] -= ((c - tmp) >> rate) as u16;
                } else {
                    cdf[i] += ((tmp - c) >> rate) as u16;
                }
            }
            cdf[n] += u16::from(cdf[n] < 32);
            symbol as usize
        }
    }

    #[test]
    fn matches_spec_decoder() {
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..200 {
            let len = (rnd() % 256 + 256) as usize;
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            let mut a = Msac::new(&data, false);
            let mut b = SpecDec::new(&data);
            // 4-symbol cdf
            let mut spec_cdf = [8000u16, 16000, 24000, 32768, 0];
            let mut inv: [u16; 4] = invert(&spec_cdf);
            let mut spec2 = [20000u16, 32768, 0];
            let mut inv2: [u16; 2] = invert(&spec2);
            for k in 0..150 {
                if k % 3 == 0 {
                    let x = a.bool_adapt(&mut inv2);
                    let y = b.read_symbol(&mut spec2);
                    assert_eq!(usize::from(x), y);
                } else {
                    let x = a.symbol(&mut inv);
                    let y = b.read_symbol(&mut spec_cdf);
                    assert_eq!(x, y, "k={k} len={len}");
                    let back: [u16; 4] = invert(&spec_cdf);
                    assert_eq!(&back[..3], &inv[..3]);
                }
            }
        }
    }
}
