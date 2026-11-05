//! Symbol encoder: the exact inverse of `dec::msac::Msac` (spec 8.2).
//!
//! CDFs use the decoder's inverted layout (`icdf[i] = 32768 - cdf[i]`, the
//! adaptation counter in the last slot) and adapt exactly as the decoder does.
//!
//! The coder keeps the interval base `low` and range `rng` (16 bits after
//! normalization). `low` holds the `15 + pend` bits of the base that are not
//! yet in `out`. Whole bytes move to `out` once `pend` reaches
//! `FLUSH_BITS`, so the 64-bit window never overflows; the bits of `low`
//! from `15 + pend` up are a carry (at most 2, since the remainder after a
//! flush plus `rng` can exceed one unit of the last byte) that ripples back
//! through `out`.

const PROB_SHIFT: u32 = 6;
const MIN_PROB: u32 = 4;
const FLUSH_BITS: u32 = 32;

/// Fixed-point fraction bits of the rate estimates (1/256 bit units).
pub const COST_SHIFT: u32 = 8;

pub struct MsacEnc {
    out: Vec<u8>,
    low: u64,
    rng: u32,
    pend: u32,
    pub allow_update: bool,
    /// With [`MsacEnc::record`]: every coded interval, independent of the
    /// coder state, so the stream can be coded again with insertions.
    log: Option<Vec<Op>>,
    /// Positions in `log` tagged by [`MsacEnc::mark`].
    marks: Vec<(usize, u32)>,
}

/// One coded interval: `u` and `v` of [`MsacEnc::encode`] as functions of
/// the range. Bit 32 set: `u` is the whole range; bit 33 set: `v` is 0.
/// Otherwise `u` comes from bits 0..16 (probability) and 40..48 (count of
/// `MIN_PROB`), `v` from bits 16..32 and 48..56.
#[derive(Clone, Copy)]
struct Op(u64);

impl Op {
    const U_FULL: u64 = 1 << 32;
    const V_ZERO: u64 = 1 << 33;

    #[inline(always)]
    fn new(u: Option<(u16, u32)>, v: Option<(u16, u32)>) -> Self {
        let mut x = 0u64;
        match u {
            Some((f, c)) => x |= u64::from(f) | (u64::from(c) << 40),
            None => x |= Self::U_FULL,
        }
        match v {
            Some((f, c)) => x |= (u64::from(f) << 16) | (u64::from(c) << 48),
            None => x |= Self::V_ZERO,
        }
        Self(x)
    }
}

impl MsacEnc {
    pub fn new(allow_update: bool) -> Self {
        Self::with_capacity(allow_update, 4096)
    }

    pub fn with_capacity(allow_update: bool, bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            low: 0,
            rng: 0x8000,
            pend: 0,
            allow_update,
            log: None,
            marks: Vec::new(),
        }
    }

    /// Starts logging every coded interval (for [`MsacEnc::replay`]).
    pub fn record(&mut self) {
        self.log = Some(Vec::new());
    }

    /// Tags the current position of the log (no-op unless recording).
    pub fn mark(&mut self, tag: u32) {
        if let Some(l) = &self.log {
            self.marks.push((l.len(), tag));
        }
    }

    /// The tags in coding order.
    pub fn marks(&self) -> impl Iterator<Item = u32> + '_ {
        self.marks.iter().map(|m| m.1)
    }

    /// Codes the logged intervals again into a new stream, inserting before
    /// each mark the literal `insert(tag) = (bits, value)`; returns the
    /// finished stream. Without a log, finishes this one unchanged.
    pub fn replay(self, insert: &dyn Fn(u32) -> (u32, u32)) -> Vec<u8> {
        let Some(log) = self.log else {
            return self.finish();
        };
        let mut e = MsacEnc::with_capacity(self.allow_update, self.out.len() + 64);
        let mut marks = self.marks.iter().peekable();
        for (i, op) in log.iter().enumerate() {
            while let Some(&&(at, tag)) = marks.peek() {
                if at != i {
                    break;
                }
                let (n, v) = insert(tag);
                e.literal(n, v);
                marks.next();
            }
            e.encode_op(*op);
        }
        for &(_, tag) in marks {
            let (n, v) = insert(tag);
            e.literal(n, v);
        }
        e.finish()
    }

    #[inline(always)]
    fn encode_op(&mut self, op: Op) {
        let r = self.rng >> 8;
        let x = op.0;
        let u = if x & Op::U_FULL != 0 {
            self.rng
        } else {
            ((r * ((x & 0xffff) as u32 >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * ((x >> 40) & 0xff) as u32
        };
        let v = if x & Op::V_ZERO != 0 {
            0
        } else {
            ((r * (((x >> 16) & 0xffff) as u32 >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * ((x >> 48) & 0xff) as u32
        };
        self.encode(u, v);
    }

    #[inline(always)]
    fn log_op(&mut self, u: Option<(u16, u32)>, v: Option<(u16, u32)>) {
        if let Some(l) = &mut self.log {
            l.push(Op::new(u, v));
        }
    }

    /// Bits the stream holds so far, including the final trailing one bit.
    pub fn tell_bits(&self) -> u64 {
        self.out.len() as u64 * 8 + u64::from(self.pend) + 1
    }

    /// Adds `c` at the last byte of `out`, rippling the carry back.
    #[inline(always)]
    fn carry(&mut self, mut c: u64) {
        for b in self.out.iter_mut().rev() {
            let t = u64::from(*b) + c;
            *b = t as u8;
            c = t >> 8;
            if c == 0 {
                return;
            }
        }
    }

    #[inline(never)]
    fn flush(&mut self) {
        let top = 15 + self.pend;
        let c = self.low >> top;
        if c != 0 {
            self.carry(c);
            self.low &= (1u64 << top) - 1;
        }
        let n = self.pend >> 3;
        let mut shift = top;
        for _ in 0..n {
            shift -= 8;
            self.out.push((self.low >> shift) as u8);
        }
        self.pend &= 7;
        self.low &= (1u64 << (15 + self.pend)) - 1;
    }

    /// Codes the sub-interval [rng - u, rng - v) of the current range.
    #[inline(always)]
    fn encode(&mut self, u: u32, v: u32) {
        let rng = self.rng;
        self.low += u64::from(rng - u);
        let r = u - v;
        let d = r.leading_zeros() - 16;
        self.low <<= d;
        self.rng = r << d;
        self.pend += d;
        if self.pend >= FLUSH_BITS {
            self.flush();
        }
    }

    /// Boolean with probability one half (read_bool / read_literal bits).
    #[inline]
    pub fn bool_equi(&mut self, bit: bool) {
        if self.log.is_some() {
            self.bool_log(16384, bit);
        }
        let v = ((self.rng >> 8) << 7) + MIN_PROB;
        if bit {
            self.encode(v, 0);
        } else {
            self.encode(self.rng, v);
        }
    }

    /// Boolean with a fixed inverted probability `f` (no adaptation).
    #[inline]
    pub fn bool_prob(&mut self, f: u32, bit: bool) {
        if self.log.is_some() {
            self.bool_log(f, bit);
        }
        let v = (((self.rng >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB;
        if bit {
            self.encode(v, 0);
        } else {
            self.encode(self.rng, v);
        }
    }

    #[inline(never)]
    fn bool_log(&mut self, f: u32, bit: bool) {
        if bit {
            self.log_op(Some((f as u16, 1)), None);
        } else {
            self.log_op(None, Some((f as u16, 1)));
        }
    }

    /// Adaptive boolean on a 2-entry inverted CDF.
    #[inline]
    pub fn bool_adapt(&mut self, cdf: &mut [u16; 2], bit: bool) {
        self.bool_prob(u32::from(cdf[0]), bit);
        if self.allow_update {
            let count = cdf[1];
            let rate = 4 + (count >> 4);
            if bit {
                cdf[0] += (32768 - cdf[0]) >> rate;
            } else {
                cdf[0] -= cdf[0] >> rate;
            }
            cdf[1] = count + u16::from(count < 32);
        }
    }

    /// Adaptive symbol `s` over an N-entry inverted CDF (N symbols).
    #[inline]
    pub fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], s: usize) {
        let n = N - 1;
        debug_assert!(s <= n);
        let r = self.rng >> 8;
        let u = if s == 0 {
            self.rng
        } else {
            ((r * (u32::from(cdf[s - 1]) >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * (n - s + 1) as u32
        };
        let v = if s >= n {
            0
        } else {
            ((r * (u32::from(cdf[s]) >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * (n - s) as u32
        };
        if self.log.is_some() {
            let lu = (s > 0).then(|| (cdf[s - 1], (n - s + 1) as u32));
            let lv = (s < n).then(|| (cdf[s], (n - s) as u32));
            self.log_op(lu, lv);
        }
        self.encode(u, v);
        if self.allow_update {
            let count = cdf[n];
            let rate = 4 + (count >> 4) + u16::from(n > 2);
            for i in 0..s.min(n) {
                cdf[i] += (32768 - cdf[i]) >> rate;
            }
            for i in s..n {
                cdf[i] -= cdf[i] >> rate;
            }
            cdf[n] = count + u16::from(count < 32);
        }
    }

    /// L(n): the low n bits of `v`, MSB first.
    #[inline]
    pub fn literal(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bool_equi((v >> i) & 1 != 0);
        }
    }

    /// The inverse of read_golomb(); `u32::MAX` is coded as 32 zeros.
    pub fn golomb(&mut self, v: u32) {
        let x = u64::from(v) + 1;
        let len = 63 - x.leading_zeros();
        for _ in 0..len {
            self.bool_equi(false);
        }
        if len >= 32 {
            return;
        }
        self.bool_equi(true);
        for i in (0..len).rev() {
            self.bool_equi((x >> i) & 1 != 0);
        }
    }

    /// NS(n) through the arithmetic coder.
    pub fn ns(&mut self, n: u32, v: u32) {
        if n <= 1 {
            return;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        if v < m {
            self.literal(w - 1, v);
        } else {
            let x = v + m;
            self.literal(w - 1, x >> 1);
            self.literal(1, x & 1);
        }
    }

    /// Flushes the coder: the shortest byte string that decodes to the coded
    /// symbols whatever follows, ending in the trailing one bit and zero
    /// padding that the exit process (spec 8.2.4) requires.
    pub fn finish(mut self) -> Vec<u8> {
        let top = 15 + self.pend;
        // A value in [low, low + rng) whose last one bit is the bit after
        // the symbol bits; rng >= 0x8000 keeps it in the interval.
        let mut e = ((self.low + 0x3fff) & !0x3fff) | 0x4000;
        let c = e >> top;
        if c != 0 {
            self.carry(c);
            e &= (1u64 << top) - 1;
        }
        let nbytes = (self.pend + 1).div_ceil(8);
        let mut shift = top as i32;
        for _ in 0..nbytes {
            shift -= 8;
            let b = if shift >= 0 {
                e >> shift
            } else {
                e << (-shift)
            };
            self.out.push(b as u8);
        }
        self.out
    }
}

/// `LOG2_FRAC[m]` = log2(1 + m / 256) in 1/256 units, rounded.
const LOG2_FRAC: [u16; 257] = log2_frac_table();

const fn log2_frac_table() -> [u16; 257] {
    let mut t = [0u16; 257];
    let mut m = 0;
    while m < 257 {
        // x in Q30 for 1 + m/256; extract 8 fraction bits by squaring.
        let mut x: u64 = (256 + m as u64) << 22;
        let mut frac = 0u16;
        let mut b = 0;
        while b < 9 {
            x = (x * x) >> 30;
            frac <<= 1;
            if x >= 2 << 30 {
                frac |= 1;
                x >>= 1;
            }
            b += 1;
        }
        t[m] = frac.div_ceil(2);
        m += 1;
    }
    t
}

/// log2(p) for p >= 1, in 1/256 units.
#[inline]
fn log2_q8(p: u32) -> u32 {
    let msb = 31 - p.leading_zeros();
    // Mantissa rounded to 8 fraction bits (256 rounds up to the next octave).
    let q9 = if msb >= 9 {
        p >> (msb - 9)
    } else {
        p << (9 - msb)
    };
    let mant = ((q9 + 1) >> 1) - 256;
    (msb << COST_SHIFT) + u32::from(LOG2_FRAC[mant as usize])
}

/// Cost of a symbol of probability `p / 32768`, in 1/256 bits.
#[inline]
pub fn prob_cost(p: u32) -> u32 {
    (15 << COST_SHIFT) - log2_q8(p.clamp(1, 32768))
}

/// Cost of `symbol` under an inverted CDF (N entries: N-1 probabilities and
/// the counter), in 1/256 bits.
#[inline]
pub fn symbol_cost(icdf: &[u16], symbol: usize) -> u32 {
    let n = icdf.len().saturating_sub(1);
    let hi = if symbol == 0 {
        32768
    } else {
        u32::from(icdf[symbol - 1])
    };
    let lo = if symbol >= n {
        0
    } else {
        u32::from(icdf[symbol])
    };
    prob_cost(hi.saturating_sub(lo))
}

/// Cost of a boolean coded with inverted probability `f` (as `bool_prob`),
/// in 1/256 bits.
#[inline]
pub fn bool_cost(f: u16, bit: bool) -> u32 {
    let f = u32::from(f);
    prob_cost(if bit { f } else { 32768 - f })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::cdf::CdfContext;
    use crate::dec::msac::Msac;

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
    }

    /// Spec 8.2 decoder on inverted probabilities, with the exit check.
    struct SpecDec<'a> {
        data: &'a [u8],
        pos: usize,
        range: u32,
        value: u32,
        max_bits: i64,
    }
    impl<'a> SpecDec<'a> {
        fn bit_at(&self, p: usize) -> u32 {
            let byte = self.data.get(p >> 3).copied().unwrap_or(0);
            u32::from((byte >> (7 - (p & 7))) & 1)
        }
        fn f(&mut self, n: u32) -> u32 {
            let mut x = 0;
            for _ in 0..n {
                x = (x << 1) | self.bit_at(self.pos);
                self.pos += 1;
            }
            x
        }
        fn new(data: &'a [u8]) -> Self {
            let mut s = Self {
                data,
                pos: 0,
                range: 1 << 15,
                value: 0,
                max_bits: 8 * data.len() as i64 - 15,
            };
            let num_bits = 15.min(8 * data.len() as u32);
            let buf = s.f(num_bits);
            s.value = ((1 << 15) - 1) ^ (buf << (15 - num_bits));
            s
        }
        /// `icdf` holds the N-1 inverted probabilities.
        fn read(&mut self, icdf: &[u16]) -> usize {
            let n = icdf.len() + 1;
            let mut cur = self.range;
            let mut symbol: i32 = -1;
            let mut prev;
            loop {
                symbol += 1;
                prev = cur;
                let f = icdf.get(symbol as usize).map_or(0, |&x| u32::from(x));
                cur = (((self.range >> 8) * (f >> 6)) >> 1) + 4 * (n as u32 - symbol as u32 - 1);
                if self.value >= cur {
                    break;
                }
            }
            self.range = prev - cur;
            self.value -= cur;
            let bits = 15 - (31 - self.range.leading_zeros());
            self.range <<= bits;
            let num_bits = (i64::from(bits)).min(self.max_bits.max(0)) as u32;
            let newdata = self.f(num_bits);
            let padded = newdata << (bits - num_bits);
            self.value = padded ^ (((self.value + 1) << bits) - 1);
            self.max_bits -= i64::from(bits);
            symbol as usize
        }
        fn exit_ok(&mut self) -> bool {
            if self.max_bits < -14 {
                return false;
            }
            let trailing = self.pos as i64 - (self.max_bits + 15).min(15);
            self.pos += self.max_bits.max(0) as usize;
            if self.pos != self.data.len() * 8 || trailing < 0 {
                return false;
            }
            let t = trailing as usize;
            self.bit_at(t) == 1 && (t + 1..self.pos).all(|p| self.bit_at(p) == 0)
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Equi(bool),
        Prob(u16, bool),
        Adapt(usize, bool),
        Sym(usize, usize),
        Lit(u32, u32),
        Golomb(u32),
        Ns(u32, u32),
    }

    fn enc_sym(e: &mut MsacEnc, cdf: &mut [u16], s: usize) {
        macro_rules! go {
            ($($n:literal)*) => {
                match cdf.len() {
                    $($n => e.symbol::<$n>(<&mut [u16; $n]>::try_from(cdf).unwrap(), s),)*
                    _ => unreachable!(),
                }
            };
        }
        go!(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }

    fn dec_sym(d: &mut Msac<'_>, cdf: &mut [u16]) -> usize {
        macro_rules! go {
            ($($n:literal)*) => {
                match cdf.len() {
                    $($n => d.symbol::<$n>(<&mut [u16; $n]>::try_from(cdf).unwrap()),)*
                    _ => unreachable!(),
                }
            };
        }
        go!(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }

    fn random_cdf(rng: &mut Rng, n: usize, extreme: bool) -> Vec<u16> {
        // n symbols: n-1 strictly decreasing inverted values in (0, 32768).
        let mut v: Vec<u16> = (0..n - 1)
            .map(|_| {
                if extreme {
                    match rng.below(3) {
                        0 => 1 + rng.below(64) as u16,
                        1 => 32767 - rng.below(64) as u16,
                        _ => 1 + rng.below(32767) as u16,
                    }
                } else {
                    1 + rng.below(32767) as u16
                }
            })
            .collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.dedup();
        while v.len() < n - 1 {
            let x = 1 + rng.below(32767) as u16;
            if !v.contains(&x) {
                v.push(x);
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        let mut cdf = v;
        cdf.push(rng.below(33) as u16);
        cdf
    }

    fn real_cdfs(q: u8) -> Vec<Vec<u16>> {
        let c = CdfContext::new(q);
        let mut v: Vec<Vec<u16>> = vec![
            c.intrabc.to_vec(),
            c.skip[0].to_vec(),
            c.tx_16x16[0].to_vec(),
            c.partition_w8[1].to_vec(),
            c.coeff_base[0][0][3].to_vec(),
            c.coeff_br[1][0][2].to_vec(),
            c.eob_pt_16[0][0].to_vec(),
            c.eob_pt_32[0][0].to_vec(),
            c.eob_pt_64[0][0].to_vec(),
            c.angle_delta[2].to_vec(),
            c.eob_pt_128[0][1].to_vec(),
            c.palette_size_8_y_color[0].to_vec(),
            c.eob_pt_256[1][0].to_vec(),
            c.partition_w32[2].to_vec(),
            c.mv_class[0][0].to_vec(),
            c.inter_tx_type_set2.to_vec(),
            c.y_mode[0].to_vec(),
            c.uv_mode_cfl_allowed[3].to_vec(),
            c.wedge_index[5].to_vec(),
            c.inter_tx_type_set1[0].to_vec(),
            c.cfl_alpha[1].to_vec(),
            c.coeff_base_eob[2][0][1].to_vec(),
        ];
        v.retain(|x| (2..=16).contains(&x.len()));
        v
    }

    fn run(seed: u64, count: usize, allow_update: bool, extreme: bool, spec_check: bool) {
        let mut rng = Rng(seed);
        let mut cdfs = real_cdfs([0, 40, 100, 200][rng.below(4) as usize]);
        for n in 2..=16 {
            for _ in 0..2 {
                cdfs.push(random_cdf(&mut rng, n, extreme));
            }
        }
        let mut bools: Vec<[u16; 2]> = (0..8)
            .map(|_| {
                let f = if extreme {
                    [1u16, 4, 32767, 32760, 100, 32700][rng.below(6) as usize]
                } else {
                    1 + rng.below(32767) as u16
                };
                [f, 0]
            })
            .collect();
        let init_cdfs = cdfs.clone();
        let init_bools = bools.clone();
        let mut ops = Vec::with_capacity(count);
        for _ in 0..count {
            let op = match rng.below(10) {
                0 => Op::Equi(rng.below(2) == 1),
                1 => {
                    let f = if extreme {
                        [1u16, 64, 32767, 32704, 16384][rng.below(5) as usize]
                    } else {
                        1 + rng.below(32767) as u16
                    };
                    Op::Prob(f, rng.below(2) == 1)
                }
                2 | 3 => {
                    let i = rng.below(bools.len() as u64) as usize;
                    let p = u64::from(bools[i][0]);
                    // Mostly follow the model so the CDF moves.
                    let bit = if rng.below(4) == 0 {
                        rng.below(2) == 1
                    } else {
                        rng.below(32768) < p
                    };
                    Op::Adapt(i, bit)
                }
                4..=7 => {
                    let i = rng.below(cdfs.len() as u64) as usize;
                    let n = cdfs[i].len();
                    let s = if extreme && rng.below(2) == 0 {
                        [0, n - 1][rng.below(2) as usize]
                    } else {
                        rng.below(n as u64) as usize
                    };
                    Op::Sym(i, s)
                }
                8 => {
                    let n = rng.below(33) as u32;
                    let v = if n == 32 {
                        rng.next() as u32
                    } else {
                        (rng.next() as u32) & ((1u32 << n) - 1)
                    };
                    if rng.below(2) == 0 {
                        Op::Lit(n, v)
                    } else {
                        let m = 1 + rng.below(1000) as u32;
                        Op::Ns(m, rng.below(u64::from(m)) as u32)
                    }
                }
                _ => Op::Golomb(match rng.below(8) {
                    0 => u32::MAX,
                    1 => u32::MAX - 1,
                    _ => (rng.next() as u32) >> rng.below(32),
                }),
            };
            // Track the encoder's CDF state to build the ops (encode below).
            if let Op::Sym(i, s) = op {
                let mut e = MsacEnc::new(allow_update);
                enc_sym(&mut e, &mut cdfs[i], s);
            }
            if let Op::Adapt(i, bit) = op {
                let mut e = MsacEnc::new(allow_update);
                e.bool_adapt(&mut bools[i], bit);
            }
            ops.push(op);
        }

        let mut ecdfs = init_cdfs.clone();
        let mut ebools = init_bools.clone();
        let mut e = MsacEnc::new(allow_update);
        for op in &ops {
            match *op {
                Op::Equi(b) => e.bool_equi(b),
                Op::Prob(f, b) => e.bool_prob(u32::from(f), b),
                Op::Adapt(i, b) => e.bool_adapt(&mut ebools[i], b),
                Op::Sym(i, s) => enc_sym(&mut e, &mut ecdfs[i], s),
                Op::Lit(n, v) => e.literal(n, v),
                Op::Golomb(v) => e.golomb(v),
                Op::Ns(n, v) => e.ns(n, v),
            }
        }
        let tell = e.tell_bits();
        let data = e.finish();
        assert_eq!(data.len() as u64, tell.div_ceil(8));
        assert_eq!(ecdfs, cdfs);
        assert_eq!(ebools, bools);

        let mut dcdfs = init_cdfs.clone();
        let mut dbools = init_bools.clone();
        let mut d = Msac::new(&data, !allow_update);
        let mut spec = SpecDec::new(&data);
        for (k, op) in ops.iter().enumerate() {
            match *op {
                Op::Equi(b) => {
                    assert_eq!(d.bool_equi(), b, "op {k}");
                    if spec_check {
                        assert_eq!(spec.read(&[16384]), usize::from(b));
                    }
                }
                Op::Prob(f, b) => {
                    assert_eq!(d.bool_prob(u32::from(f)), b, "op {k}");
                    if spec_check {
                        assert_eq!(spec.read(&[f]), usize::from(b));
                    }
                }
                Op::Adapt(i, b) => {
                    let f = dbools[i][0];
                    assert_eq!(d.bool_adapt(&mut dbools[i]), b, "op {k}");
                    if spec_check {
                        assert_eq!(spec.read(&[f]), usize::from(b));
                    }
                }
                Op::Sym(i, s) => {
                    let before = dcdfs[i].clone();
                    assert_eq!(dec_sym(&mut d, &mut dcdfs[i]), s, "op {k} {op:?}");
                    if spec_check {
                        assert_eq!(spec.read(&before[..before.len() - 1]), s);
                    }
                }
                Op::Lit(n, v) => {
                    assert_eq!(d.literal(n), v, "op {k}");
                    if spec_check {
                        for i in (0..n).rev() {
                            assert_eq!(spec.read(&[16384]) as u32, (v >> i) & 1);
                        }
                    }
                }
                Op::Golomb(v) => {
                    let dg = d.golomb();
                    if spec_check {
                        let x = u64::from(v) + 1;
                        let len = (63 - x.leading_zeros()).min(32);
                        for _ in 0..len {
                            assert_eq!(spec.read(&[16384]), 0);
                        }
                        if len < 32 {
                            assert_eq!(spec.read(&[16384]), 1);
                            for i in (0..len).rev() {
                                assert_eq!(spec.read(&[16384]) as u64, (x >> i) & 1);
                            }
                        }
                    }
                    assert_eq!(dg, v, "op {k} {:?}", &ops[k.saturating_sub(6)..=k]);
                }
                Op::Ns(n, v) => {
                    assert_eq!(d.ns(n), v, "op {k}");
                    if spec_check {
                        let w = 32 - n.leading_zeros();
                        let m = (1u32 << w) - n;
                        let (bits, val) = if v < m { (w - 1, v) } else { (w, v + m) };
                        for i in (0..bits).rev() {
                            assert_eq!(spec.read(&[16384]) as u32, (val >> i) & 1);
                        }
                    }
                }
            }
        }
        assert_eq!(dcdfs, cdfs);
        assert_eq!(dbools, bools);
        if spec_check {
            assert!(spec.exit_ok(), "exit process padding (len {})", data.len());
        }
    }

    #[test]
    fn replay_reproduces_and_inserts() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for round in 0..20 {
            let mut e = MsacEnc::new(true);
            e.record();
            let mut plain = MsacEnc::new(true);
            let mut cdf4 = [32768u16 - 8192, 32768 - 16384, 32768 - 24576, 0, 0];
            let mut cdf4b = cdf4;
            let mut cdf2 = [16384u16, 0];
            let mut cdf2b = cdf2;
            let mut ops = Vec::new();
            for i in 0..2000 {
                if i % 97 == 5 {
                    e.mark(i as u32);
                }
                match rng.below(4) {
                    0 => {
                        let v = rng.below(5) as usize;
                        e.symbol(&mut cdf4, v);
                        plain.symbol(&mut cdf4b, v);
                        ops.push((0, v as u32));
                    }
                    1 => {
                        let v = rng.below(2) == 1;
                        e.bool_adapt(&mut cdf2, v);
                        plain.bool_adapt(&mut cdf2b, v);
                        ops.push((1, u32::from(v)));
                    }
                    2 => {
                        let v = rng.below(8) as u32;
                        e.literal(3, v);
                        plain.literal(3, v);
                        ops.push((2, v));
                    }
                    _ => {
                        let v = rng.below(2) == 1;
                        e.bool_prob(30000, v);
                        plain.bool_prob(30000, v);
                        ops.push((3, u32::from(v)));
                    }
                }
            }
            let reference = plain.finish();
            let bits = round % 4;
            let ins = move |tag: u32| (bits, tag & ((1 << bits) - 1));
            let data = e.replay(&ins);
            if bits == 0 {
                assert_eq!(data, reference);
            }
            let mut d = Msac::new(&data, false);
            let mut cdf4 = [32768u16 - 8192, 32768 - 16384, 32768 - 24576, 0, 0];
            let mut cdf2 = [16384u16, 0];
            for (i, &(kind, v)) in ops.iter().enumerate() {
                if i % 97 == 5 {
                    assert_eq!(d.literal(bits), ins(i as u32).1);
                }
                let got = match kind {
                    0 => d.symbol(&mut cdf4) as u32,
                    1 => u32::from(d.bool_adapt(&mut cdf2)),
                    2 => d.literal(3),
                    _ => u32::from(d.bool_prob(30000)),
                };
                assert_eq!(got, v);
            }
        }
    }

    #[test]
    fn mixed_round_trip() {
        for seed in 1..200u64 {
            let count = (seed as usize * 37) % 3000;
            run(seed * 7919, count, seed % 2 == 0, seed % 3 == 0, true);
        }
    }

    #[test]
    fn long_streams() {
        run(0xdead_beef, 100_000, true, false, false);
        run(0xfeed_f00d, 100_000, false, false, false);
        run(0x1234_5678, 100_000, true, true, false);
    }

    /// Streams without golomb/ns checked against a spec 8.2 decoder,
    /// including the exit-process trailing bit and padding.
    #[test]
    fn spec_decoder_and_exit_padding() {
        for seed in 1..300u64 {
            let mut rng = Rng(seed * 104_729);
            let allow = seed % 2 == 0;
            let mut cdfs = real_cdfs(60);
            for n in 2..=16 {
                cdfs.push(random_cdf(&mut rng, n, seed % 3 == 0));
            }
            let init = cdfs.clone();
            let count = rng.below(if seed % 10 == 0 { 20_000 } else { 60 }) as usize;
            let mut ops = Vec::new();
            let mut e = MsacEnc::new(allow);
            for _ in 0..count {
                let op = match rng.below(4) {
                    0 => Op::Equi(rng.below(2) == 1),
                    1 => Op::Prob(1 + rng.below(32767) as u16, rng.below(2) == 1),
                    2 => Op::Lit(rng.below(9) as u32, rng.below(256) as u32),
                    _ => {
                        let i = rng.below(cdfs.len() as u64) as usize;
                        Op::Sym(i, rng.below(cdfs[i].len() as u64) as usize)
                    }
                };
                match op {
                    Op::Equi(b) => e.bool_equi(b),
                    Op::Prob(f, b) => e.bool_prob(u32::from(f), b),
                    Op::Lit(n, v) => e.literal(n, v & ((1 << n) - 1)),
                    Op::Sym(i, s) => enc_sym(&mut e, &mut cdfs[i], s),
                    _ => unreachable!(),
                }
                ops.push(match op {
                    Op::Lit(n, v) => Op::Lit(n, v & ((1 << n) - 1)),
                    o => o,
                });
            }
            let data = e.finish();
            let mut spec = SpecDec::new(&data);
            let mut dcdfs = init.clone();
            let mut d = Msac::new(&data, !allow);
            for op in &ops {
                match *op {
                    Op::Equi(b) => {
                        assert_eq!(d.bool_equi(), b);
                        assert_eq!(spec.read(&[16384]), usize::from(b));
                    }
                    Op::Prob(f, b) => {
                        assert_eq!(d.bool_prob(u32::from(f)), b);
                        assert_eq!(spec.read(&[f]), usize::from(b));
                    }
                    Op::Lit(n, v) => {
                        assert_eq!(d.literal(n), v);
                        for i in (0..n).rev() {
                            assert_eq!(spec.read(&[16384]) as u32, (v >> i) & 1);
                        }
                    }
                    Op::Sym(i, s) => {
                        let before = dcdfs[i].clone();
                        assert_eq!(dec_sym(&mut d, &mut dcdfs[i]), s);
                        assert_eq!(spec.read(&before[..before.len() - 1]), s);
                    }
                    _ => unreachable!(),
                }
            }
            assert_eq!(dcdfs, cdfs);
            assert!(spec.exit_ok(), "seed {seed}: exit padding");
        }
        // The empty stream is one byte holding the trailing bit.
        assert_eq!(MsacEnc::new(true).finish(), vec![0x80]);
    }

    #[test]
    fn carry_heavy_streams() {
        // Symbols at the top of the range produce long runs of ones in low.
        for allow in [false, true] {
            let mut e = MsacEnc::new(allow);
            for i in 0..50_000 {
                e.bool_prob(1, i % 97 != 0);
            }
            let data = e.finish();
            let mut d = Msac::new(&data, !allow);
            for i in 0..50_000 {
                assert_eq!(d.bool_prob(1), i % 97 != 0, "i {i}");
            }
        }
        run(42, 30_000, true, true, true);
    }

    #[test]
    fn single_symbol_each_cdf() {
        let mut rng = Rng(5);
        let mut cdfs = real_cdfs(60);
        for n in 2..=16 {
            cdfs.push(random_cdf(&mut rng, n, false));
        }
        for c in &cdfs {
            for s in 0..c.len() {
                for pre in 0..3 {
                    let mut e = MsacEnc::new(false);
                    for _ in 0..pre {
                        e.bool_equi(true);
                    }
                    let mut cc = c.clone();
                    enc_sym(&mut e, &mut cc, s);
                    e.literal(20, 0xabcde);
                    let data = e.finish();
                    let mut d = Msac::new(&data, true);
                    for _ in 0..pre {
                        d.bool_equi();
                    }
                    let mut cc = c.clone();
                    assert_eq!(dec_sym(&mut d, &mut cc), s, "{c:?} s {s}");
                    assert_eq!(d.literal(20), 0xabcde, "{c:?} s {s}");
                }
            }
        }
    }

    #[test]
    fn costs() {
        assert_eq!(prob_cost(16384), 256);
        assert_eq!(prob_cost(32768), 0);
        assert_eq!(prob_cost(1), 15 * 256);
        assert_eq!(bool_cost(16384, true), 256);
        assert_eq!(bool_cost(8192, true), 512);
        // 3/4 probability: 0.415 bits.
        assert!((bool_cost(8192, false) as i32 - 106).abs() <= 1);
        let cdf = [24576u16, 16384, 8192, 0];
        for s in 0..4 {
            assert_eq!(symbol_cost(&cdf, s), 512);
        }
        for p in 1..=32768u32 {
            let exact = -((f64::from(p) / 32768.0).log2()) * 256.0;
            assert!((f64::from(prob_cost(p)) - exact).abs() < 1.5, "p {p}");
        }
    }
}
