//! Symbol decoder (spec 8.2).
//!
//! The window keeps the complement of the coded bits, as dav1d does, and the
//! CDFs are stored inverted: `icdf[i] = 32768 - cdf[i]` for the first N-1
//! entries, with the adaptation counter in the last slot. A CDF for N symbols
//! is therefore an array of N `u16`.

const WIN: u32 = 64;
const PROB_SHIFT: u32 = 6;
const MIN_PROB: u32 = 4;

pub struct Msac<'a> {
    buf: &'a [u8],
    pos: usize,
    dif: u64,
    rng: u32,
    cnt: i32,
    pub allow_update: bool,
}

impl<'a> Msac<'a> {
    pub fn new(data: &'a [u8], disable_cdf_update: bool) -> Self {
        let mut s = Self {
            buf: data,
            pos: 0,
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update: !disable_cdf_update,
        };
        s.refill();
        s
    }

    #[inline]
    fn refill(&mut self) {
        let mut c = WIN as i32 - self.cnt - 24;
        let mut dif = self.dif;
        loop {
            if self.pos >= self.buf.len() {
                dif |= !(!0xffu64 << c);
                break;
            }
            dif |= u64::from(self.buf[self.pos] ^ 0xff) << c;
            self.pos += 1;
            c -= 8;
            if c < 0 {
                break;
            }
        }
        self.dif = dif;
        self.cnt = WIN as i32 - c - 24;
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
    #[inline]
    pub fn bool_equi(&mut self) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = ((r >> 8) << 7) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = dif >= vw;
        if ret {
            dif -= vw;
            v = r - v;
        }
        self.norm(dif, v);
        !ret
    }

    /// Boolean with a fixed inverted probability `f` (no adaptation).
    #[inline]
    pub fn bool_prob(&mut self, f: u32) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = (((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB;
        let vw = u64::from(v) << (WIN - 16);
        let ret = dif >= vw;
        if ret {
            dif -= vw;
            v = r - v;
        }
        self.norm(dif, v);
        !ret
    }

    /// Adaptive boolean on a 2-entry inverted CDF.
    #[inline]
    pub fn bool_adapt(&mut self, cdf: &mut [u16; 2]) -> bool {
        let bit = self.bool_prob(u32::from(cdf[0]));
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
        bit
    }

    /// Adaptive symbol over an N-entry inverted CDF (N symbols).
    #[inline]
    pub fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        let n = N - 1; // number of probability entries
        let c = (self.dif >> (WIN - 16)) as u32;
        let r = self.rng >> 8;
        // v[i] for every entry; the decoded symbol is the number of entries
        // whose v is above c (v decreases with i).
        let mut v = [0u32; N];
        let mut val = 0usize;
        for i in 0..n {
            v[i] = ((r * (u32::from(cdf[i]) >> PROB_SHIFT)) >> (7 - PROB_SHIFT))
                + MIN_PROB * (n - i) as u32;
            val += usize::from(c < v[i]);
        }
        let u = if val == 0 { self.rng } else { v[val - 1] };
        let vv = if val >= n { 0 } else { v[val] };
        self.norm(self.dif - (u64::from(vv) << (WIN - 16)), u - vv);
        if self.allow_update {
            let count = cdf[n];
            let rate = 4 + (count >> 4) + u16::from(n > 2);
            for i in 0..n {
                let x = cdf[i];
                cdf[i] = if i < val {
                    x + ((32768 - x) >> rate)
                } else {
                    x - (x >> rate)
                };
            }
            cdf[n] = count + u16::from(count < 32);
        }
        val
    }

    /// L(n): n raw bits, MSB first.
    #[inline]
    pub fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | u32::from(self.bool_equi());
        }
        x
    }

    /// read_golomb()
    pub fn golomb(&mut self) -> u32 {
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
