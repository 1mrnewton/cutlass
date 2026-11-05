//! MSB-first bit writer for OBU headers (spec 4.10, the inverse of
//! `dec::bits::BitReader`), leb128 and the OBU wrapper (spec 5.3).

/// MSB-first bit writer.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bits used in the last byte of `buf` (0 means byte aligned).
    used: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bits written so far.
    pub fn position(&self) -> usize {
        if self.used == 0 {
            self.buf.len() * 8
        } else {
            (self.buf.len() - 1) * 8 + self.used as usize
        }
    }

    pub fn is_aligned(&self) -> bool {
        self.used == 0
    }

    #[inline]
    pub fn bit(&mut self, b: bool) {
        if self.used == 0 {
            self.buf.push(0);
        }
        if let Some(last) = self.buf.last_mut() {
            *last |= u8::from(b) << (7 - self.used);
        }
        self.used = (self.used + 1) & 7;
    }

    pub fn flag(&mut self, b: bool) {
        self.bit(b);
    }

    /// f(n), n <= 32: the low n bits of `v`, MSB first.
    pub fn f(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 != 0);
        }
    }

    /// su(n): `v` as an n-bit two's complement value.
    pub fn su(&mut self, n: u32, v: i32) {
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.f(n, (v as u32) & mask);
    }

    /// ns(n): `v` in 0..n with the non-symmetric code.
    pub fn ns(&mut self, n: u32, v: u32) {
        if n <= 1 {
            return;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        if v < m {
            self.f(w - 1, v);
        } else {
            let x = v + m;
            self.f(w - 1, x >> 1);
            self.f(1, x & 1);
        }
    }

    /// le(n): n bytes little endian.
    pub fn le(&mut self, n: u32, v: u32) {
        for i in 0..n {
            self.f(8, (v >> (8 * i)) & 0xff);
        }
    }

    /// uvlc(); `u32::MAX` is written as 32 zeros.
    pub fn uvlc(&mut self, v: u32) {
        if v == u32::MAX {
            // The reader stops after 32 leading zeros.
            self.f(32, 0);
            return;
        }
        let x = u64::from(v) + 1;
        let leading = 63 - x.leading_zeros(); // FloorLog2(v + 1)
        self.f(leading, 0);
        self.bit(true);
        self.f(leading, (x - (1u64 << leading)) as u32);
    }

    /// byte_alignment(): zero bits up to the next byte boundary.
    pub fn byte_alignment(&mut self) {
        self.used = 0;
    }

    /// trailing_bits(): a one bit, then zero bits to the byte boundary.
    pub fn trailing_bits(&mut self) {
        self.bit(true);
        self.byte_alignment();
    }

    /// Appends whole bytes (the writer must be byte aligned).
    pub fn bytes_raw(&mut self, data: &[u8]) {
        self.used = 0;
        self.buf.extend_from_slice(data);
    }

    /// The bytes written; a partial last byte is zero padded.
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Number of bytes `write_leb128` uses for `v`.
pub fn leb128_size(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// leb128() in the minimal number of bytes.
pub fn write_leb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// leb128() padded to exactly `n` bytes (1..=8); `v` must fit in 7n bits.
pub fn write_leb128_fixed(out: &mut Vec<u8>, v: u64, n: usize) {
    for i in 0..n {
        let b = ((v >> (7 * i)) & 0x7f) as u8;
        out.push(if i + 1 < n { b | 0x80 } else { b });
    }
}

/// obu_header() with obu_has_size_field = 1, obu_size, then the payload.
pub fn write_obu(
    out: &mut Vec<u8>,
    obu_type: u8,
    temporal_id: u8,
    spatial_id: u8,
    with_extension: bool,
    payload: &[u8],
) {
    out.push(((obu_type & 15) << 3) | (u8::from(with_extension) << 2) | 2);
    if with_extension {
        out.push(((temporal_id & 7) << 5) | ((spatial_id & 3) << 3));
    }
    write_leb128(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::bits::{BitReader, leb128};
    use crate::dec::obu::split_obus;

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

    #[derive(Debug)]
    enum Op {
        F(u32, u32),
        Su(u32, i32),
        Ns(u32, u32),
        Le(u32, u32),
        Uvlc(u32),
        Align,
    }

    #[test]
    fn round_trip_all_codings() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..200 {
            let mut ops = Vec::new();
            for _ in 0..200 {
                let op = match rng.below(6) {
                    0 => {
                        let n = rng.below(33) as u32;
                        let v = if n == 32 {
                            rng.next() as u32
                        } else {
                            (rng.next() as u32) & ((1u32 << n) - 1)
                        };
                        Op::F(n, v)
                    }
                    1 => {
                        let n = 1 + rng.below(31) as u32;
                        let half = 1i64 << (n - 1);
                        let v = (rng.below(2 * half as u64) as i64 - half) as i32;
                        Op::Su(n, v)
                    }
                    2 => {
                        let lim = if rng.below(2) == 0 { 20 } else { 100_000 };
                        let n = 1 + rng.below(lim) as u32;
                        Op::Ns(n, rng.below(u64::from(n)) as u32)
                    }
                    3 => {
                        let n = rng.below(5) as u32;
                        let v = if n == 4 {
                            rng.next() as u32
                        } else {
                            (rng.next() as u32) & ((1u32 << (8 * n)) - 1)
                        };
                        Op::Le(n, v)
                    }
                    4 => {
                        let v = match rng.below(4) {
                            0 => rng.below(4) as u32,
                            1 => u32::MAX - rng.below(3) as u32,
                            _ => (rng.next() as u32) >> rng.below(32),
                        };
                        Op::Uvlc(v)
                    }
                    _ => Op::Align,
                };
                ops.push(op);
            }
            let mut w = BitWriter::new();
            for op in &ops {
                match *op {
                    Op::F(n, v) => w.f(n, v),
                    Op::Su(n, v) => w.su(n, v),
                    Op::Ns(n, v) => w.ns(n, v),
                    Op::Le(n, v) => w.le(n, v),
                    Op::Uvlc(v) => w.uvlc(v),
                    Op::Align => w.byte_alignment(),
                }
            }
            let total = w.position();
            let data = w.into_bytes();
            let mut r = BitReader::new(&data);
            for op in &ops {
                match *op {
                    Op::F(n, v) => assert_eq!(r.f(n).unwrap(), v, "{op:?}"),
                    Op::Su(n, v) => assert_eq!(r.su(n).unwrap(), v, "{op:?}"),
                    Op::Ns(n, v) => assert_eq!(r.ns(n).unwrap(), v, "{op:?}"),
                    Op::Le(n, v) => assert_eq!(r.le(n).unwrap(), v, "{op:?}"),
                    Op::Uvlc(v) => assert_eq!(r.uvlc().unwrap(), v, "{op:?}"),
                    Op::Align => r.byte_alignment(),
                }
            }
            assert_eq!(r.position(), total);
        }
    }

    #[test]
    fn trailing_bits_and_position() {
        let mut w = BitWriter::new();
        w.f(3, 5);
        assert_eq!(w.position(), 3);
        w.trailing_bits();
        assert_eq!(w.bytes(), &[0b1011_0000]);
        w.trailing_bits();
        assert_eq!(w.bytes(), &[0b1011_0000, 0x80]);
        assert!(w.is_aligned());
    }

    #[test]
    fn leb128_round_trip() {
        let mut rng = Rng(77);
        let mut vals: Vec<u64> = vec![0, 1, 127, 128, 16383, 16384, (1 << 56) - 1];
        for _ in 0..10_000 {
            vals.push(rng.next() >> (8 + rng.below(56)));
        }
        for v in vals {
            let mut out = Vec::new();
            write_leb128(&mut out, v);
            assert_eq!(out.len(), leb128_size(v));
            assert_eq!(leb128(&out).unwrap(), (v, out.len()));
            let n = leb128_size(v) + rng.below(3) as usize;
            if n <= 8 {
                let mut out = Vec::new();
                write_leb128_fixed(&mut out, v, n);
                assert_eq!(leb128(&out).unwrap(), (v, n));
            }
        }
    }

    #[test]
    fn obu_wrapper() {
        let mut out = Vec::new();
        write_obu(&mut out, 2, 0, 0, false, &[]);
        let big: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        write_obu(&mut out, 6, 5, 2, true, &big);
        write_obu(&mut out, 1, 0, 0, false, &[1, 2, 3]);
        let obus = split_obus(&out).unwrap();
        assert_eq!(obus.len(), 3);
        assert_eq!(obus[0].obu_type, 2);
        assert!(obus[0].payload.is_empty());
        assert_eq!(obus[1].obu_type, 6);
        assert!(obus[1].has_extension);
        assert_eq!((obus[1].temporal_id, obus[1].spatial_id), (5, 2));
        assert_eq!(obus[1].payload, &big[..]);
        assert_eq!(obus[2].payload, &[1, 2, 3]);
    }
}
