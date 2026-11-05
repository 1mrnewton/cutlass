//! The Opus range coder.
//!
//! RFC 6716 section 4.1 defines the decoder and section 5.1 the encoder.
//! Symbols are coded MSB-first from the front of the buffer. Raw bits
//! (`dec_bits`/`enc_bits`) are packed LSB-first from the back of the same
//! buffer. Both ends meet in the middle; a decoder that reads past either
//! end sees zero bytes, as the RFC requires, and the frame is still decoded.
//!
//! The state after the last symbol, `rng`, is the "final range" that the
//! official test vectors record for every packet.

/// Bits per output symbol (one byte).
const SYM_BITS: u32 = 8;
/// Bits in the coder state.
const CODE_BITS: u32 = 32;
const SYM_MAX: u32 = (1 << SYM_BITS) - 1;
const CODE_SHIFT: u32 = CODE_BITS - SYM_BITS - 1;
const CODE_TOP: u32 = 1 << (CODE_BITS - 1);
const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;
/// Bits held in the raw-bit window.
const WINDOW_SIZE: u32 = 32;
/// Bits a uniform integer codes with the range coder before raw bits.
const UINT_BITS: u32 = 8;
/// Fractional bits in [`RangeDecoder::tell_frac`].
pub(crate) const BITRES: u32 = 3;

/// Number of bits needed to hold `x`: 0 for 0, 1 for 1, 2 for 2..=3, and so on.
#[inline]
pub(crate) fn ilog(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// Range decoder over one frame's bytes.
#[derive(Clone, Debug)]
pub(crate) struct RangeDecoder<'a> {
    buf: &'a [u8],
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: u32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: u32,
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        let storage = u32::try_from(buf.len()).unwrap_or(u32::MAX);
        let mut dec = Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1
                - ((CODE_BITS - CODE_EXTRA) / SYM_BITS) * SYM_BITS) as i32,
            offs: 0,
            rng: 1 << CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: false,
        };
        dec.rem = dec.read_byte();
        dec.val = dec.rng - 1 - (dec.rem >> (SYM_BITS - CODE_EXTRA));
        dec.normalize();
        dec
    }

    /// Bytes of the buffer this decoder was given.
    pub(crate) fn storage(&self) -> u32 {
        self.storage
    }

    fn read_byte(&mut self) -> u32 {
        if self.offs < self.storage {
            let byte = self.buf.get(self.offs as usize).copied().unwrap_or(0);
            self.offs += 1;
            u32::from(byte)
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            let at = (self.storage - self.end_offs) as usize;
            u32::from(self.buf.get(at).copied().unwrap_or(0))
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS as i32;
            self.rng <<= SYM_BITS;
            let mut sym = self.rem;
            self.rem = self.read_byte();
            sym = ((sym << SYM_BITS) | self.rem) >> (SYM_BITS - CODE_EXTRA);
            self.val = ((self.val << SYM_BITS).wrapping_add(SYM_MAX & !sym)) & (CODE_TOP - 1);
        }
    }

    /// The cumulative frequency of the next symbol, for a total of `ft`.
    /// The caller must follow with [`Self::update`].
    pub(crate) fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// [`Self::decode`] for a power-of-two total `1 << bits`.
    pub(crate) fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    /// Consumes the symbol whose range is `fl..fh` out of `ft`.
    pub(crate) fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.normalize();
    }

    /// One bit whose probability of being 1 is `1 / (1 << logp)`.
    pub(crate) fn bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// One symbol from an inverse cumulative table with `1 << ftb` total.
    ///
    /// The table is decreasing and ends at 0. A table that does not end in
    /// zero is a programming error and returns its last index.
    pub(crate) fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut index = 0usize;
        let mut t;
        loop {
            t = s;
            let entry = u32::from(icdf.get(index).copied().unwrap_or(0));
            s = r.wrapping_mul(entry);
            if d >= s || index + 1 >= icdf.len() {
                break;
            }
            index += 1;
        }
        self.val = d.wrapping_sub(s);
        self.rng = t.wrapping_sub(s);
        self.normalize();
        index
    }

    /// A uniform integer in `0..ft`. `ft` is at least 2.
    ///
    /// A value past `ft - 1` in the raw bits sets the error flag and
    /// returns `ft - 1`, as the RFC specifies.
    pub(crate) fn dec_uint(&mut self, ft: u32) -> u32 {
        if ft <= 1 {
            return 0;
        }
        let ft_minus = ft - 1;
        let mut ftb = ilog(ft_minus);
        if ftb > UINT_BITS {
            ftb -= UINT_BITS;
            let ft1 = (ft_minus >> ftb) + 1;
            let s = self.decode(ft1);
            self.update(s, s + 1, ft1);
            let t = (s << ftb) | self.dec_bits(ftb);
            if t <= ft_minus {
                return t;
            }
            self.error = true;
            ft_minus
        } else {
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            s
        }
    }

    /// `bits` raw bits (at most 25) from the end of the buffer.
    pub(crate) fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if available < bits {
            loop {
                window |= self.read_byte_from_end() << available;
                available += SYM_BITS;
                if available > WINDOW_SIZE - SYM_BITS {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits).wrapping_sub(1));
        window = if bits >= 32 { 0 } else { window >> bits };
        available -= bits;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }

    /// Whole bits used so far, rounded up.
    pub(crate) fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// Bits used so far in 1/8 bit units, rounded up.
    pub(crate) fn tell_frac(&self) -> u32 {
        tell_frac(self.nbits_total, self.rng)
    }

    pub(crate) fn rng(&self) -> u32 {
        self.rng
    }

    /// Whether a uniform integer was out of range.
    pub(crate) fn error(&self) -> bool {
        self.error
    }

    /// Shrinks the buffer the raw bits are read from, as hybrid redundancy
    /// does when the redundant frame takes the end of the packet.
    pub(crate) fn shrink_storage(&mut self, by: u32) {
        self.storage = self.storage.saturating_sub(by);
    }
}

fn tell_frac(nbits_total: i32, rng: u32) -> u32 {
    let nbits = (nbits_total as u32) << BITRES;
    let mut l = ilog(rng);
    let mut r = rng >> (l - 16);
    for _ in 0..BITRES {
        r = (r * r) >> 15;
        let b = r >> 16;
        l = (l << 1) | b;
        r >>= b;
    }
    nbits.wrapping_sub(l)
}

/// Range encoder writing into a fixed-size buffer.
#[derive(Clone, Debug)]
pub(crate) struct RangeEncoder {
    buf: Vec<u8>,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: u32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: bool,
}

impl RangeEncoder {
    /// An encoder that writes at most `size` bytes.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            buf: vec![0; size],
            storage: u32::try_from(size).unwrap_or(u32::MAX),
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1) as i32,
            offs: 0,
            rng: CODE_TOP,
            val: 0,
            ext: 0,
            rem: -1,
            error: false,
        }
    }

    fn write_byte(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        if let Some(slot) = self.buf.get_mut(self.offs as usize) {
            *slot = value as u8;
        }
        self.offs += 1;
    }

    fn write_byte_at_end(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        self.end_offs += 1;
        let at = (self.storage - self.end_offs) as usize;
        if let Some(slot) = self.buf.get_mut(at) {
            *slot = value as u8;
        }
    }

    fn carry_out(&mut self, c: u32) {
        if c != SYM_MAX {
            let carry = c >> SYM_BITS;
            if self.rem >= 0 {
                self.write_byte((self.rem as u32).wrapping_add(carry));
            }
            if self.ext > 0 {
                let sym = (SYM_MAX + carry) & SYM_MAX;
                loop {
                    self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = (c & SYM_MAX) as i32;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.carry_out(self.val >> CODE_SHIFT);
            self.val = (self.val << SYM_BITS) & (CODE_TOP - 1);
            self.rng <<= SYM_BITS;
            self.nbits_total += SYM_BITS as i32;
        }
    }

    /// Codes the symbol whose range is `fl..fh` out of `ft`.
    pub(crate) fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.normalize();
    }

    /// [`Self::encode`] with a power-of-two total `1 << bits`.
    pub(crate) fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul((1 << bits) - fh));
        }
        self.normalize();
    }

    /// One bit whose probability of being 1 is `1 / (1 << logp)`.
    pub(crate) fn bit_logp(&mut self, value: bool, logp: u32) {
        let mut r = self.rng;
        let l = self.val;
        let s = r >> logp;
        r -= s;
        if value {
            self.val = l.wrapping_add(r);
        }
        self.rng = if value { s } else { r };
        self.normalize();
    }

    /// Symbol `s` of an inverse cumulative table with `1 << ftb` total.
    pub(crate) fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        let at = |index: usize| u32::from(icdf.get(index).copied().unwrap_or(0));
        if s > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(at(s - 1))));
            self.rng = r.wrapping_mul(at(s - 1).wrapping_sub(at(s)));
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(at(s)));
        }
        self.normalize();
    }

    /// A uniform integer `fl` in `0..ft`. `ft` is at least 2.
    pub(crate) fn enc_uint(&mut self, fl: u32, ft: u32) {
        if ft <= 1 {
            return;
        }
        let ft_minus = ft - 1;
        let mut ftb = ilog(ft_minus);
        if ftb > UINT_BITS {
            ftb -= UINT_BITS;
            let ft1 = (ft_minus >> ftb) + 1;
            self.encode(fl >> ftb, (fl >> ftb) + 1, ft1);
            self.enc_bits(fl & ((1 << ftb) - 1), ftb);
        } else {
            self.encode(fl, fl + 1, ft_minus + 1);
        }
    }

    /// `bits` raw bits (at most 25) at the end of the buffer.
    pub(crate) fn enc_bits(&mut self, fl: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used + bits > WINDOW_SIZE {
            loop {
                self.write_byte_at_end(window & SYM_MAX);
                window >>= SYM_BITS;
                used -= SYM_BITS;
                if used < SYM_BITS {
                    break;
                }
            }
        }
        window |= fl << used;
        used += bits;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Overwrites the first `nbits` bits already coded (at most 8).
    pub(crate) fn patch_initial_bits(&mut self, value: u32, nbits: u32) {
        let shift = SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            if let Some(first) = self.buf.first_mut() {
                *first = ((u32::from(*first) & !mask) | (value << shift)) as u8;
            }
        } else if self.rem >= 0 {
            self.rem = (((self.rem as u32) & !mask) | (value << shift)) as i32;
        } else if self.rng <= (CODE_TOP >> nbits) {
            self.val = (self.val & !(mask << CODE_SHIFT)) | (value << (CODE_SHIFT + shift));
        } else {
            self.error = true;
        }
    }

    /// Moves the raw bits so the frame ends at `size` bytes.
    pub(crate) fn shrink(&mut self, size: u32) {
        if size >= self.storage || self.offs + self.end_offs > size {
            return;
        }
        let end = self.end_offs as usize;
        let from = (self.storage - self.end_offs) as usize;
        let to = (size - self.end_offs) as usize;
        self.buf.copy_within(from..from + end, to);
        self.storage = size;
        self.buf.truncate(size as usize);
    }

    /// Whole bits used so far, rounded up.
    pub(crate) fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// Bits used so far in 1/8 bit units, rounded up.
    pub(crate) fn tell_frac(&self) -> u32 {
        tell_frac(self.nbits_total, self.rng)
    }

    pub(crate) fn rng(&self) -> u32 {
        self.rng
    }

    /// Bytes this frame may use.
    pub(crate) fn storage(&self) -> u32 {
        self.storage
    }

    /// Bytes written from the front so far.
    pub(crate) fn range_bytes(&self) -> u32 {
        self.offs
    }

    pub(crate) fn error(&self) -> bool {
        self.error
    }

    /// Flushes the coder and returns the `storage`-byte frame.
    ///
    /// The bytes between the range-coded front and the raw-bit back are
    /// zero. `error` is set when the frame did not fit.
    pub(crate) fn done(&mut self) {
        let mut l = CODE_BITS as i32 - ilog(self.rng) as i32;
        let mut msk = (CODE_TOP - 1) >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out(end >> CODE_SHIFT);
            end = (end << SYM_BITS) & (CODE_TOP - 1);
            l -= SYM_BITS as i32;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= SYM_BITS {
            self.write_byte_at_end(window & SYM_MAX);
            window >>= SYM_BITS;
            used -= SYM_BITS;
        }
        if !self.error {
            let start = self.offs as usize;
            let stop = (self.storage - self.end_offs) as usize;
            if let Some(gap) = self.buf.get_mut(start..stop) {
                gap.fill(0);
            }
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = true;
                } else {
                    let room = (-l) as u32;
                    if self.offs + self.end_offs >= self.storage && room < used {
                        window &= (1 << room) - 1;
                        self.error = true;
                    }
                    let at = (self.storage - self.end_offs - 1) as usize;
                    if let Some(slot) = self.buf.get_mut(at) {
                        *slot |= window as u8;
                    }
                }
            }
        }
    }

    /// The coded frame. Call after [`Self::done`].
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.buf[..(self.storage as usize).min(self.buf.len())]
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.buf.truncate(self.storage as usize);
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::{RangeDecoder, RangeEncoder, ilog};

    /// A deterministic sequence, so the tests need no random crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn ilog_counts_bits() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
        assert_eq!(ilog(u32::MAX), 32);
    }

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Uint(u32, u32),
        Bits(u32, u32),
        Logp(bool, u32),
        Icdf(usize),
        Freq(u32, u32, u32),
    }

    const ICDF: [u8; 5] = [200, 120, 60, 20, 0];

    fn encode(ops: &[Op], size: usize) -> (Vec<u8>, u32, i32) {
        let mut enc = RangeEncoder::new(size);
        for op in ops {
            match *op {
                Op::Uint(value, total) => enc.enc_uint(value, total),
                Op::Bits(value, bits) => enc.enc_bits(value, bits),
                Op::Logp(value, logp) => enc.bit_logp(value, logp),
                Op::Icdf(symbol) => enc.enc_icdf(symbol, &ICDF, 8),
                Op::Freq(fl, fh, ft) => enc.encode(fl, fh, ft),
            }
        }
        let tell = enc.tell();
        enc.done();
        assert!(!enc.error(), "frame of {size} bytes overflowed");
        let rng = enc.rng();
        (enc.into_bytes(), rng, tell)
    }

    fn random_ops(seed: u64, count: usize) -> Vec<Op> {
        let mut rand = Lcg(seed);
        (0..count)
            .map(|_| match rand.next() % 5 {
                0 => {
                    let total = 2 + rand.next() % 70_000;
                    Op::Uint(rand.next() % total, total)
                }
                1 => {
                    let bits = 1 + rand.next() % 24;
                    Op::Bits(rand.next() & ((1 << bits) - 1), bits)
                }
                2 => Op::Logp(rand.next() % 3 == 0, 1 + rand.next() % 15),
                3 => Op::Icdf((rand.next() % 5) as usize),
                _ => {
                    let total = 2 + rand.next() % 1000;
                    let fl = rand.next() % total;
                    let fh = fl + 1 + rand.next() % (total - fl);
                    Op::Freq(fl, fh, total)
                }
            })
            .collect()
    }

    #[test]
    fn every_symbol_kind_round_trips_with_the_same_final_range() {
        for seed in 0..200 {
            let ops = random_ops(seed, 300);
            let (bytes, enc_rng, enc_tell) = encode(&ops, 4000);
            let mut dec = RangeDecoder::new(&bytes);
            for op in &ops {
                match *op {
                    Op::Uint(value, total) => assert_eq!(dec.dec_uint(total), value),
                    Op::Bits(value, bits) => assert_eq!(dec.dec_bits(bits), value),
                    Op::Logp(value, logp) => assert_eq!(dec.bit_logp(logp), value),
                    Op::Icdf(symbol) => assert_eq!(dec.dec_icdf(&ICDF, 8), symbol),
                    Op::Freq(fl, fh, total) => {
                        let s = dec.decode(total);
                        assert!(fl <= s && s < fh, "seed {seed}");
                        dec.update(fl, fh, total);
                    }
                }
            }
            assert_eq!(dec.rng(), enc_rng, "seed {seed}");
            assert_eq!(dec.tell(), enc_tell, "seed {seed}");
            assert!(!dec.error());
        }
    }

    #[test]
    fn a_tight_frame_still_decodes() {
        let ops = random_ops(7, 40);
        let (loose, _, tell) = encode(&ops, 4000);
        let bytes = (tell as usize).div_ceil(8);
        let mut enc = RangeEncoder::new(4000);
        for op in &ops {
            match *op {
                Op::Uint(value, total) => enc.enc_uint(value, total),
                Op::Bits(value, bits) => enc.enc_bits(value, bits),
                Op::Logp(value, logp) => enc.bit_logp(value, logp),
                Op::Icdf(symbol) => enc.enc_icdf(symbol, &ICDF, 8),
                Op::Freq(fl, fh, ft) => enc.encode(fl, fh, ft),
            }
        }
        enc.shrink(bytes as u32);
        enc.done();
        assert!(!enc.error());
        let tight = enc.into_bytes();
        assert_eq!(tight.len(), bytes);
        assert!(tight.len() < loose.len());
        let mut dec = RangeDecoder::new(&tight);
        for op in &ops {
            match *op {
                Op::Uint(value, total) => assert_eq!(dec.dec_uint(total), value),
                Op::Bits(value, bits) => assert_eq!(dec.dec_bits(bits), value),
                Op::Logp(value, logp) => assert_eq!(dec.bit_logp(logp), value),
                Op::Icdf(symbol) => assert_eq!(dec.dec_icdf(&ICDF, 8), symbol),
                Op::Freq(fl, fh, total) => {
                    let s = dec.decode(total);
                    assert!(fl <= s && s < fh);
                    dec.update(fl, fh, total);
                }
            }
        }
    }

    #[test]
    fn reading_past_the_end_sees_zeros_and_does_not_panic() {
        let mut dec = RangeDecoder::new(&[]);
        for _ in 0..100 {
            let _ = dec.dec_uint(1_000_000);
            let _ = dec.dec_bits(17);
            let _ = dec.bit_logp(3);
            let _ = dec.dec_icdf(&ICDF, 8);
        }
        assert!(dec.tell() > 0);
    }

    #[test]
    fn an_overflowing_frame_reports_an_error() {
        let ops = random_ops(3, 300);
        let mut enc = RangeEncoder::new(8);
        for op in &ops {
            if let Op::Bits(value, bits) = *op {
                enc.enc_bits(value, bits);
            }
        }
        enc.done();
        assert!(enc.error());
    }

    #[test]
    fn patching_the_first_bits_changes_only_them() {
        let mut enc = RangeEncoder::new(100);
        enc.bit_logp(false, 1);
        enc.bit_logp(false, 1);
        for i in 0..50 {
            enc.enc_uint(i % 7, 7);
        }
        enc.patch_initial_bits(0b11, 2);
        enc.done();
        let bytes = enc.into_bytes();
        let mut dec = RangeDecoder::new(&bytes);
        assert!(dec.bit_logp(1));
        assert!(dec.bit_logp(1));
        for i in 0..50 {
            assert_eq!(dec.dec_uint(7), i % 7);
        }
    }
}
