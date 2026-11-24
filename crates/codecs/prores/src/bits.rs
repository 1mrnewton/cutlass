//! MSB-first bit reading and writing, and the Rice / exp-Golomb codewords.

use cutlass_decode::CodecError;

/// Reads bits from one component's data. Past the end it reads zeros, and
/// [`Self::overrun`] says so.
///
/// The next bits sit at the top of a 64-bit window. [`Self::refill`] tops
/// it up to 56 bits or more without a branch, so a caller that refills
/// once can take two codewords from the window with no memory access.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// The next byte of `data` to load into the window.
    next: usize,
    /// Unread bits, first bit highest. The top `valid` bits are counted;
    /// lower bits that a refill brought in early are the stream's too.
    window: u64,
    valid: u32,
    end: usize,
    /// Bit offset of `data` in the frame, for error positions.
    base: u64,
}

impl<'a> BitReader<'a> {
    #[inline(always)]
    pub(crate) fn new(data: &'a [u8], base: u64) -> Self {
        Self {
            data,
            next: 0,
            window: 0,
            valid: 0,
            end: data.len().saturating_mul(8),
            base,
        }
    }

    /// Tops the window up to at least 56 bits.
    #[inline(always)]
    pub(crate) fn refill(&mut self) {
        let word = match self.data.get(self.next..self.next.wrapping_add(8)) {
            Some(bytes) => match <[u8; 8]>::try_from(bytes) {
                Ok(array) => u64::from_be_bytes(array),
                Err(_) => 0,
            },
            None => tail_word(self.data, self.next),
        };
        // `valid` is 63 at most, so the shift is in range. The whole bytes
        // that fit below the counted bits are now counted.
        self.window |= word >> self.valid;
        self.next = self.next.wrapping_add(((63 - self.valid) >> 3) as usize);
        self.valid |= 56;
    }

    /// Refills when fewer than 32 bits are counted.
    #[inline(always)]
    pub(crate) fn ensure32(&mut self) {
        if self.valid < 32 {
            self.refill();
        }
    }

    /// The next 32 bits as the window holds them: right after a refill, or
    /// after [`Self::ensure32`]. Zero-filled past the end.
    #[inline(always)]
    pub(crate) fn top32(&self) -> u32 {
        (self.window >> 32) as u32
    }

    /// The next 32 bits, zero-filled past the end.
    #[inline(always)]
    pub(crate) fn peek32(&mut self) -> u32 {
        self.refill();
        self.top32()
    }

    /// Skips `bits` bits, 0 to 32.
    #[inline(always)]
    pub(crate) fn skip(&mut self, bits: u32) {
        if bits > self.valid {
            self.refill();
        }
        // After a refill 56 bits or more are counted, and `bits` is 32 at most.
        let bits = bits.min(self.valid);
        self.window <<= bits;
        self.valid -= bits;
    }

    /// Reads `bits` bits, 0 to 32.
    #[inline(always)]
    pub(crate) fn read(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        let value = self.peek32() >> (32 - bits.min(32));
        self.skip(bits);
        value
    }

    /// Bits read so far.
    #[inline(always)]
    fn pos(&self) -> usize {
        // `next` is at most a few bytes past the data, and `valid` bits of
        // what was loaded are unread, so neither step can wrap.
        self.next.wrapping_mul(8).wrapping_sub(self.valid as usize)
    }

    /// Bits left before the end. Negative after an overrun.
    #[cfg(test)]
    pub(crate) fn left(&self) -> isize {
        (self.end as isize).saturating_sub(self.pos() as isize)
    }

    /// True when a read went past the end.
    #[inline(always)]
    pub(crate) fn overrun(&self) -> bool {
        self.pos() > self.end
    }

    /// The frame bit offset of the next read.
    pub(crate) fn at(&self) -> u64 {
        self.base.saturating_add(self.pos() as u64)
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.at(),
            what,
        }
    }

    /// True when every bit left is zero and fewer than 32 remain: the
    /// component's coefficients have ended.
    #[cfg(test)]
    pub(crate) fn at_end_of_data(&mut self) -> bool {
        self.refill();
        self.ended()
    }

    /// Whether the coefficients have ended, for a window that was just
    /// refilled: nothing is left, or fewer than 32 bits and all zero.
    #[inline(always)]
    pub(crate) fn ended(&self) -> bool {
        // 32 bits or more remain when the position is 32 short of the end.
        let pos = self.pos();
        pos.wrapping_add(32) > self.end && (pos >= self.end || self.top32() == 0)
    }

    /// Decodes one codeword of the RDD 36 hybrid Rice / exp-Golomb code.
    ///
    /// The codebook byte holds the Rice order in bits 7..5, the exp-Golomb
    /// order in bits 4..2, and the switch point in bits 1..0. A codeword
    /// longer than 31 bits is refused.
    #[inline(always)]
    pub(crate) fn code(&mut self, codebook: u8) -> Result<u32, CodecError> {
        let (value, len) = codeword(self.peek32(), codebook);
        if len > 31 {
            return Err(self.malformed("codeword"));
        }
        self.skip(len);
        Ok(value)
    }
}

/// The eight bytes from `byte`, zero-filled past the end of `data`.
#[cold]
#[inline(never)]
fn tail_word(data: &[u8], byte: usize) -> u64 {
    let mut array = [0u8; 8];
    if let Some(rest) = data.get(byte..) {
        for (slot, value) in array.iter_mut().zip(rest) {
            *slot = *value;
        }
    }
    u64::from_be_bytes(array)
}

/// The codeword at the top of `buf`: its value and its length in bits. A
/// length over 31 is a codeword this crate refuses, and its value means
/// nothing.
///
/// Both forms are worked out and one is chosen, so the choice is a select
/// and not a branch on the stream's bits.
#[inline(always)]
pub(crate) fn codeword(buf: u32, codebook: u8) -> (u32, u32) {
    let switch = u32::from(codebook & 3);
    let rice = u32::from(codebook >> 5);
    let exp = u32::from((codebook >> 2) & 7);
    let zeros = buf.leading_zeros();
    // Exp-Golomb, when the zeros pass the switch point.
    let long = (exp + 2 * zeros).wrapping_sub(switch);
    let long_value = buf
        .wrapping_shr(32u32.wrapping_sub(long))
        .wrapping_sub(1 << exp)
        .wrapping_add((switch + 1) << rice);
    // Rice: the zeros in unary, a one, then `rice` bits.
    let short = zeros + 1 + rice;
    let short_value = (zeros << rice)
        .wrapping_add(buf.wrapping_shr(32u32.wrapping_sub(short)) & ((1 << rice) - 1));
    let pick = u32::from(zeros > switch).wrapping_neg();
    (
        (long_value & pick) | (short_value & !pick),
        (long & pick) | (short & !pick),
    )
}

/// Writes bits MSB first, onto the end of a byte vector.
///
/// Every write stores the pending bits as eight bytes at the end and moves
/// the end on by the whole bytes among them, so no write branches on how
/// full the writer is. The vector is kept longer than what is written and
/// cut to length by [`Self::finish`].
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Bytes of `bytes` that are written.
    len: usize,
    /// Bits not yet counted in `len`, in the low `filled` bits: at most 7
    /// between calls.
    acc: u64,
    filled: u32,
    /// Bits written since the writer started.
    #[cfg(test)]
    written: usize,
}

/// Room kept past the written bytes, so a store of eight never grows the
/// vector in the common case.
const HEADROOM: usize = 4096;

impl BitWriter {
    /// A writer that appends to `bytes`, so one buffer takes many parts.
    #[inline(always)]
    pub(crate) fn onto(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len(),
            bytes,
            acc: 0,
            filled: 0,
            #[cfg(test)]
            written: 0,
        }
    }

    /// Writes the low `bits` bits of `value`, 0 to 57.
    #[inline(always)]
    pub(crate) fn put_long(&mut self, bits: u32, value: u64) {
        let bits = bits.min(57);
        #[cfg(test)]
        {
            self.written += bits as usize;
        }
        // At most 7 bits are pending, so 64 hold them and these.
        self.acc = (self.acc << bits) | (value & ((1u64 << bits) - 1));
        self.filled += bits;
        if self.bytes.len() < self.len + 8 {
            grow(&mut self.bytes, self.len);
        }
        // The pending bits, first bit highest; `filled` is 1 to 64 here
        // unless nothing is pending, when the store is of zeros.
        let top = self.acc.checked_shl(64 - self.filled).unwrap_or(0);
        if let Some(slot) = self.bytes.get_mut(self.len..self.len + 8) {
            slot.copy_from_slice(&top.to_be_bytes());
        }
        self.len += (self.filled >> 3) as usize;
        self.filled &= 7;
    }

    /// Writes the low `bits` bits of `value`, 0 to 32.
    #[inline(always)]
    pub(crate) fn put(&mut self, bits: u32, value: u32) {
        self.put_long(bits.min(32), u64::from(value));
    }

    /// Writes `count` zero bits.
    #[inline(always)]
    pub(crate) fn zeros(&mut self, mut count: u32) {
        while count > 0 {
            let step = count.min(32);
            self.put(step, 0);
            count -= step;
        }
    }

    /// Bits written so far, from where the writer started.
    #[cfg(test)]
    pub(crate) fn len_bits(&self) -> usize {
        self.written
    }

    /// Pads with zero bits to a byte boundary and returns the bytes.
    #[inline(always)]
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.put_long(8 - self.filled, 0);
        }
        self.bytes.truncate(self.len);
        self.bytes
    }

    /// Encodes one codeword, the inverse of [`BitReader::code`].
    #[inline(always)]
    pub(crate) fn code(&mut self, codebook: u8, value: u32) {
        let (bits, zeros, payload) = code_parts(codebook, value);
        if zeros + bits <= 57 {
            // The payload is below `bits` bits, so the zeros come with it.
            self.put_long(zeros + bits, u64::from(payload));
        } else {
            self.zeros(zeros);
            self.put(bits, payload);
        }
    }

    /// Writes a run and the level and sign after it, each codeword given
    /// as [`RUN_WORDS`] and [`LEVEL_WORDS`] hold them.
    #[inline(always)]
    pub(crate) fn pair_then_bit(&mut self, run: u32, level: u32, bit: u32) {
        let (len_a, len_b) = (run & 0xff, (level & 0xff) + 1);
        let tail = (u64::from(level >> 8) << 1) | u64::from(bit & 1);
        // Two table words are at most 24 bits each.
        self.put_long(len_a + len_b, (u64::from(run >> 8) << len_b) | tail);
    }

    /// Encodes a codeword from `first`, then one from `second`, then one bit.
    #[inline(always)]
    pub(crate) fn code_pair_then_bit(&mut self, first: (u8, u32), second: (u8, u32), bit: u32) {
        let (bits_a, zeros_a, payload_a) = code_parts(first.0, first.1);
        let (bits_b, zeros_b, payload_b) = code_parts(second.0, second.1);
        let len_a = zeros_a + bits_a;
        let len_b = zeros_b + bits_b + 1;
        let tail = (u64::from(payload_b) << 1) | u64::from(bit & 1);
        if len_a + len_b <= 57 && len_b <= 33 {
            self.put_long(len_a + len_b, (u64::from(payload_a) << len_b) | tail);
        } else {
            self.zeros(zeros_a);
            self.put(bits_a, payload_a);
            self.zeros(zeros_b);
            self.put(bits_b, payload_b);
            self.put(1, bit);
        }
    }
}

/// Values each codeword table covers. Larger values are worked out.
pub(crate) const TABLE_VALUES: usize = 64;

/// A table word for `value` in `codebook`: the codeword's bits above its
/// length in the low byte. Every codeword for a value under
/// [`TABLE_VALUES`] is 24 bits or fewer.
const fn table_word(codebook: u8, value: u32) -> u32 {
    let switch = (codebook & 3) as u32;
    let rice = (codebook >> 5) as u32;
    let exp = ((codebook >> 2) & 7) as u32;
    let base = (switch + 1) << rice;
    if value < base {
        let payload = (1 << rice) | (value & ((1 << rice) - 1));
        (payload << 8) | ((value >> rice) + 1 + rice)
    } else {
        let x = value - base + (1 << exp);
        let bits = 32 - x.leading_zeros();
        (x << 8) | (2 * bits - exp + switch)
    }
}

const fn table<const N: usize>(books: [u8; N]) -> [[u32; TABLE_VALUES]; N] {
    let mut out = [[0u32; TABLE_VALUES]; N];
    let mut book = 0;
    while book < N {
        let mut value = 0;
        while value < TABLE_VALUES {
            out[book][value] = table_word(books[book], value as u32);
            value += 1;
        }
        book += 1;
    }
    out
}

/// Run codewords: `[previous run, 15 at most][run]`.
pub(crate) static RUN_WORDS: [[u32; TABLE_VALUES]; 16] = table(crate::tables::RUN_CODEBOOK);
/// Level codewords: `[previous level, 1 to 9, less 1][level less 1]`.
pub(crate) static LEVEL_WORDS: [[u32; TABLE_VALUES]; 9] = table(crate::tables::LEVEL_CODEBOOK);

/// Lengthens `bytes` to hold [`HEADROOM`] past `len`. Apart from the
/// writer, so its other fields can stay in registers.
#[cold]
#[inline(never)]
fn grow(bytes: &mut Vec<u8>, len: usize) {
    bytes.resize(len + HEADROOM, 0);
}

/// The codeword for `value`: `zeros` leading zeros, then `bits` bits of `payload`.
///
/// Both forms are worked out and one is chosen, so the choice is a select
/// and not a branch on the value.
#[inline(always)]
pub(crate) fn code_parts(codebook: u8, value: u32) -> (u32, u32, u32) {
    let switch = u32::from(codebook & 3);
    let rice = u32::from(codebook >> 5);
    let exp = u32::from((codebook >> 2) & 7);
    let base = (switch + 1) << rice;
    // Rice: the quotient in unary, a one, then `rice` bits.
    let short_zeros = value >> rice;
    let short_payload = (1 << rice) | (value & ((1u32 << rice) - 1));
    // Exp-Golomb, from the switch point up.
    let x = value.wrapping_sub(base).wrapping_add(1 << exp);
    let long_bits = 32 - x.leading_zeros();
    let long_zeros = long_bits.wrapping_sub(exp).wrapping_add(switch);
    let pick = u32::from(value < base).wrapping_neg();
    (
        ((1 + rice) & pick) | (long_bits & !pick),
        (short_zeros & pick) | (long_zeros & !pick),
        (short_payload & pick) | (x & !pick),
    )
}

/// Length of a codeword in bits.
#[cfg(test)]
pub(crate) fn code_len(codebook: u8, value: u32) -> u32 {
    let (bits, zeros, _) = code_parts(codebook, value);
    bits + zeros
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter, code_len};
    use crate::tables::{DC_CODEBOOK, FIRST_DC_CODEBOOK, LEVEL_CODEBOOK, RUN_CODEBOOK};

    #[test]
    fn every_codebook_round_trips_its_symbols() {
        let mut books = vec![FIRST_DC_CODEBOOK];
        books.extend_from_slice(&DC_CODEBOOK);
        books.extend_from_slice(&RUN_CODEBOOK);
        books.extend_from_slice(&LEVEL_CODEBOOK);
        for &book in &books {
            let mut writer = BitWriter::default();
            let values: Vec<u32> = (0..600)
                .chain([4_095, 16_383, 32_767])
                .filter(|&value| code_len(book, value) <= 31)
                .collect();
            assert!(values.len() >= 602, "book {book:#x}");
            let mut total = 0;
            for &value in &values {
                writer.code(book, value);
                total += code_len(book, value) as usize;
                assert_eq!(writer.len_bits(), total, "book {book:#x} value {value}");
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes, 0);
            for &value in &values {
                assert_eq!(reader.code(book).expect("code"), value, "book {book:#x}");
            }
            assert!(!reader.overrun());
        }
    }

    #[test]
    fn table_words_are_the_worked_out_codewords() {
        use super::{LEVEL_WORDS, RUN_WORDS, TABLE_VALUES, code_parts};
        let tables = [
            (&RUN_WORDS[..], &RUN_CODEBOOK[..]),
            (&LEVEL_WORDS[..], &LEVEL_CODEBOOK[..]),
        ];
        for (words, books) in tables {
            assert_eq!(words.len(), books.len());
            for (row, &book) in words.iter().zip(books) {
                for value in 0..TABLE_VALUES {
                    let (bits, zeros, payload) = code_parts(book, value as u32);
                    assert_eq!(row[value] & 0xff, zeros + bits, "book {book:#x} {value}");
                    assert_eq!(row[value] >> 8, payload, "book {book:#x} {value}");
                    assert!(zeros + bits <= 24);
                }
            }
        }
        // A table pair writes what two codewords and a bit write.
        for (run, level, bit) in [(0u32, 0u32, 1u32), (63, 63, 0), (5, 40, 1), (17, 2, 0)] {
            let mut tabled = BitWriter::default();
            tabled.pair_then_bit(
                RUN_WORDS[3][run as usize],
                LEVEL_WORDS[8][level as usize],
                bit,
            );
            let mut worked = BitWriter::default();
            worked.code(RUN_CODEBOOK[3], run);
            worked.code(LEVEL_CODEBOOK[8], level);
            worked.put(1, bit);
            assert_eq!(tabled.len_bits(), worked.len_bits());
            assert_eq!(tabled.finish(), worked.finish());
        }
    }

    #[test]
    fn long_and_short_writes_give_the_same_bytes() {
        // The same bits through writes of every width, against one bit at a time.
        let mut single = BitWriter::default();
        let mut mixed = BitWriter::onto(vec![0xab, 0xcd]);
        let mut state = 77u64;
        let mut total = 0;
        for round in 0..5000u32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let width = (round * 11 + 3) % 58;
            let value = state >> 3;
            mixed.put_long(width, value);
            for bit in (0..width).rev() {
                single.put(1, ((value >> bit) & 1) as u32);
            }
            total += width as usize;
            assert_eq!(mixed.len_bits(), total);
        }
        let mixed = mixed.finish();
        assert_eq!(&mixed[..2], &[0xab, 0xcd]);
        assert_eq!(&mixed[2..], &single.finish()[..]);
        assert_eq!(mixed.len(), 2 + total.div_ceil(8));
    }

    #[test]
    fn rice_and_exp_golomb_match_hand_coded_words() {
        // Codebook 0x04: Rice order 0 up to 0, then exp-Golomb order 1.
        // 0 is "1"; 1 is "010"; 2 is "011"; 3 is "00100".
        let bytes = [0b1010_0110, 0b0100_0000];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.code(0x04).expect("0"), 0);
        assert_eq!(reader.code(0x04).expect("1"), 1);
        assert_eq!(reader.code(0x04).expect("2"), 2);
        assert_eq!(reader.code(0x04).expect("3"), 3);
    }

    #[test]
    fn a_run_of_zeros_is_refused_not_wrapped() {
        let bytes = [0u8; 8];
        let mut reader = BitReader::new(&bytes, 16);
        assert!(reader.code(0x04).is_err());
        let mut reader = BitReader::new(&[], 0);
        assert!(reader.code(FIRST_DC_CODEBOOK).is_err());
    }

    #[test]
    fn the_window_reads_what_a_bit_at_a_time_reads() {
        // Mixed widths across many refills, against one bit per read.
        let bytes: Vec<u8> = (0u32..600)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let mut single = BitReader::new(&bytes, 0);
        let mut mixed = BitReader::new(&bytes, 0);
        let mut width = 0u32;
        while mixed.left() > 0 {
            width = (width * 7 + 3) % 33;
            let got = mixed.read(width);
            let mut want = 0u64;
            for _ in 0..width {
                want = (want << 1) | u64::from(single.read(1));
            }
            assert_eq!(u64::from(got), want, "width {width}");
            assert_eq!(mixed.at(), single.at());
        }
        assert!(mixed.at_end_of_data());
    }

    #[test]
    fn reads_past_the_end_are_zero_and_flagged() {
        let bytes = [0xff];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read(4), 0xf);
        assert!(!reader.at_end_of_data());
        assert_eq!(reader.read(8), 0xf0);
        assert!(reader.overrun());
        assert!(reader.at_end_of_data());
    }
}
