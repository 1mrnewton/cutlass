//! Bits for the parameter sets and slice headers the VA-API encoders hand
//! the driver as packed headers.

/// Writes RBSP syntax, most significant bit first.
#[derive(Clone, Debug, Default)]
pub struct Writer {
    bytes: Vec<u8>,
    /// Bits written into the last byte, 0 when it is full or there is none.
    used: u32,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `width` bits of `value`, at most 64.
    pub fn bits(&mut self, width: u32, value: u64) {
        for bit in (0..width).rev() {
            if self.used == 0 {
                self.bytes.push(0);
            }
            if let Some(last) = self.bytes.last_mut()
                && (value >> bit) & 1 == 1
            {
                *last |= 0x80 >> self.used;
            }
            self.used = (self.used + 1) % 8;
        }
    }

    /// u(n).
    pub fn u(&mut self, width: u32, value: u32) {
        self.bits(width, u64::from(value));
    }

    pub fn flag(&mut self, value: bool) {
        self.bits(1, u64::from(value));
    }

    /// ue(v): exp-Golomb.
    pub fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let width = 64 - code.leading_zeros();
        self.bits(width - 1, 0);
        self.bits(width, code);
    }

    /// se(v): signed exp-Golomb.
    pub fn se(&mut self, value: i32) {
        let value = i64::from(value);
        let code = if value > 0 { 2 * value - 1 } else { -2 * value };
        self.ue(u32::try_from(code).unwrap_or(u32::MAX));
    }

    /// rbsp_trailing_bits() and HEVC's byte_alignment(): a one, then zeros
    /// to the byte.
    pub fn align(&mut self) {
        self.flag(true);
        while self.used != 0 {
            self.flag(false);
        }
    }

    /// Bits written so far.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 - ((8 - self.used as usize) % 8)
    }

    /// The bytes, the last one padded with zeros.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// A NAL unit's bytes with emulation prevention: a 0x03 after any two zero
/// bytes that a byte of 0x03 or less follows.
pub fn escape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len() + nal.len() / 64 + 1);
    let mut zeros = 0;
    for &byte in nal {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

const START: [u8; 4] = [0, 0, 0, 1];

/// The data of a packed header and its length in bits, start codes and
/// emulation prevention bytes included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    pub data: Vec<u8>,
    pub bits: u32,
}

impl Packed {
    /// Whole NAL units, each escaped after a start code.
    pub fn units(units: &[Vec<u8>]) -> Self {
        let mut data = Vec::new();
        for unit in units {
            data.extend_from_slice(&START);
            data.extend(escape(unit));
        }
        let bits = u32::try_from(data.len() * 8).unwrap_or(u32::MAX);
        Self { data, bits }
    }

    /// A NAL unit that ends inside a byte, as an H.264 slice header does
    /// before the driver appends the slice data.
    pub fn partial(unit: Writer) -> Self {
        let padding = (8 - unit.bit_len() % 8) % 8;
        let mut data = START.to_vec();
        data.extend(escape(&unit.into_bytes()));
        let bits = u32::try_from(data.len() * 8 - padding).unwrap_or(u32::MAX);
        Self { data, bits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_codes_match_the_table() {
        let mut out = Writer::new();
        out.ue(0); // 1
        out.ue(1); // 010
        out.ue(2); // 011
        out.ue(7); // 0001000
        out.se(-1); // 011
        out.se(1); // 010
        out.se(0); // 1
        assert_eq!(out.bit_len(), 1 + 3 + 3 + 7 + 3 + 3 + 1);
        assert_eq!(out.into_bytes(), [0b1010_0110, 0b0010_0001, 0b1010_1000]);
    }

    #[test]
    fn the_largest_code_fits() {
        let mut out = Writer::new();
        out.ue(u32::MAX);
        assert_eq!(out.bit_len(), 65);
        let bytes = out.into_bytes();
        assert_eq!(&bytes[..4], [0, 0, 0, 0]);
        assert_eq!(bytes[4], 0x80);
    }

    #[test]
    fn alignment_is_a_one_then_zeros() {
        let mut out = Writer::new();
        out.u(3, 0b101);
        out.align();
        assert_eq!(out.bit_len(), 8);
        out.align();
        assert_eq!(out.into_bytes(), [0b1011_0000, 0x80]);
    }

    #[test]
    fn zero_runs_are_escaped() {
        assert_eq!(escape(&[0, 0, 1, 0, 0, 0]), [0, 0, 3, 1, 0, 0, 3, 0]);
        assert_eq!(escape(&[0, 0, 4]), [0, 0, 4]);
        assert_eq!(escape(&[0, 0, 0, 0]), [0, 0, 3, 0, 0]);
    }

    #[test]
    fn packed_lengths_count_start_codes_and_padding() {
        let whole = Packed::units(&[vec![0x67, 0, 0, 1], vec![0x68]]);
        assert_eq!(whole.data, [0, 0, 0, 1, 0x67, 0, 0, 3, 1, 0, 0, 0, 1, 0x68]);
        assert_eq!(whole.bits, 14 * 8);
        let mut header = Writer::new();
        header.u(8, 0x65);
        header.u(3, 0b111);
        let partial = Packed::partial(header);
        assert_eq!(partial.data, [0, 0, 0, 1, 0x65, 0b1110_0000]);
        assert_eq!(partial.bits, 32 + 11);
    }
}
