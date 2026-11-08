//! MD5 (RFC 1321) for the STREAMINFO signature.
//!
//! RFC 9639, section 8.2: the message is every sample of every channel,
//! interleaved, signed, little-endian, each sample widened to whole bytes
//! by sign extension. [`Md5::samples`] packs that message.

/// A running MD5 digest.
#[derive(Clone, Debug)]
pub struct Md5 {
    state: [u32; 4],
    buffer: [u8; 64],
    filled: usize,
    length: u64,
}

const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// `floor(abs(sin(i + 1)) * 2^32)`, the RFC 1321 table.
const SINES: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buffer: [0; 64],
            filled: 0,
            length: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self
            .length
            .wrapping_add(u64::try_from(data.len()).unwrap_or(0));
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.buffer[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            let block = self.buffer;
            self.block(&block);
            self.filled = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            self.block(block);
        }
        self.buffer[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    /// Adds interleaved samples of `bits` depth in the RFC 9639 packing.
    pub fn samples(&mut self, interleaved: &[i32], bits: u32) {
        let width = bits.div_ceil(8).clamp(1, 4) as usize;
        let mut packed = [0u8; 4096];
        let per = packed.len() / width;
        for chunk in interleaved.chunks(per) {
            let mut at = 0;
            for sample in chunk {
                packed[at..at + width].copy_from_slice(&sample.to_le_bytes()[..width]);
                at += width;
            }
            self.update(&packed[..at]);
        }
    }

    /// The digest. The running state is not changed, so more data may follow
    /// for a later digest of the longer message.
    #[must_use]
    pub fn digest(&self) -> [u8; 16] {
        let mut copy = self.clone();
        let bits = self.length.wrapping_mul(8);
        copy.update(&[0x80]);
        while copy.filled != 56 {
            copy.update(&[0]);
        }
        copy.update(&bits.to_le_bytes());
        let mut out = [0u8; 16];
        for (slot, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(copy.state) {
            *slot = word.to_le_bytes();
        }
        out
    }

    fn block(&mut self, block: &[u8; 64]) {
        let mut words = [0u32; 16];
        for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*bytes);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let sum = a
                .wrapping_add(f)
                .wrapping_add(SINES[i])
                .wrapping_add(words[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(sum.rotate_left(SHIFTS[i]));
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

#[cfg(test)]
mod tests {
    use super::Md5;

    fn hex(digest: [u8; 16]) -> String {
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn rfc_1321_test_suite() {
        let cases: [(&[u8], &str); 5] = [
            (b"", "d41d8cd98f00b204e9800998ecf8427e"),
            (b"a", "0cc175b9c0f1b6a831c399e269772661"),
            (b"abc", "900150983cd24fb0d6963f7d28e17f72"),
            (b"message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        for (input, expected) in cases {
            let mut md5 = Md5::new();
            md5.update(input);
            assert_eq!(hex(md5.digest()), expected);
        }
    }

    #[test]
    fn split_updates_match_one_update() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut whole = Md5::new();
        whole.update(&data);
        let mut parts = Md5::new();
        for chunk in data.chunks(13) {
            parts.update(chunk);
        }
        assert_eq!(whole.digest(), parts.digest());
    }

    #[test]
    fn samples_pack_sign_extended_little_endian_whole_bytes() {
        let mut packed = Md5::new();
        packed.samples(&[-1, 2, -300], 12);
        let mut raw = Md5::new();
        raw.update(&[0xFF, 0xFF, 0x02, 0x00, 0xD4, 0xFE]);
        assert_eq!(packed.digest(), raw.digest());
    }
}
