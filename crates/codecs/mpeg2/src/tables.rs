//! Annex B code tables, the scans, and the default matrices.
//!
//! Each table is written as the code strings printed in H.262 Annex B and
//! turned into lookup tables at compile time. A table entry with `len == 0`
//! is a bit pattern that is not a code.

/// One flat-table entry: the decoded value and the code length in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Vlc {
    pub value: i16,
    pub len: u8,
}

const NONE: Vlc = Vlc { value: 0, len: 0 };

/// Bits and length of a code written as `0`/`1` characters with spaces.
pub(crate) const fn parse(code: &str) -> (u32, u32) {
    let bytes = code.as_bytes();
    let mut bits = 0u32;
    let mut len = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'0' {
            bits <<= 1;
            len += 1;
        } else if bytes[i] == b'1' {
            bits = (bits << 1) | 1;
            len += 1;
        }
        i += 1;
    }
    (bits, len)
}

/// A table indexed by the next `bits` bits of the stream.
const fn flat<const N: usize>(bits: u32, codes: &[(&str, i16)]) -> [Vlc; N] {
    let mut table = [NONE; N];
    let mut k = 0;
    while k < codes.len() {
        let (code, len) = parse(codes[k].0);
        let shift = bits - len;
        let first = (code << shift) as usize;
        let count = 1usize << shift;
        let mut j = 0;
        while j < count {
            table[first + j] = Vlc {
                value: codes[k].1,
                len: len as u8,
            };
            j += 1;
        }
        k += 1;
    }
    table
}

/// Code and length for each value `0..N`, from a code list. Length 0 is no code.
const fn encode_by_value<const N: usize>(codes: &[(&str, i16)], offset: i16) -> [(u32, u8); N] {
    let mut table = [(0u32, 0u8); N];
    let mut k = 0;
    while k < codes.len() {
        let (code, len) = parse(codes[k].0);
        let index = codes[k].1 + offset;
        if index >= 0 && (index as usize) < N {
            table[index as usize] = (code, len as u8);
        }
        k += 1;
    }
    table
}

// B.1, macroblock_address_increment. Value 0 is macroblock_escape and -1
// is the 11172-2 macroblock_stuffing code.
pub(crate) const MB_ESCAPE: i16 = 0;
pub(crate) const MB_STUFFING: i16 = -1;
const MB_ADDR_CODES: &[(&str, i16)] = &[
    ("1", 1),
    ("011", 2),
    ("010", 3),
    ("0011", 4),
    ("0010", 5),
    ("0001 1", 6),
    ("0001 0", 7),
    ("0000 111", 8),
    ("0000 110", 9),
    ("0000 1011", 10),
    ("0000 1010", 11),
    ("0000 1001", 12),
    ("0000 1000", 13),
    ("0000 0111", 14),
    ("0000 0110", 15),
    ("0000 0101 11", 16),
    ("0000 0101 10", 17),
    ("0000 0101 01", 18),
    ("0000 0101 00", 19),
    ("0000 0100 11", 20),
    ("0000 0100 10", 21),
    ("0000 0100 011", 22),
    ("0000 0100 010", 23),
    ("0000 0100 001", 24),
    ("0000 0100 000", 25),
    ("0000 0011 111", 26),
    ("0000 0011 110", 27),
    ("0000 0011 101", 28),
    ("0000 0011 100", 29),
    ("0000 0011 011", 30),
    ("0000 0011 010", 31),
    ("0000 0011 001", 32),
    ("0000 0011 000", 33),
    ("0000 0001 000", MB_ESCAPE),
    ("0000 0001 111", MB_STUFFING),
];
pub(crate) const MB_ADDR_BITS: u32 = 11;
pub(crate) static MB_ADDR: [Vlc; 1 << 11] = flat(11, MB_ADDR_CODES);
pub(crate) static MB_ADDR_ENC: [(u32, u8); 34] = encode_by_value(MB_ADDR_CODES, 0);

// B.2 to B.4, macroblock_type, as flag sets.
pub(crate) const MB_QUANT: i16 = 1;
pub(crate) const MB_FORWARD: i16 = 2;
pub(crate) const MB_BACKWARD: i16 = 4;
pub(crate) const MB_PATTERN: i16 = 8;
pub(crate) const MB_INTRA: i16 = 16;

const MB_TYPE_I_CODES: &[(&str, i16)] = &[("1", MB_INTRA), ("01", MB_INTRA | MB_QUANT)];
const MB_TYPE_P_CODES: &[(&str, i16)] = &[
    ("1", MB_FORWARD | MB_PATTERN),
    ("01", MB_PATTERN),
    ("001", MB_FORWARD),
    ("0001 1", MB_INTRA),
    ("0001 0", MB_QUANT | MB_FORWARD | MB_PATTERN),
    ("0000 1", MB_QUANT | MB_PATTERN),
    ("0000 01", MB_QUANT | MB_INTRA),
];
const MB_TYPE_B_CODES: &[(&str, i16)] = &[
    ("10", MB_FORWARD | MB_BACKWARD),
    ("11", MB_FORWARD | MB_BACKWARD | MB_PATTERN),
    ("010", MB_BACKWARD),
    ("011", MB_BACKWARD | MB_PATTERN),
    ("0010", MB_FORWARD),
    ("0011", MB_FORWARD | MB_PATTERN),
    ("0001 1", MB_INTRA),
    ("0001 0", MB_QUANT | MB_FORWARD | MB_BACKWARD | MB_PATTERN),
    ("0000 11", MB_QUANT | MB_FORWARD | MB_PATTERN),
    ("0000 10", MB_QUANT | MB_BACKWARD | MB_PATTERN),
    ("0000 01", MB_QUANT | MB_INTRA),
];
pub(crate) const MB_TYPE_BITS: u32 = 6;
pub(crate) static MB_TYPE_I: [Vlc; 64] = flat(6, MB_TYPE_I_CODES);
pub(crate) static MB_TYPE_P: [Vlc; 64] = flat(6, MB_TYPE_P_CODES);
pub(crate) static MB_TYPE_B: [Vlc; 64] = flat(6, MB_TYPE_B_CODES);
pub(crate) static MB_TYPE_P_ENC: [(u32, u8); 32] = encode_by_value(MB_TYPE_P_CODES, 0);
pub(crate) static MB_TYPE_B_ENC: [(u32, u8); 32] = encode_by_value(MB_TYPE_B_CODES, 0);
pub(crate) static MB_TYPE_I_ENC: [(u32, u8); 32] = encode_by_value(MB_TYPE_I_CODES, 0);

// B.9, coded_block_pattern.
const CBP_CODES: &[(&str, i16)] = &[
    ("111", 60),
    ("1101", 4),
    ("1100", 8),
    ("1011", 16),
    ("1010", 32),
    ("1001 1", 12),
    ("1001 0", 48),
    ("1000 1", 20),
    ("1000 0", 40),
    ("0111 1", 28),
    ("0111 0", 44),
    ("0110 1", 52),
    ("0110 0", 56),
    ("0101 1", 1),
    ("0101 0", 61),
    ("0100 1", 2),
    ("0100 0", 62),
    ("0011 11", 24),
    ("0011 10", 36),
    ("0011 01", 3),
    ("0011 00", 63),
    ("0010 111", 5),
    ("0010 110", 9),
    ("0010 101", 17),
    ("0010 100", 33),
    ("0010 011", 6),
    ("0010 010", 10),
    ("0010 001", 18),
    ("0010 000", 34),
    ("0001 1111", 7),
    ("0001 1110", 11),
    ("0001 1101", 19),
    ("0001 1100", 35),
    ("0001 1011", 13),
    ("0001 1010", 49),
    ("0001 1001", 21),
    ("0001 1000", 41),
    ("0001 0111", 14),
    ("0001 0110", 50),
    ("0001 0101", 22),
    ("0001 0100", 42),
    ("0001 0011", 15),
    ("0001 0010", 51),
    ("0001 0001", 23),
    ("0001 0000", 43),
    ("0000 1111", 25),
    ("0000 1110", 37),
    ("0000 1101", 26),
    ("0000 1100", 38),
    ("0000 1011", 29),
    ("0000 1010", 45),
    ("0000 1001", 53),
    ("0000 1000", 57),
    ("0000 0111", 30),
    ("0000 0110", 46),
    ("0000 0101", 54),
    ("0000 0100", 58),
    ("0000 0011 1", 31),
    ("0000 0011 0", 47),
    ("0000 0010 1", 55),
    ("0000 0010 0", 59),
    ("0000 0001 1", 27),
    ("0000 0001 0", 39),
    ("0000 0000 1", 0),
];
pub(crate) const CBP_BITS: u32 = 9;
pub(crate) static CBP: [Vlc; 1 << 9] = flat(9, CBP_CODES);
pub(crate) static CBP_ENC: [(u32, u8); 64] = encode_by_value(CBP_CODES, 0);

// B.10, motion_code. The last bit of every code but `1` is the sign.
const MOTION_CODES: &[(&str, i16)] = &[
    ("0000 0011 001", -16),
    ("0000 0011 011", -15),
    ("0000 0011 101", -14),
    ("0000 0011 111", -13),
    ("0000 0100 001", -12),
    ("0000 0100 011", -11),
    ("0000 0100 11", -10),
    ("0000 0101 01", -9),
    ("0000 0101 11", -8),
    ("0000 0111", -7),
    ("0000 1001", -6),
    ("0000 1011", -5),
    ("0000 111", -4),
    ("0001 1", -3),
    ("0011", -2),
    ("011", -1),
    ("1", 0),
    ("010", 1),
    ("0010", 2),
    ("0001 0", 3),
    ("0000 110", 4),
    ("0000 1010", 5),
    ("0000 1000", 6),
    ("0000 0110", 7),
    ("0000 0101 10", 8),
    ("0000 0101 00", 9),
    ("0000 0100 10", 10),
    ("0000 0100 010", 11),
    ("0000 0100 000", 12),
    ("0000 0011 110", 13),
    ("0000 0011 100", 14),
    ("0000 0011 010", 15),
    ("0000 0011 000", 16),
];
pub(crate) const MOTION_BITS: u32 = 11;
pub(crate) static MOTION: [Vlc; 1 << 11] = flat(11, MOTION_CODES);
pub(crate) static MOTION_ENC: [(u32, u8); 33] = encode_by_value(MOTION_CODES, 16);

// B.11, dmvector.
const DMV_CODES: &[(&str, i16)] = &[("11", -1), ("0", 0), ("10", 1)];
pub(crate) static DMV: [Vlc; 4] = flat(2, DMV_CODES);

// B.12 and B.13, dct_dc_size.
const DC_LUMA_CODES: &[(&str, i16)] = &[
    ("100", 0),
    ("00", 1),
    ("01", 2),
    ("101", 3),
    ("110", 4),
    ("1110", 5),
    ("1111 0", 6),
    ("1111 10", 7),
    ("1111 110", 8),
    ("1111 1110", 9),
    ("1111 1111 0", 10),
    ("1111 1111 1", 11),
];
const DC_CHROMA_CODES: &[(&str, i16)] = &[
    ("00", 0),
    ("01", 1),
    ("10", 2),
    ("110", 3),
    ("1110", 4),
    ("1111 0", 5),
    ("1111 10", 6),
    ("1111 110", 7),
    ("1111 1110", 8),
    ("1111 1111 0", 9),
    ("1111 1111 10", 10),
    ("1111 1111 11", 11),
];
pub(crate) const DC_BITS: u32 = 10;
pub(crate) static DC_LUMA: [Vlc; 1 << 10] = flat(10, DC_LUMA_CODES);
pub(crate) static DC_CHROMA: [Vlc; 1 << 10] = flat(10, DC_CHROMA_CODES);
pub(crate) static DC_LUMA_ENC: [(u32, u8); 12] = encode_by_value(DC_LUMA_CODES, 0);
pub(crate) static DC_CHROMA_ENC: [(u32, u8); 12] = encode_by_value(DC_CHROMA_CODES, 0);

/// One DCT coefficient code. `len` excludes the sign bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Coef {
    pub run: u8,
    pub level: u8,
    pub len: u8,
}

pub(crate) const RUN_EOB: u8 = 64;
pub(crate) const RUN_ESCAPE: u8 = 65;
const COEF_NONE: Coef = Coef {
    run: 0,
    level: 0,
    len: 0,
};

/// Codes of B.14 and B.15 with the trailing sign bit removed:
/// `(code, run, level)`. Run 64 is End of Block and run 65 is Escape.
type CoefCode = (&'static str, u8, u8);

// The 13- to 16-bit codes that B.14 and B.15 share.
const COEF_SHARED_TAIL: &[CoefCode] = &[
    ("0000 0000 1011 0", 1, 6),
    ("0000 0000 1010 1", 1, 7),
    ("0000 0000 1010 0", 2, 5),
    ("0000 0000 1001 1", 3, 4),
    ("0000 0000 1001 0", 5, 3),
    ("0000 0000 1000 1", 9, 2),
    ("0000 0000 1000 0", 10, 2),
    ("0000 0000 1111 1", 22, 1),
    ("0000 0000 1111 0", 23, 1),
    ("0000 0000 1110 1", 24, 1),
    ("0000 0000 1110 0", 25, 1),
    ("0000 0000 1101 1", 26, 1),
    ("0000 0000 0111 11", 0, 16),
    ("0000 0000 0111 10", 0, 17),
    ("0000 0000 0111 01", 0, 18),
    ("0000 0000 0111 00", 0, 19),
    ("0000 0000 0110 11", 0, 20),
    ("0000 0000 0110 10", 0, 21),
    ("0000 0000 0110 01", 0, 22),
    ("0000 0000 0110 00", 0, 23),
    ("0000 0000 0101 11", 0, 24),
    ("0000 0000 0101 10", 0, 25),
    ("0000 0000 0101 01", 0, 26),
    ("0000 0000 0101 00", 0, 27),
    ("0000 0000 0100 11", 0, 28),
    ("0000 0000 0100 10", 0, 29),
    ("0000 0000 0100 01", 0, 30),
    ("0000 0000 0100 00", 0, 31),
    ("0000 0000 0011 000", 0, 32),
    ("0000 0000 0010 111", 0, 33),
    ("0000 0000 0010 110", 0, 34),
    ("0000 0000 0010 101", 0, 35),
    ("0000 0000 0010 100", 0, 36),
    ("0000 0000 0010 011", 0, 37),
    ("0000 0000 0010 010", 0, 38),
    ("0000 0000 0010 001", 0, 39),
    ("0000 0000 0010 000", 0, 40),
    ("0000 0000 0011 111", 1, 8),
    ("0000 0000 0011 110", 1, 9),
    ("0000 0000 0011 101", 1, 10),
    ("0000 0000 0011 100", 1, 11),
    ("0000 0000 0011 011", 1, 12),
    ("0000 0000 0011 010", 1, 13),
    ("0000 0000 0011 001", 1, 14),
    ("0000 0000 0001 0011", 1, 15),
    ("0000 0000 0001 0010", 1, 16),
    ("0000 0000 0001 0001", 1, 17),
    ("0000 0000 0001 0000", 1, 18),
    ("0000 0000 0001 0100", 6, 3),
    ("0000 0000 0001 1010", 11, 2),
    ("0000 0000 0001 1001", 12, 2),
    ("0000 0000 0001 1000", 13, 2),
    ("0000 0000 0001 0111", 14, 2),
    ("0000 0000 0001 0110", 15, 2),
    ("0000 0000 0001 0101", 16, 2),
    ("0000 0000 0001 1111", 27, 1),
    ("0000 0000 0001 1110", 28, 1),
    ("0000 0000 0001 1101", 29, 1),
    ("0000 0000 0001 1100", 30, 1),
    ("0000 0000 0001 1011", 31, 1),
];

// B.14 without the first-coefficient `1 s` entry, which the decoder handles.
const COEF_ZERO: &[CoefCode] = &[
    ("10", RUN_EOB, 0),
    ("11", 0, 1),
    ("011", 1, 1),
    ("0100", 0, 2),
    ("0101", 2, 1),
    ("0010 1", 0, 3),
    ("0011 1", 3, 1),
    ("0011 0", 4, 1),
    ("0001 10", 1, 2),
    ("0001 11", 5, 1),
    ("0001 01", 6, 1),
    ("0001 00", 7, 1),
    ("0000 110", 0, 4),
    ("0000 100", 2, 2),
    ("0000 111", 8, 1),
    ("0000 101", 9, 1),
    ("0000 01", RUN_ESCAPE, 0),
    ("0010 0110", 0, 5),
    ("0010 0001", 0, 6),
    ("0010 0101", 1, 3),
    ("0010 0100", 3, 2),
    ("0010 0111", 10, 1),
    ("0010 0011", 11, 1),
    ("0010 0010", 12, 1),
    ("0010 0000", 13, 1),
    ("0000 0010 10", 0, 7),
    ("0000 0011 00", 1, 4),
    ("0000 0010 11", 2, 3),
    ("0000 0011 11", 4, 2),
    ("0000 0010 01", 5, 2),
    ("0000 0011 10", 14, 1),
    ("0000 0011 01", 15, 1),
    ("0000 0010 00", 16, 1),
    ("0000 0001 1101", 0, 8),
    ("0000 0001 1000", 0, 9),
    ("0000 0001 0011", 0, 10),
    ("0000 0001 0000", 0, 11),
    ("0000 0001 1011", 1, 5),
    ("0000 0001 0100", 2, 4),
    ("0000 0001 1100", 3, 3),
    ("0000 0001 0010", 4, 3),
    ("0000 0001 1110", 6, 2),
    ("0000 0001 0101", 7, 2),
    ("0000 0001 0001", 8, 2),
    ("0000 0001 1111", 17, 1),
    ("0000 0001 1010", 18, 1),
    ("0000 0001 1001", 19, 1),
    ("0000 0001 0111", 20, 1),
    ("0000 0001 0110", 21, 1),
    ("0000 0000 1101 0", 0, 12),
    ("0000 0000 1100 1", 0, 13),
    ("0000 0000 1100 0", 0, 14),
    ("0000 0000 1011 1", 0, 15),
];

// B.15.
const COEF_ONE: &[CoefCode] = &[
    ("0110", RUN_EOB, 0),
    ("10", 0, 1),
    ("010", 1, 1),
    ("110", 0, 2),
    ("0010 1", 2, 1),
    ("0111", 0, 3),
    ("0011 1", 3, 1),
    ("0001 10", 4, 1),
    ("0011 0", 1, 2),
    ("0001 11", 5, 1),
    ("0000 110", 6, 1),
    ("0000 100", 7, 1),
    ("1110 0", 0, 4),
    ("0000 111", 2, 2),
    ("0000 101", 8, 1),
    ("1111 000", 9, 1),
    ("0000 01", RUN_ESCAPE, 0),
    ("1110 1", 0, 5),
    ("0001 01", 0, 6),
    ("1111 001", 1, 3),
    ("0010 0110", 3, 2),
    ("1111 010", 10, 1),
    ("0010 0001", 11, 1),
    ("0010 0101", 12, 1),
    ("0010 0100", 13, 1),
    ("0001 00", 0, 7),
    ("0010 0111", 1, 4),
    ("1111 1100", 2, 3),
    ("1111 1101", 4, 2),
    ("0000 0010 0", 5, 2),
    ("0000 0010 1", 14, 1),
    ("0000 0011 1", 15, 1),
    ("0000 0011 01", 16, 1),
    ("1111 011", 0, 8),
    ("1111 100", 0, 9),
    ("0010 0011", 0, 10),
    ("0010 0010", 0, 11),
    ("0010 0000", 1, 5),
    ("0000 0011 00", 2, 4),
    ("0000 0001 1100", 3, 3),
    ("0000 0001 0010", 4, 3),
    ("0000 0001 1110", 6, 2),
    ("0000 0001 0101", 7, 2),
    ("0000 0001 0001", 8, 2),
    ("0000 0001 1111", 17, 1),
    ("0000 0001 1010", 18, 1),
    ("0000 0001 1001", 19, 1),
    ("0000 0001 0111", 20, 1),
    ("0000 0001 0110", 21, 1),
    ("1111 1010", 0, 12),
    ("1111 1011", 0, 13),
    ("1111 1110", 0, 14),
    ("1111 1111", 0, 15),
];

/// Two-level decode table for a coefficient code book.
///
/// `short` is indexed by the next 8 bits. Every code longer than 8 bits
/// starts with six zero bits, so `long` is indexed by bits 6 to 15 of the
/// next 16.
pub(crate) struct CoefTable {
    pub short: [Coef; 256],
    pub long: [Coef; 1024],
}

const fn fill_coef(table: &mut CoefTable, codes: &[CoefCode]) {
    let mut k = 0;
    while k < codes.len() {
        let (code, len) = parse(codes[k].0);
        let entry = Coef {
            run: codes[k].1,
            level: codes[k].2,
            len: len as u8,
        };
        if len <= 8 {
            let shift = 8 - len;
            let first = (code << shift) as usize;
            let mut j = 0;
            while j < (1usize << shift) {
                table.short[first + j] = entry;
                j += 1;
            }
        } else {
            let shift = 16 - len;
            let first = ((code << shift) & 0x3ff) as usize;
            let mut j = 0;
            while j < (1usize << shift) {
                table.long[first + j] = entry;
                j += 1;
            }
        }
        k += 1;
    }
}

const fn coef_table(codes: &[CoefCode]) -> CoefTable {
    let mut table = CoefTable {
        short: [COEF_NONE; 256],
        long: [COEF_NONE; 1024],
    };
    fill_coef(&mut table, codes);
    fill_coef(&mut table, COEF_SHARED_TAIL);
    table
}

pub(crate) static COEF_TABLE_ZERO: CoefTable = coef_table(COEF_ZERO);
pub(crate) static COEF_TABLE_ONE: CoefTable = coef_table(COEF_ONE);

/// Code and length (without sign) for `[run][level]`, run 0..=31 and level
/// 1..=40. Length 0 means the pair needs an escape.
pub(crate) type CoefEncode = [[(u16, u8); 41]; 32];

const fn fill_encode(table: &mut CoefEncode, codes: &[CoefCode]) {
    let mut k = 0;
    while k < codes.len() {
        let (code, len) = parse(codes[k].0);
        let run = codes[k].1;
        let level = codes[k].2;
        if run < 32 && level >= 1 && level <= 40 {
            table[run as usize][level as usize] = (code as u16, len as u8);
        }
        k += 1;
    }
}

const fn coef_encode(codes: &[CoefCode]) -> CoefEncode {
    let mut table = [[(0u16, 0u8); 41]; 32];
    fill_encode(&mut table, codes);
    fill_encode(&mut table, COEF_SHARED_TAIL);
    table
}

pub(crate) static COEF_ENC_ZERO: CoefEncode = coef_encode(COEF_ZERO);
pub(crate) static COEF_ENC_ONE: CoefEncode = coef_encode(COEF_ONE);
/// End of Block in B.14 and in B.15.
pub(crate) const EOB_ZERO: (u32, u32) = parse("10");
pub(crate) const EOB_ONE: (u32, u32) = parse("0110");
pub(crate) const ESCAPE: (u32, u32) = parse("0000 01");

/// `scan[0][v][u]` from Figure 7-2: the scan index at each raster position.
const ZIGZAG_INDEX: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, //
    2, 4, 7, 13, 16, 26, 29, 42, //
    3, 8, 12, 17, 25, 30, 41, 43, //
    9, 11, 18, 24, 31, 40, 44, 53, //
    10, 19, 23, 32, 39, 45, 52, 54, //
    20, 22, 33, 38, 46, 51, 55, 60, //
    21, 34, 37, 47, 50, 56, 59, 61, //
    35, 36, 48, 49, 57, 58, 62, 63,
];

/// `scan[1][v][u]` from Figure 7-3.
const ALTERNATE_INDEX: [u8; 64] = [
    0, 4, 6, 20, 22, 36, 38, 52, //
    1, 5, 7, 21, 23, 37, 39, 53, //
    2, 8, 19, 24, 34, 40, 50, 54, //
    3, 9, 18, 25, 35, 41, 51, 55, //
    10, 17, 26, 30, 42, 46, 56, 60, //
    11, 16, 27, 31, 43, 47, 57, 61, //
    12, 15, 28, 32, 44, 48, 58, 62, //
    13, 14, 29, 33, 45, 49, 59, 63,
];

const fn invert(index: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut raster = 0;
    while raster < 64 {
        out[index[raster] as usize] = raster as u8;
        raster += 1;
    }
    out
}

/// Raster position (`v * 8 + u`) of scan index `n`, for each `alternate_scan`.
pub(crate) static SCAN: [[u8; 64]; 2] = [invert(&ZIGZAG_INDEX), invert(&ALTERNATE_INDEX)];

/// The default intra matrix of 6.3.11, in raster order.
pub(crate) const DEFAULT_INTRA: [u8; 64] = [
    8, 16, 19, 22, 26, 27, 29, 34, //
    16, 16, 22, 24, 27, 29, 34, 37, //
    19, 22, 26, 27, 29, 34, 34, 38, //
    22, 22, 26, 27, 29, 34, 37, 40, //
    22, 26, 27, 29, 32, 35, 40, 48, //
    26, 27, 29, 32, 35, 40, 48, 58, //
    26, 27, 29, 34, 38, 46, 56, 69, //
    27, 29, 35, 38, 46, 56, 69, 83,
];

pub(crate) const DEFAULT_NON_INTRA: [u8; 64] = [16; 64];

/// Table 7-6, `quantiser_scale` for `q_scale_type` 1.
pub(crate) const NON_LINEAR_SCALE: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64,
    72, 80, 88, 96, 104, 112,
];

/// Table 6-4, `frame_rate_value` as a ratio, by `frame_rate_code`.
pub(crate) const FRAME_RATES: [(u32, u32); 9] = [
    (0, 0),
    (24_000, 1001),
    (24, 1),
    (25, 1),
    (30_000, 1001),
    (30, 1),
    (50, 1),
    (60_000, 1001),
    (60, 1),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(table: &[Vlc], bits: u32, code: &str) -> Vlc {
        let (value, len) = parse(code);
        table[(value << (bits - len)) as usize]
    }

    fn check_flat(table: &[Vlc], bits: u32, codes: &[(&str, i16)]) {
        for &(code, value) in codes {
            let entry = lookup(table, bits, code);
            assert_eq!(entry.value, value, "{code}");
            assert_eq!(u32::from(entry.len), parse(code).1, "{code}");
        }
    }

    /// No code is a prefix of another, so a flat table can hold them all.
    fn prefix_free(codes: &[&str]) {
        let parsed: Vec<(u32, u32)> = codes.iter().map(|code| parse(code)).collect();
        for (i, &(a, la)) in parsed.iter().enumerate() {
            for (j, &(b, lb)) in parsed.iter().enumerate() {
                if i != j && la <= lb {
                    assert_ne!(b >> (lb - la), a, "{} prefixes {}", codes[i], codes[j]);
                }
            }
        }
    }

    #[test]
    fn every_flat_table_decodes_its_own_codes() {
        check_flat(&MB_ADDR, MB_ADDR_BITS, MB_ADDR_CODES);
        check_flat(&MB_TYPE_I, MB_TYPE_BITS, MB_TYPE_I_CODES);
        check_flat(&MB_TYPE_P, MB_TYPE_BITS, MB_TYPE_P_CODES);
        check_flat(&MB_TYPE_B, MB_TYPE_BITS, MB_TYPE_B_CODES);
        check_flat(&CBP, CBP_BITS, CBP_CODES);
        check_flat(&MOTION, MOTION_BITS, MOTION_CODES);
        check_flat(&DMV, 2, DMV_CODES);
        check_flat(&DC_LUMA, DC_BITS, DC_LUMA_CODES);
        check_flat(&DC_CHROMA, DC_BITS, DC_CHROMA_CODES);
        for codes in [
            MB_ADDR_CODES,
            MB_TYPE_P_CODES,
            MB_TYPE_B_CODES,
            CBP_CODES,
            MOTION_CODES,
            DC_LUMA_CODES,
            DC_CHROMA_CODES,
        ] {
            prefix_free(&codes.iter().map(|code| code.0).collect::<Vec<_>>());
        }
    }

    #[test]
    fn the_tables_have_the_sizes_annex_b_prints() {
        assert_eq!(MB_ADDR_CODES.len(), 35);
        assert_eq!(CBP_CODES.len(), 64);
        assert_eq!(MOTION_CODES.len(), 33);
        let cbps: std::collections::BTreeSet<i16> = CBP_CODES.iter().map(|code| code.1).collect();
        assert_eq!(cbps.len(), 64);
        // B.14 and B.15 each list 111 codes, End of Block, and Escape.
        assert_eq!(COEF_ZERO.len() + COEF_SHARED_TAIL.len(), 113);
        assert_eq!(COEF_ONE.len() + COEF_SHARED_TAIL.len(), 113);
    }

    fn decode_coef(table: &CoefTable, code: &str) -> Coef {
        let (value, len) = parse(code);
        let window = value << (16 - len);
        let short = table.short[(window >> 8) as usize];
        if short.len > 0 {
            short
        } else {
            table.long[(window & 0x3ff) as usize]
        }
    }

    #[test]
    fn coefficient_tables_decode_every_code_and_are_prefix_free() {
        for (table, codes) in [(&COEF_TABLE_ZERO, COEF_ZERO), (&COEF_TABLE_ONE, COEF_ONE)] {
            let all: Vec<CoefCode> = codes.iter().chain(COEF_SHARED_TAIL).copied().collect();
            prefix_free(&all.iter().map(|code| code.0).collect::<Vec<_>>());
            let mut pairs = std::collections::BTreeSet::new();
            for (code, run, level) in &all {
                let entry = decode_coef(table, code);
                assert_eq!(
                    (entry.run, entry.level, u32::from(entry.len)),
                    (*run, *level, parse(code).1),
                    "{code}"
                );
                assert!(pairs.insert((*run, *level)), "pair {run} {level} twice");
                if parse(code).1 > 8 {
                    assert!(parse(code).0 >> (parse(code).1 - 6) == 0, "{code}");
                }
            }
        }
    }

    #[test]
    fn b14_and_b15_code_the_same_run_level_pairs() {
        let pairs = |codes: &[CoefCode]| -> std::collections::BTreeSet<(u8, u8)> {
            codes
                .iter()
                .chain(COEF_SHARED_TAIL)
                .map(|code| (code.1, code.2))
                .collect()
        };
        assert_eq!(pairs(COEF_ZERO), pairs(COEF_ONE));
    }

    #[test]
    fn encode_tables_invert_the_decode_tables() {
        for run in 0..32u8 {
            for level in 1..=40u8 {
                for (enc, table) in [
                    (&COEF_ENC_ZERO, &COEF_TABLE_ZERO),
                    (&COEF_ENC_ONE, &COEF_TABLE_ONE),
                ] {
                    let (code, len) = enc[usize::from(run)][usize::from(level)];
                    if len == 0 {
                        continue;
                    }
                    let text: String = (0..len)
                        .rev()
                        .map(|bit| if code >> bit & 1 == 1 { '1' } else { '0' })
                        .collect();
                    let entry = decode_coef(table, &text);
                    assert_eq!((entry.run, entry.level), (run, level));
                }
            }
        }
        assert_eq!(MOTION_ENC[16], (1, 1));
        assert_eq!(MB_ADDR_ENC[1], (1, 1));
        assert_eq!(MB_ADDR_ENC[33], (parse("0000 0011 000").0, 11));
        assert_eq!(MOTION_ENC[0], (parse("0000 0011 001").0, 11));
        assert_eq!(DC_CHROMA_ENC[11], (parse("1111 1111 11").0, 10));
    }

    #[test]
    fn scans_are_permutations_and_start_at_dc() {
        for scan in &SCAN {
            let mut seen = [false; 64];
            for &raster in scan {
                seen[usize::from(raster)] = true;
            }
            assert!(seen.iter().all(|&hit| hit));
            assert_eq!(scan[0], 0);
            assert_eq!(scan[63], 63);
        }
        // Zigzag: 0 1 8 16 9 2 ...
        assert_eq!(&SCAN[0][..6], &[0, 1, 8, 16, 9, 2]);
        // Alternate: down the first column first.
        assert_eq!(&SCAN[1][..5], &[0, 8, 16, 24, 1]);
    }
}
