//! Spectral and scale-factor Huffman codes.
//!
//! ISO/IEC 14496-3:2009 clause 4.A. Codewords are stored low-bits-first in
//! the tables and transmitted most-significant bit first. Unsigned books
//! emit a sign bit after the codeword for every non-zero value, before any
//! escape bits. Book 11 uses the value 16 as the escape flag.

use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BitSink};
use crate::huff_tables::{
    BOOK1_CODE, BOOK1_LEN, BOOK2_CODE, BOOK2_LEN, BOOK3_CODE, BOOK3_LEN, BOOK4_CODE, BOOK4_LEN,
    BOOK5_CODE, BOOK5_LEN, BOOK6_CODE, BOOK6_LEN, BOOK7_CODE, BOOK7_LEN, BOOK8_CODE, BOOK8_LEN,
    BOOK9_CODE, BOOK9_LEN, BOOK10_CODE, BOOK10_LEN, BOOK11_CODE, BOOK11_LEN, SF_CODE, SF_LEN,
};

struct Node {
    child: [i32; 2],
    symbol: i32,
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn read(&self, bits: &mut BitReader<'_>) -> Result<usize, CodecError> {
        let mut node = 0usize;
        loop {
            if self.nodes[node].symbol >= 0 {
                return Ok(self.nodes[node].symbol as usize);
            }
            let bit = bits.read(1)? as usize;
            let next = self.nodes[node].child[bit];
            if next < 0 {
                return Err(CodecError::Malformed {
                    at_bit: bits.position() as u64,
                    what: "huffman code",
                });
            }
            node = next as usize;
        }
    }
}

fn build(codes: &[u32], lens: &[u8]) -> Result<Tree, CodecError> {
    let mut nodes = vec![Node {
        child: [-1, -1],
        symbol: -1,
    }];
    for (symbol, (&code, &len)) in codes.iter().zip(lens).enumerate() {
        if len == 0 || len > 32 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "huffman length",
            });
        }
        let mut node = 0usize;
        for shift in (0..len).rev() {
            if nodes[node].symbol >= 0 {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "huffman prefix",
                });
            }
            let bit = ((code >> shift) & 1) as usize;
            if nodes[node].child[bit] < 0 {
                let id = nodes.len() as i32;
                nodes.push(Node {
                    child: [-1, -1],
                    symbol: -1,
                });
                nodes[node].child[bit] = id;
            }
            node = nodes[node].child[bit] as usize;
        }
        if nodes[node].symbol >= 0 || nodes[node].child[0] >= 0 || nodes[node].child[1] >= 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "huffman prefix",
            });
        }
        nodes[node].symbol = symbol as i32;
    }
    Ok(Tree { nodes })
}

struct Books {
    spectral: [Tree; 11],
    scale: Tree,
}

fn books() -> Result<&'static Books, CodecError> {
    static CELL: OnceLock<Result<Books, CodecError>> = OnceLock::new();
    match CELL.get_or_init(|| {
        Ok(Books {
            spectral: [
                build(&BOOK1_CODE, &BOOK1_LEN)?,
                build(&BOOK2_CODE, &BOOK2_LEN)?,
                build(&BOOK3_CODE, &BOOK3_LEN)?,
                build(&BOOK4_CODE, &BOOK4_LEN)?,
                build(&BOOK5_CODE, &BOOK5_LEN)?,
                build(&BOOK6_CODE, &BOOK6_LEN)?,
                build(&BOOK7_CODE, &BOOK7_LEN)?,
                build(&BOOK8_CODE, &BOOK8_LEN)?,
                build(&BOOK9_CODE, &BOOK9_LEN)?,
                build(&BOOK10_CODE, &BOOK10_LEN)?,
                build(&BOOK11_CODE, &BOOK11_LEN)?,
            ],
            scale: build(&SF_CODE, &SF_LEN)?,
        })
    }) {
        Ok(books) => Ok(books),
        Err(err) => Err(*err),
    }
}

pub fn read_spectral(bits: &mut BitReader<'_>, book: usize) -> Result<usize, CodecError> {
    let books = books()?;
    let Some(tree) = books.spectral.get(book) else {
        return Err(CodecError::Malformed {
            at_bit: bits.position() as u64,
            what: "codebook",
        });
    };
    tree.read(bits)
}

pub fn read_scale_delta(bits: &mut BitReader<'_>) -> Result<i32, CodecError> {
    let symbol = books()?.scale.read(bits)?;
    Ok(symbol as i32 - 60)
}

pub fn write_code<S: BitSink>(sink: &mut S, code: u32, len: u8) -> Result<(), CodecError> {
    sink.write_bits(code, u32::from(len))
}

fn table(book: usize) -> Option<(&'static [u32], &'static [u8])> {
    Some(match book {
        0 => (&BOOK1_CODE, &BOOK1_LEN),
        1 => (&BOOK2_CODE, &BOOK2_LEN),
        2 => (&BOOK3_CODE, &BOOK3_LEN),
        3 => (&BOOK4_CODE, &BOOK4_LEN),
        4 => (&BOOK5_CODE, &BOOK5_LEN),
        5 => (&BOOK6_CODE, &BOOK6_LEN),
        6 => (&BOOK7_CODE, &BOOK7_LEN),
        7 => (&BOOK8_CODE, &BOOK8_LEN),
        8 => (&BOOK9_CODE, &BOOK9_LEN),
        9 => (&BOOK10_CODE, &BOOK10_LEN),
        10 => (&BOOK11_CODE, &BOOK11_LEN),
        _ => return None,
    })
}

pub fn spectral_code(book: usize, symbol: usize) -> Result<(u32, u8), CodecError> {
    let Some((codes, lens)) = table(book) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "codebook",
        });
    };
    let Some((&code, &len)) = codes.get(symbol).zip(lens.get(symbol)) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "huffman symbol",
        });
    };
    Ok((code, len))
}

pub fn scale_code(delta: i32) -> Result<(u32, u8), CodecError> {
    if !(-60..=60).contains(&delta) {
        return Err(CodecError::Limit {
            what: "scale factor delta",
        });
    }
    let symbol = (delta + 60) as usize;
    Ok((SF_CODE[symbol], SF_LEN[symbol]))
}

pub fn quad_digits(index: usize) -> [i32; 4] {
    let mut n = index;
    let mut out = [0; 4];
    for slot in out.iter_mut().rev() {
        *slot = (n % 3) as i32;
        n /= 3;
    }
    out
}

pub fn quad_index(values: [i32; 4]) -> usize {
    values.iter().fold(0usize, |acc, v| acc * 3 + *v as usize)
}

pub fn pair_digits(index: usize, modulus: i32) -> [i32; 2] {
    let m = modulus as usize;
    [(index / m) as i32, (index % m) as i32]
}

pub fn pair_index(a: i32, b: i32, modulus: i32) -> usize {
    (a * modulus + b) as usize
}

/// Signed 2-tuples in books 5 and 6. Index 0 is -4.
pub fn signed_pair(index: usize) -> [i32; 2] {
    let raw = pair_digits(index, 9);
    [raw[0] - 4, raw[1] - 4]
}

pub fn signed_pair_index(a: i32, b: i32) -> usize {
    pair_index(a + 4, b + 4, 9)
}

/// Escape extension for an absolute quantized value of at least 16.
/// The unary prefix length must stay below 9, so the value is at most 8191.
pub fn read_escape(bits: &mut BitReader<'_>) -> Result<i32, CodecError> {
    let mut ones = 0u32;
    while bits.read(1)? == 1 {
        ones += 1;
        if ones >= 9 {
            return Err(CodecError::Limit {
                what: "escape prefix",
            });
        }
    }
    let width = ones + 4;
    let offset = bits.read(width)?;
    let base = 1i32 << width;
    Ok(base + offset as i32)
}

pub fn write_escape<S: BitSink>(sink: &mut S, magnitude: u32) -> Result<(), CodecError> {
    if !(16..=8191).contains(&magnitude) {
        return Err(CodecError::Limit {
            what: "escape value",
        });
    }
    let log = 31 - magnitude.leading_zeros();
    let ones = log - 4;
    if ones > 0 {
        sink.write_bits((1u32 << ones) - 1, ones)?;
    }
    sink.write_bits(0, 1)?;
    let offset = magnitude - (1u32 << log);
    sink.write_bits(offset, log)
}

#[cfg(test)]
mod tests {
    use super::{
        quad_digits, quad_index, read_escape, read_scale_delta, read_spectral, scale_code,
        signed_pair, signed_pair_index, spectral_code, write_code, write_escape,
    };
    use crate::bits::{BitReader, BitWriter};
    use crate::huff_tables::{
        BOOK1_LEN, BOOK2_LEN, BOOK3_LEN, BOOK4_LEN, BOOK5_LEN, BOOK6_LEN, BOOK7_LEN, BOOK8_LEN,
        BOOK9_LEN, BOOK10_LEN, BOOK11_LEN, SF_LEN,
    };

    fn round_trip_book(book: usize, symbols: usize) {
        for symbol in 0..symbols {
            let (code, len) = spectral_code(book, symbol).unwrap();
            let mut w = BitWriter::with_cap(8);
            write_code(&mut w, code, len).unwrap();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            let got = read_spectral(&mut r, book).unwrap();
            assert_eq!(got, symbol, "book {book} symbol {symbol}");
        }
    }

    #[test]
    fn every_spectral_symbol_round_trips() {
        let counts = [
            BOOK1_LEN.len(),
            BOOK2_LEN.len(),
            BOOK3_LEN.len(),
            BOOK4_LEN.len(),
            BOOK5_LEN.len(),
            BOOK6_LEN.len(),
            BOOK7_LEN.len(),
            BOOK8_LEN.len(),
            BOOK9_LEN.len(),
            BOOK10_LEN.len(),
            BOOK11_LEN.len(),
        ];
        assert_eq!(counts, [81, 81, 81, 81, 81, 81, 64, 64, 169, 169, 289]);
        for (book, count) in counts.iter().enumerate() {
            round_trip_book(book, *count);
        }
    }

    #[test]
    fn scale_factor_deltas_round_trip() {
        assert_eq!(SF_LEN.len(), 121);
        for delta in -60..=60 {
            let (code, len) = scale_code(delta).unwrap();
            let mut w = BitWriter::with_cap(8);
            write_code(&mut w, code, len).unwrap();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_scale_delta(&mut r).unwrap(), delta);
        }
    }

    #[test]
    fn quad_index_is_base_three() {
        assert_eq!(quad_digits(0), [0, 0, 0, 0]);
        assert_eq!(quad_digits(1), [0, 0, 0, 1]);
        assert_eq!(quad_digits(3), [0, 0, 1, 0]);
        assert_eq!(quad_index([2, 2, 2, 2]), 80);
    }

    #[test]
    fn signed_pair_zero_is_the_middle_index() {
        assert_eq!(signed_pair(signed_pair_index(0, 0)), [0, 0]);
        assert_eq!(signed_pair(signed_pair_index(-4, 4)), [-4, 4]);
    }

    #[test]
    fn escape_round_trip_covers_the_flag_and_the_limit() {
        for value in [16u32, 17, 31, 32, 1000, 8191] {
            let mut w = BitWriter::with_cap(8);
            write_escape(&mut w, value).unwrap();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_escape(&mut r).unwrap(), value as i32);
        }
    }

    #[test]
    fn a_truncated_code_is_not_a_symbol() {
        let mut r = BitReader::new(&[0x00]);
        // Book 1's shortest code is not zero-length; a single zero byte may
        // or may not finish a symbol. Forcing a short buffer after a prefix
        // that needs more bits must truncate.
        let _ = r.read(0);
        let mut tiny = BitReader::new(&[]);
        assert!(read_spectral(&mut tiny, 0).is_err());
    }
}
