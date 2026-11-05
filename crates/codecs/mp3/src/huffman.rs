//! Huffman decode and encode tables built from [`crate::tables`].

use std::sync::OnceLock;

use crate::bits::BitReader;
use crate::tables::{QUAD_A, big_table};

/// One prefix code, decodable by its left-justified codewords.
#[derive(Debug)]
pub(crate) struct Code {
    /// Left-justified codeword of each entry, ascending.
    starts: Vec<u32>,
    lens: Vec<u8>,
    syms: Vec<u8>,
    /// Entry range for each 8-bit prefix.
    first: Vec<(u16, u16)>,
}

impl Code {
    /// Build from `(length, symbol)` pairs in ascending code order.
    fn canonical(entries: &[(u8, u8)]) -> Self {
        let mut starts = Vec::with_capacity(entries.len());
        let mut lens = Vec::with_capacity(entries.len());
        let mut syms = Vec::with_capacity(entries.len());
        let mut code: u64 = 0;
        for &(len, sym) in entries {
            starts.push(code as u32);
            lens.push(len);
            syms.push(sym);
            code += 1u64 << (32 - u32::from(len));
        }
        Self::finish(starts, lens, syms)
    }

    /// Build from `(length, codeword)` indexed by symbol.
    fn explicit(table: &[(u8, u8)]) -> Self {
        let mut entries: Vec<(u32, u8, u8)> = table
            .iter()
            .enumerate()
            .map(|(sym, &(len, code))| {
                let left = u32::from(code) << (32 - u32::from(len));
                (left, len, sym as u8)
            })
            .collect();
        entries.sort_unstable();
        Self::finish(
            entries.iter().map(|entry| entry.0).collect(),
            entries.iter().map(|entry| entry.1).collect(),
            entries.iter().map(|entry| entry.2).collect(),
        )
    }

    fn finish(starts: Vec<u32>, lens: Vec<u8>, syms: Vec<u8>) -> Self {
        let mut first = Vec::with_capacity(256);
        for prefix in 0u32..256 {
            let low = prefix << 24;
            let high = low | 0x00FF_FFFF;
            first.push((
                entry_at(&starts, low) as u16,
                entry_at(&starts, high) as u16,
            ));
        }
        Self {
            starts,
            lens,
            syms,
            first,
        }
    }

    /// `(length, codeword)` by symbol, right-aligned; length 0 is absent.
    #[cfg(test)]
    pub(crate) fn by_symbol(&self) -> Vec<(u8, u32)> {
        let mut table = vec![(0u8, 0u32); 256];
        for ((&start, &len), &sym) in self.starts.iter().zip(&self.lens).zip(&self.syms) {
            table[usize::from(sym)] = (len, start >> (32 - u32::from(len)));
        }
        table
    }

    /// Decode one symbol. `None` when the codeword passes the reader's end.
    pub(crate) fn decode(&self, reader: &mut BitReader<'_>) -> Option<u8> {
        let peek = reader.peek32();
        let (low, high) = self.first[(peek >> 24) as usize];
        let index = if low == high {
            usize::from(low)
        } else {
            let low = usize::from(low);
            let high = usize::from(high);
            low + entry_at(&self.starts[low..=high], peek)
        };
        let len = usize::from(self.lens[index]);
        if reader.left() < len {
            return None;
        }
        reader.skip(len, "huffman").ok()?;
        Some(self.syms[index])
    }
}

/// Index of the last start that is at most `value`.
fn entry_at(starts: &[u32], value: u32) -> usize {
    starts
        .partition_point(|start| *start <= value)
        .saturating_sub(1)
}

/// Every Layer III Huffman table.
#[derive(Debug)]
pub(crate) struct Tables {
    big: Vec<Option<Code>>,
    pub(crate) quad_a: Code,
}

impl Tables {
    /// The big-value code of table index 1..=15, 16, or 24.
    pub(crate) fn big(&self, index: u8) -> Option<&Code> {
        self.big.get(usize::from(index)).and_then(Option::as_ref)
    }
}

pub(crate) fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut big = Vec::with_capacity(25);
        for index in 0u8..25 {
            let entries = big_table(index);
            big.push(if entries.is_empty() {
                None
            } else {
                Some(Code::canonical(entries))
            });
        }
        Tables {
            big,
            quad_a: Code::explicit(&QUAD_A),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{Code, tables};
    use crate::bits::{BitReader, BitWriter};
    use crate::tables::big_table;

    fn check_complete(code: &Code) {
        let mut kraft = 0f64;
        for (index, len) in code.lens.iter().enumerate() {
            kraft += 2f64.powi(-i32::from(*len));
            if index > 0 {
                assert!(code.starts[index] > code.starts[index - 1]);
            }
        }
        assert!((kraft - 1.0).abs() < 1e-12, "kraft sum {kraft}");
    }

    #[test]
    fn every_table_is_prefix_free_and_complete() {
        let all = tables();
        for index in [1u8, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24] {
            let code = all.big(index).expect("table");
            check_complete(code);
            let size = match index {
                1 => 2,
                2 | 3 => 3,
                5 | 6 => 4,
                7..=9 => 6,
                10..=12 => 8,
                _ => 16,
            };
            assert_eq!(big_table(index).len(), size * size);
            for x in 0..size {
                for y in 0..size {
                    assert!(code.by_symbol()[x * 16 + y].0 > 0, "table {index} {x} {y}");
                }
            }
        }
        check_complete(&all.quad_a);
        assert!(all.big(4).is_none());
        assert!(all.big(14).is_none());
    }

    #[test]
    fn table_one_matches_the_standard_codewords() {
        // ISO/IEC 11172-3 Table 3-B.7, table 1: 00 -> 1, 01 -> 001, 10 -> 01, 11 -> 000.
        let code = tables().big(1).expect("table 1");
        assert_eq!(code.by_symbol()[0x00], (1, 0b1));
        assert_eq!(code.by_symbol()[0x01], (3, 0b001));
        assert_eq!(code.by_symbol()[0x10], (2, 0b01));
        assert_eq!(code.by_symbol()[0x11], (3, 0b000));
    }

    #[test]
    fn every_codeword_decodes_to_its_symbol() {
        let all = tables();
        for index in [1u8, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24] {
            let code = all.big(index).expect("table");
            let mut writer = BitWriter::new();
            let mut sent = Vec::new();
            for (sym, &(len, word)) in code.by_symbol().iter().enumerate() {
                if len > 0 {
                    writer.put(word, u32::from(len));
                    sent.push(sym as u8);
                }
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            for sym in sent {
                assert_eq!(code.decode(&mut reader), Some(sym), "table {index}");
            }
        }
        let mut writer = BitWriter::new();
        for &(len, word) in &all.quad_a.by_symbol()[..16] {
            writer.put(word, u32::from(len));
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        for sym in 0u8..16 {
            assert_eq!(all.quad_a.decode(&mut reader), Some(sym));
        }
    }

    #[test]
    fn a_codeword_cut_by_the_window_is_refused() {
        let code = tables().big(13).expect("table 13");
        let (len, word) = code.by_symbol()[0xFE];
        assert_eq!(len, 19);
        let mut writer = BitWriter::new();
        writer.put(word, 19);
        let bytes = writer.finish();
        let mut reader = BitReader::window(&bytes, 0, 18);
        assert_eq!(code.decode(&mut reader), None);
        assert_eq!(reader.pos(), 0);
    }
}
