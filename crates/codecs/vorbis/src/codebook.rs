//! Codebooks (Vorbis I section 3): entry lengths, the Huffman tree they
//! imply, and the optional vector lookup table.

use crate::bits::{BitReader, BitWriter, ilog};
use crate::header::malformed;
use cutlass_decode::CodecError;

const SYNC: u32 = 0x56_43_42;
/// The spec's limit on entries (24-bit field).
pub(crate) const MAX_ENTRIES: usize = 1 << 24;
/// Entries over all codebooks of one setup. libvorbis setups use a few
/// thousand; this bounds what a hostile header can make the decoder build.
pub(crate) const MAX_TOTAL_ENTRIES: usize = 1 << 22;
/// Vector values over all codebooks of one setup (`entries * dimensions`).
pub(crate) const MAX_TOTAL_VALUES: usize = 1 << 22;
/// Bits the decode table looks up at once.
const TABLE_BITS: u32 = 10;
/// A table slot whose codeword is longer than [`TABLE_BITS`]: the low bits
/// name the tree node to continue from.
const LONG: u32 = 0xff00_0000;
const LEAF: u32 = 0x8000_0000;
const NONE: u32 = u32::MAX;

/// One codebook, ready to decode.
#[derive(Clone, Debug)]
pub(crate) struct Codebook {
    pub dimensions: usize,
    pub entries: usize,
    /// Codeword length per entry, 0 for an unused entry.
    pub lengths: Vec<u8>,
    /// `dimensions` values per entry, when the book has a lookup table.
    pub values: Option<Vec<f32>>,
    /// The single used entry of a single-entry book.
    single: Option<u32>,
    /// Indexed by the next [`TABLE_BITS`] bits: entry and length, `LONG`
    /// plus a tree node, or `NONE`.
    table: Vec<u32>,
    /// Children of each tree node: another node, or `LEAF | entry`.
    tree: Vec<[u32; 2]>,
    table_bits: u32,
}

/// What [`Codebook::parse`] reads besides the decode structures, kept so
/// the encoder can write the same book back.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lookup {
    pub kind: u8,
    pub minimum: u32,
    pub delta: u32,
    pub value_bits: u8,
    pub sequence: bool,
    pub multiplicands: Vec<u32>,
}

/// A codebook as written in a setup header.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BookSpec {
    pub dimensions: usize,
    pub lengths: Vec<u8>,
    pub lookup: Option<Lookup>,
}

/// Running totals that bound one setup's codebooks.
#[derive(Debug, Default)]
pub(crate) struct Budget {
    entries: usize,
    values: usize,
}

impl Codebook {
    /// Reads one codebook from a setup header.
    pub(crate) fn parse(
        reader: &mut BitReader<'_>,
        budget: &mut Budget,
    ) -> Result<Self, CodecError> {
        let spec = BookSpec::parse(reader, budget)?;
        Self::build(&spec, budget)
    }

    /// Builds the decode structures for a book.
    pub(crate) fn build(spec: &BookSpec, budget: &mut Budget) -> Result<Self, CodecError> {
        let entries = spec.lengths.len();
        let dimensions = spec.dimensions;
        let used = spec.lengths.iter().filter(|&&len| len > 0).count();
        let mut book = Self {
            dimensions,
            entries,
            lengths: spec.lengths.clone(),
            values: None,
            single: None,
            table: Vec::new(),
            tree: Vec::new(),
            table_bits: 0,
        };
        if used == 1 {
            // Errata 20150226: the one used entry has length 1 and reading
            // it sinks one bit, whichever value it has.
            let Some(entry) = spec.lengths.iter().position(|&len| len > 0) else {
                return Err(malformed(0, "codebook lengths"));
            };
            if spec.lengths[entry] != 1 {
                return Err(malformed(0, "single-entry codebook length"));
            }
            book.single = Some(entry as u32);
        } else if used > 1 {
            let codes = assign_codewords(&spec.lengths)?;
            book.build_tables(&codes);
        }
        if let Some(lookup) = &spec.lookup {
            budget.values = budget
                .values
                .checked_add(entries.saturating_mul(dimensions))
                .filter(|total| *total <= MAX_TOTAL_VALUES)
                .ok_or(CodecError::Limit {
                    what: "codebook vector values",
                })?;
            book.values = Some(unpack_values(lookup, entries, dimensions));
        }
        Ok(book)
    }

    fn build_tables(&mut self, codes: &[(u32, u8)]) {
        let max_len = self.lengths.iter().copied().max().unwrap_or(0);
        let bits = u32::from(max_len).min(TABLE_BITS);
        self.table_bits = bits;
        self.table = vec![NONE; 1 << bits];
        self.tree = vec![[NONE, NONE]];
        for (entry, &(code, len)) in codes.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len32 = u32::from(len);
            // Tree: walk from the root, first bit read = top bit of code.
            let mut node = 0_usize;
            for depth in 0..len32 {
                let bit = ((code >> (len32 - 1 - depth)) & 1) as usize;
                if depth + 1 == len32 {
                    self.tree[node][bit] = LEAF | entry as u32;
                } else {
                    let next = self.tree[node][bit];
                    if next == NONE {
                        self.tree.push([NONE, NONE]);
                        let index = (self.tree.len() - 1) as u32;
                        self.tree[node][bit] = index;
                        node = index as usize;
                    } else {
                        node = next as usize;
                    }
                }
            }
            // Table: the bits as read form the codeword reversed.
            let reversed = reverse(code, len32);
            if len32 <= bits {
                let value = entry as u32 | (len32 << 24);
                for high in 0..(1_u32 << (bits - len32)) {
                    self.table[(reversed | (high << len32)) as usize] = value;
                }
            }
        }
        // Slots of long codewords continue from the node `bits` deep.
        for slot in 0..self.table.len() {
            if self.table[slot] != NONE {
                continue;
            }
            let mut node = 0_u32;
            let mut ok = true;
            for depth in 0..bits {
                let bit = (slot >> depth) & 1;
                let next = self.tree[node as usize][bit];
                if next == NONE || next & LEAF != 0 {
                    ok = false;
                    break;
                }
                node = next;
            }
            if ok {
                self.table[slot] = LONG | node;
            }
        }
    }

    /// Reads one entry number (`None` at end of packet).
    pub(crate) fn decode(&self, reader: &mut BitReader<'_>) -> Option<u32> {
        if let Some(entry) = self.single {
            reader.read(1)?;
            return Some(entry);
        }
        if self.table.is_empty() {
            reader.end();
            return None;
        }
        let (bits, have) = reader.peek(self.table_bits);
        let slot = self.table[bits as usize];
        if slot == NONE {
            reader.end();
            return None;
        }
        if slot & LONG == LONG {
            if have < self.table_bits {
                reader.end();
                return None;
            }
            reader.consume(self.table_bits);
            let mut node = slot & !LONG;
            loop {
                let bit = reader.read(1)? as usize;
                let next = self.tree[node as usize][bit];
                if next == NONE {
                    reader.end();
                    return None;
                }
                if next & LEAF != 0 {
                    return Some(next & !LEAF);
                }
                node = next;
            }
        }
        let len = slot >> 24;
        if len > have {
            reader.end();
            return None;
        }
        reader.consume(len);
        Some(slot & 0x00ff_ffff)
    }

    /// Reads one entry and returns its vector.
    pub(crate) fn decode_vector(&self, reader: &mut BitReader<'_>) -> Option<&[f32]> {
        let entry = self.decode(reader)? as usize;
        let values = self.values.as_ref()?;
        let start = entry * self.dimensions;
        values.get(start..start + self.dimensions)
    }
}

impl BookSpec {
    /// Reads the codebook fields (section 3.2.1).
    pub(crate) fn parse(
        reader: &mut BitReader<'_>,
        budget: &mut Budget,
    ) -> Result<Self, CodecError> {
        let at = reader.position();
        let sync = reader.take(24, "codebook sync")?;
        if sync != SYNC {
            return Err(malformed(at, "codebook sync pattern"));
        }
        let dimensions = reader.take(16, "codebook")? as usize;
        let entries = reader.take(24, "codebook")? as usize;
        if dimensions == 0 || entries == 0 {
            return Err(malformed(at, "empty codebook"));
        }
        budget.entries = budget
            .entries
            .checked_add(entries)
            .filter(|total| *total <= MAX_TOTAL_ENTRIES)
            .ok_or(CodecError::Limit {
                what: "codebook entries",
            })?;
        let ordered = reader.take_bit("codebook")?;
        let mut lengths = vec![0_u8; entries];
        if ordered {
            let mut current = 0_usize;
            let mut length = reader.take(5, "codebook lengths")? + 1;
            while current < entries {
                if length > 32 {
                    return Err(malformed(reader.position(), "codebook length over 32"));
                }
                let width = ilog((entries - current) as u32);
                let number = reader.take(width, "codebook lengths")? as usize;
                if number > entries - current {
                    return Err(malformed(reader.position(), "codebook length run"));
                }
                lengths[current..current + number].fill(length as u8);
                current += number;
                length += 1;
            }
        } else {
            let sparse = reader.take_bit("codebook")?;
            for length in &mut lengths {
                let used = !sparse || reader.take_bit("codebook lengths")?;
                if used {
                    *length = (reader.take(5, "codebook lengths")? + 1) as u8;
                }
            }
        }
        let kind = reader.take(4, "codebook lookup")? as u8;
        let lookup = match kind {
            0 => None,
            1 | 2 => {
                let minimum = reader.take(32, "codebook lookup")?;
                let delta = reader.take(32, "codebook lookup")?;
                let value_bits = (reader.take(4, "codebook lookup")? + 1) as u8;
                let sequence = reader.take_bit("codebook lookup")?;
                let count = if kind == 1 {
                    lookup1_values(entries, dimensions)
                } else {
                    entries.checked_mul(dimensions).ok_or(CodecError::Limit {
                        what: "codebook lookup values",
                    })?
                };
                // Every value is in the packet: check before allocating.
                let needed = count as u64 * u64::from(value_bits);
                if needed > reader.remaining() {
                    return Err(CodecError::Truncated {
                        at_bit: reader.position(),
                        what: "codebook lookup values",
                    });
                }
                let mut multiplicands = Vec::with_capacity(count);
                for _ in 0..count {
                    multiplicands
                        .push(reader.take(u32::from(value_bits), "codebook lookup values")?);
                }
                Some(Lookup {
                    kind,
                    minimum,
                    delta,
                    value_bits,
                    sequence,
                    multiplicands,
                })
            }
            _ => return Err(malformed(reader.position(), "codebook lookup type")),
        };
        Ok(Self {
            dimensions,
            lengths,
            lookup,
        })
    }

    /// Writes the book as [`Self::parse`] reads it, picking ordered or
    /// sparse length coding by size.
    pub(crate) fn write(&self, writer: &mut BitWriter) {
        writer.write(SYNC, 24);
        writer.write(self.dimensions as u32, 16);
        writer.write(self.lengths.len() as u32, 24);
        let entries = self.lengths.len();
        let ordered = self.lengths.windows(2).all(|pair| pair[0] <= pair[1])
            && self.lengths.first().is_some_and(|&len| len > 0);
        writer.write_bit(ordered);
        if ordered {
            let mut current = 0;
            let mut length = self.lengths[0];
            writer.write(u32::from(length) - 1, 5);
            while current < entries {
                let number = self.lengths[current..]
                    .iter()
                    .take_while(|&&len| len == length)
                    .count();
                writer.write(number as u32, ilog((entries - current) as u32));
                current += number;
                length += 1;
            }
        } else {
            let sparse = self.lengths.contains(&0);
            writer.write_bit(sparse);
            for &len in &self.lengths {
                if sparse {
                    writer.write_bit(len > 0);
                }
                if len > 0 {
                    writer.write(u32::from(len) - 1, 5);
                }
            }
        }
        match &self.lookup {
            None => writer.write(0, 4),
            Some(lookup) => {
                writer.write(u32::from(lookup.kind), 4);
                writer.write(lookup.minimum, 32);
                writer.write(lookup.delta, 32);
                writer.write(u32::from(lookup.value_bits) - 1, 4);
                writer.write_bit(lookup.sequence);
                for &value in &lookup.multiplicands {
                    writer.write(value, u32::from(lookup.value_bits));
                }
            }
        }
    }
}

/// Codeword and length per entry (length 0: unused). Each entry takes the
/// lowest-valued codeword of its length still free, in entry order
/// (section 3.2.1). Over- and underfull trees are refused.
pub(crate) fn assign_codewords(lengths: &[u8]) -> Result<Vec<(u32, u8)>, CodecError> {
    // A binary tree over codeword prefixes. `shallowest[n]` is the depth of
    // the shallowest empty subtree under node `n` (its own depth when empty,
    // `FULL` when nothing is free).
    const FULL: u8 = u8::MAX;
    const EMPTY_CHILD: u32 = u32::MAX;
    let mut children: Vec<[u32; 2]> = vec![[EMPTY_CHILD; 2]];
    let mut shallowest: Vec<u8> = vec![0];
    let mut leaf: Vec<bool> = vec![false];
    let mut codes = Vec::with_capacity(lengths.len());
    let mut path: Vec<u32> = Vec::with_capacity(33);
    for &len in lengths {
        if len == 0 {
            codes.push((0, 0));
            continue;
        }
        if len > 32 {
            return Err(malformed(0, "codebook length over 32"));
        }
        if shallowest[0] > len {
            return Err(malformed(0, "overfull Huffman tree"));
        }
        let mut node = 0_u32;
        let mut code = 0_u32;
        path.clear();
        path.push(0);
        for depth in 0..len {
            // A missing child is an empty subtree one deeper.
            let free = |child: u32| -> u8 {
                if child == EMPTY_CHILD {
                    depth + 1
                } else {
                    shallowest[child as usize]
                }
            };
            let [zero, one] = children[node as usize];
            let bit = if free(zero) <= len {
                0
            } else if free(one) <= len {
                1
            } else {
                return Err(malformed(0, "overfull Huffman tree"));
            };
            code = (code << 1) | bit;
            let mut child = children[node as usize][bit as usize];
            if child == EMPTY_CHILD {
                children.push([EMPTY_CHILD; 2]);
                shallowest.push(depth + 1);
                leaf.push(false);
                child = (children.len() - 1) as u32;
                children[node as usize][bit as usize] = child;
            }
            node = child;
            path.push(node);
        }
        leaf[node as usize] = true;
        shallowest[node as usize] = FULL;
        // Recompute up the path.
        for (depth, &at) in path.iter().enumerate().rev().skip(1) {
            let depth = depth as u8;
            let free = |child: u32| -> u8 {
                if child == EMPTY_CHILD {
                    depth + 1
                } else {
                    shallowest[child as usize]
                }
            };
            let [zero, one] = children[at as usize];
            shallowest[at as usize] = free(zero).min(free(one));
        }
        codes.push((code, len));
    }
    if shallowest[0] != FULL {
        return Err(malformed(0, "underfull Huffman tree"));
    }
    Ok(codes)
}

fn reverse(code: u32, len: u32) -> u32 {
    if len == 0 {
        0
    } else {
        code.reverse_bits() >> (32 - len)
    }
}

/// The greatest `r` with `r^dimensions <= entries` (`lookup1_values`).
pub(crate) fn lookup1_values(entries: usize, dimensions: usize) -> usize {
    let fits = |r: u64| -> bool {
        let mut product = 1_u64;
        for _ in 0..dimensions {
            product = match product.checked_mul(r) {
                Some(value) if value <= entries as u64 => value,
                _ => return false,
            };
        }
        true
    };
    let mut r = (entries as f64).powf(1.0 / dimensions as f64).floor() as u64;
    while r > 1 && !fits(r) {
        r -= 1;
    }
    while fits(r + 1) {
        r += 1;
    }
    r.max(1) as usize
}

/// `float32_unpack` (section 9.2.2).
pub(crate) fn float32_unpack(word: u32) -> f32 {
    let mantissa = f64::from(word & 0x001f_ffff);
    let exponent = ((word & 0x7fe0_0000) >> 21) as i32;
    let signed = if word & 0x8000_0000 != 0 {
        -mantissa
    } else {
        mantissa
    };
    (signed * 2_f64.powi(exponent - 788)) as f32
}

/// The inverse of [`float32_unpack`] for values it can represent: a 21-bit
/// mantissa and an exponent.
pub(crate) fn float32_pack(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let sign = if value < 0.0 { 0x8000_0000 } else { 0 };
    let mut magnitude = f64::from(value.abs());
    let mut exponent = 788_i32;
    while magnitude >= f64::from(1_u32 << 21) {
        magnitude /= 2.0;
        exponent += 1;
    }
    while magnitude < f64::from(1_u32 << 20) {
        magnitude *= 2.0;
        exponent -= 1;
    }
    let mantissa = magnitude.round() as u32;
    sign | ((exponent.clamp(0, 1023) as u32) << 21) | (mantissa & 0x001f_ffff)
}

/// The vector of every entry (section 3.2.1, "VQ lookup table vector
/// representation"), in `f32` arithmetic as the reference computes it.
fn unpack_values(lookup: &Lookup, entries: usize, dimensions: usize) -> Vec<f32> {
    let minimum = float32_unpack(lookup.minimum);
    let delta = float32_unpack(lookup.delta);
    let mut values = Vec::with_capacity(entries * dimensions);
    let count = lookup.multiplicands.len().max(1);
    for entry in 0..entries {
        let mut last = 0.0_f32;
        let mut divisor = 1_usize;
        for i in 0..dimensions {
            let offset = if lookup.kind == 1 {
                (entry / divisor) % count
            } else {
                entry * dimensions + i
            };
            let multiplicand = lookup.multiplicands.get(offset).copied().unwrap_or(0) as f32;
            let value = multiplicand * delta + minimum + last;
            if lookup.sequence {
                last = value;
            }
            values.push(value);
            if lookup.kind == 1 {
                divisor = divisor.saturating_mul(count);
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::{
        BookSpec, Budget, Codebook, Lookup, assign_codewords, float32_pack, float32_unpack,
        lookup1_values,
    };
    use crate::bits::{BitReader, BitWriter};

    #[test]
    fn codewords_follow_the_spec_example() {
        // Section 3.2.1: lengths 2 4 4 4 4 2 3 3.
        let codes = assign_codewords(&[2, 4, 4, 4, 4, 2, 3, 3]).unwrap();
        let expected = [
            (0b00, 2),
            (0b0100, 4),
            (0b0101, 4),
            (0b0110, 4),
            (0b0111, 4),
            (0b10, 2),
            (0b110, 3),
            (0b111, 3),
        ];
        assert_eq!(codes, expected);
    }

    #[test]
    fn a_short_code_after_long_ones_takes_the_lowest_free_slot() {
        let codes = assign_codewords(&[3, 1, 3, 2]).unwrap();
        assert_eq!(codes, [(0b000, 3), (0b1, 1), (0b001, 3), (0b01, 2)]);
    }

    #[test]
    fn over_and_underfull_trees_are_refused() {
        assert!(assign_codewords(&[1, 1, 1]).is_err());
        assert!(assign_codewords(&[1, 2]).is_err());
        assert!(assign_codewords(&[2, 2, 2]).is_err());
        assert!(assign_codewords(&[1, 0, 1]).is_ok());
    }

    fn write_codes(writer: &mut BitWriter, code: u32, len: u8) {
        for depth in (0..len).rev() {
            writer.write_bit((code >> depth) & 1 == 1);
        }
    }

    #[test]
    fn decoding_reads_every_codeword_short_and_long() {
        let mut lengths = vec![1_u8];
        for len in 2..=20 {
            lengths.push(len);
        }
        lengths.push(20);
        let spec = BookSpec {
            dimensions: 1,
            lengths: lengths.clone(),
            lookup: None,
        };
        let book = Codebook::build(&spec, &mut Budget::default()).unwrap();
        let codes = assign_codewords(&lengths).unwrap();
        let mut writer = BitWriter::new();
        let order: Vec<usize> = (0..lengths.len()).rev().chain(0..lengths.len()).collect();
        for &entry in &order {
            write_codes(&mut writer, codes[entry].0, codes[entry].1);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        for &entry in &order {
            assert_eq!(book.decode(&mut reader), Some(entry as u32));
        }
    }

    #[test]
    fn a_codeword_cut_by_the_packet_end_is_end_of_packet() {
        let spec = BookSpec {
            dimensions: 1,
            lengths: vec![1, 3, 3, 3, 3],
            lookup: None,
        };
        let book = Codebook::build(&spec, &mut Budget::default()).unwrap();
        // Entry 1 is 100, cut after two bits.
        let mut reader = BitReader::new(&[0b0100_0000]);
        reader.read(6).unwrap();
        assert_eq!(book.decode(&mut reader), None);
        assert!(reader.ended());
    }

    #[test]
    fn a_single_entry_book_sinks_one_bit_of_either_value() {
        let spec = BookSpec {
            dimensions: 1,
            lengths: vec![0, 0, 1, 0],
            lookup: None,
        };
        let book = Codebook::build(&spec, &mut Budget::default()).unwrap();
        let mut reader = BitReader::new(&[0b10]);
        assert_eq!(book.decode(&mut reader), Some(2));
        assert_eq!(book.decode(&mut reader), Some(2));
        assert_eq!(reader.position(), 2);
        let long = BookSpec {
            dimensions: 1,
            lengths: vec![3],
            lookup: None,
        };
        assert!(Codebook::build(&long, &mut Budget::default()).is_err());
    }

    #[test]
    fn lookup1_values_is_the_integer_root() {
        assert_eq!(lookup1_values(81, 4), 3);
        assert_eq!(lookup1_values(80, 4), 2);
        assert_eq!(lookup1_values(1, 8), 1);
        assert_eq!(lookup1_values(1 << 24, 1), 1 << 24);
        assert_eq!(lookup1_values(1 << 24, 2), 4096);
        assert_eq!(lookup1_values(5, 65_535), 1);
    }

    #[test]
    fn floats_unpack_as_the_spec_and_pack_back() {
        // 1.0 = mantissa 2^20, exponent 788 + 0 - 20.
        let one = (768 << 21) | (1 << 20);
        assert_eq!(float32_unpack(one), 1.0);
        assert_eq!(float32_unpack(one | 0x8000_0000), -1.0);
        for value in [1.0_f32, -1.0, 0.5, -15.25, 0.001_953_125, 1234.0, 0.375] {
            assert_eq!(float32_unpack(float32_pack(value)), value);
        }
        assert_eq!(float32_pack(0.0), 0);
    }

    #[test]
    fn books_round_trip_through_the_setup_bits() {
        let books = [
            BookSpec {
                dimensions: 2,
                lengths: vec![1, 2, 3, 4, 5, 6, 7, 8, 8],
                lookup: Some(Lookup {
                    kind: 1,
                    minimum: float32_pack(-1.0),
                    delta: float32_pack(1.0),
                    value_bits: 2,
                    sequence: false,
                    multiplicands: vec![0, 1, 2],
                }),
            },
            BookSpec {
                dimensions: 3,
                lengths: vec![2, 0, 2, 2, 2],
                lookup: Some(Lookup {
                    kind: 2,
                    minimum: float32_pack(0.5),
                    delta: float32_pack(0.25),
                    value_bits: 5,
                    sequence: true,
                    multiplicands: (0..15).collect(),
                }),
            },
            BookSpec {
                dimensions: 1,
                lengths: vec![3, 3, 2, 3, 3, 2],
                lookup: None,
            },
        ];
        let mut writer = BitWriter::new();
        for book in &books {
            book.write(&mut writer);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let mut budget = Budget::default();
        for book in &books {
            assert_eq!(&BookSpec::parse(&mut reader, &mut budget).unwrap(), book);
        }
        // Lookup type 1: entry 4 of a 3-value, 2-dimension book is (0, 0) + ... digits (1, 1).
        let built = Codebook::build(&books[0], &mut Budget::default()).unwrap();
        let values = built.values.unwrap();
        assert_eq!(&values[8..10], &[0.0, 0.0]);
        assert_eq!(&values[0..2], &[-1.0, -1.0]);
        assert_eq!(&values[2..4], &[0.0, -1.0]);
        // Lookup type 2 with sequence: running sums of each entry's values.
        let built = Codebook::build(&books[1], &mut Budget::default()).unwrap();
        let values = built.values.unwrap();
        assert_eq!(&values[3..6], &[1.25, 2.75, 4.5]);
    }

    #[test]
    fn hostile_sizes_fail_before_allocating() {
        // 2^24 entries, ordered, one length: allowed alone, but the setup
        // budget refuses it.
        let mut writer = BitWriter::new();
        writer.write(0x56_43_42, 24);
        writer.write(1, 16);
        writer.write((1 << 24) - 1, 24);
        writer.write_bit(true);
        writer.write(0, 5);
        writer.write((1 << 24) - 1, 24);
        let bytes = writer.finish();
        let err = BookSpec::parse(&mut BitReader::new(&bytes), &mut Budget::default());
        assert!(matches!(err, Err(cutlass_decode::CodecError::Limit { .. })));
        // A lookup table longer than the packet is truncated, not allocated.
        let mut writer = BitWriter::new();
        writer.write(0x56_43_42, 24);
        writer.write(4, 16);
        writer.write(4096, 24);
        writer.write_bit(true);
        writer.write(11, 5);
        writer.write(4096, 13);
        writer.write(2, 4);
        writer.write(0, 32);
        writer.write(0, 32);
        writer.write(15, 4);
        writer.write_bit(false);
        let bytes = writer.finish();
        let err = BookSpec::parse(&mut BitReader::new(&bytes), &mut Budget::default());
        assert!(matches!(
            err,
            Err(cutlass_decode::CodecError::Truncated { .. })
        ));
    }
}
