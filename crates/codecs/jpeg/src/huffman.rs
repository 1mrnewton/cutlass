//! Huffman decode and encode tables. Counts and symbols are the DHT payload.

use cutlass_decode::CodecError;

use crate::bits::Cursor;

/// Codes up to this length decode with one table lookup.
const LOOKUP_BITS: u32 = 9;

#[derive(Clone)]
pub(crate) struct HuffDecode {
    /// `length << 8 | symbol` for every `LOOKUP_BITS` prefix of a short code.
    /// Zero when the code is longer.
    lookup: Box<[u16; 1 << LOOKUP_BITS]>,
    maxcode: [i32; 17],
    /// Index into `symbols` minus the first code of that length.
    offset: [i32; 17],
    symbols: [u8; 256],
}

pub(crate) struct HuffEncode {
    pub(crate) code: [u16; 256],
    pub(crate) len: [u8; 256],
    pub(crate) counts: [u8; 16],
    pub(crate) symbols: Vec<u8>,
}

pub(crate) fn decode_table(counts: &[u8], symbols: &[u8]) -> Result<HuffDecode, CodecError> {
    if counts.len() != 16 || symbols.len() > 256 {
        return Err(malformed());
    }
    let declared: usize = counts.iter().map(|count| usize::from(*count)).sum();
    if declared != symbols.len() {
        return Err(malformed());
    }

    let mut lookup = Box::new([0u16; 1 << LOOKUP_BITS]);
    let mut maxcode = [-1i32; 17];
    let mut offset = [0i32; 17];
    let mut stored = [0u8; 256];
    stored[..symbols.len()].copy_from_slice(symbols);

    // Figure C.2. A length with no symbols still shifts the code.
    let mut code = 0u32;
    let mut index = 0usize;
    for length in 1..=16u32 {
        let count = u32::from(counts[usize::try_from(length - 1).unwrap_or(0)]);
        let Some(next) = code.checked_add(count) else {
            return Err(malformed());
        };
        // No code is all ones: that pattern is reserved as a prefix.
        if count > 0 && next >= (1 << length) {
            return Err(malformed());
        }
        let slot = usize::try_from(length).unwrap_or(0);
        if count > 0 {
            offset[slot] = i32::try_from(index).map_err(|_| malformed())?
                - i32::try_from(code).map_err(|_| malformed())?;
            maxcode[slot] = i32::try_from(next).map_err(|_| malformed())? - 1;
        }
        for c in code..next {
            let Some(&symbol) = symbols.get(index) else {
                return Err(malformed());
            };
            if length <= LOOKUP_BITS {
                let shift = LOOKUP_BITS - length;
                let first = usize::try_from(c << shift).unwrap_or(0);
                let last = usize::try_from((c + 1) << shift).unwrap_or(0);
                let entry = (u16::try_from(length).unwrap_or(0) << 8) | u16::from(symbol);
                lookup[first..last].fill(entry);
            }
            index += 1;
        }
        code = next << 1;
    }
    Ok(HuffDecode {
        lookup,
        maxcode,
        offset,
        symbols: stored,
    })
}

pub(crate) fn encode_table(counts: &[u8; 16], symbols: &[u8]) -> Result<HuffEncode, CodecError> {
    decode_table(counts, symbols)?;
    let mut code = [0u16; 256];
    let mut len = [0u8; 256];
    let mut next = 0u32;
    let mut symbol_index = 0usize;
    for length in 1..=16u8 {
        let count = usize::from(counts[usize::from(length - 1)]);
        for _ in 0..count {
            let Some(&symbol) = symbols.get(symbol_index) else {
                return Err(malformed());
            };
            code[usize::from(symbol)] = u16::try_from(next).map_err(|_| malformed())?;
            len[usize::from(symbol)] = length;
            symbol_index += 1;
            next += 1;
        }
        next <<= 1;
    }
    Ok(HuffEncode {
        code,
        len,
        counts: *counts,
        symbols: symbols.to_vec(),
    })
}

#[inline]
pub(crate) fn decode_symbol(input: &mut Cursor<'_>, table: &HuffDecode) -> Result<u8, CodecError> {
    let bits = input.peek(16);
    let entry = table.lookup[usize::try_from(bits >> (16 - LOOKUP_BITS)).unwrap_or(0)];
    if entry != 0 {
        input.consume(u32::from(entry >> 8));
        return Ok(u8::try_from(entry & 0xff).unwrap_or(0));
    }
    slow_symbol(input, table, bits)
}

fn slow_symbol(input: &mut Cursor<'_>, table: &HuffDecode, bits: u32) -> Result<u8, CodecError> {
    for length in LOOKUP_BITS + 1..=16 {
        let code = i32::try_from(bits >> (16 - length)).unwrap_or(i32::MAX);
        let slot = usize::try_from(length).unwrap_or(0);
        if code <= table.maxcode[slot] {
            input.consume(length);
            let index = usize::try_from(code + table.offset[slot]).map_err(|_| malformed())?;
            return table.symbols.get(index).copied().ok_or_else(malformed);
        }
    }
    input.check()?;
    Err(input.malformed("huffman code"))
}

fn malformed() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "huffman table",
    }
}
