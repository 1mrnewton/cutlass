//! Huffman decode and encode tables. Counts and symbols are the DHT payload.

use cutlass_decode::CodecError;

use crate::bits::Cursor;

pub(crate) struct HuffDecode {
    maxcode: [i32; 18],
    mincode: [i32; 18],
    valptr: [i16; 18],
    symbols: Vec<u8>,
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

    // Figure C.2. A length with no symbols still shifts the code.
    let mut code = 0u32;
    let mut symbol_index = 0usize;
    for length in 1..=16 {
        let count = u32::from(counts[length - 1]);
        let Some(next) = code.checked_add(count) else {
            return Err(malformed());
        };
        if count > 0 && next >= (1 << length) {
            return Err(malformed());
        }
        symbol_index += usize::from(counts[length - 1]);
        code = next << 1;
    }
    if symbol_index != symbols.len() {
        return Err(malformed());
    }

    let mut maxcode = [-1i32; 18];
    let mut mincode = [0i32; 18];
    let mut valptr = [0i16; 18];
    let mut cursor = 0i32;
    symbol_index = 0;
    for length in 1..=16 {
        let count = usize::from(counts[length - 1]);
        if count == 0 {
            cursor <<= 1;
            continue;
        }
        let Some(pointer) = i16::try_from(symbol_index).ok() else {
            return Err(malformed());
        };
        valptr[length] = pointer;
        mincode[length] = cursor;
        symbol_index += count;
        cursor += i32::try_from(count).map_err(|_| malformed())?;
        maxcode[length] = cursor - 1;
        cursor <<= 1;
    }

    Ok(HuffDecode {
        maxcode,
        mincode,
        valptr,
        symbols: symbols.to_vec(),
    })
}

pub(crate) fn encode_table(counts: &[u8; 16], symbols: &[u8]) -> Result<HuffEncode, CodecError> {
    let decode = decode_table(counts, symbols)?;
    let mut code = [0u16; 256];
    let mut len = [0u8; 256];
    let mut next = 0u32;
    let mut symbol_index = 0usize;
    for length in 1..=16 {
        let count = usize::from(counts[length - 1]);
        for _ in 0..count {
            let Some(&symbol) = symbols.get(symbol_index) else {
                return Err(malformed());
            };
            code[usize::from(symbol)] = u16::try_from(next).unwrap_or(0);
            len[usize::from(symbol)] = u8::try_from(length).unwrap_or(0);
            symbol_index += 1;
            next += 1;
        }
        next <<= 1;
    }
    let _ = decode;
    Ok(HuffEncode {
        code,
        len,
        counts: *counts,
        symbols: symbols.to_vec(),
    })
}

pub(crate) fn decode_symbol(input: &mut Cursor<'_>, table: &HuffDecode) -> Result<u8, CodecError> {
    let mut code = 0i32;
    for length in 1..=16 {
        code = (code << 1) | i32::from(input.read_bit()?);
        if code <= table.maxcode[length] {
            let offset = code - table.mincode[length];
            let index = i32::from(table.valptr[length]) + offset;
            let Some(&symbol) = usize::try_from(index)
                .ok()
                .and_then(|index| table.symbols.get(index))
            else {
                return Err(malformed());
            };
            return Ok(symbol);
        }
    }
    Err(malformed())
}

fn malformed() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "huffman table",
    }
}
