//! The variable-length LZW GIF images use, packed into data sub-blocks.

use std::collections::HashMap;

/// The image data after the image descriptor and its color table: the LZW
/// minimum code size, the sub-blocks, and the block terminator.
pub(crate) fn image_data(indexes: &[u8], palette_bits: u8) -> Vec<u8> {
    let min_size = palette_bits.max(2);
    let codes = compress(indexes, min_size);
    let mut out = Vec::with_capacity(codes.len() + codes.len() / 255 + 3);
    out.push(min_size);
    for block in codes.chunks(255) {
        out.push(u8::try_from(block.len()).unwrap_or(255));
        out.extend_from_slice(block);
    }
    out.push(0);
    out
}

struct Bits {
    out: Vec<u8>,
    held: u32,
    count: u32,
}

impl Bits {
    fn put(&mut self, code: u16, width: u32) {
        self.held |= u32::from(code) << self.count;
        self.count += width;
        while self.count >= 8 {
            self.out.push((self.held & 0xFF) as u8);
            self.held >>= 8;
            self.count -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push((self.held & 0xFF) as u8);
        }
        self.out
    }
}

/// LZW codes, least significant bit first, starting with a clear code and
/// ending with the end code. The table clears when it reaches 4096 codes.
fn compress(indexes: &[u8], min_size: u8) -> Vec<u8> {
    let clear: u16 = 1 << min_size;
    let end = clear + 1;
    let mut bits = Bits {
        out: Vec::new(),
        held: 0,
        count: 0,
    };
    let mut width = u32::from(min_size) + 1;
    let mut next = end + 1;
    let mut table: HashMap<(u16, u8), u16> = HashMap::new();
    bits.put(clear, width);
    let mut iter = indexes.iter();
    let Some(&first) = iter.next() else {
        bits.put(end, width);
        return bits.finish();
    };
    let mut prefix = u16::from(first);
    for &index in iter {
        if let Some(&code) = table.get(&(prefix, index)) {
            prefix = code;
            continue;
        }
        bits.put(prefix, width);
        if next < 4096 {
            table.insert((prefix, index), next);
            next += 1;
            if u32::from(next) > (1 << width) && width < 12 {
                width += 1;
            }
        } else {
            bits.put(clear, width);
            table.clear();
            next = end + 1;
            width = u32::from(min_size) + 1;
        }
        prefix = u16::from(index);
    }
    bits.put(prefix, width);
    // A decoder adds a table entry after the last code too, so it may widen
    // before the end code; widen with it.
    if u32::from(next) + 1 > (1 << width) && width < 12 && next < 4096 {
        width += 1;
    }
    bits.put(end, width);
    bits.finish()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Decode the sub-blocks `image_data` wrote back to color indexes.
    pub(crate) fn expand(data: &[u8]) -> Vec<u8> {
        let min_size = data[0];
        let mut packed = Vec::new();
        let mut at = 1;
        while data[at] != 0 {
            let len = usize::from(data[at]);
            packed.extend_from_slice(&data[at + 1..at + 1 + len]);
            at += 1 + len;
        }
        let clear: usize = 1 << min_size;
        let end = clear + 1;
        let mut width = u32::from(min_size) + 1;
        let mut table: Vec<Vec<u8>> = (0..clear).map(|index| vec![index as u8]).collect();
        table.push(Vec::new());
        table.push(Vec::new());
        let mut out = Vec::new();
        let mut previous: Option<usize> = None;
        let (mut held, mut count, mut byte) = (0u32, 0u32, 0usize);
        loop {
            while count < width {
                held |= u32::from(packed[byte]) << count;
                byte += 1;
                count += 8;
            }
            let code = (held & ((1 << width) - 1)) as usize;
            held >>= width;
            count -= width;
            if code == clear {
                table.truncate(end + 1);
                width = u32::from(min_size) + 1;
                previous = None;
                continue;
            }
            if code == end {
                return out;
            }
            let entry = if code < table.len() {
                table[code].clone()
            } else {
                let mut entry = table[previous.unwrap()].clone();
                entry.push(entry[0]);
                entry
            };
            if let Some(previous) = previous {
                let mut grown = table[previous].clone();
                grown.push(entry[0]);
                table.push(grown);
                if table.len() == (1 << width) && width < 12 {
                    width += 1;
                }
            }
            out.extend_from_slice(&entry);
            previous = Some(code);
        }
    }

    #[test]
    fn codes_round_trip_through_a_table_reset() {
        let mut indexes = Vec::new();
        let mut state = 7u32;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            indexes.push(((state >> 16) % 200) as u8);
        }
        assert_eq!(expand(&image_data(&indexes, 8)), indexes);
        let flat = vec![3u8; 5_000];
        assert_eq!(expand(&image_data(&flat, 2)), flat);
        assert_eq!(expand(&image_data(&[1], 1)), vec![1]);
    }
}
