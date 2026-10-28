//! zlib wrapper and deflate, for the IDAT stream and compressed metadata.
//!
//! The bit packing follows RFC 1951. The two-byte header and Adler-32 follow
//! RFC 1950. The encoder writes one final fixed-Huffman block. The decoder
//! accepts stored, fixed, and dynamic blocks, and rejects a preset dictionary
//! or a window above 32 KiB.

const ADLER_BASE: u32 = 65_521;
const MAX_WINDOW: usize = 32_768;
const MAX_BITS: usize = 15;

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZlibError {
    Truncated,
    Malformed(&'static str),
    Limit,
}

pub(crate) fn inflate(src: &[u8], limit: usize) -> Result<Vec<u8>, ZlibError> {
    if src.len() < 6 {
        return Err(ZlibError::Truncated);
    }
    let cmf = u16::from(src[0]);
    let flg = u16::from(src[1]);
    if (cmf * 256 + flg) % 31 != 0 {
        return Err(ZlibError::Malformed("zlib header"));
    }
    let method = src[0] & 0x0f;
    let info = src[0] >> 4;
    if method != 8 || info > 7 {
        return Err(ZlibError::Malformed("zlib method"));
    }
    if src[1] & 0x20 != 0 {
        return Err(ZlibError::Malformed("zlib dictionary"));
    }
    let window = 1usize << (usize::from(info) + 8);
    let mut bits = Bits {
        data: src,
        index: 2,
        acc: 0,
        bits: 0,
    };
    let mut out = Vec::new();
    loop {
        let final_block = bits.get(1).ok_or(ZlibError::Truncated)?;
        let kind = bits.get(2).ok_or(ZlibError::Truncated)?;
        match kind {
            0 => stored(&mut bits, &mut out, limit)?,
            1 => {
                let lit = fixed_lit()?;
                let dist = fixed_dist()?;
                huffman(&mut bits, &mut out, limit, window, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                huffman(&mut bits, &mut out, limit, window, &lit, &dist)?;
            }
            _ => return Err(ZlibError::Malformed("deflate block")),
        }
        if final_block == 1 {
            break;
        }
    }
    let rest = bits.rest();
    if rest.len() < 4 {
        return Err(ZlibError::Truncated);
    }
    if rest.len() != 4 {
        return Err(ZlibError::Malformed("zlib trailer"));
    }
    let stored = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    if stored != adler32(&out) {
        return Err(ZlibError::Malformed("adler32"));
    }
    Ok(out)
}

pub(crate) fn deflate(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 2 + 64);
    out.push(0x78);
    out.push(0x01);
    let mut writer = BitWriter::default();
    writer.put(1, 1);
    writer.put(1, 2);
    write_fixed(&mut writer, src);
    let (end_code, end_len) = fixed_lit_code(256);
    writer.put_code(end_code, end_len);
    writer.finish(&mut out);
    out.extend_from_slice(&adler32(src).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    let mut index = 0;
    while index < data.len() {
        let end = (index + 5552).min(data.len());
        while index < end {
            s1 += u32::from(data[index]);
            s2 += s1;
            index += 1;
        }
        s1 %= ADLER_BASE;
        s2 %= ADLER_BASE;
    }
    (s2 << 16) | s1
}

struct Bits<'a> {
    data: &'a [u8],
    index: usize,
    acc: u32,
    bits: u8,
}

impl Bits<'_> {
    fn ensure(&mut self, n: u8) -> bool {
        while self.bits < n {
            let Some(byte) = self.data.get(self.index) else {
                return false;
            };
            self.acc |= u32::from(*byte) << self.bits;
            self.index += 1;
            self.bits += 8;
        }
        true
    }

    fn get(&mut self, n: u8) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if !self.ensure(n) {
            return None;
        }
        let mask = (1u32 << n) - 1;
        let value = self.acc & mask;
        self.acc >>= n;
        self.bits -= n;
        Some(value)
    }

    fn align(&mut self) {
        let drop_bits = self.bits % 8;
        if drop_bits != 0 {
            self.acc >>= drop_bits;
            self.bits -= drop_bits;
        }
    }

    fn rest(&self) -> &[u8] {
        // Whole bytes pulled into the accumulator but not consumed are still
        // part of the trailer. A partial byte is deflate padding.
        let unread = usize::from(self.bits / 8);
        let start = self.index.saturating_sub(unread);
        &self.data[start..]
    }
}

struct Huff {
    table: Vec<u16>,
    lens: Vec<u8>,
    max: u8,
    codes: Vec<u16>,
    lengths: Vec<u8>,
}

fn build(lengths: &[u8]) -> Result<Huff, ZlibError> {
    let max = lengths.iter().copied().max().unwrap_or(0);
    if max > MAX_BITS as u8 {
        return Err(ZlibError::Malformed("huffman length"));
    }
    let mut count = [0u16; MAX_BITS + 1];
    for &len in lengths {
        count[usize::from(len)] += 1;
    }
    count[0] = 0;
    let mut left = 1i32;
    for tally in count.iter().skip(1) {
        left <<= 1;
        left -= i32::from(*tally);
        if left < 0 {
            return Err(ZlibError::Malformed("huffman tree"));
        }
    }
    if max == 0 {
        return Ok(Huff {
            table: vec![u16::MAX],
            lens: vec![0],
            max: 0,
            codes: vec![0; lengths.len()],
            lengths: lengths.to_vec(),
        });
    }
    let mut next = [0u16; MAX_BITS + 1];
    let mut code = 0u16;
    for bits in 1..=MAX_BITS {
        code = code.wrapping_add(count[bits - 1]).wrapping_shl(1);
        next[bits] = code;
    }
    let size = 1usize << max;
    let mut table = vec![u16::MAX; size];
    let mut lens = vec![0u8; size];
    let mut codes = vec![0u16; lengths.len()];
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let assigned = next[usize::from(len)];
        next[usize::from(len)] = assigned.wrapping_add(1);
        codes[symbol] = assigned;
        let rev = reverse_bits(assigned, len);
        let step = 1usize << len;
        let mut fill = usize::from(rev);
        while fill < size {
            if table[fill] != u16::MAX {
                return Err(ZlibError::Malformed("huffman tree"));
            }
            table[fill] = u16::try_from(symbol).unwrap_or(u16::MAX);
            lens[fill] = len;
            fill += step;
        }
    }
    Ok(Huff {
        table,
        lens,
        max,
        codes,
        lengths: lengths.to_vec(),
    })
}

impl Huff {
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<u16, ZlibError> {
        if self.max == 0 {
            return Err(ZlibError::Malformed("huffman symbol"));
        }
        if bits.ensure(self.max) {
            let index = (bits.acc & ((1u32 << self.max) - 1)) as usize;
            let len = self.lens[index];
            let symbol = self.table[index];
            if len == 0 || symbol == u16::MAX {
                return Err(ZlibError::Malformed("huffman symbol"));
            }
            bits.acc >>= len;
            bits.bits -= len;
            return Ok(symbol);
        }
        let mut acc = 0u16;
        let mut have = 0u8;
        loop {
            let bit = bits.get(1).ok_or(ZlibError::Truncated)?;
            acc = (acc << 1) | u16::try_from(bit).unwrap_or(0);
            have += 1;
            if have > MAX_BITS as u8 {
                return Err(ZlibError::Malformed("huffman symbol"));
            }
            for (symbol, (&len, &code)) in self.lengths.iter().zip(self.codes.iter()).enumerate() {
                if len == have && code == acc {
                    return u16::try_from(symbol)
                        .map_err(|_| ZlibError::Malformed("huffman symbol"));
                }
            }
        }
    }
}

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut value = code;
    let mut reversed = 0u16;
    for _ in 0..len {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

fn stored(bits: &mut Bits<'_>, out: &mut Vec<u8>, limit: usize) -> Result<(), ZlibError> {
    bits.align();
    let len = bits.get(16).ok_or(ZlibError::Truncated)? as usize;
    let nlen = bits.get(16).ok_or(ZlibError::Truncated)? as u16;
    if nlen != !(len as u16) {
        return Err(ZlibError::Malformed("stored length"));
    }
    if out.len().saturating_add(len) > limit {
        return Err(ZlibError::Limit);
    }
    for _ in 0..len {
        let byte = bits.get(8).ok_or(ZlibError::Truncated)?;
        out.push(byte as u8);
    }
    Ok(())
}

fn huffman(
    bits: &mut Bits<'_>,
    out: &mut Vec<u8>,
    limit: usize,
    window: usize,
    lit: &Huff,
    dist: &Huff,
) -> Result<(), ZlibError> {
    loop {
        let symbol = lit.symbol(bits)?;
        match symbol {
            0..=255 => {
                if out.len() >= limit {
                    return Err(ZlibError::Limit);
                }
                out.push(symbol as u8);
            }
            256 => return Ok(()),
            257..=285 => {
                let index = usize::from(symbol - 257);
                let extra = LEN_EXTRA[index];
                let add = bits.get(extra).ok_or(ZlibError::Truncated)? as usize;
                let len = usize::from(LEN_BASE[index]) + add;
                let distance = distance(bits, dist)?;
                copy_match(out, distance, len, limit, window)?;
            }
            _ => return Err(ZlibError::Malformed("literal")),
        }
    }
}

fn distance(bits: &mut Bits<'_>, dist: &Huff) -> Result<usize, ZlibError> {
    let symbol = dist.symbol(bits)?;
    let index = usize::from(symbol);
    if index >= DIST_BASE.len() {
        return Err(ZlibError::Malformed("distance"));
    }
    let extra = DIST_EXTRA[index];
    let add = bits.get(extra).ok_or(ZlibError::Truncated)? as usize;
    Ok(usize::from(DIST_BASE[index]) + add)
}

fn copy_match(
    out: &mut Vec<u8>,
    distance: usize,
    len: usize,
    limit: usize,
    window: usize,
) -> Result<(), ZlibError> {
    if distance == 0 || distance > out.len() || distance > window || distance > MAX_WINDOW {
        return Err(ZlibError::Malformed("distance"));
    }
    if out.len().saturating_add(len) > limit {
        return Err(ZlibError::Limit);
    }
    for _ in 0..len {
        let byte = out[out.len() - distance];
        out.push(byte);
    }
    Ok(())
}

fn dynamic(bits: &mut Bits<'_>) -> Result<(Huff, Huff), ZlibError> {
    let hlit = bits.get(5).ok_or(ZlibError::Truncated)? as usize + 257;
    let hdist = bits.get(5).ok_or(ZlibError::Truncated)? as usize + 1;
    let hclen = bits.get(4).ok_or(ZlibError::Truncated)? as usize + 4;
    if hlit > 286 || hdist > 32 || hclen > 19 {
        return Err(ZlibError::Malformed("dynamic header"));
    }
    let mut code_lengths = [0u8; 19];
    for index in CL_ORDER.iter().take(hclen) {
        code_lengths[*index] = bits.get(3).ok_or(ZlibError::Truncated)? as u8;
    }
    let alphabet = build(&code_lengths)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut index = 0;
    while index < lengths.len() {
        let symbol = alphabet.symbol(bits)?;
        match symbol {
            0..=15 => {
                lengths[index] = symbol as u8;
                index += 1;
            }
            16 => {
                if index == 0 {
                    return Err(ZlibError::Malformed("code length"));
                }
                let repeat = bits.get(2).ok_or(ZlibError::Truncated)? as usize + 3;
                let previous = lengths[index - 1];
                fill(&mut lengths, &mut index, previous, repeat)?;
            }
            17 => {
                let repeat = bits.get(3).ok_or(ZlibError::Truncated)? as usize + 3;
                fill(&mut lengths, &mut index, 0, repeat)?;
            }
            18 => {
                let repeat = bits.get(7).ok_or(ZlibError::Truncated)? as usize + 11;
                fill(&mut lengths, &mut index, 0, repeat)?;
            }
            _ => return Err(ZlibError::Malformed("code length")),
        }
    }
    let lit = build(&lengths[..hlit])?;
    let dist = build(&lengths[hlit..])?;
    Ok((lit, dist))
}

fn fill(lengths: &mut [u8], index: &mut usize, value: u8, repeat: usize) -> Result<(), ZlibError> {
    if index.saturating_add(repeat) > lengths.len() {
        return Err(ZlibError::Malformed("code length"));
    }
    for _ in 0..repeat {
        lengths[*index] = value;
        *index += 1;
    }
    Ok(())
}

fn fixed_lit() -> Result<Huff, ZlibError> {
    let mut lengths = vec![8u8; 288];
    for len in &mut lengths[144..256] {
        *len = 9;
    }
    for len in &mut lengths[256..280] {
        *len = 7;
    }
    build(&lengths)
}

fn fixed_dist() -> Result<Huff, ZlibError> {
    build(&[5u8; 32])
}

/// Canonical fixed-Huffman code for a literal or length symbol.
fn fixed_lit_code(symbol: usize) -> (u16, u8) {
    match symbol {
        0..=143 => (48 + symbol as u16, 8),
        144..=255 => (400 + (symbol - 144) as u16, 9),
        256..=279 => ((symbol - 256) as u16, 7),
        280..=287 => (192 + (symbol - 280) as u16, 8),
        _ => (0, 0),
    }
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    bits: u8,
}

impl BitWriter {
    fn put(&mut self, value: u32, n: u8) {
        if n == 0 {
            return;
        }
        let mask = (1u32 << n) - 1;
        self.acc |= (value & mask) << self.bits;
        self.bits += n;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    fn put_code(&mut self, code: u16, len: u8) {
        self.put(u32::from(reverse_bits(code, len)), len);
    }

    fn finish(mut self, into: &mut Vec<u8>) {
        if self.bits > 0 {
            self.out.push(self.acc as u8);
        }
        into.extend_from_slice(&self.out);
    }
}

fn write_fixed(writer: &mut BitWriter, src: &[u8]) {
    let mut head = vec![-1i32; 32_768];
    let mut prev = vec![-1i32; src.len()];
    let mut index = 0usize;
    while index < src.len() {
        let mut matched = 0usize;
        let mut distance = 0u16;
        if index + 3 <= src.len() {
            let hash = hash3(src, index);
            let mut pos = head[hash];
            let mut steps = 0;
            while pos >= 0 && steps < 16 {
                let earlier = pos as usize;
                if earlier >= index || index - earlier > MAX_WINDOW {
                    break;
                }
                let len = match_at(src, index, earlier);
                if len > matched {
                    matched = len;
                    distance = (index - earlier) as u16;
                }
                if len >= 64 {
                    break;
                }
                pos = prev[earlier];
                steps += 1;
            }
            prev[index] = head[hash];
            head[hash] = index as i32;
        }
        if matched >= 3 {
            for step in 1..matched {
                let at = index + step;
                if at + 3 <= src.len() {
                    let hash = hash3(src, at);
                    prev[at] = head[hash];
                    head[hash] = at as i32;
                }
            }
            put_length(writer, matched as u16);
            put_distance(writer, distance);
            index += matched;
        } else {
            let (code, len) = fixed_lit_code(usize::from(src[index]));
            writer.put_code(code, len);
            index += 1;
        }
    }
}

fn hash3(src: &[u8], index: usize) -> usize {
    let mixed = (u32::from(src[index]) << 16)
        | (u32::from(src[index + 1]) << 8)
        | u32::from(src[index + 2]);
    (mixed.wrapping_mul(0x1e35_a7bd) >> 17) as usize & 0x7fff
}

fn match_at(src: &[u8], index: usize, earlier: usize) -> usize {
    let max = (src.len() - index).min(258);
    let mut len = 0;
    while len < max && src[index + len] == src[earlier + len] {
        len += 1;
    }
    len
}

fn put_length(writer: &mut BitWriter, len: u16) {
    let mut index = 0;
    while index + 1 < LEN_BASE.len() && LEN_BASE[index + 1] <= len {
        index += 1;
    }
    let (code, bits) = fixed_lit_code(257 + index);
    writer.put_code(code, bits);
    writer.put(u32::from(len - LEN_BASE[index]), LEN_EXTRA[index]);
}

fn put_distance(writer: &mut BitWriter, distance: u16) {
    let mut index = 0;
    while index + 1 < DIST_BASE.len() && DIST_BASE[index + 1] <= distance {
        index += 1;
    }
    writer.put_code(index as u16, 5);
    writer.put(u32::from(distance - DIST_BASE[index]), DIST_EXTRA[index]);
}

#[cfg(test)]
mod tests {
    use super::{deflate, inflate};
    use std::io::Write;
    use std::process::{Command, Stdio};

    #[test]
    fn a_stored_block_inflates_to_its_bytes() {
        let raw = b"Hi";
        let mut zlib = vec![0x78, 0x01, 0x01, 0x02, 0x00, 0xfd, 0xff, b'H', b'i'];
        let sum = super::adler32(raw);
        zlib.extend_from_slice(&sum.to_be_bytes());
        let out = inflate(&zlib, 16).expect("inflate");
        assert_eq!(out, raw);
    }

    #[test]
    fn inflate_matches_python_and_python_accepts_ours() {
        let samples: [&[u8]; 4] = [b"", b"a", b"hello png", &[7u8; 200]];
        for sample in samples {
            let foreign = python_zlib(sample, true).expect("python compress");
            let out = inflate(&foreign, sample.len() + 8).expect("inflate foreign");
            assert_eq!(out, sample);
            let ours = deflate(sample);
            let back = python_zlib(&ours, false).expect("python decompress");
            assert_eq!(back, sample);
            assert_eq!(inflate(&ours, sample.len() + 8).expect("self"), sample);
        }
        for len in 0..96 {
            let mixed: Vec<u8> = (0..len)
                .map(|index: usize| (index as u8).wrapping_mul(17))
                .collect();
            let repeated = vec![9u8; len];
            for sample in [&mixed, &repeated] {
                let ours = deflate(sample);
                assert_eq!(inflate(&ours, len).expect("length"), *sample);
            }
        }
    }

    #[test]
    fn a_short_buffer_and_a_bad_checksum_fail() {
        assert!(inflate(&[0x78], 8).is_err());
        let mut zlib = deflate(b"abc");
        let last = zlib.len() - 1;
        zlib[last] ^= 0xff;
        assert!(inflate(&zlib, 8).is_err());
    }

    fn python_zlib(data: &[u8], compress: bool) -> Option<Vec<u8>> {
        let program = if compress {
            "import sys,zlib; sys.stdout.buffer.write(zlib.compress(sys.stdin.buffer.read(), 9))"
        } else {
            "import sys,zlib; sys.stdout.buffer.write(zlib.decompress(sys.stdin.buffer.read()))"
        };
        let mut child = Command::new("python3")
            .args(["-c", program])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        child.stdin.as_mut()?.write_all(data).ok()?;
        drop(child.stdin.take());
        let output = child.wait_with_output().ok()?;
        if output.status.success() {
            Some(output.stdout)
        } else {
            None
        }
    }
}
