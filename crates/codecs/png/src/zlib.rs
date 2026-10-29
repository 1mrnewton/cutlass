//! zlib wrapper and deflate, for the IDAT stream and compressed metadata.
//!
//! The bit packing follows RFC 1951. The two-byte header and Adler-32 follow
//! RFC 1950. The encoder writes one final block, fixed or dynamic Huffman,
//! whichever is smaller. The decoder accepts stored, fixed, and dynamic
//! blocks, and rejects a preset dictionary or a window above 32 KiB.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

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
const LIT_FLAG: u32 = 0x8000_0000;
const CHAIN_FLOOR: usize = 16;
const CHAIN_CEILING: usize = 32;
const LAZY_LIMIT: usize = 16;
const STRONG_MATCH: usize = 8;
const WEAK_RUN: usize = 48;
const GOOD_LEN: usize = 32;
const GOOD_EXTRA: usize = 8;
const NICE_LEN: usize = 258;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZlibError {
    Truncated,
    Malformed(&'static str),
    Limit,
}

pub(crate) fn inflate(src: &[u8], limit: usize) -> Result<Vec<u8>, ZlibError> {
    let mut out = Vec::new();
    inflate_with(src, limit, |chunk| {
        out.extend_from_slice(chunk);
        Ok(())
    })?;
    Ok(out)
}

pub(crate) fn inflate_with(
    src: &[u8],
    limit: usize,
    emit: impl FnMut(&[u8]) -> Result<(), ZlibError>,
) -> Result<(), ZlibError> {
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
    let mut sink = Sink {
        history: vec![0; window],
        pos: 0,
        written: 0,
        window,
        count: 0,
        limit,
        s1: 1,
        s2: 0,
        block: 0,
        emit,
    };
    loop {
        let final_block = bits.get(1).ok_or(ZlibError::Truncated)?;
        let kind = bits.get(2).ok_or(ZlibError::Truncated)?;
        match kind {
            0 => stored(&mut bits, &mut sink)?,
            1 => {
                let lit = fixed_lit()?;
                let dist = fixed_dist()?;
                huffman(&mut bits, &mut sink, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                huffman(&mut bits, &mut sink, &lit, &dist)?;
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
    let stored_sum = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    if stored_sum != sink.adler() {
        return Err(ZlibError::Malformed("adler32"));
    }
    Ok(())
}

struct Sink<F> {
    history: Vec<u8>,
    pos: usize,
    written: usize,
    window: usize,
    count: usize,
    limit: usize,
    s1: u32,
    s2: u32,
    block: u32,
    emit: F,
}

impl<F> Sink<F>
where
    F: FnMut(&[u8]) -> Result<(), ZlibError>,
{
    fn push(&mut self, byte: u8) -> Result<(), ZlibError> {
        if self.count >= self.limit {
            return Err(ZlibError::Limit);
        }
        self.history[self.pos] = byte;
        self.pos += 1;
        if self.pos == self.window {
            self.pos = 0;
        }
        self.written += 1;
        self.count += 1;
        self.s1 += u32::from(byte);
        self.s2 += self.s1;
        self.block += 1;
        if self.block == 5552 {
            self.s1 %= ADLER_BASE;
            self.s2 %= ADLER_BASE;
            self.block = 0;
        }
        let one = [byte];
        (self.emit)(&one)
    }

    /// Record output bytes and hand them to the caller as one slice.
    ///
    /// `bytes` must fit in the ring before it wraps. A match copies at most
    /// one such run, so a repeated byte is a fill instead of one call per byte.
    fn commit(&mut self, bytes: &[u8]) -> Result<(), ZlibError> {
        let room = self.limit.saturating_sub(self.count);
        if bytes.len() > room {
            if room > 0 {
                self.write_all(&bytes[..room])?;
            }
            return Err(ZlibError::Limit);
        }
        self.write_all(bytes)
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), ZlibError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let n = bytes.len();
        let end = self.pos.saturating_add(n);
        if end > self.window {
            return Err(ZlibError::Malformed("distance"));
        }
        self.history[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        if self.pos == self.window {
            self.pos = 0;
        }
        self.written += n;
        self.count += n;
        self.absorb(bytes);
        (self.emit)(bytes)
    }

    fn absorb(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.s1 += u32::from(byte);
            self.s2 += self.s1;
            self.block += 1;
            if self.block == 5552 {
                self.s1 %= ADLER_BASE;
                self.s2 %= ADLER_BASE;
                self.block = 0;
            }
        }
    }

    fn adler(&mut self) -> u32 {
        self.s1 %= ADLER_BASE;
        self.s2 %= ADLER_BASE;
        self.block = 0;
        (self.s2 << 16) | self.s1
    }
}

pub(crate) fn deflate(src: &[u8]) -> Vec<u8> {
    deflate_spans(src, &[])
}

/// `spans` are `(end, stride)` in order. `end` is the exclusive offset of that
/// scanline group. A stride of 0 does not probe a previous line.
pub(crate) fn deflate_spans(src: &[u8], spans: &[(usize, usize)]) -> Vec<u8> {
    let tokens = tokenize(src, spans);
    let fixed = encode_fixed(&tokens);
    let body = match encode_dynamic(&tokens) {
        Some(dynamic) if dynamic.len() < fixed.len() => dynamic,
        _ => fixed,
    };
    let mut out = Vec::with_capacity(body.len() + 6);
    out.push(0x78);
    out.push(0x01);
    out.extend_from_slice(&body);
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

fn stored<F>(bits: &mut Bits<'_>, sink: &mut Sink<F>) -> Result<(), ZlibError>
where
    F: FnMut(&[u8]) -> Result<(), ZlibError>,
{
    bits.align();
    let len = bits.get(16).ok_or(ZlibError::Truncated)? as usize;
    let nlen = bits.get(16).ok_or(ZlibError::Truncated)? as u16;
    if nlen != !(len as u16) {
        return Err(ZlibError::Malformed("stored length"));
    }
    for _ in 0..len {
        let byte = bits.get(8).ok_or(ZlibError::Truncated)?;
        sink.push(byte as u8)?;
    }
    Ok(())
}

fn huffman<F>(
    bits: &mut Bits<'_>,
    sink: &mut Sink<F>,
    lit: &Huff,
    dist: &Huff,
) -> Result<(), ZlibError>
where
    F: FnMut(&[u8]) -> Result<(), ZlibError>,
{
    loop {
        let symbol = lit.symbol(bits)?;
        match symbol {
            0..=255 => sink.push(symbol as u8)?,
            256 => return Ok(()),
            257..=285 => {
                let index = usize::from(symbol - 257);
                let extra = LEN_EXTRA[index];
                let add = bits.get(extra).ok_or(ZlibError::Truncated)? as usize;
                let len = usize::from(LEN_BASE[index]) + add;
                let distance = distance(bits, dist)?;
                copy_match(sink, distance, len)?;
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

fn copy_match<F>(sink: &mut Sink<F>, distance: usize, mut len: usize) -> Result<(), ZlibError>
where
    F: FnMut(&[u8]) -> Result<(), ZlibError>,
{
    if distance == 0 || distance > sink.written || distance > sink.window || distance > MAX_WINDOW {
        return Err(ZlibError::Malformed("distance"));
    }
    let mut tmp = [0u8; 258];
    while len > 0 {
        let until_wrap = sink.window - sink.pos;
        if until_wrap == 0 {
            return Err(ZlibError::Malformed("distance"));
        }
        let start = (sink.pos + sink.window - distance) % sink.window;
        let n = if distance == 1 {
            len.min(until_wrap).min(tmp.len())
        } else {
            let until_source = sink.window - start;
            len.min(distance)
                .min(until_wrap)
                .min(until_source)
                .min(tmp.len())
        };
        if n == 0 {
            return Err(ZlibError::Malformed("distance"));
        }
        if distance == 1 {
            tmp[..n].fill(sink.history[start]);
        } else {
            tmp[..n].copy_from_slice(&sink.history[start..start + n]);
        }
        sink.commit(&tmp[..n])?;
        len -= n;
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

fn tokenize(src: &[u8], spans: &[(usize, usize)]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut head = vec![-1i32; 32_768];
    let mut prev = vec![-1i32; src.len()];
    let mut index = 0usize;
    let mut span_at = 0usize;
    let mut chain_limit = CHAIN_CEILING;
    let mut weak = 0usize;
    let mut held: Option<(usize, u16)> = None;
    while index < src.len() {
        while span_at < spans.len() && index >= spans[span_at].0 {
            span_at += 1;
        }
        let (span_start, stride) = span_at_index(spans, span_at, src.len());
        let (matched, distance) = best_match(
            src,
            index,
            &mut head,
            &mut prev,
            stride,
            span_start,
            chain_limit,
        );
        if let Some((prev_len, prev_dist)) = held {
            if matched > prev_len {
                tokens.push(LIT_FLAG | u32::from(src[index - 1]));
                held = Some((matched, distance));
                index += 1;
                continue;
            }
            note_strength(prev_len, &mut chain_limit, &mut weak);
            push_match(&mut tokens, prev_len, prev_dist);
            let start = index - 1;
            for step in 2..prev_len {
                insert_hash(src, start + step, &mut head, &mut prev);
            }
            index = start + prev_len;
            held = None;
            continue;
        }
        if (3..LAZY_LIMIT).contains(&matched) && index + 1 < src.len() {
            held = Some((matched, distance));
            index += 1;
            continue;
        }
        if matched >= 3 {
            note_strength(matched, &mut chain_limit, &mut weak);
            push_match(&mut tokens, matched, distance);
            for step in 1..matched {
                insert_hash(src, index + step, &mut head, &mut prev);
            }
            index += matched;
        } else {
            note_strength(0, &mut chain_limit, &mut weak);
            tokens.push(LIT_FLAG | u32::from(src[index]));
            index += 1;
        }
    }
    if let Some((prev_len, prev_dist)) = held {
        push_match(&mut tokens, prev_len, prev_dist);
        let start = index - 1;
        for step in 1..prev_len {
            insert_hash(src, start + step, &mut head, &mut prev);
        }
    }
    tokens
}

fn push_match(tokens: &mut Vec<u32>, len: usize, distance: u16) {
    tokens.push((u32::from(distance) << 9) | u32::try_from(len).unwrap_or(0));
}

fn note_strength(matched: usize, chain_limit: &mut usize, weak: &mut usize) {
    if matched >= STRONG_MATCH {
        *chain_limit = CHAIN_CEILING;
        *weak = 0;
        return;
    }
    *weak = weak.saturating_add(1);
    if *weak >= WEAK_RUN {
        *chain_limit = CHAIN_FLOOR;
    }
}

fn span_at_index(spans: &[(usize, usize)], span_at: usize, len: usize) -> (usize, usize) {
    if spans.is_empty() {
        return (0, 0);
    }
    let span_start = if span_at == 0 {
        0
    } else {
        spans[span_at - 1].0
    };
    if span_at >= spans.len() {
        return (span_start.min(len), 0);
    }
    (span_start, spans[span_at].1)
}

fn best_match(
    src: &[u8],
    index: usize,
    head: &mut [i32],
    prev: &mut [i32],
    stride: usize,
    span_start: usize,
    chain_limit: usize,
) -> (usize, u16) {
    let mut matched = 0usize;
    let mut distance = 0u16;
    if index + 3 > src.len() {
        return (0, 0);
    }
    let hash = hash3(src, index);
    let mut pos = head[hash];
    let mut steps = 0usize;
    // The newest copy is usually the run. One comparison catches it before the
    // previous scanline, so a repeated byte keeps the short distance.
    if let Some(earlier) = chain_earlier(index, pos) {
        let len = record_match(src, index, earlier, &mut matched, &mut distance);
        pos = prev[earlier];
        steps = 1;
        if len >= NICE_LEN {
            prev[index] = head[hash];
            head[hash] = index as i32;
            return (matched, distance);
        }
    }
    if matched < NICE_LEN {
        probe_rows(src, index, stride, span_start, &mut matched, &mut distance);
    }
    let mut limit = chain_limit;
    if matched >= NICE_LEN {
        limit = steps;
    } else if matched >= GOOD_LEN {
        limit = steps + GOOD_EXTRA;
    }
    while pos >= 0 && steps < limit {
        let Some(earlier) = chain_earlier(index, pos) else {
            break;
        };
        let len = record_match(src, index, earlier, &mut matched, &mut distance);
        if len >= NICE_LEN {
            break;
        }
        if matched >= GOOD_LEN && limit > steps + GOOD_EXTRA {
            limit = steps + GOOD_EXTRA;
        }
        pos = prev[earlier];
        steps += 1;
    }
    prev[index] = head[hash];
    head[hash] = index as i32;
    (matched, distance)
}

fn chain_earlier(index: usize, pos: i32) -> Option<usize> {
    if pos < 0 {
        return None;
    }
    let earlier = pos as usize;
    if earlier >= index || index - earlier > MAX_WINDOW {
        None
    } else {
        Some(earlier)
    }
}

fn record_match(
    src: &[u8],
    index: usize,
    earlier: usize,
    matched: &mut usize,
    distance: &mut u16,
) -> usize {
    let len = match_at(src, index, earlier);
    let back = index - earlier;
    if worth_encoding(len, back) && better_match(len, back, *matched, *distance) {
        *matched = len;
        *distance = u16::try_from(back).unwrap_or(0);
    }
    len
}

fn probe_rows(
    src: &[u8],
    index: usize,
    stride: usize,
    span_start: usize,
    matched: &mut usize,
    distance: &mut u16,
) {
    if stride == 0 {
        return;
    }
    for rows in 1..=4 {
        let back = stride.saturating_mul(rows);
        if back == 0 || back > MAX_WINDOW || index < span_start.saturating_add(back) {
            break;
        }
        let len = match_at(src, index, index - back);
        if worth_encoding(len, back) && better_match(len, back, *matched, *distance) {
            *matched = len;
            *distance = u16::try_from(back).unwrap_or(0);
        }
        if len >= NICE_LEN {
            break;
        }
    }
}

fn better_match(len: usize, back: usize, matched: usize, distance: u16) -> bool {
    len > matched || (len == matched && len >= 3 && back < usize::from(distance))
}

/// A length-3 match far from here costs more bits than three literals.
fn worth_encoding(len: usize, back: usize) -> bool {
    len > 3 || (len == 3 && back <= 4_096)
}

fn insert_hash(src: &[u8], index: usize, head: &mut [i32], prev: &mut [i32]) {
    if index + 3 > src.len() {
        return;
    }
    let hash = hash3(src, index);
    prev[index] = head[hash];
    head[hash] = index as i32;
}

fn encode_fixed(tokens: &[u32]) -> Vec<u8> {
    let mut writer = BitWriter::default();
    writer.put(1, 1);
    writer.put(1, 2);
    for &token in tokens {
        write_fixed_token(&mut writer, token);
    }
    let (code, len) = fixed_lit_code(256);
    writer.put_code(code, len);
    let mut out = Vec::new();
    writer.finish(&mut out);
    out
}

fn write_fixed_token(writer: &mut BitWriter, token: u32) {
    if token & LIT_FLAG != 0 {
        let (code, len) = fixed_lit_code((token & 0xff) as usize);
        writer.put_code(code, len);
        return;
    }
    let (symbol, extra, bits) = length_code((token & 0x1ff) as u16);
    let (code, len) = fixed_lit_code(symbol);
    writer.put_code(code, len);
    writer.put(extra, bits);
    let (symbol, extra, bits) = distance_code((token >> 9) as u16);
    writer.put_code(symbol as u16, 5);
    writer.put(extra, bits);
}

fn encode_dynamic(tokens: &[u32]) -> Option<Vec<u8>> {
    let (lit_freq, dist_freq) = frequencies(tokens);
    let lit_len = huffman_lengths(&lit_freq, MAX_BITS);
    let dist_len = huffman_lengths(&dist_freq, MAX_BITS);
    if !covers(&lit_freq, &lit_len) || !covers(&dist_freq, &dist_len) {
        return None;
    }
    let hlit = used_count(&lit_len, 257);
    let hdist = used_count(&dist_len, 1);
    let mut combined = lit_len[..hlit].to_vec();
    combined.extend_from_slice(&dist_len[..hdist]);
    let runs = encode_runs(&combined);
    let mut code_freq = vec![0u32; 19];
    for &(symbol, _) in &runs {
        code_freq[usize::from(symbol)] += 1;
    }
    let code_len = huffman_lengths(&code_freq, 7);
    if !covers(&code_freq, &code_len) || !tokens_fit(tokens, &lit_len, &dist_len) {
        return None;
    }
    let lit_huff = build(&lit_len).ok()?;
    let dist_huff = build(&dist_len).ok()?;
    let code_huff = build(&code_len).ok()?;
    let mut send = 4usize;
    for (index, &ord) in CL_ORDER.iter().enumerate() {
        if code_len[ord] != 0 {
            send = send.max(index + 1);
        }
    }
    let mut writer = BitWriter::default();
    writer.put(1, 1);
    writer.put(2, 2);
    writer.put(u32::try_from(hlit - 257).unwrap_or(0), 5);
    writer.put(u32::try_from(hdist - 1).unwrap_or(0), 5);
    writer.put(u32::try_from(send - 4).unwrap_or(0), 4);
    for &ord in CL_ORDER.iter().take(send) {
        writer.put(u32::from(code_len[ord]), 3);
    }
    for &(symbol, extra) in &runs {
        let index = usize::from(symbol);
        let len = code_huff.lengths.get(index).copied().unwrap_or(0);
        let code = code_huff.codes.get(index).copied().unwrap_or(0);
        if len == 0 {
            return None;
        }
        writer.put_code(code, len);
        writer.put(u32::from(extra), run_extra_bits(symbol));
    }
    for &token in tokens {
        write_dynamic_token(&mut writer, token, &lit_huff, &dist_huff)?;
    }
    let eob_len = lit_huff.lengths.get(256).copied().unwrap_or(0);
    let eob_code = lit_huff.codes.get(256).copied().unwrap_or(0);
    if eob_len == 0 {
        return None;
    }
    writer.put_code(eob_code, eob_len);
    let mut out = Vec::new();
    writer.finish(&mut out);
    Some(out)
}

fn write_dynamic_token(writer: &mut BitWriter, token: u32, lit: &Huff, dist: &Huff) -> Option<()> {
    if token & LIT_FLAG != 0 {
        put_huff(writer, lit, (token & 0xff) as usize)?;
        return Some(());
    }
    let (symbol, extra, bits) = length_code((token & 0x1ff) as u16);
    put_huff(writer, lit, symbol)?;
    writer.put(extra, bits);
    let (symbol, extra, bits) = distance_code((token >> 9) as u16);
    put_huff(writer, dist, symbol)?;
    writer.put(extra, bits);
    Some(())
}

fn put_huff(writer: &mut BitWriter, huff: &Huff, symbol: usize) -> Option<()> {
    let len = huff.lengths.get(symbol).copied().unwrap_or(0);
    let code = huff.codes.get(symbol).copied().unwrap_or(0);
    if len == 0 {
        return None;
    }
    writer.put_code(code, len);
    Some(())
}

fn frequencies(tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut lit = vec![0u32; 286];
    let mut dist = vec![0u32; 30];
    lit[256] = 1;
    for &token in tokens {
        if token & LIT_FLAG != 0 {
            lit[(token & 0xff) as usize] += 1;
            continue;
        }
        let (symbol, _, _) = length_code((token & 0x1ff) as u16);
        let (distance, _, _) = distance_code((token >> 9) as u16);
        lit[symbol] += 1;
        dist[distance] += 1;
    }
    (lit, dist)
}

fn covers(freq: &[u32], lengths: &[u8]) -> bool {
    freq.iter()
        .zip(lengths)
        .all(|(&weight, &len)| weight == 0 || len > 0)
}

fn tokens_fit(tokens: &[u32], lit: &[u8], dist: &[u8]) -> bool {
    if lit.get(256).copied().unwrap_or(0) == 0 {
        return false;
    }
    for &token in tokens {
        if token & LIT_FLAG != 0 {
            if lit[(token & 0xff) as usize] == 0 {
                return false;
            }
            continue;
        }
        let (symbol, _, _) = length_code((token & 0x1ff) as u16);
        let (distance, _, _) = distance_code((token >> 9) as u16);
        if lit[symbol] == 0 || dist[distance] == 0 {
            return false;
        }
    }
    true
}

fn used_count(lengths: &[u8], min: usize) -> usize {
    let mut count = lengths.len();
    while count > min && lengths[count - 1] == 0 {
        count -= 1;
    }
    count.max(min)
}

fn huffman_lengths(freq: &[u32], max_bits: usize) -> Vec<u8> {
    let count = freq.len();
    let mut lengths = vec![0u8; count];
    let mut heap = BinaryHeap::new();
    for (symbol, &weight) in freq.iter().enumerate() {
        if weight > 0 {
            heap.push(Reverse((weight, symbol)));
        }
    }
    if heap.len() <= 1 {
        if let Some(Reverse((_, symbol))) = heap.pop() {
            lengths[symbol] = 1;
        }
        return lengths;
    }
    let mut children = Vec::new();
    let mut next = count;
    while heap.len() >= 2 {
        let Some(Reverse((left_freq, left))) = heap.pop() else {
            break;
        };
        let Some(Reverse((right_freq, right))) = heap.pop() else {
            break;
        };
        children.push((left, right));
        let parent = next;
        next += 1;
        heap.push(Reverse((left_freq.saturating_add(right_freq), parent)));
    }
    let Some(Reverse((_, root))) = heap.pop() else {
        return lengths;
    };
    let mut stack = vec![(root, 0u32)];
    while let Some((node, depth)) = stack.pop() {
        if node < count {
            lengths[node] = u8::try_from(depth.min(255)).unwrap_or(255);
            continue;
        }
        let Some(&(left, right)) = children.get(node - count) else {
            continue;
        };
        stack.push((left, depth.saturating_add(1)));
        stack.push((right, depth.saturating_add(1)));
    }
    if lengths.iter().any(|len| usize::from(*len) > max_bits) {
        cap_lengths(&mut lengths, max_bits);
    }
    lengths
}

/// Keep a prefix code inside the deflate length limit.
///
/// Shortening an over-long code spends more of the code space. The counts are
/// then repaired until they satisfy the Kraft inequality.
fn cap_lengths(lengths: &mut [u8], max_bits: usize) {
    let mut count = vec![0u32; lengths.len() + 1];
    for &len in lengths.iter() {
        if len > 0 {
            let at = usize::from(len).min(count.len() - 1);
            count[at] += 1;
        }
    }
    let mut overflow = 0u32;
    for extra in count.iter_mut().skip(max_bits + 1) {
        overflow += *extra;
        *extra = 0;
    }
    count[max_bits] += overflow;
    let mut total = 0u32;
    for (bits, tally) in count.iter().enumerate().take(max_bits + 1).skip(1) {
        total += tally << (max_bits - bits);
    }
    let target = 1u32 << max_bits;
    while total > target && count[max_bits] > 0 {
        count[max_bits] -= 1;
        let mut bits = max_bits - 1;
        while bits > 0 && count[bits] == 0 {
            bits -= 1;
        }
        if bits == 0 {
            break;
        }
        count[bits] -= 1;
        count[bits + 1] += 2;
        total -= 1;
    }
    let mut symbols: Vec<usize> = (0..lengths.len()).filter(|&sym| lengths[sym] > 0).collect();
    symbols.sort_by(|&left, &right| {
        lengths[left]
            .cmp(&lengths[right])
            .then_with(|| left.cmp(&right))
    });
    let mut assigned = Vec::new();
    for (bits, &tally) in count.iter().enumerate().take(max_bits + 1).skip(1) {
        for _ in 0..tally {
            assigned.push(u8::try_from(bits).unwrap_or(u8::MAX));
        }
    }
    for (symbol, len) in symbols.iter().zip(&assigned) {
        lengths[*symbol] = *len;
    }
    let cap = u8::try_from(max_bits).unwrap_or(u8::MAX);
    for symbol in symbols.iter().skip(assigned.len()) {
        lengths[*symbol] = cap;
    }
}

fn encode_runs(lengths: &[u8]) -> Vec<(u8, u16)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < lengths.len() {
        let value = lengths[index];
        let mut run = 1usize;
        while index + run < lengths.len() && lengths[index + run] == value && run < 138 {
            run += 1;
        }
        if value == 0 {
            push_zero_run(&mut runs, run);
        } else {
            push_repeat(&mut runs, value, run);
        }
        index += run;
    }
    runs
}

fn push_zero_run(runs: &mut Vec<(u8, u16)>, mut left: usize) {
    while left > 0 {
        if left < 3 {
            runs.push((0, 0));
            left -= 1;
        } else if left < 11 {
            let take = left.min(10);
            runs.push((17, u16::try_from(take - 3).unwrap_or(0)));
            left -= take;
        } else {
            let take = left.min(138);
            runs.push((18, u16::try_from(take - 11).unwrap_or(0)));
            left -= take;
        }
    }
}

fn push_repeat(runs: &mut Vec<(u8, u16)>, value: u8, run: usize) {
    runs.push((value, 0));
    let mut left = run - 1;
    while left >= 3 {
        let take = left.min(6);
        runs.push((16, u16::try_from(take - 3).unwrap_or(0)));
        left -= take;
    }
    while left > 0 {
        runs.push((value, 0));
        left -= 1;
    }
}

fn run_extra_bits(symbol: u8) -> u8 {
    match symbol {
        16 => 2,
        17 => 3,
        18 => 7,
        _ => 0,
    }
}

fn length_code(len: u16) -> (usize, u32, u8) {
    let mut index = 0;
    while index + 1 < LEN_BASE.len() && LEN_BASE[index + 1] <= len {
        index += 1;
    }
    (
        257 + index,
        u32::from(len - LEN_BASE[index]),
        LEN_EXTRA[index],
    )
}

fn distance_code(distance: u16) -> (usize, u32, u8) {
    let mut index = 0;
    while index + 1 < DIST_BASE.len() && DIST_BASE[index + 1] <= distance {
        index += 1;
    }
    (
        index,
        u32::from(distance - DIST_BASE[index]),
        DIST_EXTRA[index],
    )
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
        let repeated = vec![9u8; 40_000];
        let foreign = python_zlib(&repeated, true).expect("python long run");
        assert_eq!(
            inflate(&foreign, repeated.len()).expect("long run"),
            repeated
        );
        let mixed: Vec<u8> = (0..40_000)
            .map(|index| (index as u8).wrapping_mul(17))
            .collect();
        let foreign = python_zlib(&mixed, true).expect("python long mix");
        assert_eq!(inflate(&foreign, mixed.len()).expect("long mix"), mixed);
    }

    #[test]
    fn dynamic_codes_beat_fixed_codes_on_a_small_alphabet() {
        let sample: Vec<u8> = (0..8_000).map(|index| (index % 7) as u8).collect();
        let tokens = super::tokenize(&sample, &[]);
        let fixed = super::encode_fixed(&tokens);
        let dynamic = super::encode_dynamic(&tokens).expect("dynamic block");
        assert!(
            dynamic.len() < fixed.len(),
            "dynamic {} fixed {}",
            dynamic.len(),
            fixed.len()
        );
        let ours = deflate(&sample);
        assert_eq!(ours.len(), dynamic.len() + 6);
        assert_eq!(inflate(&ours, sample.len()).expect("inflate"), sample);
        let back = python_zlib(&ours, false).expect("python");
        assert_eq!(back, sample);
    }

    #[test]
    fn a_previous_scanline_is_found_past_the_hash_chain() {
        let mut row = vec![9u8; 760];
        for (index, byte) in row.iter_mut().enumerate().take(80) {
            *byte = 40u8.wrapping_add(u8::try_from(index).unwrap_or(0));
        }
        row[0] = 1;
        row[1] = 2;
        row[2] = 3;
        for decoy in 0..160 {
            let at = 80 + decoy * 4;
            row[at] = 1;
            row[at + 1] = 2;
            row[at + 2] = 3;
            row[at + 3] = 200;
        }
        let mut image = Vec::new();
        for _ in 0..30 {
            image.extend_from_slice(&row);
        }
        let plain = deflate(&image);
        let spanned = super::deflate_spans(&image, &[(image.len(), row.len())]);
        assert!(
            spanned.len() < plain.len(),
            "span {} plain {}",
            spanned.len(),
            plain.len()
        );
        assert_eq!(inflate(&spanned, image.len()).expect("inflate"), image);
        let back = python_zlib(&spanned, false).expect("python");
        assert_eq!(back, image);
    }

    #[test]
    fn a_short_match_does_not_hide_a_longer_one() {
        let sample = b"ABC_BCDEFGHIJKLMNOP_ABCDEFGHIJKLMNOP";
        let tokens = super::tokenize(sample, &[]);
        let long = tokens
            .iter()
            .any(|token| token & super::LIT_FLAG == 0 && (token & 0x1ff) >= 15);
        assert!(long, "{tokens:?}");
        let ours = deflate(sample);
        assert_eq!(inflate(&ours, sample.len()).expect("inflate"), sample);
        let back = python_zlib(&ours, false).expect("python");
        assert_eq!(back, sample);
    }

    #[test]
    fn a_far_three_byte_match_is_left_as_literals() {
        let mut sample: Vec<u8> = (0..5_000)
            .map(|index| (index as u8).wrapping_mul(3).wrapping_add(11))
            .collect();
        sample[10] = 9;
        sample[11] = 8;
        sample[12] = 7;
        sample[4_500] = 9;
        sample[4_501] = 8;
        sample[4_502] = 7;
        let tokens = super::tokenize(&sample, &[]);
        for token in &tokens {
            if token & super::LIT_FLAG == 0 {
                let len = token & 0x1ff;
                let distance = token >> 9;
                assert!(len > 3 || distance <= 4_096, "{token:#x}");
            }
        }
        let ours = deflate(&sample);
        assert_eq!(inflate(&ours, sample.len()).expect("inflate"), sample);
        let back = python_zlib(&ours, false).expect("python");
        assert_eq!(back, sample);
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
