//! Building and writing VP8L prefix codes (RFC 9649 section 3.7.2).

use super::bits::BitWriter;
use super::decode::CODE_LENGTH_ORDER;
use super::huffman::reverse;

/// Longest code-length code the format can store (three bits each).
const MAX_CODE_LENGTH_CODE: u8 = 7;

/// Code lengths for `counts`, none longer than `limit`. Unused symbols get
/// 0. A lone used symbol gets length 1, which the decoder reads in zero
/// bits.
pub(crate) fn lengths(counts: &[u32], limit: u8) -> Vec<u8> {
    let mut lengths = vec![0u8; counts.len()];
    let used: Vec<usize> = (0..counts.len()).filter(|&s| counts[s] > 0).collect();
    match used.len() {
        0 => return lengths,
        1 => {
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }
    // Raise the smallest weights until the tree fits the limit.
    let mut floor = 1u64;
    loop {
        let weights: Vec<u64> = used
            .iter()
            .map(|&s| u64::from(counts[s]).max(floor))
            .collect();
        let depths = depths(&weights);
        if depths.iter().all(|&d| d <= limit) {
            for (&symbol, depth) in used.iter().zip(depths) {
                lengths[symbol] = depth;
            }
            return lengths;
        }
        floor *= 2;
    }
}

/// Huffman depths of `weights` (two or more), by the two-queue method.
fn depths(weights: &[u64]) -> Vec<u8> {
    let n = weights.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (weights[i], i));
    // Nodes 0..n are leaves; n.. are internal. `parent` links upward.
    let mut weight: Vec<u64> = weights.to_vec();
    let mut parent = vec![usize::MAX; 2 * n - 1];
    let mut leaves = order.into_iter().peekable();
    let mut internal = std::collections::VecDeque::with_capacity(n);
    for next in n..2 * n - 1 {
        let (Some(a), Some(b)) = (
            lightest(&weight, &mut leaves, &mut internal),
            lightest(&weight, &mut leaves, &mut internal),
        ) else {
            break;
        };
        weight.push(weight[a] + weight[b]);
        parent[a] = next;
        parent[b] = next;
        internal.push_back(next);
    }
    // Depth of each node: parents come after children.
    let mut depth = vec![0u8; 2 * n - 1];
    for node in (0..2 * n - 2).rev() {
        depth[node] = depth[parent[node]].saturating_add(1);
    }
    depth.truncate(n);
    depth
}

/// The lighter of the next unused leaf and the oldest internal node.
fn lightest(
    weight: &[u64],
    leaves: &mut std::iter::Peekable<std::vec::IntoIter<usize>>,
    internal: &mut std::collections::VecDeque<usize>,
) -> Option<usize> {
    match (leaves.peek(), internal.front()) {
        (Some(&leaf), Some(&node)) if weight[leaf] <= weight[node] => leaves.next(),
        (Some(_), None) => leaves.next(),
        _ => internal.pop_front(),
    }
}

/// Codes for writing: each symbol's bits in the order the reader takes
/// them, and how many. A code with one used symbol writes nothing.
pub(crate) struct Code {
    pub bits: Vec<u16>,
    pub lengths: Vec<u8>,
}

impl Code {
    pub(crate) fn from_lengths(lengths: &[u8]) -> Self {
        let used = lengths.iter().filter(|&&l| l > 0).count();
        let mut bits = vec![0u16; lengths.len()];
        let mut out_lengths = lengths.to_vec();
        if used <= 1 {
            out_lengths.fill(0);
            return Self {
                bits,
                lengths: out_lengths,
            };
        }
        let mut count = [0u16; 16];
        for &length in lengths {
            count[usize::from(length)] += 1;
        }
        count[0] = 0;
        let mut next = [0u16; 16];
        let mut code = 0u16;
        for length in 1..16 {
            code = (code + count[length - 1]) << 1;
            next[length] = code;
        }
        for (symbol, &length) in lengths.iter().enumerate() {
            if length > 0 {
                let value = next[usize::from(length)];
                next[usize::from(length)] += 1;
                bits[symbol] = reverse(u32::from(value), u32::from(length)) as u16;
            }
        }
        Self {
            bits,
            lengths: out_lengths,
        }
    }

    #[inline]
    pub(crate) fn put(&self, writer: &mut BitWriter, symbol: usize) {
        writer.write(
            u32::from(self.bits[symbol]),
            u32::from(self.lengths[symbol]),
        );
    }
}

/// Write the code for `lengths` and return it for the symbols.
pub(crate) fn write_code(writer: &mut BitWriter, lengths: &[u8]) -> Code {
    let used: Vec<usize> = (0..lengths.len()).filter(|&s| lengths[s] > 0).collect();
    if used.len() <= 2 && used.iter().all(|&s| s < 256) {
        // Simple code: one or two 1- or 8-bit symbols.
        writer.write(1, 1);
        let first = used.first().copied().unwrap_or(0);
        writer.write(u32::from(used.len() == 2), 1);
        if first < 2 {
            writer.write(0, 1);
            writer.write(first as u32, 1);
        } else {
            writer.write(1, 1);
            writer.write(first as u32, 8);
        }
        if let Some(&second) = used.get(1) {
            writer.write(second as u32, 8);
        }
        let mut simple = vec![0u8; lengths.len()];
        for &symbol in &used {
            simple[symbol] = 1;
        }
        return Code::from_lengths(&simple);
    }
    writer.write(0, 1);
    let tokens = length_tokens(lengths);
    let mut histogram = [0u32; 19];
    for &(symbol, _) in &tokens {
        histogram[usize::from(symbol)] += 1;
    }
    let cl_lengths = self::lengths(&histogram, MAX_CODE_LENGTH_CODE);
    let mut count = CODE_LENGTH_ORDER
        .iter()
        .rposition(|&symbol| cl_lengths[symbol] > 0)
        .map_or(0, |at| at + 1);
    count = count.max(4);
    writer.write((count - 4) as u32, 4);
    for &symbol in &CODE_LENGTH_ORDER[..count] {
        writer.write(u32::from(cl_lengths[symbol]), 3);
    }
    let cl_code = Code::from_lengths(&cl_lengths);
    // Drop trailing zero runs: the reader stops after `max_symbol` tokens
    // and leaves the rest of the lengths at zero.
    let trimmed = tokens
        .iter()
        .rposition(|&(symbol, _)| !matches!(symbol, 0 | 17 | 18))
        .map_or(0, |at| at + 1);
    let mut write_count = tokens.len();
    if trimmed < tokens.len() && trimmed >= 2 {
        let value = (trimmed - 2) as u32;
        let needed = 32 - value.leading_zeros();
        let width = needed.max(2).div_ceil(2) * 2;
        writer.write(1, 1);
        writer.write((width - 2) / 2, 3);
        writer.write(value, width);
        write_count = trimmed;
    } else {
        writer.write(0, 1);
    }
    for &(symbol, extra) in &tokens[..write_count] {
        cl_code.put(writer, usize::from(symbol));
        match symbol {
            16 => writer.write(u32::from(extra), 2),
            17 => writer.write(u32::from(extra), 3),
            18 => writer.write(u32::from(extra), 7),
            _ => {}
        }
    }
    Code::from_lengths(lengths)
}

/// Run-length tokens of a length list: `(symbol, extra)` with 0 to 15
/// literal, 16 repeating the previous non-zero length, 17 and 18 zeros.
fn length_tokens(lengths: &[u8]) -> Vec<(u8, u8)> {
    let mut tokens = Vec::new();
    let mut previous = 8u8;
    let mut at = 0;
    while at < lengths.len() {
        let value = lengths[at];
        let run = lengths[at..].iter().take_while(|&&l| l == value).count();
        if value == 0 {
            let mut left = run;
            while left > 0 {
                if left >= 11 {
                    let take = left.min(138);
                    tokens.push((18, (take - 11) as u8));
                    left -= take;
                } else if left >= 3 {
                    tokens.push((17, (left - 3) as u8));
                    left = 0;
                } else {
                    tokens.push((0, 0));
                    left -= 1;
                }
            }
        } else {
            let mut left = run;
            if value != previous {
                tokens.push((value, 0));
                previous = value;
                left -= 1;
            }
            while left > 0 {
                if left >= 3 {
                    let take = left.min(6);
                    tokens.push((16, (take - 3) as u8));
                    left -= take;
                } else {
                    tokens.push((value, 0));
                    left -= 1;
                }
            }
        }
        at += run;
    }
    tokens
}

/// Estimated bits to store a histogram's symbols with its own code, and
/// the code itself: the entropy plus a per-run cost for the lengths.
pub(crate) fn cost(counts: &[u32]) -> f64 {
    let mut total = 0u64;
    let mut nonzero = 0usize;
    let mut entropy = 0.0f64;
    for &count in counts {
        if count > 0 {
            total += u64::from(count);
            nonzero += 1;
            entropy -= f64::from(count) * f64::from(count).log2();
        }
    }
    if nonzero <= 1 {
        return 12.0;
    }
    let total = total as f64;
    entropy += total * total.log2();
    // Runs of zero lengths are cheap; each used symbol costs about three
    // bits of code-length code.
    let mut runs = 0usize;
    let mut zero = false;
    for &count in counts {
        if count == 0 && !zero {
            runs += 1;
        }
        zero = count == 0;
    }
    entropy + 3.0 * nonzero as f64 + 6.0 * runs as f64 + 20.0
}

#[cfg(test)]
mod tests {
    use super::{BitWriter, lengths, write_code};
    use crate::lossless::bits::BitReader;
    use crate::lossless::huffman::Code as ReadCode;

    #[test]
    fn lengths_respect_the_limit_and_stay_complete() {
        let mut counts = vec![0u32; 40];
        for (i, count) in counts.iter_mut().enumerate() {
            *count = 1 << (i % 30);
        }
        let lengths = lengths(&counts, 15);
        assert!(lengths.iter().all(|&l| (1..=15).contains(&l)));
        let kraft: f64 = lengths.iter().map(|&l| 0.5f64.powi(i32::from(l))).sum();
        assert!((kraft - 1.0).abs() < 1e-9);
    }

    #[test]
    fn written_codes_read_back() {
        for counts in [
            vec![5u32, 0, 0, 7],
            vec![0, 0, 9],
            vec![0; 10],
            {
                let mut c = vec![0u32; 300];
                c[280] = 4;
                c
            },
            (0..280).map(|i| (i * 7 % 13) as u32).collect(),
        ] {
            let lengths = lengths(&counts, 15);
            let mut writer = BitWriter::default();
            let code = write_code(&mut writer, &lengths);
            let symbols: Vec<usize> = (0..counts.len()).filter(|&s| counts[s] > 0).collect();
            for &symbol in &symbols {
                code.put(&mut writer, symbol);
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            let read = read_code(&mut reader, counts.len());
            for &symbol in &symbols {
                assert_eq!(usize::from(read.read(&mut reader).unwrap()), symbol);
            }
        }
    }

    fn read_code(reader: &mut BitReader<'_>, alphabet: usize) -> ReadCode {
        crate::lossless::decode::read_code(reader, alphabet).unwrap()
    }
}
