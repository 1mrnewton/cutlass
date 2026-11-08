//! Codebooks the encoder writes. Lengths are Huffman codes over a model of
//! how often each entry is used, limited to [`MAX_LENGTH`] bits. A vector
//! book is a lattice (lookup type 1): every entry is a vector of integers
//! `step * (digit - side / 2)`, so a residue value is coded exactly and
//! the entry is found by arithmetic, not search.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::bits::{BitWriter, ilog};
use crate::codebook::{BookSpec, Lookup, assign_codewords, float32_pack, reverse};
use cutlass_decode::CodecError;

/// The longest codeword the encoder writes.
pub(crate) const MAX_LENGTH: u8 = 24;

/// Huffman lengths for `weights`, none longer than `limit`. A zero weight
/// is an unused entry (length 0). Two used entries at least; the code is
/// complete, as section 3.2.1 requires.
pub(crate) fn huffman_lengths(weights: &[f64], limit: u8) -> Vec<u8> {
    let used: Vec<usize> = (0..weights.len())
        .filter(|&i| weights[i] > 0.0 && weights[i].is_finite())
        .collect();
    let mut lengths = vec![0_u8; weights.len()];
    if used.len() < 2 {
        if let Some(&only) = used.first() {
            lengths[only] = 1;
        }
        return lengths;
    }
    let total: f64 = used.iter().map(|&i| weights[i]).sum();
    // Integer weights keep the tree the same on every platform.
    let scaled: Vec<u64> = used
        .iter()
        .map(|&i| ((weights[i] / total) * (1_u64 << 40) as f64) as u64 + 1)
        .collect();
    let mut parent = vec![usize::MAX; used.len() * 2];
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = scaled
        .iter()
        .enumerate()
        .map(|(node, &weight)| Reverse((weight, node)))
        .collect();
    let mut next = used.len();
    while heap.len() > 1 {
        let (Some(Reverse((wa, a))), Some(Reverse((wb, b)))) = (heap.pop(), heap.pop()) else {
            break;
        };
        parent[a] = next;
        parent[b] = next;
        heap.push(Reverse((wa + wb, next)));
        next += 1;
    }
    let root = next - 1;
    let mut depth = vec![0_usize; next];
    for node in (0..root).rev() {
        depth[node] = depth[parent[node]] + 1;
    }
    // Count leaves per length, then move the ones past the limit up
    // (ITU T.81 annex K.3): the code stays complete.
    let deepest = (0..used.len()).map(|leaf| depth[leaf]).max().unwrap_or(1);
    let mut count = vec![0_usize; deepest.max(usize::from(limit)) + 2];
    for leaf in 0..used.len() {
        count[depth[leaf]] += 1;
    }
    let limit = usize::from(limit);
    let mut i = count.len() - 1;
    while i > limit {
        while count[i] >= 2 {
            let mut j = i - 2;
            while j > 0 && count[j] == 0 {
                j -= 1;
            }
            if count[j] == 0 {
                break;
            }
            count[i] -= 2;
            count[i - 1] += 1;
            count[j + 1] += 2;
            count[j] -= 1;
        }
        i -= 1;
    }
    // The most used entries take the shortest codes.
    let mut order: Vec<usize> = (0..used.len()).collect();
    order.sort_by(|&a, &b| scaled[b].cmp(&scaled[a]).then(a.cmp(&b)));
    let mut at = 0;
    for (len, &n) in count.iter().enumerate() {
        for _ in 0..n {
            if let Some(&leaf) = order.get(at) {
                lengths[used[leaf]] = len as u8;
            }
            at += 1;
        }
    }
    lengths
}

/// Codewords of one book, bit-reversed for [`BitWriter::write`].
#[derive(Clone, Debug)]
pub(crate) struct Coder {
    codes: Vec<(u32, u8)>,
}

impl Coder {
    pub(crate) fn new(lengths: &[u8]) -> Result<Self, CodecError> {
        let codes = assign_codewords(lengths)?
            .into_iter()
            .map(|(code, len)| (reverse(code, u32::from(len)), len))
            .collect();
        Ok(Self { codes })
    }

    /// Bits of `entry`'s codeword; 0 for an unused entry.
    pub(crate) fn bits(&self, entry: usize) -> u32 {
        self.codes.get(entry).map_or(0, |&(_, len)| u32::from(len))
    }

    /// Writes `entry`, which must be used.
    pub(crate) fn write(&self, writer: &mut BitWriter, entry: usize) {
        if let Some(&(code, len)) = self.codes.get(entry) {
            writer.write(code, u32::from(len));
        }
    }
}

/// A vector book of `dimensions` integers, each `step * (digit - side / 2)`
/// for `digit < side`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lattice {
    pub dimensions: usize,
    /// Values per dimension, odd.
    pub side: usize,
    pub step: i32,
}

impl Lattice {
    pub(crate) fn entries(&self) -> usize {
        self.side.pow(self.dimensions as u32)
    }

    /// The largest magnitude one dimension holds.
    #[cfg(test)]
    pub(crate) fn reach(&self) -> i32 {
        (self.side / 2) as i32 * self.step
    }

    /// The entry whose vector is `values`; `None` when one is off the
    /// lattice.
    pub(crate) fn entry(&self, values: &[i32]) -> Option<usize> {
        let half = (self.side / 2) as i32;
        let mut entry = 0;
        let mut scale = 1;
        for &value in values.iter().take(self.dimensions) {
            if value % self.step != 0 {
                return None;
            }
            let digit = value / self.step + half;
            if !(0..self.side as i32).contains(&digit) {
                return None;
            }
            entry += digit as usize * scale;
            scale *= self.side;
        }
        Some(entry)
    }

    /// The vector of `entry`.
    pub(crate) fn values(&self, mut entry: usize, out: &mut [i32]) {
        let half = (self.side / 2) as i32;
        for value in out.iter_mut().take(self.dimensions) {
            *value = ((entry % self.side) as i32 - half) * self.step;
            entry /= self.side;
        }
    }

    /// Lengths from a weight per entry vector.
    pub(crate) fn lengths(&self, mut weight: impl FnMut(&[i32]) -> f64) -> Vec<u8> {
        let mut values = vec![0; self.dimensions];
        let weights: Vec<f64> = (0..self.entries())
            .map(|entry| {
                self.values(entry, &mut values);
                weight(&values)
            })
            .collect();
        huffman_lengths(&weights, MAX_LENGTH)
    }

    /// The book as the setup header carries it.
    pub(crate) fn spec(&self, lengths: Vec<u8>) -> BookSpec {
        let half = (self.side / 2) as i32;
        BookSpec {
            dimensions: self.dimensions,
            lengths,
            lookup: Some(Lookup {
                kind: 1,
                minimum: float32_pack(-(half * self.step) as f32),
                delta: float32_pack(self.step as f32),
                value_bits: ilog(self.side as u32 - 1).max(1) as u8,
                sequence: false,
                multiplicands: (0..self.side as u32).collect(),
            }),
        }
    }
}

/// A two-sided geometric model: `ratio^|value|`.
pub(crate) fn geometric(value: i32, ratio: f64) -> f64 {
    ratio.powi(value.abs())
}

#[cfg(test)]
mod tests {
    use super::{Coder, Lattice, MAX_LENGTH, geometric, huffman_lengths};
    use crate::bits::{BitReader, BitWriter};
    use crate::codebook::{Budget, Codebook, lookup1_values};

    fn kraft(lengths: &[u8]) -> f64 {
        lengths
            .iter()
            .filter(|&&len| len > 0)
            .map(|&len| 0.5_f64.powi(i32::from(len)))
            .sum()
    }

    #[test]
    fn huffman_lengths_make_a_complete_code_under_the_limit() {
        let skewed: Vec<f64> = (0..300).map(|i| 0.5_f64.powi(i / 4)).collect();
        for (weights, limit) in [
            (vec![1.0, 1.0], 24),
            (vec![5.0, 1.0, 1.0, 0.0, 3.0], 24),
            (skewed.clone(), 24),
            (skewed, 12),
            ((0..1089).map(|i| geometric(i - 544, 0.9)).collect(), 20),
        ] {
            let lengths = huffman_lengths(&weights, limit);
            assert!((kraft(&lengths) - 1.0).abs() < 1e-12, "{lengths:?}");
            assert!(lengths.iter().all(|&len| len <= limit));
            for (i, &weight) in weights.iter().enumerate() {
                assert_eq!(weight == 0.0, lengths[i] == 0);
            }
            assert!(Coder::new(&lengths).is_ok());
        }
    }

    #[test]
    fn more_likely_entries_never_take_longer_codes() {
        let weights = [9.0, 1.0, 4.0, 4.0, 0.5, 30.0];
        let lengths = huffman_lengths(&weights, MAX_LENGTH);
        for a in 0..weights.len() {
            for b in 0..weights.len() {
                if weights[a] > weights[b] {
                    assert!(lengths[a] <= lengths[b], "{lengths:?}");
                }
            }
        }
    }

    #[test]
    fn a_lattice_book_decodes_to_the_vector_it_was_written_for() {
        for lattice in [
            Lattice {
                dimensions: 4,
                side: 3,
                step: 1,
            },
            Lattice {
                dimensions: 2,
                side: 17,
                step: 1,
            },
            Lattice {
                dimensions: 1,
                side: 33,
                step: 1089,
            },
        ] {
            assert_eq!(
                lookup1_values(lattice.entries(), lattice.dimensions),
                lattice.side
            );
            let lengths = lattice.lengths(|v| {
                v.iter()
                    .map(|&x| geometric(x / lattice.step, 0.7))
                    .product()
            });
            let spec = lattice.spec(lengths.clone());
            assert!(lengths.iter().all(|&len| len > 0));
            let book = Codebook::build(&spec, &mut Budget::default()).expect("book");
            let coder = Coder::new(&lengths).expect("coder");
            let mut writer = BitWriter::new();
            let mut wanted = Vec::new();
            let mut values = vec![0; lattice.dimensions];
            for entry in (0..lattice.entries()).step_by(7) {
                lattice.values(entry, &mut values);
                assert_eq!(lattice.entry(&values), Some(entry));
                coder.write(&mut writer, entry);
                wanted.push(values.clone());
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            for want in wanted {
                let got = book.decode_vector(&mut reader).expect("vector");
                let got: Vec<i32> = got.iter().map(|&v| v as i32).collect();
                assert_eq!(got, want);
            }
            assert_eq!(lattice.entry(&[lattice.reach() + 1; 4]), None);
        }
    }
}
