//! Residue packet decode (Vorbis I section 8.6): classifications, then up to
//! eight cascade passes of vector-quantized partitions, in formats 0
//! (interleaved within a partition), 1 (in order), and 2 (format 1 over the
//! channels interleaved).

use crate::bits::BitReader;
use crate::codebook::Codebook;
use crate::setup::Residue;

/// Classification scratch, reused across packets.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    classes: Vec<u8>,
}

/// Decodes `vectors.len() / half` vectors of `half` values each into
/// `vectors` (zeroed by the caller), skipping vectors whose
/// `do_not_decode` is set. End of packet stops decode with what was read.
pub(crate) fn decode(
    residue: &Residue,
    books: &[Codebook],
    reader: &mut BitReader<'_>,
    half: usize,
    do_not_decode: &[bool],
    vectors: &mut [f32],
    scratch: &mut Scratch,
) {
    let channels = do_not_decode.len();
    if channels == 0 || half == 0 {
        return;
    }
    let interleaved = residue.kind == 2;
    let (vector_count, size) = if interleaved {
        if do_not_decode.iter().all(|&skip| skip) {
            return;
        }
        (1, half * channels)
    } else {
        (channels, half)
    };
    let begin = (residue.begin as usize).min(size);
    let end = (residue.end as usize).min(size);
    if end <= begin {
        return;
    }
    let partition = residue.partition_size as usize;
    let partitions = (end - begin) / partition;
    if partitions == 0 {
        return;
    }
    let classbook = &books[usize::from(residue.classbook)];
    let per_word = classbook.dimensions;
    let stride = partitions + per_word;
    scratch.classes.clear();
    scratch.classes.resize(vector_count * stride, 0);
    let classifications = u32::from(residue.classifications);
    let skip = |vector: usize| -> bool { !interleaved && do_not_decode[vector] };
    for pass in 0..8 {
        let mut count = 0;
        while count < partitions {
            if pass == 0 {
                for vector in 0..vector_count {
                    if skip(vector) {
                        continue;
                    }
                    let Some(mut word) = classbook.decode(reader) else {
                        return;
                    };
                    for i in (0..per_word).rev() {
                        scratch.classes[vector * stride + count + i] =
                            (word % classifications) as u8;
                        word /= classifications;
                    }
                }
            }
            let mut i = 0;
            while i < per_word && count < partitions {
                for vector in 0..vector_count {
                    if skip(vector) {
                        continue;
                    }
                    let class = usize::from(scratch.classes[vector * stride + count]);
                    let Some(book) = residue.books[class][pass] else {
                        continue;
                    };
                    let book = &books[usize::from(book)];
                    let offset = begin + count * partition;
                    let done = match residue.kind {
                        0 => {
                            let target = &mut vectors[vector * half..(vector + 1) * half];
                            partition_format0(book, reader, &mut target[offset..offset + partition])
                        }
                        1 => {
                            let target = &mut vectors[vector * half..(vector + 1) * half];
                            partition_format1(book, reader, &mut target[offset..offset + partition])
                        }
                        _ => partition_format2(
                            book, reader, vectors, half, channels, offset, partition,
                        ),
                    };
                    if !done {
                        return;
                    }
                }
                i += 1;
                count += 1;
            }
        }
    }
}

/// Format 0: `step = n / dimensions` codewords, value `d` of codeword `k`
/// at `k + d * step`. False at end of packet.
fn partition_format0(book: &Codebook, reader: &mut BitReader<'_>, target: &mut [f32]) -> bool {
    let dimensions = book.dimensions;
    let step = target.len() / dimensions;
    for k in 0..step {
        let Some(vector) = book.decode_vector(reader) else {
            return false;
        };
        for (d, &value) in vector.iter().enumerate() {
            target[k + d * step] += value;
        }
    }
    true
}

/// Format 1: codeword values in order until the partition is full.
fn partition_format1(book: &Codebook, reader: &mut BitReader<'_>, target: &mut [f32]) -> bool {
    let mut at = 0;
    while at < target.len() {
        let Some(vector) = book.decode_vector(reader) else {
            return false;
        };
        for &value in vector {
            if let Some(slot) = target.get_mut(at) {
                *slot += value;
            }
            at += 1;
        }
    }
    true
}

/// Format 2: format 1 over the interleaved vector, where interleaved
/// position `p` is channel `p % channels`, sample `p / channels`.
fn partition_format2(
    book: &Codebook,
    reader: &mut BitReader<'_>,
    vectors: &mut [f32],
    half: usize,
    channels: usize,
    offset: usize,
    len: usize,
) -> bool {
    let mut channel = offset % channels;
    let mut sample = offset / channels;
    let mut left = len;
    while left > 0 {
        let Some(vector) = book.decode_vector(reader) else {
            return false;
        };
        for &value in vector {
            if left == 0 {
                break;
            }
            vectors[channel * half + sample] += value;
            channel += 1;
            if channel == channels {
                channel = 0;
                sample += 1;
            }
            left -= 1;
        }
    }
    true
}
