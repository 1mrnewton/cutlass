//! Residue encode (Vorbis I section 8.6), the mirror of `residue::decode`:
//! each partition takes the classification that codes it in the fewest
//! bits, then the classifications and the passes go out in decode order.

use crate::bits::BitWriter;
use crate::encoder::books::Coder;
use crate::encoder::profile::{ClassPlan, MAX_RESIDUE, ResiduePlan};
use cutlass_decode::CodecError;

/// The value each pass of `class` codes for `value`. One pass codes the
/// value; the escape cascade splits it into balanced base-33 digits.
fn split(class: &ClassPlan, value: i32) -> [i32; 3] {
    if class.passes.len() < 3 {
        return [value, 0, 0];
    }
    let value = value.clamp(-MAX_RESIDUE, MAX_RESIDUE);
    let fine = (value + 16).rem_euclid(33) - 16;
    let rest = (value - fine) / 33;
    let middle = (rest + 16).rem_euclid(33) - 16;
    let coarse = (rest - middle) / 33;
    [coarse * 1089, middle * 33, fine]
}

/// Whether `values` (starting at an even position) fit `class`.
fn fits(class: &ClassPlan, values: &[i32]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(i, &v)| v.abs() <= class.limits[i % 2])
}

/// Bits `class` takes for `values`; `None` when they do not fit.
fn cost(class: &ClassPlan, coders: &[Coder], values: &[i32]) -> Option<u32> {
    if !fits(class, values) {
        return None;
    }
    let mut bits = 0;
    let mut chunk = [0_i32; 8];
    for (pass, &(book, lattice)) in class.passes.iter().enumerate() {
        for vector in values.chunks(lattice.dimensions) {
            for (slot, &v) in chunk.iter_mut().zip(vector) {
                *slot = split(class, v)[pass];
            }
            let entry = lattice.entry(&chunk[..lattice.dimensions])?;
            let len = coders[book].bits(entry);
            if len == 0 {
                return None;
            }
            bits += len;
        }
    }
    Some(bits)
}

/// The cheapest classification of `values`; the last one always fits.
fn classify(plan: &ResiduePlan, coders: &[Coder], values: &[i32]) -> usize {
    if values.iter().all(|&v| v == 0) {
        return 0;
    }
    let mut best = (plan.classes.len() - 1, u32::MAX);
    for (index, class) in plan.classes.iter().enumerate().skip(1) {
        if let Some(bits) = cost(class, coders, values)
            && bits < best.1
        {
            best = (index, bits);
        }
    }
    best.0
}

/// Writes the residue of one submap. `vectors` holds one vector per
/// channel (format 1) or the channels interleaved (format 2), each at
/// least `plan.end` long; vectors with `skip` set are not coded.
///
/// # Errors
///
/// `Limit` when a value is not in the class's book. Classification only
/// picks a class that contains every value, so this is a bug in the
/// quantizer's range.
pub(crate) fn write(
    plan: &ResiduePlan,
    coders: &[Coder],
    vectors: &[Vec<i32>],
    skip: &[bool],
    w: &mut BitWriter,
) -> Result<(), CodecError> {
    let partitions = (plan.end - plan.begin) / plan.partition;
    if partitions == 0 {
        return Ok(());
    }
    let part = |vector: usize, p: usize| -> &[i32] {
        let start = plan.begin + p * plan.partition;
        &vectors[vector][start..start + plan.partition]
    };
    let classes: Vec<Vec<usize>> = (0..vectors.len())
        .map(|v| {
            if skip[v] {
                Vec::new()
            } else {
                (0..partitions)
                    .map(|p| classify(plan, coders, part(v, p)))
                    .collect()
            }
        })
        .collect();
    let count = plan.classes.len();
    let passes = plan
        .classes
        .iter()
        .map(|c| c.passes.len())
        .max()
        .unwrap_or(0);
    let mut chunk = [0_i32; 8];
    for pass in 0..passes {
        let mut p = 0;
        while p < partitions {
            if pass == 0 {
                for (v, classes) in classes.iter().enumerate() {
                    if skip[v] {
                        continue;
                    }
                    let mut word = 0;
                    for j in 0..plan.per_word {
                        word = word * count + classes.get(p + j).copied().unwrap_or(0);
                    }
                    coders[plan.classbook].write(w, word);
                }
            }
            let mut i = 0;
            while i < plan.per_word && p < partitions {
                for (v, classes) in classes.iter().enumerate() {
                    if skip[v] {
                        continue;
                    }
                    let class = &plan.classes[classes[p]];
                    let Some(&(book, lattice)) = class.passes.get(pass) else {
                        continue;
                    };
                    for vector in part(v, p).chunks(lattice.dimensions) {
                        for (slot, &value) in chunk.iter_mut().zip(vector) {
                            *slot = split(class, value)[pass];
                        }
                        let Some(entry) = lattice.entry(&chunk[..lattice.dimensions]) else {
                            return Err(CodecError::Limit {
                                what: "Vorbis residue value",
                            });
                        };
                        if coders[book].bits(entry) == 0 {
                            return Err(CodecError::Limit {
                                what: "Vorbis residue value",
                            });
                        }
                        coders[book].write(w, entry);
                    }
                }
                i += 1;
                p += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{split, write};
    use crate::bits::{BitReader, BitWriter};
    use crate::codebook::{Budget, Codebook};
    use crate::encoder::profile::{MAX_RESIDUE, Profile};
    use crate::residue::{Scratch, decode};
    use crate::setup::Residue;

    fn residue_of(profile: &Profile, index: usize) -> Residue {
        profile.setup.residues[index].clone()
    }

    #[test]
    fn the_escape_cascade_sums_to_the_value() {
        let profile = Profile::new(1, 44_100, 20_000.0).expect("profile");
        let escape = profile.residues[1][0].classes.last().expect("escape");
        for value in [
            -MAX_RESIDUE,
            -545,
            -544,
            -17,
            -16,
            0,
            16,
            17,
            544,
            545,
            MAX_RESIDUE,
        ] {
            let parts = split(escape, value);
            assert_eq!(parts.iter().sum::<i32>(), value);
            for (pass, &(_, lattice)) in escape.passes.iter().enumerate() {
                assert!(
                    lattice
                        .entry(&[parts[pass]; 2][..lattice.dimensions])
                        .is_some()
                );
            }
        }
    }

    fn check(
        channels: usize,
        index: usize,
        size: usize,
        kind: usize,
        values: Vec<Vec<i32>>,
        skip: Vec<bool>,
    ) {
        let profile = Profile::new(channels, 44_100, 22_050.0).expect("profile");
        let plan = &profile.residues[size][kind];
        let mut w = BitWriter::new();
        write(plan, &profile.coders, &values, &skip, &mut w).expect("residue");
        let bytes = w.finish();
        let books: Vec<Codebook> = profile
            .setup
            .books
            .iter()
            .map(|spec| Codebook::build(spec, &mut Budget::default()).expect("book"))
            .collect();
        let half = profile.blocksizes[size] / 2;
        // Format 1 stores one vector per channel. Format 2 stores one
        // interleaved vector and the decoder writes it back per channel.
        let channels = if plan.kind == 2 {
            channels
        } else {
            values.len()
        };
        let mut out = vec![0.0_f32; half * channels];
        let mut reader = BitReader::new(&bytes);
        let do_not_decode = if plan.kind == 2 {
            vec![skip[0]; channels]
        } else {
            skip.clone()
        };
        decode(
            &residue_of(&profile, index),
            &books,
            &mut reader,
            half,
            &do_not_decode,
            &mut out,
            &mut Scratch::default(),
        );
        let end = plan.end;
        if plan.kind == 2 {
            for channel in 0..channels {
                for bin in 0..half {
                    let pos = bin * channels + channel;
                    let want = if skip[0] || pos >= end {
                        0
                    } else {
                        values[0][pos]
                    };
                    assert_eq!(
                        out[channel * half + bin] as i32,
                        want,
                        "ch {channel} bin {bin}"
                    );
                }
            }
        } else {
            for vector in 0..channels {
                for bin in 0..half {
                    let want = if skip[vector] || bin >= end {
                        0
                    } else {
                        values[vector][bin]
                    };
                    assert_eq!(
                        out[vector * half + bin] as i32,
                        want,
                        "vector {vector} bin {bin}"
                    );
                }
            }
        }
    }

    fn pattern(len: usize, seed: u32, scale: i32) -> Vec<i32> {
        let mut state = seed;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let magnitude = (state % 1000) as i32;
                match i % 64 {
                    0..16 => 0,
                    16..32 => magnitude % 2 * (1 - 2 * (magnitude % 3 % 2)),
                    32..48 => (magnitude % 9) - 4,
                    _ => (magnitude - 500) * scale / 500,
                }
            })
            .collect()
    }

    #[test]
    fn format_1_decodes_to_the_values_written() {
        for scale in [1, 20, 600, MAX_RESIDUE] {
            let values = vec![pattern(1024, 3, scale)];
            check(1, 2, 1, 0, values, vec![false]);
        }
        let values = vec![pattern(128, 5, 40), pattern(128, 9, 3)];
        check(3, 0, 0, 0, values.clone(), vec![false, false]);
        check(3, 0, 0, 0, values, vec![true, false]);
    }

    #[test]
    fn format_2_decodes_to_the_values_written() {
        for scale in [1, 2, 30, 5000] {
            let mut values = pattern(2048, 7, scale);
            for pair in values.chunks_mut(2).skip(300) {
                pair[1] = 0;
            }
            check(2, 3, 1, 1, vec![values], vec![false]);
        }
        check(2, 1, 0, 1, vec![pattern(256, 1, 9)], vec![false]);
        check(2, 1, 0, 1, vec![pattern(256, 1, 9)], vec![true]);
    }
}
