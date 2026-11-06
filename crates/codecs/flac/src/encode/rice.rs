//! Residual coding: the partition order, Rice parameters, and escapes
//! (RFC 9639 section 9.2.7).
//!
//! Each partition's cost is estimated from the sum of its folded residuals,
//! the way the reference encoder does: `n (k + 1) + sum >> k`, less half a
//! bit per sample for the dropped fractions. Sums are taken at the finest
//! partition order and merged pairwise for each coarser one.

use crate::bits::BitWriter;

/// Highest Rice parameter of coding method 0; 15 is its escape.
const RICE1_MAX: u32 = 14;
/// Highest Rice parameter of coding method 1; 31 is its escape.
const RICE2_MAX: u32 = 30;

/// How one partition is coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Rice(u32),
    /// Unencoded residuals of this many bits, 0 to 31.
    Escape(u32),
}

/// A complete residual coding.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Coding method 1 (5-bit parameters) when some parameter passes 14.
    pub wide: bool,
    pub order: u32,
    pub parts: Vec<Part>,
    /// Estimated bits, method and order fields included.
    pub bits: u64,
}

/// Reused buffers.
#[derive(Default)]
pub(crate) struct Scratch {
    sums: Vec<u64>,
    peaks: Vec<u64>,
    level_sums: Vec<u64>,
    level_peaks: Vec<u64>,
    best_parts: Vec<Part>,
    parts: Vec<Part>,
}

/// Zigzag folding: 0, -1, 1, -2 to 0, 1, 2, 3.
#[inline]
pub(crate) fn fold(residual: i32) -> u64 {
    let value = i64::from(residual);
    ((value << 1) ^ (value >> 63)) as u64
}

/// Bits that hold a folded value as a two's complement escape.
fn width(peak: u64) -> u32 {
    64 - peak.leading_zeros()
}

/// The highest partition order a `block` with predictor `order` allows:
/// the block divides evenly and the first partition is longer than the
/// predictor.
pub(crate) fn max_order(block: usize, order: usize, limit: u32) -> u32 {
    let mut po = limit;
    while po > 0 && (!block.is_multiple_of(1usize << po) || (block >> po) <= order) {
        po -= 1;
    }
    po
}

/// Cost and parameter of one Rice-coded partition of `n` residuals whose
/// folded values sum to `sum`.
fn rice_cost(n: u64, sum: u64, max_k: u32) -> (u64, u32) {
    if n == 0 {
        return (0, 0);
    }
    let mut guess = 0u32;
    while guess < max_k && (n << (guess + 1)) <= sum {
        guess += 1;
    }
    let mut best = (u64::MAX, 0);
    for k in guess.saturating_sub(1)..=(guess + 1).min(max_k) {
        let cost = if k == 0 {
            n + sum
        } else {
            (n * u64::from(k + 1) + (sum >> k)).saturating_sub(n >> 1)
        };
        if cost < best.0 {
            best = (cost, k);
        }
    }
    best
}

/// The cheapest coding of `residual`, the samples after the warm-up of an
/// `order` predictor in a `block`, over partition orders `min..=max`.
pub(crate) fn plan(
    residual: &[i32],
    block: usize,
    order: usize,
    min: u32,
    max: u32,
    scratch: &mut Scratch,
) -> Plan {
    let top = max_order(block, order, max);
    let bottom = min.min(top);
    let parts = 1usize << top;
    let size = block >> top;
    scratch.sums.clear();
    scratch.peaks.clear();
    let mut start = 0;
    for part in 0..parts {
        let len = if part == 0 { size - order } else { size };
        let slice = &residual[start..start + len];
        let mut sum = 0u64;
        let mut peak = 0u64;
        for value in slice {
            let folded = fold(*value);
            sum += folded;
            peak |= folded;
        }
        scratch.sums.push(sum);
        scratch.peaks.push(peak);
        start += len;
    }
    let mut best = Plan {
        bits: u64::MAX,
        ..Plan::default()
    };
    let mut po = top;
    loop {
        let count = 1usize << po;
        let size = (block >> po) as u64;
        let mut total = 2 + 4u64;
        let mut wide = false;
        scratch.parts.clear();
        // First pass with 5-bit parameters, to see whether any wants k > 14.
        for part in 0..count {
            let n = if part == 0 { size - order as u64 } else { size };
            let (_, k) = rice_cost(n, scratch.sums[part], RICE2_MAX);
            if k > RICE1_MAX {
                wide = true;
            }
        }
        let parameter_bits = if wide { 5 } else { 4 };
        let max_k = if wide { RICE2_MAX } else { RICE1_MAX };
        for part in 0..count {
            let n = if part == 0 { size - order as u64 } else { size };
            let (rice, k) = rice_cost(n, scratch.sums[part], max_k);
            let bits = width(scratch.peaks[part]);
            let escape = if bits <= 31 {
                5 + n * u64::from(bits)
            } else {
                u64::MAX
            };
            if escape < rice {
                total += parameter_bits + escape;
                scratch.parts.push(Part::Escape(bits));
            } else {
                total += parameter_bits + rice;
                scratch.parts.push(Part::Rice(k));
            }
        }
        if total < best.bits {
            best.bits = total;
            best.wide = wide;
            best.order = po;
            std::mem::swap(&mut scratch.best_parts, &mut scratch.parts);
        }
        if po == bottom {
            break;
        }
        // Merge to the next coarser order.
        scratch.level_sums.clear();
        scratch.level_peaks.clear();
        for pair in 0..count / 2 {
            scratch
                .level_sums
                .push(scratch.sums[2 * pair] + scratch.sums[2 * pair + 1]);
            scratch
                .level_peaks
                .push(scratch.peaks[2 * pair] | scratch.peaks[2 * pair + 1]);
        }
        std::mem::swap(&mut scratch.sums, &mut scratch.level_sums);
        std::mem::swap(&mut scratch.peaks, &mut scratch.level_peaks);
        po -= 1;
    }
    best.parts = scratch.best_parts.clone();
    best
}

/// Writes `residual` coded as `plan`.
pub(crate) fn write(
    out: &mut BitWriter,
    residual: &[i32],
    block: usize,
    order: usize,
    plan: &Plan,
) {
    out.put(2, u64::from(plan.wide));
    out.put(4, u64::from(plan.order));
    let parameter_bits = if plan.wide { 5 } else { 4 };
    let escape = (1u64 << parameter_bits) - 1;
    let size = block >> plan.order;
    let mut start = 0;
    for (index, part) in plan.parts.iter().enumerate() {
        let len = if index == 0 { size - order } else { size };
        let slice = &residual[start..start + len];
        start += len;
        match *part {
            Part::Escape(bits) => {
                out.put(parameter_bits, escape);
                out.put(5, u64::from(bits));
                for value in slice {
                    out.put_signed(bits, i64::from(*value));
                }
            }
            Part::Rice(k) => {
                out.put(parameter_bits, u64::from(k));
                let mask = (1u64 << k) - 1;
                for value in slice {
                    let folded = fold(*value);
                    let quotient = folded >> k;
                    if quotient + 1 + u64::from(k) <= 32 {
                        let len = u32::try_from(quotient).unwrap_or(0) + 1 + k;
                        out.put(len, (1u64 << k) | (folded & mask));
                    } else {
                        out.put_unary(quotient);
                        out.put(k, folded & mask);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Part, Scratch, fold, max_order, plan, write};
    use crate::bits::{BitReader, BitWriter};

    #[test]
    fn folding_matches_the_rfc() {
        assert_eq!([0, -1, 1, -2, 2].map(fold), [0, 1, 2, 3, 4]);
        assert_eq!(fold(i32::MAX), u64::from(u32::MAX) - 1);
    }

    #[test]
    fn partition_orders_respect_the_block_and_the_predictor() {
        assert_eq!(max_order(4096, 8, 8), 8);
        assert_eq!(max_order(4096, 32, 8), 6);
        assert_eq!(max_order(1000, 0, 8), 3);
        assert_eq!(max_order(17, 0, 8), 0);
        assert_eq!(max_order(16, 4, 8), 1);
    }

    #[test]
    fn silence_is_escaped_at_width_zero() {
        let residual = vec![0i32; 4096];
        let plan = plan(&residual, 4096, 0, 0, 4, &mut Scratch::default());
        assert!(plan.parts.iter().all(|part| *part == Part::Escape(0)));
        assert_eq!(plan.order, 0);
    }

    #[test]
    fn loud_noise_uses_the_wide_method() {
        let residual: Vec<i32> = (0..256)
            .map(|i: i32| i.wrapping_mul(2_654_435_761u32 as i32) >> 2)
            .collect();
        let plan = plan(&residual, 256, 0, 0, 0, &mut Scratch::default());
        assert!(
            plan.wide
                || plan
                    .parts
                    .iter()
                    .all(|part| matches!(part, Part::Escape(_)))
        );
    }

    #[test]
    fn written_residuals_read_back() {
        let residual: Vec<i32> = (0..1020).map(|i| (i * 37) % 201 - 100).collect();
        let plan = plan(&residual, 1024, 4, 0, 6, &mut Scratch::default());
        let mut out = BitWriter::with_capacity(4096);
        write(&mut out, &residual, 1024, 4, &plan);
        out.align();
        let bytes = out.into_bytes();
        let estimate = plan.bits.div_ceil(8);
        let actual = bytes.len() as u64;
        assert!(
            actual.abs_diff(estimate) * 20 < actual,
            "{actual} vs {estimate}"
        );
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(2, "m").expect("m"), u64::from(plan.wide));
        assert_eq!(reader.read(4, "o").expect("o"), u64::from(plan.order));
    }
}
