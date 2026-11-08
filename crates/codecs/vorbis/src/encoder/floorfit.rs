//! Floor 1 analysis (Vorbis I section 7): the point Y values that follow a
//! target curve, their prediction coding, and the curve a decoder draws
//! from them.
//!
//! The fit is least squares over the piecewise-linear curve through the
//! points (a tridiagonal system over the hat functions), rounded to Y
//! levels. A point whose fitted value is within `tolerance` of what its
//! neighbors predict is coded as 0, the prediction itself.

use crate::bits::{BitWriter, ilog};
use crate::encoder::books::Coder;
use crate::encoder::profile::{FLOOR_MULTIPLIER, FLOOR_RANGE, FloorPlan};
use crate::floor::{floor1_curve, gain, render_point};
use crate::tables::INVERSE_DB;

/// dB of Y 0, and dB per Y level.
pub(crate) fn y_scale() -> (f32, f32) {
    let low = 20.0 * INVERSE_DB[0].log10();
    let high = 20.0 * INVERSE_DB[255].log10();
    let per_index = (high - low) / 255.0;
    (low, per_index * f32::from(FLOOR_MULTIPLIER))
}

/// One channel's coded floor.
#[derive(Clone, Debug, Default)]
pub(crate) struct CodedFloor {
    /// Y per point in stream order, as the decoder rebuilds it.
    pub ys: Vec<i32>,
    pub drawn: Vec<bool>,
    /// The coded value per point (points 0 and 1 are Y itself).
    pub coded: Vec<i32>,
}

/// Fits the points to `target` (Y levels per bin, any real value) with
/// `weight` per bin, prunes, and codes them.
pub(crate) fn fit(plan: &FloorPlan, target: &[f32], weight: &[f32], tolerance: i32) -> CodedFloor {
    let floor = &plan.floor;
    let n = target.len();
    let points = floor.xs.len();
    let sorted = &floor.sorted;
    // Normal equations of the hat basis, by sorted position.
    let mut diagonal = vec![1e-6_f64; points];
    let mut upper = vec![0.0_f64; points];
    let mut rhs = vec![0.0_f64; points];
    for s in 0..points - 1 {
        let x0 = usize::from(floor.xs[sorted[s]]);
        let x1 = usize::from(floor.xs[sorted[s + 1]]);
        let span = (x1 - x0) as f64;
        for k in x0..x1.min(n) {
            let right = (k - x0) as f64 / span;
            let left = 1.0 - right;
            let w = f64::from(weight[k]);
            let t = f64::from(target[k]);
            diagonal[s] += w * left * left;
            diagonal[s + 1] += w * right * right;
            upper[s] += w * left * right;
            rhs[s] += w * left * t;
            rhs[s + 1] += w * right * t;
        }
    }
    // Thomas algorithm: the Gram matrix of hat functions is diagonally
    // dominant, so no pivoting.
    let mut c = vec![0.0_f64; points];
    let mut d = vec![0.0_f64; points];
    for s in 0..points {
        let below = if s > 0 { upper[s - 1] } else { 0.0 };
        let denominator = diagonal[s] - below * if s > 0 { c[s - 1] } else { 0.0 };
        c[s] = upper[s] / denominator;
        d[s] = (rhs[s] - below * if s > 0 { d[s - 1] } else { 0.0 }) / denominator;
    }
    let mut fitted = vec![0.0_f64; points];
    for s in (0..points).rev() {
        fitted[s] = d[s]
            - if s + 1 < points {
                c[s] * fitted[s + 1]
            } else {
                0.0
            };
    }
    let mut wanted = vec![0_i32; points];
    for (s, &point) in sorted.iter().enumerate() {
        wanted[point] = (fitted[s].round() as i32).clamp(0, FLOOR_RANGE - 1);
    }
    code(plan, &wanted, tolerance)
}

/// Codes `wanted` Y values: each point after the first two is the
/// difference from its prediction, or 0 within `tolerance`.
pub(crate) fn code(plan: &FloorPlan, wanted: &[i32], tolerance: i32) -> CodedFloor {
    let floor = &plan.floor;
    let points = wanted.len();
    let mut ys = wanted.to_vec();
    let mut coded = vec![0_i32; points];
    let mut drawn = vec![false; points];
    coded[0] = ys[0];
    coded[1] = ys[1];
    drawn[0] = true;
    drawn[1] = true;
    for i in 2..points {
        let (low, high) = (floor.low[i], floor.high[i]);
        let predicted = render_point(
            i32::from(floor.xs[low]),
            ys[low],
            i32::from(floor.xs[high]),
            ys[high],
            i32::from(floor.xs[i]),
        );
        let difference = ys[i] - predicted;
        if difference.abs() <= tolerance {
            ys[i] = predicted;
            continue;
        }
        let high_room = FLOOR_RANGE - predicted;
        let low_room = predicted;
        let room = high_room.min(low_room);
        coded[i] = if difference > 0 {
            if difference < room {
                difference * 2
            } else {
                difference + low_room
            }
        } else if -difference <= room {
            -difference * 2 - 1
        } else {
            -difference + high_room - 1
        };
        drawn[low] = true;
        drawn[high] = true;
        drawn[i] = true;
    }
    CodedFloor { ys, drawn, coded }
}

/// Writes a used floor (section 7.2.3).
pub(crate) fn write(plan: &FloorPlan, coders: &[Coder], coded: &CodedFloor, w: &mut BitWriter) {
    w.write_bit(true);
    let width = ilog((FLOOR_RANGE - 1) as u32);
    w.write(coded.coded[0] as u32, width);
    w.write(coded.coded[1] as u32, width);
    let mut offset = 2;
    for &class in &plan.floor.partition_classes {
        let class = &plan.floor.classes[usize::from(class)];
        let dims = usize::from(class.dimensions);
        let bits = u32::from(class.subclass_bits);
        let mut subclasses = [0_usize; 8];
        let mut cval = 0;
        for (k, sub) in subclasses.iter_mut().enumerate().take(dims) {
            let value = coded.coded[offset + k];
            *sub = plan
                .limits
                .iter()
                .position(|&limit| value < limit)
                .unwrap_or(plan.limits.len() - 1);
            cval |= *sub << (bits * k as u32);
        }
        if bits > 0 {
            coders[plan.masterbook].write(w, cval);
        }
        for (k, &sub) in subclasses.iter().enumerate().take(dims) {
            if let Some(book) = plan.subbooks[sub] {
                coders[book].write(w, coded.coded[offset + k] as usize);
            }
        }
        offset += dims;
    }
}

/// Bits [`write`] takes.
#[cfg(test)]
pub(crate) fn bits(plan: &FloorPlan, coders: &[Coder], coded: &CodedFloor) -> u32 {
    let mut w = BitWriter::new();
    write(plan, coders, coded, &mut w);
    w.len() as u32
}

/// The linear gain per bin a decoder applies for `coded`.
pub(crate) fn curve(plan: &FloorPlan, coded: &CodedFloor, out: &mut [f32]) {
    let n = out.len();
    floor1_curve(&plan.floor, &coded.ys, &coded.drawn, n, |x, y| {
        out[x] = gain(y);
    });
}

#[cfg(test)]
mod tests {
    use super::{CodedFloor, code, curve, fit, write, y_scale};
    use crate::bits::{BitReader, BitWriter};
    use crate::codebook::{Budget, Codebook};
    use crate::encoder::profile::{FLOOR_RANGE, Profile};
    use crate::floor::{ChannelFloor, Status};

    fn decode(profile: &Profile, size: usize, bytes: &[u8], n: usize) -> Vec<f32> {
        let books: Vec<Codebook> = profile
            .setup
            .books
            .iter()
            .map(|spec| Codebook::build(spec, &mut Budget::default()).expect("book"))
            .collect();
        let mut floor = ChannelFloor::default();
        let mut reader = BitReader::new(bytes);
        let plan = &profile.floors[size];
        assert_eq!(
            floor.decode_floor1(&plan.floor, &books, &mut reader),
            Status::Used
        );
        let mut spectrum = vec![1.0; n];
        floor.apply_floor1(&plan.floor, &mut spectrum);
        spectrum
    }

    fn round_trip(profile: &Profile, size: usize, coded: &CodedFloor) {
        let plan = &profile.floors[size];
        let n = profile.blocksizes[size] / 2;
        let mut w = BitWriter::new();
        write(plan, &profile.coders, coded, &mut w);
        let bytes = w.finish();
        let decoded = decode(profile, size, &bytes, n);
        let mut ours = vec![0.0; n];
        curve(plan, coded, &mut ours);
        assert_eq!(decoded, ours);
    }

    #[test]
    fn every_y_codes_from_every_prediction() {
        let profile = Profile::new(1, 44_100, 20_000.0).expect("profile");
        let plan = &profile.floors[0];
        let points = plan.floor.xs.len();
        for first in [0, 1, 40, 126, 127] {
            for value in 0..FLOOR_RANGE {
                let mut wanted = vec![first; points];
                wanted[1] = FLOOR_RANGE - 1 - first;
                for (i, y) in wanted.iter_mut().enumerate().skip(2) {
                    *y = (value + i as i32 * 37) % FLOOR_RANGE;
                }
                let coded = code(plan, &wanted, 0);
                assert_eq!(coded.ys, wanted);
                round_trip(&profile, 0, &coded);
            }
        }
    }

    #[test]
    fn the_fit_follows_a_curve_and_pruning_keeps_it_close() {
        let profile = Profile::new(2, 44_100, 20_000.0).expect("profile");
        let n = 1024;
        let target: Vec<f32> = (0..n)
            .map(|k| 90.0 - 30.0 * ((k as f32 + 1.0).ln() / 7.0) + 6.0 * (k as f32 / 40.0).sin())
            .collect();
        let weight = vec![1.0; n];
        let plan = &profile.floors[1];
        let exact = fit(plan, &target, &weight, 0);
        let pruned = fit(plan, &target, &weight, 2);
        round_trip(&profile, 1, &exact);
        round_trip(&profile, 1, &pruned);
        let (low, step) = y_scale();
        for coded in [&exact, &pruned] {
            let mut gains = vec![0.0; n];
            curve(plan, coded, &mut gains);
            let worst = (8..n)
                .map(|k| ((20.0 * gains[k].log10() - low) / step - target[k]).abs())
                .fold(0.0_f32, f32::max);
            assert!(worst < 6.0, "worst {worst}");
        }
        let zeros = |c: &CodedFloor| c.coded.iter().skip(2).filter(|&&v| v == 0).count();
        assert!(zeros(&pruned) > zeros(&exact));
        let bits = |c| super::bits(plan, &profile.coders, c);
        assert!(bits(&pruned) < bits(&exact));
    }
}
