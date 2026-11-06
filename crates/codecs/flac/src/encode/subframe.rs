//! One channel of one block to the cheapest subframe: constant, verbatim,
//! fixed, or LPC (RFC 9639 section 9.2).

use super::lpc::{self, MAX_ORDER};
use super::rice::{self, Plan};
use super::settings::Settings;
use crate::bits::BitWriter;

/// Buffers reused across subframes.
#[derive(Default)]
pub(crate) struct Workspace {
    shifted: Vec<i64>,
    residual: Vec<i32>,
    best_residual: Vec<i32>,
    windowed: Vec<f64>,
    rice: rice::Scratch,
}

#[derive(Clone, Debug)]
enum Model {
    Fixed {
        order: usize,
    },
    Lpc {
        order: usize,
        precision: u32,
        shift: u32,
        coefficients: [i32; MAX_ORDER],
    },
}

struct Best {
    bits: u64,
    model: Model,
    plan: Plan,
}

fn header(out: &mut BitWriter, kind: u64, wasted: u32) {
    out.put(1, 0);
    out.put(6, kind);
    if wasted == 0 {
        out.put(1, 0);
    } else {
        out.put(1, 1);
        out.put_unary(u64::from(wasted - 1));
    }
}

/// Codes `samples`, each `bits` wide (the frame depth, plus one for a side
/// channel), into `out`, which is cleared first.
pub(crate) fn encode(
    samples: &[i64],
    bits: u32,
    settings: &Settings,
    windows: &[Vec<f64>],
    work: &mut Workspace,
    out: &mut BitWriter,
) {
    out.clear();
    let n = samples.len();
    let Some(&first) = samples.first() else {
        return;
    };
    if samples.iter().all(|sample| *sample == first) {
        header(out, 0, 0);
        out.put_signed(bits, first);
        return;
    }
    let all = samples.iter().fold(0i64, |all, sample| all | sample);
    let wasted = all.trailing_zeros().min(bits - 1);
    let depth = bits - wasted;
    work.shifted.clear();
    work.shifted
        .extend(samples.iter().map(|sample| sample >> wasted));
    let verbatim = 8 + u64::from(wasted) + n as u64 * u64::from(depth);
    let mut best: Option<Best> = None;
    let (min_po, max_po) = (
        u32::from(settings.min_partition_order),
        u32::from(settings.max_partition_order),
    );

    // Fixed predictors: the order the residual magnitudes suggest, or all.
    let fixed_orders = if settings.exhaustive_order {
        0..5
    } else {
        let (order, _) = lpc::best_fixed_order(&work.shifted);
        order..order + 1
    };
    for order in fixed_orders {
        if order >= n || !lpc::fixed_residual(&work.shifted, order, &mut work.residual) {
            continue;
        }
        let plan = rice::plan(&work.residual, n, order, min_po, max_po, &mut work.rice);
        let cost = 8 + u64::from(wasted) + order as u64 * u64::from(depth) + plan.bits;
        if best.as_ref().is_none_or(|best| cost < best.bits) {
            std::mem::swap(&mut work.residual, &mut work.best_residual);
            best = Some(Best {
                bits: cost,
                model: Model::Fixed { order },
                plan,
            });
        }
    }

    // LPC, one candidate per window (or per order when exhaustive).
    let max_order = usize::from(settings.max_lpc_order).min(n.saturating_sub(1));
    if max_order > 0 {
        let precision = lpc::precision(depth, n);
        let mut autoc = [0.0; MAX_ORDER + 1];
        let mut coefficients = [[0.0; MAX_ORDER]; MAX_ORDER];
        let mut errors = [0.0; MAX_ORDER];
        for window in windows {
            lpc::autocorrelation(
                &work.shifted,
                window,
                max_order + 1,
                &mut work.windowed,
                &mut autoc,
            );
            let orders = lpc::levinson(&autoc, max_order, &mut coefficients, &mut errors);
            if orders == 0 {
                continue;
            }
            let candidates = if settings.exhaustive_order {
                1..orders + 1
            } else {
                let order = lpc::best_order(&errors, orders, n, depth, precision);
                order..order + 1
            };
            for order in candidates {
                let mut q = [0i32; MAX_ORDER];
                let Some(shift) =
                    lpc::quantize(&coefficients[order - 1][..order], precision, &mut q)
                else {
                    continue;
                };
                if !lpc::residual(&work.shifted, &q[..order], shift, &mut work.residual) {
                    continue;
                }
                let plan = rice::plan(&work.residual, n, order, min_po, max_po, &mut work.rice);
                let cost = 8
                    + u64::from(wasted)
                    + order as u64 * u64::from(depth + precision)
                    + 9
                    + plan.bits;
                if best.as_ref().is_none_or(|best| cost < best.bits) {
                    std::mem::swap(&mut work.residual, &mut work.best_residual);
                    best = Some(Best {
                        bits: cost,
                        model: Model::Lpc {
                            order,
                            precision,
                            shift,
                            coefficients: q,
                        },
                        plan,
                    });
                }
            }
        }
    }

    if let Some(best) = best.filter(|best| best.bits < verbatim) {
        write_model(
            out,
            &best,
            wasted,
            depth,
            &work.shifted,
            &work.best_residual,
        );
        if out.bit_len() <= verbatim {
            return;
        }
        out.clear();
    }
    header(out, 1, wasted);
    for sample in &work.shifted {
        out.put_signed(depth, *sample);
    }
}

fn write_model(
    out: &mut BitWriter,
    best: &Best,
    wasted: u32,
    depth: u32,
    samples: &[i64],
    residual: &[i32],
) {
    let n = samples.len();
    match best.model {
        Model::Fixed { order } => {
            header(out, 8 + order as u64, wasted);
            for sample in &samples[..order] {
                out.put_signed(depth, *sample);
            }
            rice::write(out, residual, n, order, &best.plan);
        }
        Model::Lpc {
            order,
            precision,
            shift,
            coefficients,
        } => {
            header(out, 31 + order as u64, wasted);
            for sample in &samples[..order] {
                out.put_signed(depth, *sample);
            }
            out.put(4, u64::from(precision - 1));
            out.put_signed(5, i64::from(shift));
            for coefficient in &coefficients[..order] {
                out.put_signed(precision, i64::from(*coefficient));
            }
            rice::write(out, residual, n, order, &best.plan);
        }
    }
}
