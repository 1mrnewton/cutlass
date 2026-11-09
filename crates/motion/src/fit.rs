//! RANSAC fits of the target's move: a similarity (move, scale, turn) or a
//! homography (four corners), from the followed points that agree.

/// A point in one frame and where it is in another.
pub(crate) type Pair = ((f64, f64), (f64, f64));

/// A point's place in the first frame, and where it was followed to.
pub(crate) type Anchored = ((f64, f64), (f32, f32));

/// A move from one frame's pixels to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Model {
    /// `x' = a x - b y + tx`, `y' = b x + a y + ty`.
    Similarity { a: f64, b: f64, tx: f64, ty: f64 },
    /// Row by row, the last entry 1.
    Homography([f64; 9]),
}

impl Model {
    pub(crate) fn apply(&self, p: (f64, f64)) -> Option<(f64, f64)> {
        match *self {
            Self::Similarity { a, b, tx, ty } => {
                Some((a * p.0 - b * p.1 + tx, b * p.0 + a * p.1 + ty))
            }
            Self::Homography(h) => {
                let w = h[6] * p.0 + h[7] * p.1 + h[8];
                if w.abs() < 1e-9 {
                    return None;
                }
                let q = (
                    (h[0] * p.0 + h[1] * p.1 + h[2]) / w,
                    (h[3] * p.0 + h[4] * p.1 + h[5]) / w,
                );
                (q.0.is_finite() && q.1.is_finite()).then_some(q)
            }
        }
    }
}

impl Model {
    /// No move: a similarity, or for `plane` a homography.
    pub(crate) fn identity(plane: bool) -> Self {
        if plane {
            Self::Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        } else {
            Self::Similarity {
                a: 1.0,
                b: 0.0,
                tx: 0.0,
                ty: 0.0,
            }
        }
    }

    /// This move, then a slide of `dx`, `dy`.
    pub(crate) fn shifted(&self, dx: f64, dy: f64) -> Self {
        match *self {
            Self::Similarity { a, b, tx, ty } => Self::Similarity {
                a,
                b,
                tx: tx + dx,
                ty: ty + dy,
            },
            Self::Homography(h) => Self::Homography([
                h[0] + dx * h[6],
                h[1] + dx * h[7],
                h[2] + dx * h[8],
                h[3] + dy * h[6],
                h[4] + dy * h[7],
                h[5] + dy * h[8],
                h[6],
                h[7],
                h[8],
            ]),
        }
    }

    /// The move back, or `None` when this one squashes the plane flat.
    pub(crate) fn inverse(&self) -> Option<Self> {
        match *self {
            Self::Similarity { a, b, tx, ty } => {
                let norm = a * a + b * b;
                if norm < 1e-12 {
                    return None;
                }
                let (ia, ib) = (a / norm, -b / norm);
                Some(Self::Similarity {
                    a: ia,
                    b: ib,
                    tx: -(ia * tx - ib * ty),
                    ty: -(ib * tx + ia * ty),
                })
            }
            Self::Homography(h) => {
                let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
                    + h[2] * (h[3] * h[7] - h[4] * h[6]);
                if det.abs() < 1e-12 || !det.is_finite() {
                    return None;
                }
                let adj = [
                    h[4] * h[8] - h[5] * h[7],
                    h[2] * h[7] - h[1] * h[8],
                    h[1] * h[5] - h[2] * h[4],
                    h[5] * h[6] - h[3] * h[8],
                    h[0] * h[8] - h[2] * h[6],
                    h[2] * h[3] - h[0] * h[5],
                    h[3] * h[7] - h[4] * h[6],
                    h[1] * h[6] - h[0] * h[7],
                    h[0] * h[4] - h[1] * h[3],
                ];
                let last = adj[8];
                if last.abs() < 1e-12 {
                    return None;
                }
                Some(Self::Homography(adj.map(|v| v / last)))
            }
        }
    }
}

/// The fit and which pairs agree with it.
pub(crate) struct Fit {
    pub(crate) model: Model,
    pub(crate) inliers: Vec<bool>,
    pub(crate) count: usize,
}

/// A small fixed-seed generator, so a track is the same on every run.
pub(crate) struct Random(u64);

impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn below(&mut self, n: usize) -> usize {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((x >> 32) as usize) % n.max(1)
    }

    /// `k` different indices below `n`.
    fn pick(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(k);
        while out.len() < k {
            let i = self.below(n);
            if !out.contains(&i) {
                out.push(i);
            }
        }
        out
    }
}

/// The best fit of `pairs` (from, to) with `plane` choosing a homography
/// over a similarity. A pair agrees when the fit puts it within
/// `threshold` pixels. `None` when no sample gives a fit.
pub(crate) fn ransac(
    pairs: &[Anchored],
    plane: bool,
    threshold: f64,
    random: &mut Random,
) -> Option<Fit> {
    let need = if plane { 4 } else { 2 };
    if pairs.len() < need {
        return None;
    }
    let pairs: Vec<Pair> = pairs
        .iter()
        .map(|&(p, q)| (p, (f64::from(q.0), f64::from(q.1))))
        .collect();
    let rounds = if plane { 300 } else { 150 };
    let mut best: Option<Fit> = None;
    for _ in 0..rounds {
        let sample: Vec<_> = random
            .pick(pairs.len(), need)
            .into_iter()
            .map(|i| pairs[i])
            .collect();
        let model = if plane {
            homography(&sample)
        } else {
            similarity(&sample)
        };
        let Some(model) = model else { continue };
        let fit = score(&pairs, model, threshold);
        if best.as_ref().is_none_or(|b| fit.count > b.count) {
            best = Some(fit);
        }
    }
    let best = best?;
    // Fit again to every pair that agreed, then count again.
    let best = refine(&pairs, best, plane, threshold);
    // The pairs that agree closely place the target best: fit once more to
    // those alone, unless that loses more than a tenth of the pairs that
    // agree at `threshold`. Which pairs agree is still judged at
    // `threshold`, so a noisy frame is not called lost.
    let close = score(&pairs, best.model, threshold / 2.0);
    if close.count >= 2 * need {
        let tight = refine(&pairs, close, plane, threshold / 2.0);
        let fit = score(&pairs, tight.model, threshold);
        if fit.count * 10 >= best.count * 9 {
            return Some(fit);
        }
    }
    Some(best)
}

/// `fit` again through the pairs it agrees with, kept when it agrees with
/// as many at `threshold`.
fn refine(pairs: &[Pair], fit: Fit, plane: bool, threshold: f64) -> Fit {
    let agreed: Vec<_> = pairs
        .iter()
        .zip(&fit.inliers)
        .filter_map(|(pair, &keep)| keep.then_some(*pair))
        .collect();
    let model = if plane {
        homography(&agreed)
    } else {
        similarity(&agreed)
    };
    match model {
        Some(model) => {
            let again = score(pairs, model, threshold);
            if again.count >= fit.count { again } else { fit }
        }
        None => fit,
    }
}

fn score(pairs: &[Pair], model: Model, threshold: f64) -> Fit {
    let inliers: Vec<bool> = pairs
        .iter()
        .map(|&(p, q)| {
            model
                .apply(p)
                .is_some_and(|m| (m.0 - q.0).powi(2) + (m.1 - q.1).powi(2) <= threshold * threshold)
        })
        .collect();
    let count = inliers.iter().filter(|&&keep| keep).count();
    Fit {
        model,
        inliers,
        count,
    }
}

/// The least-squares similarity through `pairs` (two or more).
pub(crate) fn similarity(pairs: &[Pair]) -> Option<Model> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let (mut px, mut py, mut qx, mut qy) = (0.0, 0.0, 0.0, 0.0);
    for &(p, q) in pairs {
        px += p.0;
        py += p.1;
        qx += q.0;
        qy += q.1;
    }
    let (px, py, qx, qy) = (px / n, py / n, qx / n, qy / n);
    let (mut dot, mut cross, mut norm) = (0.0, 0.0, 0.0);
    for &(p, q) in pairs {
        let (ux, uy) = (p.0 - px, p.1 - py);
        let (vx, vy) = (q.0 - qx, q.1 - qy);
        dot += ux * vx + uy * vy;
        cross += ux * vy - uy * vx;
        norm += ux * ux + uy * uy;
    }
    if norm < 1e-9 {
        return None;
    }
    let (a, b) = (dot / norm, cross / norm);
    let model = Model::Similarity {
        a,
        b,
        tx: qx - (a * px - b * py),
        ty: qy - (b * px + a * py),
    };
    [a, b].iter().all(|v| v.is_finite()).then_some(model)
}

/// The least-squares homography through `pairs` (four or more), on points
/// moved and scaled about their middle so the sums stay well conditioned.
pub(crate) fn homography(pairs: &[Pair]) -> Option<Model> {
    if pairs.len() < 4 {
        return None;
    }
    if pairs.len() == 4 && !spread(pairs) {
        return None;
    }
    let from: Vec<_> = pairs.iter().map(|&(p, _)| p).collect();
    let to: Vec<_> = pairs.iter().map(|&(_, q)| q).collect();
    let (from_shift, from_scale) = normaliser(&from)?;
    let (to_shift, to_scale) = normaliser(&to)?;
    // Normal equations of the eight unknowns, h33 = 1.
    let mut ata = [[0.0_f64; 8]; 8];
    let mut atb = [0.0_f64; 8];
    for (p, q) in from.iter().zip(&to) {
        let (x, y) = (
            (p.0 - from_shift.0) * from_scale,
            (p.1 - from_shift.1) * from_scale,
        );
        let (u, v) = ((q.0 - to_shift.0) * to_scale, (q.1 - to_shift.1) * to_scale);
        let rows = [
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ];
        for (row, rhs) in rows {
            for i in 0..8 {
                for j in 0..8 {
                    ata[i][j] += row[i] * row[j];
                }
                atb[i] += row[i] * rhs;
            }
        }
    }
    let h = solve(ata, atb)?;
    let hn = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
    // H = T_to^-1 * Hn * T_from.
    let t_from = [
        from_scale,
        0.0,
        -from_scale * from_shift.0,
        0.0,
        from_scale,
        -from_scale * from_shift.1,
        0.0,
        0.0,
        1.0,
    ];
    let t_to_inv = [
        1.0 / to_scale,
        0.0,
        to_shift.0,
        0.0,
        1.0 / to_scale,
        to_shift.1,
        0.0,
        0.0,
        1.0,
    ];
    let full = multiply(&t_to_inv, &multiply(&hn, &t_from));
    let last = full[8];
    if last.abs() < 1e-12 || !last.is_finite() {
        return None;
    }
    let out = full.map(|v| v / last);
    out.iter()
        .all(|v| v.is_finite())
        .then_some(Model::Homography(out))
}

/// Whether four points are spread out: no three of them nearly on a line.
fn spread(pairs: &[Pair]) -> bool {
    let sides = [
        pairs.iter().map(|&(p, _)| p).collect::<Vec<_>>(),
        pairs.iter().map(|&(_, q)| q).collect::<Vec<_>>(),
    ];
    sides.iter().all(|points| {
        let triangle = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
            ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() / 2.0
        };
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
            .iter()
            .all(|&(i, j, k)| triangle(points[i], points[j], points[k]) > 4.0)
    })
}

/// The middle of `points`, and the scale that puts them about √2 from it.
fn normaliser(points: &[(f64, f64)]) -> Option<((f64, f64), f64)> {
    let n = points.len() as f64;
    let middle = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let mean = points
        .iter()
        .map(|p| ((p.0 - middle.0).powi(2) + (p.1 - middle.1).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    (mean > 1e-9).then_some((middle, std::f64::consts::SQRT_2 / mean))
}

fn multiply(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    out
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..8 {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (value, &above) in a[row].iter_mut().zip(&pivot_row).skip(col) {
                *value -= f * above;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 8];
    for row in (0..8).rev() {
        let rest: f64 = (row + 1..8).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - rest) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<(f64, f64)> {
        (0..6)
            .flat_map(|y| {
                (0..6).map(move |x| (10.0 + 13.0 * f64::from(x), 7.0 + 11.0 * f64::from(y)))
            })
            .collect()
    }

    fn close(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn a_similarity_is_recovered_through_a_third_of_wrong_pairs() {
        let truth = Model::Similarity {
            a: 1.1 * 0.2_f64.cos(),
            b: 1.1 * 0.2_f64.sin(),
            tx: 5.0,
            ty: -3.0,
        };
        let mut pairs: Vec<_> = grid()
            .into_iter()
            .map(|p| {
                let q = truth.apply(p).unwrap();
                (p, (q.0 as f32, q.1 as f32))
            })
            .collect();
        for (i, pair) in pairs.iter_mut().enumerate().filter(|(i, _)| i % 3 == 0) {
            pair.1 = (pair.1.0 + 20.0 + i as f32, pair.1.1 - 15.0);
        }
        let fit = ransac(&pairs, false, 1.0, &mut Random::new(7)).unwrap();
        assert_eq!(fit.count, 24);
        assert!(close(
            fit.model.apply((50.0, 40.0)).unwrap(),
            truth.apply((50.0, 40.0)).unwrap()
        ));
    }

    #[test]
    fn a_homography_is_recovered_through_wrong_pairs() {
        let truth = Model::Homography([1.05, 0.1, 4.0, -0.05, 0.95, 2.0, 0.0008, -0.0005, 1.0]);
        let mut pairs: Vec<_> = grid()
            .into_iter()
            .map(|p| {
                let q = truth.apply(p).unwrap();
                (p, (q.0 as f32, q.1 as f32))
            })
            .collect();
        for pair in pairs.iter_mut().step_by(4) {
            pair.1 = (pair.1.0 - 30.0, pair.1.1 + 25.0);
        }
        let fit = ransac(&pairs, true, 1.0, &mut Random::new(7)).unwrap();
        assert_eq!(fit.count, 27);
        let (got, want) = (
            fit.model.apply((70.0, 60.0)).unwrap(),
            truth.apply((70.0, 60.0)).unwrap(),
        );
        assert!(
            (got.0 - want.0).abs() < 1e-3 && (got.1 - want.1).abs() < 1e-3,
            "{got:?} {want:?}"
        );
    }

    #[test]
    fn points_on_one_spot_or_one_line_give_no_fit() {
        let same = [((1.0, 1.0), (2.0, 2.0)); 5];
        assert!(similarity(&same).is_none());
        let line: Vec<_> = (0..4)
            .map(|i| ((f64::from(i), 0.0), (f64::from(i), 0.0)))
            .collect();
        assert!(homography(&line).is_none());
        assert!(ransac(&[((0.0, 0.0), (1.0, 1.0))], false, 1.0, &mut Random::new(1)).is_none());
    }
}
