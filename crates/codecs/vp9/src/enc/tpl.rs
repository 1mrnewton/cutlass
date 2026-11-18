//! libvpx's temporal dependency model (`vp9_tpl_model.c`, `enable-tpl`),
//! over the source pictures of an alt-ref's golden group: how much of
//! each block of the alt-ref the group's pictures take on, through the
//! pictures that predict from it and those that predict from them.
//!
//! The pictures are, as libvpx's `gf_picture`: the golden (or key) frame
//! before the group (a reference only), the alt-ref (which predicts from
//! the golden frame), then the group's pictures in order, each predicting
//! from the one before it (the golden frame for the first), the golden
//! frame and the alt-ref. Each 16×16 block of luma has an intra cost (the
//! SATD of the best of DC, vertical and horizontal prediction from the
//! source around it) and an inter cost (the SATD at the best whole-sample
//! vector into its best reference, at most the intra cost). From the last
//! picture back, a block's dependency cost is its intra cost and the flow
//! from the blocks that predict from it; it passes on to the blocks of its
//! reference its vector covers, by the area covered, that cost times the
//! share of the intra cost prediction saves (libvpx's `tpl_model_update`).
//! What libvpx's VP9 encoder does with it is [`Model::lambda_scales`]; the
//! encoder's frame-level use is [`Model::r0`].

use super::EncPixel;

/// A block's intra and inter cost, and the reference and vector of the
/// inter one.
type Cost = (f64, f64, (usize, (isize, isize)));

/// The side of the model's blocks.
const B: usize = 16;

/// How far the motion search reaches, in samples.
const RANGE: isize = 32;

/// The model of the alt-ref: its blocks' intra and dependency costs.
pub struct Model {
    cols: usize,
    rows: usize,
    intra: Vec<f64>,
    dep: Vec<f64>,
}

/// One picture's blocks: intra and inter cost, the reference and vector
/// of the inter one, and the flow from the pictures after.
struct Blocks {
    intra: Vec<f64>,
    inter: Vec<f64>,
    refs: Vec<(usize, (isize, isize))>,
    flow: Vec<f64>,
}

impl Model {
    /// The model over `pics` (luma, `w`×`h`, rows `w` apart): the golden
    /// frame, the alt-ref, then the group's pictures in order; on up to
    /// `threads` threads.
    pub fn new<P: EncPixel>(pics: &[&[P]], w: usize, h: usize, bd: u8, threads: usize) -> Self {
        let (cols, rows) = (w / B, h / B);
        let n = cols * rows;
        if pics.len() < 2 || n == 0 {
            return Self {
                cols,
                rows,
                intra: Vec::new(),
                dep: Vec::new(),
            };
        }
        // Each picture's references (none for the golden frame).
        let refs_of = |i: usize| -> Vec<usize> {
            match i {
                0 => Vec::new(),
                1 => vec![0],
                2 => vec![0, 1],
                _ => vec![0, i - 1, 1],
            }
        };
        let mut frames: Vec<Blocks> = (0..pics.len())
            .map(|i| costs(pics, i, &refs_of(i), (w, h), bd, threads))
            .collect();
        for i in (1..pics.len()).rev() {
            let (before, rest) = frames.split_at_mut(i);
            let f = &rest[0];
            for b in 0..n {
                let dep = f.intra[b] + f.flow[b];
                let (r, (dx, dy)) = f.refs[b];
                if f.intra[b] <= 0.0 || r >= i {
                    continue;
                }
                let flow = dep * (1.0 - f.inter[b] / f.intra[b]);
                if flow <= 0.0 {
                    continue;
                }
                let (bx, by) = ((b % cols * B) as isize, (b / cols * B) as isize);
                spread(&mut before[r].flow, (cols, rows), (bx + dx, by + dy), flow);
            }
        }
        let arf = &frames[1];
        Self {
            cols,
            rows,
            intra: arf.intra.clone(),
            dep: (0..n).map(|b| arf.intra[b] + arf.flow[b]).collect(),
        }
    }

    /// libvpx's `r0`: the alt-ref's intra cost over its dependency cost,
    /// the share of what the group takes on that it does not take from
    /// the alt-ref (1: nothing; small: the group leans on it). `None`
    /// without blocks.
    pub fn r0(&self) -> Option<f64> {
        let dep: f64 = self.dep.iter().sum();
        (dep > 0.0).then(|| self.intra.iter().sum::<f64>() / dep)
    }

    /// Each 64×64 superblock's scale of the RD lambda (`sb_cols` a row,
    /// `sb_rows` of them): its own `rk` (intra over dependency cost) over
    /// [`Model::r0`], from half to 1.5 (libvpx's `get_rdmult_delta`):
    /// superblocks the group leans on more than on the rest weigh their
    /// error more. Superblocks past the model's blocks take 1.
    pub fn lambda_scales(&self, sb_cols: usize, sb_rows: usize) -> Vec<f32> {
        let Some(r0) = self.r0() else {
            return vec![1.0; sb_cols * sb_rows];
        };
        let per = 64 / B;
        let mut out = Vec::with_capacity(sb_cols * sb_rows);
        for sr in 0..sb_rows {
            for sc in 0..sb_cols {
                let (mut intra, mut dep) = (0.0, 0.0);
                for r in sr * per..((sr + 1) * per).min(self.rows) {
                    for c in sc * per..((sc + 1) * per).min(self.cols) {
                        intra += self.intra[r * self.cols + c];
                        dep += self.dep[r * self.cols + c];
                    }
                }
                let scale = if dep > 0.0 && intra > 0.0 {
                    (intra / dep / r0).clamp(0.5, 1.5)
                } else {
                    1.0
                };
                out.push(scale as f32);
            }
        }
        out
    }
}

/// Picture `i`'s block costs against its references `refs`.
fn costs<P: EncPixel>(
    pics: &[&[P]],
    i: usize,
    refs: &[usize],
    (w, h): (usize, usize),
    bd: u8,
    threads: usize,
) -> Blocks {
    let (cols, rows) = (w / B, h / B);
    let row = |r: usize| -> Vec<Cost> {
        let mut out = Vec::with_capacity(cols);
        // (Each reference's search starts from the vector the block to the
        // left found in it too.)
        let mut left = vec![(0isize, 0isize); refs.len()];
        for c in 0..cols {
            let (bx, by) = (c * B, r * B);
            let intra = intra_cost(pics[i], w, (bx, by), bd) as f64;
            let mut best = (intra, (i, (0, 0)));
            for (j, &rf) in refs.iter().enumerate() {
                let (mv, cost) = search(pics[i], pics[rf], (w, h), (bx, by), left[j]);
                if (cost as f64) < best.0 {
                    best = (cost as f64, (rf, mv));
                }
                left[j] = mv;
            }
            out.push((intra, best.0, best.1));
        }
        out
    };
    let threads = threads.clamp(1, rows.max(1));
    let per = rows.div_ceil(threads);
    let rows_out: Vec<Vec<_>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let row = &row;
                s.spawn(move || {
                    (t * per..((t + 1) * per).min(rows))
                        .map(row)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    });
    let mut b = Blocks {
        intra: Vec::with_capacity(cols * rows),
        inter: Vec::with_capacity(cols * rows),
        refs: Vec::with_capacity(cols * rows),
        flow: vec![0.0; cols * rows],
    };
    for (intra, inter, rf) in rows_out.into_iter().flatten() {
        b.intra.push(intra);
        b.inter.push(inter.min(intra));
        b.refs.push(rf);
    }
    b
}

/// The SATD of the best of DC, vertical and horizontal prediction of the
/// block at `(bx, by)` from the source samples above and to its left.
fn intra_cost<P: EncPixel>(pic: &[P], w: usize, (bx, by): (usize, usize), bd: u8) -> u64 {
    let src = &pic[by * w + bx..];
    let mid = 1i32 << (bd - 1);
    let above: Option<Vec<i32>> =
        (by > 0).then(|| (0..B).map(|c| pic[(by - 1) * w + bx + c].get()).collect());
    let left: Option<Vec<i32>> =
        (bx > 0).then(|| (0..B).map(|r| pic[(by + r) * w + bx - 1].get()).collect());
    let dc = match (&above, &left) {
        (Some(a), Some(l)) => {
            (a.iter().sum::<i32>() + l.iter().sum::<i32>() + B as i32) / (2 * B as i32)
        }
        (Some(a), None) => (a.iter().sum::<i32>() + B as i32 / 2) / B as i32,
        (None, Some(l)) => (l.iter().sum::<i32>() + B as i32 / 2) / B as i32,
        (None, None) => mid,
    };
    let mut pred = vec![P::default(); B * B];
    let mut cost = |f: &dyn Fn(usize, usize) -> i32| {
        for r in 0..B {
            for c in 0..B {
                pred[r * B + c] = P::new(f(r, c));
            }
        }
        P::satd(src, w, &pred, B, B, B)
    };
    let mut best = cost(&|_, _| dc);
    if let Some(a) = &above {
        best = best.min(cost(&|_, c| a[c]));
    }
    if let Some(l) = &left {
        best = best.min(cost(&|r, _| l[r]));
    }
    best
}

/// The best whole-sample vector of the block at `(bx, by)` of `cur` into
/// `rf` (a diamond search from zero and from `seed`, within [`RANGE`] and
/// the picture), and the SATD there.
fn search<P: EncPixel>(
    cur: &[P],
    rf: &[P],
    (w, h): (usize, usize),
    (bx, by): (usize, usize),
    seed: (isize, isize),
) -> ((isize, isize), u64) {
    let src = &cur[by * w + bx..];
    let inside = |(dx, dy): (isize, isize)| {
        let (x, y) = (bx as isize + dx, by as isize + dy);
        dx.abs() <= RANGE
            && dy.abs() <= RANGE
            && x >= 0
            && y >= 0
            && x as usize + B <= w
            && y as usize + B <= h
    };
    let sad = |(dx, dy): (isize, isize)| {
        let (x, y) = ((bx as isize + dx) as usize, (by as isize + dy) as usize);
        P::sad(src, w, &rf[y * w + x..], w, B, B)
    };
    let mut best = (0isize, 0isize);
    let mut best_sad = sad(best);
    if seed != (0, 0) && inside(seed) {
        let v = sad(seed);
        if v < best_sad {
            best = seed;
            best_sad = v;
        }
    }
    let mut step = 8isize;
    while step >= 1 {
        loop {
            let c = best;
            for d in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let m = (c.0 + d.0, c.1 + d.1);
                if inside(m) {
                    let v = sad(m);
                    if v < best_sad {
                        best_sad = v;
                        best = m;
                    }
                }
            }
            if best == c {
                break;
            }
        }
        step /= 2;
    }
    let (x, y) = (
        (bx as isize + best.0) as usize,
        (by as isize + best.1) as usize,
    );
    let mut pred = vec![P::default(); B * B];
    for r in 0..B {
        pred[r * B..(r + 1) * B].copy_from_slice(&rf[(y + r) * w + x..(y + r) * w + x + B]);
    }
    (best, P::satd(src, w, &pred, B, B, B))
}

/// Adds `flow` to the blocks of a picture (`cols`×`rows` of them) a block
/// at `(x, y)` covers, by the area it covers of each.
fn spread(flow: &mut [f64], (cols, rows): (usize, usize), (x, y): (isize, isize), amount: f64) {
    let b = B as isize;
    let (c0, r0) = (x.div_euclid(b), y.div_euclid(b));
    for (r, c) in [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)] {
        if r < 0 || c < 0 || r as usize >= rows || c as usize >= cols {
            continue;
        }
        let ow = (b - (x - c * b).abs()).max(0);
        let oh = (b - (y - r * b).abs()).max(0);
        if ow > 0 && oh > 0 {
            flow[r as usize * cols + c as usize] += amount * (ow * oh) as f64 / (b * b) as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_the_group_copies_takes_on_its_cost() {
        // A textured picture that the next ones copy (still): every block of
        // the "alt-ref" (picture 1) is predicted exactly by the pictures
        // after it, so its dependency cost is many times its intra cost
        // (r0 far under 1); a group whose pictures are new noise each
        // time takes nothing on (r0 near 1).
        let (w, h) = (64usize, 64usize);
        let tex: Vec<u8> = (0..w * h)
            .map(|k| ((k * 2_654_435_761usize) >> 13) as u8)
            .collect();
        // (The golden frame before is another picture: the group takes
        // nothing from it.)
        let other: Vec<u8> = (0..w * h)
            .map(|k| ((k * 2_246_822_519usize) >> 11) as u8)
            .collect();
        let mut still: Vec<&[u8]> = vec![&tex; 6];
        still[0] = &other;
        let m = Model::new(&still, w, h, 8, 2);
        let r0 = m.r0().expect("blocks");
        assert!(r0 < 0.3, "still: {r0}");
        let noise: Vec<Vec<u8>> = (0..6usize)
            .map(|i| {
                (0..w * h)
                    .map(|k| {
                        let mut x = (k as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            ^ ((i as u64 + 1) << 40);
                        x ^= x >> 29;
                        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        (x >> 56) as u8
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = noise.iter().map(Vec::as_slice).collect();
        let m = Model::new(&refs, w, h, 8, 1);
        let r0 = m.r0().expect("blocks");
        assert!(r0 > 0.9, "noise: {r0}");
        assert_eq!(m.lambda_scales(1, 1).len(), 1);
    }
}
