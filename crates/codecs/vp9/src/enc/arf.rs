//! The temporal filter that makes an alt-ref frame's source (libvpx
//! `vp9_temporal_filter`): each 16×16 block of the centre picture is
//! averaged with its motion-compensated matches in the pictures around it,
//! each sample weighted by how closely its 3×3 neighbourhood matches. The
//! result keeps the picture and drops the noise that no single picture
//! predicts, so the frames around it predict well from it.

use crate::dsp::sad;

/// One 4:2:0 picture, planes tightly packed.
pub struct Picture<'a> {
    pub planes: [&'a [u8]; 3],
}

/// The filtered centre picture of `frames` (index `center`), `w`×`h`,
/// at `strength` (a larger one filters more).
pub fn filter(
    frames: &[Picture<'_>],
    center: usize,
    w: usize,
    h: usize,
    strength: u32,
) -> [Vec<u8>; 3] {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let dims = [(w, h), (cw, ch), (cw, ch)];
    let mut acc: [Vec<u32>; 3] = std::array::from_fn(|p| vec![0; dims[p].0 * dims[p].1]);
    let mut cnt: [Vec<u32>; 3] = std::array::from_fn(|p| vec![0; dims[p].0 * dims[p].1]);
    let target = &frames[center];
    for by in (0..h).step_by(16) {
        for bx in (0..w).step_by(16) {
            let bw = 16.min(w - bx);
            let bh = 16.min(h - by);
            for (f, pic) in frames.iter().enumerate() {
                // The match and how much the block trusts it.
                let (mv, weight) = if f == center {
                    ((0, 0), 2)
                } else {
                    let (mv, sse) = search(target.planes[0], pic.planes[0], w, h, bx, by, bw, bh);
                    let per = sse / (bw * bh) as u64;
                    (
                        mv,
                        if per < 156 {
                            2
                        } else if per < 312 {
                            1
                        } else {
                            0
                        },
                    )
                };
                if weight == 0 {
                    continue;
                }
                for p in 0..3 {
                    let ss = usize::from(p > 0);
                    let (pw, ph) = dims[p];
                    let (x0, y0) = (bx >> ss, by >> ss);
                    let (n_w, n_h) = (((bx + bw + ss) >> ss) - x0, ((by + bh + ss) >> ss) - y0);
                    let (dx, dy) = (mv.0 >> ss, mv.1 >> ss);
                    apply(
                        target.planes[p],
                        pic.planes[p],
                        (pw, ph),
                        (x0, y0, n_w, n_h),
                        (dx, dy),
                        if f == center { None } else { Some(strength) },
                        weight,
                        &mut acc[p],
                        &mut cnt[p],
                    );
                }
            }
        }
    }
    std::array::from_fn(|p| {
        acc[p]
            .iter()
            .zip(&cnt[p])
            .map(|(&a, &c)| (a + c / 2).checked_div(c).map_or(0, |v| v.min(255) as u8))
            .collect()
    })
}

/// Whether `to` predicts well from `from` (luma, `w`×`h`): most of a
/// sample of 16×16 blocks find a close match a short way off. An alt-ref
/// pays only then.
pub fn reachable(from: &[u8], to: &[u8], w: usize, h: usize) -> bool {
    let (mut good, mut all) = (0u32, 0u32);
    for by in (0..h.saturating_sub(15)).step_by(64) {
        for bx in (0..w.saturating_sub(15)).step_by(64) {
            let (mv, sse) = search(to, from, w, h, bx, by, 16, 16);
            all += 1;
            // Close, and near where it was: the content holds still enough
            // over the interval for one picture to stand for it.
            good += u32::from(sse / 256 < 156 && mv.0.abs() <= 16 && mv.1.abs() <= 16);
        }
    }
    all > 0 && good * 10 >= all * 7
}

/// Adds one picture's block to the accumulators: `strength` `None` is the
/// centre picture itself (full weight).
#[allow(clippy::too_many_arguments)]
fn apply(
    target: &[u8],
    pic: &[u8],
    (pw, ph): (usize, usize),
    (x0, y0, n_w, n_h): (usize, usize, usize, usize),
    (dx, dy): (isize, isize),
    strength: Option<u32>,
    weight: u32,
    acc: &mut [u32],
    cnt: &mut [u32],
) {
    let at = |x: isize, y: isize| -> u8 {
        let x = x.clamp(0, pw as isize - 1) as usize;
        let y = y.clamp(0, ph as isize - 1) as usize;
        pic[y * pw + x]
    };
    // Squared differences over the block and a sample around it.
    let mut diff = [[0u32; 18]; 18];
    if strength.is_some() {
        for (r, row) in diff.iter_mut().enumerate().take(n_h + 2) {
            for (c, d) in row.iter_mut().enumerate().take(n_w + 2) {
                let x = (x0 + c) as isize - 1;
                let y = (y0 + r) as isize - 1;
                let tx = x.clamp(0, pw as isize - 1) as usize;
                let ty = y.clamp(0, ph as isize - 1) as usize;
                let e = i32::from(at(x + dx, y + dy)) - i32::from(target[ty * pw + tx]);
                *d = (e * e) as u32;
            }
        }
    }
    for r in 0..n_h {
        for c in 0..n_w {
            let (x, y) = (x0 + c, y0 + r);
            let v = u32::from(at(x as isize + dx, y as isize + dy));
            let m = match strength {
                None => 16 * weight,
                Some(s) => {
                    let sum: u32 = diff[r..r + 3]
                        .iter()
                        .map(|row| row[c..c + 3].iter().sum::<u32>())
                        .sum();
                    let modifier = (sum * 3 / 9 + ((1 << s) >> 1)) >> s;
                    (16 - modifier.min(16)) * weight
                }
            };
            let i = y * pw + x;
            acc[i] += m * v;
            cnt[i] += m;
        }
    }
}

/// The whole-sample vector of the best match of the target's block in
/// `pic` (a diamond search from zero), and its SSE.
#[allow(clippy::too_many_arguments)]
fn search(
    target: &[u8],
    pic: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
) -> ((isize, isize), u64) {
    let cost = |dx: isize, dy: isize| -> Option<u64> {
        let (x, y) = (bx as isize + dx, by as isize + dy);
        if x < 0 || y < 0 || x as usize + bw > w || y as usize + bh > h {
            return None;
        }
        let (x, y) = (x as usize, y as usize);
        Some(u64::from(sad::sad(
            &target[by * w + bx..],
            w,
            &pic[y * w + x..],
            w,
            bw,
            bh,
        )))
    };
    let mut best = (0isize, 0isize);
    let mut best_sad = cost(0, 0).unwrap_or(u64::MAX);
    let mut step = 16isize;
    while step >= 1 {
        let mut moved = true;
        let mut iters = 0;
        while moved && iters < 32 {
            moved = false;
            iters += 1;
            let c = best;
            for (dx, dy) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let (x, y) = (c.0 + dx, c.1 + dy);
                if x.abs() > 64 || y.abs() > 64 {
                    continue;
                }
                if let Some(v) = cost(x, y)
                    && v < best_sad
                {
                    best_sad = v;
                    best = (x, y);
                    moved = true;
                }
            }
        }
        step /= 2;
    }
    let (x, y) = (
        (bx as isize + best.0) as usize,
        (by as isize + best.1) as usize,
    );
    let sse = sad::sse(&target[by * w + bx..], w, &pic[y * w + x..], w, bw, bh);
    (best, sse)
}
