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
    threads: usize,
    ss: usize,
) -> [Vec<u8>; 3] {
    let bands = h.div_ceil(16);
    let threads = threads.clamp(1, bands.max(1));
    let parts: Vec<Vec<(usize, [Vec<u8>; 3])>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    (t..bands)
                        .step_by(threads)
                        .map(|b| (b, band(frames, center, w, h, strength, b * 16, ss)))
                        .collect()
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut all: Vec<(usize, [Vec<u8>; 3])> = parts.into_iter().flatten().collect();
    all.sort_unstable_by_key(|p| p.0);
    std::array::from_fn(|p| all.iter().flat_map(|b| b.1[p].iter().copied()).collect())
}

/// The filtered rows of the 16-row band from `by` (and its chroma's).
fn band(
    frames: &[Picture<'_>],
    center: usize,
    w: usize,
    h: usize,
    strength: u32,
    by: usize,
    ss: usize,
) -> [Vec<u8>; 3] {
    // Chroma subsampled by `ss` both ways (1: 4:2:0, 0: 4:4:4).
    let (cw, ch) = (w.div_ceil(1 << ss), h.div_ceil(1 << ss));
    let dims = [(w, h), (cw, ch), (cw, ch)];
    let bh = 16.min(h - by);
    // The band's rows in each plane.
    let rows = [
        (by, by + bh),
        (by >> ss, (by + bh).div_ceil(1 << ss)),
        (by >> ss, (by + bh).div_ceil(1 << ss)),
    ];
    let mut acc: [Vec<u32>; 3] =
        std::array::from_fn(|p| vec![0; dims[p].0 * (rows[p].1 - rows[p].0)]);
    let mut cnt: [Vec<u32>; 3] =
        std::array::from_fn(|p| vec![0; dims[p].0 * (rows[p].1 - rows[p].0)]);
    let target = &frames[center];
    for bx in (0..w).step_by(16) {
        let bw = 16.min(w - bx);
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
                let ss = if p > 0 { ss } else { 0 };
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
                    rows[p].0,
                    &mut acc[p],
                    &mut cnt[p],
                );
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

/// How far a block may have moved over the interval and still count as
/// a match: the whole search range. (At 16, a steady pan of a sample a
/// frame never made an alt-ref, which takes 10 to 11% off its bits at
/// the same PSNR, Y or averaged over the planes.)
const REACH: isize = 64;

/// Whether `to` predicts well from `from` (luma, `w`×`h`): most of a
/// sample of 16×16 blocks find a close match within [`REACH`]. An alt-ref
/// pays only then.
pub fn reachable(from: &[u8], to: &[u8], w: usize, h: usize) -> bool {
    let (mut good, mut all) = (0u32, 0u32);
    for by in (0..h.saturating_sub(15)).step_by(64) {
        for bx in (0..w.saturating_sub(15)).step_by(64) {
            let (mv, sse) = search(to, from, w, h, bx, by, 16, 16);
            all += 1;
            // Close: the content holds together over the interval well
            // enough for one picture to stand for it.
            good += u32::from(sse / 256 < 156 && mv.0.abs() <= REACH && mv.1.abs() <= REACH);
        }
    }
    all > 0 && good * 10 >= all * 7
}

/// Whether `to` cuts from `from` (luma, `w`×`h`): over a sample of 16×16
/// blocks, the best match each finds in the search range leaves more than
/// twice the error of its own mean (libvpx compares inter and intra
/// errors the same way). Within one scene the match leaves at most 0.95 of
/// it, grain and all; across a cut 11 to 50 times it, even to another
/// picture of the same subject. (A cut to fine texture everywhere, which
/// its own mean predicts no better than anything does, is not seen; the
/// blocks of the frame after it choose intra all the same.)
pub fn cut(from: &[u8], to: &[u8], w: usize, h: usize) -> bool {
    let (mut inter, mut intra) = (0u64, 0u64);
    for by in (0..h.saturating_sub(15)).step_by(64) {
        for bx in (0..w.saturating_sub(15)).step_by(64) {
            let (_, sse) = search(to, from, w, h, bx, by, 16, 16);
            inter += sse;
            let (mut s, mut s2) = (0u64, 0u64);
            for r in 0..16 {
                for &v in &to[(by + r) * w + bx..(by + r) * w + bx + 16] {
                    s += u64::from(v);
                    s2 += u64::from(v) * u64::from(v);
                }
            }
            intra += s2 - s * s / 256;
        }
    }
    inter > intra * 2
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
    row0: usize,
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
            let i = (y - row0) * pw + x;
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
