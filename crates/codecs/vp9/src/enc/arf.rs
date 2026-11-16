//! The temporal filter that makes an alt-ref frame's source (libvpx
//! `vp9_temporal_filter`): each 16×16 block of the centre picture is
//! averaged with its motion-compensated matches in the pictures around it,
//! each sample weighted by how closely its 3×3 neighbourhood matches. The
//! result keeps the picture and drops the noise that no single picture
//! predicts, so the frames around it predict well from it.

use crate::dsp::pixel::Pixel;

use super::EncPixel;

/// One picture, planes tightly packed.
pub struct Picture<'a, P> {
    pub planes: [&'a [P]; 3],
}

/// The filtered centre picture of `frames` (index `center`), `w`×`h` of
/// `bd` bits a sample, at `strength` (a larger one filters more). Each
/// block's search starts from zero and from the match of the block to its
/// left; with `seeded` (a second pass's effort), a poor match is searched
/// again coarsely ([`COARSE_PER`]).
#[allow(clippy::too_many_arguments)]
pub fn filter<P: EncPixel>(
    frames: &[Picture<'_, P>],
    center: usize,
    w: usize,
    h: usize,
    strength: u32,
    threads: usize,
    ss: (usize, usize),
    bd: u8,
    seeded: bool,
) -> [Vec<P>; 3] {
    let bands = h.div_ceil(16);
    let threads = threads.clamp(1, bands.max(1));
    let small: Vec<Small> = if seeded {
        frames
            .iter()
            .map(|f| Small::new(f.planes[0], w, h))
            .collect()
    } else {
        Vec::new()
    };
    let small = &small;
    let parts: Vec<Vec<(usize, [Vec<P>; 3])>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    (t..bands)
                        .step_by(threads)
                        .map(|b| {
                            (
                                b,
                                band(frames, small, center, w, h, strength, b * 16, ss, bd),
                            )
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut all: Vec<(usize, [Vec<P>; 3])> = parts.into_iter().flatten().collect();
    all.sort_unstable_by_key(|p| p.0);
    std::array::from_fn(|p| all.iter().flat_map(|b| b.1[p].iter().copied()).collect())
}

/// The filtered rows of the 16-row band from `by` (and its chroma's).
/// Above 8 bits the error thresholds and the filter's strength scale with
/// the squared errors (libvpx's high bit depth filter adds `2 (bd - 8)`
/// to the strength).
#[allow(clippy::too_many_arguments)]
fn band<P: EncPixel>(
    frames: &[Picture<'_, P>],
    small: &[Small],
    center: usize,
    w: usize,
    h: usize,
    strength: u32,
    by: usize,
    (sx, sy): (usize, usize),
    bd: u8,
) -> [Vec<P>; 3] {
    let sq = 2 * u32::from(bd - 8);
    let max = (1u32 << bd) - 1;
    // Chroma subsampled `sx` across and `sy` down.
    let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
    let dims = [(w, h), (cw, ch), (cw, ch)];
    let bh = 16.min(h - by);
    // The band's rows in each plane.
    let rows = [
        (by, by + bh),
        (by >> sy, (by + bh).div_ceil(1 << sy)),
        (by >> sy, (by + bh).div_ceil(1 << sy)),
    ];
    let mut acc: [Vec<u32>; 3] =
        std::array::from_fn(|p| vec![0; dims[p].0 * (rows[p].1 - rows[p].0)]);
    let mut cnt: [Vec<u32>; 3] =
        std::array::from_fn(|p| vec![0; dims[p].0 * (rows[p].1 - rows[p].0)]);
    let target = &frames[center];
    // Each picture's match of the block to the left, where the search
    // also starts.
    let mut left = vec![(0isize, 0isize); frames.len()];
    for bx in (0..w).step_by(16) {
        let bw = 16.min(w - bx);
        for (f, pic) in frames.iter().enumerate() {
            // The match and how much the block trusts it.
            let (mv, weight) = if f == center {
                ((0, 0), 2)
            } else {
                let (t, p) = (target.planes[0], pic.planes[0]);
                // Seeded by the match of the block to the left: from zero
                // alone, the diamond stops in the valleys of smooth
                // content (on a pan of a sinusoid, a block in five
                // matched two samples off its exact match, with an error
                // the weights still take, and blurred the alt-ref).
                let seeds = &[left[f]];
                let (mut mv, mut sse) = search(t, p, w, h, bx, by, bw, bh, seeds, 16, FILTER_REACH);
                if (sse / (bw * bh) as u64) >> sq >= COARSE_PER
                    && let (Some(a), Some(b)) = (small.get(center), small.get(f))
                {
                    // A poor match: the best on sums of 4×4 samples within
                    // the search range, refined.
                    let c = a.find(b, bx / 4, by / 4, bw / 4, bh / 4);
                    let (m2, s2) = search(t, p, w, h, bx, by, bw, bh, &[c], 2, FILTER_REACH);
                    if s2 < sse {
                        (mv, sse) = (m2, s2);
                    }
                }
                left[f] = mv;
                let per = (sse / (bw * bh) as u64) >> sq;
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
                let (sx, sy) = if p > 0 { (sx, sy) } else { (0, 0) };
                let (pw, ph) = dims[p];
                let (x0, y0) = (bx >> sx, by >> sy);
                let (n_w, n_h) = (((bx + bw + sx) >> sx) - x0, ((by + bh + sy) >> sy) - y0);
                let (dx, dy) = (mv.0 >> sx, mv.1 >> sy);
                // A vector odd in luma falls between two chroma samples.
                let half = (sx == 1 && mv.0 & 1 != 0, sy == 1 && mv.1 & 1 != 0);
                apply(
                    target.planes[p],
                    pic.planes[p],
                    (pw, ph),
                    (x0, y0, n_w, n_h),
                    (dx, dy),
                    half,
                    if f == center {
                        None
                    } else {
                        Some(strength + sq)
                    },
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
            .map(|(&a, &c)| P::new((a + c / 2).checked_div(c).map_or(0, |v| v.min(max)) as i32))
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
pub fn reachable<P: EncPixel>(from: &[P], to: &[P], w: usize, h: usize, bd: u8) -> bool {
    let (mut good, mut all) = (0u32, 0u32);
    for by in (0..h.saturating_sub(15)).step_by(64) {
        for bx in (0..w.saturating_sub(15)).step_by(64) {
            let (mv, sse) = search(to, from, w, h, bx, by, 16, 16, &[], 16, 0);
            all += 1;
            // Close: the content holds together over the interval well
            // enough for one picture to stand for it.
            let per = (sse / 256) >> (2 * u32::from(bd - 8));
            good += u32::from(per < 156 && mv.0.abs() <= REACH && mv.1.abs() <= REACH);
        }
    }
    all > 0 && good * 10 >= all * 7
}

/// [`motion`] is 0 when under one block in this many of its sample moved:
/// a still picture with a few things moving across it. FFmpeg's
/// `testsrc2` (a twentieth of its blocks move, 12 samples a picture) had
/// golden frames every 6 pictures in two passes; libvpx gives it 16. At 12
/// (with the other changes of that round, on the README's three scenes,
/// two threads), 1.3 points fewer bits at the same PSNR-Y, 0.7 and 0.8 at
/// PSNR-U and V.
const MOVED_SHARE: usize = 10;

/// How far `to` moved from `from` (luma, `w`×`h`), in whole samples: over
/// a sample of 16×16 blocks whose best match (on sums of 4×4 samples,
/// refined) leaves under a quarter of the error of none (detail that
/// moved: on flat areas, grain and all, any vector matches), the median
/// of the larger component of their vectors. 0 when under eight blocks,
/// or under one in [`MOVED_SHARE`], moved so.
pub fn motion<P: EncPixel>(from: &[P], to: &[P], w: usize, h: usize) -> u32 {
    let (sf, st) = (Small::new(from, w, h), Small::new(to, w, h));
    let mut m = Vec::new();
    for by in (0..h.saturating_sub(15)).step_by(32) {
        for bx in (0..w.saturating_sub(15)).step_by(32) {
            let c = st.find(&sf, bx / 4, by / 4, 4, 4);
            let (mv, sse) = search(to, from, w, h, bx, by, 16, 16, &[c], 2, 0);
            let still = P::sse(&to[by * w + bx..], w, &from[by * w + bx..], w, 16, 16);
            if sse * 4 < still {
                m.push(mv.0.unsigned_abs().max(mv.1.unsigned_abs()) as u32);
            }
        }
    }
    let all = h.saturating_sub(15).div_ceil(32) * w.saturating_sub(15).div_ceil(32);
    if m.len() < 8 || m.len() * MOVED_SHARE < all {
        return 0;
    }
    m.sort_unstable();
    m[m.len() / 2]
}

/// How closely `from` predicts `to` (planes tightly packed, luma `w`×`h`,
/// chroma subsampled `sx` across and `sy` down) at whole luma samples: over
/// a sample of 16×16 blocks, each block's best whole-sample luma match
/// near `seed` (the vector the content moved by), the SSE a luma sample of
/// the luma, and the SSE of the chroma there, a chroma sample between two
/// (a vector odd in luma) as their mean, as the filters blur it.
pub fn phase_error<P: EncPixel>(
    from: [&[P]; 3],
    to: [&[P]; 3],
    w: usize,
    h: usize,
    (sx, sy): (usize, usize),
    seed: (isize, isize),
) -> (f32, u64) {
    let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
    let (mut luma, mut chroma, mut blocks) = (0u64, 0u64, 0u64);
    for by in (16..h.saturating_sub(31)).step_by(48) {
        for bx in (16..w.saturating_sub(31)).step_by(48) {
            let (mv, sse) = search(to[0], from[0], w, h, bx, by, 16, 16, &[seed], 8, 0);
            luma += sse;
            blocks += 1;
            // The chroma block and where its match starts, in halves of a
            // chroma sample.
            let (x0, y0, bw, bh) = (bx >> sx, by >> sy, 16 >> sx, 16 >> sy);
            let hx = ((bx as isize + mv.0) << 1) >> sx;
            let hy = ((by as isize + mv.1) << 1) >> sy;
            let (fx, fy) = (hx >> 1, hy >> 1);
            if fx < 0 || fy < 0 || fx as usize + bw + 1 > cw || fy as usize + bh + 1 > ch {
                continue;
            }
            let (fx, fy, ox, oy) = (
                fx as usize,
                fy as usize,
                (hx & 1) as usize,
                (hy & 1) as usize,
            );
            for p in 1..3 {
                let (t, f) = (to[p], from[p]);
                for r in 0..bh {
                    for k in 0..bw {
                        let at = |dy: usize, dx: usize| f[(fy + r + dy) * cw + fx + k + dx].get();
                        let pred = (at(0, 0) + at(0, ox) + at(oy, 0) + at(oy, ox) + 2) >> 2;
                        let d = i64::from(t[(y0 + r) * cw + x0 + k].get() - pred);
                        chroma += (d * d) as u64;
                    }
                }
            }
        }
    }
    (luma as f32 / (blocks.max(1) * 256) as f32, chroma)
}

/// The whole-sample vector `to` moved by from `from` (luma, `w`×`h`), each
/// component the median of those of a sample of 16×16 blocks of detail
/// that moved (as [`motion`]), searched near `guess` (the vector the
/// picture before moved by: a pan goes on); zero when under eight did.
pub fn shift<P: EncPixel>(
    from: &[P],
    to: &[P],
    w: usize,
    h: usize,
    guess: (isize, isize),
) -> (isize, isize) {
    let (sf, st) = (Small::new(from, w, h), Small::new(to, w, h));
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let near = (guess.0.div_euclid(4), guess.1.div_euclid(4));
    for by in (0..h.saturating_sub(15)).step_by(32) {
        for bx in (0..w.saturating_sub(15)).step_by(32) {
            let c = st.find_near(&sf, bx / 4, by / 4, 4, 4, near, SHIFT_RANGE);
            let (mv, sse) = search(to, from, w, h, bx, by, 16, 16, &[c, guess], 2, 0);
            let still = P::sse(&to[by * w + bx..], w, &from[by * w + bx..], w, 16, 16);
            if sse * 4 < still {
                xs.push(mv.0);
                ys.push(mv.1);
            }
        }
    }
    if xs.len() < 8 {
        return (0, 0);
    }
    xs.sort_unstable();
    ys.sort_unstable();
    (xs[xs.len() / 2], ys[ys.len() / 2])
}

/// A picture's luma detail by row and by column, for [`Profile::shift`]:
/// the differences of each sample from the one above summed across each
/// row, and from the one to its left summed down each column (every
/// other row).
pub struct Profile {
    cols: Vec<u32>,
    rows: Vec<u32>,
}

impl Profile {
    /// The profile of `y` (`w`×`h`, rows `stride` apart).
    pub fn new<P: Pixel>(y: &[P], stride: usize, w: usize, h: usize) -> Self {
        cutlass_simd::vp9::wide(
            #[inline(always)]
            || Self::build(y, stride, w, h),
        )
    }

    #[inline(always)]
    fn build<P: Pixel>(y: &[P], stride: usize, w: usize, h: usize) -> Self {
        // (Differences in 16 bits, a lane each.)
        let d = |a: &P, b: &P| {
            let (a, b) = (a.get() as i16, b.get() as i16);
            (a.max(b) - a.min(b)) as u16
        };
        let mut cols = vec![0u32; w];
        let mut acc = vec![0u16; w];
        let mut rows = Vec::with_capacity(h);
        rows.push(0);
        for r in 1..h {
            let line = &y[r * stride..r * stride + w];
            let up = &y[(r - 1) * stride..(r - 1) * stride + w];
            rows.push(line.iter().zip(up).map(|(a, b)| u32::from(d(a, b))).sum());
            if r % 2 == 0 {
                for ((c, a), b) in acc[1..].iter_mut().zip(&line[1..]).zip(&line[..w - 1]) {
                    *c += d(a, b);
                }
                // (Under 16 rows of 4095 a sum: carried into 32 bits.)
                if r % 32 == 0 || r + 2 >= h {
                    for (c, a) in cols.iter_mut().zip(acc.iter_mut()) {
                        *c += u32::from(std::mem::take(a));
                    }
                }
            }
        }
        Self { cols, rows }
    }

    /// The whole-sample vector the picture of `to` moved by from this
    /// one's, as [`shift`]'s (`to[p]` near `from[p + v]`), for content
    /// that moves as a whole (a pan): across and down, the offset within
    /// `range` at which the profiles differ least, a sum on average over
    /// the sums both cover (the shorter offset on a tie).
    pub fn shift(&self, to: &Profile, range: isize) -> (isize, isize) {
        #[inline(always)]
        fn best(a: &[u32], b: &[u32], range: isize) -> isize {
            let n = a.len().min(b.len()) as isize;
            let mut best = (u64::MAX, 0isize);
            for d in -range..=range {
                // b[i] against a[i + d], over the i both cover.
                let (lo, hi) = ((-d).max(0), (n - d).min(n));
                if hi - lo < n / 2 {
                    continue;
                }
                let (lo, hi) = (lo as usize, hi as usize);
                let off = (lo as isize + d) as usize;
                let s: u64 = a[off..off + (hi - lo)]
                    .iter()
                    .zip(&b[lo..hi])
                    .map(|(&x, &y)| u64::from(x.abs_diff(y)))
                    .sum();
                let cost = s * 1024 / (hi - lo) as u64 * 64 + d.unsigned_abs() as u64;
                if cost < best.0 {
                    best = (cost, d);
                }
            }
            best.1
        }
        cutlass_simd::vp9::wide(
            #[inline(always)]
            || {
                (
                    best(&self.cols, &to.cols, range),
                    best(&self.rows, &to.rows, range),
                )
            },
        )
    }
}

/// Whether `to` cuts from `from` (luma, `w`×`h`): over a sample of 16×16
/// blocks, the best match each finds in the search range leaves more than
/// twice the error of its own mean (libvpx compares inter and intra
/// errors the same way). Within one scene the match leaves at most 0.95 of
/// it, grain and all; across a cut 11 to 50 times it, even to another
/// picture of the same subject. (A cut to fine texture everywhere, which
/// its own mean predicts no better than anything does, is not seen; the
/// blocks of the frame after it choose intra all the same.)
pub fn cut<P: EncPixel>(from: &[P], to: &[P], w: usize, h: usize) -> bool {
    let rows: Vec<usize> = (0..h.saturating_sub(15)).step_by(64).collect();
    let (inter, intra) = cut_rows(from, to, w, h, &rows);
    inter > intra * 2
}

/// Each of `rows` split among `threads` threads, `f` of each share.
fn on_rows<R: Send>(rows: &[usize], threads: usize, f: impl Fn(&[usize]) -> R + Sync) -> Vec<R> {
    let share = rows.len().div_ceil(threads.max(1)).max(1);
    if threads <= 1 || rows.len() <= share {
        return vec![f(rows)];
    }
    let f = &f;
    std::thread::scope(|sc| {
        let hs: Vec<_> = rows.chunks(share).map(|c| sc.spawn(move || f(c))).collect();
        hs.into_iter()
            .map(|h| h.join())
            .collect::<Result<Vec<R>, _>>()
    })
    .unwrap_or_else(|_| vec![f(rows)])
}

/// [`cut`]'s sums over the block rows at `rows`: the best matches' errors
/// and the blocks' own.
fn cut_rows<P: EncPixel>(from: &[P], to: &[P], w: usize, h: usize, rows: &[usize]) -> (u64, u64) {
    let (mut inter, mut intra) = (0u64, 0u64);
    for &by in rows {
        for bx in (0..w.saturating_sub(15)).step_by(64) {
            let (_, sse) = search(to, from, w, h, bx, by, 16, 16, &[], 16, 0);
            inter += sse;
            let (mut s, mut s2) = (0u64, 0u64);
            for r in 0..16 {
                for &v in &to[(by + r) * w + bx..(by + r) * w + bx + 16] {
                    let v = v.get() as u64;
                    s += v;
                    s2 += v * v;
                }
            }
            intra += s2 - s * s / 256;
        }
    }
    (inter, intra)
}

/// How far (in sums of 4×4 samples) [`jump`] looks.
const JUMP_RANGE: isize = 80;

/// When `to` cuts from `from` by [`cut`] (luma, `w`×`h`), whether it is
/// the same picture moved far, as a camera that jumps: the vector (whole
/// samples, across then down) at which most 32×32 blocks of detail of a
/// sample find a match in `from` within [`JUMP_RANGE`] sums, or `None`
/// for a cut. Blocks the same vector predicts leave under a quarter of
/// their own variance; a cut to another picture leaves no vector more
/// than a few blocks agree on.
/// The blocks are measured on `threads` threads, a share of the rows each.
pub fn jump<P: EncPixel>(
    from: &[P],
    to: &[P],
    w: usize,
    h: usize,
    threads: usize,
) -> Option<(isize, isize)> {
    let (sf, st) = (Small::new(from, w, h), Small::new(to, w, h));
    let rows: Vec<usize> = (16..h.saturating_sub(47)).step_by(96).collect();
    let mut votes: Vec<(isize, isize)> = Vec::new();
    let mut detail = 0usize;
    for (v, d) in on_rows(&rows, threads, |rows| {
        jump_rows(from, to, (w, h), (&sf, &st), rows)
    }) {
        votes.extend(v);
        detail += d;
    }
    votes.sort_unstable();
    let mut best = (0usize, (0isize, 0isize));
    let mut i = 0;
    while i < votes.len() {
        let j = votes[i..].iter().take_while(|&&v| v == votes[i]).count();
        if j > best.0 {
            best = (j, votes[i]);
        }
        i += j;
    }
    (best.0 >= 4 && best.0 * 3 >= detail && best.1 != (0, 0)).then_some(best.1)
}

/// [`jump`]'s blocks of detail in the block rows at `rows`: the vectors
/// that predict them, and how many there are.
#[allow(clippy::type_complexity)]
fn jump_rows<P: EncPixel>(
    from: &[P],
    to: &[P],
    (w, h): (usize, usize),
    (sf, st): (&Small, &Small),
    rows: &[usize],
) -> (Vec<(isize, isize)>, usize) {
    let mut votes = Vec::new();
    let mut detail = 0usize;
    for &by in rows {
        for bx in (16..w.saturating_sub(47)).step_by(96) {
            let (mut s, mut s2) = (0u64, 0u64);
            for r in 0..32 {
                for &v in &to[(by + r) * w + bx..(by + r) * w + bx + 32] {
                    let v = v.get() as u64;
                    s += v;
                    s2 += v * v;
                }
            }
            let var = s2 - s * s / 1024;
            // Flat blocks match anywhere.
            if var < 1024 * 16 {
                continue;
            }
            detail += 1;
            let c = st.find_near(sf, bx / 4, by / 4, 8, 8, (0, 0), JUMP_RANGE);
            // The sums' match, to the sample.
            let mut best = (u64::MAX, c);
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let (x, y) = (bx as isize + c.0 + dx, by as isize + c.1 + dy);
                    if x < 0 || y < 0 || x as usize + 32 > w || y as usize + 32 > h {
                        continue;
                    }
                    let (x, y) = (x as usize, y as usize);
                    let sse = P::sse(&to[by * w + bx..], w, &from[y * w + x..], w, 32, 32);
                    if sse < best.0 {
                        best = (sse, (c.0 + dx, c.1 + dy));
                    }
                }
            }
            if best.0 * 4 < var {
                votes.push(best.1);
            }
        }
    }
    (votes, detail)
}

/// Adds one picture's block to the accumulators: `strength` `None` is the
/// centre picture itself (full weight).
#[allow(clippy::too_many_arguments)]
fn apply<P: Pixel>(
    target: &[P],
    pic: &[P],
    (pw, ph): (usize, usize),
    (x0, y0, n_w, n_h): (usize, usize, usize, usize),
    (dx, dy): (isize, isize),
    (hx, hy): (bool, bool),
    strength: Option<u32>,
    weight: u32,
    row0: usize,
    acc: &mut [u32],
    cnt: &mut [u32],
) {
    let one = |x: isize, y: isize| -> i32 {
        let x = x.clamp(0, pw as isize - 1) as usize;
        let y = y.clamp(0, ph as isize - 1) as usize;
        pic[y * pw + x].get()
    };
    // The match's sample, or between two (or four) of them, their mean
    // (`hx`, `hy`: a chroma match half a sample on).
    let (ox, oy) = (isize::from(hx), isize::from(hy));
    let at = |x: isize, y: isize| -> i32 {
        if hx || hy {
            (one(x, y) + one(x + ox, y) + one(x, y + oy) + one(x + ox, y + oy) + 2) >> 2
        } else {
            one(x, y)
        }
    };
    // Whether the block, a sample around it, and its match (and the
    // samples after it a half reads) all lie inside the picture: no sample
    // then needs clamping.
    let inside = x0 >= 1
        && y0 >= 1
        && x0 + n_w < pw
        && y0 + n_h < ph
        && x0 as isize + dx >= 1
        && y0 as isize + dy >= 1
        && x0 as isize + dx + (n_w as isize) + ox < pw as isize
        && y0 as isize + dy + (n_h as isize) + oy < ph as isize;
    // Inside, the match's sample at `i`.
    let (ou, ov) = (ox as usize, oy as usize * pw);
    let fetch = |i: usize| -> i32 {
        if hx || hy {
            (pic[i].get() + pic[i + ou].get() + pic[i + ov].get() + pic[i + ou + ov].get() + 2) >> 2
        } else {
            pic[i].get()
        }
    };
    // Squared differences over the block and a sample around it, and
    // their sums over three across.
    let mut across = [[0u32; 16]; 18];
    if strength.is_some() {
        let mut diff = [0u32; 18];
        for (r, row) in across.iter_mut().enumerate().take(n_h + 2) {
            let y = (y0 + r) as isize - 1;
            if inside {
                let (t, m) = (y as usize * pw + x0 - 1, (y + dy) as usize * pw);
                let m = (m as isize + x0 as isize - 1 + dx) as usize;
                for (c, d) in diff.iter_mut().enumerate().take(n_w + 2) {
                    let e = fetch(m + c) - target[t + c].get();
                    *d = (e * e) as u32;
                }
            } else {
                let ty = y.clamp(0, ph as isize - 1) as usize;
                for (c, d) in diff.iter_mut().enumerate().take(n_w + 2) {
                    let x = (x0 + c) as isize - 1;
                    let tx = x.clamp(0, pw as isize - 1) as usize;
                    let e = at(x + dx, y + dy) - target[ty * pw + tx].get();
                    *d = (e * e) as u32;
                }
            }
            for (c, a) in row.iter_mut().enumerate().take(n_w) {
                *a = diff[c] + diff[c + 1] + diff[c + 2];
            }
        }
    }
    for r in 0..n_h {
        let y = y0 + r;
        let base = (y - row0) * pw;
        let m0 = (y as isize + dy) * pw as isize + dx;
        let sums = across[r].iter().zip(&across[r + 1]).zip(&across[r + 2]);
        for (c, ((s0, s1), s2)) in sums.enumerate().take(n_w) {
            let x = x0 + c;
            let v = if inside {
                fetch((m0 + x as isize) as usize) as u32
            } else {
                at(x as isize + dx, y as isize + dy) as u32
            };
            let m = match strength {
                None => 16 * weight,
                Some(s) => {
                    let sum = s0 + s1 + s2;
                    let modifier = (sum * 3 / 9 + ((1 << s) >> 1)) >> s;
                    (16 - modifier.min(16)) * weight
                }
            };
            let i = base + x;
            acc[i] += m * v;
            cnt[i] += m;
        }
    }
}

/// How far past the picture's edges (in samples, the edge samples
/// repeated as the filter reads them) the temporal filter's search looks,
/// for content that moves in from an edge: a block at the edge of a pan
/// otherwise matched what lies inside, a few samples off. (The analyses
/// that share [`search`] keep to the picture.)
const FILTER_REACH: isize = 16;

/// The whole-sample vector of the best match of the target's block in
/// `pic` (a diamond search from zero or the best of `seeds`, its first
/// step `step`; the match up to `reach` samples past the picture's edges),
/// and its SSE.
#[allow(clippy::too_many_arguments)]
fn search<P: EncPixel>(
    target: &[P],
    pic: &[P],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    seeds: &[(isize, isize)],
    mut step: isize,
    reach: isize,
) -> ((isize, isize), u64) {
    let cost = |dx: isize, dy: isize| -> Option<u64> {
        let (x, y) = (bx as isize + dx, by as isize + dy);
        if x < 0 || y < 0 || x as usize + bw > w || y as usize + bh > h {
            // Past the edge: the edge samples repeated, as the filter reads
            // them.
            if x < -reach
                || y < -reach
                || x + (bw as isize) > w as isize + reach
                || y + (bh as isize) > h as isize + reach
            {
                return None;
            }
            return Some(edge_err(
                target,
                pic,
                (w, h),
                (bx, by),
                (dx, dy),
                (bw, bh),
                false,
            ));
        }
        let (x, y) = (x as usize, y as usize);
        Some(u64::from(P::sad(
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
    for &(x, y) in seeds {
        if (x, y) != (0, 0)
            && let Some(v) = cost(x, y)
            && v < best_sad
        {
            best_sad = v;
            best = (x, y);
        }
    }
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
    let (x, y) = (bx as isize + best.0, by as isize + best.1);
    let sse = if x < 0 || y < 0 || x as usize + bw > w || y as usize + bh > h {
        edge_err(target, pic, (w, h), (bx, by), best, (bw, bh), true)
    } else {
        let (x, y) = (x as usize, y as usize);
        P::sse(&target[by * w + bx..], w, &pic[y * w + x..], w, bw, bh)
    };
    (best, sse)
}

/// The SAD (or with `squared` the SSE) of the target's `bw`×`bh` block
/// at `(bx, by)` against `pic` moved by `(dx, dy)`, where that crosses the
/// picture's edge: the edge samples repeated.
fn edge_err<P: Pixel>(
    target: &[P],
    pic: &[P],
    (w, h): (usize, usize),
    (bx, by): (usize, usize),
    (dx, dy): (isize, isize),
    (bw, bh): (usize, usize),
    squared: bool,
) -> u64 {
    let mut total = 0u64;
    for r in 0..bh {
        let y = (by as isize + r as isize + dy).clamp(0, h as isize - 1) as usize;
        let row = &pic[y * w..(y + 1) * w];
        let t = &target[(by + r) * w + bx..(by + r) * w + bx + bw];
        for (c, a) in t.iter().enumerate() {
            let x = (bx as isize + c as isize + dx).clamp(0, w as isize - 1) as usize;
            let e = u64::from((a.get() - row[x].get()).unsigned_abs());
            total += if squared { e * e } else { e };
        }
    }
    total
}

/// A match of the filter's seeded diamond search leaving this much error
/// a sample (or more) is searched again on sums of 4×4 samples
/// ([`Small`]). From zero alone, the diamond's first steps of 16 settle
/// on fine detail far from its match: on the grainy pan (a fractal's
/// edge in fine colour, flat gradients elsewhere) its matches left 151 of
/// error a sample, and a block of that detail, weighted by its 3×3
/// errors, still blurs into the alt-ref; seeded and searched again so, 22.
const COARSE_PER: u64 = 64;

/// The coarse search's range, in sums of 4×4 samples (48 samples).
const COARSE_RANGE: isize = 12;

/// [`shift`]'s coarse range around its guess, in sums (12 samples).
const SHIFT_RANGE: isize = 3;

/// A picture's luma in sums of 4×4 samples, for the coarse search (at
/// most 16 × 4095, so in 16 bits at every depth).
struct Small {
    s: Vec<u16>,
    w: usize,
    h: usize,
}

impl Small {
    /// The sums of `y` (`w`×`h`, tight): each band of four rows summed
    /// down its columns, then across in fours (a loop the compiler
    /// vectorizes).
    fn new<P: Pixel>(y: &[P], w: usize, h: usize) -> Self {
        let (sw, sh) = (w / 4, h / 4);
        let mut s = vec![0u16; sw * sh];
        let n = sw * 4;
        let mut cols = vec![0u16; n];
        for r in 0..sh {
            let b = r * 4 * w;
            let (r0, r1, r2, r3) = (
                &y[b..b + n],
                &y[b + w..b + w + n],
                &y[b + 2 * w..b + 2 * w + n],
                &y[b + 3 * w..b + 3 * w + n],
            );
            for ((((c, a), b), d), e) in cols.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
                *c = (a.get() + b.get() + d.get() + e.get()) as u16;
            }
            for (v, q) in s[r * sw..(r + 1) * sw]
                .iter_mut()
                .zip(cols.as_chunks::<4>().0)
            {
                *v = q[0] + q[1] + q[2] + q[3];
            }
        }
        Self { s, w: sw, h: sh }
    }

    /// The vector, in whole samples of the picture, of the best match in
    /// `pic` within [`COARSE_RANGE`] of this one's `bw`×`bh` sums at
    /// `(x, y)` (ties to the shorter vector).
    fn find(&self, pic: &Small, x: usize, y: usize, bw: usize, bh: usize) -> (isize, isize) {
        self.find_near(pic, x, y, bw, bh, (0, 0), COARSE_RANGE)
    }

    /// [`Small::find`] within `range` of `near` (in sums). Each row of
    /// vectors is measured at once, sum by sum of the block across the
    /// row's offsets (in vector lanes), then taken in order.
    #[allow(clippy::too_many_arguments)]
    fn find_near(
        &self,
        pic: &Small,
        x: usize,
        y: usize,
        bw: usize,
        bh: usize,
        near: (isize, isize),
        range: isize,
    ) -> (isize, isize) {
        if bw == 0 || bh == 0 || x + bw > self.w || y + bh > self.h {
            return (0, 0);
        }
        // The offsets across that keep the block in `pic`.
        let lo = (near.0 - range).max(-(x as isize));
        let hi = (near.0 + range).min(pic.w as isize - bw as isize - x as isize);
        let mut best = (u64::MAX, (0isize, 0isize));
        if hi < lo {
            return best.1;
        }
        if bw == 4 && bh == 4 {
            // (A block of 4×4 sums, as `shift` measures, held in registers:
            // each vector's sum of differences at once.)
            let a: [[u16; 4]; 4] = std::array::from_fn(|r| {
                let at = (y + r) * self.w + x;
                [self.s[at], self.s[at + 1], self.s[at + 2], self.s[at + 3]]
            });
            for dy in near.1 - range..=near.1 + range {
                let py = y as isize + dy;
                if py < 0 || py as usize + 4 > pic.h {
                    continue;
                }
                let py = py as usize;
                for dx in lo..=hi {
                    let px = (x as isize + dx) as usize;
                    let mut t = 0u32;
                    for (r, a) in a.iter().enumerate() {
                        let row = &pic.s[(py + r) * pic.w + px..][..4];
                        for (&a, &b) in a.iter().zip(row) {
                            t += u32::from(a.abs_diff(b));
                        }
                    }
                    let sad = u64::from(t) * 64 + (dx.unsigned_abs() + dy.unsigned_abs()) as u64;
                    if sad < best.0 {
                        best = (sad, (dx * 4, dy * 4));
                    }
                }
            }
            return best.1;
        }
        let n = (hi - lo + 1) as usize;
        let mut acc = vec![0u32; n];
        for dy in near.1 - range..=near.1 + range {
            let py = y as isize + dy;
            if py < 0 || py as usize + bh > pic.h {
                continue;
            }
            let py = py as usize;
            acc.fill(0);
            let px0 = (x as isize + lo) as usize;
            for r in 0..bh {
                let a = &self.s[(y + r) * self.w + x..][..bw];
                let row = &pic.s[(py + r) * pic.w + px0..];
                for (c, &a) in a.iter().enumerate() {
                    for (t, &b) in acc.iter_mut().zip(&row[c..c + n]) {
                        *t += u32::from(a.abs_diff(b));
                    }
                }
            }
            for (i, &t) in acc.iter().enumerate() {
                let dx = lo + i as isize;
                let sad = u64::from(t) * 64 + (dx.unsigned_abs() + dy.unsigned_abs()) as u64;
                if sad < best.0 {
                    best = (sad, (dx * 4, dy * 4));
                }
            }
        }
        best.1
    }
}
