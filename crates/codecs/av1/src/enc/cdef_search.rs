//! CDEF strengths per 64x64 block.
//!
//! Every candidate strength filters the whole deblocked frame once; the
//! squared error of each 64x64 block (all planes) under each candidate, and
//! unfiltered, then picks a set of up to eight strengths and each block's
//! index into it, trading the error against the `cdef_idx` bits.

use crate::dec::frame::{FrameState, Picture};
use crate::dec::obu::{FrameHeader, SequenceHeader};
use crate::dec::postfilter::cdef;
use crate::dsp::Pixel;
use crate::dsp::cdef::CDEF_PAD_STRIDE;

/// A luma (primary, secondary) strength; chroma follows it.
pub type Strength = (u8, u8);

/// The chosen strengths and the index of each 64x64 block (in
/// `FrameState::cdef_idx` layout; -1 where nothing is coded).
pub struct CdefChoice {
    pub bits: u8,
    pub strengths: Vec<Strength>,
    pub idx: Vec<i8>,
}

/// Writes strength `k` of `s` into the header's tables.
pub fn set_strength(h: &mut FrameHeader, k: usize, s: Strength) {
    let (pri, sec) = s;
    h.cdef.y_pri[k] = pri;
    h.cdef.y_sec[k] = sec;
    h.cdef.uv_pri[k] = pri / 2;
    h.cdef.uv_sec[k] = sec.min(2) / 2 * 2;
}

/// Squared error of each 64x64 block of `a` against `b`, all planes.
fn block_sse<P: Pixel>(a: &Picture<P>, b: &Picture<P>, stride: usize, out: &mut [u64]) {
    out.fill(0);
    for (p, (pa, pb)) in a.planes.iter().zip(&b.planes).enumerate() {
        let (sx, sy) = if p > 0 {
            (usize::from(a.sub_x), usize::from(a.sub_y))
        } else {
            (0, 0)
        };
        for y in 0..pa.h {
            let ra = &pa.data[y * pa.stride..][..pa.w];
            let rb = &pb.data[y * pb.stride..][..pa.w];
            let row = ((y << sy) >> 6) * stride;
            let fw = 64 >> sx;
            for (c, (ca, cb)) in ra.chunks(fw).zip(rb.chunks(fw)).enumerate() {
                let mut e = 0u64;
                for (x, z) in ca.iter().zip(cb) {
                    let d = i64::from(x.to_i32() - z.to_i32());
                    e += (d * d) as u64;
                }
                if let Some(o) = out.get_mut(row + c) {
                    *o += e;
                }
            }
        }
    }
}

/// Searches `cands` (index 0 of the result's errors is "off").
/// `lambda` converts one bit into squared error.
#[allow(clippy::too_many_arguments)]
pub fn search<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    deblocked: &Picture<P>,
    src: &Picture<P>,
    cands: &[Strength],
    lambda: f64,
    max_bits: u8,
    threads: usize,
) -> CdefChoice {
    let st = fs.cdef_stride;
    let n = fs.cdef_idx.len();
    let coded: Vec<usize> = (0..n).filter(|&i| fs.cdef_idx[i] >= 0).collect();
    // err[k][fb]: k = 0 unfiltered, k = 1 + candidate.
    let mut err = vec![vec![0u64; n]; cands.len() + 1];
    block_sse(deblocked, src, st, &mut err[0]);
    let mut h2 = hdr.clone();
    h2.cdef.bits = 0;
    let mut out = deblocked.clone();
    for (k, &c) in cands.iter().enumerate() {
        set_strength(&mut h2, 0, c);
        // A plane whose strengths are zero is left as it is: start clean.
        out.clone_from(deblocked);
        cdef(seq, &h2, fs, deblocked, &mut out, threads);
        block_sse(&out, src, st, &mut err[k + 1]);
    }
    let nk = cands.len() + 1;
    let total = |set: &[usize]| -> u64 {
        coded
            .iter()
            .map(|&i| set.iter().map(|&k| err[k][i]).min().unwrap_or(0))
            .sum()
    };
    let mut best: Option<(f64, Vec<usize>, u8)> = None;
    for bits in 0..=max_bits {
        let want = 1usize << bits;
        if want > nk {
            break;
        }
        // Greedy growth, then one round of swaps.
        let mut set: Vec<usize> = Vec::with_capacity(want);
        while set.len() < want {
            let pick = (0..nk).filter(|k| !set.contains(k)).min_by_key(|&k| {
                let mut s = set.clone();
                s.push(k);
                total(&s)
            });
            match pick {
                Some(k) => set.push(k),
                None => break,
            }
        }
        for i in 0..set.len() {
            let mut cur = total(&set);
            for k in 0..nk {
                if set.contains(&k) {
                    continue;
                }
                let old = set[i];
                set[i] = k;
                let t = total(&set);
                if t < cur {
                    cur = t;
                } else {
                    set[i] = old;
                }
            }
        }
        let cost = total(&set) as f64 + lambda * f64::from(bits) * coded.len() as f64;
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, set, bits));
        }
    }
    let (_, set, bits) = best.unwrap_or((0.0, vec![0], 0));
    let strengths: Vec<Strength> = set
        .iter()
        .map(|&k| if k == 0 { (0, 0) } else { cands[k - 1] })
        .collect();
    let mut idx = fs.cdef_idx.clone();
    for &i in &coded {
        let j = (0..set.len()).min_by_key(|&j| err[set[j]][i]).unwrap_or(0);
        idx[i] = j as i8;
    }
    CdefChoice {
        bits,
        strengths,
        idx,
    }
}

/// How much one frame-wide CDEF strength changes the squared error, for
/// each of `luma` (luma primary, secondary) and each of `chroma` (chroma
/// primary, secondary), measured in the 64-row areas `rows`, as
/// [`crate::dec::postfilter::cdef`] would filter `deblocked` with every
/// coded 64x64 block at that strength. Each 8x8 block's direction is found
/// once for all candidates, and only the blocks a candidate changes are
/// measured. The sums do not depend on `threads`.
#[allow(clippy::too_many_arguments)]
pub fn strength_deltas<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    deblocked: &Picture<P>,
    src: &Picture<P>,
    luma: &[Strength],
    chroma: &[Strength],
    rows: &[usize],
    threads: usize,
) -> (Vec<i64>, Vec<i64>) {
    let chroma = if seq.num_planes() > 1 { chroma } else { &[] };
    let job = |i: usize, acc: &mut (Vec<i64>, Vec<i64>), pads: &mut [Vec<u16>; 3]| {
        let rw = RowWork {
            seq,
            hdr,
            fs,
            deblocked,
            src,
        };
        rw.deltas(luma, chroma, i, acc, pads);
    };
    let t = threads.clamp(1, rows.len().max(1));
    let zero = || (vec![0i64; luma.len()], vec![0i64; chroma.len()]);
    let parts: Vec<(Vec<i64>, Vec<i64>)> = if t <= 1 {
        let mut acc = zero();
        let mut pads: [Vec<u16>; 3] =
            core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]);
        for &i in rows {
            job(i, &mut acc, &mut pads);
        }
        vec![acc]
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..t)
                .map(|k| {
                    let (job, rows, zero) = (&job, &rows, &zero);
                    sc.spawn(move || {
                        let mut acc = zero();
                        let mut pads: [Vec<u16>; 3] =
                            core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]);
                        for &i in rows.iter().skip(k).step_by(t) {
                            job(i, &mut acc, &mut pads);
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|_| zero()))
                .collect()
        })
    };
    let mut out = zero();
    for (l, c) in parts {
        out.0.iter_mut().zip(l).for_each(|(a, b)| *a += b);
        out.1.iter_mut().zip(c).for_each(|(a, b)| *a += b);
    }
    out
}

/// What [`strength_deltas`] measures one 64-row area of.
struct RowWork<'a, P> {
    seq: &'a SequenceHeader,
    hdr: &'a FrameHeader,
    fs: &'a FrameState,
    deblocked: &'a Picture<P>,
    src: &'a Picture<P>,
}

impl<P: Pixel> RowWork<'_, P> {
    /// Adds the error changes of the 64-row area `sbr16` to `acc`.
    fn deltas(
        &self,
        luma: &[Strength],
        chroma: &[Strength],
        sbr16: usize,
        acc: &mut (Vec<i64>, Vec<i64>),
        pads: &mut [Vec<u16>; 3],
    ) {
        use crate::dsp::cdef::{
            CDEF_BORDER, adjust_luma_pri_strength, cdef_damping, cdef_direction, cdef_filter,
            cdef_pad_block, cdef_uv_dir,
        };
        if luma.is_empty() && chroma.is_empty() {
            return;
        }
        let (fs, deblocked, src) = (self.fs, self.deblocked, self.src);
        let ps = 64 + 2 * CDEF_BORDER;
        let bd = deblocked.bd;
        let shift = bd - 8;
        let (ssx, ssy) = (self.seq.subsampling_x, self.seq.subsampling_y);
        let np = if chroma.is_empty() { 1 } else { 3 };
        let damping = i32::from(self.hdr.cdef.damping);
        let y_damping = cdef_damping(damping, bd, false);
        let uv_damping = cdef_damping(damping, bd, true);
        let lp = &deblocked.planes[0];
        let sbr = sbr16 * 16;
        let mut buf = [P::from_i32(0); 64];
        for sbc in (0..fs.mi_cols).step_by(16) {
            if fs.cdef_idx[sbr16 * fs.cdef_stride + (sbc >> 4)] < 0 {
                continue;
            }
            let mut padded = [false; 3];
            let r_end = (sbr + 16).min(fs.mi_rows);
            let c_end = (sbc + 16).min(fs.mi_cols);
            for r in (sbr..r_end).step_by(2) {
                for c in (sbc..c_end).step_by(2) {
                    let rr = (r + 1).min(fs.mi_rows - 1);
                    let cc = (c + 1).min(fs.mi_cols - 1);
                    if fs.mi_at(r, c).skip
                        && fs.mi_at(rr, c).skip
                        && fs.mi_at(r, cc).skip
                        && fs.mi_at(rr, cc).skip
                    {
                        continue;
                    }
                    let (ydir, var) =
                        cdef_direction(&lp.data[r * 4 * lp.stride + c * 4..], lp.stride, bd);
                    for plane in 0..np {
                        let (cands, out) = if plane == 0 {
                            (luma, &mut acc.0)
                        } else {
                            (chroma, &mut acc.1)
                        };
                        if cands.is_empty() {
                            continue;
                        }
                        let (sx, sy) = if plane > 0 {
                            (usize::from(ssx), usize::from(ssy))
                        } else {
                            (0, 0)
                        };
                        let sp = &deblocked.planes[plane];
                        let sb_x = (sbc * 4) >> sx;
                        let sb_y = (sbr * 4) >> sy;
                        if !padded[plane] {
                            cdef_pad_block(
                                &mut pads[plane],
                                ps,
                                &sp.data,
                                sp.stride,
                                sb_x,
                                sb_y,
                                64 >> sx,
                                64 >> sy,
                                (fs.mi_cols * 4) >> sx,
                                (fs.mi_rows * 4) >> sy,
                            );
                            padded[plane] = true;
                        }
                        let (w, h) = (8 >> sx, 8 >> sy);
                        let (x0, y0) = ((c * 4) >> sx, (r * 4) >> sy);
                        // The visible part of the block.
                        let vw = sp.w.saturating_sub(x0).min(w);
                        let vh = sp.h.saturating_sub(y0).min(h);
                        if vw == 0 || vh == 0 {
                            continue;
                        }
                        let org = &src.planes[plane];
                        let sse = |a: &[P], a_stride: usize| -> i64 {
                            let mut e = 0i64;
                            for y in 0..vh {
                                let s = &org.data[(y0 + y) * org.stride + x0..][..vw];
                                for (p, q) in a[y * a_stride..][..vw].iter().zip(s) {
                                    let d = i64::from(p.to_i32() - q.to_i32());
                                    e += d * d;
                                }
                            }
                            e
                        };
                        let base = sse(&sp.data[y0 * sp.stride + x0..], sp.stride);
                        let pad = &pads[plane][(y0 - sb_y) * ps + (x0 - sb_x)..];
                        for (k, &(pri, sec)) in cands.iter().enumerate() {
                            let pri0 = i32::from(pri) << shift;
                            let sec = i32::from(sec) << shift;
                            let (p, dir, damping) = if plane == 0 {
                                let dir = if pri0 == 0 { 0 } else { ydir };
                                (adjust_luma_pri_strength(pri0, var), dir, y_damping)
                            } else {
                                let dir = if pri0 == 0 {
                                    0
                                } else {
                                    cdef_uv_dir(ssx, ssy, ydir)
                                };
                                (pri0, dir, uv_damping)
                            };
                            if p == 0 && sec == 0 {
                                continue;
                            }
                            cdef_filter(&mut buf, 8, pad, ps, w, h, p, sec, dir, damping, bd);
                            out[k] += sse(&buf, 8) - base;
                        }
                    }
                }
            }
        }
    }
}
