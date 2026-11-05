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
