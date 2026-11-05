//! Palette mode decisions (screen content): the luma palette and the
//! joint U/V palette of a block, each with its size chosen by an estimate
//! of distortion plus colour and colour-map bits.

use crate::dec::coder::Coder;
use crate::dec::tile::{Block, TileDec};
use crate::dsp::Pixel;

use super::me::satd;

/// A U/V palette: size, U colours (ascending), V colours, SATD cost with
/// the signalling, and the signalling bits.
type UvPalette = (u8, [u16; 8], [u16; 8], f64, f64);

/// Weight of the U/V palette's signalling bits against prediction SATD.
const UV_BITS_WEIGHT: f64 = 5.0;

/// Estimated bits of a colour map (`w` x `h`, stride `stride`) with `n`
/// colours: the map is coded in wavefront order with the left and above
/// indices as context, so runs of the same index are cheap.
fn map_bits(map: &[u8], stride: usize, w: usize, h: usize, n: usize) -> f64 {
    let ln = (n as f64).log2();
    let mut b = ln;
    for i in 0..h {
        for j in 0..w {
            if i == 0 && j == 0 {
                continue;
            }
            let v = map[i * stride + j];
            let l = (j > 0).then(|| map[i * stride + j - 1]);
            let a = (i > 0).then(|| map[(i - 1) * stride + j]);
            b += if l.is_none_or(|l| l == v) && a.is_none_or(|a| a == v) {
                0.1
            } else if l == Some(v) || a == Some(v) {
                1.0
            } else {
                1.0 + ln
            };
        }
    }
    b
}

/// Weighted k-means of points (`dims` coordinates each, with a count) into
/// `k` centres, started from points spread along the first coordinate.
fn kmeans(pts: &[([i32; 2], u32)], dims: usize, k: usize) -> Vec<[i32; 2]> {
    let total: u64 = pts.iter().map(|p| u64::from(p.1)).sum();
    // Start from weighted quantiles of the (sorted) points.
    let mut c: Vec<[f64; 2]> = Vec::with_capacity(k);
    let mut acc = 0u64;
    let mut q = 0;
    for p in pts {
        acc += u64::from(p.1);
        while q < k && acc * (2 * k as u64) > (2 * q as u64 + 1) * total {
            c.push([f64::from(p.0[0]), f64::from(p.0[1])]);
            q += 1;
        }
    }
    while c.len() < k {
        let last = c.last().copied().unwrap_or([0.0, 0.0]);
        c.push(last);
    }
    let dist = |a: [i32; 2], b: [f64; 2]| -> f64 {
        let d0 = f64::from(a[0]) - b[0];
        let d1 = if dims > 1 {
            f64::from(a[1]) - b[1]
        } else {
            0.0
        };
        d0 * d0 + d1 * d1
    };
    for _ in 0..8 {
        let mut sum = vec![[0f64; 2]; k];
        let mut cnt = vec![0f64; k];
        for p in pts {
            let mut bi = 0;
            let mut bd = f64::MAX;
            for (i, ci) in c.iter().enumerate() {
                let d = dist(p.0, *ci);
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            let w = f64::from(p.1);
            sum[bi][0] += f64::from(p.0[0]) * w;
            sum[bi][1] += f64::from(p.0[1]) * w;
            cnt[bi] += w;
        }
        let mut moved = false;
        for i in 0..k {
            if cnt[i] > 0.0 {
                let n = [sum[i][0] / cnt[i], sum[i][1] / cnt[i]];
                moved |= n != c[i];
                c[i] = n;
            }
        }
        if !moved {
            break;
        }
    }
    c.iter()
        .map(|v| [v[0].round() as i32, v[1].round() as i32])
        .collect()
}

/// Distinct values (pairs) of a list with their counts, sorted.
fn histogram(vals: &mut [[i32; 2]]) -> Vec<([i32; 2], u32)> {
    vals.sort_unstable();
    let mut out: Vec<([i32; 2], u32)> = Vec::new();
    for &v in vals.iter() {
        match out.last_mut() {
            Some(l) if l.0 == v => l.1 += 1,
            _ => out.push((v, 1)),
        }
    }
    out
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// Best luma palette for the block at (x, y): (size, ascending colors,
    /// SATD cost including an estimate of the palette and map bits). The
    /// color map is left in `self.color_map_y` (stride = block width).
    pub(crate) fn pick_palette_y(
        &mut self,
        x: usize,
        y: usize,
        bw: usize,
        bh: usize,
    ) -> Option<(u8, [u16; 8], f64, f64)> {
        let src = self.src?;
        let sp = &src.planes[0];
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        let on_w = bw.min(fw - x);
        let on_h = bh.min(fh - y);
        let mut vals: Vec<[i32; 2]> = Vec::with_capacity(on_w * on_h);
        for i in 0..on_h {
            for j in 0..on_w {
                vals.push([sp.data[(y + i) * sp.stride + x + j].to_i32(), 0]);
            }
        }
        let hist = histogram(&mut vals);
        if hist.len() < 2 {
            return None;
        }
        let lam = self.mode.lambda_satd;
        let mut map = vec![0u8; bw * bh];
        let mut pred = vec![P::default(); bw * bh];
        let mut best: Option<(u8, [u16; 8], f64, f64)> = None;
        let mut best_map = vec![0u8; bw * bh];
        let mut worse = 0;
        for k in 2..=hist.len().min(8) {
            // The cost is about convex in the size: stop once it rose twice.
            if worse >= 2 {
                break;
            }
            let mut colors: Vec<u16> = if hist.len() == k {
                hist.iter().map(|h| h.0[0] as u16).collect()
            } else {
                kmeans(&hist, 1, k).iter().map(|c| c[0] as u16).collect()
            };
            colors.sort_unstable();
            colors.dedup();
            let n = colors.len();
            if n < 2 || best.is_some_and(|b| usize::from(b.0) == n) {
                continue;
            }
            for i in 0..bh {
                for j in 0..bw {
                    let v =
                        sp.data[(y + i.min(on_h - 1)) * sp.stride + x + j.min(on_w - 1)].to_i32();
                    let mut bi = 0;
                    for (c, &cv) in colors.iter().enumerate().skip(1) {
                        if (v - i32::from(cv)).abs() < (v - i32::from(colors[bi])).abs() {
                            bi = c;
                        }
                    }
                    map[i * bw + j] = bi as u8;
                    pred[i * bw + j] = P::from_i32(i32::from(colors[bi]));
                }
            }
            let d = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &pred,
                bw,
                on_w & !3,
                on_h & !3,
            ));
            let bits =
                4.0 + n as f64 * f64::from(self.bd) * 0.8 + map_bits(&map, bw, on_w, on_h, n);
            let cost = d + lam * bits;
            if best.is_some_and(|b| cost >= b.2) {
                worse += 1;
            }
            if best.is_none_or(|b| cost < b.2) {
                let mut out = [0u16; 8];
                out[..n].copy_from_slice(&colors);
                best = Some((n as u8, out, cost, bits));
                best_map.copy_from_slice(&map);
            }
        }
        self.color_map_y[..bw * bh].copy_from_slice(&best_map);
        best
    }

    /// Joint U/V palette of an intra block: (size, U colors ascending, V
    /// colors in the same order, SATD cost with an estimate of the colour
    /// and map bits). The color map is left in `self.color_map_uv`.
    pub(crate) fn pick_palette_uv(&mut self, pb: &Block) -> Option<(u8, [u16; 8], [u16; 8], f64)> {
        let bw0 = crate::dec::types::block_width(pb.bsize);
        let bh0 = crate::dec::types::block_height(pb.bsize);
        self.pick_palette_uv_at(pb.mi_col * 4, pb.mi_row * 4, bw0, bh0)
            .map(|r| (r.0, r.1, r.2, r.3))
    }

    /// [`Self::pick_palette_uv`] for the luma area `bw0` x `bh0` at
    /// (`x0`, `y0`); also returns the signalling bits.
    fn pick_palette_uv_at(
        &mut self,
        x0: usize,
        y0: usize,
        bw0: usize,
        bh0: usize,
    ) -> Option<UvPalette> {
        let src = self.src?;
        if self.num_planes < 3 {
            return None;
        }
        let (sx, sy) = (usize::from(self.sub_x), usize::from(self.sub_y));
        let x = x0 >> sx;
        let y = y0 >> sy;
        let on_w0 = bw0.min(self.fs.mi_cols * 4 - x0);
        let on_h0 = bh0.min(self.fs.mi_rows * 4 - y0);
        let (mut bw, mut bh) = (bw0 >> sx, bh0 >> sy);
        let (mut on_w, mut on_h) = (on_w0 >> sx, on_h0 >> sy);
        if bw < 4 {
            bw += 2;
            on_w += 2;
        }
        if bh < 4 {
            bh += 2;
            on_h += 2;
        }
        if on_w < 4 || on_h < 4 {
            return None;
        }
        let (up, vp) = (&src.planes[1], &src.planes[2]);
        let at = |p: &crate::dec::frame::Plane<P>, i: usize, j: usize| -> i32 {
            p.data[(y + i.min(on_h - 1)) * p.stride + x + j.min(on_w - 1)].to_i32()
        };
        let mut vals: Vec<[i32; 2]> = Vec::with_capacity(on_w * on_h);
        for i in 0..on_h {
            for j in 0..on_w {
                vals.push([at(up, i, j), at(vp, i, j)]);
            }
        }
        let hist = histogram(&mut vals);
        if hist.len() < 2 {
            return None;
        }
        let lam = self.mode.lambda_satd;
        let mut map = vec![0u8; bw * bh];
        let mut pu = vec![P::default(); bw * bh];
        let mut pv = vec![P::default(); bw * bh];
        let mut best: Option<UvPalette> = None;
        let mut best_map = vec![0u8; bw * bh];
        let maxv = (1i32 << self.bd) - 1;
        let mut worse = 0;
        for k in 2..=hist.len().min(8) {
            if worse >= 2 {
                break;
            }
            let mut colors: Vec<[i32; 2]> = if hist.len() == k {
                hist.iter().map(|h| h.0).collect()
            } else {
                kmeans(&hist, 2, k)
            };
            for c in colors.iter_mut() {
                c[0] = c[0].clamp(0, maxv);
                c[1] = c[1].clamp(0, maxv);
            }
            colors.sort_unstable();
            colors.dedup();
            let n = colors.len();
            if n < 2 || best.is_some_and(|b| usize::from(b.0) == n) {
                continue;
            }
            for i in 0..bh {
                for j in 0..bw {
                    let (u, v) = (at(up, i, j), at(vp, i, j));
                    let mut bi = 0;
                    let mut bd = i32::MAX;
                    for (c, cv) in colors.iter().enumerate() {
                        let d = (u - cv[0]).pow(2) + (v - cv[1]).pow(2);
                        if d < bd {
                            bd = d;
                            bi = c;
                        }
                    }
                    map[i * bw + j] = bi as u8;
                    pu[i * bw + j] = P::from_i32(colors[bi][0]);
                    pv[i * bw + j] = P::from_i32(colors[bi][1]);
                }
            }
            let (cw, ch) = (on_w & !3, on_h & !3);
            let d = f64::from(satd(
                &up.data[y * up.stride + x..],
                up.stride,
                &pu,
                bw,
                cw,
                ch,
            )) + f64::from(satd(
                &vp.data[y * vp.stride + x..],
                vp.stride,
                &pv,
                bw,
                cw,
                ch,
            ));
            let bits =
                4.0 + n as f64 * f64::from(self.bd) * 1.8 + map_bits(&map, bw, on_w, on_h, n);
            // The SATD comparison with the regular chroma prediction favours
            // the palette; its colours and map pay off less than they seem
            // once the residual is coded (measured on screen content).
            let cost = d + lam * bits * UV_BITS_WEIGHT;
            if best.is_some_and(|b| cost >= b.3) {
                worse += 1;
            }
            if best.is_none_or(|b| cost < b.3) {
                let mut u = [0u16; 8];
                let mut v = [0u16; 8];
                for (i, c) in colors.iter().enumerate() {
                    u[i] = c[0] as u16;
                    v[i] = c[1] as u16;
                }
                best = Some((n as u8, u, v, cost, bits));
                best_map.copy_from_slice(&map);
            }
        }
        if self.color_map_uv.len() >= bw * bh {
            self.color_map_uv[..bw * bh].copy_from_slice(&best_map);
        }
        best
    }

    /// Partition estimate of the chroma of a screen content intra area
    /// (`pw` x `ph` luma samples at (`x`, `y`)): the better of a DC
    /// prediction from the source neighbours and a U/V palette, each with its
    /// residual transform-coded, in the units of `tx_cost`.
    pub(crate) fn est_chroma_sc(&mut self, x: usize, y: usize, pw: usize, ph: usize) -> f64 {
        let Some(src) = self.src else { return 0.0 };
        if self.num_planes < 3 {
            return 0.0;
        }
        let (sx, sy) = (usize::from(self.sub_x), usize::from(self.sub_y));
        let (cx, cy, cw, ch) = (x >> sx, y >> sy, (pw >> sx).max(4), (ph >> sy).max(4));
        let (tw, th) = (cw.min(32), ch.min(32));
        let Some(tx) = (0..19u8)
            .find(|&t| crate::dec::types::tx_w(t) == tw && crate::dec::types::tx_h(t) == th)
        else {
            return 0.0;
        };
        let lam_b = self.enc.lambda;
        let mut res = vec![0i32; cw * ch];
        // DC from the source neighbours.
        let mut dc_cost = 0.0;
        for plane in 1..3 {
            let p = &src.planes[plane];
            if cx + cw > p.w || cy + ch > p.h {
                return 0.0;
            }
            let (mut sum, mut n) = (0i64, 0i64);
            if cy > 0 {
                for j in 0..cw {
                    sum += i64::from(p.data[(cy - 1) * p.stride + cx + j].to_i32());
                }
                n += cw as i64;
            }
            if cx > 0 {
                for i in 0..ch {
                    sum += i64::from(p.data[(cy + i) * p.stride + cx - 1].to_i32());
                }
                n += ch as i64;
            }
            let dc = if n > 0 {
                ((sum + n / 2) / n) as i32
            } else {
                1 << (self.bd - 1)
            };
            for i in 0..ch {
                for j in 0..cw {
                    res[i * cw + j] = p.data[(cy + i) * p.stride + cx + j].to_i32() - dc;
                }
            }
            dc_cost += self.tx_cost_any(&res, cw, ch, tx);
        }
        let mut best = dc_cost + lam_b * 2.0;
        if !self.hdr.allow_screen_content_tools {
            return best;
        }
        if let Some((n, u, v, _, bits)) = self.pick_palette_uv_at(x, y, pw, ph) {
            let mut pal_cost = lam_b * bits;
            for (plane, colors) in [(1usize, u), (2, v)] {
                let p = &src.planes[plane];
                for i in 0..ch {
                    for j in 0..cw {
                        let k = usize::from(self.color_map_uv[i * cw + j]).min(usize::from(n) - 1);
                        res[i * cw + j] =
                            p.data[(cy + i) * p.stride + cx + j].to_i32() - i32::from(colors[k]);
                    }
                }
                pal_cost += self.tx_cost_any(&res, cw, ch, tx);
            }
            best = best.min(pal_cost);
        }
        best
    }
}
