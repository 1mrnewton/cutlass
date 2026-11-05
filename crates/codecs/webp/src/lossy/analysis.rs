//! Segments: how much each macroblock suffers from quantization, as
//! libwebp measures it, grouped into up to four segments by k-means.

use super::forward::fdct;
use super::frame::{CHROMA_PITCH, LUMA_PITCH, load_edges};
use super::predict::{self, B_DC, B_TM, Edges};
use super::source::Source;

/// Largest susceptibility value.
const MAX_ALPHA: i32 = 255;
/// Histogram bins of `|coefficient| >> 3`.
const MAX_COEFF_THRESH: usize = 31;
const MAX_ITERS_K_MEANS: usize = 6;

/// One segment's place among the others.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SegmentShape {
    /// -127 to 127: how much more than average it may be quantized.
    pub alpha: i32,
    /// 0 to 255: how busy it is, which lowers its filtering.
    pub beta: i32,
}

pub(crate) struct Analysis {
    /// Segment of each macroblock, raster order.
    pub segments: Vec<u8>,
    pub shapes: Vec<SegmentShape>,
    /// Average chroma susceptibility, around 60.
    pub uv_alpha: i32,
}

/// `alpha` of libwebp's `GetAlpha`: the spread of the coefficient
/// histogram against its peak.
fn alpha_of(histogram: &[u32; MAX_COEFF_THRESH + 1]) -> i32 {
    let max = histogram.iter().copied().max().unwrap_or(0);
    let last = histogram.iter().rposition(|&count| count > 0).unwrap_or(0);
    if max > 1 {
        (2 * MAX_ALPHA * last as i32) / max as i32
    } else {
        0
    }
}

fn add_block(histogram: &mut [u32; MAX_COEFF_THRESH + 1], coeffs: &[i16; 16]) {
    for &c in coeffs {
        let bin = (usize::from(c.unsigned_abs()) >> 3).min(MAX_COEFF_THRESH);
        histogram[bin] += 1;
    }
}

/// The best of DC and TM prediction for the luma of one macroblock.
fn luma_alpha(source: &Source, mb_x: usize, mb_y: usize) -> i32 {
    let mut edges = [0u8; LUMA_PITCH * 17];
    load_edges(
        &mut edges,
        LUMA_PITCH,
        16,
        &source.y,
        source.y_stride,
        mb_x,
        mb_y,
    );
    let mut best = -1;
    for mode in [B_DC, B_TM] {
        let mut work = edges;
        predict::block(
            &mut work,
            LUMA_PITCH + 1,
            LUMA_PITCH,
            16,
            mode,
            Edges {
                above: mb_y > 0,
                left: mb_x > 0,
            },
        );
        let mut histogram = [0u32; MAX_COEFF_THRESH + 1];
        for block in 0..16 {
            let (row, col) = (block / 4, block % 4);
            let src_at = (mb_y * 16 + row * 4) * source.y_stride + mb_x * 16 + col * 4;
            let at = (1 + row * 4) * LUMA_PITCH + 1 + col * 4;
            add_block(
                &mut histogram,
                &fdct(&source.y, src_at, source.y_stride, &work, at, LUMA_PITCH),
            );
        }
        best = best.max(alpha_of(&histogram));
    }
    best
}

fn chroma_alpha(source: &Source, mb_x: usize, mb_y: usize) -> i32 {
    let mut best = -1;
    let edges_of = |plane: &[u8]| {
        let mut edges = [0u8; CHROMA_PITCH * 9];
        load_edges(
            &mut edges,
            CHROMA_PITCH,
            8,
            plane,
            source.uv_stride,
            mb_x,
            mb_y,
        );
        edges
    };
    let planes = [
        (&source.u, edges_of(&source.u)),
        (&source.v, edges_of(&source.v)),
    ];
    for mode in [B_DC, B_TM] {
        let mut histogram = [0u32; MAX_COEFF_THRESH + 1];
        for (plane, edges) in &planes {
            let mut work = *edges;
            predict::block(
                &mut work,
                CHROMA_PITCH + 1,
                CHROMA_PITCH,
                8,
                mode,
                Edges {
                    above: mb_y > 0,
                    left: mb_x > 0,
                },
            );
            for block in 0..4 {
                let (row, col) = (block / 2, block % 2);
                let src_at = (mb_y * 8 + row * 4) * source.uv_stride + mb_x * 8 + col * 4;
                let at = (1 + row * 4) * CHROMA_PITCH + 1 + col * 4;
                add_block(
                    &mut histogram,
                    &fdct(plane, src_at, source.uv_stride, &work, at, CHROMA_PITCH),
                );
            }
        }
        best = best.max(alpha_of(&histogram));
    }
    best
}

/// Measure every macroblock and group them into `count` segments.
pub(crate) fn analyze(source: &Source, count: usize) -> Analysis {
    let (mb_w, mb_h) = (source.mb_w, source.mb_h);
    let mut alphas = vec![0u8; mb_w * mb_h];
    let mut histogram = [0u32; MAX_ALPHA as usize + 1];
    let mut uv_sum = 0i64;
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let luma = luma_alpha(source, mb_x, mb_y);
            let chroma = chroma_alpha(source, mb_x, mb_y);
            uv_sum += i64::from(chroma);
            let mixed = (3 * luma + chroma + 2) >> 2;
            let alpha = (MAX_ALPHA - mixed).clamp(0, MAX_ALPHA);
            alphas[mb_y * mb_w + mb_x] = alpha as u8;
            histogram[alpha as usize] += 1;
        }
    }
    let uv_alpha = (uv_sum / (mb_w * mb_h) as i64) as i32;
    let (map, centers, mid) = k_means(&histogram, count);
    let segments = alphas.iter().map(|&a| map[usize::from(a)]).collect();
    let (min, max) = centers
        .iter()
        .fold((i32::MAX, i32::MIN), |(lo, hi), &c| (lo.min(c), hi.max(c)));
    let max = if max == min { min + 1 } else { max };
    let shapes = centers
        .iter()
        .map(|&center| SegmentShape {
            alpha: (255 * (center - mid) / (max - min)).clamp(-127, 127),
            beta: (255 * (center - min) / (max - min)).clamp(0, 255),
        })
        .collect();
    Analysis {
        segments,
        shapes,
        uv_alpha,
    }
}

/// libwebp's `AssignSegments`: the segment of each alpha value, the
/// centers, and their weighted average.
fn k_means(histogram: &[u32; 256], count: usize) -> ([u8; 256], Vec<i32>, i32) {
    let min_a = histogram.iter().position(|&n| n > 0).unwrap_or(0) as i32;
    let max_a = histogram.iter().rposition(|&n| n > 0).unwrap_or(0) as i32;
    let range = max_a - min_a;
    let mut centers: Vec<i32> = (0..count)
        .map(|k| min_a + ((2 * k as i32 + 1) * range) / (2 * count as i32))
        .collect();
    let mut map = [0u8; 256];
    let mut weighted = 0;
    for _ in 0..MAX_ITERS_K_MEANS {
        let mut accum = vec![0i64; count];
        let mut dist = vec![0i64; count];
        let mut n = 0;
        for a in min_a..=max_a {
            let weight = histogram[a as usize];
            if weight > 0 {
                while n + 1 < count && (a - centers[n + 1]).abs() < (a - centers[n]).abs() {
                    n += 1;
                }
                map[a as usize] = n as u8;
                dist[n] += i64::from(a) * i64::from(weight);
                accum[n] += i64::from(weight);
            }
        }
        let mut displaced = 0;
        let mut total = 0i64;
        let mut sum = 0i64;
        for k in 0..count {
            if accum[k] > 0 {
                let center = ((dist[k] + accum[k] / 2) / accum[k]) as i32;
                displaced += (centers[k] - center).abs();
                centers[k] = center;
                sum += i64::from(center) * accum[k];
                total += accum[k];
            }
        }
        weighted = ((sum + total / 2) / total.max(1)) as i32;
        if displaced < 5 {
            break;
        }
    }
    (map, centers, weighted)
}

#[cfg(test)]
mod tests {
    use super::k_means;

    #[test]
    fn two_clouds_split_into_their_own_segments() {
        let mut histogram = [0u32; 256];
        histogram[20] = 50;
        histogram[22] = 50;
        histogram[200] = 30;
        histogram[205] = 30;
        let (map, centers, mid) = k_means(&histogram, 2);
        assert_eq!((map[20], map[22], map[200], map[205]), (0, 0, 1, 1));
        assert_eq!(centers, vec![21, 203]);
        assert!(mid > 21 && mid < 203);
    }
}
