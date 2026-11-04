//! Super-resolution upscaling (spec 7.16) for one plane.

use super::Pixel;
use crate::spec::tables::{
    FILTER_BITS, MI_SIZE, SUPERRES_EXTRA_BITS, SUPERRES_FILTER_OFFSET, SUPERRES_FILTER_TAPS,
    SUPERRES_SCALE_BITS, UPSCALE_FILTER,
};

/// Per-plane stepping of 7.16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperresParams {
    /// `downscaledPlaneW`.
    pub src_w: usize,
    /// `upscaledPlaneW`.
    pub dst_w: usize,
    /// `stepX`.
    pub step: i64,
    /// `initialSubpelX` (after masking).
    pub initial_subpel: i64,
    /// `maxX`: the last source column read, `(MiCols >> subX) * MI_SIZE - 1`.
    pub max_x: i64,
}

/// Derives the 7.16 parameters from the luma `FrameWidth` (downscaled) and
/// `UpscaledWidth`, and the plane's horizontal subsampling.
pub fn superres_params(frame_width: usize, upscaled_width: usize, sub_x: bool) -> SuperresParams {
    let sx = sub_x as u32;
    let src_w = ((frame_width + sx as usize) >> sx) as i64;
    let dst_w = ((upscaled_width + sx as usize) >> sx) as i64;
    let scale_bits = SUPERRES_SCALE_BITS as u32;
    let den = dst_w.max(1);
    let step = ((src_w << scale_bits) + dst_w / 2) / den;
    let err = dst_w * step - (src_w << scale_bits);
    let mut initial = (-((dst_w - src_w) << (scale_bits - 1)) + dst_w / 2) / den
        + (1 << (SUPERRES_EXTRA_BITS - 1))
        - err / 2;
    initial &= (1 << scale_bits) - 1;
    let mi_cols = 2 * ((frame_width as i64 + 7) >> 3);
    let max_x = (mi_cols >> sx) * MI_SIZE as i64 - 1;
    SuperresParams {
        src_w: src_w as usize,
        dst_w: dst_w as usize,
        step,
        initial_subpel: initial,
        max_x,
    }
}

/// Upscales `h` rows of one plane (7.16). `frame_width` / `upscaled_width`
/// are the luma `FrameWidth` (downscaled) and `UpscaledWidth`; `sub_x` is the
/// plane's horizontal subsampling. Writes `Round2(upscaled_width, sub_x)`
/// samples per row into `dst`.
///
/// Per the spec, source columns are clamped to `0..=(MiCols >> subX) * 4 - 1`
/// (MiCols of the downscaled frame), so `src` must hold decoded samples up to
/// that column, which can lie past `Round2(frame_width, sub_x) - 1`.
pub fn upscale_plane<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_stride: usize,
    frame_width: usize,
    upscaled_width: usize,
    h: usize,
    sub_x: bool,
    bd: u32,
) {
    let sp = superres_params(frame_width, upscaled_width, sub_x);
    if sp.max_x < 0 {
        return;
    }
    let scale_bits = SUPERRES_SCALE_BITS as u32;
    let mask = (1i64 << scale_bits) - 1;
    let extra = SUPERRES_EXTRA_BITS as u32;
    let taps = SUPERRES_FILTER_TAPS as usize;
    let off = SUPERRES_FILTER_OFFSET as i64;
    let fb = FILTER_BITS as u32;
    let maxv = (1i32 << bd) - 1;
    for y in 0..h {
        let row = &src[y * src_stride..y * src_stride + sp.max_x as usize + 1];
        let out = &mut dst[y * dst_stride..y * dst_stride + sp.dst_w];
        let mut src_x = -(1i64 << scale_bits) + sp.initial_subpel;
        for o in out.iter_mut() {
            let first = (src_x >> scale_bits) - off;
            let filter = &UPSCALE_FILTER[((src_x & mask) >> extra) as usize];
            let mut sum = 0i32;
            if first >= 0 && first + taps as i64 - 1 <= sp.max_x {
                let s = &row[first as usize..first as usize + taps];
                for k in 0..taps {
                    sum += s[k].to_i32() * filter[k] as i32;
                }
            } else {
                for k in 0..taps {
                    let sx = (first + k as i64).clamp(0, sp.max_x) as usize;
                    sum += row[sx].to_i32() * filter[k] as i32;
                }
            }
            *o = P::from_i32(((sum + (1 << (fb - 1))) >> fb).clamp(0, maxv));
            src_x += sp.step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::upscale_plane;
    use crate::dsp::Pixel;
    use crate::spec::tables::UPSCALE_FILTER;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
        fn range(&mut self, n: u32) -> i64 {
            (self.next() % n) as i64
        }
    }

    fn r2(x: i64, n: i64) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    /// 7.16 transliterated for one plane.
    fn ref_upscale(
        frame: &[i64],
        stride: usize,
        frame_width: i64,
        upscaled_width: i64,
        mi_cols: i64,
        plane_h: usize,
        sub_x: i64,
        bd: u32,
    ) -> Vec<i64> {
        let down = r2(frame_width, sub_x);
        let up = r2(upscaled_width, sub_x);
        let step = ((down << 14) + (up / 2)) / up;
        let err = (up * step) - (down << 14);
        let mut init = (-((up - down) << 13) + up / 2) / up + (1 << 7) - err / 2;
        init &= (1 << 14) - 1;
        let mi_w = mi_cols >> sub_x;
        let max_x = mi_w * 4 - 1;
        let mut out = vec![0i64; up as usize * plane_h];
        for y in 0..plane_h {
            for x in 0..up {
                let src_x = -(1 << 14) + init + x * step;
                let px = src_x >> 14;
                let sub = (src_x & ((1 << 14) - 1)) >> 8;
                let mut sum = 0;
                for k in 0..8 {
                    let sx = (px + (k - 3)).clamp(0, max_x);
                    sum += frame[y * stride + sx as usize]
                        * UPSCALE_FILTER[sub as usize][k as usize] as i64;
                }
                out[y * up as usize + x as usize] = r2(sum, 7).clamp(0, (1 << bd) - 1);
            }
        }
        out
    }

    fn run<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let max = (1i64 << bd) - 1;
        let mut tested = 0;
        for iter in 0..400 {
            let upscaled_width = 8 + rng.range(400);
            let denom = 9 + rng.range(8);
            let frame_width = (upscaled_width * 8 + denom / 2) / denom;
            let sub_x = rng.range(2);
            let down = r2(frame_width, sub_x);
            let up = r2(upscaled_width, sub_x);
            if up <= down {
                continue;
            }
            tested += 1;
            let mi_cols = 2 * ((frame_width + 7) >> 3);
            let src_w = ((mi_cols >> sub_x) * 4) as usize;
            let stride = src_w + rng.range(3) as usize;
            let h = 1 + rng.range(5) as usize;
            let frame: Vec<i64> = (0..stride * h)
                .map(|_| {
                    if rng.range(4) == 0 {
                        rng.range(2) * max
                    } else {
                        rng.range(max as u32 + 1)
                    }
                })
                .collect();
            let expect = ref_upscale(
                &frame,
                stride,
                frame_width,
                upscaled_width,
                mi_cols,
                h,
                sub_x,
                bd,
            );
            let src: Vec<P> = frame.iter().map(|&v| P::from_i32(v as i32)).collect();
            let mut dst = vec![P::default(); up as usize * h];
            upscale_plane(
                &mut dst,
                up as usize,
                &src,
                stride,
                frame_width as usize,
                upscaled_width as usize,
                h,
                sub_x == 1,
                bd,
            );
            let got: Vec<i64> = dst.iter().map(|v| v.to_i32() as i64).collect();
            assert_eq!(
                got, expect,
                "iter {iter} {frame_width} -> {upscaled_width} sub {sub_x}"
            );
        }
        assert!(tested > 300);
    }

    #[test]
    fn superres_matches_spec_8bit() {
        run::<u8>(8, 0xabcd_ef01_2345_6789);
    }

    #[test]
    fn superres_matches_spec_10bit() {
        run::<u16>(10, 0xbbcd_ef01_2345_6789);
    }

    #[test]
    fn superres_matches_spec_12bit() {
        run::<u16>(12, 0xcbcd_ef01_2345_6789);
    }
}
