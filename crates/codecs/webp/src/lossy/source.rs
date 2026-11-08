//! The encoder's input: RGBA to 4:2:0 YUV with libwebp's fixed-point
//! BT.601 matrix, padded to whole macroblocks by repeating the last row and
//! column.

/// Padded planes. Strides are whole macroblocks.
pub(crate) struct Source {
    pub mb_w: usize,
    pub mb_h: usize,
    pub y: Vec<u8>,
    pub y_stride: usize,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub uv_stride: usize,
}

const YUV_FIX: u32 = 16;

fn to_y(r: i32, g: i32, b: i32) -> u8 {
    ((16_839 * r + 33_059 * g + 6_420 * b + (1 << (YUV_FIX - 1)) + (16 << YUV_FIX)) >> YUV_FIX)
        as u8
}

/// Chroma from sums of four pixels.
fn to_uv(value: i32) -> u8 {
    let v = (value + (1 << (YUV_FIX + 1)) + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2);
    v.clamp(0, 255) as u8
}

impl Source {
    /// `rgba` is `width x height` straight RGBA. Chroma of a pixel pair
    /// that is partly transparent is weighted by alpha. Unless `exact`,
    /// the colour under transparent pixels is replaced, as libwebp does:
    /// see [`Self::clean_transparent`].
    pub(crate) fn new(rgba: &[u8], width: usize, height: usize, exact: bool) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let y_stride = mb_w * 16;
        let uv_stride = mb_w * 8;
        let mut y = vec![0u8; y_stride * mb_h * 16];
        let mut u = vec![0u8; uv_stride * mb_h * 8];
        let mut v = vec![0u8; uv_stride * mb_h * 8];
        let pixel = |x: usize, row: usize| {
            let at = (row.min(height - 1) * width + x.min(width - 1)) * 4;
            [
                i32::from(rgba[at]),
                i32::from(rgba[at + 1]),
                i32::from(rgba[at + 2]),
                i32::from(rgba[at + 3]),
            ]
        };
        for row in 0..height {
            for x in 0..width {
                let [r, g, b, _] = pixel(x, row);
                y[row * y_stride + x] = to_y(r, g, b);
            }
        }
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        for row in 0..ch {
            for x in 0..cw {
                let quad = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .map(|(dx, dy)| pixel(2 * x + dx, 2 * row + dy));
                let alpha: i32 = quad.iter().map(|p| p[3]).sum();
                let sum = |c: usize| {
                    if alpha == 0 || alpha == 4 * 255 {
                        quad.iter().map(|p| p[c]).sum::<i32>()
                    } else {
                        // Four times the alpha-weighted mean.
                        (4 * quad.iter().map(|p| p[c] * p[3]).sum::<i32>() + alpha / 2) / alpha
                    }
                };
                let (r, g, b) = (sum(0), sum(1), sum(2));
                u[row * uv_stride + x] = to_uv(-9_719 * r - 19_081 * g + 28_800 * b);
                v[row * uv_stride + x] = to_uv(28_800 * r - 24_116 * g - 4_684 * b);
            }
        }
        let mut source = Self {
            mb_w,
            mb_h,
            y,
            y_stride,
            u,
            v,
            uv_stride,
        };
        if !exact {
            source.clean_transparent(rgba, width, height);
        }
        pad(&mut source.y, y_stride, width, height);
        pad(&mut source.u, uv_stride, cw, ch);
        pad(&mut source.v, uv_stride, cw, ch);
        source
    }

    /// libwebp's `WebPCleanupTransparentArea`: in each 8x8 luma block,
    /// transparent pixels take the mean luma of the visible ones; a fully
    /// transparent block is flattened, in luma and chroma, to the values
    /// of the first block of its run along the row, so runs cost nothing.
    fn clean_transparent(&mut self, rgba: &[u8], width: usize, height: usize) {
        const SIZE: usize = 8;
        let alpha = |x: usize, row: usize| rgba[(row * width + x) * 4 + 3];
        for y0 in (0..height).step_by(SIZE) {
            let rows = SIZE.min(height - y0);
            let mut run: Option<[u8; 3]> = None;
            for x0 in (0..width).step_by(SIZE) {
                let cols = SIZE.min(width - x0);
                let (mut sum, mut count) = (0u32, 0u32);
                for row in y0..y0 + rows {
                    for x in x0..x0 + cols {
                        if alpha(x, row) != 0 {
                            sum += u32::from(self.y[row * self.y_stride + x]);
                            count += 1;
                        }
                    }
                }
                let whole = rows == SIZE && cols == SIZE;
                if count == 0 && whole {
                    let values = *run.get_or_insert([
                        self.y[y0 * self.y_stride + x0],
                        self.u[(y0 / 2) * self.uv_stride + x0 / 2],
                        self.v[(y0 / 2) * self.uv_stride + x0 / 2],
                    ]);
                    fill(&mut self.y, self.y_stride, x0, y0, SIZE, values[0]);
                    fill(
                        &mut self.u,
                        self.uv_stride,
                        x0 / 2,
                        y0 / 2,
                        SIZE / 2,
                        values[1],
                    );
                    fill(
                        &mut self.v,
                        self.uv_stride,
                        x0 / 2,
                        y0 / 2,
                        SIZE / 2,
                        values[2],
                    );
                    continue;
                }
                run = None;
                if count > 0 && (count as usize) < rows * cols {
                    let mean = (sum / count) as u8;
                    for row in y0..y0 + rows {
                        for x in x0..x0 + cols {
                            if alpha(x, row) == 0 {
                                self.y[row * self.y_stride + x] = mean;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn fill(plane: &mut [u8], stride: usize, x0: usize, y0: usize, size: usize, value: u8) {
    for row in y0..y0 + size {
        plane[row * stride + x0..row * stride + x0 + size].fill(value);
    }
}

/// Repeat the last column and row of the `width x height` corner of
/// `plane` into the rest of it.
fn pad(plane: &mut [u8], stride: usize, width: usize, height: usize) {
    let rows = plane.len() / stride;
    for row in 0..height {
        let line = &mut plane[row * stride..(row + 1) * stride];
        let last = line[width - 1];
        line[width..].fill(last);
    }
    for row in height..rows {
        plane.copy_within((height - 1) * stride..height * stride, row * stride);
    }
}

#[cfg(test)]
mod tests {
    use super::Source;

    #[test]
    fn grey_stays_grey_and_padding_repeats_the_edge() {
        let rgba = [128u8, 128, 128, 255].repeat(3 * 5);
        let source = Source::new(&rgba, 3, 5, true);
        assert_eq!((source.mb_w, source.mb_h), (1, 1));
        assert!(source.y.iter().all(|&y| y == source.y[0]));
        assert!(source.u.iter().chain(&source.v).all(|&c| c == 128));
        let white = Source::new(&[255, 255, 255, 255], 1, 1, true);
        assert_eq!(white.y[0], 235);
        let black = Source::new(&[0, 0, 0, 255], 1, 1, true);
        assert_eq!(black.y[0], 16);
    }
}
