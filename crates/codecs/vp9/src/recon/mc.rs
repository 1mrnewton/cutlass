//! Motion compensation.
//!
//! The sample path follows libvpx `dec_build_inter_predictors` and
//! `vpx_convolve8_c`: a full-pel offset is a copy, one subpel axis uses that
//! 8-tap filter, and both axes filter horizontally into 8-bit samples before
//! the vertical pass. A reference whose crop differs from the frame uses
//! `vp9_scale_mv` and the same stepping filters. Samples outside the
//! reference crop are replicated, which is what `build_mc_border` does.

use cutlass_decode::CodecError;

use crate::tables::SUBPEL_FILTERS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

/// One mode-info cell of the previous coded frame, libvpx `MV_REF`.
#[derive(Clone, Copy, Default)]
pub struct MotionCell {
    pub mv: [Mv; 2],
    pub refs: [u8; 2],
}

/// Motion vectors from the previous coded frame, indexed in mode-info order.
#[derive(Clone)]
pub struct Motion {
    pub cols: usize,
    pub cells: Vec<MotionCell>,
}

pub struct RefPlane {
    pub samples: Vec<u8>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

pub struct Reference {
    pub y: RefPlane,
    pub u: RefPlane,
    pub v: RefPlane,
    pub width: u32,
    pub height: u32,
}

pub struct RefSlots<'a> {
    pub frames: [Option<&'a Reference>; 8],
}

impl<'a> RefSlots<'a> {
    pub fn none() -> Self {
        Self { frames: [None; 8] }
    }
}

impl Reference {
    pub fn from_picture(pic: &super::Picture) -> Result<Self, CodecError> {
        let width = usize::try_from(pic.width).unwrap_or(0);
        let height = usize::try_from(pic.height).unwrap_or(0);
        Ok(Self {
            y: RefPlane::tight(&pic.y, width, height)?,
            u: RefPlane::tight(&pic.u, width.div_ceil(2), height.div_ceil(2))?,
            v: RefPlane::tight(&pic.v, width.div_ceil(2), height.div_ceil(2))?,
            width: pic.width,
            height: pic.height,
        })
    }
}

impl RefPlane {
    fn tight(samples: &[u8], width: usize, height: usize) -> Result<Self, CodecError> {
        let need = width.checked_mul(height).ok_or(bad("reference plane"))?;
        if samples.len() != need || width == 0 || height == 0 {
            return Err(bad("reference plane"));
        }
        Ok(Self {
            samples: samples.to_vec(),
            stride: width,
            width,
            height,
        })
    }
}

pub struct Job<'a> {
    pub pred: &'a mut [u8],
    pub pred_stride: usize,
    pub w: usize,
    pub h: usize,
    pub origin_x: i32,
    pub origin_y: i32,
    pub mv_row: i32,
    pub mv_col: i32,
    pub filter: usize,
    pub reference: &'a RefPlane,
}

/// Fixed-point reference scale, libvpx `scale_factors` for 8-bit 4:2:0.
#[derive(Clone, Copy)]
pub struct Scale {
    x_fp: i32,
    y_fp: i32,
    x_step: i32,
    y_step: i32,
}

const REF_SCALE_SHIFT: i32 = 14;
const REF_NO_SCALE: i32 = 1 << REF_SCALE_SHIFT;

impl Scale {
    /// `None` when the reference crop matches the frame. An illegal ratio is
    /// the same rejection libvpx reports for an unusable reference.
    pub fn between(
        ref_w: u32,
        ref_h: u32,
        this_w: u32,
        this_h: u32,
    ) -> Result<Option<Self>, CodecError> {
        if !legal_scale(ref_w, ref_h, this_w, this_h) {
            return Err(CodecError::Unsupported {
                what: "vp9 reference scale",
            });
        }
        let x_fp = scale_fp(ref_w, this_w)?;
        let y_fp = scale_fp(ref_h, this_h)?;
        if x_fp == REF_NO_SCALE && y_fp == REF_NO_SCALE {
            return Ok(None);
        }
        Ok(Some(Self {
            x_fp,
            y_fp,
            x_step: scale_i32(16, x_fp)?,
            y_step: scale_i32(16, y_fp)?,
        }))
    }

    pub fn x_step(self) -> i32 {
        self.x_step
    }

    pub fn y_step(self) -> i32 {
        self.y_step
    }
}

pub struct ScaledJob<'a> {
    pub pred: &'a mut [u8],
    pub pred_stride: usize,
    pub w: usize,
    pub h: usize,
    pub x0: i32,
    pub y0: i32,
    pub sub_x: i32,
    pub sub_y: i32,
    pub x_step: i32,
    pub y_step: i32,
    pub filter: usize,
    pub reference: &'a RefPlane,
}

/// Block origin in the reference after `vp9_scale_mv`.
pub fn scaled_point(
    plane_x: i32,
    plane_y: i32,
    mv_col: i32,
    mv_row: i32,
    at_x: i32,
    at_y: i32,
    scale: &Scale,
) -> Result<(i32, i32, i32, i32), CodecError> {
    let col = scale_i32(mv_col, scale.x_fp)? + scale_q4(at_x, scale.x_fp)?;
    let row = scale_i32(mv_row, scale.y_fp)? + scale_q4(at_y, scale.y_fp)?;
    let x0 = scale_i32(plane_x, scale.x_fp)? + (col >> 4);
    let y0 = scale_i32(plane_y, scale.y_fp)? + (row >> 4);
    Ok((x0, y0, col & 15, row & 15))
}

pub fn render_scaled(mut job: ScaledJob<'_>) -> Result<(), CodecError> {
    if job.filter >= SUBPEL_FILTERS.len()
        || job.w == 0
        || job.h == 0
        || !(1..=64).contains(&job.x_step)
        || !(1..=64).contains(&job.y_step)
    {
        return Err(bad("motion sample"));
    }
    let horiz = job.x_step != 16 || job.sub_x != 0;
    let vert = job.y_step != 16 || job.sub_y != 0;
    if !horiz && !vert {
        copy_scaled(&mut job)
    } else if horiz && vert {
        both_scaled(&mut job)
    } else if horiz {
        horiz_scaled(&mut job)
    } else {
        vert_scaled(&mut job)
    }
}

pub fn render(mut job: Job<'_>) -> Result<(), CodecError> {
    if job.filter >= SUBPEL_FILTERS.len() || job.w == 0 || job.h == 0 {
        return Err(bad("motion sample"));
    }
    let (dx, sx) = split_q4(job.mv_col);
    let (dy, sy) = split_q4(job.mv_row);
    let x0 = job.origin_x + dx;
    let y0 = job.origin_y + dy;
    if sx == 0 && sy == 0 {
        copy(&mut job, x0, y0)
    } else if sy == 0 {
        horiz(&mut job, x0, y0, sx)
    } else if sx == 0 {
        vert(&mut job, x0, y0, sy)
    } else {
        both(&mut job, x0, y0, sx, sy)
    }
}

fn copy_scaled(job: &mut ScaledJob<'_>) -> Result<(), CodecError> {
    for row in 0..job.h {
        let y = job.y0 + i32::try_from(row).unwrap_or(0);
        for col in 0..job.w {
            let x = job.x0 + i32::try_from(col).unwrap_or(0);
            let value = sample(job.reference, x, y)?;
            put_scaled(job, col, row, value)?;
        }
    }
    Ok(())
}

fn horiz_scaled(job: &mut ScaledJob<'_>) -> Result<(), CodecError> {
    for row in 0..job.h {
        let y = job.y0 + i32::try_from(row).unwrap_or(0);
        let mut x_q4 = job.sub_x;
        for col in 0..job.w {
            let value = horiz_tap(job, x_q4, y)?;
            put_scaled(job, col, row, value)?;
            x_q4 += job.x_step;
        }
    }
    Ok(())
}

fn vert_scaled(job: &mut ScaledJob<'_>) -> Result<(), CodecError> {
    for row in 0..job.h {
        let y_q4 = job.sub_y + i32::try_from(row).unwrap_or(0) * job.y_step;
        let phase = phase_of(y_q4)?;
        let kernel = &SUBPEL_FILTERS[job.filter][phase];
        let base_y = job.y0 + (y_q4 >> 4);
        for col in 0..job.w {
            let x = job.x0 + i32::try_from(col).unwrap_or(0);
            let value = filter_column(job.reference, x, base_y, kernel)?;
            put_scaled(job, col, row, value)?;
        }
    }
    Ok(())
}

fn both_scaled(job: &mut ScaledJob<'_>) -> Result<(), CodecError> {
    let height = i32::try_from(job.h).unwrap_or(0);
    let rows_i = ((height - 1) * job.y_step + job.sub_y) >> 4;
    let rows = usize::try_from(rows_i + 8).unwrap_or(0);
    let width = job.w;
    let cells = width.checked_mul(rows).ok_or(bad("motion sample"))?;
    if rows < 8 {
        return Err(bad("motion sample"));
    }
    let mut temp = vec![0u8; cells];
    for row in 0..rows {
        let y = job.y0 - 3 + i32::try_from(row).unwrap_or(0);
        let mut x_q4 = job.sub_x;
        for col in 0..width {
            temp[row * width + col] = horiz_tap(job, x_q4, y)?;
            x_q4 += job.x_step;
        }
    }
    for row in 0..job.h {
        let y_q4 = job.sub_y + i32::try_from(row).unwrap_or(0) * job.y_step;
        let phase = phase_of(y_q4)?;
        let kernel = &SUBPEL_FILTERS[job.filter][phase];
        let base = usize::try_from(y_q4 >> 4).unwrap_or(0);
        for col in 0..width {
            let mut sum = 0i32;
            for (tap, coef) in kernel.iter().enumerate() {
                let sample = i32::from(
                    *temp
                        .get((base + tap) * width + col)
                        .ok_or(bad("motion sample"))?,
                );
                sum += i32::from(*coef) * sample;
            }
            put_scaled(job, col, row, round7(sum))?;
        }
    }
    Ok(())
}

fn horiz_tap(job: &ScaledJob<'_>, x_q4: i32, y: i32) -> Result<u8, CodecError> {
    let phase = phase_of(x_q4)?;
    let kernel = &SUBPEL_FILTERS[job.filter][phase];
    let x = job.x0 + (x_q4 >> 4);
    filter_at(job.reference, x, y, kernel)
}

fn phase_of(q4: i32) -> Result<usize, CodecError> {
    usize::try_from(q4 & 15).map_err(|_| bad("motion sample"))
}

fn put_scaled(
    job: &mut ScaledJob<'_>,
    col: usize,
    row: usize,
    value: u8,
) -> Result<(), CodecError> {
    let at = row * job.pred_stride + col;
    let slot = job.pred.get_mut(at).ok_or(bad("motion sample"))?;
    *slot = value;
    Ok(())
}

fn legal_scale(ref_w: u32, ref_h: u32, this_w: u32, this_h: u32) -> bool {
    if ref_w == 0 || ref_h == 0 || this_w == 0 || this_h == 0 {
        return false;
    }
    let ref_w = u64::from(ref_w);
    let ref_h = u64::from(ref_h);
    let this_w = u64::from(this_w);
    let this_h = u64::from(this_h);
    this_w.saturating_mul(2) >= ref_w
        && this_h.saturating_mul(2) >= ref_h
        && this_w <= ref_w.saturating_mul(16)
        && this_h <= ref_h.saturating_mul(16)
}

fn scale_fp(other: u32, this: u32) -> Result<i32, CodecError> {
    let num = i64::from(other) << REF_SCALE_SHIFT;
    let den = i64::from(this);
    i32::try_from(num / den).map_err(|_| bad("reference scale"))
}

fn scale_i32(val: i32, fp: i32) -> Result<i32, CodecError> {
    let prod = i64::from(val) * i64::from(fp);
    i32::try_from(prod >> REF_SCALE_SHIFT).map_err(|_| bad("reference scale"))
}

fn scale_q4(pos: i32, fp: i32) -> Result<i32, CodecError> {
    let shifted = i64::from(pos) << 4;
    let prod = shifted * i64::from(fp);
    let scaled = i32::try_from(prod >> REF_SCALE_SHIFT).map_err(|_| bad("reference scale"))?;
    Ok(scaled & 15)
}

fn split_q4(delta: i32) -> (i32, usize) {
    (delta >> 4, usize::try_from(delta & 15).unwrap_or(0))
}

fn copy(job: &mut Job<'_>, x0: i32, y0: i32) -> Result<(), CodecError> {
    for row in 0..job.h {
        let y = y0 + i32::try_from(row).unwrap_or(0);
        for col in 0..job.w {
            let x = x0 + i32::try_from(col).unwrap_or(0);
            let value = sample(job.reference, x, y)?;
            put(job, col, row, value)?;
        }
    }
    Ok(())
}

fn horiz(job: &mut Job<'_>, x0: i32, y0: i32, sub: usize) -> Result<(), CodecError> {
    let kernel = SUBPEL_FILTERS[job.filter][sub];
    for row in 0..job.h {
        let y = y0 + i32::try_from(row).unwrap_or(0);
        for col in 0..job.w {
            let x = x0 + i32::try_from(col).unwrap_or(0);
            let value = filter_at(job.reference, x, y, &kernel)?;
            put(job, col, row, value)?;
        }
    }
    Ok(())
}

fn vert(job: &mut Job<'_>, x0: i32, y0: i32, sub: usize) -> Result<(), CodecError> {
    let kernel = SUBPEL_FILTERS[job.filter][sub];
    for row in 0..job.h {
        let y = y0 + i32::try_from(row).unwrap_or(0);
        for col in 0..job.w {
            let x = x0 + i32::try_from(col).unwrap_or(0);
            let value = filter_column(job.reference, x, y, &kernel)?;
            put(job, col, row, value)?;
        }
    }
    Ok(())
}

fn both(job: &mut Job<'_>, x0: i32, y0: i32, sx: usize, sy: usize) -> Result<(), CodecError> {
    let rows = (((job.h - 1) * 16 + sy) >> 4) + 8;
    let mut temp = vec![0u8; job.w.saturating_mul(rows)];
    let kernel_x = SUBPEL_FILTERS[job.filter][sx];
    for row in 0..rows {
        let y = y0 - 3 + i32::try_from(row).unwrap_or(0);
        for col in 0..job.w {
            let x = x0 + i32::try_from(col).unwrap_or(0);
            let at = row * job.w + col;
            temp[at] = filter_at(job.reference, x, y, &kernel_x)?;
        }
    }
    let kernel_y = SUBPEL_FILTERS[job.filter][sy];
    for row in 0..job.h {
        let base = (sy + row * 16) >> 4;
        for col in 0..job.w {
            let mut sum = 0i32;
            for (tap, coef) in kernel_y.iter().enumerate() {
                let sample = i32::from(
                    *temp
                        .get((base + tap) * job.w + col)
                        .ok_or(bad("motion sample"))?,
                );
                sum += i32::from(*coef) * sample;
            }
            put(job, col, row, round7(sum))?;
        }
    }
    Ok(())
}

fn filter_at(plane: &RefPlane, x: i32, y: i32, kernel: &[i16; 8]) -> Result<u8, CodecError> {
    let mut sum = 0i32;
    for (tap, coef) in kernel.iter().enumerate() {
        let sample = i32::from(sample(plane, x - 3 + i32::try_from(tap).unwrap_or(0), y)?);
        sum += i32::from(*coef) * sample;
    }
    Ok(round7(sum))
}

fn filter_column(plane: &RefPlane, x: i32, y: i32, kernel: &[i16; 8]) -> Result<u8, CodecError> {
    let mut sum = 0i32;
    for (tap, coef) in kernel.iter().enumerate() {
        let sample = i32::from(sample(plane, x, y - 3 + i32::try_from(tap).unwrap_or(0))?);
        sum += i32::from(*coef) * sample;
    }
    Ok(round7(sum))
}

fn sample(plane: &RefPlane, x: i32, y: i32) -> Result<u8, CodecError> {
    if plane.width == 0 || plane.height == 0 {
        return Err(bad("reference plane"));
    }
    let max_x = i32::try_from(plane.width - 1).unwrap_or(0);
    let max_y = i32::try_from(plane.height - 1).unwrap_or(0);
    let x = usize::try_from(x.clamp(0, max_x)).unwrap_or(0);
    let y = usize::try_from(y.clamp(0, max_y)).unwrap_or(0);
    plane
        .samples
        .get(y * plane.stride + x)
        .copied()
        .ok_or(bad("reference plane"))
}

fn put(job: &mut Job<'_>, col: usize, row: usize, value: u8) -> Result<(), CodecError> {
    let at = row * job.pred_stride + col;
    let slot = job.pred.get_mut(at).ok_or(bad("motion sample"))?;
    *slot = value;
    Ok(())
}

fn round7(sum: i32) -> u8 {
    let rounded = (sum + 64) >> 7;
    u8::try_from(rounded.clamp(0, 255)).unwrap_or(0)
}

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

#[cfg(test)]
mod tests {
    use super::Scale;
    use cutlass_decode::CodecError;

    #[test]
    fn equal_sizes_stay_on_the_unscaled_path() {
        assert!(Scale::between(64, 64, 64, 64).expect("scale").is_none());
    }

    #[test]
    fn half_and_four_fifths_steps_match_libvpx() {
        let half = Scale::between(64, 64, 32, 32)
            .expect("scale")
            .expect("scaled");
        assert_eq!((half.x_step(), half.y_step()), (32, 32));
        let four_fifths = Scale::between(64, 64, 52, 52)
            .expect("scale")
            .expect("scaled");
        assert_eq!((four_fifths.x_step(), four_fifths.y_step()), (19, 19));
    }

    #[test]
    fn an_illegal_ratio_is_unsupported() {
        let err = match Scale::between(8, 8, 200, 200) {
            Err(err) => err,
            Ok(_) => panic!("accepted an illegal ratio"),
        };
        assert!(matches!(
            err,
            CodecError::Unsupported {
                what: "vp9 reference scale"
            }
        ));
    }
}
