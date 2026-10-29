//! Unscaled motion compensation.
//!
//! The sample path follows libvpx `dec_build_inter_predictors` and
//! `vpx_convolve8_c`: a full-pel offset is a copy, one subpel axis uses that
//! 8-tap filter, and both axes filter horizontally into 8-bit samples before
//! the vertical pass. Samples outside the reference crop are replicated, which
//! is what `build_mc_border` does.

use cutlass_decode::CodecError;

use crate::tables::SUBPEL_FILTERS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
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
