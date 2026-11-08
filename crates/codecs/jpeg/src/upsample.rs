//! Component planes to full-size pixels.
//!
//! A component sampled at half the width, half the height, or both is
//! filtered with the triangle weights libjpeg uses by default: three
//! quarters of the nearer sample and one quarter of the farther one, with
//! the edge sample repeated. Other ratios repeat each sample. A component
//! two samples wide or narrower is repeated too, as in libjpeg.

use cutlass_decode::CodecError;

use crate::color::ycbcr_to_rgb;
use crate::image::{ColorModel, Component, Image};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    Copy,
    H2V1,
    H1V2,
    H2V2,
    Repeat,
}

struct Upsampler<'a> {
    comp: &'a Component,
    filter: Filter,
    max_h: usize,
    max_v: usize,
    row: Vec<u8>,
    sums: Vec<u16>,
}

impl<'a> Upsampler<'a> {
    fn new(image: &Image, comp: &'a Component) -> Result<Self, CodecError> {
        let width = usize::try_from(image.width).map_err(|_| limit())?;
        let max_h = usize::from(image.max_h);
        let max_v = usize::from(image.max_v);
        let h = usize::from(comp.h);
        let v = usize::from(comp.v);
        let integral = max_h % h == 0 && max_v % v == 0;
        let (h_ratio, v_ratio) = (max_h / h, max_v / v);
        let narrow = comp.width <= 2;
        let filter = match (integral, h_ratio, v_ratio) {
            (true, 1, 1) => Filter::Copy,
            (true, 2, 1) if !narrow => Filter::H2V1,
            (true, 1, 2) => Filter::H1V2,
            (true, 2, 2) if !narrow => Filter::H2V2,
            _ => Filter::Repeat,
        };
        let comp_width = usize::try_from(comp.width).map_err(|_| limit())?;
        let mut row = Vec::new();
        row.try_reserve_exact(width + 2).map_err(|_| limit())?;
        row.resize(width + 2, 0);
        let mut sums = Vec::new();
        if filter == Filter::H2V2 {
            sums.try_reserve_exact(comp_width).map_err(|_| limit())?;
            sums.resize(comp_width, 0);
        }
        Ok(Self {
            comp,
            filter,
            max_h,
            max_v,
            row,
            sums,
        })
    }

    fn comp_row(&self, index: usize) -> &'a [u8] {
        let width = usize::try_from(self.comp.width).unwrap_or(0);
        let height = usize::try_from(self.comp.height).unwrap_or(1);
        let index = index.min(height.saturating_sub(1));
        self.comp
            .samples
            .get(index * width..index * width + width)
            .unwrap_or(&[])
    }

    /// Full-size row `y`, at least `width` samples.
    fn row(&mut self, y: usize, width: usize) -> &[u8] {
        match self.filter {
            Filter::Copy => return self.comp_row(y),
            Filter::H2V1 => {
                let input = self.comp_row(y);
                h2v1(input, &mut self.row);
            }
            Filter::H1V2 => {
                let near = self.comp_row(y / 2);
                let far = if y.is_multiple_of(2) {
                    self.comp_row((y / 2).saturating_sub(1))
                } else {
                    self.comp_row(y / 2 + 1)
                };
                let bias = if y.is_multiple_of(2) { 1 } else { 2 };
                for ((out, a), b) in self.row.iter_mut().zip(near).zip(far) {
                    *out = narrow((u16::from(*a) * 3 + u16::from(*b) + bias) >> 2);
                }
            }
            Filter::H2V2 => {
                let near = self.comp_row(y / 2);
                let far = if y.is_multiple_of(2) {
                    self.comp_row((y / 2).saturating_sub(1))
                } else {
                    self.comp_row(y / 2 + 1)
                };
                for ((sum, a), b) in self.sums.iter_mut().zip(near).zip(far) {
                    *sum = u16::from(*a) * 3 + u16::from(*b);
                }
                h2v2(&self.sums, &mut self.row);
            }
            Filter::Repeat => {
                let h = usize::from(self.comp.h);
                let v = usize::from(self.comp.v);
                let input = self.comp_row(y * v / self.max_v);
                for (x, out) in self.row.iter_mut().enumerate().take(width) {
                    *out = input.get(x * h / self.max_h).copied().unwrap_or(0);
                }
            }
        }
        &self.row
    }
}

fn narrow(value: u16) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn h2v1(input: &[u8], out: &mut [u8]) {
    let last = input.len().saturating_sub(1);
    for (c, value) in input.iter().enumerate() {
        let here = u16::from(*value) * 3;
        let left = u16::from(input[c.saturating_sub(1)]);
        let right = u16::from(input[(c + 1).min(last)]);
        if let Some(pair) = out.get_mut(2 * c..2 * c + 2) {
            pair[0] = narrow((here + left + 1) >> 2);
            pair[1] = narrow((here + right + 2) >> 2);
        }
    }
}

fn h2v2(sums: &[u16], out: &mut [u8]) {
    let last = sums.len().saturating_sub(1);
    for (c, sum) in sums.iter().enumerate() {
        let here = u32::from(*sum) * 3;
        let left = u32::from(sums[c.saturating_sub(1)]);
        let right = u32::from(sums[(c + 1).min(last)]);
        if let Some(pair) = out.get_mut(2 * c..2 * c + 2) {
            pair[0] = u8::try_from((here + left + 8) >> 4).unwrap_or(u8::MAX);
            pair[1] = u8::try_from((here + right + 7) >> 4).unwrap_or(u8::MAX);
        }
    }
}

fn limit() -> CodecError {
    CodecError::Limit {
        what: "jpeg picture",
    }
}

fn picture_len(image: &Image, channels: usize) -> Result<usize, CodecError> {
    usize::try_from(image.width)
        .ok()
        .and_then(|width| width.checked_mul(usize::try_from(image.height).ok()?))
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(limit)
}

impl Image {
    /// Call `row` once per picture row with every component at full size.
    fn rows(&self, mut row: impl FnMut(usize, &[&[u8]])) -> Result<(), CodecError> {
        let width = usize::try_from(self.width).map_err(|_| limit())?;
        let height = usize::try_from(self.height).map_err(|_| limit())?;
        let mut upsamplers = self
            .components
            .iter()
            .map(|comp| Upsampler::new(self, comp))
            .collect::<Result<Vec<_>, _>>()?;
        for y in 0..height {
            let views: Vec<&[u8]> = upsamplers
                .iter_mut()
                .map(|upsampler| upsampler.row(y, width).get(..width).unwrap_or(&[]))
                .collect();
            if views.iter().any(|view| view.len() != width) {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "sample",
                });
            }
            row(y, &views);
        }
        Ok(())
    }

    /// Each component at full size, `width` × `height`, in frame order.
    pub fn full_planes(&self) -> Result<Vec<Vec<u8>>, CodecError> {
        let len = picture_len(self, 1)?;
        let width = usize::try_from(self.width).map_err(|_| limit())?;
        let mut planes = Vec::with_capacity(self.components.len());
        for _ in &self.components {
            let mut plane = Vec::new();
            plane.try_reserve_exact(len).map_err(|_| limit())?;
            plane.resize(len, 0);
            planes.push(plane);
        }
        self.rows(|y, rows| {
            for (plane, row) in planes.iter_mut().zip(rows) {
                if let Some(dest) = plane.get_mut(y * width..y * width + width) {
                    dest.copy_from_slice(row);
                }
            }
        })?;
        Ok(planes)
    }

    /// Grey as one sample per pixel; RGB and YCbCr as R, G, B.
    pub fn to_rgb(&self) -> Result<Vec<u8>, CodecError> {
        let channels = if self.color == ColorModel::Gray { 1 } else { 3 };
        let mut out = Vec::new();
        out.try_reserve_exact(picture_len(self, channels)?)
            .map_err(|_| limit())?;
        let color = self.color;
        self.rows(|_, rows| match (color, rows) {
            (ColorModel::Gray, [gray, ..]) => out.extend_from_slice(gray),
            (ColorModel::Rgb, [red, green, blue, ..]) => {
                for ((r, g), b) in red.iter().zip(*green).zip(*blue) {
                    out.extend_from_slice(&[*r, *g, *b]);
                }
            }
            (ColorModel::YCbCr, [luma, cb, cr, ..]) => {
                for ((y, b), r) in luma.iter().zip(*cb).zip(*cr) {
                    let (red, green, blue) =
                        ycbcr_to_rgb(i32::from(*y), i32::from(*b), i32::from(*r));
                    out.extend_from_slice(&[red, green, blue]);
                }
            }
            _ => {}
        })?;
        if out.len() != picture_len(self, channels)? {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "picture",
            });
        }
        Ok(out)
    }

    /// R, G, B, A with alpha 255, into `out`, which is resized to fit.
    pub fn to_rgba(&self, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let len = picture_len(self, 4)?;
        let stride = usize::try_from(self.width).map_err(|_| limit())? * 4;
        out.clear();
        out.try_reserve_exact(len).map_err(|_| limit())?;
        out.resize(len, 255);
        let color = self.color;
        self.rows(|y, rows| {
            let Some(line) = out.get_mut(y * stride..y * stride + stride) else {
                return;
            };
            let pixels = line.as_chunks_mut::<4>().0.iter_mut();
            match (color, rows) {
                (ColorModel::Gray, [gray, ..]) => {
                    for (pixel, value) in pixels.zip(*gray) {
                        pixel[..3].fill(*value);
                    }
                }
                (ColorModel::Rgb, [red, green, blue, ..]) => {
                    for (((pixel, r), g), b) in pixels.zip(*red).zip(*green).zip(*blue) {
                        pixel[..3].copy_from_slice(&[*r, *g, *b]);
                    }
                }
                (ColorModel::YCbCr, [luma, cb, cr, ..]) => {
                    for (((pixel, y), b), r) in pixels.zip(*luma).zip(*cb).zip(*cr) {
                        let (red, green, blue) =
                            ycbcr_to_rgb(i32::from(*y), i32::from(*b), i32::from(*r));
                        pixel[..3].copy_from_slice(&[red, green, blue]);
                    }
                }
                _ => {}
            }
        })
    }
}
