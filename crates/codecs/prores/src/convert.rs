//! Wide 4:2:2 or 4:4:4 to the 8-bit 4:2:0 the app shows.
//!
//! The row functions here are the whole conversion. The decoder calls them
//! from each slice job on the rows that job has just decoded, and
//! [`to_yuv420p`] calls them on a finished frame; both give the same bytes.

use cutlass_decode::{CodecError, PixelFormat, VideoFrame};

/// How `depth`-bit samples round to 8 bits.
#[derive(Clone, Copy)]
pub(crate) struct Narrow {
    half: u16,
    shift: u32,
}

impl Narrow {
    /// `depth` is 10 or 12.
    pub(crate) fn new(depth: u8) -> Self {
        let shift = u32::from(depth.clamp(9, 12)) - 8;
        Self {
            half: 1 << (shift - 1),
            shift,
        }
    }
}

/// One luma row. `dst` is `src` or one sample longer; the extra sample
/// repeats the last. The sums saturate, which only matters for samples far
/// above the depth: those come out 255 either way.
#[inline(always)]
pub(crate) fn luma_row(src: &[u16], dst: &mut [u8], narrow: Narrow) {
    for (out, &sample) in dst.iter_mut().zip(src) {
        *out = (sample.saturating_add(narrow.half) >> narrow.shift).min(255) as u8;
    }
    repeat_last(dst, src.len());
}

/// One 4:2:0 chroma row from two 4:2:2 rows.
#[inline(always)]
pub(crate) fn chroma_row_422(top: &[u16], bottom: &[u16], dst: &mut [u8], narrow: Narrow) {
    let round = 2 * narrow.half;
    for ((out, &a), &b) in dst.iter_mut().zip(top).zip(bottom) {
        let sum = a.saturating_add(b).saturating_add(round);
        *out = (sum >> (narrow.shift + 1)).min(255) as u8;
    }
}

/// One 4:2:0 chroma row from two 4:4:4 rows. An odd last column counts twice.
#[inline(always)]
pub(crate) fn chroma_row_444(top: &[u16], bottom: &[u16], dst: &mut [u8], narrow: Narrow) {
    let round = 4 * narrow.half;
    let (top_pairs, top_last) = top.as_chunks::<2>();
    let (bottom_pairs, bottom_last) = bottom.as_chunks::<2>();
    for ((out, a), b) in dst.iter_mut().zip(top_pairs).zip(bottom_pairs) {
        let sum = a[0]
            .saturating_add(a[1])
            .saturating_add(b[0])
            .saturating_add(b[1])
            .saturating_add(round);
        *out = (sum >> (narrow.shift + 2)).min(255) as u8;
    }
    if let (Some(out), Some(&a), Some(&b)) = (
        dst.get_mut(top_pairs.len()),
        top_last.first(),
        bottom_last.first(),
    ) {
        let sum = a
            .saturating_add(a)
            .saturating_add(b)
            .saturating_add(b)
            .saturating_add(round);
        *out = (sum >> (narrow.shift + 2)).min(255) as u8;
    }
}

/// Fills `row[filled..]` with the sample before it.
#[inline(always)]
fn repeat_last(row: &mut [u8], filled: usize) {
    if filled == 0 || filled >= row.len() {
        return;
    }
    let last = row[filled - 1];
    row[filled..].fill(last);
}

/// Reads row `y` of a 16-bit little-endian plane into `out`.
fn wide_row(samples: &[u8], stride: usize, y: usize, out: &mut [u16]) {
    let start = y * stride;
    let bytes = samples
        .get(start..start + out.len() * 2)
        .unwrap_or_default();
    let (pairs, _) = bytes.as_chunks::<2>();
    out.fill(0);
    for (value, pair) in out.iter_mut().zip(pairs) {
        *value = u16::from_le_bytes(*pair);
    }
}

/// Packs `frame` as tight 8-bit 4:2:0, Y then U then V. Odd edges grow by
/// one repeated sample to the even size 4:2:0 needs. Returns the size and bytes.
pub fn to_yuv420p(frame: &VideoFrame) -> Result<(u32, u32, Vec<u8>), CodecError> {
    let (depth, horizontal) = match frame.format {
        PixelFormat::Yuv422p10 => (10, true),
        PixelFormat::Yuv422p12 => (12, true),
        PixelFormat::Yuv444p10 => (10, false),
        PixelFormat::Yuv444p12 => (12, false),
        _ => {
            return Err(CodecError::InvalidApi {
                what: "picture format",
            });
        }
    };
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    let out_w = width.div_ceil(2) * 2;
    let out_h = height.div_ceil(2) * 2;
    let narrow = Narrow::new(depth);
    let mut bytes = vec![0u8; out_w * out_h * 3 / 2];
    let (luma, chroma) = bytes.split_at_mut(out_w * out_h);
    let mut row = vec![0u16; width];
    let samples: &[u8] = &frame.y.samples;
    for (y, out) in luma.chunks_exact_mut(out_w).enumerate() {
        wide_row(samples, frame.y.stride, y.min(height - 1), &mut row);
        luma_row(&row, out, narrow);
    }
    let chroma_w = if horizontal { width.div_ceil(2) } else { width };
    let mut top = vec![0u16; chroma_w];
    let mut bottom = vec![0u16; chroma_w];
    let (u, v) = chroma.split_at_mut(out_w / 2 * (out_h / 2));
    for (plane, out) in [(&frame.u, u), (&frame.v, v)] {
        let samples: &[u8] = &plane.samples;
        for (cy, out) in out.chunks_exact_mut(out_w / 2).enumerate() {
            wide_row(samples, plane.stride, (2 * cy).min(height - 1), &mut top);
            wide_row(
                samples,
                plane.stride,
                (2 * cy + 1).min(height - 1),
                &mut bottom,
            );
            if horizontal {
                chroma_row_422(&top, &bottom, out, narrow);
            } else {
                chroma_row_444(&top, &bottom, out, narrow);
            }
        }
    }
    Ok((out_w as u32, out_h as u32, bytes))
}

#[cfg(test)]
mod tests {
    use super::to_yuv420p;
    use cutlass_decode::{PixelFormat, Plane, VideoFrame};

    fn plane(width: u32, height: u32, value: u16) -> Plane {
        let samples = (0..width * height)
            .flat_map(|_| value.to_le_bytes())
            .collect();
        Plane::bytes(width, height, 2, samples).expect("plane")
    }

    #[test]
    fn an_odd_422_frame_grows_to_even_420() {
        let frame = VideoFrame::planar(
            PixelFormat::Yuv422p10,
            3,
            3,
            plane(3, 3, 940),
            plane(2, 3, 512),
            plane(2, 3, 64),
        )
        .expect("frame");
        let (w, h, bytes) = to_yuv420p(&frame).expect("convert");
        assert_eq!((w, h), (4, 4));
        assert_eq!(bytes.len(), 16 + 4 + 4);
        assert!(bytes[..16].iter().all(|&b| b == 235));
        assert!(bytes[16..20].iter().all(|&b| b == 128));
        assert!(bytes[20..].iter().all(|&b| b == 16));
    }

    #[test]
    fn a_12_bit_444_frame_averages_chroma() {
        let frame = VideoFrame::planar(
            PixelFormat::Yuv444p12,
            2,
            2,
            plane(2, 2, 256),
            plane(2, 2, 2048),
            plane(2, 2, 4095),
        )
        .expect("frame");
        let (w, h, bytes) = to_yuv420p(&frame).expect("convert");
        assert_eq!((w, h), (2, 2));
        assert_eq!(bytes, vec![16, 16, 16, 16, 128, 255]);
    }
}
