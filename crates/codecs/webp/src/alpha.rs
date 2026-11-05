//! The `ALPH` chunk: the alpha plane of a lossy picture.
//!
//! One header byte: two reserved bits, the pre-processing hint, the
//! filtering method, and the compression (raw bytes, or a headerless VP8L
//! stream whose green channel is the alpha). The filter is undone after
//! decompression. Pre-processing only says the encoder reduced the
//! levels; it needs nothing from a decoder.

use cutlass_decode::CodecError;

use crate::lossless::decode_headless;

pub(crate) const FILTER_NONE: u8 = 0;
pub(crate) const FILTER_HORIZONTAL: u8 = 1;
pub(crate) const FILTER_VERTICAL: u8 = 2;
pub(crate) const FILTER_GRADIENT: u8 = 3;

pub(crate) fn decode_alpha(
    chunk: &[u8],
    width: usize,
    height: usize,
    max_pixels: usize,
) -> Result<Vec<u8>, CodecError> {
    let Some((&header, data)) = chunk.split_first() else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "alph header",
        });
    };
    let compression = header & 3;
    let filter = (header >> 2) & 3;
    let preprocessing = (header >> 4) & 3;
    if header >> 6 != 0 || preprocessing > 1 || compression > 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "alph header",
        });
    }
    let pixels = width
        .checked_mul(height)
        .filter(|&p| p <= max_pixels)
        .ok_or(CodecError::Limit { what: "alph size" })?;
    let mut alpha = if compression == 0 {
        data.get(..pixels)
            .ok_or(CodecError::Truncated {
                at_bit: u64::try_from(data.len()).unwrap_or(u64::MAX) * 8,
                what: "alph raw plane",
            })?
            .to_vec()
    } else {
        decode_headless(data, width, height, max_pixels)?
            .into_iter()
            .map(|argb| ((argb >> 8) & 0xFF) as u8)
            .collect()
    };
    unfilter(&mut alpha, width, filter);
    Ok(alpha)
}

/// Undo a prediction filter in place. The first pixel is predicted from
/// zero, the rest of the top row from the left, the rest of the left column
/// from above; the method chooses for every other pixel.
pub(crate) fn unfilter(plane: &mut [u8], width: usize, filter: u8) {
    if filter == FILTER_NONE || width == 0 {
        return;
    }
    let height = plane.len() / width;
    for x in 1..width.min(plane.len()) {
        plane[x] = plane[x].wrapping_add(plane[x - 1]);
    }
    for y in 1..height {
        let row = y * width;
        plane[row] = plane[row].wrapping_add(plane[row - width]);
        for at in row + 1..row + width {
            let (left, top, corner) = (plane[at - 1], plane[at - width], plane[at - width - 1]);
            let predicted = match filter {
                FILTER_HORIZONTAL => left,
                FILTER_VERTICAL => top,
                FILTER_GRADIENT => u8::try_from(
                    (i32::from(left) + i32::from(top) - i32::from(corner)).clamp(0, 255),
                )
                .unwrap_or(0),
                _ => 0,
            };
            plane[at] = plane[at].wrapping_add(predicted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The forward filters, written from the container specification, so
    /// each inverse is checked against an independent encoder side.
    fn filter(plane: &[u8], width: usize, method: u8) -> Vec<u8> {
        let height = plane.len() / width;
        let mut out = plane.to_vec();
        for y in 0..height {
            for x in 0..width {
                let at = y * width + x;
                let predicted = if x == 0 && y == 0 {
                    0
                } else if y == 0 {
                    plane[at - 1]
                } else if x == 0 {
                    plane[at - width]
                } else {
                    let (a, b, c) = (plane[at - 1], plane[at - width], plane[at - width - 1]);
                    match method {
                        FILTER_HORIZONTAL => a,
                        FILTER_VERTICAL => b,
                        _ => (i32::from(a) + i32::from(b) - i32::from(c)).clamp(0, 255) as u8,
                    }
                };
                out[at] = plane[at].wrapping_sub(predicted);
            }
        }
        out
    }

    #[test]
    fn every_filter_inverts_its_forward_filter() {
        let width = 7;
        let plane: Vec<u8> = (0..7 * 5).map(|n| ((n * 37 + n * n) % 256) as u8).collect();
        for method in [FILTER_HORIZONTAL, FILTER_VERTICAL, FILTER_GRADIENT] {
            let mut filtered = filter(&plane, width, method);
            unfilter(&mut filtered, width, method);
            assert_eq!(filtered, plane, "method {method}");
        }
        let mut same = plane.clone();
        unfilter(&mut same, width, FILTER_NONE);
        assert_eq!(same, plane);
    }

    #[test]
    fn raw_alpha_reads_the_plane_and_refuses_short_or_bad_headers() {
        let mut chunk = vec![FILTER_HORIZONTAL << 2];
        chunk.extend_from_slice(&[10, 5, 5, 1, 1, 1]);
        assert_eq!(
            decode_alpha(&chunk, 3, 2, 100).unwrap(),
            [10, 15, 20, 11, 12, 13]
        );
        assert!(decode_alpha(&chunk[..5], 3, 2, 100).is_err());
        assert!(decode_alpha(&[], 3, 2, 100).is_err());
        assert!(decode_alpha(&[0xC0, 0, 0], 1, 2, 100).is_err());
        assert!(decode_alpha(&[2, 0, 0], 1, 2, 100).is_err());
        assert!(matches!(
            decode_alpha(&[0, 0], 1000, 1000, 100),
            Err(CodecError::Limit { .. })
        ));
    }
}
