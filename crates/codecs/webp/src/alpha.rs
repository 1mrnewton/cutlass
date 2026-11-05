//! The `ALPH` chunk of a lossy picture (RFC 9649 section 2.7.1.2).

use cutlass_decode::CodecError;

use crate::limits::pixel_count;
use crate::lossless::decode::decode_stream;

/// How each alpha value was predicted before coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Filter {
    None,
    Horizontal,
    Vertical,
    Gradient,
}

impl Filter {
    fn from_bits(bits: u8) -> Self {
        match bits & 3 {
            0 => Self::None,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            _ => Self::Gradient,
        }
    }
}

/// The alpha plane, `width * height` bytes in raster order.
///
/// The pre-processing field only says the encoder reduced the levels; a
/// decoder may smooth them, and `dwebp` does not unless asked, so neither
/// does this one.
pub(crate) fn decode(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, CodecError> {
    let Some((&header, payload)) = data.split_first() else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "webp alpha header",
        });
    };
    let compression = header & 3;
    let filter = Filter::from_bits(header >> 2);
    let preprocessing = (header >> 4) & 3;
    let reserved = header >> 6;
    if compression > 1 || preprocessing > 1 || reserved != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "webp alpha header",
        });
    }
    let count = pixel_count(width, height)?;
    let mut plane = if compression == 0 {
        let Some(raw) = payload.get(..count) else {
            return Err(CodecError::Truncated {
                at_bit: 8,
                what: "webp raw alpha",
            });
        };
        raw.to_vec()
    } else {
        decode_stream(payload, width, height)?
            .into_iter()
            .map(|argb| (argb >> 8) as u8)
            .collect()
    };
    unfilter(&mut plane, width as usize, filter);
    Ok(plane)
}

fn gradient(left: u8, top: u8, top_left: u8) -> u8 {
    (i16::from(left) + i16::from(top) - i16::from(top_left)).clamp(0, 255) as u8
}

/// Undo the prediction in place. The first pixel is predicted from 0, the
/// rest of the first row from the left, and the rest of the first column
/// from above, whatever the filter.
pub(crate) fn unfilter(plane: &mut [u8], width: usize, filter: Filter) {
    if filter == Filter::None || width == 0 {
        return;
    }
    let mut left = 0u8;
    for value in &mut plane[..width] {
        *value = value.wrapping_add(left);
        left = *value;
    }
    for row in 1..plane.len() / width {
        let (done, rest) = plane.split_at_mut(row * width);
        let above = &done[(row - 1) * width..];
        let line = &mut rest[..width];
        line[0] = line[0].wrapping_add(above[0]);
        for x in 1..width {
            let prediction = match filter {
                Filter::Horizontal => line[x - 1],
                Filter::Vertical => above[x],
                _ => gradient(line[x - 1], above[x], above[x - 1]),
            };
            line[x] = line[x].wrapping_add(prediction);
        }
    }
}

/// Apply a prediction filter, the inverse of [`unfilter`].
#[cfg(test)]
pub(crate) fn filter(plane: &[u8], width: usize, filter: Filter) -> Vec<u8> {
    let mut out = plane.to_vec();
    if filter == Filter::None {
        return out;
    }
    for (index, value) in out.iter_mut().enumerate() {
        let (x, y) = (index % width, index / width);
        let prediction = if y == 0 {
            if x == 0 { 0 } else { plane[index - 1] }
        } else if x == 0 {
            plane[index - width]
        } else {
            match filter {
                Filter::Horizontal => plane[index - 1],
                Filter::Vertical => plane[index - width],
                _ => gradient(
                    plane[index - 1],
                    plane[index - width],
                    plane[index - width - 1],
                ),
            }
        };
        *value = value.wrapping_sub(prediction);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Filter, decode, filter, unfilter};

    #[test]
    fn every_filter_round_trips() {
        let plane: Vec<u8> = (0..7 * 5).map(|i| (i * 37 % 256) as u8).collect();
        for kind in [
            Filter::None,
            Filter::Horizontal,
            Filter::Vertical,
            Filter::Gradient,
        ] {
            let mut coded = filter(&plane, 7, kind);
            unfilter(&mut coded, 7, kind);
            assert_eq!(coded, plane, "{kind:?}");
        }
    }

    #[test]
    fn raw_alpha_shorter_than_the_picture_is_truncated() {
        assert!(decode(&[0, 1, 2, 3], 2, 2).is_err());
        assert_eq!(decode(&[0, 1, 2, 3, 4], 2, 2).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn reserved_bits_and_unknown_methods_are_refused() {
        assert!(decode(&[0x40, 1], 1, 1).is_err());
        assert!(decode(&[0x02, 1], 1, 1).is_err());
        assert!(decode(&[0x20, 1], 1, 1).is_err());
        assert!(decode(&[], 1, 1).is_err());
    }
}
