//! EXIF orientation, applied to stored samples so the picture is upright.
//!
//! ImageIO reports this tag on JPEG, PNG, WebP, HEIC, and AVIF. A missing
//! tag or a value outside 1 through 8 stays upright. The stored width and
//! height are the buffer that was decoded. Orientations 5 through 8 swap
//! those edges.

use cutlass_decode::CodecError;

/// Tag value 1 through 8. Anything else is stored as upright.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation(u8);

impl Orientation {
    /// A missing tag is upright.
    #[must_use]
    pub const fn upright() -> Self {
        Self(1)
    }

    /// Keeps 1 through 8. Any other value is upright.
    #[must_use]
    pub fn from_tag(value: u32) -> Self {
        if (1..=8).contains(&value) {
            Self(value as u8)
        } else {
            Self(1)
        }
    }

    #[must_use]
    pub const fn tag(self) -> u8 {
        self.0
    }

    /// The displayed picture swaps width and height.
    #[must_use]
    pub const fn swaps_edges(self) -> bool {
        matches!(self.0, 5 | 6 | 7 | 8)
    }

    /// Where the stored sample `(x, y)` is drawn.
    ///
    /// `width` and `height` are the stored buffer. The returned point is in
    /// the upright picture.
    #[must_use]
    pub fn place(self, x: u32, y: u32, width: u32, height: u32) -> (u32, u32) {
        match self.0 {
            2 => (width - 1 - x, y),
            3 => (width - 1 - x, height - 1 - y),
            4 => (x, height - 1 - y),
            5 => (y, x),
            6 => (height - 1 - y, x),
            7 => (height - 1 - y, width - 1 - x),
            8 => (y, width - 1 - x),
            _ => (x, y),
        }
    }
}

/// Stored samples moved into an upright buffer.
///
/// `channels` is the byte count of one sample. The stored length must be
/// `width * height * channels`.
pub fn upright(
    orientation: Orientation,
    width: u32,
    height: u32,
    channels: usize,
    src: &[u8],
) -> Result<(u32, u32, Vec<u8>), CodecError> {
    if channels == 0 || channels > 8 {
        return Err(CodecError::InvalidApi {
            what: "still channels",
        });
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(CodecError::Limit {
            what: "picture pixels",
        })?;
    let bytes = pixels
        .checked_mul(channels as u64)
        .ok_or(CodecError::Limit {
            what: "picture bytes",
        })?;
    let bytes = usize::try_from(bytes).map_err(|_| CodecError::Limit {
        what: "picture bytes",
    })?;
    if src.len() != bytes {
        return Err(CodecError::Malformed {
            at_bit: u64::try_from(src.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(8),
            what: "oriented still",
        });
    }
    if orientation.tag() == 1 {
        return Ok((width, height, src.to_vec()));
    }
    let (out_w, out_h) = if orientation.swaps_edges() {
        (height, width)
    } else {
        (width, height)
    };
    let mut dst = vec![0u8; bytes];
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = orientation.place(x, y, width, height);
            let from = sample_offset(x, y, width, channels)?;
            let to = sample_offset(dx, dy, out_w, channels)?;
            dst[to..to + channels].copy_from_slice(&src[from..from + channels]);
        }
    }
    Ok((out_w, out_h, dst))
}

fn sample_offset(x: u32, y: u32, width: u32, channels: usize) -> Result<usize, CodecError> {
    let row = usize::try_from(y).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    let col = usize::try_from(x).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    let width = usize::try_from(width).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    row.checked_mul(width)
        .and_then(|pixels| pixels.checked_add(col))
        .and_then(|index| index.checked_mul(channels))
        .ok_or(CodecError::Limit {
            what: "picture bytes",
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_or_unknown_tag_stays_upright() {
        assert_eq!(Orientation::from_tag(0).tag(), 1);
        assert_eq!(Orientation::from_tag(9).tag(), 1);
        assert!(!Orientation::upright().swaps_edges());
    }

    #[test]
    fn orientations_five_through_eight_swap_the_edges() {
        for tag in 1..=4 {
            assert!(!Orientation::from_tag(tag).swaps_edges());
        }
        for tag in 5..=8 {
            assert!(Orientation::from_tag(tag).swaps_edges());
        }
    }

    #[test]
    fn the_stored_origin_lands_on_the_exif_corner() {
        let width = 4;
        let height = 3;
        let cases = [
            (1u32, (0, 0)),
            (2, (3, 0)),
            (3, (3, 2)),
            (4, (0, 2)),
            (5, (0, 0)),
            (6, (2, 0)),
            (7, (2, 3)),
            (8, (0, 3)),
        ];
        for (tag, expected) in cases {
            assert_eq!(
                Orientation::from_tag(tag).place(0, 0, width, height),
                expected,
                "tag {tag}"
            );
        }
    }

    #[test]
    fn rotate_right_four_times_is_the_original() {
        let mut x = 1u32;
        let mut y = 2u32;
        let mut width = 5u32;
        let mut height = 4u32;
        for _ in 0..4 {
            let (nx, ny) = Orientation::from_tag(6).place(x, y, width, height);
            let next_w = height;
            let next_h = width;
            x = nx;
            y = ny;
            width = next_w;
            height = next_h;
        }
        assert_eq!((x, y, width, height), (1, 2, 5, 4));
    }

    #[test]
    fn an_upright_buffer_copies_the_stored_samples() {
        let src = [1u8, 2, 3, 4];
        let (w, h, dst) = upright(Orientation::upright(), 1, 1, 4, &src).expect("copy");
        assert_eq!((w, h), (1, 1));
        assert_eq!(dst, src);
    }

    #[test]
    fn a_half_turn_exchanges_two_pixels() {
        let src = [9u8, 6];
        let (w, h, dst) = upright(Orientation::from_tag(3), 2, 1, 1, &src).expect("turn");
        assert_eq!((w, h), (2, 1));
        assert_eq!(dst, [6, 9]);
    }

    #[test]
    fn a_quarter_turn_swaps_the_edges_and_the_bytes() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let (w, h, dst) = upright(Orientation::from_tag(6), 3, 2, 1, &src).expect("quarter");
        assert_eq!((w, h), (2, 3));
        assert_eq!(dst.len(), src.len());
        assert_ne!(dst, src);
    }

    #[test]
    fn a_short_buffer_is_malformed() {
        let err = upright(Orientation::upright(), 2, 2, 4, &[0, 1, 2]).expect_err("short");
        assert!(matches!(
            err,
            CodecError::Malformed {
                what: "oriented still",
                ..
            }
        ));
    }
}
