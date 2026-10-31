//! Portable RGBA source-over.

/// Blend `source` over `destination`.
///
/// Each channel, including alpha, is
/// `output = source + destination * (1 - source.alpha)`.
/// The 8-bit alpha is divided by 255 and the remainder is truncated.
/// A channel above 255 is clamped to 255. The bytes are not premultiplied
/// first. Whole pixels are zipped, and a trailing partial pixel is ignored.
pub(crate) fn blend_over(destination: &mut [u8], source: &[u8]) {
    let (dest_pixels, _) = destination.as_chunks_mut::<4>();
    let (source_pixels, _) = source.as_chunks::<4>();
    for (dst, src) in dest_pixels.iter_mut().zip(source_pixels.iter()) {
        let alpha = src[3];
        let mut mixed = [0; 4];
        for channel in 0..4 {
            mixed[channel] = blend_channel(src[channel], dst[channel], alpha);
        }
        *dst = mixed;
    }
}

fn blend_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = 255 - source_alpha;
    let mixed = u16::from(source) + u16::from(destination) * u16::from(inverse) / 255;
    let clamped = mixed.min(255);
    u8::try_from(clamped).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::blend_over;

    fn spec(source: u8, destination: u8, source_alpha: u8) -> u8 {
        let inverse = 255u32 - u32::from(source_alpha);
        let mixed = u32::from(source) + u32::from(destination) * inverse / 255;
        u8::try_from(mixed.min(255)).unwrap_or(0)
    }

    #[test]
    fn empty_slices_stay_empty() {
        let mut destination = [0u8; 0];
        blend_over(&mut destination, &[0u8; 0]);
        assert_eq!(destination, [0u8; 0]);
    }

    #[test]
    fn draw_vectors_match_the_spec_on_each_channel() {
        let cases = [
            ([1, 2, 3, 4], [10, 20, 30, 255]),
            ([0, 0, 200, 255], [100, 0, 0, 128]),
            ([1, 2, 3, 4], [10, 20, 30, 0]),
            ([255, 255, 255, 255], [255, 255, 255, 0]),
            ([0, 0, 1, 0], [0, 0, 0, 254]),
        ];
        for (onto, source) in cases {
            let mut destination = onto;
            blend_over(&mut destination, &source);
            let alpha = source[3];
            for ((got, src), dst) in destination.iter().zip(source).zip(onto) {
                assert_eq!(*got, spec(src, dst, alpha));
            }
        }
    }

    #[test]
    fn a_partial_tail_is_left_unchanged() {
        let source = [10, 20, 30, 128, 7, 6];
        let original = [1, 2, 3, 4, 9, 8];
        let mut destination = original;
        blend_over(&mut destination, &source);
        let alpha = source[3];
        for ((got, src), dst) in destination.iter().zip(source).zip(original).take(4) {
            assert_eq!(*got, spec(src, dst, alpha));
        }
        assert_eq!(&destination[4..], &original[4..]);
    }
}
