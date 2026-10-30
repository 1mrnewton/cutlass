//! RGBA over-blend on tightly packed bytes.

/// Writes `pixel` into every complete 4-byte group. A shorter tail is left unchanged.
pub(crate) fn fill_rgba(destination: &mut [u8], pixel: [u8; 4]) {
    let (chunks, _) = destination.as_chunks_mut::<4>();
    chunks.fill(pixel);
}

/// Each channel, including alpha, is `source + destination * (1 - source.alpha)`,
/// where source alpha is byte 3 divided by 255 with the remainder truncated and a
/// channel above 255 is clamped.
/// The source color is not multiplied by alpha first.
pub(crate) fn blend_over(destination: &mut [u8], source: &[u8]) {
    let (destination_pixels, _) = destination.as_chunks_mut::<4>();
    let (source_pixels, _) = source.as_chunks::<4>();
    for (dst, src) in destination_pixels.iter_mut().zip(source_pixels) {
        let alpha = src[3];
        *dst = [
            blend_channel(src[0], dst[0], alpha),
            blend_channel(src[1], dst[1], alpha),
            blend_channel(src[2], dst[2], alpha),
            blend_channel(src[3], dst[3], alpha),
        ];
    }
}

fn blend_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = 255 - u16::from(source_alpha);
    let mixed = u16::from(source) + u16::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(255)
}

#[cfg(test)]
mod tests {
    use super::{blend_over, fill_rgba};
    use crate::framebuffer::MetalDevice;
    use cutlass_gpu::{Blend, ClearColor, Device, Extent, TexelSource};

    fn draw_over(width: u32, height: u32, onto: [u8; 4], source: &[u8]) -> Vec<u8> {
        let mut device = MetalDevice::new();
        let extent = Extent { width, height };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device.blend(pass, Blend::Over).unwrap();
        device.clear(pass, ClearColor { rgba: onto }).unwrap();
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent,
                    bytes: source,
                },
            )
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        device.read_rgba(framebuffer).unwrap().to_vec()
    }

    fn assert_over(onto: [u8; 4], source: [u8; 4], expected: [u8; 4]) {
        let mut pure = onto;
        blend_over(&mut pure, &source);
        let drawn = draw_over(1, 1, onto, &source);
        assert_eq!(&pure[..], &drawn[..]);
        assert_eq!(&drawn[..], &expected[..]);
    }

    #[test]
    fn opaque_source_replaces_every_channel() {
        assert_over([1, 2, 3, 4], [10, 20, 30, 255], [10, 20, 30, 255]);
    }

    #[test]
    fn alpha_128_truncates_the_destination_product() {
        // inverse 127; 200 * 127 / 255 = 99; alpha 128 + 255 * 127 / 255 = 255.
        assert_over([0, 0, 200, 255], [100, 0, 0, 128], [100, 0, 99, 255]);
    }

    #[test]
    fn alpha_0_adds_each_channel() {
        assert_over([1, 2, 3, 4], [10, 20, 30, 0], [11, 22, 33, 4]);
    }

    #[test]
    fn a_channel_past_255_clamps() {
        // 200 + 200 * 255 / 255 = 400, clamped to 255. Alpha 0 + 255 = 255.
        assert_over([200, 10, 10, 255], [200, 10, 10, 0], [255, 20, 20, 255]);
    }

    #[test]
    fn alpha_1_truncates_255_times_254() {
        // inverse 254; 255 * 254 / 255 = 254; alpha 1 + 254 = 255.
        assert_over([255, 255, 255, 255], [0, 0, 0, 1], [254, 254, 254, 255]);
    }

    #[test]
    fn alpha_255_is_the_source_pixel() {
        assert_over([9, 8, 7, 6], [4, 3, 2, 255], [4, 3, 2, 255]);
    }

    #[test]
    fn alpha_254_adds_one_destination_step() {
        // inverse 1; 10 + 255 * 1 / 255 = 11; alpha 254 + 0 * 1 / 255 = 254.
        assert_over([255, 0, 0, 0], [10, 0, 0, 254], [11, 0, 0, 254]);
    }

    #[test]
    fn two_by_two_pixels_do_not_bleed() {
        // pixel2 inverse 127: 1*127/255=0, 2*127/255=0, 3*127/255=1, 4*127/255=1 → [0, 0, 31, 129]
        // pixel3 inverse 255: 255+1*255/255=256→255, 255+2→255, 255+3→255; alpha 0+4*255/255=4
        let onto = [1, 2, 3, 4];
        let source = [10, 0, 0, 255, 0, 20, 0, 0, 0, 0, 30, 128, 255, 255, 255, 0];
        let mut pure = [0; 16];
        fill_rgba(&mut pure, onto);
        blend_over(&mut pure, &source);
        let drawn = draw_over(2, 2, onto, &source);
        assert_eq!(source.len(), 16);
        assert_eq!(drawn.len(), 16);
        assert_eq!(&pure[..], &drawn[..]);
        assert_eq!(&drawn[0..4], &[10, 0, 0, 255]);
        assert_eq!(&drawn[4..8], &[1, 22, 3, 4]);
        assert_eq!(&drawn[8..12], &[0, 0, 31, 129]);
        assert_eq!(&drawn[12..16], &[255, 255, 255, 4]);
    }

    #[test]
    fn empty_slices_stay_empty() {
        let mut destination = [0_u8; 0];
        fill_rgba(&mut destination, [1, 2, 3, 4]);
        assert_eq!(destination, [0_u8; 0]);
        blend_over(&mut destination, &[]);
        assert_eq!(destination, [0_u8; 0]);
        blend_over(&mut destination, &[10, 20, 30, 255]);
        assert_eq!(destination, [0_u8; 0]);
    }

    #[test]
    fn fill_rgba_keeps_a_two_byte_tail() {
        let mut six = [9, 9, 9, 9, 7, 8];
        fill_rgba(&mut six, [1, 2, 3, 4]);
        assert_eq!(six, [1, 2, 3, 4, 7, 8]);

        let mut three = [7, 8, 9];
        fill_rgba(&mut three, [1, 2, 3, 4]);
        assert_eq!(three, [7, 8, 9]);
    }

    #[test]
    fn blend_over_changes_only_the_first_of_eight_bytes() {
        let mut destination = [1, 2, 3, 4, 5, 6, 7, 8];
        blend_over(&mut destination, &[10, 20, 30, 255]);
        assert_eq!(destination, [10, 20, 30, 255, 5, 6, 7, 8]);

        let mut short = [1, 2, 3, 4];
        blend_over(&mut short, &[10, 20, 30, 0, 9, 9, 9, 9]);
        assert_eq!(short, [11, 22, 33, 4]);
    }
}
