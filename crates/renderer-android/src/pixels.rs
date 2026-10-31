//! RGBA over draws on the portable device.

use cutlass_gpu::{Blend, ClearColor, Device, Extent, TexelSource};

use super::{AndroidDevice, AndroidError};

fn extent(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

fn spec(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = 255u32 - u32::from(source_alpha);
    let mixed = u32::from(source) + u32::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(0)
}

fn source_pixel(x: u32, y: u32) -> [u8; 4] {
    [
        u8::try_from(x).unwrap(),
        u8::try_from(y).unwrap(),
        u8::try_from(x + y).unwrap(),
        u8::try_from(x * 3 + y).unwrap(),
    ]
}

fn draw_pixel(onto: [u8; 4], source: [u8; 4]) -> [u8; 4] {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device.clear(pass, ClearColor { rgba: onto }).unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &source,
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    let mut out = [0; 4];
    out.copy_from_slice(device.read_rgba(framebuffer).unwrap());
    out
}

#[test]
fn an_opaque_pixel_replaces_the_framebuffer() {
    assert_eq!(
        draw_pixel([1, 2, 3, 4], [10, 20, 30, 255]),
        [10, 20, 30, 255]
    );
}

#[test]
fn over_blends_each_channel_with_truncated_alpha() {
    // inverse of 128 is 127, and 200 * 127 / 255 = 99.
    assert_eq!(
        draw_pixel([0, 0, 200, 255], [100, 0, 0, 128]),
        [100, 0, 99, 255]
    );
}

#[test]
fn a_zero_alpha_source_is_added_to_the_destination() {
    assert_eq!(draw_pixel([1, 2, 3, 4], [10, 20, 30, 0]), [11, 22, 33, 4]);
}

#[test]
fn a_sum_above_255_is_clamped() {
    assert_eq!(
        draw_pixel([255, 255, 255, 255], [255, 255, 255, 0]),
        [255, 255, 255, 255]
    );
}

#[test]
fn a_nearly_opaque_source_truncates_a_one() {
    assert_eq!(draw_pixel([0, 0, 1, 0], [0, 0, 0, 254]), [0, 0, 0, 254]);
}

#[test]
fn a_two_pixel_row_blends_the_first_and_keeps_the_second() {
    let mut device = AndroidDevice::new();
    let size = extent(2, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    // A clear is one color, so plant [1, 2, 3, 255, 4, 5, 6, 255] first.
    device
        .clear(
            pass,
            ClearColor {
                rgba: [4, 5, 6, 255],
            },
        )
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[1, 2, 3, 255, 0, 0, 0, 0],
            },
        )
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[10, 0, 0, 255, 0, 0, 0, 0],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap(),
        &[10, 0, 0, 255, 4, 5, 6, 255]
    );
}

#[test]
fn two_draws_in_one_pass_blend_in_order() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [0, 0, 0, 0] })
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[10, 20, 30, 128],
            },
        )
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[1, 2, 3, 255],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 2, 3, 255]);
}

#[test]
fn draw_without_setting_blend_uses_over() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[10, 20, 30, 255],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[10, 20, 30, 255]);
}

#[test]
fn platform_tokens_are_refused_and_the_clear_stays() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
        .unwrap();
    for token in [0, u64::MAX] {
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Platform {
                        extent: size,
                        token,
                    },
                )
                .unwrap_err(),
            AndroidError::PlatformBuffer
        );
    }
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
}

#[test]
fn a_short_slice_is_bytes_and_the_clear_stays() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
        .unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[1, 2, 3],
                },
            )
            .unwrap_err(),
        AndroidError::Bytes
    );
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
}

#[test]
fn a_wider_extent_is_size_and_the_clear_stays() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(extent(1, 1)).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
        .unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: extent(2, 1),
                    bytes: &[0, 0, 0, 0, 0, 0, 0, 0],
                },
            )
            .unwrap_err(),
        AndroidError::Size
    );
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
}

#[test]
fn a_wrong_extent_and_a_short_slice_is_size() {
    let mut device = AndroidDevice::new();
    let framebuffer = device.create_framebuffer(extent(1, 1)).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
        .unwrap();
    // The extent is wrong and the slice is short. Size wins over Bytes.
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: extent(2, 1),
                    bytes: &[1, 2, 3],
                },
            )
            .unwrap_err(),
        AndroidError::Size
    );
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
}

#[test]
fn a_valid_draw_after_bytes_blends_onto_the_clear() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
        .unwrap();
    assert_eq!(
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: size,
                    bytes: &[1, 2, 3],
                },
            )
            .unwrap_err(),
        AndroidError::Bytes
    );
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[10, 20, 30, 255],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), &[10, 20, 30, 255]);
}

#[test]
fn an_empty_frame_draws_an_empty_slice() {
    let mut device = AndroidDevice::new();
    let size = extent(0, 0);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    let empty: &[u8] = &[];
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: empty,
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(device.read_rgba(framebuffer).unwrap(), empty);
}

#[test]
fn every_grid_byte_matches_the_spec() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 8;
    let width = usize::try_from(WIDTH).unwrap();
    let height = usize::try_from(HEIGHT).unwrap();
    let mut device = AndroidDevice::new();
    let size = extent(WIDTH, HEIGHT);
    let framebuffer = device.create_framebuffer(size).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(
            pass,
            ClearColor {
                rgba: [0, 0, 0, 255],
            },
        )
        .unwrap();
    let mut bytes = Vec::with_capacity(width * height * 4);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            bytes.extend(source_pixel(x, y));
        }
    }
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &bytes,
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    let read = device.read_rgba(framebuffer).unwrap();
    let (pixels, rest) = read.as_chunks::<4>();
    assert!(rest.is_empty());
    assert_eq!(pixels.len(), width * height);
    let destination = [0, 0, 0, 255];
    for (index, pixel) in pixels.iter().enumerate() {
        let x = u32::try_from(index % width).unwrap();
        let y = u32::try_from(index / width).unwrap();
        let source = source_pixel(x, y);
        let alpha = source[3];
        let expected = [
            spec(source[0], destination[0], alpha),
            spec(source[1], destination[1], alpha),
            spec(source[2], destination[2], alpha),
            spec(source[3], destination[3], alpha),
        ];
        assert_eq!(*pixel, expected, "pixel {x},{y}");
    }
}

#[test]
fn an_opaque_red_draw_leaves_the_other_framebuffer() {
    let mut device = AndroidDevice::new();
    let size = extent(1, 1);
    let first = device.create_framebuffer(size).unwrap();
    let second = device.create_framebuffer(size).unwrap();

    let pass = device.begin(second).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(
            pass,
            ClearColor {
                rgba: [0, 0, 255, 255],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();

    let pass = device.begin(first).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [0, 0, 0, 0] })
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent: size,
                bytes: &[255, 0, 0, 255],
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();

    assert_eq!(device.read_rgba(first).unwrap(), &[255, 0, 0, 255]);
    assert_eq!(device.read_rgba(second).unwrap(), &[0, 0, 255, 255]);
}
