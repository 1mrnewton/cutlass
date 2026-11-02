//! Pixel oracles for the CPU over-blend, read back through `WindowsDevice`.

use cutlass_gpu::{Blend, ClearColor, Device, Extent, TexelSource};

use crate::blend::{blend_over, fill_rgba};
use crate::{WindowsDevice, WindowsError};

fn draw_pixel(onto: [u8; 4], source: [u8; 4]) -> Result<[u8; 4], WindowsError> {
    let mut device = WindowsDevice::new();
    let extent = Extent {
        width: 1,
        height: 1,
    };
    let framebuffer = device.create_framebuffer(extent)?;
    let pass = device.begin(framebuffer)?;
    device.blend(pass, Blend::Over)?;
    device.clear(pass, ClearColor { rgba: onto })?;
    device.draw(
        pass,
        TexelSource::Rgba {
            extent,
            bytes: &source,
        },
    )?;
    device.end(pass)?;
    device.submit(pass)?;
    let mut out = [0; 4];
    out.copy_from_slice(device.read_rgba(framebuffer)?);
    Ok(out)
}

#[test]
fn an_opaque_pixel_replaces_the_framebuffer() {
    assert_eq!(
        draw_pixel([1, 2, 3, 4], [10, 20, 30, 255]).unwrap(),
        [10, 20, 30, 255]
    );
}

#[test]
fn over_blends_each_channel_with_truncated_alpha() {
    // source alpha 128 → inverse 127. 200 * 127 / 255 = 99.
    // alpha 128 + 255 * 127 / 255 = 255.
    assert_eq!(
        draw_pixel([0, 0, 200, 255], [100, 0, 0, 128]).unwrap(),
        [100, 0, 99, 255]
    );
}

#[test]
fn zero_alpha_adds_the_source_onto_the_destination() {
    assert_eq!(
        draw_pixel([1, 2, 3, 4], [10, 20, 30, 0]).unwrap(),
        [11, 22, 33, 4]
    );
}

#[test]
fn a_sum_past_255_is_clamped() {
    // 255 + 255 * 255 / 255 = 510, clamped to 255.
    assert_eq!(
        draw_pixel([255, 255, 255, 255], [255, 255, 255, 0]).unwrap(),
        [255, 255, 255, 255]
    );
}

#[test]
fn a_low_alpha_truncates_toward_zero() {
    // source alpha 1 → inverse 254. 255 * 254 / 255 = 254.
    assert_eq!(
        draw_pixel([0, 0, 255, 255], [0, 0, 0, 1]).unwrap(),
        [0, 0, 254, 255]
    );
}

#[test]
fn two_pixels_blend_in_row_order() {
    // A zero destination copies the drawn bytes, so the first draw is the
    // destination image. The second draw blends over it.
    let mut device = WindowsDevice::new();
    let extent = Extent {
        width: 2,
        height: 1,
    };
    let framebuffer = device.create_framebuffer(extent).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [0, 0, 0, 0] })
        .unwrap();
    let destination = [10, 0, 0, 255, 0, 10, 0, 128];
    let source = [0, 0, 10, 255, 0, 0, 10, 0];
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent,
                bytes: &destination,
            },
        )
        .unwrap();
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent,
                bytes: &source,
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap(),
        &[0, 0, 10, 255, 0, 10, 10, 128]
    );
}

#[test]
fn an_opaque_red_replaces_a_cleared_2x2() {
    let mut device = WindowsDevice::new();
    let extent = Extent {
        width: 2,
        height: 2,
    };
    let framebuffer = device.create_framebuffer(extent).unwrap();
    let pass = device.begin(framebuffer).unwrap();
    device.blend(pass, Blend::Over).unwrap();
    device
        .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
        .unwrap();
    let source = [
        255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
    ];
    device
        .draw(
            pass,
            TexelSource::Rgba {
                extent,
                bytes: &source,
            },
        )
        .unwrap();
    device.end(pass).unwrap();
    device.submit(pass).unwrap();
    assert_eq!(
        device.read_rgba(framebuffer).unwrap(),
        &[
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]
    );
}

#[test]
fn blend_over_matches_the_oracle_without_allocating() {
    // The pixels are stack arrays. `blend_over` only writes the destination.
    fn over(destination: [u8; 4], source: [u8; 4]) -> [u8; 4] {
        let mut destination = destination;
        blend_over(&mut destination, &source);
        destination
    }

    assert_eq!(over([1, 2, 3, 4], [10, 20, 30, 255]), [10, 20, 30, 255]);
    assert_eq!(over([0, 0, 200, 255], [100, 0, 0, 128]), [100, 0, 99, 255]);
    assert_eq!(over([1, 2, 3, 4], [10, 20, 30, 0]), [11, 22, 33, 4]);
    assert_eq!(
        over([255, 255, 255, 255], [255, 255, 255, 0]),
        [255, 255, 255, 255]
    );
    assert_eq!(over([0, 0, 255, 255], [0, 0, 0, 1]), [0, 0, 254, 255]);
}

#[test]
fn blend_over_blends_two_stack_pixels() {
    let mut destination = [10, 0, 0, 255, 0, 10, 0, 128];
    let source = [0, 0, 10, 255, 0, 0, 10, 0];
    blend_over(&mut destination, &source);
    assert_eq!(destination, [0, 0, 10, 255, 0, 10, 10, 128]);
}

#[test]
fn fill_rgba_leaves_a_trailing_byte() {
    let mut destination = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    fill_rgba(&mut destination, [10, 20, 30, 40]);
    assert_eq!(destination, [10, 20, 30, 40, 10, 20, 30, 40, 9]);

    let mut short = [1, 2, 3];
    fill_rgba(&mut short, [10, 20, 30, 40]);
    assert_eq!(short, [1, 2, 3]);
}

#[test]
fn blend_over_stops_at_the_shorter_slice() {
    let mut destination = [1, 2, 3, 4, 5];
    blend_over(&mut destination, &[10, 20, 30, 255, 9, 9, 9, 9]);
    assert_eq!(destination, [10, 20, 30, 255, 5]);

    let mut short = [1, 2, 3];
    blend_over(&mut short, &[10, 20, 30, 255]);
    assert_eq!(short, [1, 2, 3]);
}
