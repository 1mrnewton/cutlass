//! Preview frame helpers: engine RGBA buffers → Slint images.

use cutlass_render::RgbaImage;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

pub fn to_slint_image(frame: RgbaImage) -> Image {
    let buffer =
        SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&frame.pixels, frame.width, frame.height);
    Image::from_rgba8(buffer)
}
