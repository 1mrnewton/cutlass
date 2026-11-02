//! CPU RGBA blend for the Windows device.
//!
//! Each channel, including alpha, uses
//! `source + destination * (1 - source.alpha)`. The 8-bit alpha is divided
//! by 255 and the remainder is truncated. A channel above 255 is clamped.
//! The bytes are not premultiplied.

/// Writes `pixel` into every complete 4-byte chunk. A trailing partial pixel stays.
pub(crate) fn fill_rgba(destination: &mut [u8], pixel: [u8; 4]) {
    for chunk in destination.as_chunks_mut::<4>().0 {
        *chunk = pixel;
    }
}

/// `output = source + destination * (1 - source.alpha)` on each channel.
///
/// Alpha is the stored byte divided by 255, with the remainder truncated.
/// A channel past 255 is clamped. The bytes are not premultiplied first.
/// Complete pixels are zipped. The shorter slice ends the blend, and a
/// trailing partial pixel stays.
pub(crate) fn blend_over(destination: &mut [u8], source: &[u8]) {
    for (dst, src) in destination
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(source.as_chunks::<4>().0)
    {
        let alpha = src[3];
        let mut mixed = [0; 4];
        for channel in 0..4 {
            mixed[channel] = blend_channel(src[channel], dst[channel], alpha);
        }
        *dst = mixed;
    }
}

fn blend_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = 255_u16 - u16::from(source_alpha);
    let mixed = u16::from(source) + u16::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(255)
}
