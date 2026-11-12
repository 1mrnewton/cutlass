//! The blend one sample at a time.

/// [`super::mask_blend8`], with the 16-bit wrapping it names.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn mask_blend8(a: &[i16; 8], b: &[i16; 8], m: &[u8; 8]) -> [u8; 8] {
    core::array::from_fn(|i| blend(a[i], b[i], m[i]))
}

fn blend(a: i16, b: i16, m: u8) -> u8 {
    let d = i32::from(a.wrapping_sub(b));
    let t = ((d * i32::from(m)) >> 6) as i16;
    let s = i32::from(b.wrapping_add(t));
    ((s + 8) >> 4).clamp(0, 255) as u8
}

/// [`super::weights_pairs8`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn weights_pairs8(m: &[u8; 16]) -> [u8; 8] {
    core::array::from_fn(|i| ((u16::from(m[2 * i]) + u16::from(m[2 * i + 1]) + 1) >> 1) as u8)
}

/// [`super::weights_quads8`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn weights_quads8(a: &[u8; 16], b: &[u8; 16]) -> [u8; 8] {
    core::array::from_fn(|i| {
        let s = u16::from(a[2 * i])
            + u16::from(a[2 * i + 1])
            + u16::from(b[2 * i])
            + u16::from(b[2 * i + 1]);
        ((s + 2) >> 2) as u8
    })
}
