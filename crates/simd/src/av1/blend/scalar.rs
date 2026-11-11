//! The blend one sample at a time.

/// [`super::mask_blend8`], with the 16-bit wrapping it names.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn mask_blend8(a: &[i16; 8], b: &[i16; 8], m: &[u8; 8]) -> [u8; 8] {
    core::array::from_fn(|i| {
        let d = i32::from(a[i].wrapping_sub(b[i]));
        let t = ((d * i32::from(m[i])) >> 6) as i16;
        let s = i32::from(b[i].wrapping_add(t));
        ((s + 8) >> 4).clamp(0, 255) as u8
    })
}
