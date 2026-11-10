//! The constrained directional enhancement filter (AV1 specification 7.15).

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// Direction search on an 8x8 block of 8-bit luma (7.15.2). Returns
/// `(yDir, var)`.
#[must_use]
pub fn direction_u8(px: &[[u8; 8]; 8]) -> (u8, i32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::direction_u8(px)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::direction(&px.map(|r| r.map(u16::from)), 8)
}

/// Direction search on an 8x8 block of luma of `bd` bits, 8 to 12
/// (7.15.2). Returns `(yDir, var)`.
#[must_use]
pub fn direction_u16(px: &[[u16; 8]; 8], bd: u32) -> (u8, i32) {
    let bd = bd.clamp(8, 12);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::direction_u16(px, bd)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::direction(px, bd)
}
