//! Intra prediction of 8-bit samples (AV1 specification 7.11.2).

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// The taps of one recursive filter mode (7.11.2.3): for each of the
/// eight outputs of a 4x2 block, the weights of its seven inputs (the
/// sample above-left, the four above and the two to the left).
pub type FilterTaps = [[i8; 7]; 8];

/// Recursive intra prediction (7.11.2.3) of the `w` x `h` block at the
/// start of `dst`, rows `stride` apart. `above` holds the sample above
/// and left of the block, then the `w` samples above it; `left` the `h`
/// samples to its left. `w` is a multiple of 4 and `h` of 2, at most 32.
///
/// The weights of an output sum to 16 and the negative ones to at least
/// -12, so its sum fits 16 bits.
#[inline]
pub fn filter_intra(
    dst: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    taps: &FilterTaps,
    above: &[u8],
    left: &[u8],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::filter_intra(dst, stride, w, h, taps, above, left);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::filter_intra(dst, stride, w, h, taps, above, left);
}
