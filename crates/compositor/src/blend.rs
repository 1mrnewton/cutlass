use cutlass_gpu::Blend;

/// The pass blend. `output = source + destination * (1 - source.alpha)`.
#[must_use]
pub(crate) const fn pass_blend() -> Blend {
    Blend::Over
}
