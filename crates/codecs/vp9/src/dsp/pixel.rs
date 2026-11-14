//! Samples of 8 bits (`u8`) and of 10 or 12 bits (`u16`).

pub trait Pixel: Copy + Default + PartialEq + Send + Sync + core::fmt::Debug + 'static {
    /// Whether samples are wider than 8 bits.
    const HIGH: bool;
    fn get(self) -> i32;
    /// `v` held to `0..=max`.
    fn clip(v: i32, max: i32) -> Self;
    /// `v`, which the caller has already held to the sample range.
    fn new(v: i32) -> Self;
}

impl Pixel for u8 {
    const HIGH: bool = false;
    #[inline(always)]
    fn get(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn clip(v: i32, _max: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u8
    }
}

impl Pixel for u16 {
    const HIGH: bool = true;
    #[inline(always)]
    fn get(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn clip(v: i32, max: i32) -> Self {
        v.clamp(0, max) as u16
    }
    #[inline(always)]
    fn new(v: i32) -> Self {
        v as u16
    }
}
