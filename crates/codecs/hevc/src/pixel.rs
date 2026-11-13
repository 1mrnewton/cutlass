//! The two sample types of a picture: bytes for 8-bit streams, 16-bit
//! values for deeper ones. Every sample loop is written once over
//! [`Pixel`] and compiled for both.

use std::sync::Arc;

use cutlass_simd::shared_rows::SharedRows;

/// A sample of a picture plane.
pub trait Pixel: Copy + Default + Send + Sync + std::fmt::Debug + 'static {
    /// The sample as an integer.
    fn i(self) -> i32;
    /// A value already inside the sample range.
    fn of(v: i32) -> Self;
    /// The final rows of a picture of this sample type, if `rows` holds them.
    fn rows(rows: &AnyRows) -> Option<&Arc<SharedRows<Self>>>;
    /// The samples as bytes, for the 8-bit kernels.
    fn bytes(s: &[Self]) -> Option<&[u8]>;
    /// The samples as 16-bit words, for the kernels of deeper samples.
    fn words(s: &[Self]) -> Option<&[u16]>;
    fn bytes_mut(s: &mut [Self]) -> Option<&mut [u8]>;
    /// Bytes as samples, for 8-bit samples.
    fn of_bytes_mut(s: &mut [u8]) -> Option<&mut [Self]>;
}

impl Pixel for u8 {
    #[inline(always)]
    fn i(self) -> i32 {
        i32::from(self)
    }

    #[inline(always)]
    fn of(v: i32) -> Self {
        v as u8
    }

    fn rows(rows: &AnyRows) -> Option<&Arc<SharedRows<Self>>> {
        match rows {
            AnyRows::Eight(r) => Some(r),
            AnyRows::Deep(_) => None,
        }
    }

    #[inline(always)]
    fn bytes(s: &[Self]) -> Option<&[u8]> {
        Some(s)
    }

    #[inline(always)]
    fn words(_: &[Self]) -> Option<&[u16]> {
        None
    }

    #[inline(always)]
    fn bytes_mut(s: &mut [Self]) -> Option<&mut [u8]> {
        Some(s)
    }

    #[inline(always)]
    fn of_bytes_mut(s: &mut [u8]) -> Option<&mut [Self]> {
        Some(s)
    }
}

impl Pixel for u16 {
    #[inline(always)]
    fn i(self) -> i32 {
        i32::from(self)
    }

    #[inline(always)]
    fn of(v: i32) -> Self {
        v as u16
    }

    fn rows(rows: &AnyRows) -> Option<&Arc<SharedRows<Self>>> {
        match rows {
            AnyRows::Deep(r) => Some(r),
            AnyRows::Eight(_) => None,
        }
    }

    #[inline(always)]
    fn bytes(_: &[Self]) -> Option<&[u8]> {
        None
    }

    #[inline(always)]
    fn words(s: &[Self]) -> Option<&[u16]> {
        Some(s)
    }

    #[inline(always)]
    fn bytes_mut(_: &mut [Self]) -> Option<&mut [u8]> {
        None
    }

    #[inline(always)]
    fn of_bytes_mut(_: &mut [u8]) -> Option<&mut [Self]> {
        None
    }
}

/// The final rows of a picture of either sample type.
pub enum AnyRows {
    Eight(Arc<SharedRows<u8>>),
    Deep(Arc<SharedRows<u16>>),
}

impl AnyRows {
    /// Luma rows published so far.
    pub fn published(&self) -> usize {
        match self {
            Self::Eight(r) => r.published(),
            Self::Deep(r) => r.published(),
        }
    }
}
