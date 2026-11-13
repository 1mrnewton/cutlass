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
