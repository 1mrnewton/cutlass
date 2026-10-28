//! Compare two file-timescale instants without converting either clock.
//!
//! The ordering is `left_ticks / left_scale` against `right_ticks / right_scale`,
//! computed with a cross product. It is not a conversion to microseconds.

use cutlass_codec::Timescale;
use std::cmp::Ordering;
use std::fmt;

/// A comparison was asked of a clock that does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareError {
    /// A timescale was zero.
    Clockless,
}

impl fmt::Display for CompareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clockless => {
                formatter.write_str("cannot compare a timestamp that has no timescale")
            }
        }
    }
}

impl std::error::Error for CompareError {}

/// Orders two ticks that may use different file timescales.
///
/// Equal products compare equal. A zero timescale is [`CompareError::Clockless`].
pub fn compare_ticks(
    left_ticks: i64,
    left_scale: Timescale,
    right_ticks: i64,
    right_scale: Timescale,
) -> Result<Ordering, CompareError> {
    let left_per_second = left_scale.ticks_per_second();
    let right_per_second = right_scale.ticks_per_second();
    if left_per_second == 0 || right_per_second == 0 {
        return Err(CompareError::Clockless);
    }
    let left = i128::from(left_ticks) * i128::from(right_per_second);
    let right = i128::from(right_ticks) * i128::from(left_per_second);
    Ok(left.cmp(&right))
}

#[cfg(test)]
mod tests {
    use super::{CompareError, compare_ticks};
    use cutlass_codec::Timescale;
    use std::cmp::Ordering;

    #[test]
    fn comparison_does_not_truncate_to_microseconds() {
        let order = compare_ticks(1, Timescale::new(3), 333_333, Timescale::new(1_000_000))
            .expect("clocks");
        assert_eq!(order, Ordering::Greater);
        let truncated = 1_000_000_i64 / 3;
        assert_eq!(truncated, 333_333);
    }

    #[test]
    fn equal_instants_on_different_clocks_compare_equal() {
        let order = compare_ticks(1, Timescale::new(2), 2, Timescale::new(4)).expect("clocks");
        assert_eq!(order, Ordering::Equal);
    }

    #[test]
    fn a_zero_timescale_is_not_compared() {
        assert_eq!(
            compare_ticks(1, Timescale::new(0), 1, Timescale::new(1)),
            Err(CompareError::Clockless)
        );
    }

    #[test]
    fn extreme_ticks_do_not_overflow() {
        let order = compare_ticks(
            i64::MAX,
            Timescale::new(u32::MAX),
            i64::MIN,
            Timescale::new(1),
        )
        .expect("fits in i128");
        assert_eq!(order, Ordering::Greater);
    }
}
