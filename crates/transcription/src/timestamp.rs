/// Millisecond span of frames `start` through `end`, inclusive.
///
/// Each frame is [`crate::features::HOP_MS`] long. The end is the start of
/// the frame after `end`.
pub fn span_ms(start: usize, end: usize, hop_ms: i64) -> Result<(i64, i64), TimestampError> {
    if end < start || hop_ms <= 0 {
        return Err(TimestampError::Order);
    }
    let start = i64::try_from(start).map_err(|_| TimestampError::Overflow)?;
    let end = i64::try_from(end).map_err(|_| TimestampError::Overflow)?;
    let start_ms = start.checked_mul(hop_ms).ok_or(TimestampError::Overflow)?;
    let end_ms = end
        .checked_add(1)
        .and_then(|frame| frame.checked_mul(hop_ms))
        .ok_or(TimestampError::Overflow)?;
    Ok((start_ms, end_ms))
}

/// Why a frame span did not become milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampError {
    /// The end frame is before the start, or the hop is not positive.
    Order,
    /// The product does not fit in an `i64`.
    Overflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_of_frames_covers_their_hops() {
        assert_eq!(span_ms(0, 2, 10).unwrap(), (0, 30));
    }

    #[test]
    fn a_backwards_span_is_refused() {
        assert_eq!(span_ms(3, 1, 10).unwrap_err(), TimestampError::Order);
    }
}
