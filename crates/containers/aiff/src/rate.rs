//! 80-bit extended sample rate.
//!
//! The field is a sign bit, a 15-bit exponent biased by 16383, and a 64-bit
//! integer significand. This crate keeps a rate in `1..=u32::MAX` as an
//! integer: a fraction, such as the 22254.5454 Hz of early Macintosh
//! sound, is rounded to the nearest hertz, a half going up, which is what
//! libavformat does. A negative and a non-finite value are errors.

use std::fmt;

/// A sample rate this crate will not store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RateError {
    Zero,
    Negative,
    NotFinite,
    OutOfRange,
}

impl fmt::Display for RateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Zero => "sample rate is zero",
            Self::Negative => "sample rate is negative",
            Self::NotFinite => "sample rate is not finite",
            Self::OutOfRange => "sample rate does not fit",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RateError {}

pub(crate) fn encode_rate(rate: u32) -> Result<[u8; 10], RateError> {
    if rate == 0 {
        return Err(RateError::Zero);
    }
    let power = 31 - rate.leading_zeros();
    let exp = 16383 + power;
    // `power` is at most 31, so this shift stays inside a u64.
    let shift = 63 - power;
    let mantissa = u64::from(rate) << shift;
    let exp_bytes = u16::try_from(exp)
        .map_err(|_| RateError::OutOfRange)?
        .to_be_bytes();
    let mantissa_bytes = mantissa.to_be_bytes();
    let mut out = [0u8; 10];
    out[0] = exp_bytes[0];
    out[1] = exp_bytes[1];
    out[2..].copy_from_slice(&mantissa_bytes);
    Ok(out)
}

pub(crate) fn decode_rate(bytes: &[u8; 10]) -> Result<u32, RateError> {
    let exp_bits = u16::from_be_bytes([bytes[0], bytes[1]]);
    if exp_bits & 0x8000 != 0 {
        return Err(RateError::Negative);
    }
    let exp = u32::from(exp_bits & 0x7FFF);
    if exp == 0 || exp == 0x7FFF {
        return Err(RateError::NotFinite);
    }
    if exp > 16383 + 63 {
        return Err(RateError::OutOfRange);
    }
    let mantissa = u64::from_be_bytes([
        bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
    ]);
    // The value is mantissa / 2^shift. Below a half it rounds to zero.
    let shift = 16383 + 63 - exp;
    if shift > 64 {
        return Err(RateError::Zero);
    }
    let value = if shift == 0 {
        u128::from(mantissa)
    } else {
        (u128::from(mantissa) + (1u128 << (shift - 1))) >> shift
    };
    let rate = u32::try_from(value).map_err(|_| RateError::OutOfRange)?;
    if rate == 0 {
        return Err(RateError::Zero);
    }
    Ok(rate)
}

#[cfg(test)]
mod tests {
    use super::{RateError, decode_rate, encode_rate};

    #[test]
    fn common_rates_use_the_extended_bytes() {
        assert_eq!(
            encode_rate(44_100).expect("44100"),
            [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_rate(48_000).expect("48000"),
            [0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_rate(8_000).expect("8000"),
            [0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn an_integer_rate_round_trips() {
        for rate in [
            1,
            3,
            11_025,
            16_000,
            22_050,
            32_000,
            88_200,
            96_000,
            192_000,
            u32::MAX,
        ] {
            let bytes = encode_rate(rate).expect("encode");
            assert_eq!(decode_rate(&bytes).expect("decode"), rate);
        }
    }

    /// 244800 / 11 = 22254.5454... Hz is the rate of the first Macintosh
    /// sound hardware, and system sounds of that time are stored with it.
    #[test]
    fn a_fraction_rounds_to_the_nearest_hertz() {
        let mac = [0x40, 0x0D, 0xAD, 0xDD, 0x17, 0x45, 0xD1, 0x74, 0x5D, 0x17];
        assert_eq!(decode_rate(&mac), Ok(22_255));
        // 11127.2727 Hz, half of it.
        let half = [0x40, 0x0C, 0xAD, 0xDD, 0x17, 0x45, 0xD1, 0x74, 0x5D, 0x17];
        assert_eq!(decode_rate(&half), Ok(11_127));
        // 44100 and one unit in the last place stays 44100; 44100.5 goes up.
        let mut just_over = encode_rate(44_100).expect("rate");
        just_over[9] = 1;
        assert_eq!(decode_rate(&just_over), Ok(44_100));
        let mut tie = encode_rate(44_100).expect("rate");
        // 44100 is 0xAC44 in the top 16 bits; the next bit is one half.
        tie[4] = 0x80;
        assert_eq!(decode_rate(&tie), Ok(44_101));
        // 0.75 Hz rounds to 1, 0.25 Hz to nothing.
        assert_eq!(decode_rate(&[0x3F, 0xFE, 0xC0, 0, 0, 0, 0, 0, 0, 0]), Ok(1));
        assert_eq!(
            decode_rate(&[0x3F, 0xFD, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            Err(RateError::Zero)
        );
        assert_eq!(
            decode_rate(&[0x00, 0x01, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            Err(RateError::Zero)
        );
        // 2^32 does not fit, and neither does a rate that rounds up to it.
        assert_eq!(
            decode_rate(&[0x40, 0x1F, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            Err(RateError::OutOfRange)
        );
        assert_eq!(
            decode_rate(&[0x40, 0x1E, 0xFF, 0xFF, 0xFF, 0xFF, 0x80, 0, 0, 0]),
            Err(RateError::OutOfRange)
        );
        assert_eq!(
            decode_rate(&[0x7F, 0xFE, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            Err(RateError::OutOfRange)
        );
    }

    #[test]
    fn a_negative_and_zero_are_refused() {
        let mut negative = encode_rate(44_100).expect("rate");
        negative[0] |= 0x80;
        assert_eq!(decode_rate(&negative), Err(RateError::Negative));
        assert_eq!(encode_rate(0), Err(RateError::Zero));
        assert_eq!(decode_rate(&[0; 10]), Err(RateError::NotFinite));
    }
}
