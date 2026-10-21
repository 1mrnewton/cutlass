//! The eight project frame rates, and drop frame.

use crate::Error;

/// A frame rate the project can store. Nominal fps is the ruler's count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRate {
    /// 24000/1001. Nominal 24.
    Fps23976,
    /// 24/1.
    Fps24,
    /// 25/1.
    Fps25,
    /// 30000/1001. Nominal 30. Drop frame starts on.
    Fps2997,
    /// 30/1. The empty project's rate.
    Fps30,
    /// 50/1.
    Fps50,
    /// 60000/1001. Nominal 60. Drop frame starts on.
    Fps5994,
    /// 60/1.
    Fps60,
}

impl FrameRate {
    pub const fn numerator(self) -> u32 {
        match self {
            Self::Fps23976 => 24_000,
            Self::Fps24 => 24,
            Self::Fps25 => 25,
            Self::Fps2997 => 30_000,
            Self::Fps30 => 30,
            Self::Fps50 => 50,
            Self::Fps5994 => 60_000,
            Self::Fps60 => 60,
        }
    }

    pub const fn denominator(self) -> u32 {
        match self {
            Self::Fps23976 | Self::Fps2997 | Self::Fps5994 => 1001,
            Self::Fps24 | Self::Fps25 | Self::Fps30 | Self::Fps50 | Self::Fps60 => 1,
        }
    }

    pub const fn nominal_fps(self) -> u32 {
        match self {
            Self::Fps23976 | Self::Fps24 => 24,
            Self::Fps25 => 25,
            Self::Fps2997 | Self::Fps30 => 30,
            Self::Fps50 => 50,
            Self::Fps5994 | Self::Fps60 => 60,
        }
    }

    pub const fn allows_drop_frame(self) -> bool {
        matches!(self, Self::Fps2997 | Self::Fps5994)
    }

    /// 29.97 and 59.94 start with drop frame on. Every other rate starts off.
    pub const fn sequence(self) -> SequenceRate {
        SequenceRate {
            rate: self,
            drop_frame: self.allows_drop_frame(),
        }
    }

    pub fn from_parts(numerator: u32, denominator: u32) -> Result<Self, Error> {
        let rate = match (numerator, denominator) {
            (24_000, 1001) => Self::Fps23976,
            (24, 1) => Self::Fps24,
            (25, 1) => Self::Fps25,
            (30_000, 1001) => Self::Fps2997,
            (30, 1) => Self::Fps30,
            (50, 1) => Self::Fps50,
            (60_000, 1001) => Self::Fps5994,
            (60, 1) => Self::Fps60,
            _ => {
                return Err(Error::UnknownRate {
                    numerator,
                    denominator,
                });
            }
        };
        Ok(rate)
    }
}

/// A frame rate plus its drop-frame flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceRate {
    rate: FrameRate,
    drop_frame: bool,
}

impl SequenceRate {
    pub fn new(rate: FrameRate, drop_frame: bool) -> Result<Self, Error> {
        if drop_frame && !rate.allows_drop_frame() {
            return Err(Error::DropFrameRate);
        }
        Ok(Self { rate, drop_frame })
    }

    pub const fn rate(self) -> FrameRate {
        self.rate
    }

    pub const fn drop_frame(self) -> bool {
        self.drop_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATES: [(FrameRate, u32, u32, u32, bool); 8] = [
        (FrameRate::Fps23976, 24_000, 1001, 24, false),
        (FrameRate::Fps24, 24, 1, 24, false),
        (FrameRate::Fps25, 25, 1, 25, false),
        (FrameRate::Fps2997, 30_000, 1001, 30, true),
        (FrameRate::Fps30, 30, 1, 30, false),
        (FrameRate::Fps50, 50, 1, 50, false),
        (FrameRate::Fps5994, 60_000, 1001, 60, true),
        (FrameRate::Fps60, 60, 1, 60, false),
    ];

    #[test]
    fn the_eight_rates_are_the_closed_set() {
        for (rate, numerator, denominator, nominal, drop) in RATES {
            assert_eq!(rate.numerator(), numerator);
            assert_eq!(rate.denominator(), denominator);
            assert_eq!(rate.nominal_fps(), nominal);
            assert_eq!(rate.allows_drop_frame(), drop);
            assert_eq!(rate.sequence().drop_frame(), drop);
            assert_eq!(FrameRate::from_parts(numerator, denominator).unwrap(), rate);
        }
    }

    #[test]
    fn a_nearby_fraction_is_not_a_project_rate() {
        let error = FrameRate::from_parts(30_000, 1_000).unwrap_err();
        assert_eq!(
            error,
            Error::UnknownRate {
                numerator: 30_000,
                denominator: 1_000
            }
        );
    }

    #[test]
    fn drop_frame_can_turn_off_without_changing_the_rate() {
        let rate = SequenceRate::new(FrameRate::Fps2997, false).unwrap();
        assert_eq!(rate.rate(), FrameRate::Fps2997);
        assert!(!rate.drop_frame());
        let ntsc = SequenceRate::new(FrameRate::Fps5994, false).unwrap();
        assert!(!ntsc.drop_frame());
    }

    #[test]
    fn any_other_rate_forces_drop_frame_off() {
        for rate in [
            FrameRate::Fps23976,
            FrameRate::Fps24,
            FrameRate::Fps25,
            FrameRate::Fps30,
            FrameRate::Fps50,
            FrameRate::Fps60,
        ] {
            assert_eq!(
                SequenceRate::new(rate, true).unwrap_err(),
                Error::DropFrameRate
            );
        }
    }
}
