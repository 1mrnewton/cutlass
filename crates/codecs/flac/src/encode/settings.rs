//! Encoder settings and the compression levels 0 to 8.

use cutlass_decode::CodecError;

/// How a stereo frame picks its channel assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stereo {
    /// Left and right, always.
    Independent,
    /// Estimates the four assignments from fixed-predictor residuals and
    /// codes the one that looks cheapest.
    Estimate,
    /// Codes left, right, mid, and side and keeps the cheapest pair.
    Exhaustive,
}

/// A window applied before the LPC analysis. Each window is one candidate
/// predictor per subframe; the cheapest is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Apodization {
    /// A Tukey window whose tapered part is `p` of the block, 0 to 1.
    Tukey(f64),
    /// `n` windows, each a Tukey window over one `n`th of the block and
    /// zero elsewhere.
    PartialTukey(u8),
    /// `n` windows, each a Tukey window over the block with one `n`th of it
    /// cut out.
    PunchoutTukey(u8),
}

/// What the encoder searches.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Samples per frame, 16 to 65535. Every frame but the last has it.
    pub block_size: u16,
    /// Highest LPC order, 0 to 32. Zero uses fixed predictors only.
    pub max_lpc_order: u8,
    /// Tries every LPC order up to the maximum instead of the one the
    /// prediction error suggests.
    pub exhaustive_order: bool,
    /// Rice partition orders searched, at most 15.
    pub min_partition_order: u8,
    pub max_partition_order: u8,
    pub stereo: Stereo,
    /// Windows for the LPC analysis. Empty with an LPC order is refused.
    pub apodizations: Vec<Apodization>,
}

/// The level `FlacEncoder::open` uses.
pub const DEFAULT_LEVEL: u8 = 5;

impl Settings {
    /// The settings of `level`, 0 (fastest) to 8 (smallest).
    ///
    /// The block sizes, LPC orders, partition orders, and stereo modes are
    /// those of the reference encoder's `-0` to `-8`. Levels 6 to 8 use a
    /// fixed set of partial and punch-out Tukey windows in place of the
    /// reference's `subdivide_tukey`.
    pub fn level(level: u8) -> Result<Self, CodecError> {
        use Apodization::{PartialTukey, PunchoutTukey, Tukey};
        let (block, lpc, partitions, stereo, windows): (u16, u8, u8, Stereo, Vec<Apodization>) =
            match level {
                0 => (1152, 0, 3, Stereo::Independent, Vec::new()),
                1 => (1152, 0, 3, Stereo::Estimate, Vec::new()),
                2 => (1152, 0, 3, Stereo::Exhaustive, Vec::new()),
                3 => (4096, 6, 4, Stereo::Independent, vec![Tukey(0.5)]),
                4 => (4096, 8, 4, Stereo::Estimate, vec![Tukey(0.5)]),
                5 => (4096, 8, 5, Stereo::Exhaustive, vec![Tukey(0.5)]),
                6 => (
                    4096,
                    8,
                    6,
                    Stereo::Exhaustive,
                    vec![Tukey(0.5), PartialTukey(2), PunchoutTukey(2)],
                ),
                7 => (
                    4096,
                    12,
                    6,
                    Stereo::Exhaustive,
                    vec![Tukey(0.5), PartialTukey(2), PunchoutTukey(2)],
                ),
                8 => (
                    4096,
                    12,
                    6,
                    Stereo::Exhaustive,
                    vec![
                        Tukey(0.5),
                        PartialTukey(2),
                        PartialTukey(3),
                        PunchoutTukey(3),
                    ],
                ),
                _ => return Err(CodecError::InvalidApi { what: "flac level" }),
            };
        Ok(Self {
            block_size: block,
            max_lpc_order: lpc,
            exhaustive_order: false,
            min_partition_order: 0,
            max_partition_order: partitions,
            stereo,
            apodizations: windows,
        })
    }

    /// Refuses settings the format cannot carry.
    pub fn validate(&self) -> Result<(), CodecError> {
        if self.block_size < 16 {
            return Err(CodecError::InvalidApi { what: "block size" });
        }
        if self.max_lpc_order > 32 {
            return Err(CodecError::InvalidApi { what: "lpc order" });
        }
        if self.max_partition_order > 15 || self.min_partition_order > self.max_partition_order {
            return Err(CodecError::InvalidApi {
                what: "partition order",
            });
        }
        if self.max_lpc_order > 0 && self.apodizations.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "lpc without a window",
            });
        }
        for window in &self.apodizations {
            let ok = match *window {
                Apodization::Tukey(p) => (0.0..=1.0).contains(&p),
                Apodization::PartialTukey(n) | Apodization::PunchoutTukey(n) => {
                    (1..=8).contains(&n)
                }
            };
            if !ok {
                return Err(CodecError::InvalidApi { what: "window" });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Settings;

    #[test]
    fn every_level_is_valid_and_nine_is_refused() {
        for level in 0..=8 {
            Settings::level(level)
                .expect("level")
                .validate()
                .expect("valid");
        }
        assert!(Settings::level(9).is_err());
    }

    #[test]
    fn impossible_settings_are_refused() {
        let base = Settings::level(5).expect("five");
        let mut bad = base.clone();
        bad.block_size = 15;
        assert!(bad.validate().is_err());
        bad = base.clone();
        bad.max_lpc_order = 33;
        assert!(bad.validate().is_err());
        bad = base.clone();
        bad.min_partition_order = 6;
        assert!(bad.validate().is_err());
        bad = base;
        bad.apodizations.clear();
        assert!(bad.validate().is_err());
    }
}
