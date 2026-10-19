//! Colorimetry carried with every frame.
//!
//! A hardware decoder hands back YUV samples that are meaningless without the
//! color metadata describing how to interpret them: which RGB primaries, which
//! transfer function (gamma / PQ / HLG), which YUV→RGB matrix, and whether the
//! samples use limited (16–235) or full (0–255) range. The old pipeline tracked
//! only a `full_range` bool and assumed BT.709 everywhere — fine for SDR phone
//! clips, wrong for HDR and wide-gamut footage. We carry the full description so
//! the compositor's shader can convert (and tone-map) correctly.
//!
//! These mirror the H.273 / container code points so a platform decoder can map
//! its metadata across with no interpretation.

/// RGB color primaries (the chromaticity of the red/green/blue points).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorPrimaries {
    /// Rec.709 / sRGB primaries — the SDR default.
    #[default]
    Bt709,
    /// Rec.601 525-line (SMPTE 170M / NTSC).
    Bt601_525,
    /// Rec.601 625-line (BT.470BG / PAL).
    Bt601_625,
    /// Rec.2020 / Rec.2100 wide gamut (UHD, HDR).
    Bt2020,
    /// DCI-P3 (display P3) — common on Apple capture.
    DisplayP3,
    /// Signalled as unspecified; the compositor falls back to [`Bt709`].
    Unspecified,
}

/// Opto-electronic transfer function (how linear light maps to stored samples).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferFunction {
    /// Rec.709 gamma — the SDR default.
    #[default]
    Bt709,
    /// sRGB (still images, screen recordings).
    Srgb,
    /// SMPTE ST 2084 / PQ — HDR10.
    Pq,
    /// Hybrid Log-Gamma — broadcast HDR.
    Hlg,
    /// Signalled as unspecified; the compositor falls back to [`Bt709`].
    Unspecified,
}

/// YUV→RGB matrix coefficients.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatrixCoefficients {
    /// Rec.709 — the SDR default.
    #[default]
    Bt709,
    /// Rec.601.
    Bt601,
    /// Rec.2020 non-constant luminance (HDR / wide gamut).
    Bt2020Ncl,
    /// Identity — the samples are already RGB/GBR (no conversion).
    Identity,
    /// Signalled as unspecified; the compositor falls back to [`Bt709`].
    Unspecified,
}

/// Sample range: studio/limited vs full.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    /// Limited / studio / "video" range: luma 16–235, chroma 16–240 (8-bit).
    /// The default for camera and broadcast footage.
    #[default]
    Limited,
    /// Full / "PC" / JPEG range: 0–255. Phones, screen recordings, stills.
    Full,
}

impl ColorRange {
    pub fn is_full(self) -> bool {
        matches!(self, ColorRange::Full)
    }
}

/// The complete color description attached to a [`crate::frame::VideoFrame`] and
/// reported by [`crate::decode::SourceInfo`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorSpace {
    pub primaries: ColorPrimaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub range: ColorRange,
}

impl ColorSpace {
    /// BT.709 limited-range — the safe SDR default when a source under-signals.
    pub const BT709: ColorSpace = ColorSpace {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferFunction::Bt709,
        matrix: MatrixCoefficients::Bt709,
        range: ColorRange::Limited,
    };

    /// BT.601 limited-range (SD / older footage). Defaults to the 625-line
    /// primaries; callers with 525-line sources should override `primaries`.
    pub const BT601: ColorSpace = ColorSpace {
        primaries: ColorPrimaries::Bt601_625,
        transfer: TransferFunction::Bt709,
        matrix: MatrixCoefficients::Bt601,
        range: ColorRange::Limited,
    };

    /// HDR10: BT.2020 primaries, PQ transfer, full container range.
    pub const BT2020_PQ: ColorSpace = ColorSpace {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Pq,
        matrix: MatrixCoefficients::Bt2020Ncl,
        range: ColorRange::Limited,
    };

    /// sRGB full-range RGB (decoded stills, screen recordings).
    pub const SRGB: ColorSpace = ColorSpace {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferFunction::Srgb,
        matrix: MatrixCoefficients::Identity,
        range: ColorRange::Full,
    };

    /// True if this frame needs HDR handling (PQ or HLG transfer). The
    /// compositor uses this to decide whether to tone-map to an SDR target.
    pub fn is_hdr(self) -> bool {
        matches!(self.transfer, TransferFunction::Pq | TransferFunction::Hlg)
    }

    /// True if the samples are wide-gamut (BT.2020 primaries).
    pub fn is_wide_gamut(self) -> bool {
        matches!(self.primaries, ColorPrimaries::Bt2020)
    }

    /// Resolve `Unspecified` fields to the BT.709 SDR fallback, so downstream
    /// code never has to special-case "unknown".
    pub fn resolved(self) -> ColorSpace {
        ColorSpace {
            primaries: match self.primaries {
                ColorPrimaries::Unspecified => ColorPrimaries::Bt709,
                p => p,
            },
            transfer: match self.transfer {
                TransferFunction::Unspecified => TransferFunction::Bt709,
                t => t,
            },
            matrix: match self.matrix {
                MatrixCoefficients::Unspecified => MatrixCoefficients::Bt709,
                m => m,
            },
            range: self.range,
        }
    }
}

impl Default for ColorSpace {
    fn default() -> Self {
        ColorSpace::BT709
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bt709_limited_sdr() {
        let c = ColorSpace::default();
        assert_eq!(c, ColorSpace::BT709);
        assert!(!c.is_hdr());
        assert!(!c.is_wide_gamut());
        assert!(!c.range.is_full());
    }

    #[test]
    fn hdr10_is_flagged_hdr_and_wide_gamut() {
        let c = ColorSpace::BT2020_PQ;
        assert!(c.is_hdr());
        assert!(c.is_wide_gamut());
    }

    #[test]
    fn srgb_is_full_range_identity_matrix() {
        let c = ColorSpace::SRGB;
        assert!(c.range.is_full());
        assert_eq!(c.matrix, MatrixCoefficients::Identity);
        assert!(!c.is_hdr());
    }

    #[test]
    fn resolved_replaces_unspecified_with_bt709() {
        let under = ColorSpace {
            primaries: ColorPrimaries::Unspecified,
            transfer: TransferFunction::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
            range: ColorRange::Limited,
        };
        assert_eq!(under.resolved(), ColorSpace::BT709);
    }

    #[test]
    fn resolved_preserves_known_fields() {
        // PQ/HLG and explicit primaries must survive resolution untouched.
        assert_eq!(ColorSpace::BT2020_PQ.resolved(), ColorSpace::BT2020_PQ);
    }
}
