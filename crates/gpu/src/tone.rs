//! How a shader turns a 4:2:0 sample into the working RGB the CPU path
//! makes from it.
//!
//! This is data only. The color crate decides the values. A shader
//! repeats them: the sample is expanded from its range and goes through
//! `matrix`; each encoded RGB sample, clamped to 0..1, is then decoded
//! with `source`, multiplied by `adapt` and clamped when there is one, and
//! encoded with `dest`. [`YuvTone::NONE`] is studio swing BT.709 and
//! leaves the sample as the matrix made it.

/// A curve between encoded and linear RGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneCurve {
    /// The Rec.709 OETF and its inverse.
    Rec709,
    /// The BT.2100 HLG OETF and its inverse, without the OOTF.
    Hlg,
    /// SMPTE ST 2084, linear 1 at `nits` cd/m².
    Pq,
}

/// The Y′CbCr matrix a 4:2:0 sample goes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum YuvMatrix {
    #[default]
    Bt709,
    Bt601,
    Bt2020,
}

impl YuvMatrix {
    /// The luma weights of red and blue.
    #[must_use]
    pub fn weights(self) -> (f64, f64) {
        match self {
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt601 => (0.299, 0.114),
            Self::Bt2020 => (0.2627, 0.0593),
        }
    }
}

/// The range and matrix of a Y′CbCr sample, and the per-sample transfer
/// and gamut step after the matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuvTone {
    /// The matrix the samples are coded with.
    pub matrix: YuvMatrix,
    /// The samples use the whole range of a byte, not studio swing.
    pub full_range: bool,
    pub source: ToneCurve,
    pub dest: ToneCurve,
    /// Linear source RGB to linear destination RGB, row by row.
    pub adapt: Option<[[f64; 3]; 3]>,
    pub nits: f64,
}

impl YuvTone {
    /// Studio swing BT.709, and the sample is left as the matrix made it.
    pub const NONE: Self = Self {
        matrix: YuvMatrix::Bt709,
        full_range: false,
        source: ToneCurve::Rec709,
        dest: ToneCurve::Rec709,
        adapt: None,
        nits: 100.0,
    };

    /// Nothing changes the sample after the matrix.
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.adapt.is_none() && self.source == self.dest
    }
}

impl Default for YuvTone {
    fn default() -> Self {
        Self::NONE
    }
}
