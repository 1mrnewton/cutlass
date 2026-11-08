//! How a shader turns a converted 4:2:0 sample into the working RGB the
//! CPU path makes from it.
//!
//! This is data only. The color crate decides the values. A shader
//! repeats them: each encoded RGB sample, after the matrix and clamped to
//! 0..1, is decoded with `source`, multiplied by `adapt` and clamped when
//! there is one, and encoded with `dest`. [`YuvTone::NONE`] leaves the
//! sample as it is.

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

/// The per-sample transfer and gamut step after the Y′CbCr matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuvTone {
    pub source: ToneCurve,
    pub dest: ToneCurve,
    /// Linear source RGB to linear destination RGB, row by row.
    pub adapt: Option<[[f64; 3]; 3]>,
    pub nits: f64,
}

impl YuvTone {
    /// The sample is left as the matrix made it.
    pub const NONE: Self = Self {
        source: ToneCurve::Rec709,
        dest: ToneCurve::Rec709,
        adapt: None,
        nits: 100.0,
    };

    /// Nothing changes the sample.
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
