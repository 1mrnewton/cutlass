//! Decoder configuration. Operating point and film grain are explicit.

use crate::error::DecodeError;
use crate::limits::DecoderLimits;
use cutlass_core::Rational;

/// How packet bytes are framed. The decoder does not guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputFraming {
    /// Low-overhead OBU stream. Every OBU has `obu_has_size_field == 1`.
    #[default]
    LowOverhead,
    /// Annex B length-delimited temporal units.
    AnnexB,
}

/// Whether presentation frames include reference-process film grain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilmGrainMode {
    /// Apply grain to a copy of the restored image. This is the presentation
    /// default.
    #[default]
    Apply,
    /// Return the restored image without grain. Grain parameters are still
    /// parsed and reported on [`crate::FrameInfo`].
    Reconstruction,
}

/// What `submit` should do when the output queue is full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    pub framing: InputFraming,
    pub limits: DecoderLimits,
    pub grain: FilmGrainMode,
    /// Index into `operating_point_idc`. The default is operating point 0.
    /// Changing this between frames is rejected once a frame header has been
    /// seen for the current sequence; call `reset` first.
    pub operating_point: u8,
    /// Time base stamped on presentations when the packet does not carry one.
    /// Packet timestamps, when present, win.
    pub time_base: Rational,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            framing: InputFraming::LowOverhead,
            limits: DecoderLimits::default(),
            grain: FilmGrainMode::Apply,
            operating_point: 0,
            time_base: Rational::new(1, 1),
        }
    }
}

impl DecoderConfig {
    pub fn validate(&self) -> Result<(), DecodeError> {
        self.limits.validate()?;
        if !self.time_base.is_valid() {
            return Err(DecodeError::Misuse(
                "time_base must be a positive rational".into(),
            ));
        }
        Ok(())
    }
}

/// One accepted packet of already-framed bytes plus the host's timing.
///
/// `timestamp` is the temporal-unit presentation time in `time_base` ticks
/// when the container provided one. It is not derived from order hints.
/// Several shown frames in one packet share this timestamp only when the
/// caller passed one value; the decoder does not invent a cadence.
#[derive(Debug, Clone, Copy, Default)]
pub struct PacketTiming {
    pub timestamp: Option<i64>,
    pub duration: Option<i64>,
    pub discontinuity: bool,
}
