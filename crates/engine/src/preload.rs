//! A timeline ask that may already have an answer.
//!
//! The ask returns a stored cell or waveform. It does not decode and it does
//! not wait. Filling either cache is the segment source's job.

use cutlass_model::Micros;

use crate::error::EngineError;
use crate::filmstrip::{Filmstrip, StripCell};
use crate::waveform::Waveform;

pub(crate) struct Preload {
    strip: Filmstrip,
    wave: Option<Waveform>,
}

impl Preload {
    pub(crate) fn new() -> Self {
        Self {
            strip: Filmstrip::new(),
            wave: None,
        }
    }

    pub(crate) fn strip(&self, source: Micros) -> Result<StripCell, EngineError> {
        match self.strip.at(source.0) {
            Some(cell) => Ok(cell.clone()),
            None => Err(EngineError::NotReady),
        }
    }

    pub(crate) fn store_picture(&mut self, cell: StripCell) {
        self.strip.insert(cell);
    }

    pub(crate) fn waveform(&self) -> Result<Waveform, EngineError> {
        match &self.wave {
            Some(wave) => Ok(wave.clone()),
            None => Err(EngineError::NotReady),
        }
    }

    pub(crate) fn store_waveform(&mut self, wave: Waveform) {
        self.wave = Some(wave);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filmstrip::fit;
    use crate::waveform::Waveform;

    #[test]
    fn an_ask_before_a_fill_is_not_ready() {
        let mut preload = Preload::new();
        assert_eq!(preload.strip(Micros(0)), Err(EngineError::NotReady));
        let cell = fit(Micros(0), 1, 1, &[8, 0, 0, 255]).expect("cell");
        preload.store_picture(cell);
        let stored = preload.strip(Micros(4)).expect("later");
        assert_eq!(stored.rgba(), &[8, 0, 0, 255]);
        assert_eq!(preload.strip(Micros(-1)), Err(EngineError::NotReady));
        assert_eq!(preload.waveform(), Err(EngineError::NotReady));
        let wave = Waveform::from_interleaved(&[0.25], 1, 1, 1, false).expect("wave");
        preload.store_waveform(wave);
        assert_eq!(
            preload.waveform().expect("stored").peaks()[0].max.to_bits(),
            0.25_f32.to_bits()
        );
    }
}
