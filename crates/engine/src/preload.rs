//! A timeline ask that may already have an answer.
//!
//! The ask returns a stored cell. It does not decode and it does not wait.
//! Filling a cell is the segment source's job.

use cutlass_model::Micros;

use crate::error::EngineError;
use crate::filmstrip::{Filmstrip, StripCell};

pub(crate) struct Preload {
    strip: Filmstrip,
}

impl Preload {
    pub(crate) fn new() -> Self {
        Self {
            strip: Filmstrip::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filmstrip::fit;

    #[test]
    fn an_ask_before_a_fill_is_not_ready() {
        let mut preload = Preload::new();
        assert_eq!(preload.strip(Micros(0)), Err(EngineError::NotReady));
        let cell = fit(Micros(0), 1, 1, &[8, 0, 0, 255]).expect("cell");
        preload.store_picture(cell);
        let stored = preload.strip(Micros(4)).expect("later");
        assert_eq!(stored.rgba(), &[8, 0, 0, 255]);
        assert_eq!(preload.strip(Micros(-1)), Err(EngineError::NotReady));
    }
}
