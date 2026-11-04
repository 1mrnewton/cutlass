//! Inter prediction for one block (spec 7.11.3).

use crate::dsp::Pixel;

use super::tile::{R, TileDec};

impl<P: Pixel> TileDec<'_, P> {
    /// compute_prediction() (5.11.33). Intra blocks are predicted per transform block.
    pub fn compute_prediction(&mut self) -> R<()> {
        if !self.b.is_inter {
            return Ok(());
        }
        Err(cutlass_decode::CodecError::Unsupported { what: "inter prediction" })
    }
}
