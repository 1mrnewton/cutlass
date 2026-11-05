//! The Opus decoder behind [`cutlass_decode::Decoder`].

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Output, Packet};

/// An Opus decoder for one `OpusHead` configuration.
#[derive(Debug)]
pub struct OpusDecoder {}

impl OpusDecoder {
    /// Opens a decoder for `params`.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let _ = params;
        Err(CodecError::Unsupported {
            what: "Opus decoder",
        })
    }
}

impl Decoder for OpusDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let _ = packet;
        Err(CodecError::Unsupported {
            what: "Opus decoder",
        })
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}
}
