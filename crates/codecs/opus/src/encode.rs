//! The Opus encoder behind [`cutlass_encode::Encoder`].

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cutlass_encode::{AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet};

/// An Opus encoder for one mono or stereo stream.
#[derive(Debug)]
pub struct OpusEncoder {
    cancel: Arc<AtomicBool>,
}

impl OpusEncoder {
    /// Opens an encoder for `params`.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let _ = params;
        Err(CodecError::Unsupported {
            what: "Opus encoder",
        })
    }
}

impl Encoder for OpusEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        let _ = input;
        Err(CodecError::Unsupported {
            what: "Opus encoder",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let _ = input;
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
