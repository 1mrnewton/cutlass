//! Encoder trait and the factory that picks an implementation.
//!
//! `push` accepts any number of PCM frames, including zero. It consumes
//! every finite, in-range sample it is given, buffering less than two
//! frames internally, and returns the access units those samples completed.
//! A sample that is not legal fails the call and adds nothing from that
//! push. `drain` pads and flushes. A second `drain` is empty. `reset`
//! drops the buffer, the overlap, and the sample clock.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use cutlass_codec::{
    Channel, ChannelLayout, CodecError, CodecId, CodecParams, Packet, Pcm, SampleFormat, SideData,
};

/// PCM presented to an encoder, contiguous with the samples already accepted.
///
/// The encoder clocks packets from the count of accepted samples. `pts` on
/// the input is not a second clock: a discontinuity is a [`Encoder::reset`].
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    pub pcm: Pcm,
}

pub trait Encoder: Send {
    fn codec(&self) -> CodecId;

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError>;

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError>;

    fn reset(&mut self);

    /// A handle another thread can raise. The encoder checks it between
    /// frames, never inside a lock this crate does not take.
    fn cancel_flag(&self) -> Arc<AtomicBool>;
}

pub struct EncodeFactory {
    openers: Vec<Opener>,
}

struct Opener {
    id: CodecId,
    open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
}

impl EncodeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            openers: Vec::new(),
        }
    }

    pub fn register(
        &mut self,
        id: CodecId,
        open: fn(&CodecParams) -> Result<Box<dyn Encoder>, CodecError>,
    ) {
        self.openers.push(Opener { id, open });
    }

    pub fn open(&self, params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        for opener in &self.openers {
            if opener.id == params.codec {
                return (opener.open)(params);
            }
        }
        Err(CodecError::Unsupported {
            what: "encoder is not registered",
        })
    }
}

impl Default for EncodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Raised by the owner of [`Encoder::cancel_flag`].
pub fn request_cancel(flag: &AtomicBool) {
    flag.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{AudioInput, EncodeFactory, Encoder};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cutlass_codec::{ChannelLayout, CodecError, CodecId, CodecParams, Packet, Pcm};

    struct Copy;

    impl Encoder for Copy {
        fn codec(&self) -> CodecId {
            CodecId::Pcm
        }

        fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
            let Some(samples) = input.pcm.as_f32() else {
                return Err(CodecError::InvalidApi {
                    what: "copy encoder wants f32",
                });
            };
            if samples.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![Packet::audio(
                samples
                    .iter()
                    .flat_map(|sample| sample.to_bits().to_le_bytes())
                    .collect(),
                0,
                i64::from(input.pcm.samples),
                input.pcm.rate,
            )])
        }

        fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}

        fn cancel_flag(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    fn open_copy(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported { what: "copy" });
        }
        Ok(Box::new(Copy))
    }

    #[test]
    fn empty_push_emits_nothing() {
        let mut factory = EncodeFactory::new();
        factory.register(CodecId::Pcm, open_copy);
        let params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        let mut encoder = factory.open(&params).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), Vec::new()).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("push");
        assert!(packets.is_empty());
        assert!(encoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn missing_encoder_is_unsupported() {
        let err = match EncodeFactory::new().open(&CodecParams::audio(
            CodecId::Aac,
            48_000,
            ChannelLayout::mono(),
        )) {
            Err(err) => err,
            Ok(_) => panic!("missing"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "encoder is not registered",
            }
        );
    }
}
