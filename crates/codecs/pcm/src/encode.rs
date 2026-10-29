//! PCM encoder. The packet is the raw samples, with no framing delay.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{ChannelLayout, CodecError, CodecId, CodecParams, Packet, SampleFormat};
use cutlass_encode::{AudioInput, Encoder, FrameInput, PictureInput};

use crate::convert;
use crate::layout::Layout;
use crate::limit;

#[derive(Debug)]
pub struct PcmEncoder {
    layout: Layout,
    rate: u32,
    channels: ChannelLayout,
    format: SampleFormat,
    clock: i64,
    cancel: Arc<AtomicBool>,
}

impl PcmEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Pcm {
            return Err(CodecError::Unsupported { what: "pcm" });
        }
        limit::rate(params.sample_rate)?;
        let channels = params.layout.clone().ok_or(CodecError::InvalidApi {
            what: "channel layout",
        })?;
        if channels.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "channel layout",
            });
        }
        let layout = Layout::from_params(params)?;
        layout.coding().encode()?;
        Ok(Self {
            format: layout.coding().input_format(),
            layout,
            rate: params.sample_rate,
            channels,
            clock: 0,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn push_audio(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if input.pcm.rate != self.rate || input.pcm.layout != self.channels {
            return Err(CodecError::InvalidApi {
                what: "audio format",
            });
        }
        if input.pcm.format != self.format {
            return Err(CodecError::InvalidApi {
                what: "sample format",
            });
        }
        if input.pcm.samples == 0 {
            return Ok(Vec::new());
        }
        let bytes = convert::encode(
            self.layout,
            self.channels.len(),
            input.pcm.as_i16(),
            input.pcm.as_f32(),
        )?;
        let duration = i64::from(input.pcm.samples);
        let pts = self.clock;
        self.clock = self.clock.checked_add(duration).ok_or(CodecError::Limit {
            what: "sample clock",
        })?;
        Ok(vec![Packet::audio(bytes, pts, duration, self.rate)])
    }
}

impl Encoder for PcmEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Pcm
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        self.push_audio(input)
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "picture" })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.clock = 0;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}
