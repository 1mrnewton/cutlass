//! PCM decoder. One packet is one block of raw samples.

use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SideData,
};

use crate::convert::{self, Decoded};
use crate::layout::Layout;
use crate::limit;

#[derive(Debug)]
pub struct PcmDecoder {
    layout: Layout,
    rate: u32,
    channels: ChannelLayout,
    extra: [u8; 3],
}

impl PcmDecoder {
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
        Ok(Self {
            extra: layout.to_extra(),
            layout,
            rate: params.sample_rate,
            channels,
        })
    }

    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn push_packet(&mut self, packet: &Packet) -> Result<AudioOutput, CodecError> {
        for side in &packet.side {
            if let SideData::Config(bytes) = side
                && bytes.as_slice() != self.extra
            {
                return Err(CodecError::Unsupported {
                    what: "configuration change",
                });
            }
        }
        let decoded = convert::decode(self.layout, self.channels.len(), &packet.data)?;
        let pcm = match decoded {
            Decoded::I16(samples) => Pcm::from_i16(self.rate, self.channels.clone(), samples)
                .map_err(|_| CodecError::Limit { what: "pcm buffer" })?,
            Decoded::F32(samples) => Pcm::from_f32(self.rate, self.channels.clone(), samples)
                .map_err(|_| CodecError::Limit { what: "pcm buffer" })?,
        };
        Ok(AudioOutput {
            pcm,
            pts: packet.pts,
            timescale: packet.timescale,
            skip_leading: 0,
            skip_trailing: 0,
            concealed: false,
        })
    }
}

impl Decoder for PcmDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Pcm
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Ok(vec![Output::Audio(self.push_packet(packet)?)])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}
}
