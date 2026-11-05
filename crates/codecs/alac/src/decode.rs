//! The ALAC decoder behind the workspace [`Decoder`] trait.

use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SideData,
};

use crate::config::Cookie;
use crate::frame::FrameDecoder;

/// Largest packet accepted. An 8-channel 32-bit frame of the longest frame
/// length stored raw is a little over this, and no encoder writes one.
pub const MAX_PACKET_BYTES: usize = 16 << 20;

/// One decoded frame at the stream's bit depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Interleaved, in the frame's channel order (see [`Cookie::channel_layout`]).
    /// Each value is a signed integer of the stream's bit depth.
    pub samples: Vec<i32>,
    /// Samples per channel.
    pub frames: usize,
}

/// Decodes ALAC frames. Each packet is one frame.
///
/// ALAC frames are independent: there is no state between packets, so
/// `drain` is empty and `reset` has nothing to drop. A failed packet leaves
/// the decoder as it was.
#[derive(Debug)]
pub struct AlacDecoder {
    cookie: Cookie,
    layout: ChannelLayout,
    rate: u32,
    frame: FrameDecoder,
    samples: Vec<i32>,
}

impl AlacDecoder {
    /// A decoder for `cookie`, in any of the forms [`Cookie::parse`] reads.
    pub fn new(cookie: &[u8]) -> Result<Self, CodecError> {
        let cookie = Cookie::parse(cookie)?;
        if cookie.config.sample_rate == 0 {
            return Err(CodecError::InvalidApi {
                what: "alac sample rate",
            });
        }
        Ok(Self::from_cookie(cookie, cookie.config.sample_rate))
    }

    fn from_cookie(cookie: Cookie, rate: u32) -> Self {
        Self {
            layout: cookie.channel_layout(),
            cookie,
            rate,
            frame: FrameDecoder::default(),
            samples: Vec::new(),
        }
    }

    /// Open from stream parameters. `extra` is the magic cookie. The
    /// cookie's rate wins; a cookie rate of zero falls back to
    /// `sample_rate`. A layout whose length differs from the cookie's
    /// channel count is refused.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Alac {
            return Err(CodecError::Unsupported { what: "alac" });
        }
        if params.extra.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "alac magic cookie",
            });
        }
        let cookie = Cookie::parse(&params.extra)?;
        let rate = if cookie.config.sample_rate != 0 {
            cookie.config.sample_rate
        } else {
            params.sample_rate
        };
        if rate == 0 {
            return Err(CodecError::InvalidApi {
                what: "alac sample rate",
            });
        }
        if let Some(layout) = &params.layout
            && layout.len() != usize::from(cookie.config.channels)
        {
            return Err(CodecError::Malformed {
                at_bit: 72,
                what: "alac channel count differs from the stream",
            });
        }
        Ok(Self::from_cookie(cookie, rate))
    }

    #[must_use]
    pub fn cookie(&self) -> &Cookie {
        &self.cookie
    }

    /// Channel labels in output order.
    #[must_use]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Decode one frame to integers at the stream's bit depth.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedFrame, CodecError> {
        let frames = self.decode_into(data)?;
        Ok(DecodedFrame {
            samples: self.samples.clone(),
            frames,
        })
    }

    fn decode_into(&mut self, data: &[u8]) -> Result<usize, CodecError> {
        if data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if data.len() > MAX_PACKET_BYTES {
            return Err(CodecError::Limit {
                what: "packet bytes",
            });
        }
        self.frame
            .decode(&self.cookie.config, data, &mut self.samples)
    }

    fn check_side(&self, packet: &Packet) -> Result<(), CodecError> {
        for side in &packet.side {
            if let SideData::Config(bytes) = side {
                let next = Cookie::parse(bytes)?;
                if !next.config.decodes_like(&self.cookie.config) {
                    return Err(CodecError::Unsupported {
                        what: "configuration change",
                    });
                }
            }
        }
        Ok(())
    }

    /// The decoded integers as mix PCM. 16-bit is `I16`, exact. 20 and
    /// 24-bit are `F32` over `2^(depth - 1)`, exact. 32-bit is `F32` over
    /// `2^31`, rounded to the 24-bit float mantissa.
    fn pcm(&self) -> Result<Pcm, CodecError> {
        let depth = self.cookie.config.bit_depth;
        let pcm = if depth == 16 {
            let samples = self.samples.iter().map(|&v| v as i16).collect();
            Pcm::from_i16(self.rate, self.layout.clone(), samples)
        } else {
            let scale = 1.0 / f64::from(1u32 << (depth - 1));
            let samples = self
                .samples
                .iter()
                .map(|&v| (f64::from(v) * scale) as f32)
                .collect();
            Pcm::from_f32(self.rate, self.layout.clone(), samples)
        };
        pcm.map_err(|_| CodecError::Limit { what: "pcm buffer" })
    }
}

impl Decoder for AlacDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Alac
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        self.check_side(packet)?;
        self.decode_into(&packet.data)?;
        let pcm = self.pcm()?;
        Ok(vec![Output::Audio(AudioOutput {
            pcm,
            pts: packet.pts,
            timescale: packet.timescale,
            skip_leading: 0,
            skip_trailing: 0,
            concealed: false,
        })])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}
}
