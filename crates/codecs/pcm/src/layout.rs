//! PCM wiring. Endianness and plane order are parameters.
//!
//! Three bytes in [`cutlass_decode::CodecParams::extra`] name the layout.
//! An empty extra follows [`cutlass_decode::SampleFormat`]: `I16` is
//! interleaved little-endian s16, and `F32` is interleaved little-endian f32.
//! The container writes these bytes. This crate does not read a WAV header.

use cutlass_decode::{CodecError, CodecParams, SampleFormat};

/// Sample coding in the packet bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coding {
    /// Signed 16-bit. Decodes to [`SampleFormat::I16`].
    S16,
    /// Signed 24-bit, packed in three bytes. Decodes to [`SampleFormat::F32`].
    S24,
    /// Signed 32-bit. Decodes to [`SampleFormat::F32`].
    S32,
    /// IEEE-754 binary32. Decodes to [`SampleFormat::F32`].
    F32,
    /// IEEE-754 binary64, narrowed to [`SampleFormat::F32`].
    F64,
}

/// Byte order of one sample. Not inferred from the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

/// Channel order inside one packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrangement {
    /// Frame order: every channel of frame 0, then frame 1.
    Interleaved,
    /// Plane order: every frame of channel 0, then channel 1.
    Planar,
}

/// The layout a decoder or encoder was opened with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    coding: Coding,
    endian: Endian,
    arrangement: Arrangement,
}

impl Coding {
    /// Mix format this coding decodes into.
    #[must_use]
    pub const fn output_format(self) -> SampleFormat {
        match self {
            Self::S16 => SampleFormat::I16,
            Self::S24 | Self::S32 | Self::F32 | Self::F64 => SampleFormat::F32,
        }
    }

    /// Mix format the encoder requires on input.
    #[must_use]
    pub const fn input_format(self) -> SampleFormat {
        match self {
            Self::S16 => SampleFormat::I16,
            Self::S24 | Self::S32 | Self::F32 | Self::F64 => SampleFormat::F32,
        }
    }

    pub(crate) const fn sample_bytes(self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S24 => 3,
            Self::S32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// s24 and s32 decode into float. Encoding them would quantize, so the
    /// encoder refuses them instead of returning different samples.
    pub(crate) const fn encode(self) -> Result<(), CodecError> {
        match self {
            Self::S16 | Self::F32 | Self::F64 => Ok(()),
            Self::S24 => Err(CodecError::Unsupported { what: "s24 encode" }),
            Self::S32 => Err(CodecError::Unsupported { what: "s32 encode" }),
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::S16 => 1,
            Self::S24 => 2,
            Self::S32 => 3,
            Self::F32 => 4,
            Self::F64 => 5,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
            1 => Ok(Self::S16),
            2 => Ok(Self::S24),
            3 => Ok(Self::S32),
            4 => Ok(Self::F32),
            5 => Ok(Self::F64),
            6 => Err(CodecError::Unsupported { what: "8-bit pcm" }),
            7 => Err(CodecError::Unsupported { what: "a-law" }),
            8 => Err(CodecError::Unsupported { what: "mu-law" }),
            9 => Err(CodecError::Unsupported { what: "s64 pcm" }),
            10 => Err(CodecError::Unsupported { what: "s24 in 32" }),
            _ => Err(CodecError::Unsupported { what: "pcm coding" }),
        }
    }
}

impl Endian {
    const fn tag(self) -> u8 {
        match self {
            Self::Little => 0,
            Self::Big => 1,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
            0 => Ok(Self::Little),
            1 => Ok(Self::Big),
            _ => Err(CodecError::Malformed {
                at_bit: 8,
                what: "pcm endian",
            }),
        }
    }
}

impl Arrangement {
    const fn tag(self) -> u8 {
        match self {
            Self::Interleaved => 0,
            Self::Planar => 1,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, CodecError> {
        match tag {
            0 => Ok(Self::Interleaved),
            1 => Ok(Self::Planar),
            _ => Err(CodecError::Malformed {
                at_bit: 16,
                what: "pcm arrangement",
            }),
        }
    }
}

impl Layout {
    #[must_use]
    pub const fn new(coding: Coding, endian: Endian, arrangement: Arrangement) -> Self {
        Self {
            coding,
            endian,
            arrangement,
        }
    }

    #[must_use]
    pub const fn coding(self) -> Coding {
        self.coding
    }

    #[must_use]
    pub const fn endian(self) -> Endian {
        self.endian
    }

    #[must_use]
    pub const fn arrangement(self) -> Arrangement {
        self.arrangement
    }

    /// The three configuration bytes stored in `CodecParams.extra`.
    #[must_use]
    pub const fn to_extra(self) -> [u8; 3] {
        [self.coding.tag(), self.endian.tag(), self.arrangement.tag()]
    }

    pub fn from_params(params: &CodecParams) -> Result<Self, CodecError> {
        let layout = if params.extra.is_empty() {
            match params.sample_format {
                Some(SampleFormat::I16) => {
                    Self::new(Coding::S16, Endian::Little, Arrangement::Interleaved)
                }
                Some(SampleFormat::F32) => {
                    Self::new(Coding::F32, Endian::Little, Arrangement::Interleaved)
                }
                None => {
                    return Err(CodecError::Unsupported { what: "pcm layout" });
                }
            }
        } else {
            let bytes: [u8; 3] =
                params
                    .extra
                    .as_slice()
                    .try_into()
                    .map_err(|_| CodecError::Malformed {
                        at_bit: 0,
                        what: "pcm layout",
                    })?;
            Self::from_bytes(bytes)?
        };
        if let Some(format) = params.sample_format
            && format != layout.coding.output_format()
        {
            return Err(CodecError::InvalidApi {
                what: "pcm sample format",
            });
        }
        Ok(layout)
    }

    fn from_bytes(bytes: [u8; 3]) -> Result<Self, CodecError> {
        Ok(Self::new(
            Coding::from_tag(bytes[0])?,
            Endian::from_tag(bytes[1])?,
            Arrangement::from_tag(bytes[2])?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Arrangement, Coding, Endian, Layout};
    use cutlass_decode::{ChannelLayout, CodecError, CodecId, CodecParams, SampleFormat};

    fn params(extra: Vec<u8>, format: Option<SampleFormat>) -> CodecParams {
        let mut params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::mono());
        params.extra = extra;
        params.sample_format = format;
        params
    }

    #[test]
    fn empty_extra_follows_the_mix_format() {
        let s16 = Layout::from_params(&params(Vec::new(), Some(SampleFormat::I16))).expect("s16");
        assert_eq!(s16.coding(), Coding::S16);
        assert_eq!(s16.endian(), Endian::Little);
        assert_eq!(s16.arrangement(), Arrangement::Interleaved);
        let f32 = Layout::from_params(&params(Vec::new(), Some(SampleFormat::F32))).expect("f32");
        assert_eq!(f32.coding(), Coding::F32);
    }

    #[test]
    fn named_tools_are_refused() {
        for (tag, what) in [
            (6, "8-bit pcm"),
            (7, "a-law"),
            (8, "mu-law"),
            (9, "s64 pcm"),
            (10, "s24 in 32"),
        ] {
            let err = Layout::from_params(&params(vec![tag, 0, 0], None)).expect_err("tool");
            assert_eq!(err, CodecError::Unsupported { what });
        }
    }

    #[test]
    fn a_short_extra_is_malformed() {
        let err = Layout::from_params(&params(vec![1, 0], None)).expect_err("short");
        assert!(matches!(
            err,
            CodecError::Malformed {
                what: "pcm layout",
                ..
            }
        ));
    }

    #[test]
    fn endian_round_trips_through_extra() {
        let layout = Layout::new(Coding::S24, Endian::Big, Arrangement::Planar);
        let parsed =
            Layout::from_params(&params(layout.to_extra().to_vec(), Some(SampleFormat::F32)))
                .expect("parse");
        assert_eq!(parsed, layout);
    }
}
