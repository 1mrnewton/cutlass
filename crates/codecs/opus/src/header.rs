//! The Opus decoder configuration: `OpusHead` and `dOps`.
//!
//! `CodecParams.extra` for Opus is the identification header of RFC 7845
//! section 5.1 (`OpusHead`, little-endian), which is what the Ogg mapping
//! stores and what FFmpeg passes as extradata. The Opus-in-ISOBMFF `dOps`
//! box body (big-endian, version 0) is accepted as well, so a configuration
//! taken straight from an MP4 opens the same decoder.
//!
//! Channel mapping family 0 is mono or stereo in one stream. Family 1 is
//! one to eight channels in Vorbis order. Family 255 is any count with no
//! defined positions. Family 2 and 3 (ambisonics, RFC 8486) are refused.

use cutlass_decode::{Channel, ChannelLayout, CodecError};

/// Largest channel count a Cutlass PCM buffer carries.
pub const MAX_CHANNELS: usize = 8;

/// The identification header, in host values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpusHead {
    pub channels: u8,
    /// Samples at 48 kHz to discard from the start of the decoded stream.
    pub pre_skip: u16,
    /// The rate the encoder was fed. Informational only.
    pub input_rate: u32,
    /// Gain in Q7.8 dB applied to the decoded output.
    pub output_gain: i16,
    pub family: u8,
    pub streams: u8,
    pub coupled: u8,
    /// For each output channel, the decoded channel it takes, or 255 for silence.
    pub mapping: Vec<u8>,
}

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

impl OpusHead {
    /// A one-stream mono or stereo header.
    #[must_use]
    pub fn simple(channels: u8, pre_skip: u16, input_rate: u32) -> Self {
        Self {
            channels,
            pre_skip,
            input_rate,
            output_gain: 0,
            family: 0,
            streams: 1,
            coupled: u8::from(channels == 2),
            mapping: if channels == 2 { vec![0, 1] } else { vec![0] },
        }
    }

    /// Reads `OpusHead` or a `dOps` body.
    pub fn parse(bytes: &[u8]) -> Result<Self, CodecError> {
        if bytes.starts_with(b"OpusHead") {
            Self::parse_ogg(bytes)
        } else {
            Self::parse_dops(bytes)
        }
    }

    fn parse_ogg(bytes: &[u8]) -> Result<Self, CodecError> {
        if bytes.len() < 19 {
            return Err(CodecError::Truncated {
                at_bit: (bytes.len() as u64) * 8,
                what: "OpusHead",
            });
        }
        // RFC 7845 section 5.1: a major version above 0 (version 16 and up)
        // is an incompatible header.
        if bytes[8] >= 16 {
            return Err(CodecError::Unsupported {
                what: "OpusHead major version",
            });
        }
        let channels = bytes[9];
        let pre_skip = u16::from_le_bytes([bytes[10], bytes[11]]);
        let input_rate = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let output_gain = i16::from_le_bytes([bytes[16], bytes[17]]);
        let family = bytes[18];
        Self::finish(
            channels,
            pre_skip,
            input_rate,
            output_gain,
            family,
            &bytes[19..],
        )
    }

    fn parse_dops(bytes: &[u8]) -> Result<Self, CodecError> {
        if bytes.len() < 11 {
            return Err(CodecError::Truncated {
                at_bit: (bytes.len() as u64) * 8,
                what: "dOps",
            });
        }
        if bytes[0] != 0 {
            return Err(CodecError::Unsupported {
                what: "dOps version",
            });
        }
        let channels = bytes[1];
        let pre_skip = u16::from_be_bytes([bytes[2], bytes[3]]);
        let input_rate = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let output_gain = i16::from_be_bytes([bytes[8], bytes[9]]);
        let family = bytes[10];
        Self::finish(
            channels,
            pre_skip,
            input_rate,
            output_gain,
            family,
            &bytes[11..],
        )
    }

    fn finish(
        channels: u8,
        pre_skip: u16,
        input_rate: u32,
        output_gain: i16,
        family: u8,
        table: &[u8],
    ) -> Result<Self, CodecError> {
        if channels == 0 {
            return Err(malformed("zero output channels"));
        }
        let (streams, coupled, mapping) = if family == 0 {
            if channels > 2 {
                return Err(malformed("family 0 with more than two channels"));
            }
            let mapping = if channels == 2 { vec![0, 1] } else { vec![0] };
            (1, u8::from(channels == 2), mapping)
        } else {
            if table.len() < 2 + usize::from(channels) {
                return Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "channel mapping table",
                });
            }
            let streams = table[0];
            let coupled = table[1];
            let mapping = table[2..2 + usize::from(channels)].to_vec();
            (streams, coupled, mapping)
        };
        let head = Self {
            channels,
            pre_skip,
            input_rate,
            output_gain,
            family,
            streams,
            coupled,
            mapping,
        };
        head.validate()?;
        Ok(head)
    }

    /// Checks the stream counts and the mapping against RFC 7845 section 5.1.1.
    pub fn validate(&self) -> Result<(), CodecError> {
        match self.family {
            0 => {}
            1 => {
                if !(1..=8).contains(&self.channels) {
                    return Err(malformed("family 1 channel count"));
                }
            }
            255 => {}
            2 | 3 => {
                return Err(CodecError::Unsupported {
                    what: "ambisonic channel mapping",
                });
            }
            _ => {
                return Err(CodecError::Unsupported {
                    what: "reserved channel mapping family",
                });
            }
        }
        if self.streams == 0 {
            return Err(malformed("zero streams"));
        }
        if self.coupled > self.streams {
            return Err(malformed("more coupled streams than streams"));
        }
        let decoded = usize::from(self.streams) + usize::from(self.coupled);
        if decoded > 255 {
            return Err(malformed("more than 255 decoded channels"));
        }
        if self.mapping.len() != usize::from(self.channels) {
            return Err(malformed("mapping length"));
        }
        for &index in &self.mapping {
            if index != 255 && usize::from(index) >= decoded {
                return Err(malformed("mapping names a missing channel"));
            }
        }
        if usize::from(self.channels) > MAX_CHANNELS {
            return Err(CodecError::Limit {
                what: "more than eight output channels",
            });
        }
        Ok(())
    }

    /// The `OpusHead` packet RFC 7845 section 5.1 defines.
    #[must_use]
    pub fn to_ogg(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(21 + self.mapping.len());
        out.extend_from_slice(b"OpusHead");
        out.push(1);
        out.push(self.channels);
        out.extend_from_slice(&self.pre_skip.to_le_bytes());
        out.extend_from_slice(&self.input_rate.to_le_bytes());
        out.extend_from_slice(&self.output_gain.to_le_bytes());
        out.push(self.family);
        if self.family != 0 {
            out.push(self.streams);
            out.push(self.coupled);
            out.extend_from_slice(&self.mapping);
        }
        out
    }

    /// The `dOps` box body of the Opus-in-ISOBMFF mapping.
    #[must_use]
    pub fn to_dops(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(13 + self.mapping.len());
        out.push(0);
        out.push(self.channels);
        out.extend_from_slice(&self.pre_skip.to_be_bytes());
        out.extend_from_slice(&self.input_rate.to_be_bytes());
        out.extend_from_slice(&self.output_gain.to_be_bytes());
        out.push(self.family);
        if self.family != 0 {
            out.push(self.streams);
            out.push(self.coupled);
            out.extend_from_slice(&self.mapping);
        }
        out
    }

    /// Speaker labels for the output channels.
    ///
    /// Family 0 and 1 follow the Vorbis order of RFC 7845 section 5.1.1.2.
    /// Family 255 has no positions; its channels take the same labels by
    /// count so a buffer can be built, and a caller must not read them as
    /// speaker positions.
    pub fn layout(&self) -> Result<ChannelLayout, CodecError> {
        use Channel::{
            BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft,
            SideRight,
        };
        let channels = match self.channels {
            1 => vec![FrontCenter],
            2 => vec![FrontLeft, FrontRight],
            3 => vec![FrontLeft, FrontCenter, FrontRight],
            4 => vec![FrontLeft, FrontRight, BackLeft, BackRight],
            5 => vec![FrontLeft, FrontCenter, FrontRight, BackLeft, BackRight],
            6 => vec![FrontLeft, FrontCenter, FrontRight, BackLeft, BackRight, Lfe],
            7 => vec![
                FrontLeft,
                FrontCenter,
                FrontRight,
                SideLeft,
                SideRight,
                BackCenter,
                Lfe,
            ],
            8 => vec![
                FrontLeft,
                FrontCenter,
                FrontRight,
                SideLeft,
                SideRight,
                BackLeft,
                BackRight,
                Lfe,
            ],
            _ => {
                return Err(CodecError::Limit {
                    what: "more than eight output channels",
                });
            }
        };
        ChannelLayout::new(channels).map_err(|_| CodecError::Limit {
            what: "channel layout",
        })
    }
}

/// The default mapping RFC 7845 section 5.1.1.2 gives for `channels` in family 1.
///
/// Returns `(streams, coupled, mapping)`.
#[must_use]
pub fn vorbis_mapping(channels: u8) -> Option<(u8, u8, Vec<u8>)> {
    let (streams, coupled, mapping): (u8, u8, &[u8]) = match channels {
        1 => (1, 0, &[0]),
        2 => (1, 1, &[0, 1]),
        3 => (2, 1, &[0, 2, 1]),
        4 => (2, 2, &[0, 1, 2, 3]),
        5 => (3, 2, &[0, 4, 1, 2, 3]),
        6 => (4, 2, &[0, 4, 1, 2, 3, 5]),
        7 => (4, 3, &[0, 4, 1, 2, 3, 5, 6]),
        8 => (5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]),
        _ => return None,
    };
    Some((streams, coupled, mapping.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::{OpusHead, vorbis_mapping};
    use cutlass_decode::CodecError;

    #[test]
    fn an_ogg_header_round_trips() {
        let head = OpusHead::simple(2, 312, 44_100);
        let bytes = head.to_ogg();
        assert_eq!(bytes.len(), 19);
        assert_eq!(&bytes[..8], b"OpusHead");
        assert_eq!(OpusHead::parse(&bytes).unwrap(), head);
    }

    #[test]
    fn a_dops_body_names_the_same_configuration() {
        let (streams, coupled, mapping) = vorbis_mapping(6).unwrap();
        let head = OpusHead {
            channels: 6,
            pre_skip: 3840,
            input_rate: 48_000,
            output_gain: -256,
            family: 1,
            streams,
            coupled,
            mapping,
        };
        let dops = head.to_dops();
        assert_eq!(dops[0], 0);
        assert_eq!(u16::from_be_bytes([dops[2], dops[3]]), 3840);
        assert_eq!(OpusHead::parse(&dops).unwrap(), head);
        assert_eq!(OpusHead::parse(&head.to_ogg()).unwrap(), head);
        assert_eq!(head.layout().unwrap().len(), 6);
    }

    #[test]
    fn bad_mappings_are_refused() {
        let mut head = OpusHead::simple(2, 0, 48_000);
        head.channels = 3;
        assert!(OpusHead::parse(&head.to_ogg()).is_err());
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[18] = 1;
        bytes.extend([1, 0, 5]);
        assert!(matches!(
            OpusHead::parse(&bytes),
            Err(CodecError::Malformed { .. })
        ));
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[18] = 1;
        bytes.extend([1, 2, 0]);
        assert!(OpusHead::parse(&bytes).is_err());
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[18] = 1;
        bytes.extend([0, 0, 0]);
        assert!(OpusHead::parse(&bytes).is_err());
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[18] = 2;
        bytes.extend([1, 0, 0]);
        assert!(matches!(
            OpusHead::parse(&bytes),
            Err(CodecError::Unsupported { .. })
        ));
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[8] = 16;
        assert!(OpusHead::parse(&bytes).is_err());
        assert!(OpusHead::parse(b"OpusHead\x01").is_err());
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[9] = 0;
        assert!(OpusHead::parse(&bytes).is_err());
    }

    #[test]
    fn a_silent_channel_is_allowed() {
        let mut bytes = OpusHead::simple(1, 0, 48_000).to_ogg();
        bytes[9] = 2;
        bytes[18] = 255;
        bytes.extend([1, 0, 0, 255]);
        let head = OpusHead::parse(&bytes).unwrap();
        assert_eq!(head.mapping, vec![0, 255]);
    }
}
