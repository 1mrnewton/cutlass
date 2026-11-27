//! What differs between the codecs the hardware runs: NAL types, the
//! decoder configuration record, and which units start a picture.

use cutlass_decode::nal::{self, AvcConfig, Nal};
use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, SideData};

/// A codec these backends run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    H264,
    Hevc,
    Av1,
}

impl Kind {
    pub fn of(codec: CodecId) -> Result<Self, CodecError> {
        match codec {
            CodecId::H264 => Ok(Self::H264),
            CodecId::Hevc => Ok(Self::Hevc),
            CodecId::Av1 => Ok(Self::Av1),
            _ => Err(CodecError::Unsupported {
                what: "hardware video codec",
            }),
        }
    }

    pub fn codec(self) -> CodecId {
        match self {
            Self::H264 => CodecId::H264,
            Self::Hevc => CodecId::Hevc,
            Self::Av1 => CodecId::Av1,
        }
    }

    /// Whether units are NAL units (H.264 and HEVC) rather than OBUs.
    pub fn uses_nals(self) -> bool {
        matches!(self, Self::H264 | Self::Hevc)
    }

    /// The NAL type, from the header's low five bits or bits 1 to 6.
    pub fn nal_type(self, nal: &[u8]) -> Option<u8> {
        match self {
            Self::H264 => nal::h264_type(nal),
            Self::Hevc => nal::hevc_type(nal),
            Self::Av1 => None,
        }
    }

    pub fn is_parameter_set(self, kind: u8) -> bool {
        match self {
            Self::H264 => matches!(kind, 7 | 8),
            Self::Hevc => matches!(kind, 32..=34),
            Self::Av1 => false,
        }
    }

    pub fn is_delimiter(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 9,
            Self::Hevc => kind == 35,
            Self::Av1 => false,
        }
    }

    /// IDR for H.264; an intra random access point for HEVC.
    pub fn is_keyframe(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 5,
            Self::Hevc => matches!(kind, 16..=21),
            Self::Av1 => false,
        }
    }

    /// Parameter sets and NAL length size from an `avcC` or `hvcC`. AV1
    /// keeps its configuration OBUs whole, after the four `av1C` bytes.
    pub fn read_config(self, extra: &[u8]) -> Result<(u8, Vec<Vec<u8>>), CodecError> {
        match self {
            Self::H264 => {
                let config = nal::avc_config(extra)?;
                Ok((
                    config.length_size,
                    config.sps.into_iter().chain(config.pps).collect(),
                ))
            }
            Self::Hevc => {
                let config = nal::read_hevc_sets(extra)?;
                Ok((config.length_size, config.nals))
            }
            Self::Av1 => {
                let obus = extra.get(4..).unwrap_or_default();
                Ok((
                    0,
                    if obus.is_empty() {
                        Vec::new()
                    } else {
                        vec![obus.to_vec()]
                    },
                ))
            }
        }
    }

    /// The `avcC` or `hvcC` for parameter sets an encoder wrote.
    pub fn write_config(self, sets: &[Vec<u8>]) -> Result<Vec<u8>, CodecError> {
        match self {
            Self::H264 => {
                let (sps, pps): (Vec<_>, Vec<_>) = sets
                    .iter()
                    .cloned()
                    .partition(|set| nal::h264_type(set) == Some(7));
                let first = sps.first().cloned().unwrap_or_default();
                nal::write_avc_config(&AvcConfig {
                    length_size: 4,
                    profile: first.get(1).copied().unwrap_or(100),
                    compatibility: first.get(2).copied().unwrap_or(0),
                    level: first.get(3).copied().unwrap_or(0x28),
                    sps,
                    pps,
                })
            }
            Self::Hevc => nal::write_hevc_config(sets),
            Self::Av1 => Err(CodecError::Unsupported { what: "av1 encode" }),
        }
    }
}

/// The configuration a packet carries, if any.
pub fn config_of(packet: &Packet) -> Option<&[u8]> {
    packet.side.iter().find_map(|side| match side {
        SideData::Config(config) => Some(config.as_slice()),
        _ => None,
    })
}

/// Turns the units a demuxer hands over into what a hardware parser reads:
/// Annex B for H.264 and HEVC, with the parameter sets in front after open,
/// a new configuration, or a reset; OBUs for AV1, with the configuration
/// OBUs in front at the same points.
#[derive(Clone, Debug)]
pub struct Framing {
    kind: Kind,
    /// 0 when units arrive as Annex B.
    length_size: u8,
    /// Parameter sets or configuration OBUs, without start codes.
    sets: Vec<Vec<u8>>,
    send_sets: bool,
}

impl Framing {
    pub fn new(kind: Kind, params: &CodecParams) -> Result<Self, CodecError> {
        let (mut length_size, mut sets) = (params.nal_length_size, Vec::new());
        if !params.extra.is_empty() {
            (length_size, sets) = kind.read_config(&params.extra)?;
        }
        Ok(Self {
            kind,
            length_size,
            send_sets: !sets.is_empty(),
            sets,
        })
    }

    /// The bytes for `packet`, with the sets in front when they are due.
    pub fn unit(&mut self, packet: &Packet) -> Result<Vec<u8>, CodecError> {
        if let Some(config) = config_of(packet) {
            (self.length_size, self.sets) = self.kind.read_config(config)?;
            self.send_sets = true;
        }
        let mut unit = if self.kind.uses_nals() {
            self.annex_b(&packet.data)?
        } else {
            packet.data.clone()
        };
        if self.send_sets {
            let mut with_sets = if self.kind.uses_nals() {
                let sets: Vec<Nal> = self
                    .sets
                    .iter()
                    .map(|bytes| Nal {
                        bytes: bytes.clone(),
                    })
                    .collect();
                nal::to_annex_b(&sets)
            } else {
                self.sets.concat()
            };
            with_sets.append(&mut unit);
            unit = with_sets;
            self.send_sets = false;
        }
        Ok(unit)
    }

    /// Send the sets again before the next unit.
    pub fn reset(&mut self) {
        self.send_sets = !self.sets.is_empty();
    }

    /// The unit with start codes. Length framing is tried first, as on
    /// Apple and Windows: a 256 to 511 byte NAL has a prefix that reads as a
    /// start code.
    fn annex_b(&self, data: &[u8]) -> Result<Vec<u8>, CodecError> {
        let size = if self.length_size == 0 {
            4
        } else {
            self.length_size
        };
        if let Ok(nals) = nal::split_lengths(data, size) {
            return Ok(nal::to_annex_b(&nals));
        }
        nal::annex_b(data)?;
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::PixelFormat;

    const SPS: [u8; 4] = [0x67, 0x64, 0x00, 0x28];
    const PPS: [u8; 2] = [0x68, 0xee];
    const IDR: [u8; 3] = [0x65, 0x88, 0x80];

    fn avcc() -> Vec<u8> {
        nal::write_avc_config(&AvcConfig {
            length_size: 4,
            profile: 100,
            compatibility: 0,
            level: 0x28,
            sps: vec![SPS.to_vec()],
            pps: vec![PPS.to_vec()],
        })
        .expect("avcC")
    }

    #[test]
    fn the_sets_go_in_front_once_and_again_after_reset() {
        let mut params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        params.extra = avcc();
        let mut framing = Framing::new(Kind::H264, &params).expect("framing");
        let packet = Packet::video(
            nal::length_prefixed(
                &[Nal {
                    bytes: IDR.to_vec(),
                }],
                4,
            )
            .expect("unit"),
            0,
            1,
            30,
            true,
        );
        let first = framing.unit(&packet).expect("first");
        let nals = nal::annex_b(&first).expect("annex b");
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].bytes, SPS);
        assert_eq!(nals[2].bytes, IDR);
        let second = framing.unit(&packet).expect("second");
        assert_eq!(nal::annex_b(&second).expect("annex b").len(), 1);
        framing.reset();
        let third = framing.unit(&packet).expect("third");
        assert_eq!(nal::annex_b(&third).expect("annex b").len(), 3);
    }

    #[test]
    fn annex_b_passes_through_and_av1_keeps_its_bytes() {
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        let mut framing = Framing::new(Kind::H264, &params).expect("framing");
        let data = nal::to_annex_b(&[Nal {
            bytes: IDR.to_vec(),
        }]);
        let packet = Packet::video(data.clone(), 0, 1, 30, true);
        assert_eq!(framing.unit(&packet).expect("unit"), data);

        let mut params = CodecParams::with_format(CodecId::Av1, 64, 48, PixelFormat::Yuv420p);
        params.extra = vec![0x81, 0, 0, 0, 0x0a, 0x01, 0x00];
        let mut framing = Framing::new(Kind::Av1, &params).expect("framing");
        let packet = Packet::video(vec![0x32, 0x00], 0, 1, 30, true);
        assert_eq!(
            framing.unit(&packet).expect("unit"),
            vec![0x0a, 0x01, 0x00, 0x32, 0x00]
        );
        assert_eq!(framing.unit(&packet).expect("unit"), vec![0x32, 0x00]);
    }
}
