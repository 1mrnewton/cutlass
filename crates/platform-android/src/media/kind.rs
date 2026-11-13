//! The video codecs MediaCodec runs here, and what each needs around it.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

use std::ffi::CStr;

use cutlass_decode::nal::{self, AvcConfig};
use cutlass_decode::{CodecError, CodecId, Packet, SideData};

/// One codec, by the id the rest of the workspace uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    H264,
    Hevc,
    Vp9,
    Av1,
}

impl Kind {
    pub fn of(codec: CodecId) -> Result<Self, CodecError> {
        match codec {
            CodecId::H264 => Ok(Self::H264),
            CodecId::Hevc => Ok(Self::Hevc),
            CodecId::Vp9 => Ok(Self::Vp9),
            CodecId::Av1 => Ok(Self::Av1),
            _ => Err(CodecError::Unsupported {
                what: "mediacodec codec",
            }),
        }
    }

    pub fn codec(self) -> CodecId {
        match self {
            Self::H264 => CodecId::H264,
            Self::Hevc => CodecId::Hevc,
            Self::Vp9 => CodecId::Vp9,
            Self::Av1 => CodecId::Av1,
        }
    }

    pub fn mime(self) -> &'static CStr {
        match self {
            Self::H264 => c"video/avc",
            Self::Hevc => c"video/hevc",
            Self::Vp9 => c"video/x-vnd.on2.vp9",
            Self::Av1 => c"video/av01",
        }
    }

    /// The software decoders Android ships, newest name first.
    pub fn software_decoders(self) -> &'static [&'static str] {
        match self {
            Self::H264 => &["c2.android.avc.decoder", "OMX.google.h264.decoder"],
            Self::Hevc => &["c2.android.hevc.decoder", "OMX.google.hevc.decoder"],
            Self::Vp9 => &["c2.android.vp9.decoder", "OMX.google.vp9.decoder"],
            Self::Av1 => &["c2.android.av1.decoder", "c2.android.av1-dav1d.decoder"],
        }
    }

    /// The software encoders Android ships. VP9 and AV1 are not encoded here.
    pub fn software_encoders(self) -> &'static [&'static str] {
        match self {
            Self::H264 => &["c2.android.avc.encoder", "OMX.google.h264.encoder"],
            Self::Hevc => &["c2.android.hevc.encoder"],
            Self::Vp9 | Self::Av1 => &[],
        }
    }

    /// Whether units travel as NAL units, which MediaCodec wants with start codes.
    pub fn uses_nal(self) -> bool {
        matches!(self, Self::H264 | Self::Hevc)
    }

    /// The NAL type, from the header's low five bits or bits 1 to 6.
    pub fn nal_type(self, nal: &[u8]) -> Option<u8> {
        match self {
            Self::H264 => nal::h264_type(nal),
            Self::Hevc => nal::hevc_type(nal),
            Self::Vp9 | Self::Av1 => None,
        }
    }

    pub fn is_parameter_set(self, kind: u8) -> bool {
        match self {
            Self::H264 => matches!(kind, 7 | 8),
            Self::Hevc => matches!(kind, 32..=34),
            Self::Vp9 | Self::Av1 => false,
        }
    }

    pub fn is_delimiter(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 9,
            Self::Hevc => kind == 35,
            Self::Vp9 | Self::Av1 => false,
        }
    }

    /// IDR for H.264; an intra random access point for HEVC.
    pub fn is_keyframe(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 5,
            Self::Hevc => matches!(kind, 16..=21),
            Self::Vp9 | Self::Av1 => false,
        }
    }

    /// Whether the `avcC` or `hvcC` record names more than 8-bit 4:2:0,
    /// which Android's own software decoders refuse at the first unit:
    /// HEVC Main 10 from an iPhone, or H.264 High 10, 4:2:2 or 4:4:4.
    pub fn beyond_software(self, record: &[u8]) -> bool {
        match self {
            // profile_idc 110, 122, 244 and the CAVLC 4:4:4 intra 44.
            Self::H264 => record
                .get(1)
                .is_some_and(|profile| matches!(profile, 44 | 110 | 122 | 244)),
            // chromaFormat, then the luma and chroma bit depths less 8.
            Self::Hevc => match record.get(16..19) {
                Some(&[chroma, luma, chroma_depth]) => {
                    chroma & 3 != 1 || luma & 7 != 0 || chroma_depth & 7 != 0
                }
                _ => false,
            },
            Self::Vp9 | Self::Av1 => false,
        }
    }

    /// Parameter sets and NAL length size from an `avcC` or `hvcC`.
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
            Self::Vp9 | Self::Av1 => Ok((0, Vec::new())),
        }
    }

    /// The `avcC` or `hvcC` for these parameter sets.
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
                    profile: first.get(1).copied().unwrap_or(66),
                    compatibility: first.get(2).copied().unwrap_or(0),
                    level: first.get(3).copied().unwrap_or(0x1f),
                    sps,
                    pps,
                })
            }
            Self::Hevc => nal::write_hevc_config(sets),
            Self::Vp9 | Self::Av1 => Err(CodecError::Unsupported {
                what: "mediacodec configuration record",
            }),
        }
    }

    /// The codec-specific data MediaCodec takes at configure time: for AVC
    /// the SPS in `csd-0` and the PPS in `csd-1`, for HEVC every set in
    /// `csd-0`, all with start codes; for AV1 the `av1C` record as is.
    pub fn csd(self, sets: &[Vec<u8>], extra: &[u8]) -> Vec<Vec<u8>> {
        let annex_b = |sets: Vec<&Vec<u8>>| {
            let nals: Vec<nal::Nal> = sets
                .into_iter()
                .map(|bytes| nal::Nal {
                    bytes: bytes.clone(),
                })
                .collect();
            nal::to_annex_b(&nals)
        };
        match self {
            Self::H264 => {
                let (sps, pps): (Vec<_>, Vec<_>) =
                    sets.iter().partition(|set| nal::h264_type(set) == Some(7));
                if sps.is_empty() || pps.is_empty() {
                    return Vec::new();
                }
                vec![annex_b(sps), annex_b(pps)]
            }
            Self::Hevc if !sets.is_empty() => vec![annex_b(sets.iter().collect())],
            Self::Av1 if !extra.is_empty() => vec![extra.to_vec()],
            Self::Hevc | Self::Vp9 | Self::Av1 => Vec::new(),
        }
    }
}

/// The configuration record a packet carries, when it has one.
pub fn config_of(packet: &Packet) -> Option<&[u8]> {
    packet.side.iter().find_map(|side| match side {
        SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.as_slice()),
        _ => None,
    })
}

/// Whether a codec name is one of Android's own software codecs.
pub fn is_software(name: &str) -> bool {
    name.starts_with("c2.android.") || name.starts_with("OMX.google.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avc_sets_split_into_two_csd_buffers_with_start_codes() {
        let sps = vec![0x67, 0x42, 0x00, 0x1e];
        let pps = vec![0x68, 0xce];
        let csd = Kind::H264.csd(&[sps.clone(), pps.clone()], &[]);
        assert_eq!(csd.len(), 2);
        assert_eq!(csd[0], [&[0, 0, 0, 1][..], &sps].concat());
        assert_eq!(csd[1], [&[0, 0, 0, 1][..], &pps].concat());
        assert!(Kind::H264.csd(&[sps], &[]).is_empty(), "no PPS");
        assert_eq!(Kind::Av1.csd(&[], &[0x81, 1, 2]), vec![vec![0x81, 1, 2]]);
        assert!(Kind::Vp9.csd(&[], &[1, 2]).is_empty());
    }

    #[test]
    fn every_kind_names_a_mime_and_round_trips_its_id() {
        for codec in [CodecId::H264, CodecId::Hevc, CodecId::Vp9, CodecId::Av1] {
            let kind = Kind::of(codec).expect("kind");
            assert_eq!(kind.codec(), codec);
            assert!(kind.mime().to_bytes().starts_with(b"video/"));
            assert!(!kind.software_decoders().is_empty());
        }
        assert!(Kind::of(CodecId::Aac).is_err());
        assert!(is_software("c2.android.avc.decoder"));
        assert!(!is_software("c2.qti.avc.decoder"));
    }

    #[test]
    fn deep_or_wide_streams_are_beyond_the_software_decoders() {
        let mut hvcc = vec![0_u8; 23];
        hvcc[16] = 0xFC | 1;
        hvcc[17] = 0xF8;
        hvcc[18] = 0xF8;
        assert!(!Kind::Hevc.beyond_software(&hvcc), "Main, 8-bit 4:2:0");
        hvcc[17] = 0xF8 | 2;
        hvcc[18] = 0xF8 | 2;
        assert!(Kind::Hevc.beyond_software(&hvcc), "Main 10");
        hvcc[17] = 0xF8;
        hvcc[18] = 0xF8;
        hvcc[16] = 0xFC | 2;
        assert!(Kind::Hevc.beyond_software(&hvcc), "4:2:2");
        assert!(!Kind::Hevc.beyond_software(&hvcc[..10]), "a short record");
        assert!(!Kind::H264.beyond_software(&[1, 100, 0, 40]), "High");
        assert!(Kind::H264.beyond_software(&[1, 110, 0, 40]), "High 10");
        assert!(Kind::H264.beyond_software(&[1, 122, 0, 40]), "High 4:2:2");
        assert!(!Kind::Av1.beyond_software(&[0x81, 0, 0, 0]));
    }
}
