//! MPEG-4 Part 2 visual access units (ISO/IEC 14496-2 clause 6.2), the
//! framing of `stream_type` 0x10.
//!
//! A picture starts at a VOP start code (`0xB6`); the visual object
//! sequence, visual object, video object, video object layer and group of
//! VOP headers before it open its access unit. A keyframe is an I-VOP. The
//! picture size comes from the video object layer header, and the headers
//! up to the first VOP become the extradata, as an MP4 `esds` carries them.

use cutlass_codec::CodecId;

use super::Facts;
use super::bits::Bits;
use super::video::{Class, Syntax};

const SEQUENCE: u8 = 0xB0;
const VISUAL_OBJECT: u8 = 0xB5;
const GROUP: u8 = 0xB3;
const VOP: u8 = 0xB6;
/// Video object start codes are 0x00 to 0x1F, layers 0x20 to 0x2F.
const LAYER_FIRST: u8 = 0x20;
const LAYER_LAST: u8 = 0x2F;

/// The headers kept for the extradata are short; this bounds a hostile one.
const MAX_HEADER_BYTES: usize = 256;

#[derive(Default)]
pub struct Mpeg4 {
    sequence: Option<Vec<u8>>,
    visual_object: Option<Vec<u8>>,
    object: Option<Vec<u8>>,
    layer: Option<Vec<u8>>,
    size: Option<(u32, u32)>,
}

impl Syntax for Mpeg4 {
    const HEAD: usize = 2;

    fn classify(&mut self, head: &[u8], complete: bool) -> Class {
        let Some(&code) = head.first() else {
            return if complete {
                Class::Inside { whole: false }
            } else {
                Class::NeedMore
            };
        };
        match code {
            VOP => {
                let Some(&byte) = head.get(1) else {
                    return if complete {
                        Class::Inside { whole: false }
                    } else {
                        Class::NeedMore
                    };
                };
                Class::Picture {
                    first: true,
                    pairs: false,
                    key: byte >> 6 == 0,
                }
            }
            SEQUENCE | VISUAL_OBJECT | 0x00..=LAYER_LAST => Class::Opens {
                key: false,
                whole: true,
            },
            GROUP => Class::Opens {
                key: false,
                whole: false,
            },
            _ => Class::Inside { whole: false },
        }
    }

    fn whole(&mut self, unit: &[u8]) {
        let Some(&code) = unit.first() else {
            return;
        };
        let kept = Some(unit[..unit.len().min(MAX_HEADER_BYTES)].to_vec());
        match code {
            SEQUENCE => self.sequence = kept,
            VISUAL_OBJECT => self.visual_object = kept,
            LAYER_FIRST..=LAYER_LAST => {
                if let Some(size) = layer_size(&unit[1..]) {
                    self.size = Some(size);
                    self.layer = kept;
                }
            }
            0x00..LAYER_FIRST => self.object = kept,
            _ => {}
        }
    }

    fn facts(&self) -> Option<Facts> {
        let (width, height) = self.size?;
        let mut extradata = Vec::new();
        for unit in [
            &self.sequence,
            &self.visual_object,
            &self.object,
            &self.layer,
        ]
        .into_iter()
        .flatten()
        {
            extradata.extend_from_slice(&[0, 0, 1]);
            extradata.extend_from_slice(unit);
        }
        Some(Facts::Video {
            codec: CodecId::Mpeg4,
            width,
            height,
            extradata,
        })
    }

    fn reset(&mut self) {}
}

/// The width and height of a rectangular video object layer, from the
/// header bytes after its start code (clause 6.2.3).
fn layer_size(body: &[u8]) -> Option<(u32, u32)> {
    let mut bits = Bits::new(body);
    bits.skip(1)?; // random_accessible_vol
    bits.skip(8)?; // video_object_type_indication
    let mut verid = 1;
    if bits.bit()? {
        verid = bits.bits(4)?;
        bits.skip(3)?; // priority
    }
    if bits.bits(4)? == 0x0F {
        bits.skip(16)?; // extended pixel aspect ratio
    }
    if bits.bit()? {
        // vol_control_parameters: chroma_format, low_delay, vbv_parameters.
        bits.skip(3)?;
        if bits.bit()? {
            bits.skip(79)?;
        }
    }
    let shape = bits.bits(2)?;
    if shape == 3 && verid != 1 {
        bits.skip(4)?;
    }
    // Only a rectangular layer states its size.
    if shape != 0 {
        return None;
    }
    bits.skip(1)?; // marker
    let resolution = bits.bits(16)?;
    bits.skip(1)?; // marker
    if bits.bit()? {
        // fixed_vop_time_increment, as many bits as the resolution needs.
        let width = (u32::BITS - resolution.saturating_sub(1).leading_zeros()).max(1);
        bits.skip(width as usize)?;
    }
    bits.skip(1)?;
    let width = bits.bits(13)?;
    bits.skip(1)?;
    let height = bits.bits(13)?;
    (width > 0 && height > 0).then_some((width, height))
}

#[cfg(test)]
mod tests {
    use super::{Mpeg4, layer_size};
    use crate::es::Facts;
    use crate::es::video::{Class, Syntax};

    /// The layer header FFmpeg 8.1.1's `mpeg4` encoder writes for 96x64 at
    /// 25 fps, after its start code `00 00 01 20`.
    const LAYER: [u8; 11] = [
        0x00, 0xC4, 0x8D, 0x88, 0x00, 0xCD, 0x03, 0x04, 0x08, 0x14, 0x43,
    ];

    #[test]
    fn the_layer_header_gives_the_size() {
        assert_eq!(layer_size(&LAYER), Some((96, 64)));
        assert_eq!(layer_size(&LAYER[..4]), None);
    }

    #[test]
    fn an_i_vop_is_a_keyframe_and_headers_open_the_unit() {
        let mut mpeg4 = Mpeg4::default();
        assert_eq!(
            mpeg4.classify(&[0xB6, 0x10], false),
            Class::Picture {
                first: true,
                pairs: false,
                key: true
            }
        );
        assert!(matches!(
            mpeg4.classify(&[0xB6, 0x50], false),
            Class::Picture { key: false, .. }
        ));
        assert_eq!(
            mpeg4.classify(&[0x20], false),
            Class::Opens {
                key: false,
                whole: true
            }
        );
        let mut layer = vec![0x20];
        layer.extend_from_slice(&LAYER);
        mpeg4.whole(&layer);
        let Some(Facts::Video {
            width,
            height,
            extradata,
            ..
        }) = mpeg4.facts()
        else {
            panic!("no facts");
        };
        assert_eq!((width, height), (96, 64));
        assert_eq!(&extradata[..4], &[0, 0, 1, 0x20]);
    }
}
