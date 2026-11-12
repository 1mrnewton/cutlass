//! MPEG-1 and MPEG-2 video access units (ISO/IEC 13818-2 clause 6.2).
//!
//! A picture starts at a picture start code; a sequence or GOP header
//! before it opens its access unit. The two fields of a field-coded frame
//! stay in one access unit, as FFmpeg's parser merges them: a picture
//! header after a lone field is that field's pair. A keyframe is an I
//! picture.

use cutlass_codec::CodecId;

use super::Facts;
use super::video::{Class, Syntax};

const PICTURE: u8 = 0x00;
const SEQUENCE: u8 = 0xB3;
const EXTENSION: u8 = 0xB5;
const GOP: u8 = 0xB8;
const SEQUENCE_EXTENSION: u8 = 1;
const PICTURE_CODING_EXTENSION: u8 = 8;

/// The longest sequence header: 12 bytes and two 64-byte matrices.
const MAX_SEQUENCE_BYTES: usize = 1 + 11 + 128;

#[derive(Default)]
pub struct Mpeg2 {
    sequence: Option<Vec<u8>>,
    extension: Option<Vec<u8>>,
    size: Option<(u32, u32)>,
    size_extension: (u32, u32),
    /// The picture in progress is a field whose pair has not come.
    lone_field: bool,
    /// The picture in progress is the second field of a pair.
    second_field: bool,
}

impl Syntax for Mpeg2 {
    const HEAD: usize = 4;

    fn classify(&mut self, head: &[u8], complete: bool) -> Class {
        let need = |len: usize| {
            if complete {
                Class::Inside { whole: false }
            } else if head.len() < len {
                Class::NeedMore
            } else {
                Class::Inside { whole: false }
            }
        };
        let Some(&code) = head.first() else {
            return need(1);
        };
        match code {
            PICTURE => {
                let Some(&byte) = head.get(2) else {
                    return need(3);
                };
                let coding_type = (byte >> 3) & 7;
                let pairs = self.lone_field;
                self.lone_field = false;
                self.second_field = pairs;
                Class::Picture {
                    first: true,
                    pairs,
                    key: coding_type == 1,
                }
            }
            SEQUENCE => Class::Opens {
                key: false,
                whole: true,
            },
            GOP => Class::Opens {
                key: false,
                whole: false,
            },
            EXTENSION => {
                let Some(&byte) = head.get(1) else {
                    return need(2);
                };
                match byte >> 4 {
                    PICTURE_CODING_EXTENSION => {
                        let Some(&flags) = head.get(3) else {
                            return need(4);
                        };
                        let structure = flags & 3;
                        // 1 and 2 are a top and a bottom field, 3 a frame.
                        if structure != 3 && !self.second_field {
                            self.lone_field = true;
                        }
                        Class::Inside { whole: false }
                    }
                    SEQUENCE_EXTENSION => Class::Inside { whole: true },
                    _ => Class::Inside { whole: false },
                }
            }
            _ => Class::Inside { whole: false },
        }
    }

    fn whole(&mut self, unit: &[u8]) {
        match unit {
            [SEQUENCE, a, b, c, ..] => {
                let width = (u32::from(*a) << 4) | u32::from(b >> 4);
                let height = (u32::from(b & 0x0F) << 8) | u32::from(*c);
                self.size = Some((width, height));
                self.sequence = Some(unit[..unit.len().min(MAX_SEQUENCE_BYTES)].to_vec());
                self.extension = None;
                self.size_extension = (0, 0);
            }
            [EXTENSION, _, b2, b3, ..] => {
                let horizontal = u32::from(b2 & 1) << 1 | u32::from(b3 >> 7);
                let vertical = u32::from((b3 >> 5) & 3);
                self.size_extension = (horizontal, vertical);
                self.extension = Some(unit[..unit.len().min(16)].to_vec());
            }
            _ => {}
        }
    }

    fn facts(&self) -> Option<Facts> {
        let (width, height) = self.size?;
        let width = width | (self.size_extension.0 << 12);
        let height = height | (self.size_extension.1 << 12);
        let mut extradata = Vec::new();
        for unit in [&self.sequence, &self.extension].into_iter().flatten() {
            extradata.extend_from_slice(&[0, 0, 1]);
            extradata.extend_from_slice(unit);
        }
        Some(Facts::Video {
            codec: CodecId::Mpeg2,
            width,
            height,
            extradata,
        })
    }

    fn reset(&mut self) {
        self.lone_field = false;
        self.second_field = false;
    }
}

#[cfg(test)]
mod tests {
    use super::Mpeg2;
    use crate::es::Facts;
    use crate::es::video::{Class, Syntax};

    #[test]
    fn a_field_pair_is_one_picture() {
        let mut mpeg2 = Mpeg2::default();
        // An I picture: temporal_reference 0, picture_coding_type 1.
        let first = mpeg2.classify(&[0x00, 0x00, 0x08, 0x00], false);
        assert_eq!(
            first,
            Class::Picture {
                first: true,
                pairs: false,
                key: true
            }
        );
        // Its coding extension says top field.
        mpeg2.classify(&[0xB5, 0x8F, 0xFF, 0x01], false);
        // The next picture header is the bottom field: it pairs.
        let second = mpeg2.classify(&[0x00, 0x00, 0x10, 0x00], false);
        assert!(matches!(second, Class::Picture { pairs: true, .. }));
        mpeg2.classify(&[0xB5, 0x8F, 0xFF, 0x02], false);
        // Then a new frame.
        let third = mpeg2.classify(&[0x00, 0x40, 0x10, 0x00], false);
        assert!(matches!(third, Class::Picture { pairs: false, .. }));
    }

    #[test]
    fn the_sequence_header_and_extension_give_the_size() {
        let mut mpeg2 = Mpeg2::default();
        // 720x576.
        mpeg2.whole(&[0xB3, 0x2D, 0x02, 0x40, 0x33, 0xFF, 0xFF, 0xE0, 0x18]);
        mpeg2.whole(&[0xB5, 0x14, 0x8A, 0x00, 0x01, 0x00]);
        let Some(Facts::Video { width, height, .. }) = mpeg2.facts() else {
            panic!("no facts");
        };
        assert_eq!((width, height), (720, 576));
    }
}
