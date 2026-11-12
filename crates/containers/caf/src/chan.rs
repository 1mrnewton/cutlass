//! The `chan` chunk: `CAFChannelLayout` (CoreAudioTypes `AudioChannelLayout`).
//!
//! Twelve bytes, then `mNumberChannelDescriptions` descriptions of 20 bytes:
//! label, flags, and three `Float32` coordinates, all big-endian. The
//! coordinates are kept as their bit patterns so the layout compares and
//! round-trips exactly.

use cutlass_container::OpenError;

use crate::bytes::{be_u32, len_u64};
use crate::error::{hostile, malformed, truncated};
use crate::limits::MAX_CHANNELS;

/// `kAudioChannelLayoutTag_UseChannelDescriptions`.
pub const TAG_USE_DESCRIPTIONS: u32 = 0;
/// `kAudioChannelLayoutTag_UseChannelBitmap`.
pub const TAG_USE_BITMAP: u32 = 1 << 16;
/// `kAudioChannelLayoutTag_Mono`.
pub const TAG_MONO: u32 = (100 << 16) | 1;
/// `kAudioChannelLayoutTag_Stereo`.
pub const TAG_STEREO: u32 = (101 << 16) | 2;

const HEAD_BYTES: usize = 12;
const DESCRIPTION_BYTES: usize = 20;

/// One `AudioChannelDescription`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelDescription {
    pub label: u32,
    pub flags: u32,
    /// `mCoordinates`, as `f32` bit patterns.
    pub coordinates: [u32; 3],
}

/// A channel layout, as the file stored it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChannelLayout {
    /// `mChannelLayoutTag`: the layout in the high 16 bits and the channel
    /// count in the low 16, or one of the two "use" tags.
    pub tag: u32,
    /// `mChannelBitmap`, read when the tag is [`TAG_USE_BITMAP`].
    pub bitmap: u32,
    pub descriptions: Vec<ChannelDescription>,
}

impl ChannelLayout {
    /// A layout named only by its tag.
    #[must_use]
    pub const fn tagged(tag: u32) -> Self {
        Self {
            tag,
            bitmap: 0,
            descriptions: Vec::new(),
        }
    }

    pub(crate) fn parse(body: &[u8]) -> Result<Self, OpenError> {
        if body.len() < HEAD_BYTES {
            return Err(truncated(len_u64(HEAD_BYTES), len_u64(body.len())));
        }
        let field = |at| be_u32(body, at).ok_or_else(|| malformed("chan is truncated"));
        let tag = field(0)?;
        let bitmap = field(4)?;
        let count = field(8)?;
        if count > MAX_CHANNELS {
            return Err(hostile(u64::from(count), u64::from(MAX_CHANNELS)));
        }
        let count = usize::try_from(count).map_err(|_| malformed("chan count does not fit"))?;
        let need = HEAD_BYTES + count * DESCRIPTION_BYTES;
        if body.len() < need {
            return Err(truncated(len_u64(need), len_u64(body.len())));
        }
        let mut descriptions = Vec::with_capacity(count);
        for index in 0..count {
            let at = HEAD_BYTES + index * DESCRIPTION_BYTES;
            descriptions.push(ChannelDescription {
                label: field(at)?,
                flags: field(at + 4)?,
                coordinates: [field(at + 8)?, field(at + 12)?, field(at + 16)?],
            });
        }
        Ok(Self {
            tag,
            bitmap,
            descriptions,
        })
    }

    /// The chunk body.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEAD_BYTES + self.descriptions.len() * DESCRIPTION_BYTES);
        out.extend_from_slice(&self.tag.to_be_bytes());
        out.extend_from_slice(&self.bitmap.to_be_bytes());
        let count = u32::try_from(self.descriptions.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_be_bytes());
        for description in &self.descriptions {
            out.extend_from_slice(&description.label.to_be_bytes());
            out.extend_from_slice(&description.flags.to_be_bytes());
            for coordinate in description.coordinates {
                out.extend_from_slice(&coordinate.to_be_bytes());
            }
        }
        out
    }

    /// The layout this crate writes when the caller names none: mono and
    /// stereo, as AudioToolbox and libavformat write them.
    #[must_use]
    pub const fn default_for(channels: u16) -> Option<Self> {
        match channels {
            1 => Some(Self::tagged(TAG_MONO)),
            2 => Some(Self::tagged(TAG_STEREO)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelDescription, ChannelLayout, TAG_STEREO};
    use cutlass_container::OpenError;

    #[test]
    fn a_tagged_layout_and_a_described_layout_round_trip() {
        let stereo = ChannelLayout::default_for(2).expect("stereo");
        let bytes = stereo.to_bytes();
        assert_eq!(bytes, [0, 0x65, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ChannelLayout::parse(&bytes).expect("parse"), stereo);
        assert_eq!(stereo.tag, TAG_STEREO);
        let described = ChannelLayout {
            tag: 0,
            bitmap: 0,
            descriptions: vec![
                ChannelDescription {
                    label: 1,
                    flags: 0,
                    coordinates: [0, 0, 0],
                },
                ChannelDescription {
                    label: 2,
                    flags: 1,
                    coordinates: [1.0_f32.to_bits(), 0, (-0.5_f32).to_bits()],
                },
            ],
        };
        let bytes = described.to_bytes();
        assert_eq!(bytes.len(), 52);
        assert_eq!(ChannelLayout::parse(&bytes).expect("parse"), described);
        assert_eq!(ChannelLayout::default_for(6), None);
    }

    #[test]
    fn a_count_that_lies_is_refused() {
        let mut bytes = ChannelLayout::tagged(0).to_bytes();
        bytes[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            ChannelLayout::parse(&bytes),
            Err(OpenError::HostileSize { .. })
        ));
        bytes[8..12].copy_from_slice(&2_u32.to_be_bytes());
        assert!(matches!(
            ChannelLayout::parse(&bytes),
            Err(OpenError::Truncated { .. })
        ));
        assert!(ChannelLayout::parse(&bytes[..11]).is_err());
    }
}
