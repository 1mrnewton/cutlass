//! The magic cookie: `ALACSpecificConfig`, and the optional channel layout.
//!
//! The layout follows Apple's `ALACMagicCookieDescription.txt`. The config is
//! 24 big-endian bytes. A 24-byte `chan` atom may follow it. A CAF `kuki`
//! chunk holds that cookie as it is. An MP4 `alac` box holds a four-byte
//! version and flags, then the cookie. Older writers wrapped it in a `frma`
//! atom and a full `alac` atom. [`Cookie::parse`] accepts each of those forms.
//! [`Cookie::to_bytes`] writes the bare cookie, the form this workspace
//! stores in `CodecParams::extra` and `StreamParams::extradata`.

use cutlass_decode::{Channel, ChannelLayout, CodecError};

/// Bytes of `ALACSpecificConfig`.
pub const CONFIG_BYTES: usize = 24;
/// Bytes of the `chan` atom that may follow the config.
pub const CHANNEL_ATOM_BYTES: usize = 24;
/// Most channels one stream carries.
pub const MAX_CHANNELS: u8 = 8;
/// Longest frame this crate decodes or encodes, in samples per channel.
///
/// The format sets no limit. A tiny compressed frame can describe this many
/// samples of silence, so the bound is what one packet may expand into:
/// 2^18 samples by 8 channels by 4 bytes is 8 MiB.
pub const MAX_FRAME_LENGTH: u32 = 1 << 18;
/// The reference encoder's frame length.
pub const DEFAULT_FRAME_LENGTH: u32 = 4096;

/// Adaptation defaults the reference encoder writes: `PB0`, `MB0`, `KB0`.
pub const DEFAULT_PB: u8 = 40;
pub const DEFAULT_MB: u8 = 10;
pub const DEFAULT_KB: u8 = 14;
/// `MAX_RUN_DEFAULT`. Decoders do not read it.
pub const DEFAULT_MAX_RUN: u16 = 255;

/// One channel-layout tag per channel count, from `ALACAudioTypes.h`.
/// The high 16 bits name the layout and the low 16 bits are the count.
/// Three through eight are the tags AudioToolbox writes in an MP4 `chan`
/// box; three is `MPEG_3_0_B` (C L R), not `MPEG_3_0_A` (L R C).
pub const LAYOUT_TAGS: [u32; 8] = [
    (100 << 16) | 1,
    (101 << 16) | 2,
    (114 << 16) | 3,
    (116 << 16) | 4,
    (120 << 16) | 5,
    (124 << 16) | 6,
    (142 << 16) | 7,
    (127 << 16) | 8,
];

/// `ALACSpecificConfig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AlacConfig {
    /// Samples per channel in a frame that does not signal its own length.
    pub frame_length: u32,
    /// Must be 0.
    pub compatible_version: u8,
    /// 16, 20, 24, or 32.
    pub bit_depth: u8,
    /// Adaptation rate of the Golomb mean, scaled by each channel's factor.
    pub pb: u8,
    /// Starting Golomb mean.
    pub mb: u8,
    /// Largest Golomb parameter.
    pub kb: u8,
    /// 1 through 8.
    pub channels: u8,
    /// Unused by decoders.
    pub max_run: u16,
    /// Largest packet in the stream, or 0 when unknown.
    pub max_frame_bytes: u32,
    /// Average bits per second, or 0 when unknown.
    pub avg_bit_rate: u32,
    pub sample_rate: u32,
}

impl AlacConfig {
    /// The reference encoder's config for a stream of this shape.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8, frame_length: u32) -> Self {
        Self {
            frame_length,
            compatible_version: 0,
            bit_depth,
            pb: DEFAULT_PB,
            mb: DEFAULT_MB,
            kb: DEFAULT_KB,
            channels,
            max_run: DEFAULT_MAX_RUN,
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            sample_rate,
        }
    }

    fn read(bytes: &[u8]) -> Result<Self, CodecError> {
        let Some(bytes) = bytes.get(..CONFIG_BYTES) else {
            return Err(CodecError::Truncated {
                at_bit: u64::try_from(bytes.len()).unwrap_or(u64::MAX) * 8,
                what: "alac specific config",
            });
        };
        let word = |at: usize| {
            u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        Ok(Self {
            frame_length: word(0),
            compatible_version: bytes[4],
            bit_depth: bytes[5],
            pb: bytes[6],
            mb: bytes[7],
            kb: bytes[8],
            channels: bytes[9],
            max_run: u16::from_be_bytes([bytes[10], bytes[11]]),
            max_frame_bytes: word(12),
            avg_bit_rate: word(16),
            sample_rate: word(20),
        })
    }

    /// The 24 config bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; CONFIG_BYTES] {
        let mut out = [0u8; CONFIG_BYTES];
        out[0..4].copy_from_slice(&self.frame_length.to_be_bytes());
        out[4] = self.compatible_version;
        out[5] = self.bit_depth;
        out[6] = self.pb;
        out[7] = self.mb;
        out[8] = self.kb;
        out[9] = self.channels;
        out[10..12].copy_from_slice(&self.max_run.to_be_bytes());
        out[12..16].copy_from_slice(&self.max_frame_bytes.to_be_bytes());
        out[16..20].copy_from_slice(&self.avg_bit_rate.to_be_bytes());
        out[20..24].copy_from_slice(&self.sample_rate.to_be_bytes());
        out
    }

    /// Refuse a config this crate cannot decode, before any frame.
    pub fn validate(&self) -> Result<(), CodecError> {
        if self.compatible_version != 0 {
            return Err(CodecError::Unsupported {
                what: "alac compatible version",
            });
        }
        if self.frame_length == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "alac frame length",
            });
        }
        if self.frame_length > MAX_FRAME_LENGTH {
            return Err(CodecError::Limit {
                what: "alac frame length",
            });
        }
        if !matches!(self.bit_depth, 16 | 20 | 24 | 32) {
            return Err(CodecError::Unsupported {
                what: "alac bit depth",
            });
        }
        if self.channels == 0 {
            return Err(CodecError::Malformed {
                at_bit: 72,
                what: "alac channel count",
            });
        }
        if self.channels > MAX_CHANNELS {
            return Err(CodecError::Unsupported {
                what: "alac channel count",
            });
        }
        if self.kb > 31 {
            return Err(CodecError::Malformed {
                at_bit: 64,
                what: "alac golomb limit",
            });
        }
        Ok(())
    }

    /// `true` when `other` decodes the same way: only the size and rate
    /// statistics differ.
    #[must_use]
    pub fn decodes_like(&self, other: &Self) -> bool {
        let strip = |config: &Self| Self {
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            max_run: 0,
            ..*config
        };
        strip(self) == strip(other)
    }
}

/// The cookie: the config, and the layout tag when a `chan` atom was present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cookie {
    pub config: AlacConfig,
    pub layout_tag: Option<u32>,
}

fn atom_is(bytes: &[u8], kind: &[u8; 4]) -> bool {
    bytes.get(4..8) == Some(kind.as_slice())
}

impl Cookie {
    /// A cookie for `config`. More than two channels carry a `chan` atom
    /// with the reference layout for that count, as the reference does.
    #[must_use]
    pub fn new(config: AlacConfig) -> Self {
        let layout_tag = if config.channels > 2 {
            LAYOUT_TAGS.get(usize::from(config.channels) - 1).copied()
        } else {
            None
        };
        Self { config, layout_tag }
    }

    /// Read any cookie form named in the module notes. The config is
    /// validated; the `chan` tag must name the config's channel count.
    pub fn parse(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut rest = bytes;
        if rest.len() >= 12 && atom_is(rest, b"frma") {
            rest = &rest[12..];
        }
        if rest.len() >= 12 && atom_is(rest, b"alac") {
            rest = &rest[12..];
        } else if rest.len() >= CONFIG_BYTES + 4 && rest[..4] == [0, 0, 0, 0] {
            // An MP4 `alac` box payload: version and flags, then the cookie.
            // A bare config cannot start with zero: that is a frame length of 0.
            rest = &rest[4..];
        }
        let config = AlacConfig::read(rest)?;
        config.validate()?;
        let after = &rest[CONFIG_BYTES..];
        let mut layout_tag = None;
        if after.len() >= CHANNEL_ATOM_BYTES && atom_is(after, b"chan") {
            let size = u32::from_be_bytes([after[0], after[1], after[2], after[3]]);
            if size < 24 {
                return Err(CodecError::Malformed {
                    at_bit: (CONFIG_BYTES as u64) * 8,
                    what: "alac channel atom size",
                });
            }
            let tag = u32::from_be_bytes([after[12], after[13], after[14], after[15]]);
            if tag & 0xffff != u32::from(config.channels) {
                return Err(CodecError::Malformed {
                    at_bit: (CONFIG_BYTES as u64 + 12) * 8,
                    what: "alac channel layout count",
                });
            }
            layout_tag = Some(tag);
        }
        Ok(Self { config, layout_tag })
    }

    /// The bare cookie: the config, then the `chan` atom when there is a tag.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.config.to_bytes().to_vec();
        if let Some(tag) = self.layout_tag {
            out.extend_from_slice(&24u32.to_be_bytes());
            out.extend_from_slice(b"chan");
            out.extend_from_slice(&[0; 4]);
            out.extend_from_slice(&tag.to_be_bytes());
            out.extend_from_slice(&[0; 8]);
        }
        out
    }

    /// Speaker labels in the order the frame stores channels.
    ///
    /// A known tag gives its labels. Without a tag, or with a tag outside the
    /// reference set, the reference order for the channel count is used:
    /// the description calls that order unspecified, and FFmpeg and Apple's
    /// own encoder both use these labels.
    #[must_use]
    pub fn channel_layout(&self) -> ChannelLayout {
        default_layout(self.config.channels)
    }
}

/// The reference layout for `channels`, in frame order: C; L R; C L R;
/// C L R Cs; C L R Ls Rs; C L R Ls Rs LFE; C L R Ls Rs Cs LFE;
/// C Lc Rc L R Ls Rs LFE.
#[must_use]
pub fn default_layout(channels: u8) -> ChannelLayout {
    use Channel::{
        BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontLeftOfCenter, FrontRight,
        FrontRightOfCenter, Lfe,
    };
    let labels = match channels {
        2 => vec![FrontLeft, FrontRight],
        3 => vec![FrontCenter, FrontLeft, FrontRight],
        4 => vec![FrontCenter, FrontLeft, FrontRight, BackCenter],
        5 => vec![FrontCenter, FrontLeft, FrontRight, BackLeft, BackRight],
        6 => vec![FrontCenter, FrontLeft, FrontRight, BackLeft, BackRight, Lfe],
        7 => vec![
            FrontCenter,
            FrontLeft,
            FrontRight,
            BackLeft,
            BackRight,
            BackCenter,
            Lfe,
        ],
        8 => vec![
            FrontCenter,
            FrontLeftOfCenter,
            FrontRightOfCenter,
            FrontLeft,
            FrontRight,
            BackLeft,
            BackRight,
            Lfe,
        ],
        _ => vec![FrontCenter],
    };
    ChannelLayout::new(labels).unwrap_or_else(|_| ChannelLayout::mono())
}

#[cfg(test)]
mod tests {
    use super::{AlacConfig, Cookie, LAYOUT_TAGS, default_layout};
    use cutlass_decode::{Channel, ChannelLayout, CodecError};

    fn stereo() -> AlacConfig {
        AlacConfig::new(44_100, 2, 16, 4096)
    }

    #[test]
    fn every_container_form_reads_the_same_config() {
        let bare = stereo().to_bytes().to_vec();
        let mut versioned = vec![0, 0, 0, 0];
        versioned.extend(&bare);
        let mut atom = vec![0, 0, 0, 36];
        atom.extend(b"alac");
        atom.extend(&versioned);
        let mut legacy = vec![0, 0, 0, 12];
        legacy.extend(b"frmaalac");
        legacy.extend(&atom);
        legacy.extend([0, 0, 0, 8, 0, 0, 0, 0]);
        for form in [&bare, &versioned, &atom, &legacy] {
            let cookie = Cookie::parse(form).unwrap();
            assert_eq!(cookie.config, stereo());
            assert_eq!(cookie.layout_tag, None);
        }
    }

    #[test]
    fn apple_and_ffmpeg_cookies_parse() {
        // afconvert, 16-bit stereo, and the ffmpeg encoder's cookie.
        let apple = [
            0x00, 0x00, 0x10, 0x00, 0x00, 0x10, 0x28, 0x0a, 0x0e, 0x02, 0x00, 0xff, 0x00, 0x00,
            0x13, 0x57, 0x00, 0x04, 0x91, 0xf3, 0x00, 0x00, 0xac, 0x44,
        ];
        let cookie = Cookie::parse(&apple).unwrap();
        assert_eq!(cookie.config.max_frame_bytes, 4951);
        assert_eq!(cookie.config.avg_bit_rate, 299_507);
        assert_eq!(cookie.to_bytes(), apple);
        let ffmpeg = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x10, 0x28, 0x0a, 0x0e, 0x02,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x04, 0x00, 0x15, 0x88, 0x80, 0x00, 0x00, 0xac, 0x44,
        ];
        let cookie = Cookie::parse(&ffmpeg).unwrap();
        assert_eq!(cookie.config.max_run, 0);
        assert_eq!(cookie.config.sample_rate, 44_100);
    }

    #[test]
    fn a_channel_atom_follows_and_round_trips() {
        let config = AlacConfig::new(48_000, 6, 24, 4096);
        let cookie = Cookie::new(config);
        assert_eq!(cookie.layout_tag, Some(LAYOUT_TAGS[5]));
        let bytes = cookie.to_bytes();
        assert_eq!(bytes.len(), 48);
        assert_eq!(Cookie::parse(&bytes).unwrap(), cookie);
        assert_eq!(cookie.channel_layout(), ChannelLayout::five_one());
        let mut wrong = bytes.clone();
        wrong[24 + 15] = 5;
        assert!(matches!(
            Cookie::parse(&wrong),
            Err(CodecError::Malformed { .. })
        ));
    }

    #[test]
    fn a_short_or_impossible_config_is_refused() {
        let bytes = stereo().to_bytes();
        for len in 0..24 {
            assert!(matches!(
                Cookie::parse(&bytes[..len]),
                Err(CodecError::Truncated { .. })
            ));
        }
        let refuse = |edit: fn(&mut AlacConfig)| {
            let mut config = stereo();
            edit(&mut config);
            Cookie::parse(&config.to_bytes())
        };
        assert!(matches!(
            refuse(|c| c.compatible_version = 1),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            refuse(|c| c.bit_depth = 8),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            refuse(|c| c.bit_depth = 33),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            refuse(|c| c.channels = 0),
            Err(CodecError::Malformed { .. })
        ));
        assert!(matches!(
            refuse(|c| c.channels = 9),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            refuse(|c| c.kb = 32),
            Err(CodecError::Malformed { .. })
        ));
        assert!(matches!(
            refuse(|c| c.frame_length = u32::MAX),
            Err(CodecError::Limit { .. })
        ));
    }

    #[test]
    fn layouts_follow_the_reference_order() {
        assert_eq!(default_layout(1), ChannelLayout::mono());
        assert_eq!(default_layout(2), ChannelLayout::stereo());
        assert_eq!(default_layout(8), ChannelLayout::seven_one_wide());
        assert_eq!(default_layout(7).channels()[5], Channel::BackCenter);
        for count in 1..=8u8 {
            assert_eq!(default_layout(count).len(), usize::from(count));
        }
    }
}
