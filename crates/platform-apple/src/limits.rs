//! Bounds for one Apple codec session.
//!
//! VideoToolbox, AudioToolbox, and ImageIO will allocate for whatever size
//! the caller names. These caps reject that size before a framework call.

use cutlass_decode::CodecError;

/// Longest picture edge this crate will decode or encode.
pub const MAX_EDGE: u32 = 8_192;

/// Most pixels in one picture.
pub const MAX_PIXELS: u64 = 16_777_216;

/// Largest compressed packet this crate will hand to a framework.
pub const MAX_PACKET: usize = 16 * 1024 * 1024;

/// Most audio channels. Matches the shared layout cap.
pub const MAX_CHANNELS: usize = 8;

/// Highest sample rate, in hertz.
pub const MAX_RATE: u32 = 192_000;

/// A picture whose edges and pixel count sit inside the caps.
pub fn picture_ok(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(CodecError::Limit {
            what: "picture edge",
        });
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(CodecError::Limit {
            what: "picture pixels",
        })?;
    if pixels > MAX_PIXELS {
        return Err(CodecError::Limit {
            what: "picture pixels",
        });
    }
    Ok(())
}

/// Even edges, for codecs that sample chroma on a 2-pixel grid.
pub fn even_picture(width: u32, height: u32) -> Result<(), CodecError> {
    picture_ok(width, height)?;
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(CodecError::Unsupported {
            what: "odd video size",
        });
    }
    Ok(())
}

/// A packet small enough to copy into a framework buffer.
pub fn packet_ok(len: usize) -> Result<(), CodecError> {
    if len == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty packet",
        });
    }
    if len > MAX_PACKET {
        return Err(CodecError::Limit {
            what: "packet bytes",
        });
    }
    Ok(())
}

/// Sample rate and channel count AudioToolbox will be asked to open.
pub fn audio_ok(rate: u32, channels: usize) -> Result<(), CodecError> {
    if rate == 0 || channels == 0 {
        return Err(CodecError::InvalidApi {
            what: "audio format",
        });
    }
    if rate > MAX_RATE {
        return Err(CodecError::Limit {
            what: "sample rate",
        });
    }
    if channels > MAX_CHANNELS {
        return Err(CodecError::Limit { what: "channels" });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_inside_the_cap_is_accepted() {
        assert!(picture_ok(1, 1).is_ok());
        assert!(picture_ok(MAX_EDGE, 1).is_ok());
        assert!(even_picture(2, 2).is_ok());
    }

    #[test]
    fn an_empty_edge_is_an_invalid_call() {
        assert_eq!(
            picture_ok(0, 1),
            Err(CodecError::InvalidApi {
                what: "empty picture",
            })
        );
    }

    #[test]
    fn an_edge_past_the_cap_is_a_limit() {
        assert_eq!(
            picture_ok(MAX_EDGE + 1, 1),
            Err(CodecError::Limit {
                what: "picture edge"
            })
        );
    }

    #[test]
    fn too_many_pixels_is_a_limit() {
        let width = MAX_EDGE;
        let height = 4_096u32;
        assert!(u64::from(width) * u64::from(height) > MAX_PIXELS);
        assert_eq!(
            picture_ok(width, height),
            Err(CodecError::Limit {
                what: "picture pixels",
            })
        );
    }

    #[test]
    fn an_odd_video_size_is_unsupported() {
        assert_eq!(
            even_picture(3, 2),
            Err(CodecError::Unsupported {
                what: "odd video size",
            })
        );
    }

    #[test]
    fn an_empty_packet_is_an_invalid_call() {
        assert_eq!(
            packet_ok(0),
            Err(CodecError::InvalidApi {
                what: "empty packet",
            })
        );
    }

    #[test]
    fn a_packet_past_the_cap_is_a_limit() {
        assert_eq!(
            packet_ok(MAX_PACKET + 1),
            Err(CodecError::Limit {
                what: "packet bytes"
            })
        );
        assert!(packet_ok(MAX_PACKET).is_ok());
    }

    #[test]
    fn audio_rejects_a_zero_rate_and_a_rate_past_the_cap() {
        assert_eq!(
            audio_ok(0, 1),
            Err(CodecError::InvalidApi {
                what: "audio format",
            })
        );
        assert_eq!(
            audio_ok(MAX_RATE + 1, 1),
            Err(CodecError::Limit {
                what: "sample rate"
            })
        );
        assert_eq!(
            audio_ok(48_000, MAX_CHANNELS + 1),
            Err(CodecError::Limit { what: "channels" })
        );
        assert!(audio_ok(48_000, MAX_CHANNELS).is_ok());
    }
}
