//! Bounds for one hardware session.
//!
//! A driver will allocate for whatever size the caller names. These caps
//! reject that size before a library call, as the Windows and Apple crates
//! do.

use cutlass_decode::CodecError;

/// Longest picture edge this crate will decode or encode.
pub const MAX_EDGE: u32 = 8_192;

/// Most pixels in one picture.
pub const MAX_PIXELS: u64 = 16_777_216;

/// Largest compressed packet this crate will hand to a driver.
pub const MAX_PACKET: usize = 16 * 1024 * 1024;

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

/// Even edges inside the caps. A size past them is a decline, so the
/// factory asks the Rust codec, which has caps of its own.
pub fn video_size(width: u32, height: u32) -> Result<(u32, u32), CodecError> {
    match picture_ok(width, height) {
        Ok(()) if width.is_multiple_of(2) && height.is_multiple_of(2) => Ok((width, height)),
        Ok(()) => Err(CodecError::Unsupported {
            what: "odd video size",
        }),
        Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "picture size",
        }),
        Err(err) => Err(err),
    }
}

/// A packet small enough to copy into a driver buffer.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_inside_the_caps_pass_and_others_decline_or_fail() {
        assert_eq!(video_size(1920, 1080), Ok((1920, 1080)));
        assert!(matches!(
            video_size(MAX_EDGE + 2, 2),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            video_size(3, 2),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            video_size(0, 2),
            Err(CodecError::InvalidApi { .. })
        ));
        assert!(matches!(
            video_size(MAX_EDGE, MAX_EDGE),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn packets_must_hold_bytes_and_fit() {
        assert!(matches!(packet_ok(0), Err(CodecError::InvalidApi { .. })));
        assert!(packet_ok(MAX_PACKET).is_ok());
        assert!(matches!(
            packet_ok(MAX_PACKET + 1),
            Err(CodecError::Limit { .. })
        ));
    }
}
