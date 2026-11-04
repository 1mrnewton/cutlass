//! Framework status codes, mapped onto [`CodecError`](cutlass_decode::CodecError).
//!
//! Opening a codec the framework does not ship is [`CodecError::Unsupported`],
//! so the factory can ask the Rust opener. A unit the framework rejects after
//! the session exists is [`CodecError::Malformed`].

use cutlass_decode::CodecError;

/// `noErr`.
pub const OK: i32 = 0;

/// `paramErr`.
pub const PARAM: i32 = -50;

/// VideoToolbox has no decoder for this format.
pub const NO_VIDEO_DECODER: i32 = -12_913;

/// VideoToolbox has no encoder for this format.
pub const NO_VIDEO_ENCODER: i32 = -12_915;

/// The decoder rejected the sample's layout.
pub const BAD_VIDEO_FORMAT: i32 = -12_909;

/// The decoder rejected the sample bytes.
pub const BAD_VIDEO_DATA: i32 = -12_908;

/// The property is not on this session.
pub const PROPERTY_UNSUPPORTED: i32 = -12_900;

/// The session was invalidated. iOS does this to a hardware session when
/// the app leaves the foreground.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub const INVALID_SESSION: i32 = -12_903;

/// The hardware decoder stopped working.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub const DECODER_MALFUNCTION: i32 = -12_911;

/// AudioConverter four-character code `fmt?`.
pub const AUDIO_FORMAT: i32 = i32::from_be_bytes(*b"fmt?");

/// A status from opening a session.
///
/// Anything other than success declines the opener. The bytes were not a
/// sample yet, so this is not a malformed unit.
pub fn open_status(code: i32, what: &'static str) -> Result<(), CodecError> {
    if code == OK {
        Ok(())
    } else {
        Err(CodecError::Unsupported { what })
    }
}

/// The session can decode nothing more and has to be opened again.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub fn session_lost(code: i32) -> bool {
    matches!(code, INVALID_SESSION | DECODER_MALFUNCTION)
}

/// A status from decoding or encoding one unit that was already accepted.
pub fn unit_status(code: i32, what: &'static str) -> Result<(), CodecError> {
    match code {
        OK => Ok(()),
        NO_VIDEO_DECODER | NO_VIDEO_ENCODER | BAD_VIDEO_FORMAT | AUDIO_FORMAT | PARAM
        | PROPERTY_UNSUPPORTED => Err(CodecError::Unsupported { what }),
        BAD_VIDEO_DATA => Err(CodecError::Malformed { at_bit: 0, what }),
        _ => Err(CodecError::Malformed { at_bit: 0, what }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_invalidated_or_broken_session_is_lost() {
        assert!(session_lost(INVALID_SESSION));
        assert!(session_lost(DECODER_MALFUNCTION));
        assert!(!session_lost(OK));
        assert!(!session_lost(BAD_VIDEO_DATA));
    }

    #[test]
    fn success_opens_and_completes_a_unit() {
        assert!(open_status(OK, "videotoolbox").is_ok());
        assert!(unit_status(OK, "videotoolbox").is_ok());
    }

    #[test]
    fn an_open_failure_declines_the_opener() {
        assert_eq!(
            open_status(NO_VIDEO_ENCODER, "hevc encode"),
            Err(CodecError::Unsupported {
                what: "hevc encode"
            })
        );
        assert_eq!(
            open_status(AUDIO_FORMAT, "vorbis"),
            Err(CodecError::Unsupported { what: "vorbis" })
        );
    }

    #[test]
    fn bad_video_bytes_are_malformed() {
        assert_eq!(
            unit_status(BAD_VIDEO_DATA, "h264"),
            Err(CodecError::Malformed {
                at_bit: 0,
                what: "h264",
            })
        );
    }

    #[test]
    fn a_missing_codec_on_a_unit_is_unsupported() {
        assert_eq!(
            unit_status(NO_VIDEO_DECODER, "vp8"),
            Err(CodecError::Unsupported { what: "vp8" })
        );
        assert_eq!(
            unit_status(PROPERTY_UNSUPPORTED, "profile"),
            Err(CodecError::Unsupported { what: "profile" })
        );
    }

    #[test]
    fn an_unknown_unit_status_is_malformed() {
        assert_eq!(
            unit_status(-1, "audiotoolbox"),
            Err(CodecError::Malformed {
                at_bit: 0,
                what: "audiotoolbox",
            })
        );
    }
}
