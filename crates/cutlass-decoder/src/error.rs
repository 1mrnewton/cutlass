use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[cfg(feature = "ffmpeg")]
    #[error("failed to open media")]
    Open(#[source] ffmpeg_next::Error),

    #[cfg(feature = "ffmpeg")]
    #[error("demuxer read failed")]
    Io(#[source] ffmpeg_next::Error),

    #[cfg(feature = "ffmpeg")]
    #[error("decode failed")]
    Decode(#[source] ffmpeg_next::Error),

    #[error("unsupported: {what}")]
    Unsupported { what: String },

    #[error("hardware acceleration unavailable: {accel}")]
    HwAccelUnavailable { accel: &'static str },
}

impl DecodeError {
    pub fn unsupported(what: impl Into<String>) -> Self {
        DecodeError::Unsupported { what: what.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_wraps_message() {
        let err = DecodeError::unsupported("no video stream found");
        assert!(matches!(err, DecodeError::Unsupported { .. }));
        assert_eq!(err.to_string(), "unsupported: no video stream found");
    }

    #[test]
    fn hw_accel_unavailable_formats_accel_name() {
        let err = DecodeError::HwAccelUnavailable {
            accel: "videotoolbox",
        };
        assert_eq!(
            err.to_string(),
            "hardware acceleration unavailable: videotoolbox"
        );
    }
}
