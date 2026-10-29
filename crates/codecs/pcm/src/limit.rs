//! Caps checked before a PCM buffer is allocated.
//!
//! A hostile length fails here. It does not become a `Vec` that overflows
//! the size calculation.

use std::mem::size_of;

use cutlass_decode::CodecError;

/// Largest packet, in bytes, this decoder will read.
pub(crate) const MAX_PACKET: usize = 16 * 1024 * 1024;

/// Largest decoded or encoded buffer, in bytes.
pub(crate) const MAX_OUTPUT: usize = 32 * 1024 * 1024;

/// Highest sample rate, in hertz. 384 kHz covers DXD.
pub(crate) const MAX_RATE: u32 = 384_000;

/// Positive full scale for packed 24-bit PCM. Matches the mix, which treats
/// [`i16::MAX`] as full scale rather than `2^15`.
pub(crate) const S24_FULL: f64 = 8_388_607.0;

/// Positive full scale for 32-bit integer PCM.
pub(crate) const S32_FULL: f64 = 2_147_483_647.0;

pub(crate) fn packet_len(len: usize) -> Result<(), CodecError> {
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

pub(crate) fn rate(rate: u32) -> Result<(), CodecError> {
    if rate == 0 {
        return Err(CodecError::InvalidApi {
            what: "sample rate",
        });
    }
    if rate > MAX_RATE {
        return Err(CodecError::Limit {
            what: "sample rate",
        });
    }
    Ok(())
}

/// Bytes in one interleaved frame, or a planar frame of the same width.
pub(crate) fn frame_bytes(channels: usize, sample_bytes: usize) -> Result<usize, CodecError> {
    channels.checked_mul(sample_bytes).ok_or(CodecError::Limit {
        what: "frame bytes",
    })
}

/// Whole frames in `byte_len`. A trailing partial frame is an error.
///
/// FFmpeg drops that tail when the packet is longer than one frame, and
/// GStreamer holds it for the next buffer. The container owns framing.
/// A packet here is already one block, so a short tail is not silence.
pub(crate) fn whole_frames(byte_len: usize, frame: usize) -> Result<usize, CodecError> {
    if frame == 0 || !byte_len.is_multiple_of(frame) {
        let whole = if frame == 0 {
            0
        } else {
            byte_len - byte_len % frame
        };
        let at_bit = u64::try_from(whole).unwrap_or(u64::MAX).saturating_mul(8);
        return Err(CodecError::Truncated {
            at_bit,
            what: "pcm frame",
        });
    }
    Ok(byte_len / frame)
}

pub(crate) fn sample_count(frames: usize, channels: usize) -> Result<usize, CodecError> {
    frames.checked_mul(channels).ok_or(CodecError::Limit {
        what: "sample count",
    })
}

/// Reserve `count` elements, or reject a length that would overflow.
pub(crate) fn reserve<T>(count: usize) -> Result<Vec<T>, CodecError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(CodecError::Limit { what: "allocation" })?;
    if bytes > MAX_OUTPUT {
        return Err(CodecError::Limit { what: "allocation" });
    }
    let mut out = Vec::new();
    out.try_reserve(count)
        .map_err(|_| CodecError::Limit { what: "allocation" })?;
    Ok(out)
}

pub(crate) fn reserve_bytes(len: usize) -> Result<Vec<u8>, CodecError> {
    if len > MAX_OUTPUT {
        return Err(CodecError::Limit { what: "allocation" });
    }
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| CodecError::Limit { what: "allocation" })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{frame_bytes, reserve, whole_frames};

    #[test]
    fn a_partial_frame_is_not_a_shorter_frame() {
        let err = whole_frames(3, 2).expect_err("partial");
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Truncated {
                what: "pcm frame",
                ..
            }
        ));
        assert_eq!(whole_frames(4, 2).expect("frames"), 2);
    }

    #[test]
    fn an_overflowing_length_is_rejected_before_reserve() {
        let err = reserve::<u16>(usize::MAX).expect_err("overflow");
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Limit { what: "allocation" }
        ));
        let err = frame_bytes(usize::MAX / 2 + 1, 3).expect_err("frame");
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Limit {
                what: "frame bytes"
            }
        ));
    }
}
