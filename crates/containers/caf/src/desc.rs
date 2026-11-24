//! The `desc` chunk: `CAFAudioDescription`, 32 big-endian bytes.
//!
//! `mSampleRate` is a `Float64`. The stream timescale is that rate as a
//! whole number of ticks per second: a fraction, such as the 22254.5454 Hz
//! of early Macintosh sound, is rounded to the nearest hertz, a half going
//! up, the rule the AIFF demuxer uses. libavformat's CAF demuxer drops the
//! fraction instead, so it reports 22254 where this reports 22255.

use cutlass_container::{Limits, OpenError};

use crate::bytes::{be_u32, be_u64, fourcc};
use crate::error::{hostile, malformed, truncated};
use crate::limits::{MAX_BITS_PER_CHANNEL, MAX_CHANNELS, MAX_FRAMES_PER_PACKET};

/// Bytes in a `desc` chunk body.
pub const DESC_BYTES: u64 = 32;

/// `kCAFLinearPCMFormatFlagIsFloat`.
pub const FLAG_FLOAT: u32 = 1;
/// `kCAFLinearPCMFormatFlagIsLittleEndian`.
pub const FLAG_LITTLE_ENDIAN: u32 = 2;

/// The audio description, with the rate checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Description {
    /// `mSampleRate`, to the nearest whole number of frames per second.
    pub sample_rate: u32,
    /// `mFormatID`, such as `lpcm`, `aac ` or `alac`.
    pub format: [u8; 4],
    /// `mFormatFlags`.
    pub flags: u32,
    /// `mBytesPerPacket`; zero when packet sizes vary.
    pub bytes_per_packet: u32,
    /// `mFramesPerPacket`; zero when packet durations vary.
    pub frames_per_packet: u32,
    /// `mChannelsPerFrame`.
    pub channels: u32,
    /// `mBitsPerChannel`; zero for compressed formats.
    pub bits_per_channel: u32,
}

impl Description {
    /// Parses and checks a `desc` body.
    pub(crate) fn parse(body: &[u8], limits: &Limits) -> Result<Self, OpenError> {
        if body.len() < 32 {
            return Err(truncated(DESC_BYTES, crate::bytes::len_u64(body.len())));
        }
        let rate_bits = be_u64(body, 0).ok_or_else(|| truncated(DESC_BYTES, 0))?;
        let field = |at| be_u32(body, at).ok_or_else(|| truncated(DESC_BYTES, 0));
        let format = fourcc(body, 8).ok_or_else(|| truncated(DESC_BYTES, 0))?;
        let description = Self {
            sample_rate: nearest_rate(f64::from_bits(rate_bits))?,
            format,
            flags: field(12)?,
            bytes_per_packet: field(16)?,
            frames_per_packet: field(20)?,
            channels: field(24)?,
            bits_per_channel: field(28)?,
        };
        description.check(limits)?;
        Ok(description)
    }

    fn check(&self, limits: &Limits) -> Result<(), OpenError> {
        if self.channels == 0 {
            return Err(malformed("desc has zero channels"));
        }
        if self.channels > MAX_CHANNELS {
            return Err(hostile(u64::from(self.channels), u64::from(MAX_CHANNELS)));
        }
        if self.frames_per_packet > MAX_FRAMES_PER_PACKET {
            return Err(hostile(
                u64::from(self.frames_per_packet),
                u64::from(MAX_FRAMES_PER_PACKET),
            ));
        }
        if u64::from(self.bytes_per_packet) > limits.max_packet_bytes {
            return Err(hostile(
                u64::from(self.bytes_per_packet),
                limits.max_packet_bytes,
            ));
        }
        if self.bits_per_channel > MAX_BITS_PER_CHANNEL {
            return Err(hostile(
                u64::from(self.bits_per_channel),
                u64::from(MAX_BITS_PER_CHANNEL),
            ));
        }
        Ok(())
    }

    /// The 32 body bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0_u8; 32];
        out[0..8].copy_from_slice(&f64::from(self.sample_rate).to_bits().to_be_bytes());
        out[8..12].copy_from_slice(&self.format);
        out[12..16].copy_from_slice(&self.flags.to_be_bytes());
        out[16..20].copy_from_slice(&self.bytes_per_packet.to_be_bytes());
        out[20..24].copy_from_slice(&self.frames_per_packet.to_be_bytes());
        out[24..28].copy_from_slice(&self.channels.to_be_bytes());
        out[28..32].copy_from_slice(&self.bits_per_channel.to_be_bytes());
        out
    }

    /// True when every packet has the same byte size and the same duration.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        self.bytes_per_packet != 0 && self.frames_per_packet != 0
    }
}

/// The timescale of a `Float64` rate: the nearest whole number, a half
/// going up. The rate must be finite and round to `1..=u32::MAX`.
fn nearest_rate(rate: f64) -> Result<u32, OpenError> {
    let whole = (rate + 0.5).floor();
    if !rate.is_finite() || whole < 1.0 || whole > f64::from(u32::MAX) {
        return Err(malformed(format!(
            "desc sample rate {rate} is not a number of frames per second"
        )));
    }
    // Checked above: whole, at least 1 and at most u32::MAX, so the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(whole as u32)
}

#[cfg(test)]
mod tests {
    use super::{Description, nearest_rate};
    use cutlass_container::{Limits, OpenError};

    fn body(rate: f64, channels: u32, fpp: u32, bpp: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(rate.to_bits().to_be_bytes());
        out.extend(b"lpcm");
        out.extend(0_u32.to_be_bytes());
        out.extend(bpp.to_be_bytes());
        out.extend(fpp.to_be_bytes());
        out.extend(channels.to_be_bytes());
        out.extend(16_u32.to_be_bytes());
        out
    }

    #[test]
    fn a_description_round_trips() {
        let parsed =
            Description::parse(&body(44_100.0, 2, 1, 4), &Limits::standard()).expect("desc");
        assert_eq!(parsed.sample_rate, 44_100);
        assert_eq!(parsed.format, *b"lpcm");
        assert!(parsed.is_constant());
        assert_eq!(parsed.to_bytes().to_vec(), body(44_100.0, 2, 1, 4));
    }

    #[test]
    fn rates_that_are_not_a_timescale_are_refused() {
        for rate in [
            0.0,
            -1.0,
            0.25,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            5e9,
            f64::from(u32::MAX) + 0.5,
        ] {
            assert!(nearest_rate(rate).is_err(), "{rate}");
        }
        assert_eq!(nearest_rate(1.0), Ok(1));
        assert_eq!(nearest_rate(f64::from(u32::MAX)), Ok(u32::MAX));
        assert_eq!(nearest_rate(8_000.0), Ok(8_000));
    }

    #[test]
    fn a_fractional_rate_is_the_nearest_hertz() {
        // 244800 / 11 Hz, the first Macintosh sound hardware.
        assert_eq!(nearest_rate(244_800.0 / 11.0), Ok(22_255));
        assert_eq!(nearest_rate(122_400.0 / 11.0), Ok(11_127));
        assert_eq!(nearest_rate(44_100.5), Ok(44_101));
        assert_eq!(nearest_rate(44_100.499), Ok(44_100));
        assert_eq!(nearest_rate(0.5), Ok(1));
        assert_eq!(nearest_rate(f64::from(u32::MAX) + 0.25), Ok(u32::MAX));
    }

    #[test]
    fn hostile_fields_are_refused_before_use() {
        let limits = Limits::standard();
        assert!(matches!(
            Description::parse(&body(48_000.0, 0, 1, 2), &limits),
            Err(OpenError::Source { .. })
        ));
        assert!(matches!(
            Description::parse(&body(48_000.0, 65, 1, 130), &limits),
            Err(OpenError::HostileSize { .. })
        ));
        assert!(matches!(
            Description::parse(&body(48_000.0, 2, u32::MAX, 0), &limits),
            Err(OpenError::HostileSize { .. })
        ));
        assert!(matches!(
            Description::parse(&body(48_000.0, 2, 1, u32::MAX), &limits),
            Err(OpenError::HostileSize { .. })
        ));
        assert!(matches!(
            Description::parse(&body(48_000.0, 2, 1, 4)[..31], &limits),
            Err(OpenError::Truncated { .. })
        ));
    }
}
