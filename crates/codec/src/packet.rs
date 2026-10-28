//! One access unit, and the file clock it is stamped with.
//!
//! Timestamps are ticks of [`Timescale`]. They are not microseconds.
//! A missing timestamp stays missing. Decode order is the order a demuxer
//! returns packets, not the presentation time stored here.

use std::fmt;

/// Ticks per second of a file clock.
///
/// A timestamp of `n` is `n / ticks_per_second` seconds. Zero means the
/// packet carries no clock. This value is not a microsecond count, and
/// nothing in this crate converts it into one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Timescale(u32);

impl Timescale {
    #[must_use]
    pub const fn new(ticks_per_second: u32) -> Self {
        Self(ticks_per_second)
    }

    #[must_use]
    pub const fn ticks_per_second(self) -> u32 {
        self.0
    }

    /// `false` when the packet has no clock.
    #[must_use]
    pub const fn is_clock(self) -> bool {
        self.0 != 0
    }
}

/// Decode time, presentation time, and duration in one file timescale.
///
/// Each field is absent on its own. Absence is not a zero tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Timestamp {
    decode: Option<i64>,
    presentation: Option<i64>,
    duration: Option<i64>,
}

impl Timestamp {
    /// A negative duration is refused. A missing duration stays missing.
    /// Zero duration is a real empty span, distinct from a missing one.
    pub const fn try_new(
        decode: Option<i64>,
        presentation: Option<i64>,
        duration: Option<i64>,
    ) -> Result<Self, PacketError> {
        if let Some(duration) = duration
            && duration < 0
        {
            return Err(PacketError::NegativeDuration);
        }
        Ok(Self {
            decode,
            presentation,
            duration,
        })
    }

    #[must_use]
    pub const fn missing() -> Self {
        Self {
            decode: None,
            presentation: None,
            duration: None,
        }
    }

    #[must_use]
    pub const fn decode(self) -> Option<i64> {
        self.decode
    }

    #[must_use]
    pub const fn presentation(self) -> Option<i64> {
        self.presentation
    }

    #[must_use]
    pub const fn duration(self) -> Option<i64> {
        self.duration
    }

    /// Presentation time when the file stored one, otherwise decode time.
    #[must_use]
    pub const fn presentation_or_decode(self) -> Option<i64> {
        if self.presentation.is_some() {
            self.presentation
        } else {
            self.decode
        }
    }

    #[must_use]
    pub const fn is_missing(self) -> bool {
        self.decode.is_none() && self.presentation.is_none() && self.duration.is_none()
    }
}

/// One access unit.
///
/// `bytes` is the payload a decoder receives. `stream` is the track id
/// assigned by the container. `keyframe` is the random-access flag.
/// The packet is one access unit: a container does not pack several
/// frames into it, and this type does not split it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    stream: u32,
    bytes: Vec<u8>,
    timestamp: Timestamp,
    timescale: Timescale,
    keyframe: bool,
}

impl Packet {
    /// Builds one access unit.
    ///
    /// A clockless packet (`timescale` zero) may not carry a timestamp.
    /// A negative duration is refused by [`Timestamp::try_new`].
    pub fn try_new(
        stream: u32,
        bytes: Vec<u8>,
        timestamp: Timestamp,
        timescale: Timescale,
        keyframe: bool,
    ) -> Result<Self, PacketError> {
        if !timescale.is_clock() && !timestamp.is_missing() {
            return Err(PacketError::Clockless);
        }
        Ok(Self {
            stream,
            bytes,
            timestamp,
            timescale,
            keyframe,
        })
    }

    #[must_use]
    pub const fn stream(&self) -> u32 {
        self.stream
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    #[must_use]
    pub const fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    #[must_use]
    pub const fn timescale(&self) -> Timescale {
        self.timescale
    }

    #[must_use]
    pub const fn keyframe(&self) -> bool {
        self.keyframe
    }
}

/// A packet that does not match the file clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketError {
    /// Duration was present and negative.
    NegativeDuration,
    /// The timescale is zero and a timestamp was present.
    Clockless,
}

impl fmt::Display for PacketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDuration => formatter.write_str("packet duration is negative"),
            Self::Clockless => {
                formatter.write_str("a packet with no timescale cannot carry a timestamp")
            }
        }
    }
}

impl std::error::Error for PacketError {}

#[cfg(test)]
mod tests {
    use super::{Packet, PacketError, Timescale, Timestamp};

    fn ticks(decode: Option<i64>, presentation: Option<i64>, duration: Option<i64>) -> Timestamp {
        Timestamp::try_new(decode, presentation, duration).expect("timestamp")
    }

    #[test]
    fn a_packet_keeps_bytes_time_timescale_and_keyframe() {
        let timestamp = ticks(Some(0), Some(3_000), Some(1_001));
        let packet = Packet::try_new(1, vec![7, 8, 9], timestamp, Timescale::new(30_000), true)
            .expect("packet");
        assert_eq!(packet.stream(), 1);
        assert_eq!(packet.bytes(), &[7, 8, 9]);
        assert_eq!(packet.timestamp().decode(), Some(0));
        assert_eq!(packet.timestamp().presentation(), Some(3_000));
        assert_eq!(packet.timestamp().duration(), Some(1_001));
        assert_eq!(packet.timescale().ticks_per_second(), 30_000);
        assert!(packet.keyframe());
    }

    #[test]
    fn one_buffer_is_one_access_unit() {
        let packet = Packet::try_new(
            0,
            vec![1, 2, 3, 4],
            Timestamp::missing(),
            Timescale::new(48_000),
            true,
        )
        .expect("packet");
        assert_eq!(packet.into_bytes(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn missing_timestamps_stay_missing() {
        let timestamp = Timestamp::missing();
        assert!(timestamp.is_missing());
        assert_eq!(timestamp.decode(), None);
        assert_eq!(timestamp.presentation(), None);
        assert_eq!(timestamp.duration(), None);
        let packet = Packet::try_new(0, Vec::new(), timestamp, Timescale::new(90_000), false)
            .expect("packet");
        assert!(packet.timestamp().is_missing());
        assert_eq!(packet.timescale().ticks_per_second(), 90_000);
    }

    #[test]
    fn a_zero_duration_is_not_a_missing_duration() {
        let timestamp = ticks(None, Some(0), Some(0));
        assert_eq!(timestamp.duration(), Some(0));
        assert!(!timestamp.is_missing());
    }

    #[test]
    fn decode_and_presentation_can_differ() {
        let timestamp = ticks(Some(10), Some(0), None);
        assert_eq!(timestamp.decode(), Some(10));
        assert_eq!(timestamp.presentation(), Some(0));
        assert_eq!(timestamp.presentation_or_decode(), Some(0));
        let decode_only = ticks(Some(4), None, None);
        assert_eq!(decode_only.presentation_or_decode(), Some(4));
    }

    #[test]
    fn a_negative_duration_is_refused() {
        assert_eq!(
            Timestamp::try_new(None, Some(0), Some(-1)),
            Err(PacketError::NegativeDuration)
        );
    }

    #[test]
    fn a_clockless_packet_cannot_carry_a_timestamp() {
        let timestamp = ticks(None, Some(1), None);
        assert_eq!(
            Packet::try_new(0, vec![1], timestamp, Timescale::new(0), true),
            Err(PacketError::Clockless)
        );
        let packet = Packet::try_new(0, vec![1], Timestamp::missing(), Timescale::new(0), true)
            .expect("no clock");
        assert!(!packet.timescale().is_clock());
        assert!(packet.timestamp().is_missing());
    }

    #[test]
    fn timestamps_stay_in_the_file_timescale() {
        let presentation = 180_000;
        let timescale = Timescale::new(90_000);
        let packet = Packet::try_new(
            0,
            vec![1],
            ticks(None, Some(presentation), Some(3_000)),
            timescale,
            true,
        )
        .expect("packet");
        assert_eq!(packet.timescale().ticks_per_second(), 90_000);
        assert_eq!(packet.timestamp().presentation(), Some(180_000));
        let truncated_micros = presentation * 1_000_000 / i64::from(timescale.ticks_per_second());
        assert_eq!(truncated_micros, 2_000_000);
        assert_ne!(packet.timestamp().presentation(), Some(truncated_micros));
    }
}
