/// One coded access unit, or one container packet that holds one.
///
/// `pts` and `duration` are in ticks of `timescale`. A negative `pts` is a
/// preroll. `timescale` is zero only when the packet does not carry a clock;
/// audio packets from this workspace use the sample rate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub stream: u32,
    pub data: Vec<u8>,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub side: Vec<SideData>,
}

/// Data that rides with a packet but is not part of the codec payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SideData {
    /// Leading delay and trailing padding, in samples at `sample_rate`.
    ///
    /// Valid PCM is the decoded stream with `encoder_delay` samples removed
    /// from the start and `trailing_padding` samples removed from the end.
    Gapless {
        encoder_delay: u32,
        trailing_padding: u32,
        sample_rate: u32,
    },
    /// Codec configuration that becomes active at this packet.
    Config(Vec<u8>),
    /// CRC-16 from ISO/IEC 11172-3 over the bits named by ISO/IEC 13818-7.
    ///
    /// `header` is the 7-byte ADTS fixed and variable header when this CRC
    /// also covers that header. It is absent when the CRC covers one raw
    /// data block of a multi-block frame.
    Protection { crc: u16, header: Option<Vec<u8>> },
}

impl Packet {
    #[must_use]
    pub fn audio(data: Vec<u8>, pts: i64, duration: i64, timescale: u32) -> Self {
        Self {
            stream: 0,
            data,
            pts,
            duration,
            timescale,
            keyframe: true,
            side: Vec::new(),
        }
    }

    /// One video access unit. `keyframe` is the random-access flag: the unit
    /// can be decoded without an earlier picture.
    #[must_use]
    pub fn video(data: Vec<u8>, pts: i64, duration: i64, timescale: u32, keyframe: bool) -> Self {
        Self {
            stream: 0,
            data,
            pts,
            duration,
            timescale,
            keyframe,
            side: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Packet, SideData};

    #[test]
    fn packet_keeps_time_and_gapless_side_data() {
        let mut packet = Packet::audio(vec![1, 2, 3], 0, 1024, 48_000);
        packet.side.push(SideData::Gapless {
            encoder_delay: 1024,
            trailing_padding: 0,
            sample_rate: 48_000,
        });
        assert_eq!(packet.duration, 1024);
        assert_eq!(packet.data, vec![1, 2, 3]);
        assert!(packet.keyframe);
    }

    #[test]
    fn a_video_packet_can_be_a_non_key_picture() {
        let packet = Packet::video(vec![0, 0, 0, 1], 3_000, 1_001, 30_000, false);
        assert!(!packet.keyframe);
        assert_eq!(packet.pts, 3_000);
        assert_eq!(packet.timescale, 30_000);
    }
}
