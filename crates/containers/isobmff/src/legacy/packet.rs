/// Ticks per second of one track in a container file.
///
/// Zero is not a timescale. Constructors that read a file reject it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Timescale(pub u32);

impl Timescale {
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// One access unit, timed in the track's file timescale.
///
/// `dts` is the decode time. `pts` is the presentation time. They differ when
/// the container stores a composition offset. Neither value is microseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub track_id: u32,
    pub payload: Vec<u8>,
    pub dts: i64,
    pub pts: i64,
    pub duration: u32,
    pub timescale: Timescale,
    pub keyframe: bool,
}

#[cfg(test)]
mod tests {
    use super::super::CodecId;
    use super::{Packet, Timescale};

    #[test]
    fn a_packet_keeps_its_file_timescale() {
        let packet = Packet {
            track_id: 1,
            payload: vec![1, 2, 3],
            dts: 0,
            pts: 3_000,
            duration: 3_000,
            timescale: Timescale(30_000),
            keyframe: true,
        };
        assert_eq!(packet.timescale.get(), 30_000);
        assert_eq!(packet.pts - packet.dts, 3_000);
        assert!(CodecId::H264.is_named());
        assert!(!CodecId::Unknown(*b"xxxx").is_named());
    }
}
