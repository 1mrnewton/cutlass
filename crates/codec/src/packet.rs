/// One coded access unit.
///
/// `pts` and `duration` are ticks of `timescale`. They are not microseconds.
/// A container that does not distinguish decode time from presentation time
/// stores the presentation time in `pts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub stream: u32,
    pub data: Vec<u8>,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Packet;

    #[test]
    fn audio_packet_keeps_the_file_timescale() {
        let packet = Packet::audio(vec![1, 2, 3], 0, 1152, 44_100);
        assert_eq!(packet.data, vec![1, 2, 3]);
        assert_eq!(packet.pts, 0);
        assert_eq!(packet.duration, 1152);
        assert_eq!(packet.timescale, 44_100);
        assert_ne!(packet.timescale, 1_000_000);
        assert!(packet.keyframe);
    }
}
