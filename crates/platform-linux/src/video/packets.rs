//! Encoder output as the packets a muxer takes.
//!
//! NVENC and VA-API write Annex B with the parameter sets in band. Each unit
//! leaves as 4-byte length-prefixed NAL units without those sets or access
//! unit delimiters, and the first keyframe carries the `avcC` or `hvcC` as
//! [`SideData::Config`], as on Windows and Apple. B pictures are off, so
//! units leave in input order and take their time from the oldest picture
//! still waiting.

use std::collections::VecDeque;

use cutlass_decode::nal;
use cutlass_decode::{CodecError, Packet, SideData};
use cutlass_encode::FrameTime;

use super::kind::Kind;

/// Turns encoded access units into packets.
pub struct Packetizer {
    kind: Kind,
    attached: bool,
    /// Parameter sets known out of band, for an encoder that leaves them
    /// out of a unit.
    header: Vec<Vec<u8>>,
    /// Times of the pictures handed in and not yet out.
    times: VecDeque<FrameTime>,
}

impl Packetizer {
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            attached: false,
            header: Vec::new(),
            times: VecDeque::new(),
        }
    }

    /// Parameter sets from the encoder itself, used when a keyframe comes
    /// out without them.
    pub fn set_header(&mut self, sets: Vec<Vec<u8>>) {
        self.header = sets;
    }

    /// A picture went in at `time`.
    pub fn sent(&mut self, time: FrameTime) {
        self.times.push_back(time);
    }

    /// Forget waiting times and attach the configuration again.
    pub fn reset(&mut self) {
        self.attached = false;
        self.times.clear();
    }

    /// One encoded access unit in Annex B. `None` when it held only
    /// parameter sets or delimiters.
    pub fn packet(&mut self, annex_b: &[u8]) -> Result<Option<Packet>, CodecError> {
        let nals = nal::annex_b(annex_b)?;
        let mut sets = Vec::new();
        let mut units = Vec::new();
        let mut keyframe = false;
        for unit in nals {
            match self.kind.nal_type(&unit.bytes) {
                Some(kind) if self.kind.is_parameter_set(kind) => sets.push(unit.bytes),
                Some(kind) if self.kind.is_delimiter(kind) => {}
                kind => {
                    keyframe |= kind.is_some_and(|kind| self.kind.is_keyframe(kind));
                    units.push(unit);
                }
            }
        }
        if !sets.is_empty() {
            self.header.clone_from(&sets);
        }
        if units.is_empty() {
            return Ok(None);
        }
        let time = match self.times.pop_front() {
            Some(time) => time,
            None => FrameTime::new(0, 1, 30)?,
        };
        let mut packet = Packet {
            stream: 0,
            data: nal::length_prefixed(&units, 4)?,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
            keyframe,
            side: Vec::new(),
        };
        if keyframe
            && !self.attached
            && let Ok(config) = self.kind.write_config(&self.header)
        {
            packet.side.push(SideData::Config(config));
            self.attached = true;
        }
        Ok(Some(packet))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::nal::Nal;

    const SPS: [u8; 4] = [0x67, 0x64, 0x00, 0x28];
    const PPS: [u8; 2] = [0x68, 0xee];
    const AUD: [u8; 2] = [0x09, 0xf0];
    const IDR: [u8; 3] = [0x65, 0x88, 0x80];
    const P: [u8; 3] = [0x41, 0x9a, 0x02];

    fn annex_b(units: &[&[u8]]) -> Vec<u8> {
        let nals: Vec<Nal> = units
            .iter()
            .map(|bytes| Nal {
                bytes: bytes.to_vec(),
            })
            .collect();
        nal::to_annex_b(&nals)
    }

    #[test]
    fn the_first_keyframe_carries_the_config_and_sets_leave_the_unit() {
        let mut packets = Packetizer::new(Kind::H264);
        packets.sent(FrameTime::new(0, 1, 25).expect("time"));
        packets.sent(FrameTime::new(1, 1, 25).expect("time"));
        let first = packets
            .packet(&annex_b(&[&AUD, &SPS, &PPS, &IDR]))
            .expect("first")
            .expect("unit");
        assert!(first.keyframe);
        assert_eq!((first.pts, first.duration, first.timescale), (0, 1, 25));
        assert_eq!(
            nal::split_lengths(&first.data, 4).expect("lengths"),
            vec![Nal {
                bytes: IDR.to_vec()
            }]
        );
        let Some(SideData::Config(config)) = first.side.first() else {
            panic!("no config");
        };
        let config = nal::avc_config(config).expect("avcC");
        assert_eq!(config.sps, vec![SPS.to_vec()]);
        assert_eq!(config.pps, vec![PPS.to_vec()]);

        let second = packets
            .packet(&annex_b(&[&P]))
            .expect("second")
            .expect("unit");
        assert!(!second.keyframe);
        assert_eq!(second.pts, 1);
        assert!(second.side.is_empty());
    }

    #[test]
    fn sets_alone_are_no_packet_and_are_kept_for_the_keyframe() {
        let mut packets = Packetizer::new(Kind::H264);
        assert!(
            packets
                .packet(&annex_b(&[&SPS, &PPS]))
                .expect("sets")
                .is_none()
        );
        let unit = packets
            .packet(&annex_b(&[&IDR]))
            .expect("idr")
            .expect("unit");
        assert!(matches!(unit.side.first(), Some(SideData::Config(_))));
        packets.reset();
        let again = packets
            .packet(&annex_b(&[&IDR]))
            .expect("idr")
            .expect("unit");
        assert!(matches!(again.side.first(), Some(SideData::Config(_))));
    }
}
