//! Field pictures: the two fields of a frame, in one packet, in two, and
//! around a `sequence_end_code`.
//!
//! The streams are written here, bit by bit, from H.262: a 32×32
//! interlaced sequence whose I fields are one slice of two macroblocks,
//! every block DC only. Each field has its own luma level, so a frame
//! built from a matched pair alternates levels line by line, and a field
//! left unpaired fills the frame with one level.

#[cfg(test)]
mod fields {
    use cutlass_decode::{Packet, VideoOutput};
    use cutlass_mpeg2::Mpeg2Decoder;
    use cutlass_mpeg2::es::split;

    const TOP: u8 = 1;
    const BOTTOM: u8 = 2;
    const SEQUENCE_END: [u8; 4] = [0, 0, 1, 0xb7];

    #[derive(Default)]
    struct Bits {
        bytes: Vec<u8>,
        acc: u64,
        len: u32,
    }

    impl Bits {
        fn put(&mut self, value: u32, bits: u32) {
            self.acc = (self.acc << bits) | u64::from(value);
            self.len += bits;
            while self.len >= 8 {
                self.len -= 8;
                self.bytes.push((self.acc >> self.len) as u8);
            }
        }

        fn code(&mut self, pattern: &str) {
            for bit in pattern.bytes() {
                self.put(u32::from(bit == b'1'), 1);
            }
        }

        fn start(&mut self, code: u8) {
            self.align();
            self.bytes.extend_from_slice(&[0, 0, 1, code]);
        }

        fn align(&mut self) {
            if self.len > 0 {
                self.put(0, 8 - self.len);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            self.align();
            self.bytes
        }
    }

    /// Sequence header and extension (Main profile, 4:2:0, interlaced,
    /// 25 Hz), then a closed GOP header.
    fn sequence() -> Vec<u8> {
        let mut w = Bits::default();
        w.start(0xb3);
        w.put(32, 12);
        w.put(32, 12);
        w.put(1, 4); // square samples
        w.put(3, 4); // 25 Hz
        w.put(1000, 18);
        w.put(1, 1);
        w.put(10, 10);
        w.put(0, 3); // not constrained, default matrices
        w.start(0xb5);
        w.put(1, 4);
        w.put(0x48, 8); // Main profile, Main level
        w.put(0, 1); // progressive_sequence
        w.put(1, 2); // 4:2:0
        w.put(0, 4);
        w.put(0, 12);
        w.put(1, 1);
        w.put(0, 8);
        w.put(0, 1); // low_delay
        w.put(0, 7);
        w.start(0xb8);
        w.put(0, 13);
        w.put(1, 1);
        w.put(0, 12);
        w.put(1, 1); // closed_gop
        w.put(0, 1);
        w.finish()
    }

    /// Table B-12 code for a luma DC size.
    fn dc_size(size: u32) -> &'static str {
        [
            "100", "00", "01", "101", "110", "1110", "11110", "111110", "1111110",
        ][size as usize]
    }

    /// One I field whose luma is `level` and whose chroma is 128.
    fn field(temporal_reference: u32, structure: u8, level: u8) -> Vec<u8> {
        let mut w = Bits::default();
        w.start(0x00);
        w.put(temporal_reference, 10);
        w.put(1, 3); // I
        w.put(0xffff, 16);
        w.put(0, 1);
        w.start(0xb5);
        w.put(8, 4);
        w.put(0xffff, 16); // f_codes, unused in I
        w.put(0, 2); // 8-bit DC
        w.put(u32::from(structure), 2);
        w.put(0, 8); // flags, progressive_frame and chroma_420_type 0
        w.put(0, 1);
        w.start(0x01);
        w.put(8, 5);
        w.put(0, 1);
        let diff = i32::from(level) - 128;
        for mb in 0..2 {
            w.code("1"); // macroblock_address_increment
            w.code("1"); // intra
            for block in 0..4 {
                if mb == 0 && block == 0 && diff != 0 {
                    let size = 32 - diff.unsigned_abs().leading_zeros();
                    w.code(dc_size(size));
                    let bits = if diff > 0 {
                        diff
                    } else {
                        diff + (1 << size) - 1
                    };
                    w.put(bits as u32, size);
                } else {
                    w.code(dc_size(0));
                }
                w.code("10"); // end of block
            }
            for _ in 0..2 {
                w.code("00"); // chroma DC size 0
                w.code("10");
            }
        }
        w.finish()
    }

    fn packet(data: Vec<u8>, pts: i64) -> Packet {
        Packet {
            stream: 0,
            data,
            pts,
            duration: 1,
            timescale: 25,
            keyframe: true,
            side: Vec::new(),
        }
    }

    fn decode(packets: &[Vec<u8>]) -> Vec<VideoOutput> {
        let mut decoder = Mpeg2Decoder::default();
        let mut frames = Vec::new();
        for (pts, data) in packets.iter().enumerate() {
            frames.extend(decoder.push(&packet(data.clone(), pts as i64)).unwrap());
        }
        frames.extend(decoder.drain().unwrap());
        frames
    }

    /// Luma of the even and odd lines, each checked to be one level.
    fn lines(frame: &VideoOutput) -> (u8, u8) {
        let y = frame.picture.y();
        let row = |r: usize| &y[r * 32..r * 32 + 32];
        let (even, odd) = (row(0)[0], row(1)[0]);
        for r in 0..32 {
            let want = if r % 2 == 0 { even } else { odd };
            assert!(
                row(r).iter().all(|&s| s.abs_diff(want) <= 1),
                "row {r}: {:?}",
                row(r)
            );
        }
        assert!(frame.picture.u().iter().all(|&s| s.abs_diff(128) <= 1));
        (even, odd)
    }

    fn near(got: (u8, u8), want: (u8, u8)) -> bool {
        got.0.abs_diff(want.0) <= 1 && got.1.abs_diff(want.1) <= 1
    }

    #[test]
    fn a_field_pair_in_one_packet_is_one_frame() {
        let data = [sequence(), field(0, TOP, 128), field(0, BOTTOM, 160)].concat();
        let frames = decode(&[data]);
        assert_eq!(frames.len(), 1);
        assert!(
            near(lines(&frames[0]), (128, 160)),
            "{:?}",
            lines(&frames[0])
        );
    }

    #[test]
    fn the_second_field_may_come_in_its_own_packet() {
        let first = [sequence(), field(0, BOTTOM, 96)].concat();
        let mut decoder = Mpeg2Decoder::default();
        assert!(decoder.push(&packet(first, 0)).unwrap().is_empty());
        let mut frames = decoder.push(&packet(field(0, TOP, 160), 0)).unwrap();
        frames.extend(decoder.drain().unwrap());
        assert_eq!(frames.len(), 1);
        assert!(
            near(lines(&frames[0]), (160, 96)),
            "{:?}",
            lines(&frames[0])
        );
    }

    #[test]
    fn a_lone_field_fills_the_frame_at_drain() {
        let frames = decode(&[[sequence(), field(0, TOP, 160)].concat()]);
        assert_eq!(frames.len(), 1);
        assert!(
            near(lines(&frames[0]), (160, 160)),
            "{:?}",
            lines(&frames[0])
        );
    }

    /// Frame 0 is a pair; frame 1's fields have a sequence end between
    /// them, then another after, as at the tail of FFmpeg's FATE `t.mpg`.
    fn end_between_fields() -> Vec<Vec<u8>> {
        vec![
            [sequence(), field(0, TOP, 128), field(0, BOTTOM, 160)].concat(),
            [field(1, BOTTOM, 96).as_slice(), &SEQUENCE_END].concat(),
            [field(1, TOP, 192).as_slice(), &SEQUENCE_END].concat(),
        ]
    }

    #[test]
    fn a_sequence_end_between_fields_keeps_the_pair() {
        let frames = decode(&end_between_fields());
        let got: Vec<(i64, (u8, u8))> = frames.iter().map(|f| (f.pts, lines(f))).collect();
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].0, 0);
        assert!(near(got[0].1, (128, 160)), "{got:?}");
        assert_eq!(got[1].0, 1);
        assert!(near(got[1].1, (192, 96)), "{got:?}");
    }

    #[test]
    fn the_splitter_keeps_a_pair_across_a_sequence_end() {
        let stream = end_between_fields().concat();
        let units = split(&stream).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(
            units.iter().map(|u| u.display_index).collect::<Vec<_>>(),
            [0, 1]
        );
        let frames = decode(&units.into_iter().map(|u| u.data).collect::<Vec<_>>());
        assert_eq!(frames.len(), 2);
        assert!(near(lines(&frames[1]), (192, 96)));
    }
}
