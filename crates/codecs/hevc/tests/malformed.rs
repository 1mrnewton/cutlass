//! Malformed input returns an error and never panics: hand-built parameter
//! sets with bad or huge values, and x265 streams truncated and corrupted.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod malformed {
    use std::process::Command;

    use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder};
    use cutlass_hevc::HevcDecoder;

    /// Writes the bits of a hand-made parameter set.
    struct Writer {
        bytes: Vec<u8>,
        bits: u32,
    }

    impl Writer {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bits: 0,
            }
        }

        fn u(&mut self, n: u32, value: u64) -> &mut Self {
            for i in (0..n).rev() {
                if self.bits.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                if (value >> i) & 1 == 1 {
                    let last = self.bytes.len() - 1;
                    self.bytes[last] |= 0x80 >> (self.bits % 8);
                }
                self.bits += 1;
            }
            self
        }

        fn ue(&mut self, value: u64) -> &mut Self {
            let v = value + 1;
            let len = 64 - v.leading_zeros();
            self.u(len - 1, 0).u(len, v)
        }

        fn se(&mut self, value: i64) -> &mut Self {
            let code = if value > 0 { 2 * value - 1 } else { -2 * value };
            self.ue(code as u64)
        }

        /// RBSP trailing bits, emulation prevention, and a start code.
        fn nal(mut self, kind: u8) -> Vec<u8> {
            self.u(1, 1);
            while !self.bits.is_multiple_of(8) {
                self.u(1, 0);
            }
            let mut out = vec![0, 0, 0, 1, kind << 1, 1];
            let mut zeros = 0;
            for byte in self.bytes {
                if zeros >= 2 && byte <= 3 {
                    out.push(3);
                    zeros = 0;
                }
                zeros = if byte == 0 { zeros + 1 } else { 0 };
                out.push(byte);
            }
            out
        }
    }

    struct SpsFields {
        chroma: u64,
        width: u64,
        height: u64,
        bit_depth_minus8: u64,
        log2_min_cb_minus3: u64,
        log2_diff_cb: u64,
        log2_max_poc_lsb_minus4: u64,
        long_term: bool,
    }

    const GOOD: SpsFields = SpsFields {
        chroma: 1,
        width: 64,
        height: 64,
        bit_depth_minus8: 0,
        log2_min_cb_minus3: 0,
        log2_diff_cb: 3,
        log2_max_poc_lsb_minus4: 4,
        long_term: false,
    };

    fn sps(f: &SpsFields) -> Vec<u8> {
        let mut w = Writer::new();
        w.u(4, 0).u(3, 0).u(1, 1);
        // profile_tier_level: Main, level 4.1.
        w.u(2, 0)
            .u(1, 0)
            .u(5, 1)
            .u(32, 0x6000_0000)
            .u(4, 0b1001)
            .u(43, 0)
            .u(1, 0)
            .u(8, 123);
        w.ue(0).ue(f.chroma);
        if f.chroma == 3 {
            w.u(1, 0);
        }
        w.ue(f.width).ue(f.height).u(1, 0);
        w.ue(f.bit_depth_minus8)
            .ue(f.bit_depth_minus8)
            .ue(f.log2_max_poc_lsb_minus4);
        w.u(1, 1).ue(2).ue(0).ue(0);
        w.ue(f.log2_min_cb_minus3)
            .ue(f.log2_diff_cb)
            .ue(0)
            .ue(3)
            .ue(0)
            .ue(0);
        w.u(1, 0).u(1, 0).u(1, 0).u(1, 0);
        // No short-term sets; long-term pictures, with none listed here.
        w.ue(0).u(1, u64::from(f.long_term));
        if f.long_term {
            w.ue(0);
        }
        w.u(1, 0).u(1, 0).u(1, 0).u(1, 0);
        w.nal(33)
    }

    fn pps() -> Vec<u8> {
        pps_with_init_qp(26)
    }

    fn pps_with_init_qp(init_qp: i64) -> Vec<u8> {
        let mut w = Writer::new();
        // Ids, flags, default reference counts, init_qp_minus26.
        w.ue(0).ue(0).u(1, 0).u(1, 0).u(3, 0).u(1, 0).u(1, 0);
        w.ue(0).ue(0).se(init_qp - 26);
        // Constrained intra, transform skip, cu_qp_delta; Cb and Cr offsets 0.
        w.u(1, 0).u(1, 0).u(1, 0).ue(0).ue(0);
        // Slice chroma offsets, weights, bypass, tiles, wavefronts.
        w.u(1, 0).u(1, 0).u(1, 0).u(1, 0).u(1, 0).u(1, 0);
        // Across slices, deblocking control, scaling list, list modification.
        w.u(1, 1).u(1, 0).u(1, 0).u(1, 0).ue(0).u(1, 0).u(1, 0);
        w.nal(34)
    }

    fn decoder() -> HevcDecoder {
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.nal_length_size = 0;
        HevcDecoder::open(&params).expect("open")
    }

    fn push(decoder: &mut HevcDecoder, data: Vec<u8>) -> Result<usize, CodecError> {
        decoder
            .push(&Packet::video(data, 0, 1, 30, true))
            .map(|v| v.len())
    }

    fn one_thread_decoder() -> HevcDecoder {
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.nal_length_size = 0;
        // One thread decodes each picture inside the push that completes it.
        HevcDecoder::open_with_threads(&params, 1).expect("open")
    }

    /// Slice data found by search: one intra 64x64 CTB, then the end of the
    /// slice segment.
    const ONE_CTB: [u8; 6] = [0x10, 0x33, 0x9c, 0xc0, 0x7b, 0x42];

    /// The rest of an intra slice segment after its picture order fields.
    fn intra_slice_tail(mut w: Writer, kind: u8, qp_delta: i64) -> Vec<u8> {
        // slice_qp_delta, filtering across slices, byte alignment.
        w.se(qp_delta).u(1, 1).u(1, 1);
        while !w.bits.is_multiple_of(8) {
            w.u(1, 0);
        }
        for byte in ONE_CTB {
            w.u(8, u64::from(byte));
        }
        w.nal(kind)
    }

    /// An IDR slice segment holding CTB `ctb` of a picture two CTBs wide.
    fn idr_slice(ctb: u64, qp_delta: i64) -> Vec<u8> {
        let mut w = Writer::new();
        // First segment flag, no_output_of_prior_pics_flag, PPS id.
        w.u(1, u64::from(ctb == 0)).u(1, 0).ue(0);
        if ctb != 0 {
            w.u(1, ctb);
        }
        // An I slice.
        w.ue(2);
        intra_slice_tail(w, 19, qp_delta)
    }

    #[test]
    fn a_pps_sent_again_inside_a_picture_does_not_change_its_slices() {
        let mut dec = one_thread_decoder();
        let mut data = sps(&SpsFields { width: 128, ..GOOD });
        data.extend(pps());
        data.extend(idr_slice(0, 0));
        // A slice_qp_delta of -51 is in range for the new PPS only. The
        // picture keeps the PPS it started with, and there the QP is -25.
        data.extend(pps_with_init_qp(51));
        data.extend(idr_slice(1, -51));
        let result = push(&mut dec, data);
        assert!(
            matches!(
                result,
                Err(CodecError::Malformed {
                    what: "slice_qp_delta",
                    ..
                })
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_slice_qp_delta_past_32_bits_is_malformed() {
        for qp_delta in [i64::from(i32::MAX), -i64::from(i32::MAX)] {
            let mut data = sps(&GOOD);
            data.extend(pps());
            data.extend(idr_slice(0, qp_delta));
            let result = push(&mut one_thread_decoder(), data);
            assert!(
                matches!(result, Err(CodecError::Malformed { .. })),
                "{qp_delta}: {result:?}"
            );
        }
    }

    #[test]
    fn a_long_term_picture_past_the_order_count_range_is_malformed() {
        let mut dec = one_thread_decoder();
        let fields = SpsFields {
            long_term: true,
            ..GOOD
        };
        let mut data = sps(&fields);
        data.extend(pps());
        data.extend(idr_slice(0, 0));
        assert_eq!(push(&mut dec, data), Ok(1));
        // A trailing picture one before the IDR in output order, so its order
        // count is -1. Its long-term picture sits 2^31 before that.
        let mut w = Writer::new();
        // First segment, PPS id, I slice, order count LSBs, an empty
        // short-term set written in the header.
        w.u(1, 1).ue(0).ue(2).u(8, 255).u(1, 0).ue(0).ue(0);
        // One long-term picture: its LSBs, unused, with an MSB cycle.
        w.ue(1).u(8, 255).u(1, 0).u(1, 1).ue(1 << 23);
        let result = push(&mut dec, intra_slice_tail(w, 1, 0));
        assert!(
            matches!(
                result,
                Err(CodecError::Malformed {
                    what: "delta_poc_msb_cycle_lt",
                    ..
                })
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_picture_wider_than_the_planar_output_is_refused() {
        // A slice may leave CTBs out, so one CTB makes a whole picture.
        let picture = |width: u64| {
            let mut data = sps(&SpsFields { width, ..GOOD });
            data.extend(pps());
            data.extend(idr_slice(0, 0));
            push(&mut one_thread_decoder(), data)
        };
        assert_eq!(picture(16_880), Ok(1));
        // HEVC allows 16,888 samples; a `VideoOutput` picture holds 16,880.
        assert_eq!(
            picture(16_888),
            Err(CodecError::Limit {
                what: "picture size"
            })
        );
    }

    #[test]
    fn an_order_count_past_32_bits_is_malformed() {
        let mut dec = one_thread_decoder();
        let fields = SpsFields {
            log2_max_poc_lsb_minus4: 12,
            ..GOOD
        };
        let mut data = sps(&fields);
        data.extend(pps());
        data.extend(idr_slice(0, 0));
        assert_eq!(push(&mut dec, data), Ok(1));
        // A trailing intra picture with these order count LSBs. Its
        // short-term set is written in the header and names at most one
        // picture, the next in output order, which the slice does not use.
        let trailing = |lsb: u64, names_next: bool| {
            let mut w = Writer::new();
            w.u(1, 1).ue(0).ue(2).u(16, lsb).u(1, 0).ue(0);
            if names_next {
                w.ue(1).ue(0).u(1, 0);
            } else {
                w.ue(0);
            }
            intra_slice_tail(w, 1, 0)
        };
        // Each pair of pictures wraps the LSBs once more, which adds 2^16 to
        // the order count.
        for wrap in 1..1u32 << 15 {
            assert_eq!(push(&mut dec, trailing(0x8000, false)), Ok(1), "{wrap}");
            assert_eq!(push(&mut dec, trailing(0, false)), Ok(1), "{wrap}");
        }
        assert_eq!(push(&mut dec, trailing(0x8000, false)), Ok(1));
        // One more wrap would reach 2^31.
        let result = push(&mut dec, trailing(0, false));
        assert!(
            matches!(result, Err(CodecError::Malformed { .. })),
            "{result:?}"
        );
        // So would the picture after the one at 2^31 - 1.
        let result = push(&mut dec, trailing(0xffff, true));
        assert!(
            matches!(result, Err(CodecError::Malformed { .. })),
            "{result:?}"
        );
        assert_eq!(push(&mut dec, trailing(0xffff, false)), Ok(1));
    }

    #[test]
    fn a_good_hand_built_sps_and_pps_parse() {
        let mut data = sps(&GOOD);
        data.extend(pps());
        assert_eq!(push(&mut decoder(), data), Ok(0));
    }

    #[test]
    fn huge_and_bad_sizes_are_refused() {
        let cases = [
            (
                SpsFields {
                    width: 65_536,
                    height: 64,
                    ..GOOD
                },
                "wide",
            ),
            (
                SpsFields {
                    width: 64,
                    height: 1 << 20,
                    ..GOOD
                },
                "tall",
            ),
            (
                SpsFields {
                    width: 8192,
                    height: 8192,
                    ..GOOD
                },
                "too many samples",
            ),
            (SpsFields { width: 0, ..GOOD }, "zero width"),
            (
                SpsFields { width: 60, ..GOOD },
                "not a multiple of the coding block",
            ),
            (
                SpsFields {
                    width: u64::from(u32::MAX) + 5,
                    ..GOOD
                },
                "past 32 bits",
            ),
        ];
        for (fields, what) in cases {
            let result = push(&mut decoder(), sps(&fields));
            assert!(
                matches!(
                    result,
                    Err(CodecError::Limit { .. } | CodecError::Malformed { .. })
                ),
                "{what}: {result:?}"
            );
        }
    }

    #[test]
    fn unsupported_formats_say_so() {
        for (fields, what) in [
            (SpsFields { chroma: 2, ..GOOD }, "4:2:2"),
            (SpsFields { chroma: 3, ..GOOD }, "4:4:4"),
            (SpsFields { chroma: 0, ..GOOD }, "4:0:0"),
        ] {
            let result = push(&mut decoder(), sps(&fields));
            assert!(
                matches!(result, Err(CodecError::Unsupported { .. })),
                "{what}: {result:?}"
            );
        }
        for (fields, what) in [
            (
                SpsFields {
                    bit_depth_minus8: 9,
                    ..GOOD
                },
                "bit depth 17",
            ),
            (
                SpsFields {
                    log2_min_cb_minus3: 5,
                    ..GOOD
                },
                "coding block 256",
            ),
            (
                SpsFields {
                    log2_diff_cb: 9,
                    ..GOOD
                },
                "CTB 4096",
            ),
        ] {
            let result = push(&mut decoder(), sps(&fields));
            assert!(result.is_err(), "{what}: {result:?}");
        }
    }

    #[test]
    fn a_failed_push_leaves_the_decoder_unchanged_and_empty_packets_are_api_errors() {
        let mut dec = decoder();
        assert!(matches!(
            dec.push(&Packet::video(Vec::new(), 0, 1, 30, true)),
            Err(CodecError::InvalidApi { .. })
        ));
        let mut data = sps(&GOOD);
        data.extend(pps());
        assert_eq!(push(&mut dec, data), Ok(0));
        assert!(push(&mut dec, sps(&SpsFields { chroma: 2, ..GOOD })).is_err());
        // The good PPS survived the failed push: a slice naming it parses far
        // enough to fail on its missing data, not on a missing parameter set.
        let result = push(&mut dec, vec![0, 0, 1, 0x26, 0x01, 0xa0]);
        assert!(
            !matches!(result, Err(CodecError::Malformed { what, .. }) if what.contains("missing parameter set")),
            "{result:?}"
        );
    }

    #[test]
    fn garbage_and_tiny_inputs_do_not_panic() {
        let mut seed = 0x1234_5678_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for len in 0..64usize {
            for _ in 0..32 {
                let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
                if len >= 4 && next() % 2 == 0 {
                    data[..4].copy_from_slice(&[0, 0, 0, 1]);
                }
                let _ = decoder().push(&Packet::video(data, 0, 1, 30, true));
            }
        }
    }

    /// x265 streams cut short and with bytes flipped: errors, never panics.
    #[test]
    fn truncated_and_corrupted_streams_do_not_panic() {
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        let rounds: usize = std::env::var("HEVC_FUZZ_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let streams = [
            ("yuv420p", "bframes=2:ref=2:keyint=8:wpp=1:amp=1:weightp=1"),
            (
                "yuv420p10le",
                "bframes=3:ref=3:keyint=4:open-gop=1:slices=2:wpp=1:tskip=1",
            ),
            (
                "yuv420p",
                "lossless=1:ctu=16:scaling-list=default:keyint=8:wpp=0:temporal-mvp=0",
            ),
        ];
        for (index, (pix, params)) in streams.into_iter().enumerate() {
            let path = dir.join(format!(
                "hevc-malformed-{}-{index}.hevc",
                std::process::id()
            ));
            let encoded = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x96:rate=30,fade=in:0:6",
                ])
                .args([
                    "-frames:v",
                    "10",
                    "-pix_fmt",
                    pix,
                    "-c:v",
                    "libx265",
                    "-x265-params",
                ])
                .arg(format!("log-level=error:{params}"))
                .args(["-f", "hevc"])
                .arg(&path)
                .output();
            if !encoded.is_ok_and(|o| o.status.success()) {
                eprintln!("skipped: no ffmpeg with libx265");
                return;
            }
            let bytes = std::fs::read(&path).expect("stream");
            mangle(&bytes, rounds, index as u64);
        }
    }

    fn mangle(bytes: &[u8], rounds: usize, salt: u64) {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64 ^ (salt << 32);
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Cut the whole stream at many points, as one packet.
        for cut in (0..bytes.len()).step_by(37) {
            let mut dec = decoder();
            let _ = dec.push(&Packet::video(bytes[..cut.max(1)].to_vec(), 0, 1, 30, true));
            let _ = dec.drain();
        }
        // Flip bytes after the parameter sets.
        for round in 0..rounds {
            let mut data = bytes.to_vec();
            let flips = 1 + (round % 8);
            for _ in 0..flips {
                let at = (next() as usize) % data.len();
                data[at] ^= 1 << (next() % 8);
            }
            let mut dec = decoder();
            let _ = dec.push(&Packet::video(data, 0, 1, 30, true));
            let _ = dec.drain();
        }
    }
}
