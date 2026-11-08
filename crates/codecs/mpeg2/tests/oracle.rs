//! Streams from an independent encoder, checked against an independent decoder.
//!
//! Fixtures (`tests/fixtures`) were made with FFmpeg 8.1.1 from its
//! synthetic `testsrc2` source:
//!
//! - `progressive.m2v`: `-c:v mpeg2video -g 6 -bf 2 -q:v 3`, 96x64, Main
//!   profile, open GOPs of I, P and B frames.
//! - `interlaced.m2v`: `setfield=tff`, `-flags +ildct+ilme -top 1
//!   -alternate_scan 1 -intra_vlc 1 -non_linear_quant 1 -qmax 28 -dc 10
//!   -g 6 -bf 2 -q:v 4`: frame pictures with field and frame motion, field
//!   DCT, the alternate scan, the intra VLC table, the non-linear quantiser
//!   scale, and 10-bit intra DC.
//! - `chroma422.m2v`: `-pix_fmt yuv422p -g 4 -bf 1 -q:v 3 -inter_matrix ...`,
//!   4:2:2 profile with a loaded non-intra matrix.
//! - `chroma444.m2v`: a 16×16 High-profile 4:4:4 stream written from
//!   H.262, because FFmpeg's encoder has no 4:4:4 mode. The intra picture
//!   gives each of the twelve blocks its own level, in the Figure 6-12
//!   order, and the following P picture copies it with a zero vector.
//!   `chroma444.yuv` is FFmpeg 8.1.1's decode (`-f rawvideo -pix_fmt
//!   yuv444p`).
//! - `mpeg1.m1v`: `-c:v mpeg1video -g 6 -bf 2 -q:v 4`, 90x62, a size that
//!   is not a multiple of 16.
//!
//! Each `.yuv` is FFmpeg's own decode (`-f rawvideo`, `yuv420p` or
//! `yuv422p`, default IDCT), frames in display order. Any IEEE 1180
//! inverse DCT may differ from another by one in a sample, and those
//! differences carry through prediction, so the check is a bound, not
//! equality: no sample differs by more than 2, at most 3 % of samples
//! differ, and every plane of every frame is at least 55 dB. FFmpeg's two
//! integer IDCTs differ from each other by more than that on these streams.

#[cfg(test)]
mod oracle {
    use cutlass_decode::{Packet, VideoOutput};
    use cutlass_mpeg2::Mpeg2Decoder;
    use cutlass_mpeg2::es::split;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    fn decode(stream: &[u8]) -> Vec<VideoOutput> {
        let mut decoder = Mpeg2Decoder::default();
        let mut frames = Vec::new();
        for unit in split(stream).unwrap() {
            let packet = Packet {
                stream: 0,
                data: unit.data,
                pts: i64::try_from(unit.display_index).unwrap(),
                duration: 1,
                timescale: 25,
                keyframe: unit.coding_type == 1,
                side: Vec::new(),
            };
            frames.extend(decoder.push(&packet).unwrap());
        }
        frames.extend(decoder.drain().unwrap());
        frames
    }

    fn planes(frame: &VideoOutput) -> [&[u8]; 3] {
        if let Some(full) = &frame.yuv444 {
            [full.y(), full.u(), full.v()]
        } else if let Some(full) = &frame.yuv422 {
            [full.y(), full.u(), full.v()]
        } else {
            [frame.picture.y(), frame.picture.u(), frame.picture.v()]
        }
    }

    fn check(name: &str, size: (u32, u32), count: usize) {
        let stream = fixture(name);
        let reference = fixture(&format!("{}.yuv", name.split('.').next().unwrap()));
        let frames = decode(&stream);
        assert_eq!(frames.len(), count, "{name}: frame count");
        let mut offset = 0;
        let (mut differ, mut total) = (0usize, 0usize);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                (frame.picture.width(), frame.picture.height()),
                size,
                "{name}: size"
            );
            assert_eq!(frame.pts, index as i64, "{name}: display order");
            for (p, plane) in planes(frame).into_iter().enumerate() {
                let expected = &reference[offset..offset + plane.len()];
                offset += plane.len();
                let mut squared = 0u64;
                for (&a, &b) in plane.iter().zip(expected) {
                    let d = a.abs_diff(b);
                    assert!(d <= 2, "{name}: frame {index} plane {p} differs by {d}");
                    differ += usize::from(d != 0);
                    squared += u64::from(d) * u64::from(d);
                }
                total += plane.len();
                let mse = squared as f64 / plane.len() as f64;
                let psnr = if mse == 0.0 {
                    f64::INFINITY
                } else {
                    10.0 * (255.0 * 255.0 / mse).log10()
                };
                assert!(psnr >= 55.0, "{name}: frame {index} plane {p} {psnr:.1} dB");
            }
        }
        assert_eq!(offset, reference.len(), "{name}: reference length");
        assert!(
            differ * 100 <= total * 3,
            "{name}: {differ} of {total} samples differ"
        );
    }

    #[test]
    fn progressive_main_profile_matches_the_reference_decoder() {
        check("progressive.m2v", (96, 64), 10);
    }

    #[test]
    fn interlaced_frame_pictures_match_the_reference_decoder() {
        check("interlaced.m2v", (96, 64), 10);
    }

    #[test]
    fn chroma_444_block_placement_matches_the_reference_decoder() {
        let frames = decode(&fixture("chroma444.m2v"));
        assert!(frames.iter().all(|f| f.yuv444.is_some()));
        check("chroma444.m2v", (16, 16), 2);
    }

    #[test]
    fn chroma_422_with_a_loaded_matrix_matches_the_reference_decoder() {
        let frames = decode(&fixture("chroma422.m2v"));
        assert!(frames.iter().all(|f| f.yuv422.is_some()));
        check("chroma422.m2v", (96, 64), 8);
    }

    #[test]
    fn mpeg1_matches_the_reference_decoder() {
        check("mpeg1.m1v", (90, 62), 10);
    }

    #[test]
    fn keyframes_are_the_intra_frames() {
        let frames = decode(&fixture("progressive.m2v"));
        let keys: Vec<usize> = frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.keyframe)
            .map(|(i, _)| i)
            .collect();
        // Display order I B B P B B I B B P: the intra frames show at 0 and 6.
        assert_eq!(keys, vec![0, 6]);
    }
}
