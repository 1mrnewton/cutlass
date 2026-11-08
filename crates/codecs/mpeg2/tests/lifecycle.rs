//! Push, drain, reset, refusals, configuration, and the engine factory.
//!
//! `progressive.m2v` is in coded order I0 P3 B1 B2 I6 B4 B5 P9 B7 B8
//! (display index after each letter), two open GOPs.

#[cfg(test)]
mod lifecycle {
    use cutlass_decode::{
        CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, DecodeFactory, Output,
        Packet, SampleAspect, SequenceTiming, Transfer, VideoOutput,
    };
    use cutlass_mpeg2::Mpeg2Decoder;
    use cutlass_mpeg2::es::{AccessUnit, Splitter, split};

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    fn units(name: &str) -> Vec<AccessUnit> {
        split(&fixture(name)).unwrap()
    }

    fn packet(data: Vec<u8>, pts: u64) -> Packet {
        Packet {
            stream: 0,
            data,
            pts: i64::try_from(pts).unwrap(),
            duration: 1,
            timescale: 25,
            keyframe: false,
            side: Vec::new(),
        }
    }

    fn push_all(decoder: &mut Mpeg2Decoder, units: &[AccessUnit]) -> Vec<VideoOutput> {
        let mut out = Vec::new();
        for unit in units {
            out.extend(
                decoder
                    .push(&packet(unit.data.clone(), unit.display_index))
                    .unwrap(),
            );
        }
        out
    }

    fn clean(name: &str) -> Vec<VideoOutput> {
        let mut decoder = Mpeg2Decoder::default();
        let mut out = push_all(&mut decoder, &units(name));
        out.extend(decoder.drain().unwrap());
        out
    }

    /// Bytes before the first GOP or picture start code: the sequence
    /// header and its extensions.
    fn sequence_bytes(unit: &[u8]) -> usize {
        (0..unit.len() - 3)
            .find(|&i| unit[i..i + 3] == [0, 0, 1] && matches!(unit[i + 3], 0x00 | 0xb8))
            .unwrap()
    }

    #[test]
    fn empty_packet_is_an_invalid_call() {
        let mut decoder = Mpeg2Decoder::default();
        assert!(matches!(
            decoder.push(&packet(Vec::new(), 0)),
            Err(CodecError::InvalidApi { .. })
        ));
    }

    #[test]
    fn pictures_before_a_sequence_header_are_refused() {
        let units = units("progressive.m2v");
        let first = &units[0].data;
        let headerless = first[sequence_bytes(first)..].to_vec();
        let mut decoder = Mpeg2Decoder::default();
        assert!(matches!(
            decoder.push(&packet(headerless, 0)),
            Err(CodecError::Malformed { .. })
        ));
    }

    #[test]
    fn slices_without_a_picture_leave_the_decoder_unchanged() {
        let units = units("progressive.m2v");
        let mut decoder = Mpeg2Decoder::default();
        let mut out = push_all(&mut decoder, &units[..2]);
        let p = &units[2].data;
        let slice = (4..p.len() - 3)
            .find(|&i| p[i..i + 3] == [0, 0, 1] && (1..=0xaf).contains(&p[i + 3]))
            .unwrap();
        let error = decoder.push(&packet(p[slice..].to_vec(), 1)).unwrap_err();
        assert!(matches!(error, CodecError::Malformed { .. }), "{error}");
        out.extend(push_all(&mut decoder, &units[2..]));
        out.extend(decoder.drain().unwrap());
        assert_eq!(out, clean("progressive.m2v"));
    }

    #[test]
    fn truncated_units_are_errors_and_leave_the_decoder_unchanged() {
        for name in ["progressive.m2v", "interlaced.m2v", "chroma422.m2v"] {
            let units = units(name);
            for k in 0..units.len() {
                let mut decoder = Mpeg2Decoder::default();
                let mut out = push_all(&mut decoder, &units[..k]);
                let data = &units[k].data;
                for cut in [data.len() / 4, data.len() / 2, data.len() * 3 / 4] {
                    let result = decoder.push(&packet(data[..cut].to_vec(), 0));
                    assert!(result.is_err(), "{name}: unit {k} cut at {cut} decoded");
                }
                out.extend(push_all(&mut decoder, &units[k..]));
                out.extend(decoder.drain().unwrap());
                assert_eq!(out, clean(name), "{name}: unit {k}");
            }
        }
    }

    #[test]
    fn drain_releases_the_held_frame_once() {
        let units = units("progressive.m2v");
        let mut decoder = Mpeg2Decoder::default();
        assert!(push_all(&mut decoder, &units[..1]).is_empty());
        let drained = decoder.drain().unwrap();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].keyframe);
        assert!(decoder.drain().unwrap().is_empty());
        // Push after drain continues from the same references.
        assert!(push_all(&mut decoder, &units[1..2]).is_empty());
        let b = push_all(&mut decoder, &units[2..3]);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0], clean("progressive.m2v")[1]);
    }

    #[test]
    fn reset_waits_for_an_intra_frame_and_drops_leading_b_frames() {
        let units = units("progressive.m2v");
        let reference = clean("progressive.m2v");
        let mut decoder = Mpeg2Decoder::default();
        push_all(&mut decoder, &units[..3]);
        decoder.reset();
        // B2 needs references the reset dropped.
        assert!(push_all(&mut decoder, &units[3..4]).is_empty());
        assert!(decoder.drain().unwrap().is_empty());
        let mut out = push_all(&mut decoder, &units[4..]);
        out.extend(decoder.drain().unwrap());
        // I6 comes back; B4 and B5 lean on P3 from before the reset.
        let pts: Vec<i64> = out.iter().map(|f| f.pts).collect();
        assert_eq!(pts, vec![6, 7, 8, 9]);
        for frame in &out {
            assert_eq!(frame, &reference[usize::try_from(frame.pts).unwrap()]);
        }
    }

    #[test]
    fn p_frames_after_a_reset_wait_for_an_intra_frame() {
        let units = units("progressive.m2v");
        let mut decoder = Mpeg2Decoder::default();
        push_all(&mut decoder, &units[..1]);
        decoder.reset();
        assert!(push_all(&mut decoder, &units[1..4]).is_empty());
        assert!(decoder.drain().unwrap().is_empty());
    }

    #[test]
    fn chunked_bytes_split_into_the_same_units() {
        for name in ["progressive.m2v", "interlaced.m2v", "mpeg1.m1v"] {
            let stream = fixture(name);
            let whole = split(&stream).unwrap();
            let mut seed = 0x2545_f491u32;
            let mut splitter = Splitter::new();
            let mut pieces = Vec::new();
            let mut at = 0;
            while at < stream.len() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let size = 1 + (seed >> 16) as usize % 37;
                let end = (at + size).min(stream.len());
                pieces.extend(splitter.push(&stream[at..end]).unwrap());
                at = end;
            }
            pieces.extend(splitter.finish().unwrap());
            assert_eq!(pieces, whole, "{name}");
            assert_eq!(whole.len(), 10, "{name}");
        }
    }

    #[test]
    fn display_indices_follow_temporal_reference() {
        let order: Vec<u64> = units("progressive.m2v")
            .iter()
            .map(|u| u.display_index)
            .collect();
        assert_eq!(order, vec![0, 3, 1, 2, 6, 4, 5, 9, 7, 8]);
    }

    #[test]
    fn a_packet_with_several_frames_decodes_them_all() {
        let mut decoder = Mpeg2Decoder::default();
        let mut out = decoder
            .push(&packet(fixture("progressive.m2v"), 0))
            .unwrap();
        out.extend(decoder.drain().unwrap());
        let reference = clean("progressive.m2v");
        assert_eq!(out.len(), reference.len());
        for (a, b) in out.iter().zip(&reference) {
            assert_eq!(a.picture, b.picture);
        }
    }

    #[test]
    fn a_sequence_header_in_the_configuration_opens_headerless_packets() {
        let mut units = units("progressive.m2v");
        let cut = sequence_bytes(&units[0].data);
        let extra = units[0].data[..cut].to_vec();
        units[0].data.drain(..cut);
        let mut params = CodecParams::video(CodecId::Mpeg2, 96, 64);
        params.extra = extra;
        let mut decoder = Mpeg2Decoder::open(&params).unwrap();
        let mut out = push_all(&mut decoder, &units);
        out.extend(decoder.drain().unwrap());
        assert_eq!(out, clean("progressive.m2v"));

        params.extra = vec![0, 0, 1, 0xb8, 0, 0, 0, 0];
        assert!(Mpeg2Decoder::open(&params).is_err());
        params.codec = CodecId::H264;
        params.extra.clear();
        assert!(matches!(
            Mpeg2Decoder::open(&params),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn pictures_above_the_pixel_limit_are_refused() {
        let units = units("progressive.m2v");
        let mut decoder = Mpeg2Decoder::new(96 * 64 - 1);
        assert!(matches!(
            decoder.push(&packet(units[0].data.clone(), 0)),
            Err(CodecError::Limit { .. })
        ));
        let mut decoder = Mpeg2Decoder::new(96 * 64);
        assert!(decoder.push(&packet(units[0].data.clone(), 0)).is_ok());
    }

    #[test]
    fn a_size_change_flushes_and_continues() {
        let mut decoder = Mpeg2Decoder::default();
        let mut out = push_all(&mut decoder, &units("progressive.m2v"));
        out.extend(push_all(&mut decoder, &units("mpeg1.m1v")));
        out.extend(decoder.drain().unwrap());
        assert_eq!(out.len(), 20);
        assert_eq!(&out[..10], &clean("progressive.m2v")[..]);
        assert_eq!(&out[10..], &clean("mpeg1.m1v")[..]);
    }

    #[test]
    fn scalable_layers_are_unsupported() {
        let units = units("progressive.m2v");
        let mut data = units[0].data.clone();
        let at = sequence_bytes(&data);
        // sequence_scalable_extension, spatial scalability.
        data.splice(at..at, [0, 0, 1, 0xb5, 0x54, 0, 0, 0, 0, 0, 0]);
        let mut decoder = Mpeg2Decoder::default();
        assert!(matches!(
            decoder.push(&packet(data, 0)),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn sequence_error_code_is_refused() {
        let units = units("progressive.m2v");
        let mut data = units[0].data.clone();
        data.extend_from_slice(&[0, 0, 1, 0xb4]);
        let mut decoder = Mpeg2Decoder::default();
        assert!(matches!(
            decoder.push(&packet(data, 0)),
            Err(CodecError::Malformed { .. })
        ));
    }

    #[test]
    fn sequence_end_releases_the_held_frame() {
        let units = units("progressive.m2v");
        let mut data = units[0].data.clone();
        data.extend_from_slice(&[0, 0, 1, 0xb7]);
        let mut decoder = Mpeg2Decoder::default();
        assert_eq!(decoder.push(&packet(data, 0)).unwrap().len(), 1);
        assert!(decoder.drain().unwrap().is_empty());
    }

    #[test]
    fn tags_come_from_the_sequence_header() {
        let frame = &clean("progressive.m2v")[0];
        assert_eq!(frame.sample_aspect, SampleAspect::ratio(1, 1));
        assert_eq!(
            frame.timing,
            SequenceTiming {
                num_units: 1,
                time_scale: 50,
                fixed: true,
            }
        );
        assert_eq!(frame.primaries, ColorPrimaries::Unspecified);
        assert_eq!(frame.transfer, Transfer::Unspecified);
        assert_eq!(frame.matrix, ColorMatrix::Unspecified);
        // MPEG-1 aspect code 1 is square samples.
        assert_eq!(
            clean("mpeg1.m1v")[0].sample_aspect,
            SampleAspect::ratio(1, 1)
        );
    }

    #[test]
    fn the_factory_opens_both_decode_paths() {
        let mut factory = DecodeFactory::new();
        cutlass_mpeg2::register_decoders(&mut factory);
        let params = CodecParams::video(CodecId::Mpeg2, 96, 64);
        let units = units("progressive.m2v");
        let reference = clean("progressive.m2v");

        let mut decoder = factory.open(&params).unwrap();
        let mut pictures = Vec::new();
        for unit in &units {
            pictures.extend(
                decoder
                    .push(&packet(unit.data.clone(), unit.display_index))
                    .unwrap(),
            );
        }
        pictures.extend(decoder.drain().unwrap());
        assert_eq!(pictures.len(), reference.len());
        for (output, frame) in pictures.into_iter().zip(&reference) {
            let Output::Picture(picture) = output else {
                panic!("audio from a video decoder");
            };
            assert_eq!(picture.info().width, 96);
            assert_eq!(picture.pts(), frame.pts);
            let mut packed = frame.picture.y().to_vec();
            packed.extend_from_slice(frame.picture.u());
            packed.extend_from_slice(frame.picture.v());
            assert_eq!(picture.into_cpu().unwrap(), packed);
        }

        let mut video = factory.open_video(&params).unwrap();
        assert_eq!(video.codec(), CodecId::Mpeg2);
        let mut frames = Vec::new();
        for unit in &units {
            frames.extend(
                video
                    .push(&packet(unit.data.clone(), unit.display_index))
                    .unwrap(),
            );
        }
        frames.extend(video.drain().unwrap());
        assert_eq!(frames, reference);
    }
}
