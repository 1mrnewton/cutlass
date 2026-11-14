//! The description published with each shown picture: colour, chroma
//! position, display size, timing, and level. Pictures are compared with
//! libdav1d through `libdav1d-digests.txt` (`tests/fixtures.rs`); the
//! digests here are the same `ffmpeg -c:v libdav1d -threads 1` frames.

// Clippy allows `expect` and `panic` in `cfg(test)` code; the helpers
// live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;

#[cfg(test)]
mod metadata {
    use super::common;
    use cutlass_av1::Av1Decoder;
    use cutlass_av1::dec::obu::{OBU_FRAME, OBU_FRAME_HEADER, OBU_METADATA, split_obus};
    use cutlass_av1::enc::bitwriter::write_obu;
    use cutlass_decode::{
        ChromaSamplePosition, CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries,
        ColorRange, ContentLight, ItuT35, MasteringDisplay, PixelFormat, Transfer,
    };

    /// `packet` with its OBUs passed through `edit`, which returns the OBUs
    /// (type and payload) to write in place of each one.
    fn edit_obus(packet: &[u8], mut edit: impl FnMut(u8, &[u8]) -> Vec<(u8, Vec<u8>)>) -> Vec<u8> {
        let mut out = Vec::new();
        for obu in split_obus(packet).expect("obus") {
            for (obu_type, payload) in edit(obu.obu_type, obu.payload) {
                write_obu(&mut out, obu_type, 0, 0, false, &payload);
            }
        }
        out
    }

    /// `packet` with `extra` metadata payloads before its `nth` frame or
    /// frame header OBU.
    fn metadata_before_frame(packet: &[u8], nth: usize, extra: &[Vec<u8>]) -> Vec<u8> {
        let mut seen = 0;
        edit_obus(packet, |obu_type, payload| {
            let mut out = Vec::new();
            if obu_type == OBU_FRAME || obu_type == OBU_FRAME_HEADER {
                if seen == nth {
                    out.extend(extra.iter().map(|m| (OBU_METADATA, m.clone())));
                }
                seen += 1;
            }
            out.push((obu_type, payload.to_vec()));
            out
        })
    }

    fn without_metadata(packet: &[u8]) -> Vec<u8> {
        edit_obus(packet, |obu_type, payload| {
            if obu_type == OBU_METADATA {
                Vec::new()
            } else {
                vec![(obu_type, payload.to_vec())]
            }
        })
    }

    /// A `metadata_itut_t35()` payload with its trailing bits.
    fn t35(country: u8, extension: Option<u8>, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![4, country];
        out.extend(extension);
        out.extend_from_slice(payload);
        out.push(0x80);
        out
    }

    fn message(country: u8, extension: Option<u8>, payload: &[u8]) -> ItuT35 {
        ItuT35 {
            country_code: country,
            country_code_extension: extension,
            payload: payload.to_vec(),
        }
    }

    /// What `ffprobe -c:v libdav1d -show_frames` reports for `hdr10.ivf`.
    const HDR10_LIGHT: ContentLight = ContentLight {
        max_cll: 1000,
        max_fall: 400,
    };
    const HDR10_DISPLAY: MasteringDisplay = MasteringDisplay {
        primaries: [[44564, 20972], [17367, 45220], [9830, 3932]],
        white_point: [20493, 21561],
        max_luminance: 256_000,
        min_luminance: 82,
    };

    #[test]
    fn the_light_level_and_mastering_display_match_libdav1d() {
        // SVT-AV1 4.1.0 through FFmpeg, 64x64 10-bit, four frames with a
        // key frame every two, each key frame behind a light level and a
        // mastering display OBU: G(0.265,0.69) B(0.15,0.06) R(0.68,0.32)
        // WP(0.3127,0.329) L(1000,0.005), light level 1000 and 400. The
        // constants are the side data libdav1d attaches to each frame.
        let pictures = common::decode(&common::stream("hdr10.ivf"), 64, 64).expect("decode");
        assert_eq!(pictures.len(), 4);
        for picture in &pictures {
            let info = picture.info();
            assert_eq!(info.pixel, PixelFormat::Yuv420p10);
            assert_eq!(info.range, ColorRange::Limited);
            assert_eq!(info.content_light, Some(HDR10_LIGHT));
            assert_eq!(info.mastering_display, Some(HDR10_DISPLAY));
            assert!(picture.t35().is_empty());
        }
        let display = HDR10_DISPLAY;
        assert!((display.max_nits() - 1000.0).abs() < 1e-9);
        assert!((display.min_nits() - 0.005).abs() < 1e-4);
    }

    #[test]
    fn a_stream_without_metadata_publishes_none_and_names_its_range() {
        let packets: Vec<Vec<u8>> = common::stream("hdr10.ivf")
            .iter()
            .map(|p| without_metadata(p))
            .collect();
        let pictures = common::decode(&packets, 64, 64).expect("decode");
        assert_eq!(pictures.len(), 4);
        for picture in &pictures {
            assert_eq!(picture.info().content_light, None);
            assert_eq!(picture.info().mastering_display, None);
            assert_eq!(picture.info().range, ColorRange::Limited);
        }
    }

    #[test]
    fn later_metadata_replaces_earlier_and_holds_from_where_it_appears() {
        let mut packets = common::stream("hdr10.ivf");
        // The second key frame's light level becomes 2000 and 300.
        packets[2] = edit_obus(&packets[2], |obu_type, payload| {
            if obu_type == OBU_METADATA && payload[0] == 1 {
                vec![(obu_type, vec![1, 0x07, 0xd0, 0x01, 0x2c, 0x80])]
            } else {
                vec![(obu_type, payload.to_vec())]
            }
        });
        let pictures = common::decode(&packets, 64, 64).expect("decode");
        let later = ContentLight {
            max_cll: 2000,
            max_fall: 300,
        };
        let lights: Vec<_> = pictures.iter().map(|p| p.info().content_light).collect();
        assert_eq!(
            lights,
            [
                Some(HDR10_LIGHT),
                Some(HDR10_LIGHT),
                Some(later),
                Some(later)
            ]
        );
        assert!(
            pictures
                .iter()
                .all(|p| p.info().mastering_display == Some(HDR10_DISPLAY))
        );
    }

    #[test]
    fn metadata_stays_through_a_repeated_header_and_a_reset_and_ends_with_its_sequence() {
        let hdr = common::stream("hdr10.ivf");
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        common::decode_with(&mut decoder, &hdr[..2]).expect("first key frame");
        // The second key frame repeats the sequence header. Without its own
        // metadata OBUs it is still described by the first key frame's.
        let bare = vec![without_metadata(&hdr[2]), hdr[3].clone()];
        let pictures = common::decode_with(&mut decoder, &bare).expect("second key frame");
        assert!(
            pictures
                .iter()
                .all(|p| p.info().content_light == Some(HDR10_LIGHT)
                    && p.info().mastering_display == Some(HDR10_DISPLAY))
        );
        // A reset keeps the sequence header and what describes it.
        use cutlass_decode::Decoder as _;
        decoder.reset();
        let pictures = common::decode_with(&mut decoder, &bare).expect("after the reset");
        assert_eq!(pictures.len(), 2);
        assert_eq!(pictures[0].info().content_light, Some(HDR10_LIGHT));
        // Another sequence header starts another sequence: none of it.
        let other = common::decode_with(&mut decoder, &common::stream("color.ivf")).expect("8-bit");
        assert_eq!(other.len(), 2);
        for picture in &other {
            assert_eq!(picture.info().pixel, PixelFormat::Yuv420p);
            assert_eq!(picture.info().content_light, None);
            assert_eq!(picture.info().mastering_display, None);
        }
    }

    #[test]
    fn metadata_in_the_av1c_record_describes_the_stream() {
        // The record's configuration OBUs: the sequence header and the two
        // metadata OBUs of the first key frame. The samples carry none.
        let hdr = common::stream("hdr10.ivf");
        let mut record = vec![0x81, 0x00, 0x4c, 0x00];
        record.extend(edit_obus(&hdr[0], |obu_type, payload| {
            if obu_type == OBU_METADATA || obu_type == 1 {
                vec![(obu_type, payload.to_vec())]
            } else {
                Vec::new()
            }
        }));
        let mut params = CodecParams::video(CodecId::Av1, 64, 64);
        params.extra = record;
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let bare: Vec<Vec<u8>> = hdr.iter().map(|p| without_metadata(p)).collect();
        let pictures = common::decode_with(&mut decoder, &bare).expect("decode");
        assert_eq!(pictures.len(), 4);
        for picture in &pictures {
            assert_eq!(picture.info().content_light, Some(HDR10_LIGHT));
            assert_eq!(picture.info().mastering_display, Some(HDR10_DISPLAY));
        }
    }

    #[test]
    fn a_t35_message_goes_out_with_the_frame_after_it() {
        let mut packets = common::stream("hdr10.ivf");
        // An HDR10+ style message (United States, then the provider's own
        // bytes) on the second picture; two on the fourth, one with an
        // extended country code and one whose padding follows its trailing
        // bits.
        let mut padded = t35(0xb5, None, &[9, 0, 8]);
        padded.extend_from_slice(&[0, 0, 0]);
        packets[1] =
            metadata_before_frame(&packets[1], 0, &[t35(0xb5, None, &[0, 0x3c, 0, 1, 4, 1])]);
        packets[3] =
            metadata_before_frame(&packets[3], 0, &[t35(0xff, Some(0x26), &[1, 2]), padded]);
        let pictures = common::decode(&packets, 64, 64).expect("decode");
        assert_eq!(pictures.len(), 4);
        assert!(pictures[0].t35().is_empty());
        assert_eq!(
            pictures[1].t35(),
            [message(0xb5, None, &[0, 0x3c, 0, 1, 4, 1])]
        );
        assert!(pictures[2].t35().is_empty());
        assert_eq!(
            pictures[3].t35(),
            [
                message(0xff, Some(0x26), &[1, 2]),
                message(0xb5, None, &[9, 0, 8])
            ]
        );
        // The samples are the ones libdav1d shows without the messages.
        let plain = common::decode(&common::stream("hdr10.ivf"), 64, 64).expect("decode");
        for (a, b) in pictures.iter().zip(&plain) {
            assert_eq!(common::samples(a), common::samples(b));
        }
    }

    #[test]
    fn a_t35_message_stays_with_a_hidden_frame_until_it_is_shown() {
        use cutlass_av1::Av1Encoder;
        use cutlass_encode::{CpuFrame, Encoder as _, FrameInput, FrameTime};
        // A mini-GOP of four: the unit of picture 1 codes the hidden
        // pictures 4 and 2 before it, and pictures 2 and 4 are shown later
        // by show_existing_frame.
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(4);
        let mut packets = Vec::new();
        for i in 0..5u32 {
            let bytes: Vec<u8> = (0..64 * 64 * 3 / 2)
                .map(|k| ((k * 7 + i * 13) % 200 + 20) as u8)
                .collect();
            let time = FrameTime::new(i64::from(i), 1, 30).expect("time");
            let frame = CpuFrame::new(64, 64, PixelFormat::Yuv420p, bytes, time).expect("frame");
            packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        let mut data: Vec<Vec<u8>> = packets.into_iter().map(|p| p.data).collect();
        assert_eq!(data.len(), 5);
        let frames = |packet: &[u8]| -> Vec<bool> {
            split_obus(packet)
                .expect("obus")
                .iter()
                .filter(|o| o.obu_type == OBU_FRAME || o.obu_type == OBU_FRAME_HEADER)
                .map(|o| o.payload[0] & 0x80 != 0)
                .collect()
        };
        // Three coded frames, then a shown stored one, a coded one, a
        // shown stored one.
        assert_eq!(frames(&data[1]), [false, false, false]);
        assert_eq!(frames(&data[2]), [true]);
        assert_eq!(frames(&data[4]), [true]);
        let unit = data[1].clone();
        let unit = metadata_before_frame(&unit, 0, &[t35(0xb5, None, b"four")]);
        let unit = metadata_before_frame(&unit, 1, &[t35(0xb5, None, b"two")]);
        data[1] = metadata_before_frame(&unit, 2, &[t35(0xb5, None, b"one")]);
        data[4] = metadata_before_frame(&data[4], 0, &[t35(0xb5, None, b"shown")]);
        let pictures = common::decode(&data, 64, 64).expect("decode");
        assert_eq!(pictures.len(), 5);
        assert!(pictures[0].t35().is_empty());
        assert_eq!(pictures[1].t35(), [message(0xb5, None, b"one")]);
        assert_eq!(pictures[2].t35(), [message(0xb5, None, b"two")]);
        assert!(pictures[3].t35().is_empty());
        assert_eq!(
            pictures[4].t35(),
            [message(0xb5, None, b"four"), message(0xb5, None, b"shown")]
        );
    }

    #[test]
    fn metadata_this_decoder_does_not_read_is_left_alone() {
        let mut packets = common::stream("hdr10.ivf");
        // Scalability (3) and timecode (5) are not read, nor an
        // unregistered type; a message after the last frame of its unit
        // belongs to no frame.
        packets[1] = metadata_before_frame(
            &packets[1],
            0,
            &[vec![3, 0, 0x80], vec![5, 0, 0, 0, 0x80], vec![31, 1, 2, 3]],
        );
        packets[1].extend_from_slice(&{
            let mut tail = Vec::new();
            write_obu(
                &mut tail,
                OBU_METADATA,
                0,
                0,
                false,
                &t35(0xb5, None, b"late"),
            );
            tail
        });
        let pictures = common::decode(&packets, 64, 64).expect("decode");
        assert_eq!(pictures.len(), 4);
        assert!(pictures.iter().all(|p| p.t35().is_empty()));
        let plain = common::decode(&common::stream("hdr10.ivf"), 64, 64).expect("decode");
        for (a, b) in pictures.iter().zip(&plain) {
            assert_eq!(common::samples(a), common::samples(b));
            assert_eq!(a.info(), b.info());
        }
    }

    #[test]
    fn cut_short_metadata_is_refused_and_leaves_the_decoder_as_it_was() {
        let packets = common::stream("hdr10.ivf");
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        for bad in [
            vec![1, 0x03],       // a light level of one byte
            vec![2, 0, 1, 2, 3], // a mastering display of four
            vec![4],             // a T.35 message without its country
            vec![4, 0xff],       // and without its extension byte
            Vec::new(),          // no metadata_type at all
        ] {
            let mut decoder = Av1Decoder::open(&params).expect("open");
            common::decode_with(&mut decoder, &packets[..1]).expect("key frame");
            let broken = metadata_before_frame(&packets[1], 0, std::slice::from_ref(&bad));
            let error = common::decode_with(&mut decoder, &[broken]).expect_err("cut short");
            assert!(
                matches!(error, CodecError::Truncated { .. }),
                "{bad:?}: {error:?}"
            );
            // The good unit still decodes, with the metadata from before.
            let pictures = common::decode_with(&mut decoder, &packets[1..2]).expect("good unit");
            assert_eq!(pictures.len(), 1);
            assert_eq!(pictures[0].info().content_light, Some(HDR10_LIGHT));
        }
        // Metadata of a unit that fails does not stay.
        let mut decoder = Av1Decoder::open(&params).expect("open");
        common::decode_with(&mut decoder, &packets[..1]).expect("key frame");
        let mut failing =
            metadata_before_frame(&packets[1], 0, &[vec![1, 0x07, 0xd0, 0x01, 0x2c, 0x80]]);
        failing.truncate(failing.len() - 20);
        assert!(common::decode_with(&mut decoder, &[failing]).is_err());
        let pictures = common::decode_with(&mut decoder, &packets[1..2]).expect("good unit");
        assert_eq!(pictures[0].info().content_light, Some(HDR10_LIGHT));
    }

    #[test]
    fn colour_timing_and_render_size_stay_on_the_picture() {
        // aomenc 64×64, one keyframe and one inter frame, with BT.709 colour,
        // colocated chroma, and a constant 30 Hz timing_info. The keyframe's
        // display size is 80×40 (libdav1d reports a 2:1 sample aspect for it);
        // the inter frame's is its frame size. The samples stay 64×64.
        let pictures = common::decode(&common::stream("color.ivf"), 64, 64).expect("decode");
        assert_eq!(pictures.len(), 2);
        let renders = [(80, 40), (64, 64)];
        let digests = [0x1b33_c4d2_4867_47f8, 0x7963_d39f_992f_c21f];
        for (index, picture) in pictures.iter().enumerate() {
            let info = picture.info();
            assert_eq!(info.pixel, PixelFormat::Yuv420p);
            assert_eq!((info.width, info.height), (64, 64));
            assert_eq!(
                (info.render_width, info.render_height),
                renders[index],
                "frame {index}"
            );
            assert_eq!(info.primaries, ColorPrimaries::Bt709);
            assert_eq!(info.transfer, Transfer::Bt709);
            assert_eq!(info.matrix, ColorMatrix::Bt709);
            assert_eq!(info.chroma_position, ChromaSamplePosition::Colocated);
            assert_eq!(
                (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
                (1, 1, 1)
            );
            assert_eq!(info.num_units_in_display_tick, 1);
            assert_eq!(info.time_scale, 30);
            assert!(info.equal_picture_interval);
            assert_eq!(info.num_ticks_per_picture_minus_1, Some(0));
            assert_eq!(info.num_units_in_decoding_tick, 0);
            let bytes = common::samples(picture);
            assert_eq!(bytes.len(), 64 * 64 * 3 / 2);
            assert_eq!(common::fnv1a(bytes), digests[index], "frame {index}");
        }
    }

    #[test]
    fn decoder_model_info_is_published_and_the_picture_matches_dav1d() {
        // aomenc --timing-info=model, 64×64, two frames. The model is parsed,
        // not scheduled; its decoding tick and the unequal picture interval
        // reach the picture.
        let pictures = common::decode(&common::stream("model.ivf"), 64, 64).expect("decode");
        let digests = [0x1b33_c4d2_4867_47f8, 0x7963_d39f_992f_c21f];
        assert_eq!(pictures.len(), 2);
        for (index, picture) in pictures.iter().enumerate() {
            let info = picture.info();
            assert_eq!(info.num_units_in_decoding_tick, 1);
            assert_eq!(info.num_units_in_display_tick, 1);
            assert_eq!(info.time_scale, 30);
            assert!(!info.equal_picture_interval);
            assert_eq!(info.num_ticks_per_picture_minus_1, None);
            assert_eq!(
                (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
                (2, 2, 2)
            );
            assert_eq!(info.matrix, ColorMatrix::Unspecified);
            assert_eq!(
                common::fnv1a(common::samples(picture)),
                digests[index],
                "frame {index}"
            );
        }
    }

    #[test]
    fn a_level_three_picture_wider_than_4096_matches_dav1d() {
        // aomenc --target-seq-level-idx=4, 4352×64, two frames. Level 3.0
        // allows this edge, and the level reaches the picture.
        let pictures = common::decode(&common::stream("level.ivf"), 4352, 64).expect("decode");
        let digests = [0x419d_ca99_d7ff_bfd7, 0x47af_41d4_871f_de86];
        assert_eq!(pictures.len(), 2);
        for (index, picture) in pictures.iter().enumerate() {
            let info = picture.info();
            assert_eq!((info.width, info.height), (4352, 64));
            assert_eq!(info.seq_level_idx, 4);
            // No timing info in this sequence.
            assert_eq!((info.num_units_in_display_tick, info.time_scale), (0, 0));
            let bytes = common::samples(picture);
            assert_eq!(bytes.len(), 4352 * 64 * 3 / 2);
            assert_eq!(common::fnv1a(bytes), digests[index], "frame {index}");
        }
    }

    #[test]
    fn vertical_chroma_position_matches_dav1d() {
        // aomenc --chroma-sample-position=vertical, 128×128.
        let pictures =
            common::decode(&common::stream("chroma-vert.ivf"), 128, 128).expect("decode");
        assert_eq!(pictures.len(), 1);
        let info = pictures[0].info();
        assert_eq!(info.chroma_position, ChromaSamplePosition::Vertical);
        let bytes = common::samples(&pictures[0]);
        assert_eq!(bytes.len(), 128 * 128 * 3 / 2);
        assert_eq!(common::fnv1a(bytes), 0xee51_8a06_f71e_a2a3);
    }

    #[test]
    fn an_identity_matrix_is_published_as_gbr() {
        // aomenc --i444 --matrix-coefficients=identity --film-grain-test=1,
        // 128×128. libdav1d calls the picture gbrp. The CICP bytes are the
        // header's: BT.709 primaries and transfer, matrix 0.
        let pictures =
            common::decode(&common::stream("grain-identity.ivf"), 128, 128).expect("decode");
        assert_eq!(pictures.len(), 1);
        let info = pictures[0].info();
        assert_eq!(info.pixel, PixelFormat::Gbrp);
        assert_eq!(info.matrix, ColorMatrix::Identity);
        assert_eq!(
            (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
            (1, 1, 0)
        );
        let bytes = common::samples(&pictures[0]);
        assert_eq!(bytes.len(), 128 * 128 * 3);
        assert_eq!(common::fnv1a(bytes), 0xb4c1_cb35_9263_7630);
    }

    #[test]
    fn still_pictures_publish_their_header() {
        // aomenc --limit=1, 128×128, with the full and with the reduced still
        // picture header. Neither carries colour or timing; the reduced one
        // has a single level and nothing else of the operating points.
        for (name, digest) in [
            ("still-full.ivf", 0xefec_d6e6_78fd_c52au64),
            ("still-reduced.ivf", 0x7e4f_ad64_6443_12cc),
        ] {
            let pictures = common::decode(&common::stream(name), 128, 128).expect(name);
            assert_eq!(pictures.len(), 1, "{name}");
            let info = pictures[0].info();
            assert_eq!((info.width, info.height), (128, 128), "{name}");
            assert_eq!(
                (info.render_width, info.render_height),
                (128, 128),
                "{name}"
            );
            assert_eq!(
                (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
                (2, 2, 2),
                "{name}"
            );
            assert_eq!(
                info.chroma_position,
                ChromaSamplePosition::Unknown,
                "{name}"
            );
            assert_eq!((info.num_units_in_display_tick, info.time_scale), (0, 0));
            assert!(!info.equal_picture_interval);
            assert_eq!(info.num_ticks_per_picture_minus_1, None);
            assert_eq!(info.num_units_in_decoding_tick, 0);
            assert_eq!(info.seq_level_idx, 0, "{name}");
            assert_eq!(
                common::fnv1a(common::samples(&pictures[0])),
                digest,
                "{name}"
            );
        }
    }

    #[test]
    fn a_frame_size_from_a_reference_keeps_the_render_size() {
        // The sequence is 64×64 and every frame is 32×32. Later frames copy
        // their size from a reference, render size included.
        let pictures = common::decode(&common::stream("size.obu"), 64, 64).expect("decode");
        assert_eq!(pictures.len(), 4);
        let first = pictures[0].info();
        for picture in &pictures {
            let info = picture.info();
            assert_eq!((info.width, info.height), (32, 32));
            assert_eq!(
                (info.render_width, info.render_height),
                (first.render_width, first.render_height)
            );
        }
    }

    #[test]
    fn an_operating_point_keeps_its_temporal_layer() {
        // Two operating points. Point 0 is temporal layer 0. Point 1 also
        // includes temporal layer 1, a second copy of the keyframe. Digests
        // are libdav1d at `-oppoint 0` and `-oppoint 1`.
        let packets = common::stream("operating-point.ivf");
        assert_eq!(packets.len(), 3);
        let base = [0xf021_b9cb_b654_3518u64, 0x21e9_8a09_7459_161c];
        let params = CodecParams::video(CodecId::Av1, 64, 64);

        let mut decoder = Av1Decoder::open(&params).expect("open");
        let pictures = common::decode_with(&mut decoder, &packets).expect("point 0");
        let digests: Vec<u64> = pictures
            .iter()
            .map(|picture| common::fnv1a(common::samples(picture)))
            .collect();
        assert_eq!(digests, base, "point 0 drops temporal layer 1");

        let mut layered = Av1Decoder::open(&params).expect("open");
        layered.select_operating_point(1).expect("point 1");
        let pictures = common::decode_with(&mut layered, &packets).expect("point 1");
        let digests: Vec<u64> = pictures
            .iter()
            .map(|picture| common::fnv1a(common::samples(picture)))
            .collect();
        assert_eq!(digests, [base[0], base[1], base[0]], "point 1");

        // Past the last point: stored before the sequence is known, refused
        // at its sequence header, and refused at once after it.
        let mut refused = Av1Decoder::open(&params).expect("open");
        refused.select_operating_point(2).expect("stored");
        let error =
            common::decode_with(&mut refused, &packets[..1]).expect_err("past the last point");
        assert_eq!(
            error,
            CodecError::Unsupported {
                what: "operating point"
            }
        );
        let mut known = Av1Decoder::open(&params).expect("open");
        common::decode_with(&mut known, &packets[..1]).expect("point 0");
        assert_eq!(
            known.select_operating_point(2),
            Err(CodecError::Unsupported {
                what: "operating point"
            })
        );
        assert_eq!(
            known.select_operating_point(32),
            Err(CodecError::Unsupported {
                what: "operating point"
            })
        );
        // The refused selection left point 0 in place.
        let pictures = common::decode_with(&mut known, &packets[1..]).expect("point 0 still");
        assert_eq!(pictures.len(), 1);
    }
}
