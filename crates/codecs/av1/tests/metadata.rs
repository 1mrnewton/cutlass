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
    use cutlass_decode::{
        ChromaSamplePosition, CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries,
        PixelFormat, Transfer,
    };

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
