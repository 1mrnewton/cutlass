//! OBU framing through the public decoder: reserved OBUs, an OBU without
//! a size field, and the refused large-scale tile list. Digests are the
//! libdav1d frames in `libdav1d-digests.txt`.

// Clippy allows `expect` and `panic` in `cfg(test)` code; the helpers
// live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;

#[cfg(test)]
mod obu {
    use super::common;
    use cutlass_decode::CodecError;

    /// libdav1d's picture of `sb128_flat_0.obu`, a flat 256×256 keyframe
    /// with 128×128 superblocks.
    const FLAT: u64 = 0xf206_7a5e_2040_2325;

    fn flat() -> Vec<u8> {
        std::fs::read(common::fixtures().join("sb128_flat_0.obu")).expect("fixture")
    }

    fn reserved_obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            (kind << 3) | 0x02,
            u8::try_from(payload.len()).expect("size"),
        ];
        bytes.extend_from_slice(payload);
        bytes
    }

    /// The packet with its last OBU's size field dropped.
    fn without_last_size(packet: &[u8]) -> Vec<u8> {
        let (mut index, mut last) = (0, 0);
        while index < packet.len() {
            last = index;
            assert_eq!(packet[index] & 0x04, 0, "extension");
            assert_ne!(packet[index] & 0x02, 0, "size");
            assert!(packet[index + 1] < 128, "one-byte leb128");
            index += 2 + usize::from(packet[index + 1]);
        }
        assert_eq!(index, packet.len());
        let mut out = packet[..last].to_vec();
        out.push(packet[last] & !0x02);
        out.extend_from_slice(&packet[last + 2..]);
        out
    }

    fn one_picture(packet: Vec<u8>) -> Result<u64, CodecError> {
        let pictures = common::decode(&[packet], 256, 256)?;
        assert_eq!(pictures.len(), 1);
        let bytes = common::samples(&pictures[0]);
        assert_eq!(bytes.len(), 256 * 256 * 3 / 2);
        Ok(common::fnv1a(bytes))
    }

    #[test]
    fn a_reserved_obu_is_skipped() {
        let frame = flat();
        assert_eq!(one_picture(frame.clone()).expect("plain"), FLAT);
        // Bytes 0..11 are the temporal delimiter and the sequence header.
        // A reserved type 9 OBU goes between them and the frame, and a
        // reserved type 0 OBU before everything.
        let mut with_reserved = reserved_obu(0, &[0x80]);
        with_reserved.extend_from_slice(&frame[..11]);
        with_reserved.extend_from_slice(&reserved_obu(9, &[0xff, 0x01]));
        with_reserved.extend_from_slice(&frame[11..]);
        assert_eq!(one_picture(with_reserved).expect("reserved"), FLAT);
        // A reserved OBU whose size runs past the packet is refused.
        let mut overrun = reserved_obu(14, &[0x80]);
        overrun[1] = 50;
        overrun.extend_from_slice(&frame);
        assert!(common::decode(&[overrun], 256, 256).is_err());
    }

    #[test]
    fn an_obu_without_a_size_uses_the_packet_end() {
        let open = without_last_size(&flat());
        assert_eq!(one_picture(open.clone()).expect("open"), FLAT);
        // A missing trailing byte is refused.
        let mut short = open;
        short.pop();
        assert!(common::decode(&[short], 256, 256).is_err());
    }

    #[test]
    fn a_tile_list_is_refused() {
        // Large-scale tile decoding (the tile list OBU) is not implemented;
        // libdav1d lacks it too. After the anchor frames of `lst.obu`
        // decode, a tile list OBU is refused with a typed error.
        let mut packets = common::stream("lst.obu");
        assert_eq!(packets.len(), 2);
        let mut tile_list = vec![0x12, 0x00];
        tile_list.extend_from_slice(&reserved_obu(8, &[0, 0, 0, 0]));
        packets.push(tile_list);
        let error = common::decode(&packets, 128, 128).expect_err("tile list");
        assert_eq!(
            error,
            CodecError::Unsupported {
                what: "large scale tile list"
            }
        );
    }

    /// The OBUs of a packet whose OBUs all have sizes and no extension.
    fn split(packet: &[u8]) -> Vec<&[u8]> {
        let (mut index, mut out) = (0, Vec::new());
        while index < packet.len() {
            assert_eq!(packet[index] & 0x06, 0x02, "sized, no extension");
            let (mut size, mut at, mut shift) = (0usize, index + 1, 0);
            loop {
                let byte = packet[at];
                size |= usize::from(byte & 0x7f) << shift;
                shift += 7;
                at += 1;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            out.push(&packet[index..at + size]);
            index = at + size;
        }
        out
    }

    /// The packet cut before its last OBU.
    fn without_last_obu(packet: &[u8]) -> Vec<u8> {
        let obus = split(packet);
        obus[..obus.len() - 1].concat()
    }

    #[test]
    fn a_frame_cut_short_leaves_the_next_unit_its_frame_header() {
        // `tg.obu` codes each frame as a frame header OBU and tile group
        // OBUs. A unit that loses its last tile group leaves its frame
        // unfinished; the next unit starts a new temporal unit (7.5), so
        // its frame header is a new frame's and not a copy of the last.
        let packets = common::stream("tg.obu");
        assert_eq!(packets.len(), 2);
        let whole = common::decode(&packets, 128, 128).expect("whole");
        let cut = vec![
            packets[0].clone(),
            without_last_obu(&packets[1]),
            packets[1].clone(),
        ];
        let pictures = common::decode(&cut, 128, 128).expect("cut");
        assert_eq!(pictures.len(), 2);
        for (a, b) in pictures.iter().zip(&whole) {
            assert_eq!(common::samples(a), common::samples(b));
        }
    }

    #[test]
    fn a_new_sequence_drops_the_references_of_the_last() {
        // A 4:2:0 key frame, then the sequence header of the monochrome
        // `mono.ivf` and its second frame, an inter frame. A sequence
        // header that differs starts a new coded video sequence, whose
        // frames cannot refer to the last one's: libdav1d and libaom
        // refuse the frame, and so must this decoder, not predict the
        // monochrome frame from a 4:2:0 picture.
        let colour = common::stream("tg.obu");
        let mono = common::stream("mono.ivf");
        let sequence = split(&mono[0])[1];
        assert_eq!(sequence[0] >> 3 & 15, 1, "sequence header");
        let obus = split(&mono[1]);
        let inter = [obus[0], sequence, obus[1]].concat();
        let params = cutlass_decode::CodecParams::video(cutlass_decode::CodecId::Av1, 128, 128);
        let mut decoder = cutlass_av1::Av1Decoder::open(&params).expect("decoder");
        decoder.set_threads(1);
        let first = common::decode_with(&mut decoder, &colour[..1]).expect("key frame");
        assert_eq!(first.len(), 1);
        assert!(common::decode_with(&mut decoder, &[inter]).is_err());
        // The new sequence decodes from its key frame on.
        let pictures = common::decode_with(&mut decoder, &mono).expect("mono");
        let alone = common::decode(&mono, 128, 128).expect("alone");
        assert_eq!(pictures.len(), alone.len());
        for (a, b) in pictures.iter().zip(&alone) {
            assert_eq!(common::samples(a), common::samples(b));
        }
    }

    #[test]
    fn a_frame_larger_than_its_sequence_allows_is_refused() {
        // `scale.obu` opens with a 32x32 key frame whose size its header
        // codes under a 64x64 sequence maximum. With
        // `max_frame_width_minus_1` (6 bits from bit 37 of the sequence
        // header, file byte 8) lowered from 63 to 15, the key frame is
        // wider than the maximum (5.9.5), which libaom refuses.
        let mut packets = common::stream("scale.obu");
        assert_eq!(packets[0][4..9], [0, 0, 0, 2, 0xaf]);
        assert_eq!(common::decode(&packets, 64, 64).expect("plain").len(), 4);
        packets[0][8] &= !0x06;
        let error = common::decode(&packets[..1], 64, 64).expect_err("past the maximum");
        assert!(matches!(error, CodecError::Malformed { .. }), "{error:?}");
    }

    /// The OBU with an extension header naming spatial layer `spatial_id`.
    fn in_layer(obu: &[u8], spatial_id: u8) -> Vec<u8> {
        let mut out = vec![obu[0] | 0x04, spatial_id << 3];
        out.extend_from_slice(&obu[1..]);
        out
    }

    #[test]
    fn a_unit_of_two_spatial_layers_shows_the_higher() {
        // The 32x32 key frame of `scale.obu` as spatial layer 0 and its
        // 64x64 second frame, which predicts from it, as layer 1 of the
        // same temporal unit, under one operating point of both layers
        // (`operating_point_idc` 0x301: bits 12..24 of the sequence
        // header, file bytes 5 and 6). libdav1d and libaom output only
        // the 64x64 picture.
        let packets = common::stream("scale.obu");
        let (first, second) = (split(&packets[0]), split(&packets[1]));
        let mut sequence = first[1].to_vec();
        assert_eq!(sequence[3..5], [0, 0], "operating_point_idc 0");
        sequence[3] |= 0x03;
        sequence[4] = 0x01;
        let unit = [
            first[0].to_vec(),
            sequence,
            in_layer(first[2], 0),
            in_layer(second[1], 1),
        ]
        .concat();
        let pictures = common::decode(&[unit], 64, 64).expect("layers");
        assert_eq!(pictures.len(), 1);
        let plain = common::decode(&packets[..2], 64, 64).expect("plain");
        assert_eq!(common::samples(&pictures[0]), common::samples(&plain[1]));
    }
}
