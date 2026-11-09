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
}
