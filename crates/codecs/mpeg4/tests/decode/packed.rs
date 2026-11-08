//! Packed bitstreams: the AVI layout DivX and Xvid write, where a packet
//! carries a P-VOP and the B-VOP before it, and a not-coded VOP stands in
//! the packet that would have held the P-VOP. Each packet's time is the
//! next display time, so the pictures must leave in display order with
//! those times.

use cutlass_decode::{CodecId, CodecParams, Decoder};
use cutlass_mpeg4::EngineDecoder;

use crate::common::{Picture, fixture, packet, pictures, split};

fn open() -> EngineDecoder {
    EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0)).expect("open")
}

fn vop_type(packet: &[u8]) -> u8 {
    let at = packet
        .windows(4)
        .position(|w| w == [0, 0, 1, 0xB6])
        .expect("a VOP");
    packet[at + 4] >> 6
}

/// A not-coded P-VOP of a layer with five time increment bits (the
/// fixture's 25 Hz): `01` P, modulo `0`, marker, increment `00000`,
/// marker, `vop_coded` 0, then stuffing.
const N_VOP: [u8; 6] = [0, 0, 1, 0xB6, 0b0101_0000, 0b0100_1111];

/// Packs `vops`, one VOP per packet in decode order, the way DivX does:
/// a reference followed by B-VOPs travels with the first of them, and a
/// not-coded VOP follows the last of them. Returns the packets and, for
/// each, whether it is a packed pair.
fn pack(vops: &[Vec<u8>]) -> Vec<(Vec<u8>, bool)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < vops.len() {
        let bs = vops[i + 1..]
            .iter()
            .take_while(|v| vop_type(v) == 2)
            .count();
        if bs == 0 {
            out.push((vops[i].clone(), false));
            i += 1;
            continue;
        }
        let mut pair = vops[i].clone();
        pair.extend_from_slice(&vops[i + 1]);
        out.push((pair, true));
        for b in &vops[i + 2..i + 1 + bs] {
            out.push((b.clone(), false));
        }
        out.push((N_VOP.to_vec(), false));
        i += bs + 1;
    }
    out
}

fn decode(decoder: &mut EngineDecoder, packets: &[Vec<u8>]) -> (Vec<Picture>, Vec<usize>) {
    let mut got = Vec::new();
    let mut failed = Vec::new();
    for (n, data) in packets.iter().enumerate() {
        match decoder.push(&packet(data.clone(), n as i64)) {
            Ok(out) => got.extend(pictures(out)),
            Err(_) => failed.push(n),
        }
    }
    got.extend(pictures(decoder.drain().expect("drain")));
    (got, failed)
}

#[test]
fn packed_p_and_b_vops_keep_their_display_times() {
    let vops = split(&fixture("asp.m4v"));
    let packed = pack(&vops);
    let pairs = packed.iter().filter(|(_, pair)| *pair).count();
    assert!(pairs >= 2, "the fixture packs {pairs} pairs");
    assert_eq!(packed.len(), vops.len(), "one placeholder for each pair");
    let packets: Vec<Vec<u8>> = packed.into_iter().map(|(data, _)| data).collect();
    let (got, failed) = decode(&mut open(), &packets);
    assert!(failed.is_empty(), "{failed:?}");
    let times: Vec<i64> = got.iter().map(|p| p.2).collect();
    assert_eq!(times, (0..10).collect::<Vec<_>>());
    let joined: Vec<u8> = got.iter().flat_map(|p| p.3.iter().copied()).collect();
    assert!(joined == fixture("asp.yuv"), "packed decode differs");
}

#[test]
fn a_placeholder_without_a_pair_is_an_ordinary_not_coded_vop() {
    // With nothing packed before it, a not-coded VOP is not a placeholder:
    // it takes no time and gives no picture.
    let vops = split(&fixture("asp.m4v"));
    let mut decoder = open();
    decoder.push(&packet(vops[0].clone(), 0)).expect("I");
    let out = decoder.push(&packet(N_VOP.to_vec(), 1)).expect("N");
    assert!(out.is_empty());
    let shown = pictures(decoder.drain().expect("drain"));
    assert_eq!(shown.len(), 1);
    assert_eq!(shown[0].2, 0);
}
