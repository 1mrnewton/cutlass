//! Small fixtures through the decoder's whole life: display order and
//! times, drain, reset, damaged and hostile packets, and headers out of
//! band.
//!
//! Fixtures (`tests/fixtures`), FFmpeg 8.1.1 `mpeg4` encodes of
//! `testsrc2`, 10 pictures at 25 Hz, all with `-flags +bitexact`:
//!
//! - `asp.m4v`: 48x32, `-flags +mv4+qpel -bf 2 -mpeg_quant 1 -q:v 4
//!   -g 10`. Advanced Simple: B-VOPs, quarter samples, four vectors, the
//!   MPEG quantiser.
//! - `partitioned.m4v`: 48x32, `-data_partitioning 1 -ps 60 -q:v 6 -g 5`.
//!   Simple: video packets, data partitioning, I-VOPs at 0 and 5.
//! - `interlaced.m4v`: 48x64, `-flags +ildct+ilme -top 1 -bf 1 -q:v 4
//!   -g 10`. Field DCT, field prediction, B-VOPs.
//!
//! Each `.yuv` is FFmpeg's decode of its stream with `-threads 1
//! -cpuflags 0 -flags +bitexact -idct simple -fps_mode passthrough
//! -noautoscale -f rawvideo -pix_fmt yuv420p`.

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, SideData};
use cutlass_mpeg4::EngineDecoder;

use crate::common::{Picture, decode_all, fixture, packet, pictures, split};

const FRAME: usize = 48 * 32 * 3 / 2;

fn open() -> EngineDecoder {
    EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0)).expect("open")
}

fn joined(pictures: &[Picture]) -> Vec<u8> {
    pictures.iter().flat_map(|p| p.3.iter().copied()).collect()
}

/// `vop_coding_type` of each packet's VOP.
fn coding_types(packets: &[Vec<u8>]) -> Vec<u8> {
    packets
        .iter()
        .map(|p| {
            let at = p
                .windows(4)
                .position(|w| w == [0, 0, 1, 0xB6])
                .expect("a VOP");
            p[at + 4] >> 6
        })
        .collect()
}

/// Display indices of packets in decode order: each reference shows after
/// the B-VOPs that follow it.
fn display_order(types: &[u8]) -> Vec<i64> {
    let mut out = vec![0; types.len()];
    let mut last_ref = -1i64;
    let mut i = 0;
    while i < types.len() {
        let bs = types[i + 1..].iter().take_while(|&&t| t == 2).count();
        out[i] = last_ref + bs as i64 + 1;
        for b in 0..bs {
            out[i + 1 + b] = last_ref + 1 + b as i64;
        }
        last_ref = out[i];
        i += bs + 1;
    }
    out
}

#[test]
fn fixtures_decode_like_ffmpeg() {
    for name in ["asp", "partitioned", "interlaced"] {
        let packets = split(&fixture(&format!("{name}.m4v")));
        assert_eq!(packets.len(), 10, "{name}");
        let got = decode_all(&mut open(), &packets);
        assert_eq!(got.len(), 10, "{name}");
        assert!(
            joined(&got) == fixture(&format!("{name}.yuv")),
            "{name} differs from FFmpeg"
        );
    }
}

#[test]
fn b_vops_leave_in_display_order_with_their_times() {
    let packets = split(&fixture("asp.m4v"));
    let types = coding_types(&packets);
    assert!(types.contains(&2), "the fixture has B-VOPs");
    let display = display_order(&types);
    let mut decoder = open();
    let mut got = Vec::new();
    for (data, &pts) in packets.iter().zip(&display) {
        got.extend(pictures(
            decoder.push(&packet(data.clone(), pts)).expect("push"),
        ));
    }
    // The last reference waits for a B-VOP that never comes.
    assert_eq!(got.len(), 9);
    got.extend(pictures(decoder.drain().expect("drain")));
    let times: Vec<i64> = got.iter().map(|p| p.2).collect();
    assert_eq!(times, (0..10).collect::<Vec<_>>());
    assert!(joined(&got) == fixture("asp.yuv"));
    assert!(decoder.drain().expect("drain").is_empty());
}

#[test]
fn reset_keeps_the_headers_and_resumes_at_the_next_i_vop() {
    let packets = split(&fixture("partitioned.m4v"));
    assert_eq!(coding_types(&packets)[5], 0, "an I-VOP at 5");
    let want = fixture("partitioned.yuv");
    let mut decoder = open();
    for (n, data) in packets.iter().enumerate().take(3) {
        decoder.push(&packet(data.clone(), n as i64)).expect("push");
    }
    decoder.reset();
    // A P-VOP right after a reset has nothing to predict from.
    let after = decoder.push(&packet(packets[3].clone(), 3));
    assert!(
        matches!(after, Err(CodecError::Malformed { .. })),
        "{after:?}"
    );
    let mut got = Vec::new();
    for (n, data) in packets.iter().enumerate().skip(5) {
        got.extend(pictures(
            decoder.push(&packet(data.clone(), n as i64)).expect("push"),
        ));
    }
    got.extend(pictures(decoder.drain().expect("drain")));
    assert_eq!(got.iter().map(|p| p.2).collect::<Vec<_>>(), [5, 6, 7, 8, 9]);
    assert!(joined(&got) == want[5 * FRAME..]);
}

#[test]
fn damage_lasts_until_the_next_i_vop() {
    let packets = split(&fixture("partitioned.m4v"));
    let want = fixture("partitioned.yuv");
    for k in 1..5 {
        let len = packets[k].len();
        let truncations = [1, len / 3, len / 2, len - 1].map(|cut| packets[k][..cut].to_vec());
        let mut flipped = packets[k].clone();
        flipped[len / 2] ^= 0x5A;
        for damaged in truncations.into_iter().chain([flipped]) {
            let mut decoder = open();
            let mut got = Vec::new();
            for (n, data) in packets.iter().enumerate() {
                let data = if n == k {
                    damaged.clone()
                } else {
                    data.clone()
                };
                match decoder.push(&packet(data, n as i64)) {
                    Ok(out) => got.extend(pictures(out)),
                    Err(e) => assert_eq!(n, k, "packet {n} failed after damage to {k}: {e}"),
                }
            }
            got.extend(pictures(decoder.drain().expect("drain")));
            let healed: Vec<Picture> = got.into_iter().filter(|p| p.2 >= 5).collect();
            assert_eq!(healed.len(), 5, "damage to {k}");
            assert!(joined(&healed) == want[5 * FRAME..], "damage to {k}");
        }
    }
}

#[test]
fn hostile_bytes_fail_closed() {
    let headers = {
        let asp = fixture("asp.m4v");
        let vop = asp
            .windows(4)
            .position(|w| w == [0, 0, 1, 0xB6])
            .expect("a VOP");
        asp[..vop].to_vec()
    };
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..400 {
        let len = (next() % 300) as usize;
        let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        // Start most packets where a parser goes deep: a VOP, a VOL, or
        // a short header.
        let prefix: &[u8] = match round % 4 {
            0 => &[0, 0, 1, 0xB6],
            1 => &[0, 0, 1, 0x20],
            2 => &[0, 0, 0x80, 0x02],
            _ => &[],
        };
        data.splice(0..0, prefix.iter().copied());
        // Half the rounds have real headers in force first.
        let mut decoder = open();
        if round % 2 == 0 {
            decoder.push(&packet(headers.clone(), 0)).expect("headers");
        }
        if let Ok(out) = decoder.push(&packet(data, 1)) {
            for p in pictures(out) {
                assert_eq!(p.3.len(), p.0 as usize * p.1 as usize * 3 / 2);
            }
        }
        let _ = decoder.drain();
    }
}

#[test]
fn an_empty_packet_and_a_vop_before_any_vol_change_nothing() {
    let packets = split(&fixture("asp.m4v"));
    let mut decoder = open();
    assert!(
        decoder
            .push(&packet(Vec::new(), 0))
            .expect("empty")
            .is_empty()
    );
    let vop_only = {
        let p = &packets[1];
        let at = p
            .windows(4)
            .position(|w| w == [0, 0, 1, 0xB6])
            .expect("a VOP");
        p[at..].to_vec()
    };
    let early = decoder.push(&packet(vop_only, 0));
    assert!(
        matches!(early, Err(CodecError::Malformed { .. })),
        "{early:?}"
    );
    let got = decode_all(&mut decoder, &packets);
    assert!(joined(&got) == fixture("asp.yuv"));
}

#[test]
fn headers_may_come_from_the_configuration() {
    let packets = split(&fixture("asp.m4v"));
    let at = packets[0]
        .windows(4)
        .position(|w| w == [0, 0, 1, 0xB6])
        .expect("a VOP");
    let (headers, first) = packets[0].split_at(at);
    let mut bare = packets.clone();
    bare[0] = first.to_vec();

    let mut params = CodecParams::video(CodecId::Mpeg4, 48, 32);
    params.extra = headers.to_vec();
    let mut decoder = EngineDecoder::open(&params).expect("open");
    assert!(joined(&decode_all(&mut decoder, &bare)) == fixture("asp.yuv"));

    let mut decoder = open();
    let mut configured = packet(bare[0].clone(), 0);
    configured.side.push(SideData::Config(headers.to_vec()));
    let mut got = pictures(decoder.push(&configured).expect("push"));
    for (n, data) in bare.iter().enumerate().skip(1) {
        got.extend(pictures(
            decoder.push(&packet(data.clone(), n as i64)).expect("push"),
        ));
    }
    got.extend(pictures(decoder.drain().expect("drain")));
    assert!(joined(&got) == fixture("asp.yuv"));

    // A configuration holding a picture is not a configuration.
    params.extra = packets[0].clone();
    assert!(EngineDecoder::open(&params).is_err());
}

#[test]
fn opens_only_mpeg4_visual() {
    let refused = EngineDecoder::open(&CodecParams::video(CodecId::H264, 48, 32));
    assert!(matches!(refused, Err(CodecError::Unsupported { .. })));
    assert_eq!(open().codec(), CodecId::Mpeg4);
}
