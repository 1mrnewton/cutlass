//! Hand-built frames: every refusal path, truncation at every byte, and
//! the decoder lifecycle.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoder, Output, Packet, SampleFormat, SideData,
};

use super::{FlacDecoder, Md5Check};
use crate::bits::BitWriter;
use crate::crc::crc16;
use crate::header::{ChannelAssignment, FrameHeader};
use crate::md5::Md5;
use crate::streaminfo::StreamInfo;

fn header(assignment: ChannelAssignment, block: u32, bits: u8, number: u64) -> FrameHeader {
    FrameHeader {
        variable_block: false,
        block_size: block,
        sample_rate: Some(44_100),
        assignment,
        bits: Some(bits),
        number,
        len: 0,
    }
}

fn mono(block: u32) -> FrameHeader {
    header(ChannelAssignment::Independent(1), block, 16, 0)
}

/// Header, `body`, zero padding to a byte, and the CRC-16.
fn frame(header: FrameHeader, body: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
    let mut out = BitWriter::with_capacity(64);
    header.write(&mut out).expect("header");
    body(&mut out);
    out.align();
    let mut bytes = out.into_bytes();
    let crc = crc16(&bytes);
    bytes.extend_from_slice(&crc.to_be_bytes());
    bytes
}

fn subframe(out: &mut BitWriter, kind: u64, wasted: u64) {
    out.put(1, 0);
    out.put(6, kind);
    if wasted == 0 {
        out.put(1, 0);
    } else {
        out.put(1, 1);
        out.put_unary(wasted - 1);
    }
}

/// Rice-codes `residuals` with parameter `k` in one partition (method 0).
fn rice(out: &mut BitWriter, k: u32, residuals: &[i64]) {
    out.put(2, 0);
    out.put(4, 0);
    out.put(4, u64::from(k));
    for residual in residuals {
        let folded = if *residual >= 0 {
            (*residual as u64) << 1
        } else {
            ((-*residual as u64) << 1) - 1
        };
        out.put_unary(folded >> k);
        out.put(k, folded & ((1u64 << k) - 1));
    }
}

fn decode(bytes: &[u8]) -> Result<Vec<i32>, CodecError> {
    FlacDecoder::new(None)
        .decode(bytes)
        .map(|frame| frame.samples)
}

fn malformed(bytes: &[u8]) -> &'static str {
    match decode(bytes) {
        Err(CodecError::Malformed { what, .. }) => what,
        other => panic!("expected a malformed frame, got {other:?}"),
    }
}

/// A two-channel left/side frame touching LPC, a fixed predictor, wasted
/// bits, partitioned Rice coding, and an escaped partition.
fn rich() -> (Vec<u8>, Vec<i32>) {
    let block = 64usize;
    let left_residual: Vec<i64> = (0..block as i64).map(|i| (i * 7) % 11 - 5).collect();
    let side_residual: Vec<i64> = (0..block as i64).map(|i| (i * 5) % 7 - 3).collect();
    let bytes = frame(header(ChannelAssignment::LeftSide, 64, 16, 0), |out| {
        // Left: LPC order 2, coefficients [1, 0], precision 4, shift 0,
        // four partitions, the third escaped at 5 bits.
        subframe(out, 32 + 1, 0);
        out.put_signed(16, 100);
        out.put_signed(16, 90);
        out.put(4, 3);
        out.put_signed(5, 0);
        out.put_signed(4, 1);
        out.put_signed(4, 0);
        out.put(2, 1);
        out.put(4, 2);
        for part in 0..4 {
            let start = if part == 0 { 2 } else { part * 16 };
            let values = &left_residual[start..(part + 1) * 16];
            if part == 2 {
                out.put(5, 31);
                out.put(5, 5);
                for value in values {
                    out.put_signed(5, *value);
                }
            } else {
                out.put(5, 2);
                for value in values {
                    let folded = if *value >= 0 {
                        (*value as u64) << 1
                    } else {
                        ((-*value as u64) << 1) - 1
                    };
                    out.put_unary(folded >> 2);
                    out.put(2, folded & 3);
                }
            }
        }
        // Side (17 bits): fixed order 2, one wasted bit.
        subframe(out, 8 + 2, 1);
        out.put_signed(16, -4);
        out.put_signed(16, 3);
        rice(out, 1, &side_residual[2..]);
    });
    let mut left = vec![100i64, 90];
    for i in 2..block {
        left.push(left[i - 1] + left_residual[i]);
    }
    let mut side = vec![-4i64, 3];
    for i in 2..block {
        side.push(side_residual[i] + 2 * side[i - 1] - side[i - 2]);
    }
    let expected = left
        .iter()
        .zip(&side)
        .flat_map(|(l, s)| [*l as i32, (l - (s << 1)) as i32])
        .collect();
    (bytes, expected)
}

#[test]
fn a_rich_frame_decodes_to_the_values_built_into_it() {
    let (bytes, expected) = rich();
    assert_eq!(decode(&bytes).expect("rich"), expected);
}

#[test]
fn every_truncation_is_refused() {
    let (bytes, _) = rich();
    for cut in 0..bytes.len() {
        assert!(decode(&bytes[..cut]).is_err(), "raw cut at {cut}");
        // The same prefix with a fresh CRC-16 reaches the subframe readers.
        if cut > 2 {
            let mut prefix = bytes[..cut - 2].to_vec();
            let crc = crc16(&prefix);
            prefix.extend_from_slice(&crc.to_be_bytes());
            assert!(decode(&prefix).is_err(), "cut at {cut} with a fresh crc");
        }
    }
}

#[test]
fn every_flipped_bit_is_refused_or_decodes_without_panicking() {
    let (bytes, _) = rich();
    for bit in 0..bytes.len() * 8 {
        let mut damaged = bytes.clone();
        damaged[bit / 8] ^= 0x80 >> (bit % 8);
        assert!(decode(&damaged).is_err(), "bit {bit} passed the crc");
        // With the CRC-16 refreshed the damage reaches the body.
        let end = damaged.len() - 2;
        let crc = crc16(&damaged[..end]);
        damaged[end..].copy_from_slice(&crc.to_be_bytes());
        let _ = decode(&damaged);
    }
}

#[test]
fn a_bad_crc_16_is_refused() {
    let (mut bytes, _) = rich();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    assert_eq!(malformed(&bytes), "frame crc");
}

#[test]
fn reserved_subframe_types_are_refused() {
    for kind in (2..8).chain(13..32) {
        let bytes = frame(mono(16), |out| {
            subframe(out, kind, 0);
            for _ in 0..16 {
                out.put(16, 0);
            }
        });
        assert_eq!(malformed(&bytes), "reserved subframe type", "kind {kind}");
    }
}

#[test]
fn a_set_subframe_padding_bit_is_refused() {
    let bytes = frame(mono(16), |out| {
        out.put(1, 1);
        out.put(6, 0);
        out.put(1, 0);
        out.put(16, 0);
    });
    assert_eq!(malformed(&bytes), "subframe padding bit");
}

#[test]
fn wasted_bits_must_leave_sample_bits() {
    let bytes = frame(mono(16), |out| {
        subframe(out, 0, 16);
        out.put(1, 0);
    });
    assert_eq!(malformed(&bytes), "wasted bits leave no sample bits");
    // Fifteen wasted bits leave a one-bit constant: legal.
    let bytes = frame(mono(16), |out| {
        subframe(out, 0, 15);
        out.put(1, 1);
    });
    assert_eq!(decode(&bytes).expect("one bit"), vec![-(1 << 15); 16]);
    // A unary run longer than the depth stops at the limit.
    let bytes = frame(mono(16), |out| {
        out.put(8, 1);
        out.put(32, 0);
    });
    assert!(decode(&bytes).is_err());
}

#[test]
fn reserved_residual_methods_are_refused() {
    for method in [2, 3] {
        let bytes = frame(mono(16), |out| {
            subframe(out, 8, 0);
            out.put(2, method);
            out.put(4, 0);
            out.put(4, 0);
        });
        assert_eq!(malformed(&bytes), "reserved residual coding method");
    }
}

#[test]
fn partitions_must_divide_the_block_and_outlast_the_predictor() {
    let bytes = frame(mono(24), |out| {
        subframe(out, 8, 0);
        out.put(2, 0);
        out.put(4, 4);
    });
    assert_eq!(
        malformed(&bytes),
        "partition order does not divide the block"
    );
    let bytes = frame(mono(16), |out| {
        subframe(out, 8 + 4, 0);
        for _ in 0..4 {
            out.put(16, 0);
        }
        out.put(2, 0);
        out.put(4, 3);
    });
    assert_eq!(
        malformed(&bytes),
        "first partition shorter than the predictor"
    );
    // A first partition exactly as long as the order holds no residuals;
    // libFLAC and ffmpeg both accept it.
    let bytes = frame(mono(16), |out| {
        subframe(out, 8 + 2, 0);
        out.put(16, 5);
        out.put(16, 5);
        out.put(2, 0);
        out.put(4, 3);
        for part in 0..8 {
            out.put(4, 0);
            for _ in 0..if part == 0 { 0 } else { 2 } {
                out.put_unary(0);
            }
        }
    });
    assert_eq!(decode(&bytes).expect("empty first partition"), vec![5; 16]);
}

#[test]
fn a_predictor_longer_than_the_block_is_refused() {
    let bytes = frame(mono(2), |out| {
        subframe(out, 8 + 4, 0);
        out.put(16, 0);
        out.put(16, 0);
    });
    assert_eq!(malformed(&bytes), "predictor order above block size");
    let bytes = frame(mono(16), |out| {
        subframe(out, 63, 0);
        for _ in 0..16 {
            out.put(16, 0);
        }
    });
    assert_eq!(malformed(&bytes), "predictor order above block size");
}

#[test]
fn lpc_precision_fifteen_and_a_negative_shift_are_refused() {
    let lpc = |precision: u64, shift: i64| {
        frame(mono(16), move |out| {
            subframe(out, 32, 0);
            out.put(16, 0);
            out.put(4, precision);
            out.put_signed(5, shift);
            out.put(4, 0);
        })
    };
    assert_eq!(malformed(&lpc(15, 0)), "forbidden lpc precision");
    assert_eq!(malformed(&lpc(3, -1)), "negative lpc shift");
}

#[test]
fn a_residual_outside_32_bits_is_refused() {
    // k = 30: a quotient of 3 with all low bits set folds to 2^32 - 1.
    let bytes = frame(mono(16), |out| {
        subframe(out, 8, 0);
        out.put(2, 1);
        out.put(4, 0);
        out.put(5, 30);
        out.put_unary(3);
        out.put(30, (1 << 30) - 1);
    });
    assert_eq!(malformed(&bytes), "residual outside 32 bits");
    // A quotient past the limit stops at the unary limit.
    let bytes = frame(mono(16), |out| {
        subframe(out, 8, 0);
        out.put(2, 1);
        out.put(4, 0);
        out.put(5, 30);
        out.put_unary(4);
    });
    assert!(matches!(decode(&bytes), Err(CodecError::Malformed { .. })));
}

#[test]
fn predictions_and_decorrelation_stay_inside_the_depth() {
    let bytes = frame(mono(16), |out| {
        subframe(out, 8 + 1, 0);
        out.put_signed(16, 32_767);
        rice(out, 0, &[1; 15]);
    });
    assert_eq!(malformed(&bytes), "fixed prediction leaves the bit depth");
    let bytes = frame(mono(16), |out| {
        subframe(out, 32, 0);
        out.put_signed(16, 32_767);
        out.put(4, 3);
        out.put_signed(5, 0);
        out.put_signed(4, 1);
        rice(out, 0, &[1; 15]);
    });
    assert_eq!(malformed(&bytes), "lpc prediction leaves the bit depth");
    // Left 32767 with side -1 is right 32768.
    let bytes = frame(header(ChannelAssignment::LeftSide, 16, 16, 0), |out| {
        subframe(out, 0, 0);
        out.put_signed(16, 32_767);
        subframe(out, 0, 0);
        out.put_signed(17, -1);
    });
    assert_eq!(
        malformed(&bytes),
        "decorrelated sample leaves the bit depth"
    );
}

#[test]
fn padding_and_trailing_bytes_are_refused() {
    // Fixed order 0 with sixteen zero residuals ends at bit 34 of the
    // body, so the three bits after it are padding, and nonzero.
    let mut bytes = frame(mono(16), |out| {
        subframe(out, 8, 0);
        rice(out, 0, &[0; 16]);
        out.put(3, 0b101);
    });
    assert_eq!(malformed(&bytes), "frame padding");
    bytes = frame(mono(16), |out| {
        subframe(out, 0, 0);
        out.put(16, 7);
        out.put(8, 0);
    });
    assert_eq!(malformed(&bytes), "bytes after the last subframe");
}

#[test]
fn escaped_partitions_of_width_zero_are_silent() {
    let bytes = frame(mono(16), |out| {
        subframe(out, 8, 0);
        out.put(2, 0);
        out.put(4, 0);
        out.put(4, 15);
        out.put(5, 0);
    });
    assert_eq!(decode(&bytes).expect("silent"), vec![0; 16]);
}

#[test]
fn a_header_must_agree_with_streaminfo() {
    let info = |rate: u32, channels: u8, bits: u8, max_block: u16| StreamInfo {
        min_block: 16,
        max_block,
        min_frame: 0,
        max_frame: 0,
        sample_rate: rate,
        channels,
        bits,
        total_samples: 0,
        md5: [0; 16],
    };
    let bytes = frame(mono(16), |out| {
        subframe(out, 0, 0);
        out.put(16, 0);
    });
    let refused = |info: StreamInfo| match FlacDecoder::new(Some(info)).decode(&bytes) {
        Err(CodecError::Malformed { what, .. }) => what,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert_eq!(
        refused(info(48_000, 1, 16, 4096)),
        "sample rate differs from streaminfo"
    );
    assert_eq!(
        refused(info(44_100, 2, 16, 4096)),
        "channel count differs from streaminfo"
    );
    assert_eq!(
        refused(info(44_100, 1, 24, 4096)),
        "bit depth differs from streaminfo"
    );
    assert_eq!(
        refused(info(44_100, 1, 16, 15)),
        "block larger than the streaminfo maximum"
    );
    assert!(
        FlacDecoder::new(Some(info(44_100, 1, 16, 16)))
            .decode(&bytes)
            .is_ok()
    );
    // A header deferring to a STREAMINFO nobody gave is refused.
    let mut deferred = mono(16);
    deferred.sample_rate = None;
    let bytes = frame(deferred, |out| {
        subframe(out, 0, 0);
        out.put(16, 0);
    });
    assert_eq!(malformed(&bytes), "sample rate needs streaminfo");
}

fn constant_frame(number: u64, value: i64, rate: u32) -> Vec<u8> {
    let mut header = mono(16);
    header.number = number;
    header.sample_rate = Some(rate);
    frame(header, |out| {
        subframe(out, 0, 0);
        out.put_signed(16, value);
    })
}

fn params(extra: Vec<u8>, format: Option<SampleFormat>) -> CodecParams {
    let layout = crate::pcm::layout(1).expect("mono");
    let mut params = CodecParams::audio(CodecId::Flac, 44_100, layout);
    params.extra = extra;
    params.sample_format = format;
    params
}

fn mono_info(total: u64, md5: [u8; 16]) -> StreamInfo {
    StreamInfo {
        min_block: 16,
        max_block: 16,
        min_frame: 0,
        max_frame: 0,
        sample_rate: 44_100,
        channels: 1,
        bits: 16,
        total_samples: total,
        md5,
    }
}

fn signature(values: &[i64]) -> [u8; 16] {
    let mut md5 = Md5::new();
    for value in values {
        md5.samples(&[*value as i32; 16], 16);
    }
    md5.digest()
}

#[test]
fn the_signature_is_checked_only_over_a_complete_in_order_stream() {
    let info = mono_info(32, signature(&[3, -3]));
    let mut decoder = FlacDecoder::new(Some(info));
    decoder.decode(&constant_frame(0, 3, 44_100)).expect("0");
    assert_eq!(decoder.md5_check(), Md5Check::Incomplete);
    decoder.decode(&constant_frame(1, -3, 44_100)).expect("1");
    assert_eq!(decoder.md5_check(), Md5Check::Match);

    let mut decoder = FlacDecoder::new(Some(info));
    decoder.decode(&constant_frame(0, 3, 44_100)).expect("0");
    decoder.decode(&constant_frame(1, -2, 44_100)).expect("1");
    assert_eq!(decoder.md5_check(), Md5Check::Mismatch);

    // A skipped frame number leaves the check incomplete.
    let mut decoder = FlacDecoder::new(Some(info));
    decoder.decode(&constant_frame(1, -3, 44_100)).expect("1");
    decoder.decode(&constant_frame(2, -3, 44_100)).expect("2");
    assert_eq!(decoder.md5_check(), Md5Check::Incomplete);
    // Reset starts the check again.
    decoder.reset();
    decoder.decode(&constant_frame(0, 3, 44_100)).expect("0");
    decoder.decode(&constant_frame(1, -3, 44_100)).expect("1");
    assert_eq!(decoder.md5_check(), Md5Check::Match);

    assert_eq!(
        FlacDecoder::new(Some(mono_info(32, [0; 16]))).md5_check(),
        Md5Check::Unknown
    );
    assert_eq!(FlacDecoder::new(None).md5_check(), Md5Check::Unknown);
}

#[test]
fn a_failed_push_leaves_the_decoder_as_it_was() {
    let info = mono_info(32, signature(&[3, -3]));
    let mut decoder = FlacDecoder::open(&params(info.to_bytes().to_vec(), None)).expect("open");
    let packet = |data: Vec<u8>| Packet::audio(data, 0, 16, 44_100);
    decoder
        .push(&packet(constant_frame(0, 3, 44_100)))
        .expect("0");
    let mut bad = constant_frame(1, -3, 44_100);
    bad[6] ^= 1;
    assert!(decoder.push(&packet(bad)).is_err());
    assert!(
        decoder
            .push(&packet(constant_frame(1, -3, 48_000)))
            .is_err()
    );
    let mut bad_config = packet(constant_frame(1, -3, 44_100));
    let mut stereo = info;
    stereo.channels = 2;
    bad_config
        .side
        .push(SideData::Config(stereo.to_bytes().to_vec()));
    assert!(decoder.push(&bad_config).is_err());
    assert_eq!(decoder.stream_info(), Some(&info));
    decoder
        .push(&packet(constant_frame(1, -3, 44_100)))
        .expect("1");
    assert_eq!(decoder.md5_check(), Md5Check::Match);
}

#[test]
fn push_drain_and_empty_packets() {
    let mut decoder =
        FlacDecoder::open(&params(Vec::new(), Some(SampleFormat::I16))).expect("open");
    assert!(matches!(
        decoder.push(&Packet::audio(Vec::new(), 0, 0, 44_100)),
        Err(CodecError::InvalidApi { .. })
    ));
    let out = decoder
        .push(&Packet::audio(
            constant_frame(0, -7, 44_100),
            160,
            16,
            44_100,
        ))
        .expect("push");
    let [Output::Audio(audio)] = out.as_slice() else {
        panic!("one audio output");
    };
    assert_eq!(audio.pts, 160);
    assert_eq!(audio.pcm.as_i16().expect("i16"), &[-7i16; 16]);
    assert!(decoder.drain().expect("drain").is_empty());
    // A config packet installs a STREAMINFO.
    let info = mono_info(16, signature(&[2]));
    let mut first = Packet::audio(constant_frame(0, 2, 44_100), 0, 16, 44_100);
    first.side.push(SideData::Config(info.to_bytes().to_vec()));
    decoder.reset();
    decoder.push(&first).expect("config");
    assert_eq!(decoder.md5_check(), Md5Check::Match);
}

#[test]
fn open_refuses_what_it_cannot_produce() {
    let mut deep = mono_info(0, [0; 16]);
    deep.bits = 24;
    assert!(matches!(
        FlacDecoder::open(&params(deep.to_bytes().to_vec(), Some(SampleFormat::I16))),
        Err(CodecError::Unsupported { .. })
    ));
    let mut stereo = mono_info(0, [0; 16]);
    stereo.channels = 2;
    assert!(matches!(
        FlacDecoder::open(&params(stereo.to_bytes().to_vec(), None)),
        Err(CodecError::InvalidApi { .. })
    ));
    let mut wrong = params(Vec::new(), None);
    wrong.codec = CodecId::Pcm;
    assert!(FlacDecoder::open(&wrong).is_err());
    assert!(FlacDecoder::open(&params(vec![1, 2, 3], None)).is_err());
}

#[test]
fn f32_output_is_scaled_to_the_positive_maximum() {
    let mut decoder = FlacDecoder::open(&params(Vec::new(), None)).expect("open");
    let out = decoder
        .push(&Packet::audio(
            constant_frame(0, 32_767, 44_100),
            0,
            16,
            44_100,
        ))
        .expect("push");
    let [Output::Audio(audio)] = out.as_slice() else {
        panic!("one audio output");
    };
    assert_eq!(audio.pcm.as_f32().expect("f32"), &[1.0f32; 16]);
}
